# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests of the spill connector's fault containment (review findings B2, B3, B6, B7 + pin cap, breaker).

Scheduler side runs the real ``SpillScheduler`` against a minimal fake BlockPool/coordinator (the few methods it
calls, with vLLM's ref-count semantics); worker side runs the real ``SpillWorker`` on CPU tensors and real files.
Needs vLLM importable (production image, PYTHONPATH = a rig work tree) for the base classes and prodlayout.

usage: python3 test_sched_faults.py --dir <scratch> [--module <connector .py>] [--only b6]
  --module defaults to code/glm53_nvme_spill_connector.py of this repo; pass the pre-fix module with --only b6 to
  show the ordering test catches the race there.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import stat
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prodlayout  # noqa: E402
import torch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
HBS = 4608


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


# ------------------------------------------------------------------------------------------------ fake engine
class FB:
    def __init__(self, i):
        self.block_id = i
        self.ref_cnt = 0
        self.block_hash = None
        self.block_hash_num_tokens = None
        self.is_null = i == 0


class FakeMap:
    def __init__(self):
        self.m = {}

    def get_one_block(self, key):
        return self.m.get(bytes(key))


class FakePool:
    hash_block_size = HBS

    def __init__(self, n):
        self.blocks = [FB(i) for i in range(n)]
        self.blocks[0].ref_cnt = 1
        self.num_gpu_blocks = n
        self.cached_block_hash_to_block = FakeMap()
        self.cached_block_hashes_by_block: dict[int, set] = {}
        self.order: list[int] = list(range(1, n))   # free order (LRU first)

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0:
                self.order.remove(b.block_id)
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            assert b.ref_cnt > 0, f"double free of block {b.block_id}"
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.order.append(b.block_id)

    def get_num_free_blocks(self):
        return len(self.order)

    def _evict(self, b):
        if b.block_hash is not None:
            for k in [b.block_hash, *self.cached_block_hashes_by_block.pop(b.block_id, ())]:
                if self.cached_block_hash_to_block.m.get(bytes(k)) is b:
                    del self.cached_block_hash_to_block.m[bytes(k)]
        b.block_hash = None
        b.block_hash_num_tokens = None

    def get_new_blocks(self, n):
        assert len(self.order) >= n
        out = []
        for _ in range(n):
            bid = self.order.pop(0)
            b = self.blocks[bid]
            self._evict(b)
            b.ref_cnt = 1
            out.append(b)
        return out

    def _insert_block_hash(self, key, block, num_tokens):
        key = bytes(key)
        if block.block_hash is None:
            block.block_hash = key
            block.block_hash_num_tokens = num_tokens
        else:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(key)
        self.cached_block_hash_to_block.m[key] = block

    def evict_all_cached(self):
        for bid in list(self.order):
            self._evict(self.blocks[bid])


def K(h: bytes, g: int) -> bytes:
    return bytes(h) + g.to_bytes(4, "big")


class FakeCoord:
    """Deepest prefix whose MLA blocks (group 0) are cached AND whose boundary mamba state (group 2) is cached."""

    def __init__(self, pool):
        self.pool = pool

    def find_longest_cache_hit(self, block_hashes, max_len):
        m = self.pool.cached_block_hash_to_block
        mla = []
        for h in block_hashes[: max_len // HBS]:
            b = m.get_one_block(K(h, 0))
            if b is None:
                break
            mla.append(b)
        while mla:
            st = m.get_one_block(K(block_hashes[len(mla) - 1], 2))
            if st is not None:
                groups = [mla, [], [st], [], [], [], []]
                return groups, len(mla) * HBS, None
            mla.pop()
        return [[] for _ in range(7)], 0, None


class Req(SimpleNamespace):
    pass


def make_req(rid, n_blocks, tag=b"c"):
    hashes = [tag + i.to_bytes(4, "big") for i in range(n_blocks + 1)]
    n = n_blocks * HBS + 7
    return Req(request_id=rid, block_hashes=hashes, num_tokens=n, num_computed_tokens=n, num_prompt_tokens=n,
               skip_reading_prefix_cache=False)


def compute(pool, req, n_blocks):
    """The engine computed req's first n_blocks: keyed MLA blocks + the boundary mamba state, then released."""
    blocks = []
    for i in range(n_blocks):
        b = pool.get_new_blocks(1)[0]
        pool._insert_block_hash(K(req.block_hashes[i], 0), b, HBS)
        blocks.append(b)
    s = pool.get_new_blocks(1)[0]
    pool._insert_block_hash(K(req.block_hashes[n_blocks - 1], 2), s, HBS)
    blocks.append(s)
    pool.free_blocks(blocks)
    return blocks


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def new_sched(spill, root, n=64, **kw):
    cfg = spill.SpillConfig(root=root, cap_bytes=1 << 40, min_tokens=2 * HBS, min_gain=HBS, save_policy="all",
                            log=False, **kw)
    cfg.log_every_s = kw.get("log_every_s", 0)
    kvc = prodlayout.build_kv_cache_config(n)
    vc = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))
    s = spill.SpillScheduler(cfg, vc, kvc)
    pool = FakePool(n)
    s.bind(block_pool=pool, kv_cache_manager=SimpleNamespace(coordinator=FakeCoord(pool), block_pool=pool))
    clk, io = Clock(), Clock()
    s.clock = clk
    if hasattr(s, "io_clock"):
        s.io_clock = io
    return s, pool, clk, io


def answer(spill, s, job, ok, fail):
    s.update_connector_output(spill.SpillWorkerMeta(done={job: [ok, fail]}))


def refs(pool, blocks):
    return [b.ref_cnt for b in blocks]


def saved_rp(spill, s, pool, req, nb):
    """Compute + save + both ranks ok -> one ok RP. Returns the computed blocks."""
    blocks = compute(pool, req, nb)
    s.save_request(req)
    m = s.build_meta()
    if not m.saves:
        m = s.build_meta()
    (job,) = list(m.saves)
    answer(spill, s, job, 2, 0)
    return blocks


# ------------------------------------------------------------------------------------------------ tests
def t_b6_ordering(spill, root):
    s, pool, clk, io = new_sched(spill, root + "/b6")
    req = make_req("r", 3)
    compute(pool, req, 3)
    s.tracked = {}
    s._scan_prefill_done = lambda: s.save_request(req)      # a save planned INSIDE build_meta (prefill-done scan)
    m1 = s.build_meta()
    s._scan_prefill_done = lambda: None
    pending = s.has_pending()
    m2 = s.build_meta()
    check("b6_save_planned_in_build_meta_not_shipped_same_step", not m1.saves, f"same-step saves={list(m1.saves)}")
    check("b6_save_shipped_next_step", len(m2.saves) == 1, f"next-step saves={list(m2.saves)}")
    check("b6_engine_keeps_stepping_while_unsent", pending)


def t_b6_between_steps(spill, root):
    s, pool, clk, io = new_sched(spill, root + "/b6b")
    req = make_req("r", 3)
    compute(pool, req, 3)
    s.save_request(req)                     # request finish / update_from_output: between steps
    check("b6_unsent_counts_as_pending", s.has_pending())
    m1 = s.build_meta()
    check("b6_between_step_save_ships_with_next_build_meta", len(m1.saves) == 1)


def t_save_deadline(spill, root):
    s, pool, clk, io = new_sched(spill, root + "/sd", save_deadline_s=60)
    req = make_req("r", 3)
    blocks = compute(pool, req, 3)
    s.save_request(req)
    m = s.build_meta()
    (job,) = list(m.saves)
    check("sd_pinned_while_saving", all(r == 1 for r in refs(pool, blocks)), f"{refs(pool, blocks)}")
    answer(spill, s, job, 1, 0)             # rank 0 answers, rank 1 never does
    io.t += 59
    s.build_meta()
    check("sd_still_pinned_before_deadline", all(r == 1 for r in refs(pool, blocks)) and s.breaker is None)
    io.t += 2
    m = s.build_meta()
    m2 = s.build_meta()
    check("sd_pins_released_at_deadline", all(r == 0 for r in refs(pool, blocks)), f"{refs(pool, blocks)}")
    check("sd_breaker_open", s.breaker is not None and s.stats["save_timeouts"] == 1, f"{s.breaker}")
    dels = set(m.deletes) | set(m2.deletes)
    check("sd_files_deleted_after_hung_job", len(dels) == 4, f"deletes={len(dels)}")
    check("sd_no_pending_after_expiry", not s.has_pending())
    answer(spill, s, job, 1, 0)             # the late rank finally answers
    check("sd_late_answer_ignored", s.stats["late_answers"] == 1 and s.pinned_blocks() == 0
          and all(r == 0 for r in refs(pool, blocks)))
    req2 = make_req("r2", 3, tag=b"d")
    compute(pool, req2, 3)
    s.save_request(req2)
    check("sd_breaker_blocks_new_saves", not s.save_jobs and not s.build_meta().saves)


def t_load_deadline(spill, root, missing_ack: bool):
    tag = "noack" if missing_ack else "hang"
    s, pool, clk, io = new_sched(spill, root + "/ld" + tag, load_deadline_s=30)
    req = make_req("r", 3)
    saved_rp(spill, s, pool, req, 3)
    check(f"ld_{tag}_rp_ok", s.stats["rp_saved"] == 1)
    pool.evict_all_cached()
    req2 = make_req("r2", 4)                # the next turn extends it
    r = s.get_num_new_matched_tokens(req2, 0)
    check(f"ld_{tag}_restore_started", r == (None, False) and s.stats["restore_started"] == 1, f"{r}")
    st = s.restores["r2"]
    nb = list(st.new_blocks)
    m = s.build_meta()
    (job,) = list(m.loads)
    if missing_ack:
        answer(spill, s, job, 1, 0)         # rank 0 answered; rank 1's answer is lost
    io.t += 31
    s.build_meta()
    r = s.get_num_new_matched_tokens(req2, 0)
    check(f"ld_{tag}_request_detached_to_prefill", r == (0, False) and "r2" in s.no_restore, f"{r}")
    check(f"ld_{tag}_fresh_blocks_quarantined", all(b.ref_cnt == 1 and b.block_hash is None for b in nb)
          and job in s.quarantine, f"{refs(pool, nb)}")
    check(f"ld_{tag}_breaker_open_no_busy_loop", s.breaker is not None and not s.has_pending())
    r3 = s.get_num_new_matched_tokens(make_req("r3", 4), 0)
    check(f"ld_{tag}_no_new_restore_after_breaker", r3 == (0, False) and s.stats["restore_started"] == 1)
    if missing_ack:
        io.t += 3600
        s.build_meta()
        check("ld_noack_quarantine_held_without_answer", all(b.ref_cnt == 1 for b in nb))
    else:
        answer(spill, s, job, 1, 0)
        check("ld_hang_one_answer_not_enough", all(b.ref_cnt == 1 for b in nb))
        answer(spill, s, job, 1, 0)
        check("ld_hang_late_answer_frees_quarantine", all(b.ref_cnt == 0 and b.block_hash is None for b in nb)
              and not s.quarantine and s.pinned_blocks() == 0, f"{refs(pool, nb)}")
    s.request_finished(req2)
    check(f"ld_{tag}_request_finish_clean", "r2" not in s.no_restore)


def t_consecutive_failures(spill, root):
    s, pool, clk, io = new_sched(spill, root + "/cf", n=128, breaker_fails=6)
    pinned = []
    for i in range(7):
        req = make_req(f"r{i}", 2, tag=bytes([65 + i]))
        pinned += compute(pool, req, 2)
        s.save_request(req)
        m = s.build_meta()
        for job in m.saves:
            answer(spill, s, job, 1, 1)
        if s.breaker is not None:
            break
    check("cf_breaker_after_6_failures", s.breaker is not None and s.stats["jobs_failed"] == 6, f"{s.breaker}")
    check("cf_all_pins_released", all(b.ref_cnt == 0 for b in pinned) and s.pinned_blocks() == 0)


def t_success_resets(spill, root):
    s, pool, clk, io = new_sched(spill, root + "/sr", n=128, breaker_fails=3)
    for i in range(8):
        req = make_req(f"r{i}", 2, tag=bytes([65 + i]))
        compute(pool, req, 2)
        s.save_request(req)
        m = s.build_meta()
        for job in m.saves:
            answer(spill, s, job, 2, 0) if i % 2 else answer(spill, s, job, 1, 1)
    check("cf_interleaved_failures_do_not_trip", s.breaker is None and s.stats["jobs_failed"] == 4)


def t_worker_exc(spill, root):
    s, pool, clk, io = new_sched(spill, root + "/we")
    s.update_connector_output(spill.SpillWorkerMeta(done={}, exc=1))
    check("we_worker_exception_trips_breaker", s.breaker is not None and s.stats["worker_exc"] == 1)


def t_b7(spill, root):
    s, pool, clk, io = new_sched(spill, root + "/b7")
    req = make_req("r", 3)
    compute(pool, req, 3)
    s.save_request(req)
    m = s.build_meta()
    (job,) = list(m.saves)
    rp = next(iter(s.rps.values()))
    s._forget_bad_files([rp.blocks[0].fname])   # a read elsewhere found one of its files bad while it was saving
    answer(spill, s, job, 2, 0)
    m2 = s.build_meta()
    check("b7_no_disk_leak_after_dropped_rp_saved", s.disk_bytes == 0 and not s.disk,
          f"disk_bytes={s.disk_bytes} files={len(s.disk)} deletes={len(m2.deletes)}")


def t_pin_cap(spill, root):
    # cached-free blocks (their request finished): pinning them makes them exclusive -> gated
    s, pool, clk, io = new_sched(spill, root + "/pc", max_pinned_blocks=3)
    req = make_req("r", 3)
    compute(pool, req, 3)
    s.save_request(req)
    check("pc_save_of_exclusive_blocks_over_cap_skipped", not s.save_jobs and s.stats["rp_skipped_pins"] == 1)
    # the same RP while its request still holds the blocks (prompt just computed): no exclusive pin -> not gated
    s1, pool1, _, _ = new_sched(spill, root + "/pc1", max_pinned_blocks=3)
    req = make_req("r", 3)
    blocks = compute(pool1, req, 3)
    pool1.touch(blocks)                     # the running request's own references
    s1.save_request(req)
    check("pc_save_of_blocks_in_use_not_gated", len(s1.save_jobs) == 1 and s1.save_pinned_exclusive() == 0)
    pool1.free_blocks(blocks)               # the request finishes while the copy is in flight
    check("pc_exclusive_count_follows_ref_cnt", s1.save_pinned_exclusive() == 4)
    req2 = make_req("r2", 2, tag=b"y")
    compute(pool1, req2, 2)
    s1.save_request(req2)
    check("pc_next_save_gated_while_exclusive_over_cap", len(s1.save_jobs) == 1 and s1.stats["rp_skipped_pins"] == 1)
    # restores are not gated by the save-pin cap
    s2, pool2, _, _ = new_sched(spill, root + "/pc2", max_pinned_blocks=6)
    saved_rp(spill, s2, pool2, make_req("r", 3), 3)
    s2.max_pinned = 1
    pool2.evict_all_cached()
    r = s2.get_num_new_matched_tokens(make_req("r2", 4), 0)
    check("pc_restore_not_gated", r == (None, False) and s2.stats["restore_started"] == 1, f"{r}")


def t_stats_line(spill, root):
    s, pool, clk, io = new_sched(spill, root + "/sl")
    line = s.stats_line()
    need = ["breaker=closed", "pinned=", "oldest_save_s=", "oldest_load_s=", "worker_exc=", "written_gib_1h=",
            "restores_ok=", "restores_failed=", "restores_fallback="]
    check("stats_line_fields", all(k in line for k in need), line)


# ------------------------------------------------------------------------------------------------ worker
def make_worker(spill, root, rank=0):
    cfg = spill.SpillConfig(root=root, cap_bytes=8 << 30, min_free_bytes=0, direct=False, log_every_s=0)
    kvc = prodlayout.build_kv_cache_config(16, scale=576)
    w = spill.SpillWorker(cfg, kvc, rank)
    raws = [torch.randint(0, 255, (kvt.size,), dtype=torch.uint8) for kvt in kvc.kv_cache_tensors]
    views = {n: raws[t] for t, kvt in enumerate(kvc.kv_cache_tensors) for n in kvt.shared_by}
    w.register(views, sig="x")
    return w


def wait_meta(w, jobs, timeout=10):
    got, exc = {}, 0
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        m = w.take_meta()
        if m is not None:
            got.update(m.done)
            exc += m.exc
        if all(j in got for j in jobs):
            break
        time.sleep(0.005)
    return got, exc


def t_worker(spill, root):
    M = spill.SpillMeta
    w = make_worker(spill, root + "/w1")
    w.handle(M(saves={1: [(0, 3, "a.blk")]}))
    got, _ = wait_meta(w, [1])
    check("w_save_ok", got.get(1) == [1, 0], f"{got}")
    d = w.store.dir
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)        # unlink of an EXISTING file -> EACCES (review B2 repro)
    try:
        w.handle(M(deletes=["a.blk"]))
        w.drain(5)
        w.handle(M(saves={2: [(0, 4, "b.blk")]}))   # write fails (EACCES), cleanup unlink fails too
        got, exc = wait_meta(w, [2])
    finally:
        os.chmod(d, stat.S_IRWXU)
    check("w_writer_alive_after_eacces_delete", w._threads[0].is_alive(), f"stats={w.stats['delete_failed']}")
    check("w_failed_save_answered", got.get(2) == [0, 1] and exc == 0, f"{got} exc={exc}")
    w.handle(M(saves={3: [(0, 5, "c.blk")]}))
    got, _ = wait_meta(w, [3])
    check("w_next_save_ok", got.get(3) == [1, 0], f"{got}")
    # an unexpected exception inside a job: answered as a failure, counted, thread survives
    orig = w.store.write_block
    w.store.write_block = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    w.handle(M(saves={4: [(0, 6, "d.blk")]}))
    got, exc = wait_meta(w, [4])
    w.store.write_block = orig
    check("w_unexpected_exception_answered_and_reported", got.get(4) == [0, 1] and exc == 1 and w._threads[0].is_alive(),
          f"{got} exc={exc}")
    origr = w.store.read_block
    w.store.read_block = lambda *a, **k: (_ for _ in ()).throw(KeyError("boom"))
    w.handle(M(loads={5: [(0, 7, "c.blk")]}))
    got, exc = wait_meta(w, [5])
    w.store.read_block = origr
    check("w_reader_exception_answered", got.get(5) == [0, 1] and exc == 1 and w._threads[1].is_alive(), f"{got}")
    # an exception escaping _save itself (the harsher review injection: delete raising out of the method)
    w.store.write_block = lambda *a, **k: (_ for _ in ()).throw(OSError(30, "ro"))
    w.store.delete = lambda f: (_ for _ in ()).throw(OSError(30, "ro unlink"))
    w.handle(M(saves={6: [(0, 8, "e.blk")]}))
    got, exc = wait_meta(w, [6])
    check("w_exception_out_of_cleanup_answered_once", got.get(6) == [0, 1] and w._threads[0].is_alive(),
          f"{got} exc={exc}")
    line = w.stats_line()
    check("w_stats_line", "threads_alive=2/2" in line and "thread_exc=" in line, line)
    w.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--module", default=os.path.join(HERE, "..", "code", "glm53_nvme_spill_connector.py"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("spill_under_test", a.module)
    spill = importlib.util.module_from_spec(spec)
    sys.modules["spill_under_test"] = spill
    spec.loader.exec_module(spill)
    root = os.path.abspath(a.dir)
    os.makedirs(root, exist_ok=True)
    if a.only == "b6":
        t_b6_ordering(spill, root)
    else:
        t_b6_ordering(spill, root)
        t_b6_between_steps(spill, root)
        t_save_deadline(spill, root)
        t_load_deadline(spill, root, missing_ack=False)
        t_load_deadline(spill, root, missing_ack=True)
        t_consecutive_failures(spill, root)
        t_success_resets(spill, root)
        t_worker_exc(spill, root)
        t_b7(spill, root)
        t_pin_cap(spill, root)
        t_stats_line(spill, root)
        t_worker(spill, root)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"SUMMARY {len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
