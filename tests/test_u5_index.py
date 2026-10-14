# SPDX-License-Identifier: AGPL-3.0-only
"""u5 (2026-10-14): every resume point confirmed on all ranks is in the on-disk index within a bounded time, also when
the engine goes idle right after the confirmation and is then SIGKILLed (production: the restart script waits idle, then
start.sh `docker rm -f` -> no shutdown hook).

Production finding (2026-10-14): probe prepare saved 4 RPs; the last index write came 4 s before the 4th RP's save, the
engine went idle, `docker rm -f` 30 s after that save -> "adopted 3 resume points ... orphan files deleted per rank
{0: 23, 1: 23}". Cause: the index was written only from build_meta (engine-step clock, throttled to INDEX_EVERY_S); the
answer that confirmed the last RP arrived in the last step, after which EngineCore blocked idle (no build_meta), and the
shutdown flush never runs under SIGKILL.

Real SpillScheduler against the fake pool of test_sched_faults.py (as test_ttl_persist.py), real SpillWorkers for the
end-to-end SIGKILL restart.

usage: python3 test_u5_index.py --dir <scratch> [--module <connector .py>] [--old]
  --old: the module is a pre-u5 connector (bd0f589e): run only the reproduction checks, which then must show the bug
         (the expectations are inverted: 'repro_*' PASS = the bug is present in that module).
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shutil
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sched_faults as T  # noqa: E402
import test_ttl_persist as P  # noqa: E402

HBS = T.HBS
check = T.check
OLD = False


def index_rps(spill, s):
    idx, err = spill._read_index(s.cfg.index_path())
    return None if idx is None else sorted(r["x"]["n"] // HBS for r in idx["rps"])


def ok_rps(s):
    return sorted(rp.n_tokens // HBS for rp in s.rps.values() if rp.state == "ok")


def flush(s):
    s._index_writer.flush(10.0)


def booted(spill, root, **kw):
    w = P.Wall()
    s, pool, clk, io, m1, m2 = P.boot(spill, root, w, **kw)
    s.build_meta()          # the wipe decision's urgent index write
    flush(s)
    return s, pool, clk, io, w


def confirm_in_last_step(spill, s, pool, rid, hashes, nb):
    """save planned -> shipped by build_meta -> both ranks answer ok in update_connector_output of THAT step: the RP is
    confirmed by the engine's last step (no build_meta after it)."""
    r = P.req(rid, hashes)
    T.compute(pool, r, nb)
    s.save_request(r)
    m = s.build_meta()
    (job,) = list(m.saves)
    T.answer(spill, s, job, 2, 0)
    return job


# ------------------------------------------------------------------------------------------------ reproduction
def t_repro_last_step(spill, root):
    """The production sequence on one scheduler: RP1 confirmed + indexed, RP2 confirmed in the last step, EngineCore asks
    has_requests() (-> has_pending) and blocks idle; then SIGKILL (no shutdown)."""
    s, pool, clk, io, w = booted(spill, root + "/rl")
    confirm_in_last_step(spill, s, pool, "a", P.chain(b"A", 6)[:5], 4)
    s.build_meta()          # one more step: RP1 is written (urgent? no: first change after the wipe write -> cadence)
    clk.t += 6
    s.build_meta()
    flush(s)
    before = index_rps(spill, s)
    clk.t += 1              # RP2 arrives 1 s later, within INDEX_EVERY_S of the last write
    confirm_in_last_step(spill, s, pool, "b", P.chain(b"B", 7)[:6], 5)
    pend = s.has_pending()  # EngineCore: has_work() -> Scheduler.has_requests() -> connector.has_pending_push_work()
    flush(s)                # the idle period (production: >= 20 s until docker rm -f)
    after = index_rps(spill, s)
    have = ok_rps(s)
    if OLD:
        check("repro_old_last_rp_missing_from_index_at_idle", before == [4] and have == [4, 5] and after == [4],
              f"index before={before} at kill={after} confirmed={have}")
    else:
        check("u5_idle_index_has_every_confirmed_rp", before == [4] and have == [4, 5] and after == [4, 5]
              and pend is False, f"index before={before} at kill={after} confirmed={have} pending={pend}")
        check("u5_idle_write_counted", s.stats.get("index_writes_idle", 0) == 1, f"{s.stats.get('index_writes_idle')}")


def t_repro_e2e_sigkill(spill, root):
    """Two real ranks: 2 conversations, the 2nd confirmed in the last step, idle, SIGKILL (no shutdown) -> new workers +
    new scheduler: every confirmed RP adopted, no orphan file (old: the last one's files are orphans)."""
    d = root + "/e2e"
    w = P.Wall()
    nb = 48

    def world(seed):
        ws = [P.make_worker(spill, d, r, nblocks=nb, seed=seed) for r in (0, 1)]
        return [x[0] for x in ws], [x[1] for x in ws]

    def step(s, workers, m):
        import copy
        for wk in workers:
            wk.handle(copy.deepcopy(m))
        for wk in workers:
            wk.drain(60)
        agg = None
        for wk in workers:
            mm = wk.take_meta()
            if mm is not None:
                agg = mm if agg is None else agg.aggregate(mm)
        s.update_connector_output(agg)

    workers, _ = world(1)
    for wk in workers:
        P.take_hello(wk, 20)
        wk._hello_sent = False
    s, pool, clk, io = P.sched(spill, d, w, persist=True, n=nb)
    s._ident_thread.join(60)
    for _ in range(3):
        step(s, workers, s.build_meta())
    for k, (tag, nbk) in enumerate(((b"X", 3), (b"Y", 4))):
        r = P.req(f"c{k}", P.chain(tag, 12)[:nbk + 1], {"ttl_s": 86400, "session": f"S{k}", "epoch": 0})
        T.compute(pool, r, nbk)
        s.save_request(r)
        step(s, workers, s.build_meta())        # ships the save; both ranks write + answer in this step
        if k == 0:
            clk.t += 6
            step(s, workers, s.build_meta())    # a later step writes the index with conversation 0
    confirmed = ok_rps(s)
    pend = s.has_pending()                      # EngineCore goes idle
    flush(s)
    files_before = sorted(os.listdir(workers[0].store.dir))
    for wk in workers:                          # SIGKILL: no s.shutdown(), the workers' threads just end
        wk.shutdown()
    workers2, _ = world(2)
    s2, pool2, _, _ = P.sched(spill, d, w, persist=True, n=nb)
    s2._ident_thread.join(60)
    for _ in range(3):
        step(s2, workers2, s2.build_meta())
    files_after = sorted(os.listdir(workers2[0].store.dir))
    adopted = s2.stats["persist_adopted_rps"]
    orph = s2.stats["persist_orphans"]
    det = (f"confirmed={confirmed} pending_at_idle={pend} adopted={adopted} orphans={orph} reason={s2.persist_reason} "
           f"files {len(files_before)}->{len(files_after)}")
    if OLD:
        check("repro_old_sigkill_after_idle_loses_last_rp", confirmed == [3, 4] and adopted == 1 and orph > 0
              and len(files_after) < len(files_before), det)
    else:
        check("u5_sigkill_after_idle_adopts_every_confirmed_rp", confirmed == [3, 4] and adopted == 2 and orph == 0
              and files_after == files_before and s2.persist_reason == "adopted", det)
    for wk in workers2:
        wk.shutdown()


# ------------------------------------------------------------------------------------------------ idle hook details
def t_idle_rules(spill, root):
    s, pool, clk, io, w = booted(spill, root + "/ir")
    n0 = s.stats["index_writes"]
    s.has_pending()
    flush(s)
    check("idle_no_write_when_unchanged", s.stats["index_writes"] == n0, f"{n0} {s.stats['index_writes']}")
    # a save in flight: has_pending True (engine keeps stepping) and no idle write; the RP is not indexed before its
    # answer, and is indexed at the idle after it
    r = P.req("u", P.chain(b"U", 6)[:5])
    T.compute(pool, r, 4)
    s.save_request(r)
    m = s.build_meta()
    (job,) = list(m.saves)
    p1 = s.has_pending()
    flush(s)
    check("idle_no_write_while_save_in_flight", p1 is True and s.stats.get("index_writes_idle", 0) == 0
          and index_rps(spill, s) == [], f"pending={p1} idx={index_rps(spill, s)}")
    T.answer(spill, s, job, 1, 0)          # one rank only: not confirmed yet
    p2 = s.has_pending()
    flush(s)
    check("idle_half_answered_not_indexed", p2 is True and index_rps(spill, s) == [], f"{p2} {index_rps(spill, s)}")
    T.answer(spill, s, job, 1, 0)
    p3 = s.has_pending()
    flush(s)
    check("idle_after_last_answer_indexed", p3 is False and index_rps(spill, s) == [4], f"{p3} {index_rps(spill, s)}")
    n1 = s.stats["index_writes"]
    for _ in range(5):
        s.has_pending()
    flush(s)
    check("idle_repeated_asks_write_once", s.stats["index_writes"] == n1, f"{n1} {s.stats['index_writes']}")
    # a failed save: nothing indexed, the index stays as it was
    r = P.req("f", P.chain(b"F", 6)[:5])
    T.compute(pool, r, 4)
    s.save_request(r)
    (job,) = list(s.build_meta().saves)
    T.answer(spill, s, job, 1, 1)
    s.has_pending()
    flush(s)
    check("idle_failed_save_not_indexed", index_rps(spill, s) == [4], f"{index_rps(spill, s)}")
    # every written index verifies (len + sha256 line) and is the file the next start reads
    idx, err = spill._read_index(s.cfg.index_path())
    check("idle_index_verifies", idx is not None and not os.path.exists(s.cfg.index_path() + ".tmp"), err)


def t_idle_never_raises(spill, root):
    s, pool, clk, io, w = booted(spill, root + "/nr")
    confirm_in_last_step(spill, s, pool, "a", P.chain(b"A", 6)[:5], 4)
    orig = s._index_payload

    def boom():
        raise RuntimeError("payload boom")
    s._index_payload = boom
    f0 = s.stats.get("index_write_failed", 0)
    try:
        p = s.has_pending()
        raised = None
    except Exception as e:  # noqa: BLE001
        p, raised = None, e
    check("idle_write_error_does_not_reach_vllm", raised is None and p is False
          and s.stats.get("index_write_failed", 0) == f0 + 1, f"{raised!r} {p}")
    s._index_payload = orig
    s.has_pending()
    flush(s)
    check("idle_retries_after_error", index_rps(spill, s) == [4], f"{index_rps(spill, s)}")


def t_idle_persist_off(spill, root):
    s, pool, clk, io = P.sched(spill, root + "/off", P.Wall(), persist=False)
    confirm_in_last_step(spill, s, pool, "a", P.chain(b"A", 6)[:5], 4)
    p = s.has_pending()
    check("idle_persist_off_no_index", p is False and s._index_writer is None
          and not os.path.exists(s.cfg.index_path()))


def t_log_line(spill, root):
    recs = []

    class H(logging.Handler):
        def emit(self, rec):
            recs.append(rec.getMessage())
    h = H(level=logging.INFO)
    spill.logger.addHandler(h)
    old = spill.logger.level
    spill.logger.setLevel(logging.INFO)
    try:
        s, pool, clk, io, w = booted(spill, root + "/log")
        confirm_in_last_step(spill, s, pool, "a", P.chain(b"A", 6)[:5], 4)
        s.has_pending()
        flush(s)
    finally:
        spill.logger.removeHandler(h)
        spill.logger.setLevel(old)
    lines = [m for m in recs if "persist: index written (idle: " in m]
    check("log_idle_write_line_after_rename", len(lines) == 1 and "(idle: 1 resume points, " in lines[0]
          and "bytes on disk)" in lines[0], f"{lines}")


# ------------------------------------------------------------------------------------------------ timer + lock
def t_timer(spill, root):
    s, pool, clk, io, w = booted(spill, root + "/tm")
    s.start_index_timer()
    try:
        confirm_in_last_step(spill, s, pool, "a", P.chain(b"A", 6)[:5], 4)     # dirty, engine call just now
        time.sleep(3 * spill.INDEX_TIMER_TICK_S + 0.3)
        flush(s)
        check("timer_waits_while_engine_recent", index_rps(spill, s) == [] and s.stats.get("index_writes_timer", 0) == 0,
              f"{index_rps(spill, s)}")
        io.t += spill.INDEX_IDLE_S + 0.1            # no engine call for INDEX_IDLE_S (no has_pending asked)
        t_end = time.monotonic() + 5
        while time.monotonic() < t_end and s.stats.get("index_writes_timer", 0) == 0:
            time.sleep(0.05)
        flush(s)
        check("timer_writes_changed_index_when_engine_silent", index_rps(spill, s) == [4]
              and s.stats.get("index_writes_timer", 0) == 1, f"{index_rps(spill, s)} {s.stats.get('index_writes_timer')}")
        time.sleep(3 * spill.INDEX_TIMER_TICK_S)
        check("timer_writes_once_per_change", s.stats.get("index_writes_timer", 0) == 1)
        # the timer only reads the state between two connector calls: while the lock is held it does not write
        confirm_in_last_step(spill, s, pool, "b", P.chain(b"B", 7)[:6], 5)
        io.t += spill.INDEX_IDLE_S + 0.1
        with s.lock:
            time.sleep(4 * spill.INDEX_TIMER_TICK_S)
            flush(s)
            held = index_rps(spill, s)
        t_end = time.monotonic() + 5
        while time.monotonic() < t_end and s.stats.get("index_writes_timer", 0) < 2:
            time.sleep(0.05)
        flush(s)
        check("timer_blocked_by_lock_then_writes", held == [4] and index_rps(spill, s) == [4, 5], f"{held} {index_rps(spill, s)}")
    finally:
        s.shutdown()
    check("timer_thread_stopped_by_shutdown", s._index_timer is not None and not s._index_timer.is_alive())


def t_connector_lock(spill, root):
    """Every scheduler-side entry point of the vLLM connector holds SpillScheduler.lock."""
    s, pool, clk, io, w = booted(spill, root + "/cl")
    conn = object.__new__(spill.Glm53NvmeSpillConnector)
    conn._sched = s
    conn._worker = None
    from types import SimpleNamespace
    r = P.req("q", P.chain(b"Q", 6)[:5])
    calls = {
        "has_pending_push_work": lambda: conn.has_pending_push_work(),
        "build_connector_meta": lambda: conn.build_connector_meta(None),
        "update_connector_output": lambda: conn.update_connector_output(SimpleNamespace(kv_connector_worker_meta=None)),
        "get_num_new_matched_tokens": lambda: conn.get_num_new_matched_tokens(r, 0),
        "update_state_after_alloc": lambda: conn.update_state_after_alloc(r, None, 0),
        "request_finished": lambda: conn.request_finished(r, []),
        "request_finished_all_groups": lambda: conn.request_finished_all_groups(r, ([],)),
        "bind_gpu_block_pool": lambda: conn.bind_gpu_block_pool(pool),
    }
    bad = []
    for name, fn in calls.items():
        ev = threading.Event()
        rel = threading.Event()

        def holder():
            with s.lock:
                ev.set()
                rel.wait(5)
        th = threading.Thread(target=holder)
        th.start()
        ev.wait(5)
        done = threading.Event()

        def caller():
            fn()
            done.set()
        tc = threading.Thread(target=caller)
        tc.start()
        blocked = not done.wait(0.3)
        rel.set()
        th.join(5)
        tc.join(5)
        if not (blocked and done.is_set()):
            bad.append(name)
    check("connector_entry_points_hold_the_lock", not bad, f"not locked: {bad}")


def t_timer_stress(spill, root):
    """The engine thread saves / answers / drops RPs under the lock while the timer writes as often as it can: every
    index written verifies and is self-consistent (every RP's files in 'files', every RP confirmed at that time)."""
    s, pool, clk, io, w = booted(spill, root + "/st", n=512, cap_bytes=1 << 40)
    tick, idle = spill.INDEX_TIMER_TICK_S, spill.INDEX_IDLE_S
    spill.INDEX_TIMER_TICK_S, spill.INDEX_IDLE_S = 0.002, 0.0
    problems = []
    orig_write = s._index_writer._write

    def checked_write(payload):
        try:
            nl = payload.index(b"\n")
            import hashlib
            import json
            top = json.loads(payload[:nl])
            body = payload[nl + 1:]
            assert top["len"] == len(body) and top["sha256"] == hashlib.sha256(body).hexdigest()
            idx = json.loads(body)
            for rp in idx["rps"]:
                for b in rp["x"]["b"]:
                    assert b[3] in idx["files"], "file of an RP missing from files"
        except Exception as e:  # noqa: BLE001
            problems.append(repr(e))
        orig_write(payload)
    s._index_writer._write = checked_write
    s.start_index_timer()
    n_rp = 0
    try:
        for k in range(60):
            with s.lock:
                tag = bytes([0x10 + k % 200, k // 200])
                r = P.req(f"s{k}", P.chain(tag, 6)[:5])
                T.compute(pool, r, 4)
                s.save_request(r)
                m = s.build_meta()
                for job in list(m.saves):
                    T.answer(spill, s, job, 2, 0)
                    n_rp += 1
                if k % 7 == 3 and s.rps:
                    s._drop_rp(next(iter(s.rps.values())), "lru")
                io.t += 1
            time.sleep(0.004)
        time.sleep(0.1)
    finally:
        spill.INDEX_TIMER_TICK_S, spill.INDEX_IDLE_S = tick, idle
        s.shutdown()
    idx_n = index_rps(spill, s)
    check("stress_timer_wrote_many", s.stats.get("index_writes_timer", 0) >= 5, f"{s.stats.get('index_writes_timer')}")
    check("stress_every_index_consistent", not problems and s.stats.get("index_write_failed", 0) == 0,
          f"{problems[:3]}")
    check("stress_final_index_equals_state", idx_n == ok_rps(s), f"{len(idx_n or [])} vs {len(ok_rps(s))}")


def t_shutdown(spill, root):
    s, pool, clk, io, w = booted(spill, root + "/sd")
    s.start_index_timer()
    confirm_in_last_step(spill, s, pool, "a", P.chain(b"A", 6)[:5], 4)
    r = P.req("b", P.chain(b"B", 7)[:6])
    T.compute(pool, r, 5)
    s.save_request(r)
    s.build_meta()                       # shipped, never answered (the workers are gone at Scheduler.shutdown)
    t0 = time.monotonic()
    s.shutdown()
    dt = time.monotonic() - t0
    check("shutdown_writes_confirmed_only_synchronously", index_rps(spill, s) == [4]
          and s.stats.get("index_writes_shutdown", 0) == 1, f"{index_rps(spill, s)}")
    check("shutdown_fast_and_timer_joined", dt < 3.0 and not s._index_timer.is_alive(), f"{dt:.2f}s")


def main() -> int:
    global OLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--module", default=os.path.join(HERE, "..", "code", "glm53_nvme_spill_connector.py"))
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    OLD = a.old
    spec = importlib.util.spec_from_file_location("spill_under_test", a.module)
    spill = importlib.util.module_from_spec(spec)
    sys.modules["spill_under_test"] = spill
    spec.loader.exec_module(spill)
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    assert os.environ.get("PYTHONHASHSEED"), "run with PYTHONHASHSEED set"
    init_none_hash(sha256)
    root = os.path.abspath(a.dir)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    tests = [("repro", lambda: t_repro_last_step(spill, root)), ("e2e", lambda: t_repro_e2e_sigkill(spill, root))]
    if not OLD:
        tests += [("rules", lambda: t_idle_rules(spill, root)), ("noraise", lambda: t_idle_never_raises(spill, root)),
                  ("off", lambda: t_idle_persist_off(spill, root)), ("log", lambda: t_log_line(spill, root)),
                  ("timer", lambda: t_timer(spill, root)), ("lock", lambda: t_connector_lock(spill, root)),
                  ("stress", lambda: t_timer_stress(spill, root)), ("shutdown", lambda: t_shutdown(spill, root))]
    for name, fn in tests:
        if a.only and name not in a.only.split(","):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{name}_raised", False, repr(e))
    n_fail = sum(1 for _, ok, _ in T.RESULTS if not ok)
    print(f"SUMMARY {len(T.RESULTS) - n_fail}/{len(T.RESULTS)} passed{' (old module: repro checks)' if OLD else ''}",
          flush=True)
    shutil.rmtree(root, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
