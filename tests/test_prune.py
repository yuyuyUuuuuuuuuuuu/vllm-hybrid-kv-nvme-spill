# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests of superseded resume-point pruning (GLM53_NVME_SPILL_KEEP_PER_CHAIN, 2026-10-13).

Real ``SpillScheduler`` against the fake BlockPool/coordinator of test_sched_faults.py (vLLM ref-count semantics);
the delete path is checked on the scheduler side (SpillMeta.deletes, the single list every rank receives) and, at the
end, with a real ``SpillWorker`` on CPU tensors and real files (prune deletes reach the rank's writer FIFO ahead of a
later re-write of the same file).

usage: python3 test_prune.py --dir <scratch> [--module <connector .py>]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sched_faults as T  # noqa: E402

HBS = T.HBS
check = T.check


def req_h(rid, hashes):
    """A request whose block hash chain is ``hashes`` (len = blocks + 1, like make_req)."""
    n = (len(hashes) - 1) * HBS + 7
    return T.Req(request_id=rid, block_hashes=list(hashes), num_tokens=n, num_computed_tokens=n, num_prompt_tokens=n,
                 skip_reading_prefix_cache=False)


def chain(tag: bytes, n: int) -> list[bytes]:
    return [tag + i.to_bytes(4, "big") for i in range(n)]


def save(spill, s, pool, req, nb, answer_ok=True):
    """compute + save; returns (job or None, meta of the step that shipped it). answer_ok=None leaves it in flight."""
    T.compute(pool, req, nb)
    s.save_request(req)
    m = s.build_meta()
    if not m.saves:
        return None, m
    (job,) = list(m.saves)
    if answer_ok is not None:
        T.answer(spill, s, job, 2 if answer_ok else 1, 0 if answer_ok else 1)
    return job, m


def state_fname(spill, h):
    return spill.block_file_name(T.K(h, 2))


def mla_fname(spill, h):
    return spill.block_file_name(T.K(h, 0))


def ns(s):
    return sorted(rp.n_tokens // HBS for rp in s.rps.values())


# ------------------------------------------------------------------------------------------------ tests
def t_linear(spill, root, keep):
    s, pool, _, _ = T.new_sched(spill, f"{root}/lin{keep}", n=96, keep_per_chain=keep)
    c = chain(b"a", 12)
    sizes = []
    for nb in (3, 4, 5, 6):
        save(spill, s, pool, req_h(f"t{nb}", c[:nb + 1]), nb)
        sizes.append(ns(s))
    want = {0: [[3], [3, 4], [3, 4, 5], [3, 4, 5, 6]],
            1: [[3], [4], [5], [6]],
            2: [[3], [3, 4], [4, 5], [5, 6]],
            3: [[3], [3, 4], [3, 4, 5], [4, 5, 6]]}[keep]
    check(f"lin_k{keep}_rps_after_each_turn", sizes == want, f"{sizes} want {want}")
    n_pruned = {0: 0, 1: 3, 2: 2, 3: 1}[keep]
    st_bytes = s.file_bytes[2]
    check(f"lin_k{keep}_pruned_count_and_bytes", s.stats["rp_pruned"] == n_pruned
          and s.stats["prune_freed_bytes"] == n_pruned * st_bytes,
          f"pruned={s.stats['rp_pruned']} freed={s.stats['prune_freed_bytes']} (state file {st_bytes})")
    # only the pruned RPs' own state files are gone; every MLA file is still referenced by the newest RP
    kept_states = {state_fname(spill, c[k - 1]) for k in want[-1]}
    all_states = {state_fname(spill, c[k - 1]) for k in (3, 4, 5, 6)}
    check(f"lin_k{keep}_disk_has_exactly_kept_states", {f for f in all_states if f in s.disk} == kept_states)
    check(f"lin_k{keep}_mla_files_all_kept", all(mla_fname(spill, c[i]) in s.disk for i in range(6)))
    exp = sum(s.file_bytes[0] for _ in range(6)) + len(kept_states) * st_bytes
    check(f"lin_k{keep}_disk_bytes_accounting", s.disk_bytes == exp, f"{s.disk_bytes} != {exp}")
    return s


def t_deletes_shipped(spill, root):
    """Prune deletes leave in the next SpillMeta (one list, the same for every rank) and only for refs==0 files."""
    s, pool, _, _ = T.new_sched(spill, f"{root}/ship", n=96, keep_per_chain=2)
    c = chain(b"b", 10)
    save(spill, s, pool, req_h("t3", c[:4]), 3)
    save(spill, s, pool, req_h("t4", c[:5]), 4)
    T.compute(pool, req_h("t5", c[:6]), 5)
    s.save_request(req_h("t5", c[:6]))
    m = s.build_meta()
    (job,) = list(m.saves)
    check("ship_no_prune_while_new_rp_saving", s.stats["rp_pruned"] == 0 and not m.deletes)
    T.answer(spill, s, job, 1, 0)
    check("ship_no_prune_on_one_rank_answer", s.stats["rp_pruned"] == 0)
    T.answer(spill, s, job, 1, 0)
    m2 = s.build_meta()
    check("ship_prune_after_all_ranks_ok", s.stats["rp_pruned"] == 1 and m2.deletes == [state_fname(spill, c[2])],
          f"deletes={m2.deletes}")
    check("ship_seen_keeps_pruned_key_for_repeat_policy", bytes(c[2]) in s.seen)


def t_failed_new_save(spill, root):
    """A new RP whose save fails on a rank never prunes anything."""
    s, pool, _, _ = T.new_sched(spill, f"{root}/fail", n=96, keep_per_chain=1)
    c = chain(b"f", 10)
    save(spill, s, pool, req_h("t3", c[:4]), 3)
    save(spill, s, pool, req_h("t4", c[:5]), 4, answer_ok=False)
    check("fail_older_rp_kept_when_new_save_fails", ns(s) == [3] and s.stats["rp_pruned"] == 0, f"{ns(s)}")


def t_busy(spill, root):
    """An RP being restored (loading, ready+pinned, admitted with its request still running) is never pruned; it is
    pruned at a later save of the chain once its request finished."""
    s, pool, _, _ = T.new_sched(spill, f"{root}/busy", n=128, keep_per_chain=2)
    c = chain(b"c", 12)
    save(spill, s, pool, req_h("t3", c[:4]), 3)
    save(spill, s, pool, req_h("t4", c[:5]), 4)
    # an edit of turn 4: branches after block 3 -> only RP(3) is its prefix
    pool.evict_all_cached()
    edit = req_h("edit", c[:3] + chain(b"E", 3))
    r = s.get_num_new_matched_tokens(edit, 0)
    st = s.restores.get("edit")
    check("busy_edit_restores_previous_rp", r == (None, False) and st is not None and st.rp.n_tokens == 3 * HBS,
          f"{r} {st and st.rp.n_tokens}")
    mload = s.build_meta()
    (ljob,) = list(mload.loads)
    save(spill, s, pool, req_h("t5", c[:6]), 5)
    check("busy_loading_rp_not_pruned", ns(s) == [3, 4, 5] and s.stats["rp_prune_skipped_busy"] == 1, f"{ns(s)}")
    T.answer(spill, s, ljob, 2, 0)
    check("busy_restore_ready", s.restores["edit"].state == "ready")
    pinned = list(s.restores["edit"].new_blocks) + list(s.restores["edit"].pins)
    save(spill, s, pool, req_h("t6", c[:7]), 6)
    check("busy_ready_pinned_rp_not_pruned", ns(s) == [3, 5, 6] and all(b.ref_cnt >= 1 for b in pinned),
          f"{ns(s)}")
    s.update_state_after_alloc(edit)        # admitted: pins released, request still running
    save(spill, s, pool, req_h("t7", c[:8]), 7)
    check("busy_admitted_running_rp_not_pruned", 3 in ns(s), f"{ns(s)}")
    s.request_finished(edit)
    save(spill, s, pool, req_h("t8", c[:9]), 8)
    check("busy_pruned_after_request_finished", ns(s) == [7, 8], f"{ns(s)}")
    check("busy_no_leaked_pins", s.pinned_blocks() == 0)


def t_branch_keep(spill, root):
    """A sibling branch (an edited turn) keeps its predecessor; files referenced by an in-flight save survive."""
    s, pool, _, _ = T.new_sched(spill, f"{root}/br2", n=128, keep_per_chain=2)
    c = chain(b"d", 12)
    save(spill, s, pool, req_h("t3", c[:4]), 3)
    save(spill, s, pool, req_h("t4", c[:5]), 4)
    sib = c[:3] + chain(b"S", 3)                       # branches after RP(3): blocks c0 c1 c2 S0 S1
    jb, _ = save(spill, s, pool, req_h("sib", sib), 5, answer_ok=None)     # in flight
    save(spill, s, pool, req_h("t5", c[:6]), 5)
    check("branch_sibling_in_flight_keeps_its_predecessor", 3 in ns(s) and s.stats["rp_prune_kept_branch"] == 1,
          f"{ns(s)} kept_branch={s.stats['rp_prune_kept_branch']}")
    T.answer(spill, s, jb, 2, 0)
    sib_rp = s.rps.get(bytes(sib[4]))
    check("branch_sibling_ok", sib_rp is not None and sib_rp.state == "ok")
    save(spill, s, pool, req_h("t6", c[:7]), 6)
    check("branch_predecessor_still_kept_for_live_sibling", 3 in ns(s), f"{ns(s)}")
    # the sibling goes away (LRU / bad file): the next save of the main chain prunes the old predecessor
    s._drop_rp(sib_rp, "lru")
    save(spill, s, pool, req_h("t7", c[:8]), 7)
    check("branch_predecessor_pruned_after_sibling_gone", sorted(n for n in ns(s)) == [6, 7], f"{ns(s)}")

    # K=1: the in-flight sibling does not protect, but its files (shared MLA) must survive by refcount
    s, pool, _, _ = T.new_sched(spill, f"{root}/br1", n=128, keep_per_chain=1)
    save(spill, s, pool, req_h("t3", c[:4]), 3)
    jb, _ = save(spill, s, pool, req_h("sib", sib), 5, answer_ok=None)
    save(spill, s, pool, req_h("t4", c[:5]), 4)
    m = s.build_meta()
    shared = {mla_fname(spill, h) for h in c[:3]}
    check("k1_rp3_pruned", ns(s) == [4, 5], f"{ns(s)}")
    check("k1_shared_files_of_in_flight_save_not_deleted", not (shared & set(m.deletes))
          and all(f in s.disk for f in shared), f"deletes={len(m.deletes)}")
    T.answer(spill, s, jb, 2, 0)
    sib_rp = s.rps.get(bytes(sib[4]))
    check("k1_sibling_completes_and_is_restorable", sib_rp is not None and sib_rp.state == "ok"
          and all(s.disk[e.fname].state == "ok" for e in sib_rp.blocks))


def t_regen_same_key(spill, root):
    """A regenerate (the same prompt again) re-finds the same RP: no new RP, nothing pruned."""
    s, pool, _, _ = T.new_sched(spill, f"{root}/regen", n=96, keep_per_chain=1)
    c = chain(b"g", 10)
    save(spill, s, pool, req_h("t3", c[:4]), 3)
    save(spill, s, pool, req_h("t4", c[:5]), 4)
    save(spill, s, pool, req_h("t4b", c[:5]), 4)
    check("regen_same_rp_no_prune", ns(s) == [4] and s.stats["rp_pruned"] == 1, f"{ns(s)}")


def t_other_conversations(spill, root):
    """Pruning never touches another conversation (even one sharing the head blocks)."""
    s, pool, _, _ = T.new_sched(spill, f"{root}/other", n=128, keep_per_chain=1)
    a = chain(b"h", 10)
    b = a[:2] + chain(b"B", 8)       # shares the first two blocks (the common head), then its own conversation
    save(spill, s, pool, req_h("b3", b[:4]), 3)
    save(spill, s, pool, req_h("a3", a[:4]), 3)
    save(spill, s, pool, req_h("a4", a[:5]), 4)
    save(spill, s, pool, req_h("a5", a[:6]), 5)
    check("other_conversation_untouched", bytes(b[2]) in s.rps and ns(s) == [3, 5], f"{ns(s)}")
    check("other_shared_head_files_kept", all(mla_fname(spill, h) in s.disk for h in a[:2]))


def t_cap_crowding(spill, root):
    """Under the capacity cap, old turns of one conversation no longer push another conversation out (LRU)."""
    out = {}
    for keep in (0, 2):
        s, pool, _, _ = T.new_sched(spill, f"{root}/cap{keep}", n=128, keep_per_chain=keep)
        fb = s.file_bytes
        # B (3 MLA + 1 state) first = LRU-oldest; then A's turns; cap fits B + A's 6 MLA + 3 states
        s.cfg.cap_bytes = 9 * fb[0] + 4 * fb[2]
        b = chain(b"q", 6)
        a = chain(b"p", 10)
        save(spill, s, pool, req_h("b3", b[:4]), 3)
        for nb in (3, 4, 5, 6):
            save(spill, s, pool, req_h(f"a{nb}", a[:nb + 1]), nb)
        out[keep] = (bytes(b[2]) in s.rps, s.stats["rp_evicted"], s.disk_bytes <= s.cfg.cap_bytes)
    check("cap_k0_lru_evicts_other_conversation", out[0][0] is False and out[0][1] >= 1, f"{out[0]}")
    check("cap_k2_other_conversation_survives", out[2][0] is True and out[2][1] == 0 and out[2][2], f"{out[2]}")


def t_cap_shared_head(spill, root):
    """Every DSH conversation shares the head blocks. At the cap, LRU must still evict another conversation's RP (its
    own files go, the shared head files stay because the new RP references them) instead of skipping the new save
    (before: _make_room skipped every RP sharing a file with the new one = every DSH RP -> saves silently stopped)."""
    for keep in (0, 2):
        s, pool, _, _ = T.new_sched(spill, f"{root}/caph{keep}", n=128, keep_per_chain=keep)
        fb = s.file_bytes
        head = chain(b"H", 2)
        b = head + chain(b"Y", 6)
        a = head + chain(b"X", 8)
        s.cfg.cap_bytes = 7 * fb[0] + 2 * fb[2]
        dels: set[str] = set()
        orig_del = s._delete_disk

        def rec_del(db, _o=orig_del):
            dels.add(db.fname)
            return _o(db)

        s._delete_disk = rec_del
        save(spill, s, pool, req_h("b4", b[:5]), 4)      # B: head(2) + 2 own MLA + state = 4 MLA + 1 state
        save(spill, s, pool, req_h("a4", a[:5]), 4)      # A: 2 more own MLA + state -> 6 MLA + 2 states (fits)
        save(spill, s, pool, req_h("a5", a[:6]), 5)      # +1 MLA +1 state -> over the cap: B must go
        a5 = s.rps.get(bytes(a[4]))
        check(f"caph_k{keep}_new_save_not_skipped", a5 is not None and a5.state == "ok" and s.stats["rp_skipped_cap"] == 0,
              f"skipped_cap={s.stats['rp_skipped_cap']} rps={ns(s)}")
        check(f"caph_k{keep}_other_conversation_evicted", bytes(b[3]) not in s.rps and s.stats["rp_evicted"] >= 1,
              f"evicted={s.stats['rp_evicted']}")
        check(f"caph_k{keep}_shared_head_files_kept", all(mla_fname(spill, h) in s.disk and mla_fname(spill, h) not in dels
                                                         for h in head))
        check(f"caph_k{keep}_evicted_own_files_deleted", {mla_fname(spill, b[2]), mla_fname(spill, b[3]),
                                                          state_fname(spill, b[3])} <= dels)
        check(f"caph_k{keep}_every_file_of_live_rps_present", all(e.fname in s.disk for rp in s.rps.values()
                                                                  for e in rp.blocks))
        check(f"caph_k{keep}_under_cap_and_refs_consistent", s.disk_bytes <= s.cfg.cap_bytes and all(
            db.refs == sum(1 for rp in s.rps.values() for e in rp.blocks if e.fname == f) for f, db in s.disk.items()),
            f"disk={s.disk_bytes} cap={s.cfg.cap_bytes}")
    # a save that can not fit even after evicting everything evictable leaves no tentative reference behind
    s, pool, _, _ = T.new_sched(spill, f"{root}/caph_fail", n=128, keep_per_chain=2)
    fb = s.file_bytes
    head = chain(b"H", 2)
    a = head + chain(b"X", 8)
    s.cfg.cap_bytes = 4 * fb[0] + 1 * fb[2]
    save(spill, s, pool, req_h("a4", a[:5]), 4)
    save(spill, s, pool, req_h("a7", a[:8]), 7)        # needs 3 more MLA + state: can not fit (a4 is its prefix)
    check("caph_unfit_save_evicts_nothing", bytes(a[3]) in s.rps and s.stats["rp_evicted"] == 0, f"rps={ns(s)}")
    check("caph_unfit_save_skipped_refs_restored", s.stats["rp_skipped_cap"] == 1 and all(
        db.refs == sum(1 for rp in s.rps.values() for e in rp.blocks if e.fname == f) for f, db in s.disk.items())
        and s.disk_bytes <= s.cfg.cap_bytes, f"skipped={s.stats['rp_skipped_cap']} rps={ns(s)}")


def t_breaker(spill, root):
    s, pool, _, _ = T.new_sched(spill, f"{root}/brk", n=96, keep_per_chain=1)
    c = chain(b"k", 8)
    save(spill, s, pool, req_h("t3", c[:4]), 3)
    s._trip("test")
    check("breaker_forgets_all_no_prune_counted", not s.rps and s.stats["rp_pruned"] == 0)


def t_config(spill, root):
    os.environ["GLM53_NVME_SPILL_DIR"] = root
    for raw, want in (("", 2), (None, 2), ("0", 0), ("3", 3)):
        if raw is None:
            os.environ.pop("GLM53_NVME_SPILL_KEEP_PER_CHAIN", None)
        else:
            os.environ["GLM53_NVME_SPILL_KEEP_PER_CHAIN"] = raw
        check(f"cfg_keep_{raw!r}", spill.SpillConfig.from_env().keep_per_chain == want)
    os.environ["GLM53_NVME_SPILL_KEEP_PER_CHAIN"] = "-1"
    try:
        spill.SpillConfig.from_env()
        check("cfg_keep_negative_rejected", False)
    except ValueError:
        check("cfg_keep_negative_rejected", True)
    os.environ.pop("GLM53_NVME_SPILL_KEEP_PER_CHAIN", None)
    # (2026-10-14: the TTL / persistence knobs are appended after it; the order of the earlier knobs is unchanged)
    check("knob_listed_for_start_sh", "KEEP_PER_CHAIN" in spill.KNOBS
          and spill.KNOBS.index("KEEP_PER_CHAIN") == 22)


def t_stats_line(spill, root):
    s = t_linear(spill, root + "/sl", 2)
    line = s.stats_line()
    old = ["breaker", "pinned", "save", "save_excl", "restore", "quarantined", "peak", "excl_peak", "cap",
           "inflight_save", "oldest_save_s", "inflight_load", "oldest_load_s", "worker_exc", "jobs_failed",
           "save_timeouts", "load_timeouts", "written_gib_1h", "disk_gib", "rps", "restores_ok", "restores_failed",
           "restores_fallback", "restore_short", "saves_ok", "saves_failed"]
    new = ["rp_pruned", "pruned_gib", "rp_evicted", "keep", "skipped_cap"]
    keys = [kv.split("=", 1)[0] for kv in line.split(" stats ", 1)[1].split()]
    # (2026-10-14: TTL / persistence fields are appended after these; a prefix match keeps monitor.sh compatible)
    check("stats_line_old_fields_unchanged_new_appended", keys[:len(old + new)] == old + new, f"{keys}")
    check("stats_line_values", "rp_pruned=2 " in line and f"pruned_gib={2 * s.file_bytes[2] / (1 << 30):.2f}" in line,
          line[-120:])


def t_worker_fifo(spill, root):
    """Real worker: a prune delete and a later re-write of the same file in one step end with the file written."""
    w = T.make_worker(spill, root + "/wk")
    st = w.store
    fname = "x" * 40 + ".blk"
    w.handle(spill.SpillMeta(saves={1: [(2, 3, fname)]}))
    w.drain(30)
    m = w.take_meta()
    ok1 = m is not None and m.done.get(1) == [1, 0] and os.path.exists(os.path.join(st.dir, fname))
    w.handle(spill.SpillMeta(deletes=[fname], saves={2: [(2, 4, fname)]}))
    w.drain(30)
    m = w.take_meta()
    check("worker_delete_then_rewrite_same_step", ok1 and m is not None and m.done.get(2) == [1, 0]
          and os.path.exists(os.path.join(st.dir, fname)))
    w.handle(spill.SpillMeta(deletes=[fname]))
    w.drain(30)
    check("worker_prune_delete_removes_file", not os.path.exists(os.path.join(st.dir, fname)))
    w.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--module", default=os.path.join(HERE, "..", "code", "glm53_nvme_spill_connector.py"))
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("spill_under_test", a.module)
    spill = importlib.util.module_from_spec(spec)
    sys.modules["spill_under_test"] = spill
    spec.loader.exec_module(spill)
    root = os.path.abspath(a.dir)
    os.makedirs(root, exist_ok=True)
    for keep in (0, 1, 2, 3):
        t_linear(spill, root, keep)
    t_deletes_shipped(spill, root)
    t_failed_new_save(spill, root)
    t_busy(spill, root)
    t_branch_keep(spill, root)
    t_regen_same_key(spill, root)
    t_other_conversations(spill, root)
    t_cap_crowding(spill, root)
    t_cap_shared_head(spill, root)
    t_breaker(spill, root)
    t_config(spill, root)
    t_stats_line(spill, root)
    t_worker_fifo(spill, root)
    n_fail = sum(1 for _, ok, _ in T.RESULTS if not ok)
    print(f"SUMMARY {len(T.RESULTS) - n_fail}/{len(T.RESULTS)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
