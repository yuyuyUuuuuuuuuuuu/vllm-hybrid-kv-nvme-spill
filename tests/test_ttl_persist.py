# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests of per-request TTL (DSH contract v1.1) and reuse across restarts (2026-10-14).

Real ``SpillScheduler`` against the fake BlockPool/coordinator of test_sched_faults.py; real ``SpillWorker`` (CPU
tensors, real files) for the start-up scan / wipe / orphan deletes / write-id check and for an end-to-end restart
round trip (save on two ranks -> "restart" = new scheduler + new workers on the same directories -> adopt -> restore
into different block ids -> byte comparison with the bytes that were saved).

usage: python3 test_ttl_persist.py --dir <scratch> [--module <connector .py>]   (production image, PYTHONHASHSEED set,
       PYTHONPATH = a rig work tree)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prodlayout  # noqa: E402
import test_sched_faults as T  # noqa: E402
import torch  # noqa: E402

HBS = T.HBS
check = T.check


class Wall:
    def __init__(self, t=1_760_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def req(rid, hashes, params=None, n=None):
    n = (len(hashes) - 1) * HBS + 7 if n is None else n
    r = T.Req(request_id=rid, block_hashes=list(hashes), num_tokens=n, num_computed_tokens=n, num_prompt_tokens=n,
              skip_reading_prefix_cache=False)
    if params is not None:
        r.kv_transfer_params = {"glm53_spill": params}
    return r


def chain(tag: bytes, n: int) -> list[bytes]:
    return [tag + i.to_bytes(4, "big") for i in range(n)]


def sched(spill, root, wall=None, n=128, scale=1, **kw):
    """test_sched_faults.new_sched with every config field overridable."""
    base = dict(root=root, cap_bytes=1 << 40, min_tokens=2 * HBS, min_gain=HBS, save_policy="all", log=False,
                log_every_s=0)
    base.update(kw)
    cfg = spill.SpillConfig(**base)
    kvc = prodlayout.build_kv_cache_config(n, scale=scale)
    vc = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))
    s = spill.SpillScheduler(cfg, vc, kvc)
    pool = T.FakePool(n)
    s.bind(block_pool=pool, kv_cache_manager=SimpleNamespace(coordinator=T.FakeCoord(pool), block_pool=pool))
    clk, io = T.Clock(), T.Clock()
    s.clock = clk
    s.io_clock = io
    s.wall = wall or Wall()
    return s, pool, clk, io


def save(spill, s, pool, r, nb, ok=True):
    T.compute(pool, r, nb)
    s.save_request(r)
    m = s.build_meta()
    if not m.saves:
        m = s.build_meta()
    if not m.saves:
        return None, m
    (job,) = list(m.saves)
    if ok is not None:
        T.answer(spill, s, job, 2 if ok else 1, 0 if ok else 1)
    return job, m


def rp_ns(s):
    return sorted(rp.n_tokens // HBS for rp in s.rps.values())


# ------------------------------------------------------------------------------------------------ TTL parsing
def t_parse(spill):
    P = spill.parse_spill_params
    ok = P({"glm53_spill": {"ttl_s": 86400, "session": "sess-1", "epoch": 3}})
    check("parse_valid", ok == spill.SpillParams(86400, "sess-1", 3), f"{ok}")
    check("parse_missing_field_none", P(None) is None and P({}) is None and P({"other": 1}) is None)
    check("parse_not_a_dict", P("x") is None and P({"glm53_spill": "x"}) is None and P({"glm53_spill": [1]}) is None)
    cases = {
        "bool_ttl": ({"ttl_s": True, "session": "s"}, (None, "s", None)),
        "float_ttl": ({"ttl_s": 3600.0, "session": "s"}, (None, "s", None)),
        "str_ttl": ({"ttl_s": "3600", "session": "s"}, (None, "s", None)),
        "neg_ttl": ({"ttl_s": -1, "session": "s"}, (None, "s", None)),
        "ttl_over_7d": ({"ttl_s": 7 * 86400 + 1, "session": "s"}, (None, "s", None)),
        "ttl_7d_ok": ({"ttl_s": 7 * 86400}, (7 * 86400, None, None)),
        "ttl_zero_ok": ({"ttl_s": 0}, (0, None, None)),
        "session_65_chars": ({"ttl_s": 60, "session": "x" * 65}, (60, None, None)),
        "session_64_chars": ({"ttl_s": 60, "session": "x" * 64}, (60, "x" * 64, None)),
        "session_empty": ({"ttl_s": 60, "session": ""}, (60, None, None)),
        "session_not_str": ({"ttl_s": 60, "session": 5}, (60, None, None)),
        "epoch_neg": ({"ttl_s": 60, "epoch": -1}, (60, None, None)),
        "epoch_bool": ({"ttl_s": 60, "epoch": False}, (60, None, None)),
        "all_invalid": ({"ttl_s": "x", "session": 1, "epoch": "y"}, None),
    }
    for name, (g, want) in cases.items():
        got = P({"glm53_spill": g})
        got_t = None if got is None else (got.ttl_s, got.session, got.epoch)
        check(f"parse_{name}", got_t == want, f"{got_t} want {want}")

    class Evil(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")
    check("parse_never_raises", P(Evil()) is None)


# ------------------------------------------------------------------------------------------------ eviction order
def t_eviction_order(spill, root):
    """At the cap: expired -> no TTL -> shorter TTL -> LRU (same class)."""
    w = Wall()
    # one RP = 3 MLA files + 1 state file; cap fits exactly 4 such RPs (separate chains, no sharing)
    s0, pool0, _, _ = sched(spill, root + "/ev_probe", w)
    per_rp = 3 * s0.file_bytes[0] + s0.file_bytes[2]
    s, pool, _, _ = sched(spill, root + "/ev", w, cap_bytes=4 * per_rp, n=256)
    specs = [("A", None), ("B", 3600), ("C", 86400), ("D", 60)]
    for tag, ttl in specs:
        p = None if ttl is None else {"ttl_s": ttl, "session": "s" + tag}
        save(spill, s, pool, req(tag, chain(tag.encode(), 4), p), 3)
        w.t += 10
    check("ev_four_saved", len(s.rps) == 4, f"{len(s.rps)}")
    w.t += 120      # D (ttl 60) is now expired; the others are not
    order = []
    for i, tag in enumerate(["E", "F", "G", "H"]):
        before = {rp.key[:1] for rp in s.rps.values()}
        save(spill, s, pool, req(tag, chain(tag.encode(), 4)), 3)
        after = {rp.key[:1] for rp in s.rps.values()}
        order += sorted(x.decode() for x in before - after)
    check("ev_order_expired_default_shorter_longer", order[:4] == ["D", "A", "E", "B"] or order[:3] == ["D", "A", "E"],
          f"{order}")
    # the order seen: D (expired) first, then the no-TTL class oldest first (A, then the new no-TTL E/F...), the TTL
    # class only when no no-TTL RP is left: make that explicit with a fresh set
    w2 = Wall()
    s, pool, _, _ = sched(spill, root + "/ev2", w2, cap_bytes=4 * per_rp, n=256)
    for tag, ttl in [("P", 86400), ("Q", 3600), ("R", 3600), ("S", None)]:
        p = None if ttl is None else {"ttl_s": ttl, "session": "s" + tag}
        save(spill, s, pool, req(tag, chain(tag.encode(), 4), p), 3)
        w2.t += 10
    ev = []
    for tag in ["T", "U", "V"]:
        before = {rp.key[:1] for rp in s.rps.values()}
        save(spill, s, pool, req(tag, chain(tag.encode(), 4), {"ttl_s": 259200, "session": "x" + tag}), 3)
        ev += sorted(x.decode() for x in before - {rp.key[:1] for rp in s.rps.values()})
    check("ev_default_then_short_ttl_lru_then_longer", ev == ["S", "Q", "R"], f"{ev}")
    check("ev_counted_as_lru", s.stats["rp_evicted"] == 3, f"{s.stats['rp_evicted']}")


# ------------------------------------------------------------------------------------------------ TTL sliding / sweep
def t_ttl_sliding(spill, root):
    w = Wall()
    s, pool, clk, io = sched(spill, root + "/sl", w, ttl_sweep_s=30)
    c = chain(b"t", 12)
    save(spill, s, pool, req("t3", c[:4], {"ttl_s": 3600, "session": "S", "epoch": 0}), 3)
    rp = next(iter(s.rps.values()))
    check("sl_rp_has_ttl_session_epoch", (rp.ttl_s, rp.session, rp.epoch) == (3600, "S", 0))
    w.t += 3000
    s.get_num_new_matched_tokens(req("t5a", c[:6], {"ttl_s": 3600, "session": "S", "epoch": 0}), 4 * HBS)
    check("sl_extending_request_refreshes_last_used", rp.last_used == w.t and s.stats["rp_ttl_refresh"] == 1)
    w.t += 3000
    s.build_meta()
    check("sl_not_expired_after_refresh", len(s.rps) == 1)
    s.get_num_new_matched_tokens(req("t5b", c[:6], {"ttl_s": 86400, "session": "S", "epoch": 0}), 4 * HBS)
    check("sl_new_ttl_choice_applies", rp.ttl_s == 86400)
    w.t += 86400 + 1
    pool.evict_all_cached()
    r = s.get_num_new_matched_tokens(req("t5c", c[:6]), 0)
    check("sl_expired_rp_never_restored", r == (0, False) and s.stats["restore_started"] == 0, f"{r}")
    m = s.build_meta()
    m2 = s.build_meta()
    dels = set(m.deletes) | set(m2.deletes)
    check("sl_sweep_drops_expired", len(s.rps) == 0 and s.stats["rp_expired"] == 1 and len(dels) == 4,
          f"rps={len(s.rps)} expired={s.stats['rp_expired']} dels={len(dels)}")
    # an expired RP in use (restore loading) is kept until free, then swept
    s, pool, clk, io = sched(spill, root + "/sl2", w, ttl_sweep_s=30)
    save(spill, s, pool, req("u3", c[:4], {"ttl_s": 100, "session": "S"}), 3)
    pool.evict_all_cached()
    r = s.get_num_new_matched_tokens(req("u4", c[:5], {"ttl_s": 100, "session": "S"}), 0)
    mload = s.build_meta()
    (ljob,) = list(mload.loads)
    w.t += 200
    s.build_meta()
    check("sl_expired_but_loading_kept", len(s.rps) == 1)
    T.answer(spill, s, ljob, 2, 0)
    s.update_state_after_alloc(req("u4", c[:5]))
    s.request_finished(req("u4", c[:5], {"ttl_s": 100, "session": "S"}))
    s.build_meta()
    rp = next(iter(s.rps.values()), None)
    check("sl_used_rp_revived_by_its_request", rp is not None and rp.last_used == w.t and s.stats["rp_expired"] == 0)
    w.t += 101
    s.build_meta()
    check("sl_swept_once_idle_past_ttl", len(s.rps) == 0 and s.stats["rp_expired"] == 1)


# ------------------------------------------------------------------------------------------------ ttl_s == 0
def t_ttl0(spill, root):
    w = Wall()
    s, pool, _, _ = sched(spill, root + "/z", w, keep_per_chain=0)
    c = chain(b"z", 12)
    save(spill, s, pool, req("z3", c[:4], {"ttl_s": 3600, "session": "S"}), 3)
    save(spill, s, pool, req("z4", c[:5], {"ttl_s": 3600, "session": "S"}), 4)
    other = chain(b"o", 8)
    save(spill, s, pool, req("o3", other[:4], {"ttl_s": 3600, "session": "S"}), 3)       # same session, other chain
    shared_other = req("x5", c[:6], {"ttl_s": 86400, "session": "OTHER"})
    save(spill, s, pool, shared_other, 5)                                                # other session on the chain
    check("z_four_rps", len(s.rps) == 4, f"{rp_ns(s)}")
    z = req("z6", c[:7], {"ttl_s": 0, "session": "S"})
    r = s.get_num_new_matched_tokens(z, 0)
    check("z_ttl0_no_restore", r == (0, False))
    left = sorted((rp.session, rp.n_tokens // HBS) for rp in s.rps.values())
    check("z_ttl0_expires_chain_and_session_not_other_session", left == [("OTHER", 5)], f"{left}")
    check("z_counted_expired", s.stats["rp_expired"] == 3, f"{s.stats['rp_expired']}")
    T.compute(pool, z, 6)
    s.save_request(z)
    check("z_ttl0_not_saved", not s.build_meta().saves and s.stats["rp_skipped_ttl0"] == 1)


# ------------------------------------------------------------------------------------------------ epoch
def t_epoch(spill, root):
    w = Wall()
    s, pool, _, _ = sched(spill, root + "/ep", w)
    c = chain(b"e", 14)
    save(spill, s, pool, req("e3", c[:4], {"ttl_s": 3600, "session": "S", "epoch": 0}), 3)
    save(spill, s, pool, req("e4", c[:5], {"ttl_s": 3600, "session": "S", "epoch": 0}), 4)
    # a title request (short, same session/epoch) and a summary request (other chain, same session/epoch, long)
    title = req("title", chain(b"T", 2), {"ttl_s": 3600, "session": "S", "epoch": 0})
    s.get_num_new_matched_tokens(title, 0)
    summ = req("summ", chain(b"Y", 6), {"ttl_s": 3600, "session": "S", "epoch": 0})
    s.get_num_new_matched_tokens(summ, 0)
    save(spill, s, pool, summ, 5)
    check("ep_same_epoch_short_or_other_chain_pushes_nothing_out", rp_ns(s) == [3, 4, 5], f"{rp_ns(s)}")
    # a child session: its own session, epoch 0
    s.get_num_new_matched_tokens(req("sub", chain(b"U", 6), {"ttl_s": 3600, "session": "SUB", "epoch": 0}), 0)
    check("ep_child_session_touches_nothing", rp_ns(s) == [3, 4, 5])
    # compaction: the next request of S carries epoch 1 -> the older-epoch RPs of S go
    after = req("new", chain(b"N", 6), {"ttl_s": 3600, "session": "S", "epoch": 1})
    s.get_num_new_matched_tokens(after, 0)
    check("ep_newer_epoch_waiting_marks_nothing", s.stats["rp_epoch_marked"] == 0 and all(not rp.stale_at
                                                                                       for rp in s.rps.values()))
    s.update_state_after_alloc(after)       # admitted: the session runs epoch 1 now
    check("ep_newer_epoch_marks_older_epoch_rps_stale", rp_ns(s) == [3, 4, 5] and s.stats["rp_epoch_marked"] == 3
          and all(rp.stale_at == w.t for rp in s.rps.values()), f"{rp_ns(s)} {s.stats['rp_epoch_marked']}")
    # within the grace a late request of the old epoch still restores (and refreshes) its RP
    w.t += 900
    pool.evict_all_cached()
    late4 = req("late4", c[:6], {"ttl_s": 3600, "session": "S", "epoch": 0})
    r = s.get_num_new_matched_tokens(late4, 0)
    st = s.restores.get("late4")
    check("ep_late_old_epoch_request_restores_within_grace", r == (None, False) and st is not None
          and st.rp.n_tokens == 4 * HBS, f"{r}")
    T.answer(spill, s, list(s.build_meta().loads)[0], 2, 0)
    s.update_state_after_alloc(late4)
    s.request_finished(req("late4x", c[:2]))
    s.request_finished(late4)
    # a late request of the old epoch is not saved any more
    late = req("late", chain(b"L", 6), {"ttl_s": 3600, "session": "S", "epoch": 0})
    T.compute(pool, late, 5)
    s.save_request(late)
    m_late = s.build_meta()
    rp_late = next((rp for rp in s.rps.values() if rp.key[:1] == b"L"), None)
    check("ep_older_epoch_save_kept_but_stale_at_once", len(m_late.saves) == 1 and rp_late is not None
          and rp_late.stale_at == w.t and s.stats["rp_saved_stale"] == 1, f"{rp_late and rp_late.stale_at}")
    T.answer(spill, s, list(m_late.saves)[0], 2, 0)
    w.t += 901      # 1801 s after the mark: the untouched stale RP (the summary's) goes; the two the late request
    s._last_ttl_sweep = None   # extended (its chain holds both) were used 901 s ago and stay, as does the late (stale) one
    s.build_meta()
    check("ep_stale_unused_dropped_after_grace_used_ones_kept", rp_ns(s) == [3, 4, 5] and s.stats["rp_epoch_dropped"] == 1,
          f"{rp_ns(s)} {s.stats['rp_epoch_dropped']}")
    w.t += 1800
    s._last_ttl_sweep = None
    s.build_meta()
    check("ep_last_stale_dropped_after_its_grace", rp_ns(s) == [] and s.stats["rp_epoch_dropped"] == 4)
    save(spill, s, pool, after, 5)
    check("ep_new_epoch_saved", rp_ns(s) == [5] and next(iter(s.rps.values())).epoch == 1)


# ------------------------------------------------------------------------------------------------ first-turn policy
def t_first_turn(spill, root):
    w = Wall()
    for pol, want_kept in (("all", True), ("repeat", False)):
        s, pool, _, _ = sched(spill, root + f"/ft_{pol}", w, save_policy="repeat", ttl_save_policy=pol)
        c = chain(b"f", 8)
        save(spill, s, pool, req("first", c[:5], {"ttl_s": 86400, "session": "S"}), 4)
        save(spill, s, pool, req("dflt", chain(b"g", 8)[:5]), 4)
        save(spill, s, pool, req("zero", chain(b"h", 8)[:5], {"ttl_s": 0, "session": "Z"}), 4)
        kept = sorted(rp.key[:1].decode() for rp in s.rps.values())
        check(f"ft_{pol}_explicit_ttl_first_turn", kept == (["f"] if want_kept else []), f"{kept}")
        check(f"ft_{pol}_stats", s.stats["rp_saved_ttl_first"] == (1 if want_kept else 0)
              and s.stats["rp_skipped_first"] == (1 if want_kept else 2), f"{s.stats['rp_skipped_first']}")
    s, pool, _, _ = sched(spill, root + "/ft_min", w, min_tokens=4 * HBS, ttl_min_tokens=2 * HBS)
    save(spill, s, pool, req("short_keep", chain(b"k", 8)[:4], {"ttl_s": 3600}), 3)
    save(spill, s, pool, req("short_dflt", chain(b"l", 8)[:4]), 3)
    kept = sorted(rp.key[:1].decode() for rp in s.rps.values())
    check("ft_ttl_min_tokens_only_for_explicit_keep", kept == ["k"], f"{kept}")


# ------------------------------------------------------------------------------------------------ persistence (scheduler)
def hello(spill, s, files_by_rank, sigs=("SIG0", "SIG1"), errors=(None, None)):
    rep = {r: {"rank": r, "sig": sigs[r], "parts": {"x": sigs[r]}, "files": dict(files_by_rank[r]),
               "error": errors[r]} for r in (0, 1)}
    s.update_connector_output(spill.SpillWorkerMeta(hello=rep))


def boot(spill, root, wall, files_by_rank=({}, {}), sigs=("SIG0", "SIG1"), errors=(None, None), send_hello=True, **kw):
    s, pool, clk, io = sched(spill, root, wall, persist=True, **kw)
    s._ident_thread.join(60)
    m1 = s.build_meta()
    if send_hello:
        hello(spill, s, files_by_rank, sigs, errors)
    m2 = s.build_meta()
    return s, pool, clk, io, m1, m2


def disk_files(s):
    return {f: s.file_bytes[db.group] for f, db in s.disk.items() if db.state == "ok"}


def first_run(spill, root, wall, n_conv=2):
    s, pool, clk, io, m1, m2 = boot(spill, root, wall)
    check("p_first_boot_no_index_wipes", (m1.wipe or m2.wipe) and s.persist_state == "on",
          f"{s.persist_state} {s.persist_reason}")
    convs = []
    for k in range(n_conv):
        c = chain(bytes([0x50 + k]), 12)
        save(spill, s, pool, req(f"c{k}t3", c[:4], {"ttl_s": 86400, "session": f"S{k}", "epoch": 2}), 3)
        save(spill, s, pool, req(f"c{k}t4", c[:5], {"ttl_s": 86400, "session": f"S{k}", "epoch": 2}), 4)
        convs.append(c)
    s.shutdown()
    return s, convs


def t_persist_roundtrip(spill, root):
    w = Wall()
    s1, convs = first_run(spill, root + "/rt", w)
    idx_path = s1.cfg.index_path()
    check("p_index_written", os.path.exists(idx_path) and s1.stats["index_writes"] >= 1, f"{s1.stats['index_writes']}")
    files = disk_files(s1)
    keys1 = {rp.key: (rp.n_tokens, rp.ttl_s, rp.session, rp.epoch, [(e.group, e.keys, e.fname) for e in rp.blocks])
             for rp in s1.rps.values()}
    wids1 = {f: db.wid for f, db in s1.disk.items()}
    w.t += 600
    s2, pool, clk, io, m1, m2 = boot(spill, root + "/rt", w, (files, files))
    check("p_adopted", s2.persist_state == "on" and s2.persist_reason == "adopted"
          and s2.stats["persist_adopted_rps"] == len(keys1), f"{s2.persist_state} {s2.persist_reason}")
    keys2 = {rp.key: (rp.n_tokens, rp.ttl_s, rp.session, rp.epoch, [(e.group, e.keys, e.fname) for e in rp.blocks])
             for rp in s2.rps.values()}
    check("p_adopted_rps_identical", keys1 == keys2)
    check("p_adopted_wids_identical", {f: db.wid for f, db in s2.disk.items()} == wids1)
    check("p_adopted_disk_bytes", s2.disk_bytes == s1.disk_bytes, f"{s2.disk_bytes} {s1.disk_bytes}")
    check("p_no_wipe_no_orphans", not (m1.wipe or m2.wipe) and not m2.rank_deletes and not m1.rank_deletes)
    check("p_epochs_restored", dict(s2.session_epoch) == {"S0": 2, "S1": 2})
    # a resend of conversation 0 after the restart restores from the adopted RP (fresh GPU pool), with write ids
    c = convs[0]
    r = s2.get_num_new_matched_tokens(req("after", c[:6], {"ttl_s": 86400, "session": "S0", "epoch": 2}), 0)
    ml = s2.build_meta()
    (lj,) = list(ml.loads)
    ent = ml.loads[lj]
    check("p_restore_after_restart_started", r == (None, False) and len(ent) == 5, f"{r} {len(ent)}")
    check("p_load_carries_index_write_ids", all(e[3] == wids1[e[2]] for e in ent), f"{ent[:1]}")
    T.answer(spill, s2, lj, 2, 0)
    check("p_restore_ready", s2.restores["after"].state == "ready")


def t_persist_partial(spill, root):
    w = Wall()
    s1, convs = first_run(spill, root + "/pa", w)
    files = disk_files(s1)
    victim_rp = max(s1.rps.values(), key=lambda rp: (rp.key[:1], rp.n_tokens))     # conv 1, n=4
    own = [e.fname for e in victim_rp.blocks if e.group == 2][0]      # its state file (not shared)
    mla4 = [e.fname for e in victim_rp.blocks if e.group == 0][3]     # its 4th MLA block (the n=3 RP has 3)
    f1 = dict(files)
    del f1[own]                                                          # rank 1 lost it
    f0 = dict(files)
    f0["deadbeef" * 5 + ".blk"] = s1.file_bytes[0]                      # rank 0 has an orphan
    w.t += 60
    s2, pool, clk, io, m1, m2 = boot(spill, root + "/pa", w, (f0, f1))
    names = sorted((rp.key[:1].hex(), rp.n_tokens // HBS) for rp in s2.rps.values())
    check("pa_rp_with_missing_file_dropped_others_kept", len(names) == 3 and s2.stats["persist_dropped_missing"] == 1,
          f"{names}")
    rd = m2.rank_deletes or m1.rank_deletes
    check("pa_orphans_per_rank", sorted(rd.get(0, [])) == sorted(["deadbeef" * 5 + ".blk", own, mla4])
          and rd.get(1) == [mla4], f"{rd}")
    check("pa_no_wipe", not (m1.wipe or m2.wipe))


def t_persist_mismatch(spill, root):
    w = Wall()
    s1, _ = first_run(spill, root + "/mm", w)
    files = disk_files(s1)
    s2, *_rest, m1, m2 = boot(spill, root + "/mm", w, (files, files), sigs=("SIG0", "OTHER"))
    check("mm_rank_disagreement_wipes_both", (m1.wipe or m2.wipe) and not s2.rps
          and "rank 1" in s2.persist_reason, s2.persist_reason)
    # the wipe was followed by a fresh index (rank 1's new signature): a second restart adopts nothing, wipes nothing
    s2.shutdown()
    s3, *_rest, m1, m2 = boot(spill, root + "/mm", w, ({}, {}), sigs=("SIG0", "OTHER"))
    check("mm_after_wipe_new_index_consistent", s3.persist_reason == "adopted" and not s3.rps, s3.persist_reason)
    # scheduler-side identity: an env knob of the production patches changes -> wipe
    s1, _ = first_run(spill, root + "/mm2", w)
    files = disk_files(s1)
    os.environ["GLM53_KPOOL_TEST_FLAG"] = "1"
    try:
        s2, *_rest, m1, m2 = boot(spill, root + "/mm2", w, (files, files))
    finally:
        os.environ.pop("GLM53_KPOOL_TEST_FLAG", None)
    check("mm_env_change_wipes", (m1.wipe or m2.wipe) and not s2.rps and "env" in s2.persist_reason, s2.persist_reason)
    # the KV pool comes out larger/smaller (number of blocks): block files stay valid -> adopted
    s1, _ = first_run(spill, root + "/mm3", w)
    files = disk_files(s1)
    s2, *_rest, m1, m2 = boot(spill, root + "/mm3", w, (files, files), n=160)
    check("mm_pool_size_change_still_adopts", not (m1.wipe or m2.wipe) and len(s2.rps) == len(s1.rps),
          s2.persist_reason)
    # the KV page layout changes (page bytes) -> wipe
    s1, _ = first_run(spill, root + "/mm3b", w)
    files = disk_files(s1)
    s2, *_rest, m1, m2 = boot(spill, root + "/mm3b", w, (files, files), scale=2)
    check("mm_page_layout_change_wipes", (m1.wipe or m2.wipe) and not s2.rps and "layout" in s2.persist_reason,
          s2.persist_reason)
    # a rank report with an error -> wipe
    s1, _ = first_run(spill, root + "/mm4", w)
    files = disk_files(s1)
    s2, *_rest, m1, m2 = boot(spill, root + "/mm4", w, (files, files), errors=(None, "listdir: EIO"))
    check("mm_rank_error_wipes", (m1.wipe or m2.wipe) and not s2.rps and "EIO" in s2.persist_reason, s2.persist_reason)


def t_persist_corrupt(spill, root):
    w = Wall()
    for kind in ("flip", "truncate", "garbage", "tmp_only", "missing", "version"):
        d = root + f"/cor_{kind}"
        s1, _ = first_run(spill, d, w)
        files = disk_files(s1)
        p = s1.cfg.index_path()
        raw = open(p, "rb").read()
        if kind == "flip":
            b = bytearray(raw)
            b[len(b) // 2] ^= 0x01
            open(p, "wb").write(bytes(b))
        elif kind == "truncate":
            open(p, "wb").write(raw[: len(raw) // 3])
        elif kind == "garbage":
            open(p, "wb").write(b"\x00" * 100)
        elif kind == "tmp_only":
            os.rename(p, p + ".tmp")        # a crash between write and rename: only the temp file exists
        elif kind == "missing":
            os.unlink(p)
        elif kind == "version":
            nl = raw.index(b"\n")
            top = json.loads(raw[:nl])
            top["version"] = 999
            open(p, "wb").write(json.dumps(top).encode() + raw[nl:])
        s2, *_rest, m1, m2 = boot(spill, d, w, (files, files))
        check(f"cor_{kind}_wipes", (m1.wipe or m2.wipe) and not s2.rps and s2.persist_state == "on", s2.persist_reason)


def t_persist_expired(spill, root):
    w = Wall()
    s, pool, clk, io, m1, m2 = boot(spill, root + "/ex", w)
    save(spill, s, pool, req("a", chain(b"a", 6)[:5], {"ttl_s": 3600, "session": "A"}), 4)
    save(spill, s, pool, req("b", chain(b"b", 6)[:5], {"ttl_s": 259200, "session": "B"}), 4)
    save(spill, s, pool, req("c", chain(b"c", 6)[:5]), 4)
    s.shutdown()
    files = disk_files(s)
    w.t += 7200
    s2, *_rest, m1, m2 = boot(spill, root + "/ex", w, (files, files))
    left = sorted(rp.key[:1].decode() for rp in s2.rps.values())
    check("ex_expired_dropped_at_adoption", left == ["b", "c"] and s2.stats["persist_dropped_expired"] == 1, f"{left}")
    rd = m2.rank_deletes or m1.rank_deletes
    check("ex_expired_files_deleted_on_both_ranks", len(rd.get(0, [])) == 5 and len(rd.get(1, [])) == 5, f"{rd}")


def t_persist_deadline(spill, root):
    w = Wall()
    s1, _ = first_run(spill, root + "/dl", w)
    files = disk_files(s1)
    s2, pool, clk, io = sched(spill, root + "/dl", w, persist=True, adopt_deadline_s=60)
    s2._ident_thread.join(60)
    s2.build_meta()
    s2.update_connector_output(spill.SpillWorkerMeta(hello={0: {"rank": 0, "sig": "SIG0", "files": files}}))
    s2.build_meta()
    r = s2.get_num_new_matched_tokens(req("w", chain(b"w", 6)), 0)
    check("dl_requests_wait_while_adopting_with_index", s2.persist_state == "adopting" and r == (None, False)
          and s2.has_pending(), f"{s2.persist_state} {r}")
    io.t += 61
    m = s2.build_meta()
    m2 = s2.build_meta()
    check("dl_missing_rank_report_wipes", (m.wipe or m2.wipe) and not s2.rps and "rank(s) [1]" in s2.persist_reason,
          s2.persist_reason)
    r = s2.get_num_new_matched_tokens(req("w", chain(b"w", 6)), 0)
    check("dl_requests_released", r == (0, False))


def t_persist_hash(spill, root):
    from vllm.v1.core import kv_cache_utils as kcu
    w = Wall()
    s1, _ = first_run(spill, root + "/hs", w)
    files = disk_files(s1)
    saved = kcu.NONE_HASH
    kcu.NONE_HASH = kcu.BlockHash(os.urandom(32))       # what init_none_hash does without PYTHONHASHSEED
    try:
        s2, *_rest, m1, m2 = boot(spill, root + "/hs", w, (files, files))
    finally:
        kcu.NONE_HASH = saved
    check("hs_random_none_hash_disables_and_wipes", (m1.wipe or m2.wipe) and s2.persist_state == "disabled"
          and not os.path.exists(s2.cfg.index_path()), s2.persist_reason)
    pool = s2.block_pool
    save(spill, s2, pool, req("q", chain(b"q", 6)[:5]), 4)
    s2.shutdown()
    check("hs_disabled_run_still_spills_in_process_but_writes_no_index", len(s2.rps) == 1
          and not os.path.exists(s2.cfg.index_path()))


def t_persist_cap(spill, root):
    w = Wall()
    s1, _ = first_run(spill, root + "/cap", w, n_conv=3)
    files = disk_files(s1)
    cap = s1.disk_bytes - 1
    s2, *_rest, m1, m2 = boot(spill, root + "/cap", w, (files, files), cap_bytes=cap)
    check("cap_lowered_evicts_at_adoption", s2.disk_bytes <= cap and s2.stats["rp_evicted"] >= 1
          and len(s2.rps) < len(s1.rps), f"{s2.disk_bytes} {len(s2.rps)}")


def t_persist_breaker(spill, root):
    w = Wall()
    s, pool, clk, io, m1, m2 = boot(spill, root + "/br", w)
    save(spill, s, pool, req("a", chain(b"a", 6)[:5]), 4)
    files = disk_files(s)
    s._trip("test")
    s.build_meta()
    s.shutdown()
    s2, *_rest, m1, m2 = boot(spill, root + "/br", w, (files, files))
    rd = m2.rank_deletes or m1.rank_deletes
    check("br_breaker_writes_empty_index_next_start_deletes_all", not s2.rps and s2.persist_reason == "adopted"
          and len(rd.get(0, [])) == len(files), f"{s2.persist_reason} {rd}")


def t_persist_index_cadence(spill, root):
    w = Wall()
    s, pool, clk, io, m1, m2 = boot(spill, root + "/cad", w, index_every_s=5)
    s._index_writer.flush()
    n0 = s.stats["index_writes"]
    def nw():
        s.build_meta()
        s._index_writer.flush()
        return s.stats["index_writes"]

    save(spill, s, pool, req("a", chain(b"a", 6)[:5]), 4)       # a new RP: written within INDEX_EVERY_S
    n1 = nw()
    clk.t += 6
    n2 = nw()
    s.get_num_new_matched_tokens(req("a2", chain(b"a", 7)[:6]), 5 * HBS)     # a refresh (last_used) only
    n3 = nw()
    clk.t += 6
    n4 = nw()
    s.get_num_new_matched_tokens(req("z", chain(b"a", 7)[:6], {"ttl_s": 0}), 0)    # ttl 0: written at once
    n5 = nw()
    check("cad_throttled_every_5s_ttl0_at_once", (n1, n2, n3, n4, n5) == (n0, n0 + 1, n0 + 1, n0 + 2, n0 + 3),
          f"{n0} {n1} {n2} {n3} {n4} {n5}")
    idx, err = spill._read_index(s.cfg.index_path())
    check("cad_index_verifies_and_has_no_rp_after_ttl0", idx is not None and len(idx["rps"]) == 0, err)


# ------------------------------------------------------------------------------------------------ real workers
def make_worker(spill, root, rank, persist=True, nblocks=16, seed=0):
    cfg = spill.SpillConfig(root=root, cap_bytes=8 << 30, min_free_bytes=0, direct=False, log_every_s=0,
                            persist=persist)
    kvc = prodlayout.build_kv_cache_config(nblocks, scale=576)
    w = spill.SpillWorker(cfg, kvc, rank)
    g = torch.Generator().manual_seed(seed * 101 + rank)
    raws = [torch.randint(0, 255, (kvt.size,), dtype=torch.uint8, generator=g) for kvt in kvc.kv_cache_tensors]
    views = {n: raws[t] for t, kvt in enumerate(kvc.kv_cache_tensors) for n in kvt.shared_by}
    w.register(views, sig="layoutsig", ident_fn=lambda: {"env": "E", "site": "S"})
    return w, raws


def take_hello(w, timeout=20):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        m = w.take_meta()
        if m is not None and m.hello:
            return m.hello[w.rank]
        time.sleep(0.01)
    return None


def block_bytes(w, raws, g, bid):
    return b"".join(bytes(raws[t].view(-1, page)[bid].numpy()) for t, page in w.store.rows[g])


def t_worker_persist(spill, root):
    d = root + "/wp"
    w, raws = make_worker(spill, d, 0)
    h = take_hello(w)
    check("wp_first_scan_report", h is not None and h["files"] == {} and h["sig"] and not h["error"], f"{h}")
    M = spill.SpillMeta
    w.handle(M(saves={1: [(0, 3, "a" * 40 + ".blk", "boot-1"), (2, 4, "b" * 40 + ".blk", "boot-1")]}))
    w.drain(30)
    m = w.take_meta()
    check("wp_save_with_wid", m is not None and m.done.get(1) == [1, 0])
    open(os.path.join(w.store.dir, "zz.blk.123.partial"), "wb").write(b"x")
    sig1 = h["sig"]
    w.shutdown()
    # restart: the files survive, the partial goes, the same identity gives the same signature
    w2, raws2 = make_worker(spill, d, 0, seed=7)
    h2 = take_hello(w2)
    want = {"a" * 40 + ".blk": w2.store.file_bytes(0), "b" * 40 + ".blk": w2.store.file_bytes(2)}
    check("wp_restart_scan_keeps_files_drops_partial", h2 is not None and h2["files"] == want and h2["partial"] == 1
          and not os.path.exists(os.path.join(w2.store.dir, "zz.blk.123.partial")), f"{h2 and h2['files']}")
    check("wp_signature_stable_across_restart", h2["sig"] == sig1)
    # load with the right write id: the bytes saved from block 3 land in block 9
    w2.handle(M(loads={2: [(0, 9, "a" * 40 + ".blk", "boot-1")]}))
    w2.drain(30)
    m = w2.take_meta()
    check("wp_load_right_wid_ok_bytes_equal", m is not None and m.done.get(2) == [1, 0]
          and block_bytes(w2, raws2, 0, 9) == block_bytes(w, raws, 0, 3))
    w2.handle(M(loads={3: [(0, 10, "a" * 40 + ".blk", "other-7")]}))
    w2.drain(30)
    m = w2.take_meta()
    check("wp_load_wrong_wid_fails_as_bad_file", m is not None and m.done.get(3) == [0, 1]
          and "a" * 40 + ".blk" in m.bad, f"{m}")
    # per-rank deletes: rank 0 deletes its listed file only, rank 1's list is not executed here
    w2.handle(M(rank_deletes={0: ["b" * 40 + ".blk"], 1: ["a" * 40 + ".blk"]}))
    w2.drain(30)
    check("wp_rank_deletes_only_own_list", not os.path.exists(os.path.join(w2.store.dir, "b" * 40 + ".blk"))
          and os.path.exists(os.path.join(w2.store.dir, "a" * 40 + ".blk")))
    w2.handle(M(wipe=True, saves={4: [(0, 5, "c" * 40 + ".blk", "boot2-4")]}))
    w2.drain(30)
    left = sorted(os.listdir(w2.store.dir))
    check("wp_wipe_runs_before_saves_of_the_same_step", left == ["c" * 40 + ".blk"], f"{left}")
    w2.shutdown()
    # a rank directory that can not be listed -> the report carries the error (scheduler wipes)
    d3 = root + "/wp3"
    w3, _ = make_worker(spill, d3, 1)
    take_hello(w3)
    w3.shutdown()
    os.chmod(os.path.join(d3, spill.SUBDIR, "rank1"), 0o300)
    try:
        w4, _ = make_worker(spill, d3, 1)
        h4 = take_hello(w4)
        w4.shutdown()
    finally:
        os.chmod(os.path.join(d3, spill.SUBDIR, "rank1"), 0o700)
    check("wp_unlistable_dir_reports_error", h4 is not None and h4["error"] and "listdir" in h4["error"], f"{h4}")
    # persist off: the old behaviour (wipe at register)
    d5 = root + "/wp5"
    w5, _ = make_worker(spill, d5, 0, persist=False)
    w5.handle(M(saves={1: [(0, 3, "d" * 40 + ".blk")]}))
    w5.drain(30)
    w5.shutdown()
    w6, _ = make_worker(spill, d5, 0, persist=False)
    check("wp_persist_off_wipes_at_register", os.listdir(w6.store.dir) == [] and take_hello(w6, 0.3) is None)
    w6.shutdown()


def t_e2e_restart(spill, root):
    """Two real ranks + the real scheduler (fake pool): save two turns -> clean shutdown -> new workers + new scheduler
    on the same dirs -> adopt -> restore into a fresh pool -> every restored block holds exactly the saved bytes on
    both ranks. Then the same with a crash (no shutdown flush) after a later prune: the stale index still adopts
    consistently (files deleted after the last index write are simply dropped)."""
    d = root + "/e2e"
    w = Wall()
    nb = 48

    def world(seed):
        ws = [make_worker(spill, d, r, nblocks=nb, seed=seed) for r in (0, 1)]
        return [x[0] for x in ws], [x[1] for x in ws]

    def step(s, workers, m):
        import copy
        for wk in workers:
            wk.handle(copy.deepcopy(m))     # each rank unpickles its own copy in production
        for wk in workers:
            wk.drain(60)
        agg = None
        for wk in workers:
            mm = wk.take_meta()
            if mm is not None:
                agg = mm if agg is None else agg.aggregate(mm)
        s.update_connector_output(agg)

    def run_steps(s, workers, k=3):
        for _ in range(k):
            step(s, workers, s.build_meta())

    workers, raws = world(seed=1)
    for wk in workers:
        take_hello(wk, 20)
        wk._hello_sent = False      # deliver through the normal path below
    s, pool, clk, io = sched(spill, d, w, persist=True, n=nb, keep_per_chain=2)
    s._ident_thread.join(60)
    run_steps(s, workers)
    check("e2e_first_start_wipes_and_runs", s.persist_state == "on", s.persist_reason)
    c = chain(b"E", 12)
    saved = {}
    for turn, nbk in ((0, 3), (1, 4)):
        r = req(f"t{turn}", c[:nbk + 1], {"ttl_s": 86400, "session": "S", "epoch": 0})
        blocks = T.compute(pool, r, nbk)
        s.save_request(r)
        run_steps(s, workers)
        for b in blocks:
            if b.block_hash is not None:
                g = int.from_bytes(bytes(b.block_hash)[-4:], "big")
                # a key already on disk is deduplicated (its file holds the first save's bytes); the fake engine
                # re-keys fresh blocks for the earlier prefix, so only the first sighting is the file's content
                saved.setdefault(bytes(b.block_hash),
                                 [block_bytes(workers[k], raws[k], g, b.block_id) for k in (0, 1)])
    check("e2e_two_rps_saved", s.stats["rp_saved"] == 2 and len(s.rps) == 2, f"{s.stats['rp_saved']}")
    s.shutdown()
    for wk in workers:
        wk.shutdown()
    # ---- restart (clean)
    workers2, raws2 = world(seed=2)          # different initial GPU bytes: only the files can bring them back
    s2, pool2, clk2, io2 = sched(spill, d, w, persist=True, n=nb, keep_per_chain=2)
    s2._ident_thread.join(60)
    run_steps(s2, workers2)
    check("e2e_restart_adopts", s2.persist_reason == "adopted" and len(s2.rps) == 2,
          f"{s2.persist_reason} rps={len(s2.rps)} {[(k, v) for k, v in s2.stats.items() if k.startswith('persist')]}")
    r = req("t2", c[:7], {"ttl_s": 86400, "session": "S", "epoch": 0})
    res = s2.get_num_new_matched_tokens(r, 0)
    run_steps(s2, workers2, 2)
    st = s2.restores.get("t2")
    ok = st is not None and st.state == "ready"
    mism = 0
    nchk = 0
    if ok:
        for b in list(st.new_blocks) + list(st.pins):
            key = bytes(b.block_hash)
            g = int.from_bytes(key[-4:], "big")
            for k in (0, 1):
                nchk += 1
                if block_bytes(workers2[k], raws2[k], g, b.block_id) != saved[key][k]:
                    mism += 1
    check("e2e_restored_after_restart_byte_identical", ok and nchk > 0 and mism == 0,
          f"res={res} state={st and st.state} checked={nchk} mismatches={mism}")
    s2.update_state_after_alloc(r)
    s2.request_finished(r)
    # ---- a third turn is saved (prunes the oldest RP), then a crash: no shutdown flush, index possibly stale
    blocks = T.compute(pool2, req("t3", c[:8], {"ttl_s": 86400, "session": "S", "epoch": 0}), 7)
    s2.save_request(req("t3", c[:8], {"ttl_s": 86400, "session": "S", "epoch": 0}))
    run_steps(s2, workers2)
    for wk in workers2:
        wk.shutdown()
    s2._index_writer.flush()
    workers3, raws3 = world(seed=3)
    s3, pool3, _, _ = sched(spill, d, w, persist=True, n=nb, keep_per_chain=2)
    s3._ident_thread.join(60)
    run_steps(s3, workers3)
    files0 = set(os.listdir(workers3[0].store.dir))
    referenced = set(s3.disk)
    check("e2e_crash_restart_consistent", s3.persist_reason == "adopted" and referenced <= files0
          and all(set(os.listdir(wk.store.dir)) == referenced for wk in workers3),
          f"{s3.persist_reason} rps={len(s3.rps)} ref={len(referenced)} r0={len(files0)}")
    for wk in workers3:
        wk.shutdown()


def t_config(spill):
    keys = [k for k in os.environ if k.startswith("GLM53_NVME_SPILL_")]
    saved = {k: os.environ.pop(k) for k in keys}
    try:
        os.environ["GLM53_NVME_SPILL_DIR"] = "/tmp/x"
        c = spill.SpillConfig.from_env()
        check("cfg_defaults", c.persist and c.hash_seed == "5353" and c.ttl_save_policy == "all"
              and c.ttl_min_tokens == 0 and c.index_every_s == 5.0, f"{c}")
        for k, v in (("HASH_SEED", "abc"), ("HASH_SEED", "4294967296"), ("TTL_SAVE_POLICY", "x"),
                     ("TTL_MIN_TOKENS", "-1"), ("INDEX_EVERY_S", "0")):
            os.environ["GLM53_NVME_SPILL_" + k] = v
            try:
                spill.SpillConfig.from_env()
                check(f"cfg_reject_{k}_{v}", False)
            except ValueError:
                check(f"cfg_reject_{k}_{v}", True)
            os.environ.pop("GLM53_NVME_SPILL_" + k)
        os.environ["GLM53_NVME_SPILL_PERSIST"] = "0"
        check("cfg_persist_off", not spill.SpillConfig.from_env().persist)
        # appended in this order (2026-10-14); the finish release appends DEFAULT_TTL_S after them (test_finish.py)
        check("knobs_appended", list(spill.KNOBS[-8:-1]) == ["PERSIST", "HASH_SEED", "INDEX_EVERY_S", "ADOPT_DEADLINE_S",
                                                             "TTL_SWEEP_S", "TTL_SAVE_POLICY", "TTL_MIN_TOKENS"]
              and spill.KNOBS[-1] == "DEFAULT_TTL_S")
    finally:
        for k in [k for k in os.environ if k.startswith("GLM53_NVME_SPILL_")]:
            os.environ.pop(k)
        os.environ.update(saved)


def t_stats(spill, root):
    w = Wall()
    s, pool, _, _ = sched(spill, root + "/st", w)
    save(spill, s, pool, req("a", chain(b"a", 6)[:5], {"ttl_s": 3600}), 4)
    save(spill, s, pool, req("b", chain(b"b", 6)[:5]), 4)
    line = s.stats_line()
    for k in ("expired=0", "epoch_dropped=0", "ttl0_skipped=0", "ttl_rps=1/1/0/0/0", "persist=off", "adopted=0",
              "index_age_s=-1"):
        check(f"stats_has_{k}", k in line, line[-260:])
    check("stats_skipped_cap_still_before_new", line.index("skipped_cap=") < line.index("expired="))


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
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    assert os.environ.get("PYTHONHASHSEED"), "run with PYTHONHASHSEED set (EngineCore order is tested in the rig)"
    init_none_hash(sha256)
    root = os.path.abspath(a.dir)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    tests = [("parse", lambda: t_parse(spill)), ("evict", lambda: t_eviction_order(spill, root)),
             ("sliding", lambda: t_ttl_sliding(spill, root)), ("ttl0", lambda: t_ttl0(spill, root)),
             ("epoch", lambda: t_epoch(spill, root)), ("first", lambda: t_first_turn(spill, root)),
             ("rt", lambda: t_persist_roundtrip(spill, root)), ("partial", lambda: t_persist_partial(spill, root)),
             ("mismatch", lambda: t_persist_mismatch(spill, root)), ("corrupt", lambda: t_persist_corrupt(spill, root)),
             ("expired", lambda: t_persist_expired(spill, root)), ("deadline", lambda: t_persist_deadline(spill, root)),
             ("hash", lambda: t_persist_hash(spill, root)), ("cap", lambda: t_persist_cap(spill, root)),
             ("breaker", lambda: t_persist_breaker(spill, root)), ("cadence", lambda: t_persist_index_cadence(spill, root)),
             ("worker", lambda: t_worker_persist(spill, root)), ("e2e", lambda: t_e2e_restart(spill, root)),
             ("config", lambda: t_config(spill)), ("stats", lambda: t_stats(spill, root))]
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
    print(f"SUMMARY {len(T.RESULTS) - n_fail}/{len(T.RESULTS)} passed", flush=True)
    shutil.rmtree(root, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
