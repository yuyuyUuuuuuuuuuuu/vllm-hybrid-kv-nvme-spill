# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests of the 2026-10-14 finish changes of the spill connector:
  * B8 policy gate (SpillScheduler._policy_skips / get_num_new_matched_tokens): a restore is not started in a step in
    which the scheduler's decode-floor policy skips the request anyway; checked against the REAL policy class
    (vllm/v1/core/sched/scheduler.py _Glm53MixedPrefill, the production code in the rig work tree) on random states:
    "gate says skip" must imply "the policy returns cap 0" (never hold back a request the scheduler would admit), and in
    skip mode the two must agree exactly.
  * identity "code" part (_code_digest): comment / docstring / formatting edits of the connector do not change it, any
    executable token does; _site_digest leaves the connector's own file to it and stays byte-strict for every other file.
Needs vLLM importable (production image, PYTHONPATH = a rig work tree).

usage: python3 test_finish.py --dir <scratch> [--module <connector .py>]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import shutil
import sys
from types import ModuleType, SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sched_faults as tsf  # noqa: E402  (fake pool / coordinator / requests)
import test_ttl_persist as tp  # noqa: E402  (sched / save / req / chain / Wall / boot / disk_files)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


# ------------------------------------------------------------------------------------------------ code digest
def t_code_digest(spill, root):
    src = open(spill.__file__).read()
    d = os.path.join(root, "code")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)

    def dig(text, name):
        p = os.path.join(d, name + ".py")
        with open(p, "w") as f:
            f.write(text)
        return spill._code_digest(p)

    base = dig(src, "base")
    check("code_digest_of_module_file_equals_copy", base == spill._code_digest(), base)
    same = {
        "comment_added": src.replace("\nclass SpillScheduler:", "\n# a new comment line\nclass SpillScheduler:  # trailing", 1),
        "module_docstring_changed": src.replace("GLM-5.3-Flash NVMe spill tier", "GLM-5.3-Flash NVMe spill tier (edited)", 1),
        "function_docstring_changed": src.replace("True when ``other`` is a strict prefix", "True iff ``other`` is a strict prefix", 1),
        "blank_lines_added": src.replace("\ndef _up(", "\n\n\n\ndef _up(", 1),
        "line_break_reformatted": src.replace("return -(-x // a) * a", "return (\n        -(-x // a) * a\n    )", 1),
    }
    for name, text in same.items():
        check(f"code_digest_unchanged_{name}", text != src and dig(text, name) == base)
    diff = {
        "log_format_string": src.replace('"%s rank %d: dir=%s', '"%s rank %d:  dir=%s', 1),
        "constant": src.replace("HDR = 4096", "HDR = 8192", 1),
        "name": src.replace("def _up(x: int, a: int = ALIGN) -> int:\n    return -(-x // a) * a",
                            "def _up(y: int, a: int = ALIGN) -> int:\n    return -(-y // a) * a", 1),
        "operator": src.replace("return -(-x // a) * a", "return -(-x // a) * a + 0", 1),
        "default": src.replace("keep_per_chain: int = 2", "keep_per_chain: int = 3", 1),
        "key_derivation": src.replace('hashlib.sha256(bytes(primary_key)).hexdigest()', 'hashlib.sha1(bytes(primary_key)).hexdigest()', 1),
        "string_in_code_not_docstring": src.replace('MAGIC = b"GLM53SPL"', 'MAGIC = b"GLM53SPX"', 1),
    }
    for name, text in diff.items():
        if text == src:
            check(f"code_digest_changes_{name}", False, "test edit did not apply (anchor drift)")
            continue
        check(f"code_digest_changes_{name}", dig(text, name) != base)


def t_site_digest(spill, root):
    """_site_digest on a fake site-packages: the connector's own file counts by name only (its code is the "code" part),
    every other file by content."""
    import importlib.util as iu

    site = os.path.join(root, "site")
    shutil.rmtree(site, ignore_errors=True)
    v1 = os.path.join(site, "vllm", "distributed", "kv_transfer", "kv_connector", "v1")
    os.makedirs(v1)
    with open(os.path.join(site, "vllm", "__init__.py"), "w") as f:
        f.write("x = 1\n")
    with open(os.path.join(site, "vllm", "scheduler.py"), "w") as f:
        f.write("y = 2\n")
    me = os.path.join(v1, "glm53_nvme_spill_connector.py")
    shutil.copy(spill.__file__, me)
    real_find, real_file = iu.find_spec, spill.__file__
    iu.find_spec = lambda name, *a, **k: SimpleNamespace(origin=os.path.join(site, "vllm", "__init__.py")) \
        if name == "vllm" else real_find(name, *a, **k)
    spill.__file__ = me
    try:
        a = spill._site_digest()
        with open(me, "a") as f:
            f.write("\n# a comment appended to the connector\n")
        b = spill._site_digest()
        with open(me, "a") as f:
            f.write("ZZ_NEW = 1\n")          # a code change of the connector: site unchanged, code part changes
        c = spill._site_digest()
        code_c = spill._code_digest()
        with open(os.path.join(site, "vllm", "scheduler.py"), "a") as f:
            f.write("# comment in ANOTHER file\n")
        d = spill._site_digest()
    finally:
        iu.find_spec, spill.__file__ = real_find, real_file
    check("site_digest_ignores_connector_bytes", a == b == c, f"{a} {b} {c}")
    check("site_digest_counts_connector_by_name", a.endswith("/3"), a)
    check("code_part_sees_connector_code_change", code_c != spill._code_digest(real_file))
    check("site_digest_strict_for_other_files_even_comments", d != a, f"{a} -> {d}")


def t_identity_parts(spill, root):
    parts = spill._identity(SimpleNamespace(), SimpleNamespace(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[]))
    check("identity_has_code_part", "code" in parts and len(parts["code"]) == 16 and not parts["code"].startswith("err"),
          str(parts.get("code")))
    check("identity_version_is_finish", parts["version"].startswith("2026-10-14.finish."), parts["version"])


# ------------------------------------------------------------------------------------------------ policy gate
def _req(rid, prompt, computed, decoding=False):
    n = prompt + (5 if decoding else 0)
    return SimpleNamespace(request_id=rid, num_prompt_tokens=prompt, num_tokens=n,
                           num_computed_tokens=(n if decoding else computed))


def t_policy_equivalence(spill, root):
    """Random states of the REAL decode-floor policy: gate True => cap_for == 0; skip mode: gate == (cap_for == 0)."""
    smod = sys.modules.get("vllm.v1.core.sched.scheduler")
    if smod is None:
        import vllm.v1.core.sched.scheduler as smod  # noqa: F811
    cls = smod._Glm53MixedPrefill
    s, pool, clk, io = tsf.new_sched(spill, os.path.join(root, "peq"))
    rng = random.Random(7)
    saved = os.environ.get("GLM53_MIXED_PREFILL_CHUNK")
    orig = smod._GLM53_MIXED
    stats = {}
    try:
        for mode in ("skip", "fair", "0", "1024"):
            os.environ["GLM53_MIXED_PREFILL_CHUNK"] = mode
            unsound = mismatch = n = gated = 0
            for trial in range(400):
                t = [100.0 + trial]
                pol = cls(now=lambda: t[0])
                smod._GLM53_MIXED = pol
                if mode == "fair" and rng.random() < 0.5:
                    pol.mixed_samples = [((1, 1, 1), 256, 0.2), ((1, 1, 1), 1024, 0.6)]
                    pol.credit = rng.uniform(-1, 2)
                running = [_req(f"d{i}", 40000, 0, decoding=True) for i in range(rng.randint(0, 3))]
                running += [_req(f"p{i}", 90000, rng.randrange(0, 80000, 4608)) for i in range(rng.randint(0, 2))]
                waiting = [_req(f"w{i}", rng.randrange(50000, 400000), 0) for i in range(rng.randint(1, 4))]
                sched = SimpleNamespace(running=running, waiting=waiting, skipped_waiting=[], current_step=trial,
                                        requests={r.request_id: r for r in running + waiting})
                if mode == "fair" and rng.random() < 0.3:
                    pol.inflight[1] = {"prefill_tokens": {"x": 5}, "had_decode": True}
                pol.begin_step(sched)
                for w in waiting:
                    comp = rng.randrange(0, w.num_prompt_tokens - 1, 4608)
                    g = s._policy_skips(w, comp)
                    cap = pol.cap_for(sched, w, comp)
                    n += 1
                    gated += g
                    if g and cap != 0:
                        unsound += 1
                    if mode == "skip" and g != (cap == 0):
                        mismatch += 1
            stats[mode] = (n, gated)
            check(f"policy_gate_sound_mode_{mode}", unsound == 0, f"n={n} gated={gated} unsound={unsound}")
            if mode == "skip":
                check("policy_gate_exact_in_skip_mode", mismatch == 0 and gated > 0, f"mismatch={mismatch}")
            if mode in ("0", "1024"):
                check(f"policy_gate_never_in_mode_{mode}", gated == 0, f"gated={gated}")
        check("policy_gate_fires_in_fair_mode", stats["fair"][1] > 0, str(stats["fair"]))
    finally:
        smod._GLM53_MIXED = orig
        if saved is None:
            os.environ.pop("GLM53_MIXED_PREFILL_CHUNK", None)
        else:
            os.environ["GLM53_MIXED_PREFILL_CHUNK"] = saved


class _FakePolicy:
    def __init__(self, mode, had_decode):
        self.mode = mode
        self._open_rec = {"had_decode": had_decode, "grants": {}}
        self.step_mode = "solo"
        self.selected = set()

    @staticmethod
    def prefill_remaining(request, computed=None):
        return max(0, request.num_prompt_tokens - int(computed if computed is not None else 0))


def t_restore_gated(spill, root):
    """get_num_new_matched_tokens: policy skips -> (None, False), no restore, no block taken; policy lets it through ->
    the restore starts; no policy module / gate off -> restores as before."""
    key = "vllm.v1.core.sched.scheduler"
    real = sys.modules.get(key)
    fake = ModuleType(key)
    try:
        for case in ("skip_blocked", "skip_free", "no_module", "gate_off", "fair_not_turn"):
            kw = {"policy_gate": False} if case == "gate_off" else {}
            s, pool, clk, io = tsf.new_sched(spill, os.path.join(root, "rg-" + case), **kw)
            req = tsf.make_req("a", 6)
            tsf.saved_rp(spill, s, pool, req, 6)
            pool.evict_all_cached()
            nxt = tsf.make_req("b", 7)          # next turn of the same conversation
            nxt.num_computed_tokens = 0
            nxt.num_prompt_tokens = nxt.num_tokens
            if case == "no_module":
                sys.modules.pop(key, None)
            else:
                fake._GLM53_MIXED = _FakePolicy("fair" if case == "fair_not_turn" else "skip", case != "skip_free")
                if case == "fair_not_turn":
                    fake._GLM53_MIXED.step_mode = "decode_only"
                sys.modules[key] = fake
            free0 = pool.get_num_free_blocks()
            out = s.get_num_new_matched_tokens(nxt, 0)
            started = s.stats["restore_started"]
            blocked = case in ("skip_blocked", "fair_not_turn")
            if blocked:
                ok = out == (None, False) and started == 0 and pool.get_num_free_blocks() == free0 \
                    and s.stats["restore_deferred_policy"] == 1 and not s.restores
                check(f"restore_gate_{case}_defers_without_taking_blocks", ok,
                      f"out={out} started={started} free {free0}->{pool.get_num_free_blocks()}")
                # the next step the policy lets it through -> the restore starts
                fake._GLM53_MIXED = _FakePolicy("skip", False)
                out2 = s.get_num_new_matched_tokens(nxt, 0)
                check(f"restore_gate_{case}_starts_when_policy_allows", out2 == (None, False)
                      and s.stats["restore_started"] == 1 and "b" in s.restores)
            else:
                check(f"restore_gate_{case}_restores", out == (None, False) and started == 1
                      and s.stats["restore_deferred_policy"] == 0, f"out={out} started={started}")
            sys.modules.pop(key, None)
            if real is not None:
                sys.modules[key] = real
    finally:
        sys.modules.pop(key, None)
        if real is not None:
            sys.modules[key] = real


def t_stats_and_config(spill, root):
    s, pool, clk, io = tsf.new_sched(spill, os.path.join(root, "st"))
    line = s.stats_line()
    check("stats_line_ends_with_b8_counters", line.rstrip().split(" ")[-3:] == [
        "restores_started=0", "restores_admitted=0", "restores_deferred_policy=0"], line[-120:])
    check("policy_gate_not_an_operator_knob", "POLICY_GATE" not in "".join(spill.KNOBS)
          and spill.KNOBS[-2:] == ("TTL_MIN_TOKENS", "DEFAULT_TTL_S"), str(spill.KNOBS[-3:]))
    os.environ["GLM53_NVME_SPILL_DIR"] = os.path.join(root, "cfg")
    try:
        check("from_env_policy_gate_default_on", spill.SpillConfig.from_env().policy_gate is True)
        check("from_env_policy_gate_test_off", spill.SpillConfig.from_env({"no_policy_gate": 1}).policy_gate is False)
    finally:
        os.environ.pop("GLM53_NVME_SPILL_DIR", None)


# ------------------------------------------------------------------------------------------------ DEFAULT_TTL_S
def t_default_ttl(spill, root):
    """Resume points of requests without the DSH fields: sliding expiry after DEFAULT_TTL_S unused (0 = never), save
    policy unchanged (no first-turn writes), eviction order at the cap unchanged (first after expired), class label."""
    HBS = tsf.HBS
    W = tp.Wall
    # off (0): a no-field RP never expires
    w = W()
    s, pool, _, _ = tp.sched(spill, root + "/dt0", w, ttl_sweep_s=30)
    tp.save(spill, s, pool, tp.req("c", tp.chain(b"c", 6)[:5]), 4)
    w.t += 7 * 86400
    s.build_meta()
    check("dt_off_no_field_rp_never_expires", len(s.rps) == 1 and "ttl_classes=none/" in s.stats_line())
    # on (100 s): sliding, restore refused once expired, swept; an explicit-TTL RP keeps its own TTL
    w = W()
    s, pool, _, _ = tp.sched(spill, root + "/dt1", w, ttl_sweep_s=1, default_ttl_s=100)
    c = tp.chain(b"d", 12)
    tp.save(spill, s, pool, tp.req("d4", c[:5]), 4)
    tp.save(spill, s, pool, tp.req("e4", tp.chain(b"e", 6)[:5], {"ttl_s": 3600, "session": "E"}), 4)
    rp_d = s.rps[bytes(c[3])]
    check("dt_on_no_field_rp_keeps_ttl_none (class 'default', not an explicit TTL)", rp_d.ttl_s is None
          and "ttl_classes=default/1h/1d/3d/exp" in s.stats_line() and "ttl_rps=1/1/0/0/0" in s.stats_line(),
          s.stats_line()[-160:])
    w.t += 90
    s.get_num_new_matched_tokens(tp.req("d6a", c[:7]), 4 * HBS)        # the conversation's next request (no fields)
    check("dt_on_extending_request_refreshes_last_used", rp_d.last_used == w.t and s.stats["rp_ttl_refresh"] >= 1)
    w.t += 90
    s.build_meta()
    check("dt_on_alive_90s_after_refresh", bytes(c[3]) in s.rps)
    w.t += 11
    pool.evict_all_cached()
    r = s.get_num_new_matched_tokens(tp.req("d6b", c[:7]), 0)
    check("dt_on_expired_never_restored", r == (0, False) and s.stats["restore_started"] == 0, f"{r}")
    s.build_meta()
    s.build_meta()
    check("dt_on_sweep_drops_only_the_default_class", bytes(c[3]) not in s.rps and len(s.rps) == 1
          and s.stats["rp_expired"] == 1, f"rps={len(s.rps)} expired={s.stats['rp_expired']}")
    # an expired default RP saved again by its conversation is used again (revived), not rewritten
    w = W()
    s, pool, _, _ = tp.sched(spill, root + "/dt2", w, ttl_sweep_s=1e9, default_ttl_s=100)
    r4 = tp.req("r4", tp.chain(b"r", 6)[:5])
    tp.save(spill, s, pool, r4, 4)
    w.t += 150
    s.save_request(r4)
    rp = next(iter(s.rps.values()))
    check("dt_resave_revives_expired_default_rp", rp.last_used == w.t and not s._is_expired(rp))
    # save policy unchanged: SAVE_POLICY=repeat writes no first turn of a no-field request, explicit keep does
    w = W()
    s, pool, _, _ = tp.sched(spill, root + "/dt3", w, save_policy="repeat", ttl_save_policy="all", default_ttl_s=86400)
    tp.save(spill, s, pool, tp.req("dflt", tp.chain(b"g", 8)[:5]), 4)
    tp.save(spill, s, pool, tp.req("keep", tp.chain(b"f", 8)[:5], {"ttl_s": 86400, "session": "S"}), 4)
    kept = sorted(x.key[:1].decode() for x in s.rps.values())
    check("dt_save_policy_unchanged_no_first_turn_for_no_field", kept == ["f"] and s.stats["rp_skipped_first"] == 1, f"{kept}")
    # eviction order at the cap: identical with DEFAULT_TTL_S 0 and 86400 (nothing expired): default class first
    s0, _p0, _, _ = tp.sched(spill, root + "/dt_probe", W())
    per_rp = 3 * s0.file_bytes[0] + s0.file_bytes[2]
    orders = {}
    for dt in (0, 86400):
        w = W()
        s, pool, _, _ = tp.sched(spill, root + f"/dt_ev{dt}", w, cap_bytes=4 * per_rp, n=256, default_ttl_s=dt)
        for tag, ttl in [("P", 86400), ("Q", 3600), ("R", 3600), ("S", None)]:
            p = None if ttl is None else {"ttl_s": ttl, "session": "s" + tag}
            tp.save(spill, s, pool, tp.req(tag, tp.chain(tag.encode(), 4), p), 3)
            w.t += 10
        ev = []
        for tag in ["T", "U", "V"]:
            before = {x.key[:1] for x in s.rps.values()}
            tp.save(spill, s, pool, tp.req(tag, tp.chain(tag.encode(), 4), {"ttl_s": 259200, "session": "x" + tag}), 3)
            ev += sorted(x.decode() for x in before - {x.key[:1] for x in s.rps.values()})
        orders[dt] = ev
    check("dt_cap_order_unchanged_default_first", orders[0] == orders[86400] == ["S", "Q", "R"], f"{orders}")
    # an expired default RP goes first (with the expired), ahead of an older explicit one
    w = W()
    s, pool, _, _ = tp.sched(spill, root + "/dt_ev_exp", w, cap_bytes=2 * per_rp, n=256, default_ttl_s=50,
                             ttl_sweep_s=1e9)
    tp.save(spill, s, pool, tp.req("A", tp.chain(b"A", 4), {"ttl_s": 3600, "session": "sA"}), 3)
    w.t += 10
    tp.save(spill, s, pool, tp.req("B", tp.chain(b"B", 4)), 3)
    w.t += 60
    tp.save(spill, s, pool, tp.req("C", tp.chain(b"C", 4), {"ttl_s": 3600, "session": "sC"}), 3)
    left = sorted(x.key[:1].decode() for x in s.rps.values())
    check("dt_cap_expired_default_evicted_first", left == ["A", "C"], f"{left}")
    # reuse across restarts: an expired default RP is dropped at adoption; with DEFAULT_TTL_S=0 it is adopted
    for dt, want in ((3600, ["b"]), (0, ["b", "c"])):
        w = W()
        rr = root + f"/dt_px{dt}"
        s, pool, _clk, _io, m1, m2 = tp.boot(spill, rr, w, default_ttl_s=dt)
        tp.save(spill, s, pool, tp.req("b", tp.chain(b"b", 6)[:5], {"ttl_s": 259200, "session": "B"}), 4)
        tp.save(spill, s, pool, tp.req("c", tp.chain(b"c", 6)[:5]), 4)
        s.shutdown()
        files = tp.disk_files(s)
        w.t += 7200
        s2, *_r, m1, m2 = tp.boot(spill, rr, w, (files, files), default_ttl_s=dt)
        left = sorted(x.key[:1].decode() for x in s2.rps.values())
        check(f"dt_adoption_default_ttl_{dt}", left == want, f"{left} {s2.persist_state} {s2.persist_reason[:200]}")
    # config
    os.environ["GLM53_NVME_SPILL_DIR"] = os.path.join(root, "cfgd")
    try:
        check("dt_from_env_default_0", spill.SpillConfig.from_env().default_ttl_s == 0)
        os.environ["GLM53_NVME_SPILL_DEFAULT_TTL_S"] = "86400"
        check("dt_from_env_86400", spill.SpillConfig.from_env().default_ttl_s == 86400)
        os.environ["GLM53_NVME_SPILL_DEFAULT_TTL_S"] = ""
        check("dt_from_env_empty_is_default_0", spill.SpillConfig.from_env().default_ttl_s == 0)
        for bad in ("-1", str(7 * 86400 + 1), "1.5x"):
            os.environ["GLM53_NVME_SPILL_DEFAULT_TTL_S"] = bad
            try:
                spill.SpillConfig.from_env()
                check(f"dt_from_env_rejects_{bad}", False)
            except ValueError:
                check(f"dt_from_env_rejects_{bad}", True)
    finally:
        os.environ.pop("GLM53_NVME_SPILL_DIR", None)
        os.environ.pop("GLM53_NVME_SPILL_DEFAULT_TTL_S", None)
    check("dt_knob_forwarded (last in KNOBS)", spill.KNOBS[-1] == "DEFAULT_TTL_S")


# ------------------------------------------------------------------------------------------------ backed-first eviction
def t_backed_first(spill, root):
    """_take_blocks on the REAL vLLM BlockPool: never-hashed free blocks first, then NVMe-backed cached blocks (LRU), only
    then blocks that exist only in GPU memory (LRU); ref counts / free queue / hash map as get_new_blocks leaves them."""
    from vllm.v1.core.block_pool import BlockPool

    def setup(backed_first=True):
        s, _fp, clk, io = tsf.new_sched(spill, os.path.join(root, "bf"), backed_first=backed_first)
        pool = BlockPool(16, True, hash_block_size=tsf.HBS)
        s.block_pool = pool
        blocks = pool.get_new_blocks(15)                     # ids 1..15
        for b in blocks[:10]:                                # 1..10 cached (key per block)
            pool._insert_block_hash(tsf.K(b"k" + bytes([b.block_id]), 0), b, tsf.HBS)
        pool.free_blocks(blocks[:10] + blocks[10:])          # queue: 11..15 (unhashed, head), then 1..10 (LRU)
        for bid in (3, 7, 9):                                # their bytes are in ok files on this tier
            key = bytes(pool.blocks[bid].block_hash)
            s.disk[spill.block_file_name(key)] = spill._DiskBlock(spill.block_file_name(key), 0, 1, state="ok", refs=1)
        return s, pool

    s, pool = setup()
    got = [b.block_id for b in s._take_blocks(7)]
    check("bf_unhashed_then_backed", sorted(got) == [3, 7, 11, 12, 13, 14, 15], f"{got}")
    check("bf_unbacked_older_kept_cached", all(pool.blocks[i].block_hash is not None for i in (1, 2, 4, 5, 6, 8, 10)))
    check("bf_taken_evicted_and_referenced", all(pool.blocks[i].block_hash is None and pool.blocks[i].ref_cnt == 1
                                                 for i in got) and pool.get_num_free_blocks() == 15 - 7)
    got2 = [b.block_id for b in s._take_blocks(4)]
    check("bf_then_rest_in_lru_order", got2 == [9, 1, 2, 4], f"{got2}")
    check("bf_stats_took_backed", s.stats["restore_took_backed"] == 3, str(s.stats["restore_took_backed"]))
    q = pool.free_block_queue
    walk, node = [], q.fake_free_list_head.next_free_block
    while node is not q.fake_free_list_tail:
        walk.append(node.block_id)
        node = node.next_free_block
    check("bf_free_queue_consistent", walk == [5, 6, 8, 10] and q.num_free_blocks == 4
          and all(pool.blocks[i].ref_cnt == 0 for i in walk), f"{walk}")
    mapped = {bid for bid in range(1, 16) if pool.blocks[bid].block_hash is not None}
    check("bf_hash_map_matches_blocks", mapped == {5, 6, 8, 10} and all(
        pool.cached_block_hash_to_block.get_one_block(pool.blocks[i].block_hash) is pool.blocks[i] for i in mapped))
    s, pool = setup(backed_first=False)
    got = [b.block_id for b in s._take_blocks(7)]
    check("bf_disabled_is_plain_get_new_blocks", got == [11, 12, 13, 14, 15, 1, 2], f"{got}")
    try:
        s, pool = setup()
        s._take_blocks(16)
        check("bf_too_many_raises_like_get_new_blocks", False)
    except ValueError:
        check("bf_too_many_raises_like_get_new_blocks", pool.get_num_free_blocks() == 15)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--module", default=os.path.join(HERE, "..", "code", "glm53_nvme_spill_connector.py"))
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("spill_under_test", a.module)
    spill = importlib.util.module_from_spec(spec)
    sys.modules["spill_under_test"] = spill
    spec.loader.exec_module(spill)
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    assert os.environ.get("PYTHONHASHSEED"), "run with PYTHONHASHSEED set (as test_ttl_persist.py)"
    init_none_hash(sha256)          # the EngineCore order; the persist decision checks NONE_HASH
    root = os.path.abspath(a.dir)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    for t in (t_code_digest, t_site_digest, t_identity_parts, t_policy_equivalence, t_restore_gated,
              t_stats_and_config, t_default_ttl, t_backed_first):
        try:
            t(spill, root)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(t.__name__ + "_raised", False, repr(e))
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"SUMMARY {len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
