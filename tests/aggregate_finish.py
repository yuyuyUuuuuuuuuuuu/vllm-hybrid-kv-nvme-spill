#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Aggregate the 2026-10-14 finish rig runs (B8 policy gate) against the earlier OFF / ON runs of the same seeds.

usage: aggregate_finish.py [results dir]   (default: results/ of this repository)
OFF and the previous ON (connector 1762e23f, the final version) come from results/fix, results/ttl, results/ttl/final;
the new connector from results/finish (fz_* 4 h x 5 seeds, ff_* fault suite + hit-hash reruns; gate_only/fm_* = the
policy gate alone, an intermediate connector used for the ablation). Same seed = the same
request stream (the rig is deterministic in sync mode; the instrumented rig reproduces the old runs exactly, e.g.
dg_mx_off_s1 / dg_mx_on_s1 == ttl/final mx2_off_s1 / mx2_on_s1 on every counter).

Classes: "one-shot" requests share no prefix with anything (sim/workload.py), so their re-read is prompt - hit with
hit = 0 except a re-admission after a preemption: the class total grows with the NUMBER of one-shots admitted in the
4 h (throughput), which is why the earlier "+23 %" is reported here per admitted request as well. "ttl 0" = the 4th
conversation c3 of the TTL mix (its requests carry ttl_s=0: never written, never restored, GPU cache only).
"""
import glob
import json
import os
import statistics as stt
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def load(sub, prefix, seeds=(1, 2, 3, 4, 5)):
    out = {}
    for s in seeds:
        p = os.path.join(R, sub, f"{prefix}_s{s}.json")
        if os.path.exists(p):
            try:
                out[s] = json.load(open(p))
            except ValueError:
                pass
    return out


m = lambda xs: stt.mean(xs) if xs else float("nan")  # noqa: E731


def one(d):
    adm = d["admissions"] - d["dsh_admissions"]
    rr = d["re_read_tokens"] - d["dsh_re_read_tokens"]
    return adm, rr


def sp(d, k):
    return (d.get("spill") or {}).get(k, 0) or 0


def plain_table():
    off, old, new = load("fix", "m_off"), load("fix", "m_on"), load("finish", "fz_m_on")
    print("## (1) normal workload, 4 h x 5 seeds (--no-drain): OFF / ON 1762e23f-equivalent / ON new (policy gate)")
    print("| seed | DSH re-read OFF / old / new | collapses OFF/old/new | one-shots admitted OFF/old/new | one-shot re-read per admitted "
          "OFF/old/new | restores started/admitted (old -> new) | loaded blocks old -> new | deferred_policy | verify fail |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(new):
        a, b, c = off.get(s), old.get(s), new[s]
        if not a or not b:
            continue
        oa, ob, oc = one(a), one(b), one(c)
        print(f"| {s} | {a['dsh_re_read_tokens']:,} / {b['dsh_re_read_tokens']:,} / {c['dsh_re_read_tokens']:,} | "
              f"{a['dsh_falls_le23040']}/{b['dsh_falls_le23040']}/{c['dsh_falls_le23040']} | {oa[0]}/{ob[0]}/{oc[0]} | "
              f"{oa[1] // max(1, oa[0]):,}/{ob[1] // max(1, ob[0]):,}/{oc[1] // max(1, oc[0]):,} | "
              f"{sp(b, 'restore_started')}/{sp(b, 'restore_admitted')} -> {sp(c, 'restore_started')}/{sp(c, 'restore_admitted')} | "
              f"{sp(b, 'restore_loaded_blocks')} -> {sp(c, 'restore_loaded_blocks')} | {sp(c, 'restore_deferred_policy')} | "
              f"{sp(c, 'verify_fail')}/{sp(c, 'file_verify_fail')} |")
    ss = [s for s in new if s in off and s in old]
    if ss:
        f = lambda runs, fn: m([fn(runs[s]) for s in ss])  # noqa: E731
        print(f"| mean | {f(off, lambda d: d['dsh_re_read_tokens']):,.0f} / {f(old, lambda d: d['dsh_re_read_tokens']):,.0f} / "
              f"{f(new, lambda d: d['dsh_re_read_tokens']):,.0f} | {f(off, lambda d: d['dsh_falls_le23040']):.1f}/"
              f"{f(old, lambda d: d['dsh_falls_le23040']):.1f}/{f(new, lambda d: d['dsh_falls_le23040']):.1f} | "
              f"{sum(one(off[s])[0] for s in ss)}/{sum(one(old[s])[0] for s in ss)}/{sum(one(new[s])[0] for s in ss)} (sum) | "
              + "/".join(f"{sum(one(x[s])[1] for s in ss) / max(1, sum(one(x[s])[0] for s in ss)):,.0f}" for x in (off, old, new))
              + f" (pooled) | {f(old, lambda d: sp(d, 'restore_started')):.1f}/{f(old, lambda d: sp(d, 'restore_admitted')):.1f} -> "
              f"{f(new, lambda d: sp(d, 'restore_started')):.1f}/{f(new, lambda d: sp(d, 'restore_admitted')):.1f} | "
              f"{f(old, lambda d: sp(d, 'restore_loaded_blocks')):.0f} -> {f(new, lambda d: sp(d, 'restore_loaded_blocks')):.0f} | | |")
        print(f"    DSH re-read vs OFF: old {(f(old, lambda d: d['dsh_re_read_tokens']) / f(off, lambda d: d['dsh_re_read_tokens']) - 1) * 100:+.1f}% "
              f"new {(f(new, lambda d: d['dsh_re_read_tokens']) / f(off, lambda d: d['dsh_re_read_tokens']) - 1) * 100:+.1f}%; "
              f"one-shot re-read total vs OFF: old {(sum(one(old[s])[1] for s in ss) / sum(one(off[s])[1] for s in ss) - 1) * 100:+.1f}% "
              f"new {(sum(one(new[s])[1] for s in ss) / sum(one(off[s])[1] for s in ss) - 1) * 100:+.1f}%; finished OFF/old/new "
              f"{sum(off[s]['finished'] for s in ss)}/{sum(old[s]['finished'] for s in ss)}/{sum(new[s]['finished'] for s in ss)}")


def conv_table(title, off_sub, off_p, old_sub, old_p, new_p):
    off, old, new = load(off_sub, off_p), load(old_sub, old_p), load("finish", new_p)
    ss = [s for s in sorted(new) if s in off and s in old]
    if not ss:
        return
    print(f"\n## {title} ({off_sub}/{off_p} / {old_sub}/{old_p} / finish/{new_p}, seeds {ss})")
    print("| conv | ttl | re-read OFF | old ON | new ON | old vs OFF | new vs OFF | collapses OFF/old/new | admissions OFF/old/new | "
          "per admission OFF/old/new |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    ttl = {"0": "1h", "1": "1d", "2": "3d", "3": "0"}
    for c in ("0", "1", "2", "3"):
        g = lambda runs, k: [runs[s]["dsh_by_conv"].get(c, {}).get(k, 0) for s in ss]  # noqa: E731
        a, b, x = m(g(off, "re_read")), m(g(old, "re_read")), m(g(new, "re_read"))
        na, nb, nx = sum(g(off, "admissions")), sum(g(old, "admissions")), sum(g(new, "admissions"))
        print(f"| c{c} | {ttl[c]} | {a:,.0f} | {b:,.0f} | {x:,.0f} | {(b / a - 1) * 100:+.1f}% | {(x / a - 1) * 100:+.1f}% | "
              f"{sum(g(off, 'falls'))}/{sum(g(old, 'falls'))}/{sum(g(new, 'falls'))} | {na}/{nb}/{nx} | "
              f"{sum(g(off, 're_read')) / max(1, na):,.0f}/{sum(g(old, 're_read')) / max(1, nb):,.0f}/{sum(g(new, 're_read')) / max(1, nx):,.0f} |")
    t = lambda runs: m([runs[s]["dsh_re_read_tokens"] for s in ss])  # noqa: E731
    print(f"| all DSH | | {t(off):,.0f} | {t(old):,.0f} | {t(new):,.0f} | {(t(old) / t(off) - 1) * 100:+.1f}% | {(t(new) / t(off) - 1) * 100:+.1f}% | "
          f"{sum(off[s]['dsh_falls_le23040'] for s in ss)}/{sum(old[s]['dsh_falls_le23040'] for s in ss)}/{sum(new[s]['dsh_falls_le23040'] for s in ss)} | | |")
    print("    per seed c3 re-read OFF/old/new: " + ", ".join(
        f"s{s} {off[s]['dsh_by_conv'].get('3', {}).get('re_read', 0):,}/{old[s]['dsh_by_conv'].get('3', {}).get('re_read', 0):,}/"
        f"{new[s]['dsh_by_conv'].get('3', {}).get('re_read', 0):,}" for s in ss))
    oo = [one(off[s]) for s in ss]; ol = [one(old[s]) for s in ss]; on = [one(new[s]) for s in ss]  # noqa: E702
    print(f"    one-shots admitted OFF/old/new {sum(x[0] for x in oo)}/{sum(x[0] for x in ol)}/{sum(x[0] for x in on)}, re-read per "
          f"admitted {sum(x[1] for x in oo) / sum(x[0] for x in oo):,.0f}/{sum(x[1] for x in ol) / sum(x[0] for x in ol):,.0f}/"
          f"{sum(x[1] for x in on) / sum(x[0] for x in on):,.0f}; restores started/admitted old {m([sp(old[s], 'restore_started') for s in ss]):.1f}/"
          f"{m([sp(old[s], 'restore_admitted') for s in ss]):.1f} new {m([sp(new[s], 'restore_started') for s in ss]):.1f}/"
          f"{m([sp(new[s], 'restore_admitted') for s in ss]):.1f}; loaded blocks old {m([sp(old[s], 'restore_loaded_blocks') for s in ss]):.0f} "
          f"new {m([sp(new[s], 'restore_loaded_blocks') for s in ss]):.0f}; verify fail new {sum(sp(new[s], 'verify_fail') + sp(new[s], 'file_verify_fail') for s in ss)}")
    de = [new[s].get("diag_evict") for s in ss if new[s].get("diag_evict")]
    if de:
        ev = lambda dd, own: sum(dd["evicted_by_owner_cause"].get(own, {}).values())  # noqa: E731
        rc = lambda dd: sum(v for o in dd["evicted_by_owner_cause"].values() for k, v in o.items() if k.startswith("restore"))  # noqa: E731
        print(f"    evictions (new, diag): c3 blocks evicted {m([ev(dd, 'c3') for dd in de]):.0f}/seed, by restores {m([rc(dd) for dd in de]):.0f}/seed")


def restart_table():
    old = {**load("ttl", "rs_p1"), **load("ttl/final", "rs_p1")}
    p0, off, new = load("ttl", "rs_p0"), load("ttl", "rs_off"), load("finish", "fz_rs_p1")
    ss = [s for s in sorted(new) if s in old]
    if not ss:
        return
    print(f"\n## (4) restart at 2 h (10 min down, PERSIST=1), seeds {ss}: DSH re-read after the restart")
    g = lambda runs, k: [runs[s]["restart"][k] for s in ss if s in runs]  # noqa: E731
    print(f"| | after restart | first turn of each conversation after it | collapses after | 4 h DSH re-read | adopted RPs |")
    print("|---|---|---|---|---|---|")
    for name, runs in (("OFF", off), ("PERSIST=0", p0), ("old PERSIST=1", old), ("new PERSIST=1", new)):
        rr = g(runs, "dsh_re_read_after")
        if not rr:
            continue
        print(f"| {name} (n={len(rr)}) | {m(rr):,.0f} {rr} | {m(g(runs, 'dsh_first_after_re_read')):,.0f} | {m(g(runs, 'dsh_falls_after')):.1f} | "
              f"{m([runs[s]['dsh_re_read_tokens'] for s in ss if s in runs]):,.0f} | "
              f"{m([sp(runs[s], 'persist_adopted_rps') for s in ss if s in runs]):.1f} |")
    a, b = m(g(p0, "dsh_re_read_after")), m(g(new, "dsh_re_read_after"))
    print(f"    new vs PERSIST=0 after the restart: {(b / a - 1) * 100:+.1f}% (old: {(m(g(old, 'dsh_re_read_after')) / a - 1) * 100:+.1f}%)")


def async_table():
    old, new = load("fix", "x_on"), load("finish", "fz_x_on")
    ss = [s for s in sorted(new) if s in old]
    if not ss:
        return
    print(f"\n## (5) async completion + rank latency skew, 4 h, seeds {ss}: DSH re-read old {[old[s]['dsh_re_read_tokens'] for s in ss]} "
          f"new {[new[s]['dsh_re_read_tokens'] for s in ss]}; collapses new {[new[s]['dsh_falls_le23040'] for s in ss]}; "
          f"verify fail new {[sp(new[s], 'verify_fail') for s in ss]}; restores started/admitted new "
          f"{[(sp(new[s], 'restore_started'), sp(new[s], 'restore_admitted')) for s in ss]}")


def fault_table():
    print("\n## (6) fault suite + hit-hash reruns (0.5 h, drain, strict per-step invariants), new connector")
    print("| run | restores ok/failed/fallback | saves ok/failed | breaker | verify blocks/fail, files/fail | END free pinned quarantined unfinished | "
          "inv/hashviol/acct | preemptions / re-admissions / re-admitted with a smaller hit | decision |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in sorted(x for x in glob.glob(os.path.join(R, "finish", "ff_*.json")) if not x.endswith(".viol.json")):
        try:
            d = json.load(open(p))
        except ValueError:
            continue
        s = d.get("spill") or {}
        print(f"| {os.path.basename(p)[3:-5]} | {s.get('restore_ok')}/{s.get('restore_failed')}/{s.get('restore_fallback')} | "
              f"{s.get('rp_saved')}/{s.get('rp_save_failed')} | {'OPEN' if s.get('breaker') else 'closed'} | {s.get('verify_blocks')}/"
              f"{s.get('verify_fail')}, {s.get('file_verify')}/{s.get('file_verify_fail')} | {d.get('end_free_blocks')} "
              f"{s.get('connector_pinned_end')} {s.get('quarantined_blocks_end')} {d.get('pending_unfinished')} | "
              f"{len(d.get('invariant_errors', []))}/{d.get('hit_hash_violations')}/{d.get('accounting_violations')} | "
              f"{d.get('preemptions')}/{d.get('readmissions', '-')}/{d.get('readmit_hit_shrank', '-')} | {(s.get('persist_reason') or '-')[:50]} |")


def ablation_table():
    print("\n## (0) ablation, seed 1 (2): policy gate only (gate_only/fm_*, connector c65c4fe5) vs gate + backed-first (fz_*)")
    for name, a, b in (("normal", "fm_m_on", "fz_m_on"), ("TTL mix", "fm_mx2_on", "fz_mx2_on"),
                       ("overloaded", "fm_mxo_on", "fz_mxo_on"), ("restart", "fm_rs_p1", "fz_rs_p1")):
        ga, fb = load("finish/gate_only", a), load("finish", b)
        for s_ in sorted(set(ga) & set(fb)):
            x, y = ga[s_], fb[s_]
            c3 = lambda d: d["dsh_by_conv"].get("3", {}).get("re_read", "-")  # noqa: E731
            print(f"    {name} s{s_}: DSH re-read {x['dsh_re_read_tokens']:,} -> {y['dsh_re_read_tokens']:,}, collapses "
                  f"{x['dsh_falls_le23040']} -> {y['dsh_falls_le23040']}, c3 {c3(x)} -> {c3(y)}, loaded blocks "
                  f"{sp(x, 'restore_loaded_blocks')} -> {sp(y, 'restore_loaded_blocks')} (backed taken {sp(y, 'restore_took_backed')})")


def dt_table():
    print("\n## (7) DEFAULT_TTL_S (4 h, seeds 1-2)")
    keys = ("dsh_re_read_tokens", "re_read_tokens", "dsh_falls_le23040", "admissions", "finished")
    spk = ("restore_started", "restore_ok", "rp_saved", "rp_evicted", "rp_expired", "rp_pruned", "save_bytes")

    def same(a, b):
        return all(a.get(k) == b.get(k) for k in keys) and all(sp(a, k) == sp(b, k) for k in spk) \
            and a.get("dsh_by_conv") == b.get("dsh_by_conv")
    for title, pa, pb in (("no-field mix: DEFAULT_TTL_S 0 vs 86400 (nothing unused for a day in 4 h -> identical)", "dt_nf0", "dt_nf86400"),
                          ("cap 8 GiB, c0/c3 no field + c1 1d / c2 3d: 0 vs 86400 (eviction order unchanged)", "dt_cap0", "dt_cap86400"),
                          ("normal workload: default 0 (fz_m_on) vs DEFAULT_TTL_S=86400", "fz_m_on", "dt_m86400")):
        a, b = load("finish", pa, (1, 2)), load("finish", pb, (1, 2))
        res = {s_: same(a[s_], b[s_]) for s_ in sorted(set(a) & set(b))}
        print(f"    {title}: identical per seed {res}; rp_evicted {[sp(a[s_], 'rp_evicted') for s_ in sorted(a)]} / "
              f"{[sp(b[s_], 'rp_evicted') for s_ in sorted(b)]}")
    a, b = load("finish", "dt_nf0", (1, 2)), load("finish", "dt_nf1800", (1, 2))
    for s_ in sorted(set(a) & set(b)):
        x, y = a[s_], b[s_]
        print(f"    no-field mix, DEFAULT_TTL_S 1800 vs 0, s{s_}: rp_expired {sp(x, 'rp_expired')} -> {sp(y, 'rp_expired')} "
              f"(expired GiB {sp(y, 'expired_bytes') / (1 << 30):.2f}), c0 (idle 100 min) re-read "
              f"{x['dsh_by_conv']['0']['re_read']:,} -> {y['dsh_by_conv']['0']['re_read']:,}, others "
              f"{sum(v['re_read'] for k, v in x['dsh_by_conv'].items() if k != '0'):,} -> "
              f"{sum(v['re_read'] for k, v in y['dsh_by_conv'].items() if k != '0'):,}")


ablation_table()
plain_table()
conv_table("(2) TTL mix, 4 conversations (c3 ttl 0, 150k; c0 1h idle 100 min; c1 1d compaction), 4 h x 5 seeds",
           "ttl/final", "mx2_off", "ttl/final", "mx2_on", "fz_mx2_on")
conv_table("(3) TTL mix overloaded (c3 300k; c0 idle 75 min at 1 h; c1 compaction at 2 h), 4 h x 5 seeds", "ttl", "mx_off",
           "ttl/final", "mxo_on", "fz_mxo_on")
restart_table()
async_table()
dt_table()
fault_table()
