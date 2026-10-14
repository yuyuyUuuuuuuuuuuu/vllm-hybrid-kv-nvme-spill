#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Aggregate the TTL / restart-reuse rig runs (results/ttl/*.json, 2026-10-14).

usage: aggregate_ttl.py <results dir>
Tables: (1) 4h x seeds per condition: DSH re-read, collapses, writes, disk, restores, TTL counters;
(2) restart at 2 h (10 min down): re-read after the restart (all DSH admissions after it / the first admission of each
conversation after it), collapses after it, adoption; (3) TTL-mix per conversation OFF vs ON; (4) correctness/fault
runs (0.5 h, drain, strict): end state + persistence decision.
"""
import glob
import json
import os
import re
import statistics as stt
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ttl")


def load(prefix):
    out = []
    for p in sorted(glob.glob(os.path.join(R, prefix + "_s*.json"))):
        if re.fullmatch(re.escape(prefix) + r"_s\d+\.json", os.path.basename(p)):
            try:
                out.append(json.load(open(p)))
            except (OSError, ValueError):
                pass
    return out


m = lambda xs: stt.mean(xs) if xs else 0.0  # noqa: E731


def written_gib(d):
    """Bytes the scheduler shipped to each rank for saving (production file sizes; the rig's pages are 1/576)."""
    tot = 0
    for sp in (d.get("spill_before_restart"), d.get("spill")):
        if sp:
            tot += sp.get("save_bytes", 0)
    return tot / (1 << 30)


def main_table():
    print("## (1) 4h x seeds (--no-drain): DSH re-read / collapses / writes")
    print("| condition | n | DSH re-read | collapses [per seed] | all re-read | written GiB/rank | disk peak GiB/rank | "
          "restores ok | expired | epoch_dropped | ttl0_skipped | first-turn saves (TTL) | verify fail |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("m_off", "m_k2", "m_new", "m_ttl3", "m_ttl3r", "mx_off", "mx_on", "mx2_off", "mx2_on", "mx2_nottl", "rs_off",
                 "rs_p0", "rs_p1", "rs_cr"):
        runs = load(name)
        if not runs and name in ("m_off", "m_k2"):
            continue
        if not runs:
            print(f"| {name} | 0 | - |")
            continue
        sp = [(r.get("spill") or {}) for r in runs]
        spb = [(r.get("spill_before_restart") or {}) for r in runs]

        def s2(k):
            return m([(a.get(k, 0) or 0) + (b.get(k, 0) or 0) for a, b in zip(sp, spb)])
        print(f"| {name} | {len(runs)} | {m([r['dsh_re_read_tokens'] for r in runs]):,.0f} | "
              f"{m([r['dsh_falls_le23040'] for r in runs]):.1f} {[r['dsh_falls_le23040'] for r in runs]} | "
              f"{m([r['re_read_tokens'] for r in runs]):,.0f} | {m([written_gib(r) for r in runs]):.2f} | "
              f"{m([max(a.get('disk_gib_peak', 0), b.get('disk_gib_peak', 0) or 0) for a, b in zip(sp, spb)]):.2f} | "
              f"{s2('restore_ok'):.1f} | {s2('rp_expired'):.1f} | {s2('rp_epoch_dropped'):.1f} | {s2('rp_skipped_ttl0'):.1f} | "
              f"{s2('rp_saved_ttl_first'):.1f} | {s2('verify_fail'):.0f}/{s2('file_verify_fail'):.0f} |")


def restart_table():
    print("\n## (2) restart at 7,200 s (engine down 600 s; in-flight requests resent after it)")
    print("| condition | n | DSH re-read after restart | first turn of each conversation after it | collapses after | "
          "decision | adopted RPs | adopted GiB | dropped (missing/expired) | restores ok after | verify fail |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("rs_off", "rs_p0", "rs_p1", "rs_cr"):
        runs = load(name)
        if not runs:
            continue
        rs = [r["restart"] or {} for r in runs]
        sp = [r.get("spill") or {} for r in runs]
        dec = sorted({(x.get("persist_reason") or "-")[:40] for x in sp})
        print(f"| {name} | {len(runs)} | {m([x.get('dsh_re_read_after', 0) for x in rs]):,.0f} "
              f"{[x.get('dsh_re_read_after', 0) for x in rs]} | {m([x.get('dsh_first_after_re_read', 0) for x in rs]):,.0f} | "
              f"{m([x.get('dsh_falls_after', 0) for x in rs]):.1f} | {dec} | {m([x.get('persist_adopted_rps', 0) for x in sp]):.1f} | "
              f"{m([x.get('persist_adopted_bytes', 0) / (1 << 30) for x in sp]):.2f} | "
              f"{m([x.get('persist_dropped_missing', 0) for x in sp]):.1f}/{m([x.get('persist_dropped_expired', 0) for x in sp]):.1f} | "
              f"{m([x.get('restore_ok', 0) for x in sp]):.1f} | {sum(x.get('verify_fail', 0) for x in sp)}/"
              f"{sum(x.get('file_verify_fail', 0) for x in sp)} |")
    for name in ("rs_p0", "rs_p1"):
        for r in load(name):
            fa = (r["restart"] or {}).get("dsh_first_after", {})
            print(f"    {name} s{r['seed']}: first after restart " + ", ".join(
                f"c{c} {v['prompt']:,}->hit {v['hit']:,}" for c, v in sorted(fa.items())))


def ttlmix_table(off_p="mx_off", on_p="mx_on", nottl_p=None, title="TTL mix"):
    off, on = load(off_p), load(on_p)
    nt = load(nottl_p) if nottl_p else []
    if not off or not on:
        return
    print(f"\n## (3) {title} per conversation ({off_p} / {on_p}" + (f" / {nottl_p}" if nt else "") + f", n={len(off)}/{len(on)}"
          + (f"/{len(nt)}" if nt else "") + ")")
    print("| conv | ttl | DSH re-read OFF | ON (fields) | change | ON (no fields) | collapses OFF / ON / ON no fields | admissions |")
    print("|---|---|---|---|---|---|---|---|")
    ttl = {"0": "1h (idle)", "1": "1d (compaction)", "2": "3d", "3": "0"}
    g = lambda runs, c, k: m([r["dsh_by_conv"].get(c, {}).get(k, 0) for r in runs])  # noqa: E731
    for c in ("0", "1", "2", "3"):
        a, b, x = g(off, c, "re_read"), g(on, c, "re_read"), g(nt, c, "re_read")
        print(f"| c{c} | {ttl[c]} | {a:,.0f} | {b:,.0f} | {(b - a) / a * 100 if a else 0:+.1f}% | "
              f"{x:,.0f} | {g(off, c, 'falls'):.1f} / {g(on, c, 'falls'):.1f} / {g(nt, c, 'falls'):.1f} | "
              f"{g(off, c, 'admissions'):.1f}/{g(on, c, 'admissions'):.1f}/{g(nt, c, 'admissions'):.1f} |")
    tot = lambda runs: m([r["dsh_re_read_tokens"] for r in runs])  # noqa: E731
    print(f"| all | | {tot(off):,.0f} | {tot(on):,.0f} | {(tot(on) - tot(off)) / tot(off) * 100:+.1f}% | {tot(nt):,.0f} | | |")
    sp = [r.get("spill") or {} for r in on]
    print("    ON (fields): " + " ".join(f"{k}={m([x.get(k, 0) for x in sp]):.1f}" for k in (
        "rp_expired", "rp_epoch_dropped", "rp_skipped_ttl0", "rp_saved_stale", "rp_epoch_marked", "rp_marked_expired_busy",
        "rp_saved_ttl_first", "restore_started", "restore_admitted", "restore_timeouts")))


def fault_table():
    print("\n## (4) correctness / fault runs (0.5 h, drain, strict per-step invariants)")
    print("| run | injected | decision | adopted/dropped/orphans | restores ok/failed/fallback | saves ok/failed | breaker | "
          "verify blocks/fail, files/fail | END free/582 pinned quarantined unfinished | inv/hashviol/acct |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(glob.glob(os.path.join(R, "*.json"))):
        n = os.path.basename(p)[:-5]
        if n.startswith(("m_", "mx_", "mx2_", "rs_")) or n.startswith("bench_persist"):
            continue
        try:
            d = json.load(open(p))
        except ValueError:
            continue
        if not isinstance(d, dict) or "spill" not in d:
            continue
        s = d.get("spill") or {}
        inj = ((d.get("restart") or {}).get("inject") or "") or str(s.get("faults_injected", ""))
        print(f"| {n} | {inj or '-'} | {(s.get('persist_reason') or '-')[:60]} | {s.get('persist_adopted_rps', 0)}/"
              f"{s.get('persist_dropped_missing', 0)}/{s.get('persist_orphans', 0)} | {s.get('restore_ok')}/{s.get('restore_failed')}/"
              f"{s.get('restore_fallback')} | {s.get('rp_saved')}/{s.get('rp_save_failed')} | "
              f"{'OPEN' if s.get('breaker') else 'closed'} | {s.get('verify_blocks')}/{s.get('verify_fail')}, "
              f"{s.get('file_verify')}/{s.get('file_verify_fail')} | {d.get('end_free_blocks')} {s.get('connector_pinned_end')} "
              f"{s.get('quarantined_blocks_end')} {d.get('pending_unfinished')} | {len(d.get('invariant_errors', []))}/"
              f"{d.get('hit_hash_violations')}/{d.get('accounting_violations')} |")


main_table()
restart_table()
ttlmix_table(title="TTL mix, overloaded (4 conversations incl. c3 300k; idle 75 min at 1 h; compaction at 2 h)")
ttlmix_table("mx2_off", "mx2_on", "mx2_nottl", "TTL mix (c3 150k; c0 idle 100 min after its 3rd turn; c1 compaction at its 7th turn)")
fault_table()
