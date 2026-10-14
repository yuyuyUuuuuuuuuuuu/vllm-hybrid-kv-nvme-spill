#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Aggregate the superseded-RP pruning rig runs (results/prune/*.json).

usage: aggregate_prune.py <results dir> <prefix> [<prefix> ...]
  prefix = run name without "_s<seed>" (e.g. m_k2 -> m_k2_s1..m_k2_s5). One table row per prefix: means over seeds of
  DSH re-read, collapses, all re-read, stall estimate, restores, per-rank disk peak / steady (mean of the 5-min samples in
  the second half of the run) / end, RPs at the end, pruned RPs and bytes, LRU evictions; then the DSH re-read split by
  turn kind and which RP the restores used (newer_alive = newer RPs of the same chain still on disk).
"""
import glob
import json
import os
import re
import statistics as stt
import sys


def load(d, prefix):
    out = []
    for p in sorted(glob.glob(os.path.join(d, prefix + "_s*.json"))):
        if not re.fullmatch(re.escape(prefix) + r"_s\d+\.json", os.path.basename(p)):
            continue
        try:
            out.append(json.load(open(p)))
        except (OSError, ValueError):
            pass
    return out


def steady(tl):
    if not tl:
        return 0.0
    t_end = tl[-1][0]
    xs = [g for t, g, _ in tl if t >= t_end / 2]
    return stt.mean(xs) if xs else 0.0


def main():
    d = sys.argv[1]
    rows = []
    for prefix in sys.argv[2:]:
        runs = load(d, prefix)
        if not runs:
            print(f"{prefix}: no runs")
            continue
        sp = [r["spill"] or {} for r in runs]
        m = lambda xs: stt.mean(xs) if xs else 0.0
        row = {
            "name": prefix, "n": len(runs),
            "dsh": m([r["dsh_re_read_tokens"] for r in runs]),
            "all": m([r["re_read_tokens"] for r in runs]),
            "falls": m([r["dsh_falls_le23040"] for r in runs]),
            "falls_list": [r["dsh_falls_le23040"] for r in runs],
            "stall": m([r["stalled_s_est"] for r in runs]),
            "fin": m([r["finished"] for r in runs]),
            "rest": m([s.get("restore_started", 0) for s in sp]),
            "rest_ok": m([s.get("restore_ok", 0) for s in sp]),
            "peak": m([s.get("disk_gib_peak", 0) for s in sp]),
            "peak_max": max([s.get("disk_gib_peak", 0) for s in sp] or [0]),
            "steady": m([steady(s.get("disk_timeline", [])) for s in sp]),
            "end": m([s.get("disk_gib_accounted", 0) for s in sp]),
            "rps": m([s.get("rps_end", 0) for s in sp]),
            "pruned": m([s.get("rp_pruned", 0) for s in sp]),
            "pruned_gib": m([s.get("prune_freed_bytes", 0) / (1 << 30) for s in sp]),
            "evicted": m([s.get("rp_evicted", 0) for s in sp]),
            "skip_busy": m([s.get("rp_prune_skipped_busy", 0) for s in sp]),
            "kept_branch": m([s.get("rp_prune_kept_branch", 0) for s in sp]),
            "verify_fail": sum(s.get("verify_fail", 0) + s.get("file_verify_fail", 0) for s in sp),
            "restore_short": sum(s.get("restore_short", 0) for s in sp),
            "restore_failed": sum(s.get("restore_failed", 0) for s in sp),
            "pinned_end": sum(s.get("connector_pinned_end", 0) for s in sp),
            "inv": sum(len(r.get("invariant_errors", [])) + r.get("hit_hash_violations", 0)
                       + r.get("accounting_violations", 0) for r in runs),
            "branch": {}, "newer": {},
        }
        for r in runs:
            for k, v in (r.get("dsh_by_branch") or {}).items():
                b = row["branch"].setdefault(k, {"admissions": 0, "re_read": 0, "falls": 0})
                for kk in b:
                    b[kk] += v.get(kk, 0)
        for s in sp:
            for k, v in (s.get("restore_newer_alive_by_branch") or {}).items():
                b = row["newer"].setdefault(k, {})
                for kk, n in v.items():
                    b[kk] = b.get(kk, 0) + n
        rows.append(row)
    print("| run | n | DSH re-read | all re-read | collapses (per seed) | stall s | finished | restores (ok) | "
          "disk GiB/rank peak (max) | steady | end | RPs end | pruned RPs / GiB | LRU evicted | prune skipped busy / "
          "kept for branch | verify fail | restore short/failed | pinned end | invariant viol. |")
    print("|" + "---|" * 20)
    for r in rows:
        print(f"| {r['name']} | {r['n']} | {r['dsh']:,.0f} | {r['all']:,.0f} | {r['falls']:.1f} {r['falls_list']} | "
              f"{r['stall']:,.0f} | {r['fin']:.1f} | {r['rest']:.1f} ({r['rest_ok']:.1f}) | {r['peak']:.2f} "
              f"({r['peak_max']:.2f}) | {r['steady']:.2f} | {r['end']:.2f} | {r['rps']:.1f} | {r['pruned']:.1f} / "
              f"{r['pruned_gib']:.2f} | {r['evicted']:.1f} | {r['skip_busy']:.1f} / {r['kept_branch']:.1f} | "
              f"{r['verify_fail']} | {r['restore_short']}/{r['restore_failed']} | {r['pinned_end']} | {r['inv']} |")
    if any(len(r["branch"]) > 1 for r in rows):
        print()
        print("DSH re-read by turn kind (sum over seeds: admissions / re-read tokens / collapses):")
        for r in rows:
            parts = [f"{k}: {v['admissions']} / {v['re_read']:,} / {v['falls']}" for k, v in sorted(r["branch"].items())]
            print(f"  {r['name']}: " + " | ".join(parts))
    print()
    print("restores by turn kind -> newer RPs of the same chain alive at restore time (0 = restored the newest):")
    for r in rows:
        parts = [f"{k}: " + ",".join(f"{kk}:{n}" for kk, n in sorted(v.items())) for k, v in sorted(r["newer"].items())]
        print(f"  {r['name']}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
