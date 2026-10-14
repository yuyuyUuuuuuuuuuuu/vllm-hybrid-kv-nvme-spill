# SPDX-License-Identifier: AGPL-3.0-only
"""u4 rig regression: every field of each results/u4/u4_<job>.json equals results/finish/ff_<job>.json (the same job with
the finish connector eb2e1084) except timings and the index size (the per-name breakdown makes the index larger), and
the wipe reasons now carry names where a breakdown exists.

usage: python3 compare_u4.py [results dir]   (host)
"""
from __future__ import annotations

import json
import os
import re
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
VOLATILE = re.compile(r"(_s|_s\[\d+\]|_ms_max|^wall_s|index_bytes)$")


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(d, list) and len(d) < 50:
        for i, v in enumerate(d):
            out.update(flat(v, f"{p}[{i}]"))
    else:
        out[p] = d
    return out


def main() -> int:
    bad = 0
    jobs = sorted(f[3:-5] for f in os.listdir(os.path.join(R, "u4")) if f.startswith("u4_") and f.endswith(".json"))
    for j in jobs:
        a = flat(json.load(open(os.path.join(R, "finish", f"ff_{j}.json"))))
        b = flat(json.load(open(os.path.join(R, "u4", f"u4_{j}.json"))))
        for d in (a, b):        # the injected index damage reports the index size in bytes ("corrupt_index (85984 B)")
            if isinstance(d.get("restart.inject"), str):
                d["restart.inject"] = re.sub(r" \(\d+ B\)$", " (N B)", d["restart.inject"])
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and not VOLATILE.search(k))
        # a crash stops the asynchronous index writer wherever it is: how many periodic writes completed before the
        # crash instant is timing (the adoption that follows is compared field by field like everything else)
        timing = [k for k in diff if "crash" in j and k == "spill_before_restart.index_writes"]
        diff = [k for k in diff if k not in timing]
        reason = [k for k in diff if k.endswith("persist_reason")]
        other = [k for k in diff if k not in reason]
        ib = (a.get("spill.index_bytes"), b.get("spill.index_bytes"))
        ok = not other
        bad += not ok
        print(f"{'SAME' if ok else 'DIFF'} {j}: {len(a)} fields, non-timing differences {len(other)}"
              f"{' ' + str(other[:8]) if other else ''}; index_bytes {ib[0]} -> {ib[1]}")
        for k in timing:
            print(f"     timing: {k}: {a.get(k)} -> {b.get(k)} (writes completed before the crash instant)")
        for k in reason:
            print(f"     {k}: {a.get(k)!r} -> {b.get(k)!r}")
    print(f"SUMMARY {len(jobs) - bad}/{len(jobs)} jobs identical apart from timings / index size / wipe-reason wording")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
