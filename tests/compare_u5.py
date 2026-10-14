# SPDX-License-Identifier: AGPL-3.0-only
"""u5 rig results.

1. Regression: every field of each results/u5/u5_<job>.json equals results/u4/u4_<job>.json (the same job with the
   u4 connector bd0f589e) except timings, the index size and the u5-only counters (index_writes, index_writes_<why>:
   the rig's drain loop asks has_pending_push_work, so the u5 connector writes the index when the drain ends).
2. Idle-at-end restart (pr_idle_*: a production-style restart = wait idle -> docker rm -f, after two probe-like requests): the RPs
   complete on every rank before the stop vs what the next start adopts. lost = ok_before - adopted - dropped_expired
   (an RP whose TTL ran out during the downtime is dropped by the TTL rule, not lost). *_u4code = connector bd0f589e.

usage: python3 compare_u5.py [results dir]   (host)
"""
from __future__ import annotations

import json
import os
import re
import sys

from compare_u4 import VOLATILE, flat

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
U5_ONLY = re.compile(r"(index_writes|index_writes_(idle|timer|shutdown))$")


def regression() -> int:
    bad = 0
    jobs = sorted(f[3:-5] for f in os.listdir(os.path.join(R, "u4")) if f.startswith("u4_") and f.endswith(".json"))
    for j in jobs:
        a = flat(json.load(open(os.path.join(R, "u4", f"u4_{j}.json"))))
        b = flat(json.load(open(os.path.join(R, "u5", f"u5_{j}.json"))))
        for d in (a, b):
            if isinstance(d.get("restart.inject"), str):
                d["restart.inject"] = re.sub(r" \(\d+ B\)$", " (N B)", d["restart.inject"])
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and not VOLATILE.search(k))
        u5 = [k for k in diff if U5_ONLY.search(k)]
        other = [k for k in diff if k not in u5]
        ok = not other
        bad += not ok
        print(f"{'SAME' if ok else 'DIFF'} {j}: {len(a)} fields, differences other than timing/index size/u5 counters "
              f"{len(other)}{' ' + str(other[:8]) if other else ''}")
        for k in u5:
            print(f"     u5 counter {k}: {a.get(k)} -> {b.get(k)}")
    print(f"REGRESSION {len(jobs) - bad}/{len(jobs)} jobs identical apart from timings / index size / u5 index counters")
    return bad


def idle() -> int:
    bad = 0
    names = sorted(f[:-5] for f in os.listdir(os.path.join(R, "u5")) if f.startswith("u5_pr_idle_") and f.endswith(".json"))
    for n in names:
        d = json.load(open(os.path.join(R, "u5", f"{n}.json")))
        i = d["restart"]["idle"]
        sp = d["spill"]
        lost = i["rps_ok_before"] - i["adopted_rps"] - i["dropped_expired"]
        old = n.endswith("_u4code")
        clean = (lost == 0 and i["ok_rps_missing_from_index"] == 0 and i["dropped_missing"] == 0
                 and sp.get("verify_fail") == 0 and sp.get("file_verify_fail") == 0 and not d.get("errors")
                 and not d.get("invariant_errors"))
        want = (not clean) if old else clean
        bad += not want
        print(f"{'OK  ' if want else 'BAD '} {n:30s} {'bd0f589e' if old else 'u5      '} ok_before={i['rps_ok_before']} "
              f"in_index_at_kill={i['ok_rps_in_index_at_kill']} adopted={i['adopted_rps']} "
              f"expired_in_downtime={i['dropped_expired']} lost={lost} orphan_files={i['orphan_files']} "
              f"idle_index_writes={i['index_writes_idle']} verify_fail={sp.get('verify_fail')}"
              f"{'  (reproduces the bug)' if old and want else ''}")
    print(f"IDLE {len(names) - bad}/{len(names)} as expected (u5: lost 0; bd0f589e: the bug reproduced)")
    return bad


if __name__ == "__main__":
    b = regression() + idle()
    sys.exit(1 if b else 0)
