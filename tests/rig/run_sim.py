#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run one replay condition and print/serialize its summary.

usage: python3 run_sim.py --seed 1 [--hours 4] [--fix1] [--fix2] [--out r.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--max-steps", type=int, default=600_000)
    ap.add_argument("--fix1", action="store_true")
    ap.add_argument("--fix2", action="store_true")
    ap.add_argument("--cond", default="")
    ap.add_argument("--no-drain", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/ (prodlayout)
    from sim.driver import Rig

    rig = Rig(
        seed=args.seed,
        horizon_s=args.hours * 3600,
        max_steps=args.max_steps,
        drain=not args.no_drain,
        fix_hold_on_wait=args.fix1,
        fix_defer_boundary_free=args.fix2,
        cond=args.cond,
    )
    summary = rig.run()
    if rig.world is not None:
        rig.world.shutdown()        # join the I/O threads (an exit mid-copy aborts the interpreter)
    if os.environ.get("APCREREAD_ADM_DUMP"):
        with open(os.environ["APCREREAD_ADM_DUMP"], "w") as f:
            json.dump(rig.admissions, f, indent=0)
    if os.environ.get("APCREREAD_VIOL_DUMP"):
        with open(os.environ["APCREREAD_VIOL_DUMP"], "w") as f:
            json.dump({"hit_hash": rig.hit_hash_violations[:10],
                       "acct": rig.accounting_violations[:10],
                       "inv": rig.invariant_errors[:10],
                       "lookups": [r for r in rig.lookup_records if r[1]=="dsh0t0"][:12],
                       "allocs": [r for r in rig.alloc_records if r[1]=="dsh0t0"][:12],
                       "adms": [a for a in rig.admissions if a["req"]=="dsh0t0"][:5]}, f, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
