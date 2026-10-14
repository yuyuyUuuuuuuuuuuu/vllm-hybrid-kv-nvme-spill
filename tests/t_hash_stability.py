# SPDX-License-Identifier: AGPL-3.0-only
"""Proof: are vLLM's prefix-cache block hashes the same in the next process (the precondition of reuse across restarts)?

vLLM (prod_src v1/core/kv_cache_utils.py) chains every block hash from NONE_HASH, which ``init_none_hash`` sets to
``os.urandom(32)`` unless PYTHONHASHSEED is set (then ``hash_fn(PYTHONHASHSEED)``). EngineCore (v1/engine/core.py)
constructs the Scheduler -- and with it the KV connector -- BEFORE it calls ``init_none_hash``. The connector (persist
on) sets PYTHONHASHSEED in its own process environment if unset, so NONE_HASH becomes deterministic, while Python's own
str hashing (fixed at interpreter start) stays randomized.

This script runs fresh interpreters WITHOUT PYTHONHASHSEED in their environment; each builds the real Scheduler of the
replay rig (production vLLM tree, connector installed by patch_nvme_spill.py, the EngineCore order: Scheduler, then
init_none_hash) and hashes the same token list. It also checks the order statically in the production core.py.

usage (prod image, PYTHONPATH = rig work tree, rig dir on sys.path, model dir mounted, NO PYTHONHASHSEED):
  python3 t_hash_stability.py --spill /spill/hs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def child() -> None:
    sys.path.insert(0, os.path.join(HERE, "rig"))
    sys.path.insert(0, HERE)
    from sim.configs import build_scheduler
    from vllm.v1.core import kv_cache_utils as kcu
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams

    env_before = os.environ.get("PYTHONHASHSEED")
    scheduler, hasher, _vc, _kvc = build_scheduler()       # Scheduler (+ connector) first, then init_none_hash
    toks = [(i * 7919) % 100_000 for i in range(3 * 4608 + 5)]
    r = Request(request_id="x", prompt_token_ids=toks, sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None, arrival_time=0.0, block_hasher=hasher)
    conn = scheduler.connector
    chk = conn._sched._check_hash() if conn is not None else None
    print(json.dumps({"none_hash": bytes(kcu.NONE_HASH).hex()[:32], "first": bytes(r.block_hashes[0]).hex()[:32],
                      "last": bytes(r.block_hashes[-1]).hex()[:32], "n": len(r.block_hashes),
                      "str_hash": hash("glm53-nvme-spill"), "env_before": env_before,
                      "env_after": os.environ.get("PYTHONHASHSEED"), "check": chk,
                      "seed_source": getattr(conn._sched, "seed_source", None) if conn is not None else None}))


def run(env_extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env.update(env_extra)
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env=env, capture_output=True,
                         text=True, timeout=600)
    line = [x for x in out.stdout.splitlines() if x.startswith("{")]
    if out.returncode or not line:
        raise RuntimeError(f"child failed rc={out.returncode}: {out.stderr[-2000:]}")
    return json.loads(line[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--spill", default="/spill/hs")
    a = ap.parse_args()
    if a.child:
        child()
        return 0
    check("parent_env_has_no_pythonhashseed", "PYTHONHASHSEED" not in os.environ)
    # static order in the production EngineCore: Scheduler(...) constructed before init_none_hash(...)
    import vllm.v1.engine.core as core_mod
    src = open(core_mod.__file__).read()
    i_sched = src.find("self.scheduler: SchedulerInterface = Scheduler(")
    i_none = src.find("init_none_hash(caching_hash_fn)")
    check("prod_core_order_scheduler_before_init_none_hash", 0 <= i_sched < i_none, f"{i_sched} < {i_none}")
    base = {"NVMESPILL_ON": "1", "GLM53_NVME_SPILL_LOG": "0", "VLLM_LOGGING_LEVEL": "ERROR"}
    on = [run({**base, "GLM53_NVME_SPILL_DIR": f"{a.spill}/on{i}", "GLM53_NVME_SPILL_PERSIST": "1"}) for i in (0, 1)]
    off = [run({**base, "GLM53_NVME_SPILL_DIR": f"{a.spill}/off{i}", "GLM53_NVME_SPILL_PERSIST": "0"}) for i in (0, 1)]
    for i, r in enumerate(on + off):
        print(f"  run {i}: {r}")
    check("persist_off_block_hashes_differ_between_processes", off[0]["first"] != off[1]["first"]
          and off[0]["none_hash"] != off[1]["none_hash"], "(stock vLLM without PYTHONHASHSEED: random NONE_HASH)")
    check("persist_off_check_reports_unstable", off[0]["check"] is None or off[0]["check"][0] is False,
          f"{off[0]['check']}")
    check("persist_on_block_hashes_identical_between_processes",
          on[0]["first"] == on[1]["first"] and on[0]["last"] == on[1]["last"] and on[0]["none_hash"] == on[1]["none_hash"])
    check("persist_on_seed_set_by_connector", on[0]["env_before"] is None and on[0]["env_after"] == "5353"
          and on[0]["seed_source"] == "connector")
    check("persist_on_check_reports_stable", on[0]["check"] and on[0]["check"][0] is True, f"{on[0]['check']}")
    check("python_str_hash_still_randomized", on[0]["str_hash"] != on[1]["str_hash"],
          f"{on[0]['str_hash']} vs {on[1]['str_hash']}")
    # an operator-set PYTHONHASHSEED wins (setdefault) and is also stable
    ext = [run({**base, "GLM53_NVME_SPILL_DIR": f"{a.spill}/ext{i}", "PYTHONHASHSEED": "77"}) for i in (0, 1)]
    check("operator_seed_respected_and_stable", ext[0]["first"] == ext[1]["first"] and ext[0]["seed_source"] == "env"
          and ext[0]["first"] != on[0]["first"])
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"SUMMARY {len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
