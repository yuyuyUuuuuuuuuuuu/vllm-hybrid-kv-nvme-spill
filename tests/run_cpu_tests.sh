#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# CPU test suite of the spill connector, run from this tree inside the serving image (one container at a time,
# no network, 4 GiB, no swap). Needs the rig work tree (tests/rig/setup_work.sh) for vLLM and RIG_MODELS_DIR (model +
# drafter config directories, read-only; only the two tests that build a VllmConfig read them).
#   usage: RIG_MODELS_DIR=<dir> tests/run_cpu_tests.sh [test ...]     (default: all)
#   tests: store sched_faults prune ttl_persist hash_stability alloc_conf bench_persist quality_probe
# Logs: <repo>/test-out/<test>.txt; one "SUMMARY"/result line per test on stdout. test_start_sh.sh is separate
# (host only, needs a launcher kit): tests/test_start_sh.sh <scratch> <kit dir> [<launcher dir>].
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/test-out"; mkdir -p "$OUT/scratch"
IMG="${RIG_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
[ -d "$REPO/tests/rig/work/vllm" ] || { echo "run_cpu_tests: no rig work tree; run tests/rig/setup_work.sh first" >&2; exit 2; }
MODELS="${RIG_MODELS_DIR:-}"
run() {   # run <name> <hashseed: 0|none> <python args...>
  local name="$1" hs="$2"; shift 2
  local mv=() he=(-e PYTHONHASHSEED=0)
  [ -n "$MODELS" ] && mv=(-v "$MODELS:/mnt/models:ro")
  [ "$hs" = none ] && he=()
  rm -rf "$OUT/scratch/$name"
  timeout 1800 docker run --rm --network none --memory 4g --memory-swap 4g --cpus 2 -u "$(id -u):$(id -g)" \
    --name "nvmespill-cpu-$name" --tmpfs /full:size=40m,uid="$(id -u)" \
    -v "$REPO:/w" -w /w -v "$OUT/scratch:/spill" "${mv[@]}" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 "${he[@]}" \
    -e PYTHONPATH=/w/tests/rig/work -e VLLM_LOGGING_LEVEL=WARNING \
    --entrypoint python3 "$IMG" "$@" > "$OUT/$name.txt" 2>&1
  local rc=$?
  rm -rf "$OUT/scratch/$name"
  echo "$name rc=$rc $(grep -E '^(SUMMARY|ALLOC)' "$OUT/$name.txt" | tr '\n' ' ')"
}
want="${*:-store sched_faults prune ttl_persist hash_stability alloc_conf bench_persist quality_probe}"
for t in $want; do
  case "$t" in
    store)          run store 0 tests/test_store.py --dir /spill/store --full-dir /full --device cpu ;;
    sched_faults)   run sched_faults 0 tests/test_sched_faults.py --dir /spill/sched_faults ;;
    prune)          run prune 0 tests/test_prune.py --dir /spill/prune ;;
    ttl_persist)    run ttl_persist 0 tests/test_ttl_persist.py --dir /spill/ttl_persist ;;
    hash_stability) run hash_stability none tests/t_hash_stability.py --spill /spill/hash_stability ;;
    alloc_conf)     run alloc_conf 0 tests/t_alloc_conf.py published-tree ;;
    bench_persist)  run bench_persist 0 tests/bench_persist.py --dir /spill/bench_persist --rps 60 --blocks 170 --files 6000 ;;
    quality_probe)  python3 "$REPO/tests/test_quality_probe.py" > "$OUT/quality_probe.txt" 2>&1
                    echo "quality_probe rc=$? $(grep -E '^SUMMARY' "$OUT/quality_probe.txt")" ;;
    *) echo "unknown test $t" >&2; exit 2 ;;
  esac
done
