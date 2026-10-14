#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# GPU tests of the spill worker on one GB10 (production page sizes), one container at a time.
#   usage: tests/run_gpu_tests.sh <scratch dir on the NVMe under test> <mode ...>
#   modes: store    test_store.py --device cuda + t_expandable_gpu.py (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
#          restart  t_restart_gpu.py: phase 1 and phase 2 in two separate containers (= a restart)
#          bench    bench_gpu.py: write/read of a 600k-token resume point (3.72 GB of GPU tensors + a 2 GiB
#                   bandwidth kernel; needs ~10 GB of free unified memory)
# The tests import the image's installed vLLM (with its compiled extensions), not the CPU rig work tree.
# Env: RIG_IMAGE, GPU_TEST_MEM (container memory, default 12g), GPU_GUARD_DIR (optional: a directory with a
# sitecustomize.py that caps torch CUDA allocations at GPU_MEM_CAP_GB; mounted on PYTHONPATH).
# Logs: <repo>/test-out/gpu_<mode>.txt. The scratch files are removed at the end.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP="${1:?scratch dir}"; shift
IMG="${RIG_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
mkdir -p "$SP" "$REPO/test-out"
[ -z "$(docker ps -q --filter name=nvmespill-gpu)" ] || { echo "refuse: another nvmespill GPU container is running" >&2; exit 4; }
guard=(); pp=
if [ -n "${GPU_GUARD_DIR:-}" ]; then
  guard=(-v "$GPU_GUARD_DIR:/opt/gpuguard:ro" -e "GPU_MEM_CAP_GB=${GPU_MEM_CAP_GB:-8}"); pp=/opt/gpuguard
fi
gpu() {   # gpu <python args...>
  timeout 1800 docker run --rm --gpus all --network none --name nvmespill-gpu \
    -u "$(id -u):$(id -g)" -v "$REPO:/w" -w /w -v "$SP:/spill" --memory "${GPU_TEST_MEM:-12g}" --memory-swap "${GPU_TEST_MEM:-12g}" \
    "${guard[@]}" -e PYTHONPATH="$pp" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONHASHSEED=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_LOGGING_LEVEL=WARNING \
    --entrypoint python3 "$IMG" "$@"
}
for mode in "$@"; do
  log="$REPO/test-out/gpu_$mode.txt"
  case "$mode" in
    store)   { gpu tests/test_store.py --dir /spill/ut --device cuda; echo "test_store rc=$?"
               gpu tests/t_expandable_gpu.py /spill/exp; echo "t_expandable_gpu rc=$?"; } > "$log" 2>&1
             rm -rf "$SP/ut" "$SP/exp" ;;
    restart) rm -rf "$SP/gpu_rt"
             { for ph in 1 2; do gpu tests/t_restart_gpu.py --dir /spill/gpu_rt --phase $ph; echo "PHASE $ph rc=$?"; done; } > "$log" 2>&1
             rm -rf "$SP/gpu_rt" ;;
    bench)   { gpu tests/bench_gpu.py --dir /spill/bench --out /spill/bench.json; echo "bench rc=$?"; cat "$SP/bench.json"; } > "$log" 2>&1
             rm -rf "$SP/bench" "$SP/bench.json" ;;
    *) echo "unknown mode $mode" >&2; exit 2 ;;
  esac
  echo "$mode: $(grep -E '^(SUMMARY|PHASE|bench rc|test_store rc|t_expandable_gpu rc)' "$log" | tr '\n' ' ')"
done
