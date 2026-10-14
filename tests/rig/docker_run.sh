#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# docker_run.sh <env-file-or-empty> <args for run_one.sh: mode seed hours name [extra]> : one CPU-only container
# (the serving image, 3 CPUs, RIG_MEM (default 8 GiB), no swap, no network), named nvmespill-rig-<name>.
# Both arms get the kit start.sh's allocator setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (override with
# RIG_ALLOC_CONF, e.g. RIG_ALLOC_CONF= for an unpatched-tree control): the rig builds VllmConfig WITH the
# kv_transfer_config, so VllmConfig._verify_kv_transfer_compat runs exactly as at production start.
# Required: RIG_MODELS_DIR (host dir holding the model + drafter config directories, mounted read-only at /mnt/models;
# see REPRODUCE.md). Optional: NVMESPILL_SCRATCH (host spill scratch dir, default <repo>/rig-scratch),
# RIG_IMAGE, RIG_PYTHONPATH (container path of the rig work tree), NVMESPILL_RESULTS (container results dir).
W="$(cd "$(dirname "$0")/../.." && pwd)"
ENVF="$1"; shift
EARGS=""
[ -n "$ENVF" ] && EARGS="--env-file $ENVF"
MODELS="${RIG_MODELS_DIR:?set RIG_MODELS_DIR (REPRODUCE.md)}"
SP="${NVMESPILL_SCRATCH:-$W/rig-scratch}"
IMG="${RIG_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor}"
mkdir -p "$SP"
exec docker run --rm --name "nvmespill-rig-$4" --memory "${RIG_MEM:-8g}" --memory-swap "${RIG_MEM:-8g}" --cpus 3 --network none \
  --entrypoint sh -u "$(id -u):$(id -g)" \
  -v "$W:/w" -v "$MODELS:/mnt/models:ro" -v "$SP:/spill" \
  -e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="${RIG_PYTHONPATH:-/w/tests/rig/work}" \
  -e HOME=/tmp -e "PYTORCH_CUDA_ALLOC_CONF=${RIG_ALLOC_CONF-expandable_segments:True}" \
  -e "NVMESPILL_RESULTS=${NVMESPILL_RESULTS:-/w/rig-out}" \
  -e NVMESPILL_RIG_FREE_LOG="${NVMESPILL_RIG_FREE_LOG:-1}" -e GLM53_NVME_SPILL_MIN_FREE_GIB=10 -e GLM53_NVME_SPILL_LOG=0 -e VLLM_LOGGING_LEVEL=WARNING $EARGS \
  "$IMG" /w/tests/rig/run_one.sh "$@"
