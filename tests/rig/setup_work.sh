#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# (Re)create the rig's work copy of the production vLLM tree and install the spill connector into it with the
# same patch script the serving container runs at start (code/patch_nvme_spill.py).
#   usage: PROD_SRC=<dir holding the patched vllm package> tests/rig/setup_work.sh [work dir name, default work]
# PROD_SRC is the image's vllm package after the launcher's start-up overlays (see REPRODUCE.md, "rig").
set -e
RIG="$(cd "$(dirname "$0")" && pwd)"
SRC="${PROD_SRC:?set PROD_SRC to the patched vllm package directory (REPRODUCE.md)}"
WORK="$RIG/${1:-work}"
[ -f "$SRC/v1/core/sched/scheduler.py" ] || { echo "setup_work: $SRC does not look like a vllm package" >&2; exit 2; }
rm -rf "$WORK/vllm"
mkdir -p "$WORK"
cp -a "$SRC" "$WORK/vllm"
GLM53_NVME_SPILL=1 GLM53_SITEPKG="$WORK" python3 "$RIG/../../code/patch_nvme_spill.py"
# idempotence: a second run must be a no-op
GLM53_NVME_SPILL=1 GLM53_SITEPKG="$WORK" python3 "$RIG/../../code/patch_nvme_spill.py"
