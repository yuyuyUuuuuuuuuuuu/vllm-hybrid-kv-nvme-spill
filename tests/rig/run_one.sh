#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# run_one.sh <on|off> <seed> <hours> <outname> [extra run_sim args]   (inside the serving image, CPU only)
set -e
MODE="$1"; S="$2"; H="$3"; NAME="$4"; shift 4
R="${NVMESPILL_RESULTS:-/w/rig-out}"
mkdir -p "$R"
OUT=$R/$NAME.json
if [ "$MODE" = on ]; then
  export NVMESPILL_ON=1
  export GLM53_NVME_SPILL_DIR=/spill/$NAME
  rm -rf "/spill/$NAME"
fi
# an env file can raise the engine log level (docker -e VLLM_LOGGING_LEVEL wins over --env-file entries)
[ -n "${NVMESPILL_LOGLEVEL:-}" ] && export VLLM_LOGGING_LEVEL="$NVMESPILL_LOGLEVEL"
rc=0
python3 /w/tests/rig/run_sim.py --seed "$S" --hours "$H" --max-steps 400000 --out "$OUT" "$@" \
  > "$R/$NAME.log" 2>&1 || rc=$?
[ "$MODE" = on ] && rm -rf "/spill/$NAME"
echo "DONE $NAME rc=$rc"
exit $rc
