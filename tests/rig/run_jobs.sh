#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# run_jobs.sh <job list> <done marker> : run rig jobs RIG_PAR at a time (default 2; we ran 2 x 8 GiB containers).
# Job line: <env file under env/ or -> <on|off> <seed> <hours> <name> [extra run_sim args]; '#' lines are skipped.
# A job whose .rc already holds 0 is skipped. Each job: timeout ${JOB_TIMEOUT:-3600} s (4 h runs took longer on
# slower hosts; raise it). Results: <repo>/${RIG_OUT:-rig-out}/<name>.{json,log,rc} (never into results/).
# Jobs named *_oldcode* need a work tree of the pre-fix connector at tests/rig/work_old (not shipped; they reproduce
# the B6 race of the initial version, see docs/review-findings.md).
R="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$R/../.." && pwd)"
LIST="$1"; DONE="$2"
OUT="${RIG_OUT:-rig-out}"
RES_HOST="$REPO/$OUT"; mkdir -p "$RES_HOST"
run1() {
  set -- $1
  local envf="$1" name="$5"
  [ -f "$RES_HOST/$name.rc" ] && grep -q '^0$' "$RES_HOST/$name.rc" && { echo "skip $name"; return 0; }
  [ "$envf" = - ] && envf="" || envf="$R/env/$envf"
  local pp="" ac="expandable_segments:True"
  case "$name" in *_oldcode*) pp=/w/tests/rig/work_old; ac="";; esac
  local t0=$(date +%s)
  NVMESPILL_RESULTS="/w/$OUT" RIG_PYTHONPATH="${pp:-${RIG_WORK:-/w/tests/rig/work}}" RIG_ALLOC_CONF="$ac" \
    timeout ${JOB_TIMEOUT:-3600} "$R/docker_run.sh" "$envf" "${@:2}" > /dev/null 2>&1
  local rc=$?
  echo $rc > "$RES_HOST/$name.rc"
  echo "$(date +%T) $name rc=$rc $(( $(date +%s) - t0 ))s"
}
export -f run1; export R RES_HOST OUT
grep -vE '^\s*(#|$)' "$LIST" | xargs -P "${RIG_PAR:-2}" -I{} bash -c 'run1 "$@"' _ {}
echo "MATRIX_DONE $(date +%T)" > "$DONE"
