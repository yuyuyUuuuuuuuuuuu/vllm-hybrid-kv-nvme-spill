#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# B4 regression: code/apply_start_sh.py against a launcher kit's start.sh (we measured the r16z8p kit), then the
# patched start.sh's own validate_numeric_config / write_inner_scripts / launch_cluster run against fakes
# (docker/ssh/scp record their argv; nothing is started, no node is contacted), and every GLM53_NVME_SPILL* name is
# traced into BOTH ranks' `docker run` arguments with a distinct value.
#   usage: tests/test_start_sh.sh <scratch dir> <kit dir> [<launcher dir>]
#     <kit dir>       holds launcher/start.sh (+ MANIFEST.sha256, site/, overlay/, launcher/overlay/*.py for part 2)
#     <launcher dir>  the deployed launcher tree (start.sh, overlay/, env_nonsecret.txt = its non-secret .env lines);
#                     without it only part 1 (apply / idempotence / fail-closed / knob list) runs.
# Host only (bash, python3). Prints PASS/FAIL lines and SUMMARY; exit 1 on any FAIL.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="${1:?scratch dir}"; KIT="${2:?kit dir}"
PROD="${3:-${PROD_LAUNCHER:-}}"
APPLY="$REPO/code/apply_start_sh.py"
rm -rf "$D"; mkdir -p "$D"; D="$(cd "$D" && pwd)"
npass=0; nfail=0
ck() { if [ "$1" = 0 ]; then echo "PASS $2"; npass=$((npass+1)); else echo "FAIL $2"; nfail=$((nfail+1)); fi; }

KS="$KIT/launcher/start.sh"
if [ -f "$KIT/MANIFEST.sha256" ]; then
  (cd "$KIT" && grep -E '  \./launcher/start\.sh$' MANIFEST.sha256 | sha256sum --quiet -c -); ck $? "kit start.sh matches the kit MANIFEST ($(sha256sum "$KS" | cut -c1-16))"
else
  echo "SKIP no MANIFEST.sha256 in the kit (start.sh sha256 $(sha256sum "$KS" | cut -c1-16))"
fi

# --- 1. apply / idempotence / fail-closed -------------------------------------------------------------------------
python3 "$APPLY" "$KS" "$D/start.sh" > "$D/apply1.log" 2>&1; ck $? "apply to the kit start.sh: $(cat "$D/apply1.log")"
bash -n "$D/start.sh"; ck $? "patched start.sh passes bash -n"
python3 "$APPLY" "$D/start.sh" "$D/start2.sh" > "$D/apply2.log" 2>&1 && cmp -s "$D/start.sh" "$D/start2.sh"
ck $? "second apply is a no-op: $(cat "$D/apply2.log")"
sed 's/GLM53_NVME_SPILL_LOG_EVERY_S="${GLM53_NVME_SPILL_LOG_EVERY_S:-}"/GLM53_NVME_SPILL_LOG_EVERY_S=x/' "$D/start.sh" > "$D/drift.sh"
python3 "$APPLY" "$D/drift.sh" "$D/drift_out.sh" > "$D/apply3.log" 2>&1; rc=$?
[ $rc != 0 ] && [ ! -e "$D/drift_out.sh" ]; ck $? "a drifted patched start.sh aborts, nothing written: $(tail -1 "$D/apply3.log")"
sed 's/; do  # \[tf-exl3-fork\]$/; do  # [tf-exl3-fork] (moved)/' "$KS" > "$D/anchor.sh"
python3 "$APPLY" "$D/anchor.sh" "$D/anchor_out.sh" > "$D/apply4.log" 2>&1; rc=$?
[ $rc != 0 ] && [ ! -e "$D/anchor_out.sh" ]; ck $? "a kit start.sh without an anchor aborts, nothing written: $(tail -1 "$D/apply4.log")"
python3 - "$REPO" "$D/start.sh" > "$D/knobs.log" 2>&1 <<'EOF'
import ast, re, sys
repo, sh = sys.argv[1], sys.argv[2]
src = open(f"{repo}/code/glm53_nvme_spill_connector.py").read()
read = set(re.findall(r'(?<!environ\.)get\("([A-Z_]+)"', src))   # SpillConfig.from_env's get(), not os.environ.get
knobs = None
for n in ast.parse(src).body:
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "KNOBS" for t in n.targets):
        knobs = set(ast.literal_eval(n.value))
s = open(sh).read()
fwd_w = set(re.findall(r"\bGLM53_NVME_SPILL(?:_([A-Z_]+))?\b", s.split("; do  # [tf-exl3-fork] [nvme-spill]")[0].rsplit("for v in", 1)[1]))
print("from_env reads:", sorted(read)); print("KNOBS:", sorted(knobs))
missing = read - knobs
print("read but not in KNOBS:", sorted(missing))
sys.exit(1 if missing or not knobs else 0)
EOF
ck $? "every knob SpillConfig.from_env reads is in KNOBS (=forwarded): $(grep 'read but' "$D/knobs.log")"

# --- 2. the patched start.sh against fakes ------------------------------------------------------------------------
if [ -z "$PROD" ]; then
  echo "SKIP part 2 (no launcher dir given)"
  echo "SUMMARY $npass/$((npass+nfail)) passed (part 1 only)"
  [ $nfail = 0 ]; exit
fi
L="$D/launcher"; mkdir -p "$L" "$D/bin" "$D/rec" "$D/spill"
cp -a "$PROD/start.sh" "$PROD/overlay" "$L/"
rm -rf "$L/overlay/tf/site" "$L/overlay/tf/overlay"
cp -a "$KIT/site" "$L/overlay/tf/site"; cp -a "$KIT/overlay" "$L/overlay/tf/overlay"
cp -p "$KIT"/launcher/overlay/*.py "$L/overlay/"
cp -p "$D/start.sh" "$L/start.sh"
# the nvme-spill bundle files (ROLLOUT: overlay/*.py -> overlay/tf/overlay; the branch patch_tf_bundle.py)
cp -p "$REPO/code/glm53_nvme_spill_connector.py" "$REPO/code/patch_nvme_spill.py" "$L/overlay/tf/overlay/"
cp -p "$REPO/code/patch_tf_bundle.py" "$L/overlay/patch_tf_bundle.py"
# .env: production's non-secret env, no memprep (it would run on this host), a dummy key, the switch + every knob
{ grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$PROD/env_nonsecret.txt" | grep -v '^GLM53_MEMPREP='
  echo "VLLM_API_KEY=dummy-not-a-secret"
  echo "GLM53_NVME_SPILL=1"; echo "GLM53_NVME_SPILL_DIR=$D/spill"
  i=0
  for k in $(python3 -c "import ast;src=open('$REPO/code/glm53_nvme_spill_connector.py').read()
for n in ast.parse(src).body:
    if isinstance(n, ast.Assign) and any(getattr(t,'id','')=='KNOBS' for t in n.targets): print(' '.join(ast.literal_eval(n.value)))"); do
    [ "$k" = DIR ] && continue
    i=$((i+1)); echo "GLM53_NVME_SPILL_$k=v$i$k"
  done; } > "$L/.env"
for f in docker ssh scp rsync curl nvidia-smi; do
  cat > "$D/bin/$f" <<EOF
#!/usr/bin/env bash
n=\$(ls "$D/rec" | wc -l)
printf '%s\0' "$f" "\$@" > "$D/rec/\$(printf %04d \$n).$f"
case "$f \${1:-}" in
  "docker image"|"docker inspect") echo "sha256:0000";;
esac
exit 0
EOF
  chmod +x "$D/bin/$f"
done
cat > "$D/drive.sh" <<EOF
set -uo pipefail
export PATH="$D/bin:\$PATH"
cd "$L"
sed '\$d' start.sh > start_lib.sh     # every function, no main
source ./start_lib.sh
set +e
validate_numeric_config; echo "VALIDATE_RC=\$?"
write_inner_scripts; echo "INNER_RC=\$?"
MODEL_DIR=/models/x; DFLASH_MODEL_DIR=/models/d
# launcher-side files production has but this staged tree may not (chat template, ...): empty placeholders
for v in \$(compgen -v | grep -E '_HOST\$'); do
  p="\${!v}"
  case "\$p" in "$L"/*) [ -e "\$p" ] || { mkdir -p "\$(dirname "\$p")"; : > "\$p"; echo "placeholder \$v"; } ;; esac
done
mkdir -p "$L/ablit"
( launch_cluster ); echo "LAUNCH_RC=\$?"
EOF
env -i HOME="$D" USER="$(id -un)" PATH="/usr/bin:/bin" bash "$D/drive.sh" > "$D/drive.log" 2>&1
grep -q "VALIDATE_RC=0" "$D/drive.log"; ck $? "validate_numeric_config accepts GLM53_NVME_SPILL=1 with the bundle files + a writable dir"
grep -q "LAUNCH_RC=0" "$D/drive.log"; ck $? "launch_cluster ran to the end against the fakes ($(grep -c . "$D/drive.log") log lines)"
python3 - "$D" "$L/.env" > "$D/trace.log" 2>&1 <<'EOF'
import glob, os, re, sys
d, envf = sys.argv[1], sys.argv[2]
want = {}
for line in open(envf):
    if line.startswith("GLM53_NVME_SPILL"):
        k, v = line.rstrip("\n").split("=", 1)
        want[k] = v
recs = sorted(glob.glob(f"{d}/rec/*"))
head = worker = None
for r in recs:
    argv = open(r, "rb").read().split(b"\0")[:-1]
    argv = [a.decode() for a in argv]
    if argv[0] == "docker" and argv[1:3] == ["run", "-d"]:
        head = argv
    if argv[0] == "ssh" and any("docker run -d --name" in a for a in argv):
        worker = " ".join(argv)
ok = True
if head is None or worker is None:
    print("MISSING head docker run" if head is None else "", "MISSING worker docker run" if worker is None else "")
    sys.exit(1)
hv = {}
for i, a in enumerate(head):
    if a == "-e" and i + 1 < len(head):
        k, _, v = head[i + 1].partition("=")
        hv[k] = v
for k, v in sorted(want.items()):
    h = hv.get(k)
    w = re.search(r"-e %s='([^']*)'" % re.escape(k), worker)
    w = w.group(1) if w else None
    good = h == v and w == v
    ok &= good
    print(f"{'ok ' if good else 'BAD'} {k}: head={h!r} worker={w!r} want={v!r}")
spill = want["GLM53_NVME_SPILL_DIR"]
hm = any(head[i] == "-v" and head[i + 1] == f"{spill}:{spill}" for i in range(len(head) - 1))
wm = f"-v '{spill}:{spill}'" in worker
print(f"{'ok ' if hm else 'BAD'} head mounts {spill} rw; {'ok ' if wm else 'BAD'} worker mounts it rw")
ok &= hm and wm
wa = re.search(r"PYTORCH_CUDA_ALLOC_CONF=(\S+)", worker)
print(f"both A/B arms keep the kit allocator: head PYTORCH_CUDA_ALLOC_CONF={hv.get('PYTORCH_CUDA_ALLOC_CONF')!r}; "
      f"worker {wa.group(1) if wa else None!r}")
sys.exit(0 if ok else 1)
EOF
ck $? "every GLM53_NVME_SPILL* value reaches both ranks' docker run (+ the rw mount): $(grep -c '^ok ' "$D/trace.log") ok, $(grep -c '^BAD' "$D/trace.log") bad"
for s in "$L/.glm53-exl3-head.inner.sh" "$L/.glm53-exl3-worker.inner.sh"; do
  grep -q '"kv_connector":"Glm53NvmeSpillConnector"' "$s"; ck $? "$(basename "$s") passes --kv-transfer-config for GLM53_NVME_SPILL=1"
done
# GLM53_NVME_SPILL unset: no connector arg is added by the inner scripts and validation stays stock
cat > "$D/drive_off.sh" <<EOF
cd "$L"; source ./start_lib.sh; set +e; validate_numeric_config; echo "OFF_VALIDATE_RC=\$? GLM53_NVME_SPILL=[\${GLM53_NVME_SPILL-unset}]"
EOF
env -i HOME="$D" USER="$(id -un)" PATH="$D/bin:/usr/bin:/bin" GLM53_NVME_SPILL= bash "$D/drive_off.sh" > "$D/off.log" 2>&1
grep -q "OFF_VALIDATE_RC=0 GLM53_NVME_SPILL=\[\]" "$D/off.log"; ck $? "validate_numeric_config with GLM53_NVME_SPILL empty (caller override): $(grep OFF_ "$D/off.log")"
cat > "$D/drive_bad.sh" <<EOF
cd "$L"; source ./start_lib.sh; set +e; ( validate_numeric_config ) >/dev/null 2>&1; echo "BAD_VALIDATE_RC=\$?"
EOF
env -i HOME="$D" USER="$(id -un)" PATH="$D/bin:/usr/bin:/bin" GLM53_NVME_SPILL_DIR=/nonexistent/spill bash "$D/drive_bad.sh" > "$D/bad.log" 2>&1
grep -q "BAD_VALIDATE_RC=[1-9]" "$D/bad.log"; ck $? "validate_numeric_config refuses GLM53_NVME_SPILL=1 with a missing dir: $(grep BAD_ "$D/bad.log")"
grep -q 'if \[ "${GLM53_NVME_SPILL:-}" = "1" \]; then  # \[nvme-spill\]' "$L/.glm53-exl3-head.inner.sh"; ck $? "inner script guards the connector arg on GLM53_NVME_SPILL=1 (container env)"
diff -u "$KS" "$D/start.sh" > "$D/start.sh.nvmespill.patch"
echo "SUMMARY $npass/$((npass+nfail)) passed  (trace: $D/trace.log, drive log: $D/drive.log, patch: $D/start.sh.nvmespill.patch)"
[ $nfail = 0 ]
