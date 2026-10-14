#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Add the NVMe spill tier switch (GLM53_NVME_SPILL) to the kit start.sh of the GLM-5.3-Flash EXL3
launcher (measured on the r16z8p kit, launcher/start.sh sha256 ae9b8f86...).

usage: apply_start_sh.py <in start.sh> <out start.sh>     (in == out is allowed)

Edits (each anchor must occur exactly as often as listed, else ABORT and nothing is written; the result must pass
bash -n; replaying the edits backwards must give the input byte for byte):
  V  validation next to the other bundle switches: GLM53_NVME_SPILL 0|1; for 1 the overlay files, the bundle
     registration and an absolute, writable GLM53_NVME_SPILL_DIR on the head node
  W  worker serve_env names (appended to the forwarded-name loop, whatever its last entry is)
  H  head -e for the same names
     names = GLM53_NVME_SPILL + GLM53_NVME_SPILL_<k> for EVERY k of the connector's KNOBS tuple
     (code/glm53_nvme_spill_connector.py, read with ast: a knob added there is forwarded here automatically)
  M  GLM53_NVME_SPILL_DIR bind-mounted rw at the same path in both containers (worker dir checked over ssh first)
  A  --kv-transfer-config naming Glm53NvmeSpillConnector in BOTH inner scripts (identical on both ranks)
Idempotent: an input that already carries exactly these edits (the backwards replay gives a text that re-applies to
it byte for byte) is written unchanged ("already applied"); any other input holding an [nvme-spill] marker aborts.
All of it is inert unless GLM53_NVME_SPILL=1.
"""
import ast
import os
import subprocess
import sys

CONNECTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glm53_nvme_spill_connector.py")


def connector_knobs(path: str = CONNECTOR) -> list[str]:
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KNOBS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit(f"ABORT: no KNOBS tuple in {path}")


VARS = ["GLM53_NVME_SPILL"] + [f"GLM53_NVME_SPILL_{k}" for k in connector_knobs()]

V_OLD = '''    _glm53_validate_spinwait_ms || return
    _glm53_validate_bool_flag GLM53_APC_NO_STORE'''
# The comment line below is inserted into start.sh verbatim. It names the design doc of our internal fork (in this
# repository: docs/design.md, docs/operations.md); it is kept unchanged so that the generated start.sh stays byte-identical
# to the one running in production (and a re-run on that start.sh is recognised as "already applied").
V_NEW = '''    # [nvme-spill] resume points of long conversations on each node's local NVMe (docs/NVME_SPILL.md of tf-exl3-fork).
    # Unset/empty/0 = stock (no KV connector, nothing installed); 1 = the bundle overlay patch_nvme_spill.py on both ranks
    # + --kv-transfer-config in both inner scripts + GLM53_NVME_SPILL_DIR bind-mounted rw into both containers.
    if [ -n "${GLM53_NVME_SPILL:-}" ]; then
        _glm53_validate_bool_flag GLM53_NVME_SPILL "$GLM53_NVME_SPILL" || return
    fi
    if [ "${GLM53_NVME_SPILL:-}" = "1" ]; then
        if [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/patch_nvme_spill.py" ] \\
           || [ ! -s "$TF_BUNDLE_DIR_HOST/overlay/glm53_nvme_spill_connector.py" ]; then
            echo "GLM53_NVME_SPILL=1 requires $TF_BUNDLE_DIR_HOST/overlay/{patch_nvme_spill.py,glm53_nvme_spill_connector.py}" >&2
            return 2
        fi
        if ! grep -qF '("patch_nvme_spill.py", "GLM53_NVME_SPILL")' "$TF_BUNDLE_PATCH_HOST" 2>/dev/null; then
            echo "GLM53_NVME_SPILL=1 requires $TF_BUNDLE_PATCH_HOST to run patch_nvme_spill.py (the nvme-spill patch_tf_bundle.py)" >&2
            return 2
        fi
        case "${GLM53_NVME_SPILL_DIR:-}" in
            /?*) ;;
            *) echo "GLM53_NVME_SPILL=1 requires GLM53_NVME_SPILL_DIR (absolute path on the local NVMe of each node)" >&2; return 2 ;;
        esac
        if [ ! -d "$GLM53_NVME_SPILL_DIR" ] || [ ! -w "$GLM53_NVME_SPILL_DIR" ]; then
            echo "GLM53_NVME_SPILL_DIR=$GLM53_NVME_SPILL_DIR is not a writable directory on the head node" >&2
            return 2
        fi
    fi
''' + V_OLD

W_OLD = '''; do  # [tf-exl3-fork]
        serve_env+=" -e $v='${!v:-}'"'''
W_NEW = (" \\\n            " + " ".join(VARS) + '''; do  # [tf-exl3-fork] [nvme-spill]
        serve_env+=" -e $v='${!v:-}'"''')

H_OLD = '''        -e GLM53_KDA_STRIDED_QKV="${GLM53_KDA_STRIDED_QKV:-}" \\
'''
H_NEW = H_OLD + "".join(f'        -e {v}="${{{v}:-}}" \\\n' for v in VARS)

M1_OLD = '''    log "starting worker on ${WORKER_SSH} (NCCL if=${WORKER_CX7_IF} hca=${WORKER_CX7_IB}) ..."
'''
M1_NEW = '''    # [nvme-spill] each rank writes its own resume-point files to its node's local NVMe (same path on both nodes)
    worker_spill_mount=""
    head_spill_mount=()
    if [ "${GLM53_NVME_SPILL:-}" = "1" ]; then
        if ! worker_ssh "test -d '$GLM53_NVME_SPILL_DIR' && test -w '$GLM53_NVME_SPILL_DIR'"; then
            die "GLM53_NVME_SPILL_DIR=$GLM53_NVME_SPILL_DIR is not a writable directory on the worker node"
        fi
        worker_spill_mount="-v '$GLM53_NVME_SPILL_DIR:$GLM53_NVME_SPILL_DIR'"
        head_spill_mount=(-v "$GLM53_NVME_SPILL_DIR:$GLM53_NVME_SPILL_DIR")
    fi
''' + M1_OLD

M2_OLD = '''        ${worker_preload} \\
        ${worker_nccl} \\'''
M2_NEW = '''        ${worker_preload} \\
        ${worker_spill_mount} \\
        ${worker_nccl} \\'''

M3_OLD = '''    VLLM_API_KEY="$VLLM_API_KEY" docker run -d --name "$CONTAINER_HEAD" \\
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \\
        --device /dev/infiniband --cap-add IPC_LOCK \\
        --ulimit memlock=-1 --ulimit stack=67108864 \\
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \\
        -v "$CACHE_ROOT:/root/.cache/vllm" \\
        -v "$TRITON_HOST_CACHE:/root/.triton/cache" \\
        -v "$TILELANG_HOST_CACHE:/root/.tilelang/cache" \\
'''
M3_NEW = M3_OLD + '''        "${head_spill_mount[@]}" \\
'''

A_OLD = '''[ -n "${PREFIX_MATCH_UNIT:-}" ] && ARGS+=(--prefix-match-unit "${PREFIX_MATCH_UNIT}")
'''
A_NEW = A_OLD + '''if [ "${GLM53_NVME_SPILL:-}" = "1" ]; then  # [nvme-spill] identical on both ranks (part of the config hash)
    ARGS+=(--kv-transfer-config '{"kv_connector":"Glm53NvmeSpillConnector","kv_connector_module_path":"vllm.distributed.kv_transfer.kv_connector.v1.glm53_nvme_spill_connector","kv_role":"kv_both"}')
fi
'''

EDITS = [("V", V_OLD, V_NEW, 1), ("W", W_OLD, W_NEW, 1), ("H", H_OLD, H_NEW, 1), ("M1", M1_OLD, M1_NEW, 1),
         ("M2", M2_OLD, M2_NEW, 1), ("M3", M3_OLD, M3_NEW, 1), ("A", A_OLD, A_NEW, 2)]


def unapply(s: str) -> str:
    back = s
    for name, old, new, n in reversed(EDITS):
        back = back.replace(new, old)
    return back


def apply(s: str) -> str:
    if "[nvme-spill]" in s:
        base = unapply(s)
        if "[nvme-spill]" not in base and base != s:
            try:
                again = apply(base)
            except SystemExit:
                again = None
            if again == s:
                return s        # already applied, exactly these edits
        sys.exit("ABORT: [nvme-spill] markers present but not exactly this tool's edits (drifted/partial/older)")
    out = s
    for name, old, new, n in EDITS:
        c = out.count(old)
        if c != n:
            sys.exit(f"ABORT: anchor {name} found {c}x (need {n})")
        out = out.replace(old, new)
    r = subprocess.run(["bash", "-n", "/dev/stdin"], input=out.encode(), capture_output=True)
    if r.returncode:
        sys.exit(f"ABORT: bash -n: {r.stderr.decode()}")
    if unapply(out) != s:
        sys.exit("ABORT: reverse replay differs from the input")
    return out


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = open(sys.argv[1]).read()
    out = apply(src)
    state = "already applied (unchanged)" if out == src else f"{len(EDITS)} edits applied"
    tmp = sys.argv[2] + ".nvmespill-tmp"
    with open(tmp, "w") as f:
        f.write(out)
    if os.path.exists(sys.argv[1]):
        os.chmod(tmp, os.stat(sys.argv[1]).st_mode & 0o7777)
    os.replace(tmp, sys.argv[2])
    print(f"nvme-spill: {state} ({out.count('[nvme-spill]')} markers, {len(VARS)} forwarded names)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
