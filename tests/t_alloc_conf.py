# SPDX-License-Identifier: AGPL-3.0-only
"""B1 regression (review finding B1): the kit start.sh passes PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to both
ranks; VllmConfig.__post_init__ -> _verify_kv_transfer_compat rejected ANY kv connector with it.

Runs in the production image (CPU) against a rig work tree (PYTHONPATH): builds the rig's production VllmConfig WITH
kv_transfer_config in the constructor (so __post_init__ runs the check, as at production start) under both allocator
settings, then checks that another connector name is still rejected by the same (patched) function.
usage: PYTHONPATH=<work tree> python3 t_alloc_conf.py <label>   -> prints ALLOC <label> ... lines
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rig"))
sys.path.insert(0, HERE)
os.environ["NVMESPILL_ON"] = "1"
os.environ.setdefault("GLM53_NVME_SPILL_DIR", "/spill/t_alloc")
label = sys.argv[1] if len(sys.argv) > 1 else "tree"
from sim import configs  # noqa: E402

res = {}
for val in ("", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = val
    try:
        vc = configs.build_vllm_config()
        res[val] = "accepted"
        print(f"ALLOC {label} PYTORCH_CUDA_ALLOC_CONF={val!r} -> VllmConfig(kv_transfer_config=spill) accepted "
              f"(kv_connector={vc.kv_transfer_config.kv_connector})")
    except ValueError as e:
        res[val] = "ValueError"
        msg = str(e)
        i = msg.find("KV connector")
        print(f"ALLOC {label} PYTORCH_CUDA_ALLOC_CONF={val!r} -> ValueError: {msg[i:i + 130] if i >= 0 else msg[:200]}")
if res.get("expandable_segments:True") == "accepted":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    vc.kv_transfer_config.kv_connector = "NixlConnector"
    try:
        vc._verify_kv_transfer_compat()
        print(f"ALLOC {label} other connector (NixlConnector) + expandable -> accepted (WRONG: exemption too wide)")
    except ValueError as e:
        print(f"ALLOC {label} other connector (NixlConnector) + expandable -> still ValueError (exemption is narrow)")
