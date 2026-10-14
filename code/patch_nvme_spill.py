#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Install the GLM-5.3 NVMe spill-tier KV connector (docs/design.md, docs/operations.md).

What it does (GLM53_NVME_SPILL=1)
---------------------------------
1. Copies ``glm53_nvme_spill_connector.py`` (next to this file) to
   ``vllm/distributed/kv_transfer/kv_connector/v1/glm53_nvme_spill_connector.py`` in site-packages. The engine only
   loads it when started with ``--kv-transfer-config`` naming ``Glm53NvmeSpillConnector`` (start.sh does that only for
   GLM53_NVME_SPILL=1), so installing alone changes nothing.
2. ``vllm/v1/core/sched/scheduler.py``: right after the scheduler binds the GPU block pool to the KV connector, also
   hands the connector the KVCacheManager (the connector asks the engine's own coordinator which blocks make up a
   finished request's deepest resume point). A connector without the hook method is untouched (``hasattr``).
3. ``vllm/config/vllm.py`` ``VllmConfig._verify_kv_transfer_compat`` (run by ``__post_init__`` on every rank): exempts
   ONLY ``kv_connector == "Glm53NvmeSpillConnector"`` from the "any KV connector + PYTORCH_CUDA_ALLOC_CONF=
   expandable_segments:True -> ValueError" rule, so both A/B arms keep the kit's allocator setting (start.sh passes
   expandable_segments:True to both ranks). The rule exists because connectors that register KV memory with a NIC or
   another process (NIXL ibv_reg_mr, Mooncake, CUDA IPC) keep physical-page registrations that a VMM remap makes
   stale. This connector registers nothing: it reads/writes the KV tensors only through torch views inside the
   worker process (every access is a cudaMemcpyAsync on the tensor's virtual address, resolved through the current
   mapping; a live caching-allocator block is never remapped), and its staging buffers are pinned HOST memory from
   torch's host allocator, which expandable_segments does not govern. Every other connector name still raises.

States (both targets): pristine -> patched; patched (marker + exact patched text / identical module) -> no-op;
anything else (drifted anchor, partial marker, a foreign file at the module path) -> SystemExit. Atomic replace, pyc
cleared. ``GLM53_NVME_SPILL`` unset/empty or ``0`` -> nothing is touched; any value other than 0/1 -> SystemExit.

Env overrides for tests: GLM53_SITEPKG, GLM53_SCHEDULER_PY, GLM53_VLLM_CONFIG_PY, GLM53_NVME_SPILL_MODULE_DST.
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

ENV = "GLM53_NVME_SPILL"
MARK = "# [glm53-nvme-spill]"
HERE = Path(__file__).resolve().parent
SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
SCHED = Path(os.environ.get("GLM53_SCHEDULER_PY", SITEPKG / "vllm/v1/core/sched/scheduler.py"))
VCFG = Path(os.environ.get("GLM53_VLLM_CONFIG_PY", SITEPKG / "vllm/config/vllm.py"))
MOD_SRC = HERE / "glm53_nvme_spill_connector.py"
MOD_DST = Path(os.environ.get(
    "GLM53_NVME_SPILL_MODULE_DST",
    SITEPKG / "vllm/distributed/kv_transfer/kv_connector/v1/glm53_nvme_spill_connector.py"))
MOD_TAG = "[glm53-nvme-spill]"

ANCHOR = """        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
"""
PATCHED = """        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
            if hasattr(self.connector, "glm53_bind_kv_cache_manager"):  # [glm53-nvme-spill]
                self.connector.glm53_bind_kv_cache_manager(self.kv_cache_manager)
"""


CFG_ANCHOR = """        if self.model_config is not None and (self.model_config.enable_cumem_allocator):
            return

        raise ValueError(
            f"KV connector {self.kv_transfer_config.kv_connector} is "
            "incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
"""
CFG_PATCHED = """        if self.model_config is not None and (self.model_config.enable_cumem_allocator):
            return
        # [glm53-nvme-spill] this connector registers no KV memory with a NIC/driver/other process (only torch views
        # inside the worker; pinned host staging): a VMM remap can not invalidate anything it holds.
        if self.kv_transfer_config.kv_connector == "Glm53NvmeSpillConnector":
            return

        raise ValueError(
            f"KV connector {self.kv_transfer_config.kv_connector} is "
            "incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
"""


def _clear_pyc(path: Path) -> None:
    pyc = path.parent / "__pycache__"
    if pyc.is_dir():
        for f in pyc.glob(path.stem + ".*.pyc"):
            f.unlink()


def _atomic_write(path: Path, text: str, mode: int | None = None) -> None:
    tmp = path.with_name(path.name + ".glm53tmp")
    if mode is None:
        mode = stat.S_IMODE(path.stat().st_mode)
    tmp.write_text(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    _clear_pyc(path)


def apply_scheduler(src: str) -> tuple[str, str]:
    n_mark = src.count(MARK)
    if n_mark:
        if n_mark == 1 and src.count(PATCHED) == 1:
            return src, "present"
        raise SystemExit(f"{MARK} scheduler.py: marker present but patched text drifted (marks={n_mark})")
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"{MARK} scheduler.py: anchor found {n} times (need exactly 1) -- refusing")
    return src.replace(ANCHOR, PATCHED), "patched"


def apply_config(src: str) -> tuple[str, str]:
    n_mark = src.count(MARK)
    if n_mark:
        if n_mark == 1 and src.count(CFG_PATCHED) == 1:
            return src, "present"
        raise SystemExit(f"{MARK} config/vllm.py: marker present but patched text drifted (marks={n_mark})")
    if src.count("def _verify_kv_transfer_compat(self)") != 1:
        raise SystemExit(f"{MARK} config/vllm.py: _verify_kv_transfer_compat not found exactly once -- refusing")
    n = src.count(CFG_ANCHOR)
    if n != 1:
        raise SystemExit(f"{MARK} config/vllm.py: anchor found {n} times (need exactly 1) -- refusing")
    return src.replace(CFG_ANCHOR, CFG_PATCHED), "patched"


def install_module() -> str:
    text = MOD_SRC.read_text()
    if MOD_TAG not in text:
        raise SystemExit(f"{MARK} {MOD_SRC} does not look like the spill connector")
    if MOD_DST.exists():
        cur = MOD_DST.read_text()
        if cur == text:
            return "present"
        if MOD_TAG not in cur:
            raise SystemExit(f"{MARK} {MOD_DST} exists and is not ours -- refusing")
        _atomic_write(MOD_DST, text)
        return "replaced"
    if not MOD_DST.parent.is_dir():
        raise SystemExit(f"{MARK} {MOD_DST.parent} missing (not the expected vLLM tree)")
    _atomic_write(MOD_DST, text, mode=0o644)
    return "installed"


def main(argv: list[str] | None = None) -> int:
    val = os.environ.get(ENV, "").strip()
    if val in ("", "0"):
        print(f"{MARK} {ENV}={val or '(unset)'} -> nothing to do")
        return 0
    if val != "1":
        raise SystemExit(f"{MARK} {ENV} must be 0 or 1, got {val!r}")
    # refuse before touching anything when the module path holds a foreign file
    if MOD_DST.exists() and MOD_TAG not in MOD_DST.read_text():
        raise SystemExit(f"{MARK} {MOD_DST} exists and is not ours -- refusing")
    # check every target before writing any (a drifted tree must not end half patched)
    src = SCHED.read_text()
    new, state = apply_scheduler(src)
    csrc = VCFG.read_text()
    cnew, cstate = apply_config(csrc)
    if state == "patched":
        _atomic_write(SCHED, new)
    if cstate == "patched":
        _atomic_write(VCFG, cnew)
    mstate = install_module()
    sha = hashlib.sha256(MOD_DST.read_bytes()).hexdigest()[:12]
    print(f"{MARK} scheduler.py {state}; config/vllm.py {cstate}; module {mstate} at {MOD_DST} (sha256 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
