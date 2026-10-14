# SPDX-License-Identifier: AGPL-3.0-only
"""GLM-5.3-Flash NVMe spill tier: a KV connector that puts a long conversation's resume point on local NVMe and
brings it back into the GPU prefix cache before the conversation's next turn is admitted.

Why (docs/design.md, "Problem"): a deep prefix-cache hit on this hybrid model
needs ALL of {the MLA+indexer blocks up to the boundary, the boundary state of the four mamba (KDA) groups, the DFlash
drafter window}. Under a saturated pool the boundary states go first and the next turn re-reads 0.5-0.8M tokens
(5-15 min during which every other stream stalls).

What it stores: one *resume point* (RP) = exactly the blocks the engine's own coordinator returns for the finished
request (``HybridKVCacheCoordinator.find_longest_cache_hit(request.block_hashes, num_tokens-1)``): MLA+indexer blocks
0..N, the mamba block of each group at N, the drafter window blocks at N, each with every prefix-cache key the block
carries. Blocks are files named by their primary key (content addressed), so the turns of one conversation share
their MLA blocks on disk and a later turn writes only what is new (its mamba/drafter state + new MLA blocks).
An RP is usable only when every one of its blocks is on disk on every rank ("one missing -> not used").

How it comes back: when a waiting request's local hit is shallower than a saved RP that its prompt extends, the
scheduler-side connector pins the RP blocks still in GPU memory, allocates fresh blocks for the missing ones, has
each rank read them (O_DIRECT -> small pinned staging buffer -> GPU block rows, CRC-32 per 8 MiB checked), and on
success on every rank registers the original keys on the fresh blocks. The request is then re-polled and the
ordinary local lookup finds the deep hit (counted as a prefix-cache hit; mamba CoW/seed, DFlash boundary lookup,
retention all run their normal code). Bytes are copied, never transformed: the restored blocks hold exactly the bytes
the evicted blocks held. Any failure (a rank, a CRC, a short file, no free blocks) leaves the request on the normal
prefill path; failed blocks never get a key.

Consensus over TP ranks: every decision is made once, by the scheduler (EngineCore) process, and shipped to every
rank in the connector metadata; each rank reports per job {ok, fail} counts that the executor sums over ranks
(``SpillWorkerMeta.aggregate``); a job counts only when ok == world size.

When it writes: right after a request's prompt is computed (the step after its last prompt chunk was scheduled; the
boundary state of the last chunk is released at prefill end and may be evicted before the request finishes), and at
request finish for requests not taken then, if the RP is >= MIN_TOKENS long and the pool usage is >= HIGHWATER (0 =
always); below the high-water mark the RP is remembered and written by a sweep once usage crosses it (if its blocks
are still cached). The RP blocks to copy are pinned (ref_cnt+1) until every rank has them off the GPU (or the save
deadline passes), so they can not be reallocated while being copied. A copy planned inside build_meta is shipped with
the NEXT step's metadata, so the worker's "all queued device work" event is recorded after the forward that writes
any block keyed in the planning step.

Fault containment: every job a rank receives is answered (the I/O threads catch everything per job and never die);
a save unanswered by every rank for SAVE_DEADLINE_S releases its pins; a load unanswered for LOAD_DEADLINE_S sends the
request down the normal prefill path while its fresh blocks stay quarantined until every rank answers (a late write
must not land in a block someone else owns); any deadline miss, a worker-side unexpected exception or BREAKER_FAILS
consecutive failed jobs open a latched breaker: no new save or restore on any rank until restart. Pinned blocks are
capped (MAX_PINNED_BLOCKS). A periodic "[glm53-nvme-spill] stats" line (scheduler) and "rank <r> io:" line (each
rank) carry the counters the production A/B abort criteria read (docs/operations.md).

Default OFF: nothing happens unless the engine is started with ``--kv-transfer-config`` naming this connector
(start.sh does that only for GLM53_NVME_SPILL=1).

Credits / provenance: the design (one file set per resume point on local NVMe per rank, a scheduler-side decision
shipped to all ranks with all-rank agreement on every read, header-padded files written via O_DIRECT to ``.partial``,
fsynced and renamed, a CRC-32 per 8 MiB checked on every read, small fixed staging buffers between NVMe and GPU) follows
TensorFold's CUDA spill tier (patch 0088, ``tensorfold/cuda/spill.py``, Apache-2.0, written by wojo (Robert
Wojciechowski); its multi-rank design credits patch 0250 of jayleaton/glm53-tensorfold-spark by Jay Leaton, Apache-2.0;
file conventions from TensorFold's MLX spill PR #68 by gilby and issue #155 by raymondkpwong). No code is copied
verbatim; this module is a re-implementation for vLLM's KV connector API. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import queue
import struct
import threading
import time
import zlib
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

TAG = "[glm53-nvme-spill]"
FORMAT_VERSION = 1
MAGIC = b"GLM53SPL"
HDR = 4096                      # header bytes; data starts here (O_DIRECT alignment)
ALIGN = 4096
CRC_BLOCK = 8 << 20             # bytes one CRC-32 covers
SUBDIR = "glm53-nvme-spill"
# 2026-10-14: per-request TTL (DSH contract v1.1, kv_transfer_params.glm53_spill) + reuse across restarts (index)
CONNECTOR_VERSION = "2026-10-14.ttl-persist.1"
INDEX_NAME = "index.json"
INDEX_MAGIC = "glm53-nvme-spill-index"
INDEX_VERSION = 1
TTL_MAX_S = 7 * 86400           # contract: ttl_s > 7 days is invalid (-> server default)
SESSION_MAX = 64                # contract: session id is an opaque string of at most 64 characters
EPOCH_GRACE_S = 1800.0          # an older-epoch RP is deleted only after this long unused (a late request of that epoch)


# Every operator knob (env GLM53_NVME_SPILL_<name>); code/apply_start_sh.py forwards exactly these to both
# ranks (tests/test_start_sh.sh checks the two lists agree). GLM53_NVME_SPILL_RANK is a test-only override.
KNOBS = ("DIR", "GIB", "MIN_TOKENS", "MIN_GAIN", "HIGHWATER", "MIN_FREE_GIB", "PIN_TIMEOUT_S", "RESERVE_BLOCKS",
         "SPACE_WAIT_S", "STAGE_MIB", "DIRECT", "MAX_WRITE_GIB_PER_H", "MAX_RESTORE_FAILURES",
         "MAX_RESTORES_PER_REQUEST", "MAX_STATE_PINS", "SAVE_POLICY", "LOG", "SAVE_DEADLINE_S", "LOAD_DEADLINE_S",
         "BREAKER_FAILS", "MAX_PINNED_BLOCKS", "LOG_EVERY_S", "KEEP_PER_CHAIN",
         # 2026-10-14 (TTL + restart reuse)
         "PERSIST", "HASH_SEED", "INDEX_EVERY_S", "ADOPT_DEADLINE_S", "TTL_SWEEP_S", "TTL_SAVE_POLICY",
         "TTL_MIN_TOKENS")


def _up(x: int, a: int = ALIGN) -> int:
    return -(-x // a) * a


# ----------------------------------------------------------------------------------------------------------------
# configuration (env; read identically by the scheduler and every worker)
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class SpillConfig:
    root: str
    cap_bytes: int
    min_tokens: int = 65536
    min_gain: int = 16384
    highwater: float = 0.0
    min_free_bytes: int = 50 << 30
    pin_timeout_s: float = 5.0          # restored blocks stay pinned this long waiting for the request's admission
    reserve_blocks: int = 8
    space_wait_s: float = 20.0          # a restore that does not fit yet keeps the request waiting this long
    stage_bytes: int = 32 << 20
    direct: bool = True
    max_write_bytes_per_h: int = 0      # 0 = unlimited
    max_restore_failures: int = 2       # failed restores per request before it only takes the normal path
    max_restores_per_request: int = 16  # restores per request (re-restore after its blocks were evicted while waiting)
    max_state_pins: int = 64            # recurrent/drafter state blocks kept pinned for restored, not yet admitted requests
    save_policy: str = "repeat"         # repeat: only conversations seen before (a later turn) | all
    log: bool = True
    # fault containment (a rank that stops answering or keeps failing must never hold KV blocks forever)
    save_deadline_s: float = 120.0      # a save job unanswered by every rank this long: pins released, breaker opens
    load_deadline_s: float = 30.0       # a load job unanswered this long: request -> normal prefill, blocks quarantined
    breaker_fails: int = 6              # consecutive failed jobs (any rank) that open the breaker
    max_pinned_blocks: int = 0          # save-pinned blocks only the connector holds, gate for new saves (0 = half pool)
    log_every_s: float = 60.0           # periodic stats line (0 = off)
    # superseded resume points: when a new RP of a conversation is saved (ok on every rank), the older RPs of the same
    # block-hash chain beyond the newest KEEP_PER_CHAIN are deleted (0 = never, LRU only). 2 keeps the one before the
    # newest, which an edited / regenerated last turn branches from.
    keep_per_chain: int = 2
    # reuse across restarts (2026-10-14): persist the resume-point index and re-adopt the files at the next start when
    # the identity signature (model, KV layout, site-packages sources, env, block-hash seed, ranks) matches exactly.
    # The dataclass default is off (unit tests build configs directly); from_env defaults it on.
    persist: bool = False
    hash_seed: str = "5353"             # PYTHONHASHSEED the scheduler process sets (if unset) before vLLM seeds NONE_HASH
    index_every_s: float = 5.0          # write the index at most this often while it changed
    adopt_deadline_s: float = 60.0      # rank reports later than this (from the first step): wipe instead of adopting
    # per-request TTL (kv_transfer_params.glm53_spill.ttl_s, DSH contract v1.1)
    ttl_sweep_s: float = 30.0           # how often expired resume points are deleted
    ttl_save_policy: str = "all"        # requests with an explicit ttl_s > 0: all = also their first turn | repeat
    ttl_min_tokens: int = 0             # MIN_TOKENS for requests with an explicit ttl_s > 0 (0 = MIN_TOKENS)

    def index_path(self) -> str:
        return os.path.join(self.root, SUBDIR, INDEX_NAME)

    @classmethod
    def from_env(cls, extra: dict[str, Any] | None = None) -> "SpillConfig":
        extra = dict(extra or {})

        def get(name: str, default):
            key = name.lower()
            if key in extra:
                return extra[key]
            raw = os.environ.get("GLM53_NVME_SPILL_" + name)
            # start.sh forwards every knob, unset ones as an empty string: empty = default
            return default if raw is None or raw.strip() == "" else raw.strip()

        root = str(get("DIR", "") or "")
        if not root or not os.path.isabs(root) or os.path.realpath(root) == "/":
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_DIR must be an absolute directory, got {root!r}")
        cfg = cls(
            root=root,
            cap_bytes=int(float(get("GIB", 64)) * (1 << 30)),
            min_tokens=int(get("MIN_TOKENS", 65536)),
            min_gain=int(get("MIN_GAIN", 16384)),
            highwater=float(get("HIGHWATER", 0.0)),
            min_free_bytes=int(float(get("MIN_FREE_GIB", 50)) * (1 << 30)),
            pin_timeout_s=float(get("PIN_TIMEOUT_S", 5)),
            reserve_blocks=int(get("RESERVE_BLOCKS", 8)),
            space_wait_s=float(get("SPACE_WAIT_S", 20)),
            stage_bytes=int(float(get("STAGE_MIB", 32)) * (1 << 20)),
            direct=str(get("DIRECT", "1")) != "0",
            max_write_bytes_per_h=int(float(get("MAX_WRITE_GIB_PER_H", 0)) * (1 << 30)),
            max_restore_failures=int(get("MAX_RESTORE_FAILURES", 2)),
            max_restores_per_request=int(get("MAX_RESTORES_PER_REQUEST", 16)),
            max_state_pins=int(get("MAX_STATE_PINS", 64)),
            save_policy=str(get("SAVE_POLICY", "repeat")),
            log=str(get("LOG", "1")) != "0",
            save_deadline_s=float(get("SAVE_DEADLINE_S", 120)),
            load_deadline_s=float(get("LOAD_DEADLINE_S", 30)),
            breaker_fails=int(get("BREAKER_FAILS", 6)),
            max_pinned_blocks=int(get("MAX_PINNED_BLOCKS", 0)),
            log_every_s=float(get("LOG_EVERY_S", 60)),
            keep_per_chain=int(get("KEEP_PER_CHAIN", 2)),
            persist=str(get("PERSIST", "1")) != "0",
            hash_seed=str(get("HASH_SEED", "5353")),
            index_every_s=float(get("INDEX_EVERY_S", 5)),
            adopt_deadline_s=float(get("ADOPT_DEADLINE_S", 60)),
            ttl_sweep_s=float(get("TTL_SWEEP_S", 30)),
            ttl_save_policy=str(get("TTL_SAVE_POLICY", "all")),
            ttl_min_tokens=int(get("TTL_MIN_TOKENS", 0)),
        )
        if cfg.cap_bytes <= 0:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_GIB must be > 0")
        if not 0.0 <= cfg.highwater <= 1.0:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_HIGHWATER must be in [0, 1]")
        if cfg.save_policy not in ("repeat", "all"):
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_SAVE_POLICY must be repeat or all")
        if cfg.stage_bytes % ALIGN:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_STAGE_MIB must give a multiple of 4096 bytes")
        if cfg.save_deadline_s <= 0 or cfg.load_deadline_s <= 0:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_SAVE_DEADLINE_S / LOAD_DEADLINE_S must be > 0")
        if cfg.breaker_fails < 1 or cfg.max_pinned_blocks < 0 or cfg.log_every_s < 0:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_BREAKER_FAILS >= 1, MAX_PINNED_BLOCKS >= 0, LOG_EVERY_S >= 0")
        if cfg.keep_per_chain < 0:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_KEEP_PER_CHAIN must be >= 0 (0 = no pruning)")
        if not (cfg.hash_seed.isdigit() and 0 <= int(cfg.hash_seed) <= 4294967295):
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_HASH_SEED must be an integer in [0, 4294967295] "
                             "(it becomes PYTHONHASHSEED)")
        if cfg.index_every_s <= 0 or cfg.adopt_deadline_s <= 0 or cfg.ttl_sweep_s <= 0:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_INDEX_EVERY_S / ADOPT_DEADLINE_S / TTL_SWEEP_S must be > 0")
        if cfg.ttl_save_policy not in ("repeat", "all"):
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_TTL_SAVE_POLICY must be repeat or all")
        if cfg.ttl_min_tokens < 0:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_TTL_MIN_TOKENS must be >= 0 (0 = MIN_TOKENS)")
        return cfg

    def rank_dir(self, rank: int) -> str:
        return os.path.join(self.root, SUBDIR, f"rank{rank}")


# ----------------------------------------------------------------------------------------------------------------
# block-row geometry: which bytes of which KV tensor make up "block b of group g"
# ----------------------------------------------------------------------------------------------------------------
def group_rows(kv_cache_config: "KVCacheConfig") -> tuple[list[int], list[list[tuple[int, int]]]]:
    """Return (tensor page bytes, per-group rows).

    Every KV tensor is addressed page by page: block id ``b`` of any layer sharing tensor ``t`` occupies bytes
    ``[b * page_t, (b + 1) * page_t)`` of that tensor (gpu_model_runner / v1/worker/gpu/attn_utils.py: mamba layers
    are ``raw[:nb*page].view(nb, page)``, padded/KV-first attention layers are built page-first). A group's block is
    therefore the page ``b`` of each tensor its layers live in. Rows are ``(tensor_index, page_bytes)`` in the
    group's layer order, de-duplicated."""
    num_blocks = kv_cache_config.num_blocks
    layer_to_tensor: dict[str, int] = {}
    pages: list[int] = []
    for t, kvt in enumerate(kv_cache_config.kv_cache_tensors):
        if getattr(kvt, "block_stride", 0):
            raise NotImplementedError(f"{TAG} packed KV tensors are not supported")
        if kvt.size % num_blocks:
            raise ValueError(f"{TAG} tensor {t} size {kvt.size} is not a multiple of {num_blocks} blocks")
        pages.append(kvt.size // num_blocks)
        for name in kvt.shared_by:
            layer_to_tensor[name] = t
    rows: list[list[tuple[int, int]]] = []
    for group in kv_cache_config.kv_cache_groups:
        seen: list[int] = []
        for name in group.layer_names:
            t = layer_to_tensor.get(name)
            if t is None:
                continue
            if t not in seen:
                seen.append(t)
        rows.append([(t, pages[t]) for t in seen])
    return pages, rows


def block_file_name(primary_key: bytes) -> str:
    return hashlib.sha256(bytes(primary_key)).hexdigest()[:40] + ".blk"


# ----------------------------------------------------------------------------------------------------------------
# per-request spill parameters (DSH contract v1.1): request.kv_transfer_params["glm53_spill"] = {ttl_s, session, epoch}
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpillParams:
    ttl_s: int | None = None        # None = server default (no TTL, LRU only); 0 = do not keep; else seconds (sliding)
    session: str | None = None
    epoch: int | None = None

    @property
    def explicit_keep(self) -> bool:
        return self.ttl_s is not None and self.ttl_s > 0


def parse_spill_params(raw: Any) -> SpillParams | None:
    """Defensive: any malformed field is ignored (server default for that field), never an exception. Integers must be
    real ints (not bool/float/str); ttl_s in [0, 7 days]; session a non-empty str of <= 64 chars; epoch >= 0."""
    try:
        if not isinstance(raw, dict):
            return None
        g = raw.get("glm53_spill")
        if not isinstance(g, dict):
            return None
        ttl, ses, ep = g.get("ttl_s"), g.get("session"), g.get("epoch")
        ttl = ttl if type(ttl) is int and 0 <= ttl <= TTL_MAX_S else None
        ses = ses if isinstance(ses, str) and 0 < len(ses) <= SESSION_MAX else None
        ep = ep if type(ep) is int and 0 <= ep < (1 << 53) else None
        if ttl is None and ses is None and ep is None:
            return None
        return SpillParams(ttl, ses, ep)
    except Exception:  # noqa: BLE001 - never fail a request over its spill hints
        return None


# ----------------------------------------------------------------------------------------------------------------
# identity signature for reuse across restarts: every input that decides the bytes or the keys of a block file
# ----------------------------------------------------------------------------------------------------------------
_ENV_PREFIXES = ("GLM53_", "EXL3_", "TF_", "DFLASH", "MTP_", "SPEC_", "KV_CACHE", "QUANTIZATION", "PREFIX_MATCH",
                 "LANGUAGE_MODEL_ONLY", "VLLM_", "PYTORCH_CUDA_ALLOC_CONF", "TORCH_CUDA_ARCH_LIST", "CUDA_VERSION",
                 "ENFORCE_EAGER", "MAX_MODEL_LEN")
_ENV_SKIP = ("KEY", "TOKEN", "SECRET", "PASS", "AUTH", "CRED", "COOKIE", "SESSION", "PRIVATE", "LOG", "PORT", "_IP",
             "HOST", "GLM53_NVME_SPILL")


def _h(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _env_digest() -> str:
    """Names + values of the env that selects patches / kernels / KV formats (no secrets, logging, ports, hosts, nor
    this connector's own knobs). Only a digest leaves the process."""
    items = sorted((k, v) for k, v in os.environ.items()
                   if k.startswith(_ENV_PREFIXES) and not any(s in k for s in _ENV_SKIP))
    return _h(items)


def _site_digest() -> str:
    """Every *.py (content) and shared object (name + size) of the site-packages dir holding the imported vllm: the
    image's vLLM plus every production overlay patch (they rewrite files there at container start), this connector
    included. ~4 s cold on the production image, run off the hot path (scheduler thread / rank writer thread)."""
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.origin:
        return "no-vllm"
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    h = hashlib.sha256()
    n = 0
    for dp, dn, fn in os.walk(root):
        dn.sort()
        if "__pycache__" in dp:
            continue
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root)
            try:
                if f.endswith(".py"):
                    with open(p, "rb") as fh:
                        h.update(rel.encode() + b"\0" + hashlib.sha256(fh.read()).digest())
                    n += 1
                elif f.endswith(".so") or ".so." in f:
                    h.update(f"{rel}:{os.stat(p).st_size}".encode())
            except OSError:
                h.update(f"{rel}:unreadable".encode())
    return f"{h.hexdigest()[:16]}/{n}"


def _dir_digest(path: str | None) -> str:
    """A model directory: every top-level file's name, size and mtime; small files (configs, tokenizer) by content."""
    if not path or not os.path.isdir(path):
        return f"name:{path}"
    items = []
    for f in sorted(os.listdir(path)):
        p = os.path.join(path, f)
        try:
            st = os.stat(p)
            if not os.path.isfile(p):
                continue
            if st.st_size <= (1 << 20):
                with open(p, "rb") as fh:
                    items.append((f, st.st_size, hashlib.sha256(fh.read()).hexdigest()[:16]))
            else:
                items.append((f, st.st_size, st.st_mtime_ns))
        except OSError:
            items.append((f, "unreadable"))
    return _h([os.path.realpath(path), items])


def _identity(vllm_config: Any, kv_cache_config: Any) -> dict[str, str]:
    """Identity parts shared by the scheduler and every rank (each part a short digest; a part that can not be read
    becomes an error string, which is stable only if the error is, i.e. a mismatch -> wipe, the safe side)."""
    import sys

    def part(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return f"err:{type(e).__name__}"

    def versions():
        import vllm  # noqa: F401
        return _h([sys.version, torch.__version__, getattr(vllm, "__version__", "?")])

    def config():
        mc = getattr(vllm_config, "model_config", None)
        cc = getattr(vllm_config, "cache_config", None)
        sc = getattr(vllm_config, "speculative_config", None)
        pc = getattr(vllm_config, "parallel_config", None)
        g = lambda o, *names: [str(getattr(o, n, None)) for n in names]  # noqa: E731
        return _h([g(mc, "model", "tokenizer", "dtype", "quantization", "revision", "hf_config_path", "hf_overrides"),
                   g(cc, "cache_dtype", "block_size", "mamba_cache_mode", "mamba_ssm_cache_dtype",
                     "prefix_caching_hash_algo", "prefix_match_unit"),
                   g(sc, "method", "model", "num_speculative_tokens"),
                   g(pc, "tensor_parallel_size", "pipeline_parallel_size", "world_size")])

    def spec_repr(spec):
        import dataclasses
        if dataclasses.is_dataclass(spec):
            return repr(spec)       # vLLM's KV specs: block size, heads, head size, dtype, mamba shapes/dtypes
        return f"{type(spec).__name__}{sorted((k, repr(v)) for k, v in vars(spec).items())}"

    def layout():
        nb = max(1, int(kv_cache_config.num_blocks))     # page bytes, not the pool size (see _signature)
        return _h([[(kvt.size // nb, list(kvt.shared_by)) for kvt in kv_cache_config.kv_cache_tensors],
                   [(spec_repr(g.kv_cache_spec), list(g.layer_names)) for g in kv_cache_config.kv_cache_groups]])

    def models():
        mc = getattr(vllm_config, "model_config", None)
        sc = getattr(vllm_config, "speculative_config", None)
        return _h([_dir_digest(getattr(mc, "model", None)), _dir_digest(getattr(sc, "model", None))])

    return {"version": f"{CONNECTOR_VERSION}/f{FORMAT_VERSION}/i{INDEX_VERSION}", "python": part(versions),
            "config": part(config), "layout": part(layout), "env": part(_env_digest), "site": part(_site_digest),
            "model": part(models)}


def _sig_of(parts: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# scheduler <-> worker messages
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class SpillMeta(KVConnectorMetadata):
    # job_id -> [(group, block_id, file name, write id)]  (the write id is in the file header; a load checks it)
    saves: dict[int, list[tuple]] = field(default_factory=dict)
    loads: dict[int, list[tuple]] = field(default_factory=dict)
    deletes: list[str] = field(default_factory=list)
    # start-up decision on reuse across restarts (scheduler): wipe every block file of every rank, or delete the files
    # one rank holds that the adopted index does not reference (orphans differ per rank)
    wipe: bool = False
    rank_deletes: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class SpillWorkerMeta(KVConnectorWorkerMetadata):
    # job_id -> [ok ranks, failed ranks]
    done: dict[int, list[int]] = field(default_factory=dict)
    # files a read found unusable on some rank (missing, short, header, CRC): every RP using them is forgotten
    bad: list[str] = field(default_factory=list)
    # unexpected exceptions caught on the I/O threads since the last report (summed over ranks): opens the breaker
    exc: int = 0
    # rank -> its start-up report (identity signature + the block files it holds), sent once (reuse across restarts)
    hello: dict[int, dict] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, SpillWorkerMeta)
        merged = {k: list(v) for k, v in self.done.items()}
        for k, (ok, bad) in other.done.items():
            cur = merged.setdefault(k, [0, 0])
            cur[0] += ok
            cur[1] += bad
        hello = dict(getattr(self, "hello", None) or {})
        hello.update(getattr(other, "hello", None) or {})
        return SpillWorkerMeta(done=merged, bad=sorted(set(self.bad) | set(other.bad)),
                               exc=int(self.exc) + int(getattr(other, "exc", 0)), hello=hello)


# ----------------------------------------------------------------------------------------------------------------
# on-disk file format (worker side)
# ----------------------------------------------------------------------------------------------------------------
def _crcs(mv: memoryview) -> list[int]:
    return [zlib.crc32(mv[i:i + CRC_BLOCK]) for i in range(0, len(mv), CRC_BLOCK)]


def _pack_header(meta: dict) -> bytes:
    body = json.dumps(meta, separators=(",", ":")).encode()
    if len(MAGIC) + 8 + len(body) > HDR:
        raise ValueError(f"{TAG} header too large ({len(body)} B)")
    raw = MAGIC + struct.pack("<II", FORMAT_VERSION, len(body)) + body
    return raw + b"\0" * (HDR - len(raw))


def _unpack_header(raw: bytes) -> dict:
    if raw[:len(MAGIC)] != MAGIC:
        raise ValueError("not a spill block file")
    ver, n = struct.unpack("<II", raw[len(MAGIC):len(MAGIC) + 8])
    if ver != FORMAT_VERSION:
        raise ValueError(f"spill format {ver} != {FORMAT_VERSION}")
    off = len(MAGIC) + 8
    return json.loads(raw[off:off + n])


class _Staging:
    """One page-aligned host buffer (pinned when CUDA is up) used as an O_DIRECT bounce buffer."""

    def __init__(self, nbytes: int, pin: bool):
        raw = torch.empty(nbytes + ALIGN, dtype=torch.uint8, pin_memory=pin)
        off = (-raw.data_ptr()) % ALIGN
        self._raw = raw
        self.t = raw[off:off + nbytes]
        assert self.t.data_ptr() % ALIGN == 0
        self.mv = memoryview(self.t.numpy()).cast("B")


class BlockFileStore:
    """This rank's block files and the byte movement GPU rows <-> file. All methods run on worker threads."""

    def __init__(self, cfg: SpillConfig, rank: int, views: list[torch.Tensor], rows: list[list[tuple[int, int]]],
                 sig: str):
        self.cfg = cfg
        self.rank = rank
        self.dir = cfg.rank_dir(rank)
        self.views = views          # tensor index -> (num_blocks, page) uint8 view of the KV tensor
        self.rows = rows            # group -> [(tensor index, page bytes)]
        self.sig = sig
        self.group_bytes = [sum(n for _, n in r) for r in rows]
        self.cuda = bool(views) and views[0].is_cuda
        os.makedirs(self.dir, exist_ok=True)

    # -- housekeeping ---------------------------------------------------------------------------------------------
    def wipe(self) -> int:
        n = 0
        for name in os.listdir(self.dir):
            if name.endswith(".blk") or name.endswith(".partial"):
                try:
                    os.unlink(os.path.join(self.dir, name))
                    n += 1
                except FileNotFoundError:
                    pass
        return n

    def disk_free(self) -> int:
        st = os.statvfs(self.dir)
        return st.f_bavail * st.f_frsize

    def file_bytes(self, group: int) -> int:
        return HDR + _up(self.group_bytes[group])

    def delete(self, fname: str) -> bool:
        """Best effort: a file that can not be removed (read-only/failed filesystem, EACCES, EIO) is only logged; the
        scheduler has already forgotten it, so it is never read again (and the next start wipes the directory)."""
        try:
            os.unlink(os.path.join(self.dir, fname))
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.warning("%s rank %d: could not delete %s: %r", TAG, self.rank, fname, e)
            return False

    def fsync_dir(self) -> None:
        try:
            fd = os.open(self.dir, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def _open(self, path: str, flags: int) -> tuple[int, bool]:
        if self.cfg.direct and hasattr(os, "O_DIRECT"):
            try:
                return os.open(path, flags | os.O_DIRECT, 0o600), True
            except OSError as e:
                if e.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
                    raise
        return os.open(path, flags, 0o600), False

    # -- write ------------------------------------------------------------------------------------------------------
    def write_block(self, group: int, block_id: int, fname: str, stage: _Staging, stream, wid: str | None = None) -> None:
        nbytes = self.group_bytes[group]
        total = HDR + _up(nbytes)
        if total > stage.t.numel():
            raise ValueError(f"{TAG} block of group {group} ({total} B) exceeds the staging buffer")
        data = stage.t[HDR:HDR + nbytes]
        off = 0
        ctx = torch.cuda.stream(stream) if stream is not None else _null()
        with ctx:
            for t, page in self.rows[group]:
                data[off:off + page].copy_(self.views[t][block_id], non_blocking=stream is not None)
                off += page
        if stream is not None:
            stream.synchronize()
        if total > HDR + nbytes:
            stage.t[HDR + nbytes:total].zero_()
        crcs = _crcs(stage.mv[HDR:HDR + nbytes])
        meta = {"f": fname, "g": group, "rank": self.rank, "sig": self.sig, "n": nbytes,
                "rows": [list(r) for r in self.rows[group]], "crc": crcs}
        if wid is not None:
            meta["w"] = wid     # which save wrote it: a re-adopted index names it, the load checks it
        stage.mv[:HDR] = _pack_header(meta)
        path = os.path.join(self.dir, fname)
        tmp = f"{path}.{threading.get_ident()}.partial"
        fd, _ = self._open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            mv = stage.mv[:total]
            done = 0
            while done < total:
                w = os.pwrite(fd, mv[done:], done)
                if w <= 0:
                    raise OSError(errno.EIO, "short write")
                done += w
            os.fdatasync(fd)
        except BaseException:
            os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass        # read-only/failed filesystem: the .partial is never read (start-up wipes it)
            raise
        os.close(fd)
        os.rename(tmp, path)

    # -- read -------------------------------------------------------------------------------------------------------
    def read_block(self, group: int, block_id: int, fname: str, stage: _Staging, stream, wid: str | None = None) -> None:
        nbytes = self.group_bytes[group]
        total = HDR + _up(nbytes)
        if total > stage.t.numel():
            raise ValueError(f"{TAG} block of group {group} ({total} B) exceeds the staging buffer")
        path = os.path.join(self.dir, fname)
        fd, _ = self._open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if size != total:
                raise ValueError(f"{TAG} {fname}: size {size} != {total}")
            mv = stage.mv[:total]
            done = 0
            while done < total:
                r = os.preadv(fd, [mv[done:]], done)
                if r <= 0:
                    raise OSError(errno.EIO, "short read")
                done += r
        finally:
            os.close(fd)
        meta = _unpack_header(bytes(stage.mv[:HDR]))
        if (meta.get("f") != fname or meta.get("g") != group or meta.get("rank") != self.rank
                or meta.get("sig") != self.sig or meta.get("n") != nbytes
                or [tuple(r) for r in meta.get("rows", [])] != list(self.rows[group])):
            raise ValueError(f"{TAG} {fname}: header does not match this rank/layout")
        if wid is not None and meta.get("w") != wid:
            # the file was re-written by another save than the one the scheduler's index names (e.g. a stale index after
            # a crash between a delete and a re-write on one rank): never mix two computations of a prefix across ranks
            raise ValueError(f"{TAG} {fname}: written by another save ({meta.get('w')!r} != {wid!r})")
        if _crcs(stage.mv[HDR:HDR + nbytes]) != meta.get("crc"):
            raise ValueError(f"{TAG} {fname}: CRC mismatch")
        data = stage.t[HDR:HDR + nbytes]
        off = 0
        ctx = torch.cuda.stream(stream) if stream is not None else _null()
        with ctx:
            for t, page in self.rows[group]:
                self.views[t][block_id].copy_(data[off:off + page], non_blocking=stream is not None)
                off += page
        if stream is not None:
            stream.synchronize()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class SpillWorker:
    """Worker-rank side: executes the scheduler's jobs on two threads (writer, reader) and reports results."""

    def __init__(self, cfg: SpillConfig, kv_cache_config: "KVCacheConfig", rank: int):
        self.cfg = cfg
        self.kv_cache_config = kv_cache_config
        self.rank = rank
        self.store: BlockFileStore | None = None
        self._done_lock = threading.Lock()
        self._done: dict[int, list[int]] = {}
        self._bad: set[str] = set()
        self._wq: queue.Queue = queue.Queue()
        self._rq: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._exc_new = 0                       # unexpected exceptions not yet reported to the scheduler
        self._answered_jobs: deque[int] = deque(maxlen=4096)
        self._busy: dict[str, tuple[str, int, float]] = {}   # thread tag -> (kind, job, start) of the job it runs
        self._last_log = time.monotonic()
        self.stats = {"written": 0, "write_failed": 0, "loaded": 0, "load_failed": 0, "deleted": 0,
                      "delete_failed": 0, "write_s": 0.0, "load_s": 0.0, "written_bytes": 0, "loaded_bytes": 0,
                      "thread_exc": 0, "wiped": 0, "scan_s": 0.0}
        self._hello: dict | None = None          # start-up report (persist), sent once with the next worker meta
        self._hello_sent = False

    def register(self, kv_caches: dict[str, torch.Tensor], sig: str, ident_fn=None) -> None:
        pages, rows = group_rows(self.kv_cache_config)
        num_blocks = self.kv_cache_config.num_blocks
        views: list[torch.Tensor | None] = [None] * len(pages)
        for t, kvt in enumerate(self.kv_cache_config.kv_cache_tensors):
            name = next((n for n in kvt.shared_by if n in kv_caches), None)
            if name is None:
                continue
            storage = kv_caches[name].untyped_storage()
            if storage.nbytes() < kvt.size:
                raise ValueError(f"{TAG} tensor {t}: storage {storage.nbytes()} < {kvt.size}")
            raw = torch.empty(0, dtype=torch.uint8, device=kv_caches[name].device).set_(storage)
            views[t] = raw[:kvt.size].view(num_blocks, pages[t])
        for g, r in enumerate(rows):
            for t, _ in r:
                if views[t] is None:
                    raise ValueError(f"{TAG} group {g} uses tensor {t} which no registered layer maps")
        self.store = BlockFileStore(self.cfg, self.rank, views, rows, sig)  # type: ignore[arg-type]
        need = max(HDR + _up(b) for b in self.store.group_bytes) if self.store.group_bytes else HDR
        if self.cfg.stage_bytes < need:
            raise ValueError(f"{TAG} GLM53_NVME_SPILL_STAGE_MIB too small: need {need} B")
        if self.cfg.persist:
            # Reuse across restarts: wipe nothing yet. The writer thread's first job scans this rank's files and
            # computes its identity; the scheduler decides (adopt / wipe) from every rank's report and ships the
            # decision ahead of any other job, through this same FIFO.
            wiped = "deferred"
            self._wq.put(("scan", 0, ident_fn, None))
        else:
            wiped = self.store.wipe()
        pin = self.store.cuda
        dev = views[0].device if views and views[0] is not None else None
        for target, q, name in ((self._writer, self._wq, "w"), (self._reader, self._rq, "r")):
            stage = _Staging(self.cfg.stage_bytes, pin)
            stream = torch.cuda.Stream(device=dev) if pin else None
            th = threading.Thread(target=target, args=(q, stage, stream), name=f"glm53-spill-{name}{self.rank}",
                                  daemon=True)
            th.start()
            self._threads.append(th)
        logger.info("%s rank %d: dir=%s wiped=%s group_bytes=%s stage=%d MiB direct=%s", TAG, self.rank,
                    self.store.dir, wiped, self.store.group_bytes, self.cfg.stage_bytes >> 20, self.cfg.direct)

    # called first thing in the step (before the forward and before any write into new blocks)
    def handle(self, meta: SpillMeta) -> None:
        if self.store is None or meta is None or getattr(meta, "_glm53_handled", False):
            return
        meta._glm53_handled = True  # type: ignore[attr-defined]
        wipe = bool(getattr(meta, "wipe", False))
        rank_deletes = (getattr(meta, "rank_deletes", None) or {}).get(self.rank, ())
        if not (meta.saves or meta.loads or meta.deletes or wipe or rank_deletes):
            return
        ev = None
        if self.store.cuda and (meta.saves or meta.loads):
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
        # the start-up decision first (it is shipped before any save of this run)
        if wipe:
            self._wq.put(("wipe", 0, None, None))
        for fname in rank_deletes:
            self._wq.put(("delete", 0, fname, None))
        # deletes first: a file deleted and re-written in the same step must end up written
        for fname in meta.deletes:
            self._wq.put(("delete", 0, fname, None))
        for job_id, entries in meta.saves.items():
            self._wq.put(("save", job_id, entries, ev))
        for job_id, entries in meta.loads.items():
            self._rq.put(("load", job_id, entries, ev))

    def _finish(self, job_id: int, ok: bool) -> None:
        with self._done_lock:
            cur = self._done.setdefault(job_id, [0, 0])
            cur[0 if ok else 1] += 1
            self._answered_jobs.append(job_id)

    def _thread_init(self, stream) -> None:
        if stream is not None:
            torch.cuda.set_device(stream.device)

    def _note_exc(self, where: str, e: BaseException) -> None:
        with self._done_lock:
            self._exc_new += 1
            self.stats["thread_exc"] += 1
        logger.error("%s rank %d: unexpected exception in %s: %r", TAG, self.rank, where, e)

    def _loop(self, tag: str, q: queue.Queue, stage: _Staging, stream) -> None:
        """One I/O thread. It never dies on a job: every job is answered (ok or failed) whatever is raised, so the
        scheduler always gets this rank's result and releases what it pinned. Only "stop" ends the loop."""
        store = self.store
        try:
            self._thread_init(stream)
        except BaseException as e:      # noqa: BLE001 - keep answering (every job fails below)
            self._note_exc(f"{tag} init", e)
            stage = None    # type: ignore[assignment]
        while True:
            kind, job_id, payload, ev = q.get()
            self._busy[tag] = (kind, job_id, time.monotonic())
            try:
                if kind == "stop":
                    return
                if store is None or (stage is None and kind in ("save", "load")):
                    raise RuntimeError("I/O thread not initialised")
                if kind == "delete":
                    if store.delete(payload):
                        self.stats["deleted"] += 1
                    else:
                        self.stats["delete_failed"] += 1
                elif kind == "scan":
                    self._scan(store, payload)
                elif kind == "wipe":
                    n = store.wipe()
                    self.stats["wiped"] += n
                    logger.info("%s rank %d: wiped %d block files (start-up decision of the scheduler)", TAG,
                                self.rank, n)
                elif kind == "save":
                    self._save(job_id, payload, ev, store, stage, stream)
                elif kind == "load":
                    self._load(job_id, payload, ev, store, stage, stream)
                else:
                    raise RuntimeError(f"unknown job kind {kind!r}")
            except BaseException as e:      # noqa: BLE001 - _save/_load answer themselves; this is the last net
                self._note_exc(f"{tag} {kind} job {job_id}", e)
                if kind in ("save", "load") and not self._answered(job_id):
                    self._finish(job_id, False)
                if kind == "scan" and self._hello is None:
                    self._set_hello({"rank": self.rank, "sig": None, "error": f"scan: {type(e).__name__}",
                                     "files": {}})
            finally:
                self._busy.pop(tag, None)
                q.task_done()

    def _writer(self, q: queue.Queue, stage: _Staging, stream) -> None:
        self._loop("w", q, stage, stream)

    def _reader(self, q: queue.Queue, stage: _Staging, stream) -> None:
        self._loop("r", q, stage, stream)

    def _set_hello(self, rep: dict) -> None:
        with self._done_lock:
            self._hello = rep

    def _scan(self, store: "BlockFileStore", ident_fn) -> None:
        """Start-up report of this rank (reuse across restarts): its identity signature and the block files it holds
        (name -> size). Leftover .partial files are removed (never referenced). Nothing else is touched."""
        t0 = time.perf_counter()
        files: dict[str, int] = {}
        partial = 0
        err = None
        try:
            for name in os.listdir(store.dir):
                p = os.path.join(store.dir, name)
                if name.endswith(".partial"):
                    try:
                        os.unlink(p)
                        partial += 1
                    except OSError:
                        pass
                elif name.endswith(".blk"):
                    try:
                        files[name] = os.stat(p).st_size
                    except OSError:
                        pass
        except OSError as e:
            err = f"listdir: {e!r}"
        parts: dict[str, Any] = {}
        try:
            parts = dict(ident_fn()) if ident_fn is not None else {}
        except Exception as e:  # noqa: BLE001
            err = err or f"identity: {type(e).__name__}"
        parts.update({"rank": self.rank, "rows": _h([list(map(list, r)) for r in store.rows]), "header_sig": store.sig})
        dt = time.perf_counter() - t0
        self.stats["scan_s"] = dt
        self._set_hello({"rank": self.rank, "sig": _sig_of(parts), "parts": {k: str(v)[:24] for k, v in parts.items()},
                         "files": files, "file_bytes": [store.file_bytes(g) for g in range(len(store.rows))],
                         "partial": partial, "scan_s": round(dt, 3), "error": err})
        logger.info("%s rank %d: start-up scan: %d block files (%.2f GiB), %d partial removed, %.2f s%s", TAG,
                    self.rank, len(files), sum(files.values()) / (1 << 30), partial, dt, f" ERROR {err}" if err else "")

    def _answered(self, job_id: int) -> bool:
        with self._done_lock:
            return job_id in self._answered_jobs

    def _save(self, job_id, payload, ev, store, stage, stream) -> None:
        t0 = time.perf_counter()
        ok = False
        try:
            if stream is not None:
                # The blocks are pinned and immutable, but the forward that wrote the newest of them may still be
                # queued (async scheduling): let everything already on the device finish first.
                if ev is not None:
                    stream.wait_event(ev)
                torch.cuda.synchronize(stream.device)
            need = sum(store.file_bytes(ent[0]) for ent in payload)
            if store.disk_free() - need < self.cfg.min_free_bytes:
                raise OSError(errno.ENOSPC, f"free-disk floor ({self.cfg.min_free_bytes >> 30} GiB)")
            for ent in payload:
                g, bid, fname = ent[0], ent[1], ent[2]
                store.write_block(g, bid, fname, stage, stream, ent[3] if len(ent) > 3 else None)
                self.stats["written_bytes"] += store.file_bytes(g)
            store.fsync_dir()
            self.stats["written"] += 1
            ok = True
        except BaseException as e:      # noqa: BLE001 - any failure: the job fails, the scheduler forgets it
            self.stats["write_failed"] += 1
            if not isinstance(e, (OSError, ValueError)):
                self._note_exc(f"save job {job_id}", e)
            else:
                logger.warning("%s rank %d: save job %d failed: %r", TAG, self.rank, job_id, e)
            for ent in payload:
                store.delete(ent[2])    # never raises
        finally:
            self.stats["write_s"] += time.perf_counter() - t0
            self._finish(job_id, ok)

    def _load(self, job_id, payload, ev, store, stage, stream) -> None:
        t0 = time.perf_counter()
        ok = False
        cur = None
        try:
            if stream is not None:
                # The fresh blocks may have been freed by a request still in the in-flight batch (async
                # scheduling, abort): let everything already queued on the device drain before writing them.
                if ev is not None:
                    stream.wait_event(ev)
                torch.cuda.synchronize(stream.device)
            for ent in payload:
                g, bid, fname = ent[0], ent[1], ent[2]
                cur = fname
                store.read_block(g, bid, fname, stage, stream, ent[3] if len(ent) > 3 else None)
                self.stats["loaded_bytes"] += store.file_bytes(g)
            self.stats["loaded"] += 1
            ok = True
        except BaseException as e:      # noqa: BLE001
            if cur is not None and isinstance(e, (OSError, ValueError)):
                with self._done_lock:
                    self._bad.add(cur)
            self.stats["load_failed"] += 1
            if not isinstance(e, (OSError, ValueError)):
                self._note_exc(f"load job {job_id}", e)
            else:
                logger.warning("%s rank %d: load job %d failed: %r", TAG, self.rank, job_id, e)
        finally:
            self.stats["load_s"] += time.perf_counter() - t0
            self._finish(job_id, ok)

    def take_meta(self) -> SpillWorkerMeta | None:
        if self.cfg.log_every_s > 0 and time.monotonic() - self._last_log >= self.cfg.log_every_s:
            self._last_log = time.monotonic()
            logger.info("%s", self.stats_line())
        with self._done_lock:
            hello = {}
            if self._hello is not None and not self._hello_sent:
                hello = {self.rank: self._hello}
                self._hello_sent = True
            if not self._done and not self._bad and not self._exc_new and not hello:
                return None
            done, self._done = self._done, {}
            bad, self._bad = sorted(self._bad), set()
            exc, self._exc_new = self._exc_new, 0
        return SpillWorkerMeta(done=done, bad=bad, exc=exc, hello=hello)

    def stats_line(self) -> str:
        now = time.monotonic()
        busy = " ".join(f"{t}={k}:{j}/{now - t0:.1f}s" for t, (k, j, t0) in sorted(self._busy.items())) or "idle"
        st = self.stats
        return (f"{TAG} rank {self.rank} io: {busy} queued w={self._wq.qsize()} r={self._rq.qsize()} "
                f"written={st['written']} ({st['written_bytes'] / (1 << 30):.2f} GiB) write_failed={st['write_failed']} "
                f"loaded={st['loaded']} ({st['loaded_bytes'] / (1 << 30):.2f} GiB) load_failed={st['load_failed']} "
                f"delete_failed={st['delete_failed']} thread_exc={st['thread_exc']} "
                f"threads_alive={sum(th.is_alive() for th in self._threads)}/{len(self._threads)}")

    def drain(self, timeout: float = 60.0) -> bool:
        """Test helper: wait until both queues are empty and idle."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if self._wq.unfinished_tasks == 0 and self._rq.unfinished_tasks == 0:
                return True
            time.sleep(0.005)
        return False

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop both I/O threads after the job each is running (bounded wait: a thread stuck in the kernel on a dead
        disk is left behind, it is a daemon). Exiting the process while a thread is inside a torch copy aborts the
        interpreter at finalization ("terminate called without an active exception", seen in the async rig)."""
        for q in (self._wq, self._rq):
            q.put(("stop", 0, None, None))
        t_end = time.monotonic() + timeout
        for th in self._threads:
            th.join(max(0.0, t_end - time.monotonic()))


# ----------------------------------------------------------------------------------------------------------------
# resume-point index (reuse across restarts): one file on the scheduler's node, written atomically
# ----------------------------------------------------------------------------------------------------------------
def _read_index(path: str) -> tuple[dict | None, str]:
    """(index, "") or (None, why). The first line carries the body's length and sha256; anything that does not verify
    (missing, truncated, corrupted, other version) is unusable -> the caller wipes."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None, "no index file"
    except OSError as e:
        return None, f"unreadable index: {e!r}"
    try:
        nl = raw.index(b"\n")
        top = json.loads(raw[:nl])
        body = raw[nl + 1:]
        if top.get("magic") != INDEX_MAGIC or top.get("version") != INDEX_VERSION:
            return None, "index of another format/version"
        if top.get("len") != len(body) or top.get("sha256") != hashlib.sha256(body).hexdigest():
            return None, "index truncated or corrupted (length/sha256)"
        idx = json.loads(body)
        if idx.get("magic") != INDEX_MAGIC or not isinstance(idx.get("rps"), list):
            return None, "index body malformed"
        return idx, ""
    except (ValueError, TypeError, AttributeError) as e:
        return None, f"index unparsable: {type(e).__name__}"


class _IndexWriter:
    """Writes the index off the scheduler thread: <path>.tmp -> fsync -> rename -> fsync(dir). The newest pending
    payload wins (an older one not yet written is dropped)."""

    def __init__(self, path: str):
        self.path = path
        self._cv = threading.Condition()
        self._pending: tuple[bytes, dict] | None = None
        self._stop = False
        self._busy = False
        self._th = threading.Thread(target=self._run, name="glm53-spill-index", daemon=True)
        self._th.start()

    def _write(self, payload: bytes) -> None:
        d = os.path.dirname(self.path)
        os.makedirs(d, exist_ok=True)
        tmp = f"{self.path}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            mv = memoryview(payload)
            done = 0
            while done < len(payload):
                done += os.write(fd, mv[done:])
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(tmp, self.path)
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def _do(self, payload: bytes, stats: dict) -> None:
        try:
            self._write(payload)
            stats["index_writes"] = stats.get("index_writes", 0) + 1
        except Exception as e:  # noqa: BLE001 - a lost index only costs reuse at the next start
            stats["index_write_failed"] = stats.get("index_write_failed", 0) + 1
            logger.warning("%s index write failed: %r", TAG, e)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._pending is None and self._stop:
                    return
                payload, stats = self._pending
                self._pending = None
                self._busy = True
            try:
                self._do(payload, stats)
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def submit(self, payload: bytes, stats: dict, sync: bool = False) -> None:
        with self._cv:
            self._pending = (payload, stats)
            self._cv.notify_all()
            if sync:
                t_end = time.monotonic() + 10.0
                while (self._pending is not None or self._busy) and time.monotonic() < t_end:
                    self._cv.wait(0.1)

    def flush(self, timeout: float = 10.0) -> None:
        with self._cv:
            t_end = time.monotonic() + timeout
            while (self._pending is not None or self._busy) and time.monotonic() < t_end:
                self._cv.wait(0.1)

    def stop(self) -> None:
        self.flush()
        with self._cv:
            self._stop = True
            self._cv.notify_all()


# ----------------------------------------------------------------------------------------------------------------
# scheduler side
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class _DiskBlock:
    fname: str
    group: int
    nbytes: int
    state: str = "pending"      # pending | ok
    refs: int = 0               # resume points referencing it
    busy: int = 0               # in-flight loads reading it
    wid: str = ""               # id of the save that wrote it (in the file header; loads check it)


@dataclass
class _RPBlock:
    group: int
    keys: tuple[bytes, ...]     # primary key first, then the block's extra keys
    num_tokens: int | None
    fname: str


@dataclass
class _RP:
    key: bytes                  # request.block_hashes[N // hash_block_size - 1]
    n_tokens: int
    blocks: list[_RPBlock]
    state: str = "saving"       # saving | ok
    last_used: float = 0.0
    pending_jobs: set[int] = field(default_factory=set)
    # request.block_hashes[:n // hash_block_size]: another RP X is a prefix of this one (same conversation, an
    # earlier turn) iff X.n < n and chain[X.n // hash_block_size - 1] == X.key (the hashes are chained)
    chain: tuple[bytes, ...] = ()
    # per-request TTL (DSH contract v1.1): None = server default (no expiry, LRU only); 0 = expired; else seconds since
    # last_used (sliding: a request extending it or a restore refreshes last_used). session/epoch: DSH session + epoch.
    ttl_s: int | None = None
    session: str | None = None
    epoch: int | None = None
    drop_reason: str = ""       # set when it was marked expired while in use (dropped by the sweep once free)
    stale_at: float = 0.0       # wall time a newer epoch of its session was seen (0 = current); deleted after the grace
    ser: str = ""               # cached JSON of the immutable part (index writes)

    def extends(self, other: "_RP", hbs: int) -> bool:
        """True when ``other`` is a strict prefix of this resume point (an older RP of the same chain)."""
        if other.n_tokens >= self.n_tokens or other.n_tokens <= 0 or other.n_tokens % hbs:
            return False
        i = other.n_tokens // hbs - 1
        return i < len(self.chain) and self.chain[i] == other.key

    def blocks_by_fname(self, items) -> list[_RPBlock | None]:
        by = {e.fname: e for e in self.blocks}
        return [by.get(fname) for _g, _b, fname in items]


@dataclass
class _Restore:
    rp: _RP
    job_id: int
    local_before: int
    new_blocks: list[Any]
    pins: list[Any]
    entries: list[_RPBlock]
    state: str = "loading"      # loading | ready | done | failed
    t0: float = 0.0
    t_ready: float = 0.0
    orphan: bool = False
    failures: int = 0
    restores: int = 1
    main_released: bool = False     # pin timeout: the full-attention blocks were unpinned, the state blocks kept


class SpillScheduler:
    def __init__(self, cfg: SpillConfig, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        self.cfg = cfg
        self.kv_cache_config = kv_cache_config
        self.clock = time.monotonic      # replaceable (the CPU replay rig drives a simulated clock)
        self.io_clock = time.monotonic   # job deadlines / stats cadence: always wall time (I/O runs in real time)
        self.wall = time.time            # TTL / last_used (persisted across restarts); replaceable (rig: simulated)
        self.vllm_config = vllm_config
        self.world = int(vllm_config.parallel_config.world_size)
        pages, rows = group_rows(kv_cache_config)
        self.group_bytes = [sum(n for _, n in r) for r in rows]
        self.file_bytes = [HDR + _up(b) for b in self.group_bytes]
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        def _inner(spec):
            specs = getattr(spec, "kv_cache_specs", None)
            return next(iter(specs.values())) if isinstance(specs, dict) and specs else spec

        # full-attention (MLA) groups: many blocks per RP, evicted last; the others hold one boundary state each
        self.main_groups = {i for i, g in enumerate(kv_cache_config.kv_cache_groups)
                            if isinstance(_inner(g.kv_cache_spec), FullAttentionSpec)}
        self.block_pool = None
        self.kv_cache_manager = None
        self.hash_block_size: int | None = None
        self._job = 0
        self._meta = SpillMeta()
        self.disk: dict[str, _DiskBlock] = {}
        self.disk_bytes = 0
        self.rps: OrderedDict[bytes, _RP] = OrderedDict()     # LRU order (oldest first)
        self.save_jobs: dict[int, tuple[list[tuple[int, Any, str]], list[Any], _RP]] = {}
        self.restores: dict[str, _Restore] = {}
        self.restore_by_job: dict[int, str] = {}
        self.space_wait: dict[str, float] = {}      # req_id -> first time its restore did not fit
        self.gave_up: set[str] = set()
        self.rp_loading: dict[bytes, tuple[int, float]] = {}   # RP key -> (job id, start) of the restore in flight
        # Admitted requests, to take their resume point right after the prompt is computed: the boundary state of
        # the last prompt chunk is released (cached, evictable) at prefill end and may be gone by the time the
        # request finishes its decode (observed in the replay rig).
        self.tracked: dict[str, Any] = {}
        self.prefill_noted: set[str] = set()       # last prompt chunk scheduled in the previous step
        self.saved_once: set[str] = set()
        self.orphan_restores: dict[int, _Restore] = {}
        self.job_acc: dict[int, list[int]] = {}
        self.deferred: OrderedDict[str, tuple[bytes, int]] = OrderedDict()   # req_id -> (rp key, n) below highwater
        self.write_log: deque[tuple[float, int]] = deque()
        self.seen: OrderedDict[bytes, None] = OrderedDict()    # RP keys of finished requests (policy "repeat")
        # fault containment
        self.job_sent: dict[int, float] = {}       # job -> io_clock when it left in a SchedulerOutput
        self.expired: set[int] = set()             # jobs given up at their deadline; their late answers are ignored
        self.quarantine: dict[int, _Restore] = {}  # timed-out loads: fresh blocks held until every rank answers
        self.no_restore: set[str] = set()          # requests detached to the normal prefill path
        self.breaker: str | None = None            # None = closed; else why the tier was disabled
        self.consec_fail = 0
        self.max_pinned = cfg.max_pinned_blocks or 0
        self._last_log = self.io_clock()
        self.stats = {
            "rp_saved": 0, "rp_save_failed": 0, "rp_skipped_short": 0, "rp_skipped_cap": 0, "rp_skipped_first": 0,
            "rp_skipped_rate": 0, "rp_dedup_blocks": 0, "save_bytes": 0, "rp_evicted": 0,
            "restore_started": 0, "restore_ok": 0, "restore_failed": 0, "restore_short": 0,
            "restore_no_space": 0, "restore_admitted": 0, "bad_files": 0, "restore_coalesced_waits": 0, "restore_space_wait_steps": 0, "restore_space_giveup": 0, "restored_tokens": 0, "restore_loaded_blocks": 0,
            "restore_pinned_blocks": 0, "restore_timeouts": 0, "deferred_written": 0, "save_moved_blocks": 0,
            "save_timeouts": 0, "load_timeouts": 0, "restore_fallback": 0, "rp_skipped_pins": 0,
            "pinned_excl_peak": 0, "breaker_trips": 0, "late_answers": 0, "worker_exc": 0, "jobs_failed": 0,
            "pinned_peak": 0, "quarantine_freed": 0, "orphan_files_deleted": 0,
            "rp_pruned": 0, "prune_freed_bytes": 0, "rp_prune_skipped_busy": 0, "rp_prune_kept_branch": 0,
            # 2026-10-14: TTL / epoch (DSH contract v1.1) and reuse across restarts
            "req_with_params": 0, "rp_expired": 0, "expired_bytes": 0, "rp_epoch_dropped": 0, "rp_skipped_ttl0": 0,
            "rp_skipped_old_epoch": 0, "rp_saved_stale": 0, "rp_ttl_refresh": 0, "rp_epoch_marked": 0, "rp_saved_ttl_first": 0, "rp_marked_expired_busy": 0,
            "persist_adopted_rps": 0, "persist_adopted_bytes": 0, "persist_dropped_missing": 0,
            "persist_dropped_expired": 0, "persist_orphans": 0, "persist_wipes": 0, "index_writes": 0,
            "index_write_failed": 0,
        }
        # per-request spill parameters and their one-time effects (refresh / ttl 0 / epoch)
        self.req_params: dict[str, SpillParams | None] = {}
        self.req_noted: set[str] = set()
        self.session_epoch: OrderedDict[str, int] = OrderedDict()
        self._last_ttl_sweep: float | None = None
        # reuse across restarts
        self.boot = os.urandom(4).hex()
        self.persist_state = "off"           # off | adopting | on | disabled
        self.persist_reason = ""
        self.hello: dict[int, dict] = {}
        self.hash_info: tuple[bool, str] | None = None
        self.seed_source = ""
        self._ident: dict[str, str] | None = None
        self._ident_thread: threading.Thread | None = None
        self._index: dict | None = None
        self._index_err = ""
        self._adopt_t0: float | None = None
        self._adopt_s = 0.0
        self._ident_s = 0.0
        self._index_dirty = False
        self._index_urgent = False
        self._index_last = 0.0
        self._index_written_wall = 0.0
        self._index_writer: _IndexWriter | None = None
        if cfg.persist:
            self.persist_state = "adopting"
            self._index, self._index_err = _read_index(cfg.index_path())
            self._ident_thread = threading.Thread(target=self._compute_ident, name="glm53-spill-ident", daemon=True)
            self._ident_thread.start()
            self._index_writer = _IndexWriter(cfg.index_path())

    # -- binding ----------------------------------------------------------------------------------------------------
    def bind(self, block_pool=None, kv_cache_manager=None) -> None:
        if block_pool is not None:
            self.block_pool = block_pool
            self.hash_block_size = block_pool.hash_block_size
            if not self.cfg.max_pinned_blocks:
                self.max_pinned = max(16, int(block_pool.num_gpu_blocks) // 2)
        if kv_cache_manager is not None:
            self.kv_cache_manager = kv_cache_manager
            if self.block_pool is None:
                self.bind(block_pool=kv_cache_manager.block_pool)

    @property
    def ready(self) -> bool:
        return self.block_pool is not None and self.kv_cache_manager is not None

    def _next_job(self) -> int:
        self._job += 1
        return self._job

    # -- save -------------------------------------------------------------------------------------------------------
    def _plan_rp(self, request: "Request") -> tuple[bytes, int, list[tuple[_RPBlock, Any]]] | None:
        return self._plan_from(request.block_hashes, request.num_tokens)

    def _plan_from(self, block_hashes, num_tokens: int,
                   min_tokens: int | None = None) -> tuple[bytes, int, list[tuple[_RPBlock, Any]]] | None:
        coord = self.kv_cache_manager.coordinator
        hbs = self.hash_block_size
        max_len = num_tokens - 1
        if max_len <= 0 or not block_hashes:
            return None
        blocks_by_group, n, _ = coord.find_longest_cache_hit(block_hashes, max_len)
        if n < (self.cfg.min_tokens if min_tokens is None else min_tokens) or n % hbs or n // hbs > len(block_hashes):
            return None
        rp_key = bytes(block_hashes[n // hbs - 1])
        pool = self.block_pool
        out: list[tuple[_RPBlock, Any]] = []
        for g, blocks in enumerate(blocks_by_group):
            for b in blocks:
                if b.is_null:
                    continue
                if b.block_hash is None:
                    return None
                extras = sorted(pool.cached_block_hashes_by_block.get(b.block_id, ()))
                keys = (bytes(b.block_hash), *[bytes(k) for k in extras if k != b.block_hash])
                out.append((_RPBlock(g, keys, b.block_hash_num_tokens, block_file_name(keys[0])), b))
        if not out:
            return None
        return rp_key, n, out

    def _rate_ok(self, nbytes: int) -> bool:
        cap = self.cfg.max_write_bytes_per_h
        if cap <= 0:
            return True
        now = self.clock()
        while self.write_log and now - self.write_log[0][0] > 3600:
            self.write_log.popleft()
        return sum(b for _, b in self.write_log) + nbytes <= cap

    def _drop_rp(self, rp: _RP, reason: str) -> int:
        """Forget an RP; files no other RP references (and no load reads) are deleted. Returns the bytes deleted now
        (a file a load is still reading is deleted when that load finishes)."""
        if self.rps.get(rp.key) is not rp:
            return 0
        del self.rps[rp.key]
        freed = 0
        for e in rp.blocks:
            db = self.disk.get(e.fname)
            if db is None:
                continue
            db.refs -= 1
            if db.refs <= 0 and db.busy == 0 and db.state in ("ok", "bad"):
                freed += db.nbytes
                self._delete_disk(db)
        if reason == "lru":
            self.stats["rp_evicted"] += 1
        elif reason == "expired":
            self.stats["rp_expired"] += 1
            self.stats["expired_bytes"] += freed
        elif reason == "epoch":
            self.stats["rp_epoch_dropped"] += 1
        self._index_dirty = True
        return freed

    # -- per-request TTL / session epoch (DSH contract v1.1) --------------------------------------------------------
    def _params(self, request: "Request") -> SpillParams | None:
        rid = request.request_id
        if rid in self.req_params:
            return self.req_params[rid]
        p = parse_spill_params(getattr(request, "kv_transfer_params", None))
        self.req_params[rid] = p
        if p is not None:
            self.stats["req_with_params"] += 1
        return p

    def _min_tokens(self, p: SpillParams | None) -> int:
        if p is not None and p.explicit_keep and self.cfg.ttl_min_tokens > 0:
            return self.cfg.ttl_min_tokens
        return self.cfg.min_tokens

    def _is_expired(self, rp: _RP, now: float | None = None) -> bool:
        if rp.ttl_s == 0:
            return True
        if rp.stale_at or rp.ttl_s is not None:
            now = self.wall() if now is None else now
            if rp.stale_at and now - max(rp.last_used, rp.stale_at) >= EPOCH_GRACE_S:
                return True
            if rp.ttl_s is not None:
                return now - rp.last_used >= rp.ttl_s
        return False

    def _expire_rp(self, rp: _RP, reason: str) -> None:
        """TTL 0 / a newer epoch: drop it now, or (in use / still saving) mark it expired -- it is never restored again
        and the sweep drops it once free."""
        if self.rps.get(rp.key) is not rp:
            return
        if rp.state == "ok" and not self._rp_in_use(rp):
            self._drop_rp(rp, reason)
        else:
            rp.ttl_s = 0
            rp.drop_reason = reason
            self.stats["rp_marked_expired_busy"] += 1
        self._index_dirty = self._index_urgent = True

    def _chain_rps(self, block_hashes) -> list[_RP]:
        """The RPs ``block_hashes`` extends (its earlier boundaries), deepest last."""
        hbs = self.hash_block_size
        out = []
        if not hbs or not self.rps:
            return out
        for i, h in enumerate(block_hashes):
            rp = self.rps.get(bytes(h))
            if rp is not None and rp.n_tokens == (i + 1) * hbs:
                out.append(rp)
        return out

    def _note_epoch(self, session: str, epoch: int) -> None:
        cur = self.session_epoch.get(session)
        if cur is not None and epoch <= cur:
            return
        self.session_epoch[session] = epoch
        self.session_epoch.move_to_end(session)
        while len(self.session_epoch) > 4096:
            self.session_epoch.popitem(last=False)
        self._index_dirty = True
        if cur is None:
            return
        # A strictly newer epoch (DSH compaction): the session's older-epoch RPs can no longer be hit by its new
        # requests. They are marked stale and deleted by the sweep once unused for EPOCH_GRACE_S (a request of the old
        # epoch still in flight -- e.g. sent before the compaction finished -- can still use / refresh them), and they
        # go first at the cap.
        now = self.wall()
        for rp in self.rps.values():
            if rp.session == session and rp.epoch is not None and rp.epoch < epoch and not rp.stale_at:
                rp.stale_at = now
                self.stats["rp_epoch_marked"] += 1

    def _note_request(self, request: "Request") -> SpillParams | None:
        """Once per request (first lookup): epoch bookkeeping, ttl_s == 0 (expire this chain's and session's RPs) or
        the sliding refresh of the RPs it extends."""
        p = self._params(request)
        rid = request.request_id
        if rid in self.req_noted or self.persist_state == "adopting":
            return p
        self.req_noted.add(rid)
        # (the session epoch is noted when a request is admitted or saved, not at its first lookup: a request of a
        # newer epoch can sit in the waiting queue behind requests of the older one)
        chain = self._chain_rps(request.block_hashes)
        if p is not None and p.ttl_s == 0:
            victims = {id(r): r for r in chain if r.session is None or r.session == p.session}
            if p.session is not None:
                victims.update({id(r): r for r in self.rps.values() if r.session == p.session})
            for rp in victims.values():
                self._expire_rp(rp, "expired")
            return p
        now = self.wall()
        for rp in chain:
            if self._is_expired(rp, now):
                continue
            rp.last_used = now
            if p is not None and p.explicit_keep and (rp.session is None or rp.session == p.session):
                rp.ttl_s = p.ttl_s
            self.stats["rp_ttl_refresh"] += 1
            self._index_dirty = True
        return p

    def _sweep_expired(self) -> None:
        now = self.wall()
        if self._last_ttl_sweep is not None and now - self._last_ttl_sweep < self.cfg.ttl_sweep_s:
            return
        self._last_ttl_sweep = now
        for rp in list(self.rps.values()):
            if rp.state == "ok" and self._is_expired(rp, now) and not self._rp_in_use(rp):
                self._drop_rp(rp, rp.drop_reason or ("epoch" if rp.stale_at else "expired"))

    # -- superseded resume points ----------------------------------------------------------------------------------
    def _rp_in_use(self, rp: _RP) -> bool:
        """An RP a restore (loading / waiting for admission / admitted but its request still running / quarantined)
        or a load job refers to, or one with a save job in flight, is never pruned."""
        if rp.pending_jobs or rp.state != "ok" or rp.key in self.rp_loading:
            return True
        for st in list(self.restores.values()) + list(self.orphan_restores.values()) + list(self.quarantine.values()):
            if st.rp is rp:
                return True
        return False

    def _prune_superseded(self, rp: _RP) -> None:
        """``rp`` just became ok on every rank. Delete the older RPs of its chain (its strict prefixes = earlier turns of
        the same conversation) beyond the newest ``keep_per_chain`` (``rp`` itself counts as one). An older RP is kept
        when it is in use (_rp_in_use), or when it is still among the ``keep_per_chain - 1`` nearest predecessors of an
        RP on another branch (a sibling made by an edited / regenerated turn: that branch keeps its "one before").
        Decided here, in the scheduler only, so every rank gets the same deletes, through the normal delete path
        (_drop_rp -> SpillMeta.deletes -> each rank's writer FIFO, ahead of later writes); shared files survive by their
        reference count."""
        keep = self.cfg.keep_per_chain
        hbs = self.hash_block_size
        if keep <= 0 or not hbs or self.breaker is not None or self.rps.get(rp.key) is not rp or not rp.chain:
            return
        live = list(self.rps.values())
        anc = sorted((x for x in live if x is not rp and x.state == "ok" and rp.extends(x, hbs)),
                     key=lambda x: x.n_tokens, reverse=True)
        cands = anc[keep - 1:]
        if not cands:
            return
        mine = {id(x) for x in anc} | {id(rp)}
        others = [y for y in live if id(y) not in mine and y.chain]
        for x in cands:
            if self._rp_in_use(x):
                self.stats["rp_prune_skipped_busy"] += 1
                continue
            needed = False
            for y in others:
                if not y.extends(x, hbs):
                    continue
                between = sum(1 for z in live if z is not y and z is not x and y.extends(z, hbs)
                              and z.n_tokens > x.n_tokens)
                if between < keep - 1:
                    needed = True
                    break
            if needed:
                self.stats["rp_prune_kept_branch"] += 1
                continue
            freed = self._drop_rp(x, "pruned")
            self.stats["rp_pruned"] += 1
            self.stats["prune_freed_bytes"] += freed
            # still a conversation seen before for SAVE_POLICY=repeat (a later edit that branches below the kept RPs)
            self.seen[x.key] = None
            self.seen.move_to_end(x.key)
            while len(self.seen) > 4096:
                self.seen.popitem(last=False)
            if self.cfg.log:
                logger.info("%s prune rp n=%d (superseded by n=%d) freed=%d disk=%.2f GiB rps=%d", TAG, x.n_tokens,
                            rp.n_tokens, freed, self.disk_bytes / (1 << 30), len(self.rps))

    def _delete_disk(self, db: _DiskBlock) -> None:
        self.disk.pop(db.fname, None)
        self.disk_bytes -= db.nbytes
        self._meta.deletes.append(db.fname)

    def _make_room(self, nbytes: int, keep: set[bytes]) -> bool:
        """Evict least recently used RPs until ``nbytes`` more fit under the cap. The caller has already referenced the
        files the new RP shares (the common head, earlier turns), so an evicted RP that shares them only loses its own
        files. The eviction is planned on a copy of the reference counts first and done only if it makes room (an
        unfit save evicts nothing). (Before 2026-10-13 an RP sharing any file with the new RP was never evicted; every
        DSH conversation shares the head blocks, so at the cap DSH saves were skipped instead: rp_skipped_cap.)"""
        cap = self.cfg.cap_bytes
        if nbytes > cap:
            return False
        if self.disk_bytes + nbytes <= cap:
            return True
        refs: dict[str, int] = {}
        gone: set[str] = set()
        victims: list[_RP] = []
        freed = 0
        # eviction order (2026-10-14): expired -> no TTL (server default) -> shorter TTL -> least recently used
        now = self.wall()

        def order(item):
            i, rp = item
            if rp.stale_at or self._is_expired(rp, now):
                return (0, 0, i)
            if rp.ttl_s is None:
                return (1, 0, i)
            return (2, rp.ttl_s, i)

        for _i, rp in sorted(enumerate(self.rps.values()), key=order):
            if self.disk_bytes - freed + nbytes <= cap:
                break
            if rp.key in keep or self._rp_in_use(rp):
                continue
            victims.append(rp)
            for e in rp.blocks:
                db = self.disk.get(e.fname)
                if db is None or e.fname in gone:
                    continue
                refs[e.fname] = refs.get(e.fname, db.refs) - 1
                if refs[e.fname] <= 0 and db.busy == 0 and db.state in ("ok", "bad"):
                    gone.add(e.fname)
                    freed += db.nbytes
        if self.disk_bytes - freed + nbytes > cap:
            return False
        for rp in victims:
            self._drop_rp(rp, "lru")
        return self.disk_bytes + nbytes <= cap

    def save_request(self, request: "Request") -> None:
        if not self.ready or self.breaker is not None or self.persist_state == "adopting":
            return
        p = self._params(request)
        if request.num_computed_tokens < self._min_tokens(p):
            return
        self._maybe_save(request.block_hashes, request.num_tokens, p)

    def _extends_known(self, block_hashes, n: int) -> bool:
        """True when the prompt up to n extends the RP of an earlier finished request (a later turn of a
        conversation) or a saved RP. One-shot requests never repeat, so they are never written."""
        hbs = self.hash_block_size
        m = n - hbs
        while m >= self.cfg.min_tokens:
            k = bytes(block_hashes[m // hbs - 1])
            if k in self.seen or k in self.rps:
                return True
            m -= hbs
        return False

    def _maybe_save(self, block_hashes, num_tokens: int, params: SpillParams | None = None) -> None:
        if self.breaker is not None or self.persist_state == "adopting":
            return
        if params is not None and params.ttl_s == 0:
            self.stats["rp_skipped_ttl0"] += 1      # the session asked not to keep anything
            return
        stale = False
        if params is not None and params.session is not None and params.epoch is not None:
            self._note_epoch(params.session, params.epoch)
            # a request of an older epoch than the session's newest (still in flight when the compaction was seen):
            # saved, but stale at once -- usable by other late requests of that epoch, gone after EPOCH_GRACE_S unused
            stale = params.epoch < self.session_epoch.get(params.session, params.epoch)
        plan = self._plan_from(block_hashes, num_tokens, self._min_tokens(params))
        if plan is None:
            self.stats["rp_skipped_short"] += 1
            return
        rp_key, n, entries = plan
        chain = tuple(bytes(h) for h in block_hashes[:n // self.hash_block_size])
        if self.cfg.save_policy == "repeat" and rp_key not in self.rps and not self._extends_known(block_hashes, n):
            if params is not None and params.explicit_keep and self.cfg.ttl_save_policy == "all":
                # an explicitly kept session (ttl_s > 0): its first turn is written too (TTL_SAVE_POLICY=all)
                self.stats["rp_saved_ttl_first"] += 1
            else:
                self.seen[rp_key] = None
                self.seen.move_to_end(rp_key)
                while len(self.seen) > 4096:
                    self.seen.popitem(last=False)
                self.stats["rp_skipped_first"] += 1
                self._index_dirty = True
                return
        self._save_plan(rp_key, n, entries, chain=chain, params=params, stale=stale)

    def _save_plan(self, rp_key: bytes, n: int, entries: list[tuple[_RPBlock, Any]],
                   chain: tuple[bytes, ...] = (), params: SpillParams | None = None, stale: bool = False) -> None:
        old = self.rps.get(rp_key)
        if old is not None and old.drop_reason and not self._rp_in_use(old):
            # marked for deletion (ttl 0 / an older epoch) while it was in use, now asked for again: start over
            self._drop_rp(old, old.drop_reason)
            old = None
        if old is not None:
            if old.drop_reason:
                pass                    # still in use: dropped by the sweep once free
            elif self._is_expired(old) and not (params is not None and params.explicit_keep):
                pass                    # timed out and nobody asks to keep it: the sweep drops it
            else:
                # a request of the conversation saves the same resume point again: it is used (sliding TTL), even
                # if its time had run out meanwhile (the files are still there: revive rather than rewrite)
                old.last_used = self.wall()
                if params is not None and params.explicit_keep and (old.session is None
                                                                     or old.session == params.session):
                    old.ttl_s = params.ttl_s
                self._index_dirty = True
            self.rps.move_to_end(rp_key)
            return
        to_copy: list[tuple[_RPBlock, Any]] = []
        seen: set[str] = set()
        for e, b in entries:
            if e.fname in seen:
                continue
            seen.add(e.fname)
            if e.fname not in self.disk:
                to_copy.append((e, b))
            else:
                self.stats["rp_dedup_blocks"] += 1
        nbytes = sum(self.file_bytes[e.group] for e, _ in to_copy)
        if not self._rate_ok(nbytes):
            self.stats["rp_skipped_rate"] += 1
            return
        # Pin cap (review B2): count only blocks nobody but the connector's save jobs holds (a running request's own
        # blocks cost nothing extra); new pins on cached-free blocks become exclusive at once. Restores are not gated
        # here: their pins are bounded by PIN_TIMEOUT_S / MAX_STATE_PINS / the load deadline, and they replace a
        # re-read that would take the same blocks anyway.
        if to_copy and self.save_pinned_exclusive() + sum(1 for _, b in to_copy if b.ref_cnt == 0) > self.max_pinned:
            self.stats["rp_skipped_pins"] += 1
            return
        # reference the files this RP dedups against BEFORE making room, so LRU eviction can never delete them
        pre = [i for i, (e, _) in enumerate(entries) if e.fname in self.disk]
        for i in pre:
            self.disk[entries[i][0].fname].refs += 1
        if not self._make_room(nbytes, keep={rp_key}):
            for i in pre:
                self.disk[entries[i][0].fname].refs -= 1     # _make_room evicted nothing: the others still hold them
            self.stats["rp_skipped_cap"] += 1
            return
        rp = _RP(key=rp_key, n_tokens=n, blocks=[e for e, _ in entries], last_used=self.wall(), chain=chain,
                 ttl_s=params.ttl_s if params is not None else None,
                 session=params.session if params is not None else None,
                 epoch=params.epoch if params is not None else None)
        if stale:
            rp.stale_at = rp.last_used
            self.stats["rp_saved_stale"] += 1
        job = self._next_job() if to_copy else 0
        wid = f"{self.boot}-{job}"
        for e in rp.blocks:
            k = f"rp_blocks_g{e.group}"
            self.stats[k] = self.stats.get(k, 0) + 1
        pre_set = set(pre)
        for i, e in enumerate(rp.blocks):
            if i in pre_set:
                continue        # referenced above
            db = self.disk.get(e.fname)
            if db is None:
                db = _DiskBlock(e.fname, e.group, self.file_bytes[e.group], wid=wid)
                self.disk[e.fname] = db
                self.disk_bytes += db.nbytes
            db.refs += 1
        self.rps[rp_key] = rp
        if to_copy:
            pins = [b for _, b in to_copy]
            self.block_pool.touch(pins)
            self.save_jobs[job] = ([(e.group, b, e.fname) for e, b in to_copy], pins, rp)
            # Shipped with the NEXT build_meta (see build_meta): the worker records its "all prior device work" event
            # in that step's pre_forward, i.e. after this step's forward that may still write these blocks.
            self._meta.saves[job] = [(e.group, b.block_id, e.fname, wid) for e, b in to_copy]
            self._note_pinned()
            rp.pending_jobs.add(job)
            self.write_log.append((self.clock(), nbytes))
            self.stats["save_bytes"] += nbytes
        else:
            self._check_rp(rp)
        if self.cfg.log:
            logger.info("%s save rp n=%d blocks=%d new=%d bytes=%d disk=%.2f GiB rps=%d", TAG, n, len(rp.blocks),
                        len(to_copy), nbytes, self.disk_bytes / (1 << 30), len(self.rps))

    def _check_rp(self, rp: _RP) -> None:
        if rp.pending_jobs or rp.state == "ok":
            return
        if all(self.disk.get(e.fname) is not None and self.disk[e.fname].state == "ok" for e in rp.blocks):
            rp.state = "ok"
            self.stats["rp_saved"] += 1
            self._index_dirty = True
            self._prune_superseded(rp)

    def _finish_save(self, job: int, ok: bool) -> None:
        items, pins, rp = self.save_jobs.pop(job)
        self.job_sent.pop(job, None)
        if ok:
            # A block whose key moved away while it was being copied (mamba CoW of a running producer:
            # move_block_hashes, after which the owner writes the old block) may have been copied after it changed:
            # discard the whole job (conservative, the RP is simply not saved).
            for e, b in zip(rp.blocks_by_fname(items), pins):
                if e is None or b.block_hash != e.keys[0]:
                    ok = False
                    self.stats["save_moved_blocks"] += 1
                    break
        self.block_pool.free_blocks(pins)
        rp.pending_jobs.discard(job)
        for _g, _b, fname in items:
            db = self.disk.get(fname)
            if db is None:
                continue
            if ok:
                db.state = "ok"
                if db.refs <= 0 and db.busy == 0:
                    # its RP was dropped while the copy ran (bad file, breaker): nobody references the new file
                    self.stats["orphan_files_deleted"] += 1
                    self._delete_disk(db)
        if not ok:
            self.stats["rp_save_failed"] += 1
            # forget the RP and every block it brought (files already removed by the worker on failure)
            for _g, _b, fname in items:
                db = self.disk.get(fname)
                if db is not None and db.state != "ok":
                    self.disk.pop(fname, None)
                    self.disk_bytes -= db.nbytes
                    self._meta.deletes.append(fname)
            self._forget_saving(rp)
            # other RPs still saving that counted on these files (dedup against a pending file) can never complete
            for other in list(self.rps.values()):
                if other.state == "saving" and any(e.fname not in self.disk for e in other.blocks):
                    self._forget_saving(other)
            return
        # Other RPs waiting on the same files
        for other in list(self.rps.values()):
            if other.state == "saving":
                self._check_rp(other)

    def _forget_saving(self, rp: _RP) -> None:
        if self.rps.get(rp.key) is not rp:
            return
        del self.rps[rp.key]
        for e in rp.blocks:
            db = self.disk.get(e.fname)
            if db is not None:
                db.refs -= 1
                if db.refs <= 0 and db.busy == 0 and db.state == "ok":
                    self._delete_disk(db)

    # -- sweep below/above high-water ------------------------------------------------------------------------------
    def note_finished_below_highwater(self, request: "Request") -> None:
        # Remember what is needed to re-plan the RP later from the cache alone (the request object goes away).
        self.deferred[request.request_id] = (list(request.block_hashes), request.num_tokens, self._params(request))
        while len(self.deferred) > 64:
            self.deferred.popitem(last=False)

    def sweep(self) -> None:
        """Past the high-water mark, write the oldest deferred RP (the next eviction candidate) if its blocks are
        all still cached; the plan is re-derived from the cache, so a partly evicted RP simply comes out shallower
        or is skipped (min_tokens)."""
        if not self.deferred or self.occupancy() < self.cfg.highwater:
            return
        _rid, (hashes, num_tokens, params) = self.deferred.popitem(last=False)
        before = self.stats["rp_saved"] + len(self.save_jobs)
        self._maybe_save(hashes, num_tokens, params)
        if self.stats["rp_saved"] + len(self.save_jobs) != before:
            self.stats["deferred_written"] += 1

    # -- restore ----------------------------------------------------------------------------------------------------
    def _find_rp(self, request: "Request", num_local: int) -> _RP | None:
        hbs = self.hash_block_size
        bh = request.block_hashes
        top = min(request.num_tokens - 1, len(bh) * hbs)
        top -= top % hbs
        floor = num_local + self.cfg.min_gain
        n = top
        now = None
        while n >= floor and n > 0:
            rp = self.rps.get(bytes(bh[n // hbs - 1]))
            if rp is not None and rp.state == "ok" and rp.n_tokens == n:
                if rp.ttl_s is not None:
                    now = self.wall() if now is None else now
                    if self._is_expired(rp, now):
                        n -= hbs
                        continue
                return rp
            n -= hbs
        return None

    def get_num_new_matched_tokens(self, request: "Request", num_local: int) -> tuple[int | None, bool]:
        if not self.ready:
            return 0, False
        if self.persist_state == "adopting":
            # an index of the previous run exists and the start-up decision is a step or two away: let the request wait
            # (bounded by ADOPT_DEADLINE_S) rather than admit it with a re-read its adopted resume point would avoid
            return (None, False) if self._index is not None and self.breaker is None else (0, False)
        rid = request.request_id
        if rid in self.no_restore:
            return 0, False
        p = self._note_request(request)
        if p is not None and p.ttl_s == 0:
            return 0, False         # the session asked not to keep anything: nothing is restored for it either
        st = self.restores.get(rid)
        if st is not None:
            if st.state == "loading":
                return None, False      # bounded: the load deadline detaches the request (no_restore)
            if st.state == "ready":
                if num_local >= st.rp.n_tokens:
                    return 0, False
                if not st.main_released:
                    # The engine's own lookup does not reach the restored boundary although every block is pinned
                    # (should not happen: the keys are the ones the lookup found when the RP was taken).
                    self.stats["restore_short"] += 1
                    logger.warning("%s restore short: req=%s local=%d rp=%d", TAG, rid, num_local, st.rp.n_tokens)
                    self._release(st)
                    st.state = "done"
                    return 0, False
                # waited past the pin timeout and lost some full-attention blocks: release the kept state pins and
                # restore again below (only the missing blocks are read)
                self._release(st)
                st.state = "done"
            if st.failures >= self.cfg.max_restore_failures or st.restores >= self.cfg.max_restores_per_request:
                return 0, False
        if self.breaker is not None or getattr(request, "skip_reading_prefix_cache", False):
            return 0, False
        rp = self._find_rp(request, num_local)
        if rp is None:
            return 0, False
        inflight = self.rp_loading.get(rp.key)
        if inflight is not None:
            # another request is already bringing this RP back: wait for it instead of reading the same blocks again
            # (bounded: the in-flight load finishes or hits its deadline, which removes the entry)
            self.stats["restore_coalesced_waits"] += 1
            return None, False
        res = self._start_restore(request, rp, num_local, prev=st)
        if res == "started":
            self.space_wait.pop(rid, None)
            return None, False
        if res == "no_space":
            # Admitting now would re-read rp.n_tokens - num_local tokens; waiting a little for blocks to free up
            # (running requests finish / decode) is far cheaper. Bounded, then the normal path.
            t_first = self.space_wait.setdefault(rid, self.clock())
            if self.clock() - t_first < self.cfg.space_wait_s:
                self.stats["restore_space_wait_steps"] += 1
                return None, False
            if rid not in self.gave_up:
                self.gave_up.add(rid)
                self.stats["restore_space_giveup"] += 1
        return 0, False

    def _start_restore(self, request: "Request", rp: _RP, num_local: int, prev: _Restore | None) -> str:
        pool = self.block_pool
        pins: list[Any] = []
        need: list[_RPBlock] = []
        pinned_free = 0
        for e in rp.blocks:
            existing = pool.cached_block_hash_to_block.get_one_block(e.keys[0])
            if existing is not None:
                pins.append(existing)
                pinned_free += existing.ref_cnt == 0
            else:
                need.append(e)
        for e in need:
            db = self.disk.get(e.fname)
            if db is None or db.state != "ok":
                return "unusable"
        if pool.get_num_free_blocks() - pinned_free - len(need) < self.cfg.reserve_blocks:
            self.stats["restore_no_space"] += 1
            return "no_space"
        pool.touch(pins)
        new_blocks = pool.get_new_blocks(len(need)) if need else []
        job = self._next_job()
        st = _Restore(rp=rp, job_id=job, local_before=num_local, new_blocks=new_blocks, pins=pins, entries=need,
                      t0=self.clock(), failures=prev.failures if prev else 0,
                      restores=(prev.restores + 1) if prev else 1)
        for e in need:
            self.disk[e.fname].busy += 1
        rp.last_used = self.wall()
        self._index_dirty = True
        self.rps.move_to_end(rp.key)
        self.restores[request.request_id] = st
        self.restore_by_job[job] = request.request_id
        self.stats["restore_started"] += 1
        self.stats["restore_loaded_blocks"] += len(need)
        self.stats["restore_pinned_blocks"] += len(pins)
        self._note_pinned()
        if need:
            self._meta.loads[job] = [(e.group, b.block_id, e.fname, self.disk[e.fname].wid or None)
                                     for e, b in zip(need, new_blocks)]
            self.rp_loading[rp.key] = (job, self.clock())
        else:
            self._finish_load(job, True)
        if self.cfg.log:
            logger.info("%s restore start req=%s local=%d -> rp=%d load=%d pinned=%d", TAG, request.request_id,
                        num_local, rp.n_tokens, len(need), len(pins))
        return "started"

    def _finish_load(self, job: int, ok: bool) -> None:
        self.job_sent.pop(job, None)
        for k in [k for k, v in self.rp_loading.items() if v[0] == job]:
            del self.rp_loading[k]
        rid = self.restore_by_job.pop(job, None)
        st = self.restores.get(rid) if rid is not None else None
        if st is None or st.job_id != job:
            st = self.orphan_restores.pop(job, None)
            if st is None:
                return
        for e in st.entries:
            db = self.disk.get(e.fname)
            if db is not None:
                db.busy -= 1
                if db.busy == 0 and db.refs <= 0 and db.state in ("ok", "bad"):
                    self._delete_disk(db)
        pool = self.block_pool
        if ok:
            keep_new: list[Any] = []
            for e, nb in zip(st.entries, st.new_blocks):
                existing = pool.cached_block_hash_to_block.get_one_block(e.keys[0])
                if existing is not None:
                    # computed meanwhile by someone else: use theirs, drop our copy (unhashed -> plain free)
                    pool.touch([existing])
                    st.pins.append(existing)
                    pool.free_blocks([nb])
                    continue
                for i, k in enumerate(e.keys):
                    pool._insert_block_hash(k, nb, e.num_tokens if i == 0 else None)
                keep_new.append(nb)
            st.new_blocks = keep_new
            st.state = "ready"
            st.t_ready = self.clock()
            self.stats["restore_ok"] += 1
            self.stats["restored_tokens"] += max(0, st.rp.n_tokens - st.local_before)
            if self.cfg.log:
                logger.info("%s restore ready req=%s rp=%d in %.3f s", TAG, rid, st.rp.n_tokens, st.t_ready - st.t0)
        else:
            self.stats["restore_failed"] += 1
            self.stats["restore_fallback"] += 1
            st.state = "failed"
            st.failures += 1
            self._release(st)
            # a failed read means a bad file on some rank: forget the RP
            self._drop_rp(st.rp, "bad")
            logger.warning("%s restore failed req=%s rp=%d: falling back to prefill", TAG, rid, st.rp.n_tokens)
        if st.orphan:
            self._release(st)
            st.state = "done"

    def _block_group(self, b) -> int:
        return int.from_bytes(bytes(b.block_hash)[-4:], "big") if b.block_hash is not None else -1

    def _release(self, st: _Restore, main_only: bool = False) -> None:
        if main_only:
            blocks = [b for b in list(st.pins) + list(st.new_blocks) if self._block_group(b) in self.main_groups]
            ids = {id(b) for b in blocks}
            st.pins = [b for b in st.pins if id(b) not in ids]
            st.new_blocks = [b for b in st.new_blocks if id(b) not in ids]
        else:
            blocks = list(st.pins) + list(st.new_blocks)
            st.pins, st.new_blocks = [], []
        if blocks:
            self.block_pool.free_blocks(blocks)

    def update_state_after_alloc(self, request: "Request") -> None:
        self.tracked[request.request_id] = request
        p = self._params(request)
        if p is not None and p.session is not None and p.epoch is not None:
            self._note_epoch(p.session, p.epoch)      # admitted: the session now runs this epoch
        st = self.restores.get(request.request_id)
        if st is not None and st.state == "ready":
            self.stats["restore_admitted"] += 1
            self._release(st)
            st.state = "done"

    def request_finished(self, request: "Request") -> None:
        self.no_restore.discard(request.request_id)
        st = self.restores.pop(request.request_id, None)
        if st is not None:
            if st.state == "loading":
                st.orphan = True
                self.orphan_restores[st.job_id] = st
                self.restore_by_job.pop(st.job_id, None)
            elif st.state == "ready":
                self._release(st)
        rid = request.request_id
        self.space_wait.pop(rid, None)
        self.gave_up.discard(rid)
        self.tracked.pop(rid, None)
        self.prefill_noted.discard(rid)
        self.req_noted.discard(rid)
        p = self._params(request)
        self.req_params.pop(rid, None)
        if rid in self.saved_once:
            self.saved_once.discard(rid)
            return
        if request.num_computed_tokens < self._min_tokens(p) or not self.ready:
            self.deferred.pop(rid, None)
            return
        self.req_params[rid] = p        # _save_or_defer reads it again
        try:
            self._save_or_defer(request)
        finally:
            self.req_params.pop(rid, None)

    def occupancy(self) -> float:
        """Fraction of the pool that holds something (running or cached): 1 - never-hashed free blocks / pool. Past
        the high-water mark the next allocations evict cached prefixes. (BlockPool.get_usage counts cached free
        blocks as free and stays low on a saturated prefix cache.)"""
        if self.cfg.highwater <= 0:
            return 1.0
        q = self.block_pool.free_block_queue
        node = q.fake_free_list_head.next_free_block
        empty = 0
        n = 0
        while node is not None and node is not q.fake_free_list_tail and n <= self.block_pool.num_gpu_blocks:
            if node.block_hash is None:
                empty += 1
            node = node.next_free_block
            n += 1
        return 1.0 - empty / max(1, self.block_pool.num_gpu_blocks - 1)

    def _save_or_defer(self, request: "Request") -> None:
        if self.occupancy() >= self.cfg.highwater:
            self.deferred.pop(request.request_id, None)
            self.save_request(request)
        else:
            self.note_finished_below_highwater(request)

    def _scan_prefill_done(self) -> None:
        """Save the RP of requests whose last prompt chunk was scheduled in an earlier step. build_meta runs inside
        schedule() before _update_after_schedule, so a request is noted here one step after its last chunk was
        scheduled and saved one step later; the copy itself is shipped with the step after that (build_meta)."""
        if self.persist_state == "adopting":
            return      # keep prefill_noted: those resume points are taken once the start-up decision is made
        ready = [rid for rid in self.prefill_noted if rid in self.tracked]
        self.prefill_noted = set()
        for rid in ready:
            req = self.tracked.get(rid)
            if req is None or rid in self.saved_once:
                continue
            self.saved_once.add(rid)
            if req.num_computed_tokens >= self._min_tokens(self._params(req)) and self.breaker is None:
                self._save_or_defer(req)
        for rid, req in self.tracked.items():
            if rid in self.saved_once:
                continue
            if (req.num_prompt_tokens >= self._min_tokens(self._params(req))
                    and req.num_computed_tokens >= req.num_prompt_tokens):
                self.prefill_noted.add(rid)

    def _forget_bad_files(self, names) -> None:
        for fname in names:
            db = self.disk.get(fname)
            if db is None:
                continue
            db.state = "bad"
            self.stats["bad_files"] += 1
            for rp in [r for r in self.rps.values() if any(e.fname == fname for e in r.blocks)]:
                self._drop_rp(rp, "bad")

    def update_connector_output(self, meta: SpillWorkerMeta | None) -> None:
        if meta is None:
            return
        hello = getattr(meta, "hello", None)
        if hello:
            self._note_hello(hello)
        exc = int(getattr(meta, "exc", 0) or 0)
        if exc:
            self.stats["worker_exc"] += exc
            self._trip(f"{exc} unexpected exception(s) on a worker I/O thread")
        if meta.bad:
            self._forget_bad_files(meta.bad)
        for job, (ok, bad) in meta.done.items():
            acc = self.job_acc.setdefault(job, [0, 0])
            acc[0] += ok
            acc[1] += bad
            if acc[0] + acc[1] < self.world:
                continue
            del self.job_acc[job]
            all_ok = acc[1] == 0
            if job in self.quarantine:
                self._finish_quarantine(job, all_ok)
                continue
            if job in self.expired:
                self.expired.discard(job)
                self.stats["late_answers"] += 1
                continue
            known = job in self.save_jobs or job in self.restore_by_job or job in self.orphan_restores
            if known:
                if all_ok:
                    self.consec_fail = 0
                else:
                    self.stats["jobs_failed"] += 1
                    self.consec_fail += 1
            if job in self.save_jobs:
                self._finish_save(job, all_ok)
            else:
                self._finish_load(job, all_ok)
            if self.consec_fail >= self.cfg.breaker_fails:
                self._trip(f"{self.consec_fail} consecutive failed spill jobs")

    # -- fault containment ----------------------------------------------------------------------------------------
    def _note_pinned(self) -> None:
        n = self.pinned_blocks()
        if n > self.stats["pinned_peak"]:
            self.stats["pinned_peak"] = n
        x = self.save_pinned_exclusive()
        if x > self.stats["pinned_excl_peak"]:
            self.stats["pinned_excl_peak"] = x

    def save_pinned_exclusive(self) -> int:
        """Blocks held ONLY by the connector's save pins (ref_cnt == our pins on it): what a stuck rank would hold."""
        cnt: dict[int, list] = {}
        for _, pins, _ in self.save_jobs.values():
            for b in pins:
                e = cnt.setdefault(id(b), [b, 0])
                e[1] += 1
        return sum(1 for b, n in cnt.values() if b.ref_cnt <= n)

    def _trip(self, reason: str) -> None:
        """Open the breaker: no new save or restore on any rank (every decision is made here, once, so both ranks
        agree), unsent saves are dropped, every resume point is forgotten (its files are deleted once nothing reads
        them). Jobs already on the workers finish or hit their deadline. Latched until the engine restarts."""
        if self.breaker is not None:
            return
        self.breaker = reason
        self.stats["breaker_trips"] += 1
        logger.error("%s BREAKER OPEN: %s -> the spill tier is disabled on every rank until restart "
                     "(requests take the normal prefill path)", TAG, reason)
        for job in [j for j in self._meta.saves if j in self.save_jobs]:
            del self._meta.saves[job]
            self._finish_save(job, False)
        for rp in list(self.rps.values()):
            if rp.state == "ok":
                self._drop_rp(rp, "breaker")
            else:
                self._forget_saving(rp)
        self.deferred.clear()
        self.prefill_noted.clear()
        self._index_dirty = self._index_urgent = True     # the index is rewritten empty: the next start wipes

    def _check_deadlines(self, t: float) -> None:
        for job in [j for j in self.save_jobs if j in self.job_sent]:
            age = t - self.job_sent[job]
            if age < self.cfg.save_deadline_s:
                continue
            acc = self.job_acc.get(job, [0, 0])
            self.stats["save_timeouts"] += 1
            self.expired.add(job)
            # Release the pins: the workers only READ these blocks, so a late copy can at worst produce a garbage
            # file -- and the deletes _finish_save queues reach each rank's writer FIFO after that job.
            self._finish_save(job, False)
            logger.error("%s save job %d unanswered after %.1f s (ok=%d fail=%d of %d ranks): pins released",
                         TAG, job, age, acc[0], acc[1], self.world)
            self._trip(f"save job {job} unanswered after {age:.0f} s (answers ok={acc[0]} fail={acc[1]} "
                       f"of {self.world})")
        loading = [(rid, st) for rid, st in self.restores.items() if st.state == "loading"]
        loading += [(None, st) for st in self.orphan_restores.values()]
        for rid, st in loading:
            sent = self.job_sent.get(st.job_id)
            if sent is None or t - sent < self.cfg.load_deadline_s:
                continue
            self._expire_load(rid, st, t - sent)

    def _expire_load(self, rid: str | None, st: _Restore, age: float) -> None:
        job = st.job_id
        acc = self.job_acc.get(job, [0, 0])
        self.stats["load_timeouts"] += 1
        self.job_sent.pop(job, None)
        for k in [k for k, v in self.rp_loading.items() if v[0] == job]:
            del self.rp_loading[k]
        self.restore_by_job.pop(job, None)
        self.orphan_restores.pop(job, None)
        # Blocks that were already cached are never written by a load: unpin them now. The fresh blocks may still be
        # written by a rank whose read has not returned: they stay allocated (quarantined) until every rank answers.
        if st.pins:
            self.block_pool.free_blocks(st.pins)
            st.pins = []
        st.orphan = True
        st.state = "failed"
        self.quarantine[job] = st
        if rid is not None:
            self.restores.pop(rid, None)
            self.no_restore.add(rid)
            self.stats["restore_fallback"] += 1
        logger.error("%s load job %d unanswered after %.1f s (ok=%d fail=%d of %d ranks): req=%s -> normal prefill, "
                     "%d blocks quarantined", TAG, job, age, acc[0], acc[1], self.world, rid, len(st.new_blocks))
        self._trip(f"load job {job} unanswered after {age:.0f} s (answers ok={acc[0]} fail={acc[1]} of {self.world})")

    def _finish_quarantine(self, job: int, ok: bool) -> None:
        st = self.quarantine.pop(job)
        self.stats["late_answers"] += 1
        for e in st.entries:
            db = self.disk.get(e.fname)
            if db is not None:
                db.busy -= 1
                if db.busy == 0 and db.refs <= 0 and db.state in ("ok", "bad"):
                    self._delete_disk(db)
        # every rank has stopped writing them: plain free (no key, whatever the outcome)
        self.stats["quarantine_freed"] += len(st.new_blocks)
        self._release(st)
        logger.warning("%s load job %d answered late (ok=%s): %d quarantined blocks freed", TAG, job, ok,
                       self.stats["quarantine_freed"])

    def stats_line(self) -> str:
        t = self.io_clock()
        save_ages = [t - self.job_sent[j] for j in self.save_jobs if j in self.job_sent]
        load_jobs = [st.job_id for st in self.restores.values() if st.state == "loading"]
        load_jobs += list(self.orphan_restores)
        load_ages = [t - self.job_sent[j] for j in load_jobs if j in self.job_sent]
        p_save = sum(len(p) for _, p, _ in self.save_jobs.values())
        p_rest = sum(len(st.pins) + len(st.new_blocks)
                     for st in list(self.restores.values()) + list(self.orphan_restores.values()))
        p_quar = sum(len(st.new_blocks) for st in self.quarantine.values())
        now = self.clock()
        w_1h = sum(b for ts, b in self.write_log if now - ts <= 3600)
        s = self.stats
        return (f"{TAG} stats breaker={'closed' if self.breaker is None else 'OPEN'} "
                f"pinned={p_save + p_rest + p_quar} save={p_save} save_excl={self.save_pinned_exclusive()} "
                f"restore={p_rest} quarantined={p_quar} peak={s['pinned_peak']} excl_peak={s['pinned_excl_peak']} "
                f"cap={self.max_pinned} "
                f"inflight_save={len(save_ages)} oldest_save_s={max(save_ages, default=0.0):.1f} "
                f"inflight_load={len(load_ages)} oldest_load_s={max(load_ages, default=0.0):.1f} "
                f"worker_exc={s['worker_exc']} jobs_failed={s['jobs_failed']} "
                f"save_timeouts={s['save_timeouts']} load_timeouts={s['load_timeouts']} "
                f"written_gib_1h={w_1h / (1 << 30):.2f} disk_gib={self.disk_bytes / (1 << 30):.2f} rps={len(self.rps)} "
                f"restores_ok={s['restore_ok']} restores_failed={s['restore_failed']} "
                f"restores_fallback={s['restore_fallback']} restore_short={s['restore_short']} "
                f"saves_ok={s['rp_saved']} saves_failed={s['rp_save_failed']} "
                # appended 2026-10-13 (superseded-RP pruning); fields above keep their order for monitor.sh
                f"rp_pruned={s['rp_pruned']} pruned_gib={s['prune_freed_bytes'] / (1 << 30):.2f} "
                f"rp_evicted={s['rp_evicted']} keep={self.cfg.keep_per_chain} skipped_cap={s['rp_skipped_cap']} "
                # appended 2026-10-14 (TTL + reuse across restarts); fields above keep their order for monitor.sh
                f"expired={s['rp_expired']} epoch_dropped={s['rp_epoch_dropped']} ttl0_skipped={s['rp_skipped_ttl0']} "
                f"{self._ttl_classes()} persist={self.persist_state} adopted={s['persist_adopted_rps']} "
                f"adopted_gib={s['persist_adopted_bytes'] / (1 << 30):.2f} "
                f"index_age_s={(self.wall() - self._index_written_wall) if self._index_written_wall else -1:.0f}"
                + (f" reason={self.breaker!r}" if self.breaker is not None else ""))

    def _ttl_classes(self) -> str:
        """RP counts and the GiB of the files they reference, by TTL class: none (server default) / <=1h / <=1d / >1d /
        expired-pending (a file shared by classes counts in each)."""
        names = ("none", "1h", "1d", "3d", "exp")
        cnt = [0] * 5
        files: list[set[str]] = [set() for _ in range(5)]
        now = self.wall()
        for rp in self.rps.values():
            if self._is_expired(rp, now):
                c = 4
            elif rp.ttl_s is None:
                c = 0
            else:
                c = 1 if rp.ttl_s <= 3600 else 2 if rp.ttl_s <= 86400 else 3
            cnt[c] += 1
            files[c].update(e.fname for e in rp.blocks)
        gib = [sum(self.disk[f].nbytes for f in fs if f in self.disk) / (1 << 30) for fs in files]
        return (f"ttl_rps={'/'.join(str(c) for c in cnt)} ttl_gib={'/'.join(f'{g:.2f}' for g in gib)} "
                f"ttl_classes={'/'.join(names)}")

    def _maybe_log(self) -> None:
        if self.cfg.log_every_s <= 0:
            return
        t = self.io_clock()
        if t - self._last_log < self.cfg.log_every_s:
            return
        self._last_log = t
        if self.breaker is not None:
            logger.warning("%s", self.stats_line())
        else:
            logger.info("%s", self.stats_line())

    # -- reuse across restarts ----------------------------------------------------------------------------------
    def _compute_ident(self) -> None:
        t0 = time.perf_counter()
        try:
            parts = _identity(self.vllm_config, self.kv_cache_config)
        except Exception as e:  # noqa: BLE001
            parts = {"error": f"{type(e).__name__}"}
        parts["file_bytes"] = _h(self.file_bytes)
        parts["world"] = str(self.world)
        self._ident_s = time.perf_counter() - t0
        self._ident = parts

    def _check_hash(self) -> tuple[bool, str] | None:
        """Are this process's prefix-cache block hashes reproducible by the next process? vLLM chains every block hash
        from kv_cache_utils.NONE_HASH, which init_none_hash() draws from os.urandom(32) unless PYTHONHASHSEED is set
        (then hash_fn(PYTHONHASHSEED)). The connector sets PYTHONHASHSEED (if unset) in the scheduler process before
        EngineCore calls init_none_hash; here we verify the outcome. None = not seeded yet (ask again next step)."""
        try:
            from vllm.v1.core import kv_cache_utils as kcu
            nh = getattr(kcu, "NONE_HASH", None)
            if nh is None:
                return None
            algo = str(getattr(getattr(self.vllm_config, "cache_config", None), "prefix_caching_hash_algo", "sha256"))
            from vllm.utils.hashing import get_hash_fn_by_name
            seed = os.environ.get("PYTHONHASHSEED")
            stable = seed is not None and bytes(nh) == bytes(get_hash_fn_by_name(algo)(seed))
            return stable, f"{algo}:{hashlib.sha256(bytes(nh)).hexdigest()[:16]}"
        except Exception as e:  # noqa: BLE001
            return False, f"err:{type(e).__name__}"

    def _note_hello(self, hello: dict) -> None:
        for r, rep in hello.items():
            self.hello[int(r)] = rep
        if self.persist_state == "adopting":
            self._persist_poll()

    def _sched_parts(self) -> dict[str, str]:
        parts = dict(self._ident or {})
        parts["hash"] = self.hash_info[1] if self.hash_info else "?"
        parts["hbs"] = str(self.hash_block_size)
        return parts

    def _persist_poll(self) -> None:
        """Start-up decision (scheduler only, once): adopt the previous run's resume points if the identity signatures
        of the scheduler and of EVERY rank equal the ones the index was written with; otherwise wipe every rank. No
        save and no restore happens before it (the decision leaves ahead of any job, in the same FIFO)."""
        t = self.io_clock()
        if self._adopt_t0 is None:
            self._adopt_t0 = t
        if self.breaker is not None:
            self._persist_wipe(f"breaker open before the start-up decision ({self.breaker})")
            return
        late = t - self._adopt_t0 >= self.cfg.adopt_deadline_s
        if self.hash_info is None:
            self.hash_info = self._check_hash()
            if self.hash_info is None:
                if late:
                    self._persist_wipe("block-hash seed never initialised", disable=True)
                return
        if not self.hash_info[0]:
            self._persist_wipe(f"block hashes are not reproducible across restarts ({self.hash_info[1]}, "
                               f"PYTHONHASHSEED={'set' if 'PYTHONHASHSEED' in os.environ else 'unset'})", disable=True)
            return
        idx = self._index
        if idx is None:
            self._persist_wipe(f"no usable index ({self._index_err})")
            return
        if self._ident is None:
            if late:
                self._persist_wipe("scheduler identity not computed in time")
            return
        mine = self._sched_parts()
        if idx.get("sched_sig") != _sig_of(mine):
            old = idx.get("sched_parts") or {}
            diff = sorted(k for k in set(old) | set(mine) if old.get(k) != mine.get(k))
            self._persist_wipe(f"scheduler signature differs: {','.join(diff) or '?'}")
            return
        if int(idx.get("world", -1)) != self.world:
            self._persist_wipe(f"world size {idx.get('world')} -> {self.world}")
            return
        if len(self.hello) < self.world:
            if late:
                missing = [r for r in range(self.world) if r not in self.hello]
                self._persist_wipe(f"no start-up report from rank(s) {missing} within {self.cfg.adopt_deadline_s:.0f} s")
            return
        want = idx.get("rank_sigs") or {}
        bad = []
        for r in range(self.world):
            rep = self.hello.get(r) or {}
            if rep.get("error") or not rep.get("sig") or want.get(str(r)) != rep.get("sig"):
                old = (idx.get("rank_parts") or {}).get(str(r)) or {}
                cur = rep.get("parts") or {}
                diff = sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k))
                bad.append(f"rank {r}: {rep.get('error') or ','.join(diff) or 'signature'}")
        if bad:
            self._persist_wipe("rank signature differs (" + "; ".join(bad) + ")")
            return
        self._adopt(idx)

    def _persist_wipe(self, reason: str, disable: bool = False) -> None:
        self._meta.wipe = True
        self.stats["persist_wipes"] += 1
        self.persist_reason = reason
        self.persist_state = "disabled" if disable else "on"
        self._index = None
        self._adopt_s = (self.io_clock() - self._adopt_t0) if self._adopt_t0 is not None else 0.0
        if disable:
            try:
                os.unlink(self.cfg.index_path())
            except OSError:
                pass
        else:
            self._index_dirty = self._index_urgent = True
        logger.info("%s persist: wipe every rank (%s)%s", TAG, reason,
                    " - reuse across restarts disabled for this run" if disable else "")

    def _adopt(self, idx: dict) -> None:
        t0 = time.perf_counter()
        now = self.wall()
        files_tbl = idx.get("files") or {}
        reports = [self.hello[r].get("files") or {} for r in range(self.world)]
        # each rank's own file size per group (its pages; the rank signature already pins its layout)
        sizes = [self.hello[r].get("file_bytes") or self.file_bytes for r in range(self.world)]
        ok_files: dict[str, tuple[int, str]] = {}
        for fname, rec in files_tbl.items():
            try:
                g, wid = int(rec[0]), str(rec[1])
                want = [sz[g] for sz in sizes]
            except (TypeError, ValueError, IndexError):
                continue
            if all(rep.get(fname) == want[r] for r, rep in enumerate(reports)):
                ok_files[fname] = (g, wid)
        adopted: list[_RP] = []
        for rec in idx.get("rps") or []:
            try:
                ttl, lu, ses, ep = rec["m"][:4]
                stale = float(rec["m"][4]) if len(rec["m"]) > 4 else 0.0
                x = rec["x"]
                rp = _RP(key=bytes.fromhex(x["k"]), n_tokens=int(x["n"]),
                         blocks=[_RPBlock(int(b[0]), tuple(bytes.fromhex(k) for k in b[1]),
                                          None if b[2] is None else int(b[2]), str(b[3])) for b in x["b"]],
                         state="ok", last_used=float(lu), chain=tuple(bytes.fromhex(h) for h in x["c"]),
                         ttl_s=None if ttl is None else int(ttl), session=ses, epoch=None if ep is None else int(ep))
                rp.ser = json.dumps(x, separators=(",", ":"))
                rp.stale_at = stale
            except (KeyError, TypeError, ValueError, IndexError):
                self.stats["persist_dropped_missing"] += 1
                continue
            if self._is_expired(rp, now):
                self.stats["persist_dropped_expired"] += 1
                continue
            if not rp.blocks or any(e.fname not in ok_files or ok_files[e.fname][0] != e.group for e in rp.blocks):
                self.stats["persist_dropped_missing"] += 1
                continue
            if rp.key in self.rps:
                continue
            self.rps[rp.key] = rp
            adopted.append(rp)
            for e in rp.blocks:
                db = self.disk.get(e.fname)
                if db is None:
                    db = _DiskBlock(e.fname, e.group, self.file_bytes[e.group], state="ok", wid=ok_files[e.fname][1])
                    self.disk[e.fname] = db
                    self.disk_bytes += db.nbytes
                db.refs += 1
        for h in idx.get("seen") or []:
            try:
                self.seen[bytes.fromhex(h)] = None
            except ValueError:
                pass
        while len(self.seen) > 4096:
            self.seen.popitem(last=False)
        for ses, ep in (idx.get("epochs") or {}).items():
            if isinstance(ses, str) and type(ep) is int:
                self.session_epoch[ses] = max(ep, self.session_epoch.get(ses, ep))
        # every block file a rank holds that no adopted RP references is deleted on that rank only
        orphans = {}
        for r, rep in enumerate(reports):
            extra = sorted(f for f in rep if f not in self.disk)
            if extra:
                orphans[r] = extra
                self.stats["persist_orphans"] += len(extra)
        self._meta.rank_deletes = orphans
        self.persist_state = "on"
        self._index = None
        self.stats["persist_adopted_rps"] = len(adopted)
        self.stats["persist_adopted_bytes"] = self.disk_bytes
        if self.disk_bytes > self.cfg.cap_bytes:
            self._make_room(0, keep=set())      # the cap was lowered: evict in the usual order
        self._adopt_s = (self.io_clock() - self._adopt_t0) if self._adopt_t0 is not None else 0.0
        self._index_dirty = self._index_urgent = True
        self.persist_reason = "adopted"
        logger.info("%s persist: adopted %d resume points (%d files, %.2f GiB/rank) from the previous run; dropped "
                    "%d expired + %d incomplete; orphan files deleted per rank %s; decision %.2f s after the first "
                    "step (adopt %.3f s)", TAG, len(adopted), len(self.disk), self.disk_bytes / (1 << 30),
                    self.stats["persist_dropped_expired"], self.stats["persist_dropped_missing"],
                    {r: len(v) for r, v in orphans.items()}, self._adopt_s, time.perf_counter() - t0)

    def _rp_ser(self, rp: _RP) -> str:
        if not rp.ser:
            rp.ser = json.dumps({"k": rp.key.hex(), "n": rp.n_tokens, "c": [h.hex() for h in rp.chain],
                                 "b": [[e.group, [k.hex() for k in e.keys], e.num_tokens, e.fname] for e in rp.blocks]},
                                separators=(",", ":"))
        return rp.ser

    def _index_payload(self) -> bytes | None:
        """The index: only resume points that are ok on every rank (all their files written and acknowledged), with
        the files' write ids, TTL / last_used / session / epoch; plus the 'seen' keys and session epochs. Written in LRU
        order (oldest first)."""
        if self._ident is None or self.hash_info is None or len(self.hello) < self.world:
            return None
        sched = self._sched_parts()
        ranks = {str(r): self.hello[r].get("sig") for r in range(self.world)}
        rparts = {str(r): self.hello[r].get("parts") or {} for r in range(self.world)}
        if any(not v or self.hello[int(r)].get("error") for r, v in ranks.items()):
            return None
        rps = [] if self.breaker is not None else [
            rp for rp in self.rps.values() if rp.state == "ok" and not self._is_expired(rp)
            and all(self.disk.get(e.fname) is not None and self.disk[e.fname].state == "ok" for e in rp.blocks)]
        files = {}
        for rp in rps:
            for e in rp.blocks:
                files[e.fname] = [e.group, self.disk[e.fname].wid]
        head = {"magic": INDEX_MAGIC, "version": INDEX_VERSION, "boot": self.boot, "written": self.wall(),
                "world": self.world, "sched_sig": _sig_of(sched), "sched_parts": sched, "rank_sigs": ranks,
                "rank_parts": rparts}
        body = (json.dumps(head, separators=(",", ":"))[:-1]
                + ',"files":' + json.dumps(files, separators=(",", ":"))
                + ',"rps":[' + ",".join('{"m":%s,"x":%s}' % (json.dumps([rp.ttl_s, rp.last_used, rp.session, rp.epoch,
                                                                                         rp.stale_at]),
                                                            self._rp_ser(rp)) for rp in rps) + "]"
                + ',"seen":' + json.dumps([k.hex() for k in self.seen])
                + ',"epochs":' + json.dumps(dict(self.session_epoch)) + "}").encode()
        top = json.dumps({"magic": INDEX_MAGIC, "version": INDEX_VERSION, "len": len(body),
                          "sha256": hashlib.sha256(body).hexdigest()}).encode()
        return top + b"\n" + body

    def _maybe_write_index(self, force: bool = False) -> None:
        if self.persist_state != "on" or self._index_writer is None or not (self._index_dirty or force):
            return
        t = self.clock()        # engine time (the replay rig's simulated clock; wall-monotonic in production)
        if not force and not self._index_urgent and t - self._index_last < self.cfg.index_every_s:
            return
        t0 = time.perf_counter()
        payload = self._index_payload()
        if payload is None:
            return
        ms = (time.perf_counter() - t0) * 1000.0
        self.stats["index_build_ms_max"] = max(self.stats.get("index_build_ms_max", 0.0), ms)
        self.stats["index_bytes"] = len(payload)
        self._index_dirty = self._index_urgent = False
        self._index_last = t
        self._index_written_wall = self.wall()
        self._index_writer.submit(payload, self.stats, sync=force)

    def shutdown(self) -> None:
        """Clean engine shutdown: write the index now (a crash keeps the last periodic one)."""
        try:
            self._maybe_write_index(force=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s index write at shutdown failed: %r", TAG, e)
        if self._index_writer is not None:
            self._index_writer.stop()

    def build_meta(self) -> SpillMeta:
        # Everything planned up to now leaves with THIS step: saves/deletes planned between steps (request finish,
        # update_from_output) and this schedule()'s loads. What build_meta plans below (prefill-done scan, sweep,
        # deadline cleanups) goes out with the NEXT step, after this step's forward has been queued on the workers:
        # a block keyed in this schedule() is written by this step's forward, and a copy shipped now could read it
        # before it is written (the worker's event is recorded in this step's pre_forward, before the forward).
        out, self._meta = self._meta, SpillMeta()
        t = self.io_clock()
        for job in list(out.saves) + list(out.loads):
            self.job_sent[job] = t
        self._check_deadlines(t)
        now = self.clock()
        held = []
        for rid, st in list(self.restores.items()):
            if st.state == "ready" and not st.main_released and now - st.t_ready > self.cfg.pin_timeout_s:
                self.stats["restore_timeouts"] += 1
                self._release(st, main_only=True)
                st.main_released = True
            if st.state == "ready" and st.main_released:
                held.append(st)
        n_state = sum(len(st.pins) + len(st.new_blocks) for st in held)
        for st in sorted(held, key=lambda x: x.t_ready):
            if n_state <= self.cfg.max_state_pins:
                break
            n_state -= len(st.pins) + len(st.new_blocks)
            self._release(st)
            st.state = "done"
        if self.persist_state == "adopting":
            self._persist_poll()
        if self.ready and self.breaker is None:
            self._scan_prefill_done()
            if self.cfg.highwater > 0:
                self.sweep()
            if self.persist_state != "adopting":
                self._sweep_expired()
        self._maybe_write_index()
        self._note_pinned()
        self._maybe_log()
        return out

    def has_pending(self) -> bool:
        # keeps EngineCore stepping (Scheduler.has_requests) while jobs are in flight or planned work waits for the
        # next SchedulerOutput; quarantined loads do not count (a rank that never answers must not keep the engine
        # busy-stepping forever: their late answers are picked up by any later step)
        m = self._meta
        return (bool(self.save_jobs) or bool(self.orphan_restores)
                or bool(m.saves or m.loads or m.deletes or m.wipe or m.rank_deletes)
                or self.persist_state == "adopting")

    def pinned_blocks(self) -> int:
        n = sum(len(p) for _, p, _ in self.save_jobs.values())
        for st in list(self.restores.values()) + list(self.orphan_restores.values()):
            n += len(st.pins) + len(st.new_blocks)
        for st in self.quarantine.values():
            n += len(st.pins) + len(st.new_blocks)
        return n


# ----------------------------------------------------------------------------------------------------------------
# the vLLM connector
# ----------------------------------------------------------------------------------------------------------------
def _signature(vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig") -> str:
    parts = [f"v{FORMAT_VERSION}"]
    try:
        parts.append(str(vllm_config.model_config.model))
        sc = vllm_config.speculative_config
        parts.append(str(getattr(sc, "model", None)))
        parts.append(str(vllm_config.cache_config.cache_dtype))
    except Exception:  # noqa: BLE001
        pass
    # page bytes per tensor, not the pool size: a block file holds the page rows of one block, so it stays valid when
    # the KV pool comes out a few blocks larger/smaller at the next start (2026-10-14, reuse across restarts)
    nb = max(1, int(kv_cache_config.num_blocks))
    for kvt in kv_cache_config.kv_cache_tensors:
        parts.append(f"{kvt.size // nb}:{','.join(kvt.shared_by)}")
    for g in kv_cache_config.kv_cache_groups:
        parts.append(f"{type(g.kv_cache_spec).__name__}:{g.kv_cache_spec.block_size}:{','.join(g.layer_names)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _tp_rank(vllm_config: "VllmConfig") -> int:
    forced = os.environ.get("GLM53_NVME_SPILL_RANK")
    if forced:
        return int(forced)
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    return int(get_tensor_model_parallel_rank())


class Glm53NvmeSpillConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM KV connector (``--kv-transfer-config '{"kv_connector":"Glm53NvmeSpillConnector",
    "kv_connector_module_path":"vllm.distributed.kv_transfer.kv_connector.v1.glm53_nvme_spill_connector",
    "kv_role":"kv_both"}'``)."""

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = dict(getattr(self._kv_transfer_config, "kv_connector_extra_config", None) or {})
        self.cfg = SpillConfig.from_env(extra)
        self.sig = _signature(vllm_config, kv_cache_config)
        self._sched: SpillScheduler | None = None
        self._worker: SpillWorker | None = None
        self._vllm_config = vllm_config
        self._kv_cache_config = kv_cache_config
        if role == KVConnectorRole.SCHEDULER:
            seed_src = "-"
            if self.cfg.persist:
                # Reuse across restarts needs block hashes that the next process computes identically. vLLM chains
                # every block hash from NONE_HASH = os.urandom(32) unless PYTHONHASHSEED is set (init_none_hash, called
                # by EngineCore right AFTER the scheduler -- and this connector -- is constructed). Setting it here, in
                # this process's environment only, makes NONE_HASH = hash_fn(seed); Python's own str hashing was fixed
                # at interpreter start and is not affected. SpillScheduler._check_hash verifies the outcome.
                if os.environ.get("PYTHONHASHSEED") is None:
                    os.environ["PYTHONHASHSEED"] = self.cfg.hash_seed
                    seed_src = "connector"
                else:
                    seed_src = "env"
            self._sched = SpillScheduler(self.cfg, vllm_config, kv_cache_config)
            self._sched.seed_source = seed_src
            logger.info("%s scheduler: dir=%s cap=%.1f GiB min_tokens=%d min_gain=%d highwater=%.2f world=%d "
                        "keep_per_chain=%d persist=%s hash_seed_from=%s ttl_save_policy=%s ttl_min_tokens=%d "
                        "version=%s", TAG, self.cfg.root, self.cfg.cap_bytes / (1 << 30), self.cfg.min_tokens,
                        self.cfg.min_gain, self.cfg.highwater, self._sched.world, self.cfg.keep_per_chain,
                        int(self.cfg.persist), seed_src, self.cfg.ttl_save_policy, self.cfg.ttl_min_tokens,
                        CONNECTOR_VERSION)
            if self.cfg.persist:
                logger.info("%s persist: index %s (%s)", TAG, self.cfg.index_path(),
                            "found" if self._sched._index is not None else self._sched._index_err)
        else:
            self._worker = SpillWorker(self.cfg, kv_cache_config, _tp_rank(vllm_config))

    # ---------------- worker side ----------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self._worker is not None
        vc, kvc = self._vllm_config, self._kv_cache_config
        self._worker.register(kv_caches, self.sig, ident_fn=lambda: _identity(vc, kvc))

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        if self._worker is not None and isinstance(kv_connector_metadata, SpillMeta):
            self._worker.handle(kv_connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        # Jobs are taken in handle_preemptions (first thing in the step on both runners); loads run on the
        # worker's reader thread, nothing to do here.
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata",
                      **kwargs: Any) -> None:
        return

    def wait_for_save(self):
        return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        return self._worker.take_meta() if self._worker is not None else None

    def shutdown(self):
        if self._worker is not None:
            self._worker.shutdown()
        if self._sched is not None:
            self._sched.shutdown()

    # ---------------- scheduler side ----------------
    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        if self._sched is not None:
            self._sched.bind(block_pool=gpu_block_pool)

    def glm53_bind_kv_cache_manager(self, kv_cache_manager) -> None:  # [glm53-nvme-spill] scheduler.py hook
        if self._sched is not None:
            self._sched.bind(kv_cache_manager=kv_cache_manager)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        assert self._sched is not None
        return self._sched.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self._sched is not None
        self._sched.update_state_after_alloc(request)

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        assert self._sched is not None
        return self._sched.build_meta()

    def update_connector_output(self, connector_output):
        assert self._sched is not None
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        self._sched.update_connector_output(meta if isinstance(meta, SpillWorkerMeta) else None)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self._sched is not None
        self._sched.request_finished(request)
        return False, None

    def request_finished_all_groups(self, request: "Request",
                                    block_ids: tuple[list[int], ...]) -> tuple[bool, dict[str, Any] | None]:
        assert self._sched is not None
        self._sched.request_finished(request)
        return False, None

    def has_pending_push_work(self) -> bool:
        return self._sched is not None and self._sched.has_pending()

    def take_events(self):
        return ()
