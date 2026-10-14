# SPDX-License-Identifier: AGPL-3.0-only
"""Build a production-mirror VllmConfig + KVCacheConfig for the CPU replay rig.

All values are transcribed from the production deployment (its environment, its KV start-up log and our earlier
re-read investigation; those artifacts are not shipped). Nothing here contacts GPUs: the platform/device is pinned
to CPU (a recorded substitution, see tests/rig/README.md).

The KV cache specs (7 groups, 582 usable blocks) mirror the production KV start-up log:
  group 0: MLAAttentionSpec   layers=22 block_size=4608 page=152,064 B
  group 1: KpoolTailSpec      layers=11 block_size=16   page=152,064 B (scratch)
  group 2-5: MambaSpec        layers=9/9/8/8  block_size=4608 align, spec blocks=7
  group 6: SlidingWindowSpec  layers=5  block_size=1152 window=2048 eagle
retention_by_group=[32256 x6, 0].
"""

from __future__ import annotations

import os

NUM_BLOCKS_INCL_NULL = 583  # usable 582 + null block (prod startup log)
MLA_LAYERS = 22
KPOOL_TAIL_LAYERS = 11
MAMBA_LAYERS = (9, 9, 8, 8)
DRAFTER_LAYERS = 5
BLOCK_SIZE = 4608  # mamba/MLA page size in tokens (segment alignment)
KPOOL_BLOCK = 16  # index_kpool slots per kpool tail block
DRAFTER_BLOCK_SIZE = 1152
DRAFTER_WINDOW = 2048
MAMBA_SPEC_BLOCKS = 7  # running 1 + speculative 7 (capacity section of the re-read investigation)
RETENTION = 32256
MLA_PAGE_BYTES = 152064
KPOOL_PAGE_BYTES = 152064
MAMBA_PAGE_BYTES = 2359296
DRAFTER_PAGE_BYTES = 2359296

MAX_NUM_SEQS = 8
MAX_NUM_BATCHED_TOKENS = 16384  # prod chunk budget: 13824 = floor(16384/4608)*4608


def _set_env() -> None:
    """Subset of the production environment + CPU-only platform overrides.

    GLM53_APC_RETENTION_INTERVAL(-_SWA) are the start.sh spellings; the runtime
    reads VLLM_PREFIX_CACHE_RETENTION_INTERVAL(-_SWA) (kv_cache_coordinator.py).
    """
    os.environ.setdefault("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", str(RETENTION))
    os.environ.setdefault("VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA", "0")
    os.environ.setdefault("VLLM_MAMBA_CACHE_MODE", "align")
    # prod startup log group 6 prints "lookup=boundary" == GLM53_DRAFT_KV_COMPACT=1
    os.environ.setdefault("GLM53_DRAFT_KV_COMPACT", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_booted = False


def _boot() -> None:
    """Force a neutral platform before anything constructs DeviceConfig.

    The prod image detects CUDA via pynvml; without --gpus no plugin activates
    and the builtin UnspecifiedPlatform has an empty device_type, which makes
    the default DeviceConfig() raise. We pre-instantiate UnspecifiedPlatform so
    `current_platform.device_type` checks in DeviceConfig find an object whose
    `device_type` is empty but whose check_and_update_config() is a no-op — the
    CPU platform would wrongly rewrite the config (block_size=16, prefix
    caching disabled). DeviceConfig is given device='cpu' explicitly.
    Recorded as a substitution in tests/rig/README.md.
    """
    global _booted
    if _booted:
        return
    _set_env()
    import vllm.platforms as pf
    from vllm.platforms.interface import UnspecifiedPlatform

    pf._current_platform = UnspecifiedPlatform()
    _booted = True


_boot()


def _override_arch(hf_config):
    # Architecture inspection only: swap the Glm5Next / DFlash draft model
    # registrations (their module chains import GPU kernels — glm53_flashkda,
    # triton, vllm._C) for supported stand-ins. The rig never instantiates
    # either model; Scheduler/KVCacheManager read no architecture hereafter.
    name = str(getattr(hf_config, "name_or_path", "")).lower()
    is_drafter = "dflash" in name or hasattr(hf_config, "dflash_config")
    if is_drafter:
        # 'DFlash'-prefixed archs survive EAGLEConfig(method='dflash') as-is.
        hf_config.architectures = ["DFlash2DraftModel"]
    else:
        hf_config.architectures = ["LlamaForCausalLM"]
    return hf_config


def build_vllm_config(device: str = "cpu"):
    _boot()
    import torch

    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ObservabilityConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )
    from vllm.config.device import DeviceConfig

    _set_env()
    model_dir = os.environ.get(
        "APCREREAD_MODEL_DIR",
        "/mnt/models/GLM-5.3-Flash-EXL3-TR3-4bpw-partial",
    )
    drafter_dir = os.environ.get(
        "APCREREAD_DRAFTER_DIR",
        "/mnt/models/GLM-5.3-Flash-DFlash2-dc77ff1c",
    )

    model_config = ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        max_model_len=1_000_000,
        dtype="bfloat16",
        trust_remote_code=True,
        # Architecture inspection only: swap the Glm5Next registration (whose
        # module chain imports GPU kernels, e.g. glm53_flashkda + triton) for a
        # plain causal-LM architecture. The rig never instantiates the model;
        # Scheduler/KVCacheManager read no architecture beyond this check.
        hf_overrides=_override_arch,
    )

    scheduler_config = SchedulerConfig(
        1_000_000,  # max_model_len InitVar (mirrors model_config)
        False,  # is_encoder_decoder InitVar
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=True,
        scheduler_reserve_full_isl=True,
        long_prefill_token_threshold=0,
        policy="fcfs",
    )
    cache_config = CacheConfig(
        block_size=DRAFTER_BLOCK_SIZE,
        cache_dtype="fp8",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        prefix_match_unit=DRAFTER_BLOCK_SIZE,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    device_config = DeviceConfig(device=device)

    speculative_config = SpeculativeConfig(
        model=drafter_dir,
        method="dflash",
        num_speculative_tokens=7,
        draft_tensor_parallel_size=2,
        enforce_eager=True,
        target_model_config=model_config,
        target_parallel_config=parallel_config,
    )

    kv_transfer_config = None
    if os.environ.get("NVMESPILL_ON") == "1":
        # [nvme-spill rig] the connector exactly as start.sh passes it (module installed by patch_nvme_spill.py),
        # handed to the CONSTRUCTOR so VllmConfig.__post_init__ runs every connector check production runs
        # (_verify_kv_transfer_compat against PYTORCH_CUDA_ALLOC_CONF, the HMA/SupportsHMA check, PIECEWISE forcing).
        # The rig's docker_run.sh passes the kit's PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to both arms.
        from vllm.config import KVTransferConfig

        kv_transfer_config = KVTransferConfig(
            kv_connector="Glm53NvmeSpillConnector",
            kv_connector_module_path=(
                "vllm.distributed.kv_transfer.kv_connector.v1.glm53_nvme_spill_connector"
            ),
            kv_role="kv_both",
        )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        device_config=device_config,
        speculative_config=speculative_config,
        observability_config=ObservabilityConfig(),
        kv_transfer_config=kv_transfer_config,
    )
    # The scheduler asserts this to be set by the worker pass; we size the
    # pool directly from the production startup log.
    cache_config.num_gpu_blocks = NUM_BLOCKS_INCL_NULL
    cache_config.num_cpu_blocks = 0
    return vllm_config


def build_kv_cache_config(vllm_config):
    """7-group KVCacheConfig mirroring groups 0-6 of the production KV start-up log.

    Built directly (not via the model-runner spec path) because the real spec
    derivation imports the Glm5Next model implementation, whose module chain
    needs GPU kernels. Group geometry, page sizes, block sizes, and
    spec-decode/mamba fields are transcribed from that log (its group lines);
    the single shared pool of 582 usable + 1 null block matches its pool line.
    """
    import torch

    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        KpoolTailSpec,
        MLAAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    def mla_spec():
        return MLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=33,  # sized to page below; scheduler reads page bytes only
            dtype=torch.float8_e4m3fn,
        )

    specs = [
        mla_spec(),
        KpoolTailSpec(
            block_size=KPOOL_BLOCK,
            num_kv_heads=1,
            head_size=95,  # sized to page below
            dtype=torch.bfloat16,
            sliding_window=DRAFTER_WINDOW,
        ),
    ]
    for _layers in MAMBA_LAYERS:
        specs.append(
            MambaSpec(
                block_size=BLOCK_SIZE,
                shapes=((1, 2, 128, MAMBA_PAGE_BYTES // (128 * 128 * 2)),),
                dtypes=(torch.bfloat16,),
                mamba_cache_mode="align",
                num_speculative_blocks=MAMBA_SPEC_BLOCKS,
            )
        )
    specs.append(
        SlidingWindowSpec(
            block_size=DRAFTER_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=512,  # 1152 tok * (512+512) * 2 B = 2,359,296 B page below
            dtype=torch.bfloat16,
            sliding_window=DRAFTER_WINDOW,
        )
    )
    # Pin page_size_bytes to the production values so pool arithmetic does not
    # depend on the exact head geometry (prod log lines 4-11).
    for spec, page in zip(
        specs,
        [MLA_PAGE_BYTES, KPOOL_PAGE_BYTES]
        + [MAMBA_PAGE_BYTES] * 4
        + [DRAFTER_PAGE_BYTES],
    ):
        object.__setattr__(spec, "page_size_padded", page)

    # [nvme-spill rig] production tensor layout and layer names (kv_cache_utils._glm5_next_tensor_layout): the
    # scheduler never reads tensors, but the spill connector sizes its files from them (real page sizes here; the
    # simulated worker ranks use the same structure with pages scaled down, see sim/spill_world.py).
    import prodlayout

    return prodlayout.build_kv_cache_config(NUM_BLOCKS_INCL_NULL, scale=1, specs=specs, eagle_groups=(6,))


def build_scheduler():
    """Real Scheduler + real KVCacheManager on the production config."""
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        resolve_kv_cache_block_sizes,
    )
    from vllm.v1.core.sched.scheduler import Scheduler

    vllm_config = build_vllm_config()
    kv_cache_config = build_kv_cache_config(vllm_config)
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    # Real scheduler; StructuredOutputManager replaced by a no-op stub (needs a
    # tokenizer / grammar engine which a model-less CPU rig cannot provide).
    struct_stub = _StructuredOutputStub()
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=struct_stub,  # type: ignore[arg-type]
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        log_stats=True,
    )
    hasher = get_request_block_hasher(hash_block_size, sha256)
    # Mirror engine core: initialize the NONE_HASH chain seed before hashing.
    from vllm.v1.core.kv_cache_utils import init_none_hash

    init_none_hash(sha256)
    return scheduler, hasher, vllm_config, kv_cache_config


class _StructuredOutputStub:
    """Stand-in for StructuredOutputManager (grammar engine).

    The rig only issues plain chat/turn requests without JSON schema, so the
    manager is queried but never engaged; every hook returns the no-grammar
    answer. Recorded as a substitution in tests/rig/README.md.
    """

    def should_advance(self, request, new_token_ids=None, *a, **k) -> bool:
        return False

    def grammar_bitmask(self, requests, structured_output_request_ids, *a, **k):
        return None

    def finish_request(self, request, *a, **k):
        return None
