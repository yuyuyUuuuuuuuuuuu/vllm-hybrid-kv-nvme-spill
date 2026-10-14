# SPDX-License-Identifier: AGPL-3.0-only
"""Production-shaped KV layout for the spill tests (production KV start-up log + kv_cache_utils._glm5_next_tensor_layout).

11 MLA tensors (page 2,359,296 B) each co-owned by MLA layer i, mamba layer i of each of the 4 mamba groups (9/9/8/8
layers) and drafter layer i (5 layers, exact-fit page); 11 indexer tensors (page 152,064 B) co-owned by indexer
layer i and kpool-tail layer i. Groups: 0 = MLA+indexer (22 layers, block 4608), 1 = kpool tail (block 16, scratch),
2-5 = mamba (block 4608, align), 6 = DFlash2 drafter SWA (block 1152, window 2048).

``scale`` divides both page sizes (integration rig: tiny pages, same structure)."""
from __future__ import annotations

MLA_PAGE = 2_359_296
IDX_PAGE = 152_064
N_MLA = 11
MAMBA_LAYERS = (9, 9, 8, 8)
N_DRAFT = 5


def layer_names():
    mla = [f"mla.{i}" for i in range(N_MLA)]
    idx = [f"idx.{i}" for i in range(N_MLA)]
    tail = [f"tail.{i}" for i in range(N_MLA)]
    mamba = [[f"m{g}.{i}" for i in range(n)] for g, n in enumerate(MAMBA_LAYERS)]
    draft = [f"draft.{i}" for i in range(N_DRAFT)]
    return mla, idx, tail, mamba, draft


def kv_tensors(num_blocks: int, scale: int = 1):
    """[(size_bytes, shared_by)] in the production order."""
    mla, idx, tail, mamba, draft = layer_names()
    mp, ip = MLA_PAGE // scale, IDX_PAGE // scale
    out = []
    for i in range(N_MLA):
        shared = [mla[i]] + [g[i] for g in mamba if i < len(g)] + ([draft[i]] if i < N_DRAFT else [])
        out.append((mp * num_blocks, shared))
    for i in range(N_MLA):
        out.append((ip * num_blocks, [idx[i], tail[i]]))
    return out


def group_layers():
    mla, idx, tail, mamba, draft = layer_names()
    return [mla + idx, tail, *mamba, draft]


def build_kv_cache_config(num_blocks: int, scale: int = 1, specs=None, eagle_groups=(6,)):
    """A KVCacheConfig with the production tensor layout. ``specs``: optional list of 7 KVCacheSpec objects (the
    scheduler rig passes its own); default = minimal stand-ins (only the worker reads this config)."""
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

    groups_l = group_layers()
    if specs is None:
        specs = [_StubSpec(4608), _StubSpec(16), *[_StubSpec(4608)] * 4, _StubSpec(1152)]
    groups = [KVCacheGroupSpec(layer_names=list(n), kv_cache_spec=s, is_eagle_group=(i in eagle_groups))
              for i, (n, s) in enumerate(zip(groups_l, specs))]
    tensors = [KVCacheTensor(size=sz, shared_by=list(sb)) for sz, sb in kv_tensors(num_blocks, scale)]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)


class _StubSpec:
    def __init__(self, block_size):
        self.block_size = block_size


def allocate(kv_cache_config, device="cpu", seed=0):
    """Allocate the raw tensors like v1/worker/gpu/attn_utils._allocate_kv_cache, fill with seeded random bytes,
    return {layer_name: tensor}."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    raws = []
    for kvt in kv_cache_config.kv_cache_tensors:
        t = torch.randint(0, 256, (kvt.size,), dtype=torch.uint8, generator=g).to(device).view(torch.int8)
        raws.append(t)
        for name in kvt.shared_by:
            out[name] = t
    return out, raws
