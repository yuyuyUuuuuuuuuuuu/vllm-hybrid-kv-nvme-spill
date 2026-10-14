#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Origin: modified copy of code/fork/overlay/patch_tf_bundle.py of the companion repository
# github.com/yuyuyUuuuuuuuuuuu/glm-5.3-flash-dgx-spark-vllm-optimization (itself derived from the AGPL-3.0 launcher);
# change: the GLM53_NVME_SPILL entry. The docs/*.md references below are files under code/fork/docs/ of that repository.
"""[glm53-tf-bundle] Install the tf-exl3-fork bundle inside the serving container (idempotent, fail closed).

1. Copy /opt/glm53/tf/site/* into site-packages: the fork modules, its AOT extension (sm_121a) and the dist-info that
   registers the vllm.general_plugins entry point. The fork stays INERT unless TF_EXL3_MOE is 1/on/true/yes
   (integrate.install reads it when vLLM loads general plugins, in every process).
2. Run an overlay patch from /opt/glm53/tf/overlay/ ONLY when its env var is explicitly non-empty:
     patch_spec_resample_noise.py   GLM53_SPEC_RESAMPLE_INDEPENDENT (0|1)
     patch_spec_block_keys.py       GLM53_REJECTION_METHOD (standard|block): block -> row-keyed randomness of the DFlash2
                                    walk and the rejection sampler (cross-step exactness of block verification; needs
                                    the resample-noise patch above, runs after it); standard -> prints and touches
                                    nothing. docs/BLOCK_VERIFY.md
     patch_drafter_fp8.py           GLM53_DRAFT_FP8 (off|1|layers|layers,fc)
     patch_drafter_lmhead_fp8.py    GLM53_DRAFT_LMHEAD_FP8 (0|1)
     patch_kpool_tail_seed_stride.py GLM53_KPOOL_SEED_STRIDE (1) — upstream MiaAI-Lab #264 (vLLM #57477 backport),
                                    verbatim copy of upstream overlay/patch_kpool_tail_seed_stride.py (sha256 c8f05397...)
     patch_kpool_tail_ring.py       GLM53_KPOOL_RING (1) — vLLM #58454 backport: the kpool indexer tail ring holds
                                    kpool * next_pow2(cdiv(kpool + num_spec, kpool)) slots (16 at k=7) so rejected
                                    drafts cannot overwrite committed pool keys. Needs the seed-stride fix above
                                    (fails closed without it); runs after it. docs/KPOOL_RING.md
     patch_kda_strided_qkv.py       GLM53_KDA_STRIDED_QKV (0|1) — vLLM #55736 backport (KDA half): the recurrent
                                    decode kernel reads token-strided q/k/v/beta in place instead of paying four
                                    .contiguous() copies per KDA layer per step (bitwise-equal outputs). Operator
                                    switch (env.r16 #switch, r16k start.sh forwards it to both ranks); 0 -> prints and
                                    touches nothing. docs/KDA_STRIDED_QKV.md
     patch_kpool_drop_lowest.py     GLM53_KPOOL_DROP_LOWEST (0|1) — the kpool indexer keeps the select_k-1
                                    HIGHEST-scored pools before expand_pools_and_append_tail (the top-k ops return a
                                    deterministic SET in a nondeterministic ORDER, so the old pool_ids[:,
                                    :select_k-1] dropped an arbitrary pool per run). Deterministic, no host sync,
                                    FULL-graph safe. Operator switch (env.r16 #switch, the r16l start.sh forwards it
                                    to both ranks); 0/unset -> prints and touches nothing. docs/KPOOL_DROP_LOWEST.md
     patch_mla_exactlens.py         GLM53_MLA_EXACT_LENS (0|1) — installs glm53_mla_exactlens.py and arms the site
                                    integrate.py: production's FA2 sparse-MLA plan uses the exact selected-key counts
                                    (2044 + ctx % 4 once ctx >= index_topk, not index_topk + ctx % 4), so decode rows
                                    stop attending 4 - ctx % 4 slot-0 copies plus the next row's (or a stale call's)
                                    first keys. Self-tested on the real ops at the first build. 0 -> removes and
                                    disarms; unset -> skipped. Operator switch (env.r16 #switch, the r16z2x start.sh
                                    forwards it to both ranks). docs/MLA_EXACT_LENS.md
     patch_dense_w8a8.py            GLM53_DENSE_W8A8 (0|1) — installs fp8_w8a8.py + tf_fp8_w8a8_ext into
                                    site-packages: W8A8 (per-token fp8 activations x the stored per-channel fp8
                                    weights, cutlass_scaled_mm) for the dense + shared-expert FP8 linears on the
                                    PREFILL path only; 1.1-2.4x per GEMM against production's TileLang W8A16 /
                                    Marlin path, the standard fp8 operand repacked per call from the Marlin payload
                                    (no resident copy, KV pool 2.00M -> 1.96M). Decode stays Marlin. 1 -> also arms
                                    the site integrate.py (the marked fp8_w8a8 import in plugin_register); 0 ->
                                    removes any installed copy and disarms. Operator switch (env.r16 #switch, the
                                    r16y start.sh forwards it to both ranks); unset -> skipped. docs/DENSE_W8A8.md
     patch_flashkda.py              GLM53_KDA_FLASHKDA (0|1) — FlashKDA 17a037d (fp32 recurrent state, vLLM
                                    #58846; built for sm_121a as _flashkda_fp32_C so it cannot touch the image's
                                    pre-fix bf16 vllm/_flashkda_C) replaces the Triton chunk_kda_with_fused_gate
                                    chain for the KDA chunked prefill: 2.4x per layer at the production shapes
                                    through the wrapper (3.5x with the kda_conv quick win, which this patch also
                                    keeps installable by extending glm53_prefill_quickwins.VERIFIED), ~0.45-0.56 s
                                    per 13,824-token chunk over 34 KDA layers; same inputs/outputs contract (RAW
                                    g1/beta, in-kernel l2norm + bounded gate, fp32 [N,H,D,D] final state for
                                    decode); the recurrent decode path is untouched. Operator switch (env.r16
                                    #switch, the r16n start.sh forwards it to both ranks); 0/unset -> prints and
                                    touches nothing (the extension is not installed either). docs/KDA_FLASHKDA.md.
                                    WHICH build stages is the r16z operator switch GLM53_KDA_FLASHKDA_V
                                    (env.r16 #switch; unset/1 = the shipped r16x build = production bytes; 2 =
                                    the fkda2 precision build; 3 = the fkda3 build + the direct-output kda.py):
                                    all three install under the same site-packages names, each wrapper pins its
                                    extension sha at boot. docs/KDA_FLASHKDA3.md
     patch_mhc_sp.py                GLM53_MHC_SP (0|1) — sequence-parallel prefill for the mHC bookkeeping over
                                    TP (docs/MHC_SP.md): a >= 1024-token forward keeps the residual stream SHARDED
                                    across the TP group (the per-token mHC family runs on T/2 rows; the
                                    attention/MLP all-reduces become reduce-scatter + all-gather pairs, the same
                                    wire bytes at TP=2); decode and every CUDA-graph capture stay plain TP
                                    byte-identically. Also extends the quickwins / moeglue fingerprint tables with
                                    the SP-patched forwards (else those features silently disarm on their drift
                                    checks). Operator switch (env.r16 #switch, the r16msp start.sh forwards it to
                                    both ranks); 0/unset -> prints and touches nothing.
     patch_moe_fused16.py           GLM53_MOE_FUSED16 (0|1) — the P16 routed-MoE prefill (production arithmetic, faster
                                    schedules; h2 bit-identical): copies glm53_moe_fused16.py + the shared extension into
                                    site-packages and arms it in integrate.py (before glm53_moe_e4m3's block). Prefill
                                    E3 tier only; decode untouched. 0 -> prints and touches nothing. docs/MOE3.md
     patch_moe_e4m3.py              GLM53_MOE_E4M3 (0|1) — the e4m3 routed-MoE prefill: copies glm53_moe_e4m3.py + its
                                    extension from this overlay dir into site-packages and arms it in integrate.py
                                    (plugin_install after every other plugin step). Prefill calls only (tokens > the
                                    fused cap); decode untouched. Operator switch (env.r16 #switch, the r16e4 start.sh
                                    forwards it to both ranks); 0 -> prints and touches nothing. docs/MOE_E4M3.md
     patch_mhc_sp2.py               GLM53_MHC_SP2 (0|1) — needs GLM53_MHC_SP=1 (refuses otherwise): pipelined SP prefill
                                    (k=2 interleaved sub-chunks per rank shard, the reduce-scatter / all-gather of
                                    each sub-chunk on a side stream overlapping the neighbouring sub-chunk's mHC;
                                    bitwise the r16x SP result at sub-chunks >= 1,537 rows) and odd-T SP. docs/MHC_SP2.md
     patch_mhc_fused.py             GLM53_MHC_FUSED (0|1) — the fused mHC post + prenorm GEMM on eager prefill calls
                                    (opt-kdamhc): copies glm53_mhc_fused.py + its extension into site-packages and
                                    arms it in integrate.py after patch_moe_e4m3's block; residual_cur bitwise
                                    production's, the 24 mixing logits in the decode branch's fp32 arithmetic.
     patch_kpool_tail_positions.py  GLM53_KPOOL_TAIL_POSITIONS (0|1|2) — 2 = graph-safe (use 2; 1 is not read by FULL decode graphs): prepare_attn passes the
                                    token positions, so KpoolTailMetadataBuilder emits per-request circular tail slots
                                    (stock: every tail write past the first ring goes to block 0 / stale block ids and
                                    overwrites other requests' pooled indexer keys). docs/PREFIX_HIT_TAIL.md
     patch_mamba_align_seed.py      GLM53_MAMBA_ALIGN_SEED (0|1) — MambaHybridModelState.add_request seeds a prefix-hit /
                                    resumed request's running KDA column in mamba blocks (cache_config.mamba_block_size)
                                    instead of cache_config.block_size (576 once the engine core recomputes it; an
                                    in-process worker then pre-copies a zero/foreign state). docs/PREFIX_HIT_TAIL.md
     patch_nvme_spill.py            GLM53_NVME_SPILL (0|1) — installs glm53_nvme_spill_connector.py into vLLM's kv_connector/v1
                                    and hands the KV connector the KVCacheManager (scheduler.py). Inert unless start.sh
                                    also passes --kv-transfer-config (it does for GLM53_NVME_SPILL=1). docs/design.md of this repository
   Empty/unset -> skipped, the image stays stock for that feature.
Any failure raises SystemExit, which stops the container start (same contract as the launcher's other patches).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

MARK = "[glm53-tf-bundle]"
OPT = Path(os.environ.get("GLM53_OPT", "/opt/glm53")) / "tf"
SITEPKG = Path(os.environ.get("GLM53_SITEPKG", "/usr/local/lib/python3.12/dist-packages"))
PATCHES = (
    ("patch_spec_resample_noise.py", "GLM53_SPEC_RESAMPLE_INDEPENDENT"),
    ("patch_spec_block_keys.py", "GLM53_REJECTION_METHOD"),  # after resample-noise: anchors on its patched text
    ("patch_drafter_fp8.py", "GLM53_DRAFT_FP8"),
    ("patch_drafter_lmhead_fp8.py", "GLM53_DRAFT_LMHEAD_FP8"),
    ("patch_kpool_tail_seed_stride.py", "GLM53_KPOOL_SEED_STRIDE"),
    ("patch_kpool_tail_ring.py", "GLM53_KPOOL_RING"),  # after seed-stride: it requires the #57477 seed
    ("patch_kda_strided_qkv.py", "GLM53_KDA_STRIDED_QKV"),  # opt-in (r16k): vLLM #55736 KDA strided decode inputs
    ("patch_kpool_drop_lowest.py", "GLM53_KPOOL_DROP_LOWEST"),  # opt-in (r16l): drop the lowest-scored kpool, deterministic
    ("patch_dense_w8a8.py", "GLM53_DENSE_W8A8"),  # opt-in (r16y): W8A8 prefill dense GEMMs (installs + arms the module)
    ("patch_flashkda.py", "GLM53_KDA_FLASHKDA"),  # opt-in (r16n): FlashKDA 17a037d chunked prefill, decode untouched
    ("patch_mhc_sp.py", "GLM53_MHC_SP"),  # opt-in (r16msp): sequence-parallel mHC prefill over TP, decode byte-identical
    ("patch_mhc_sp2.py", "GLM53_MHC_SP2"),  # opt-in (mhc2): pipelined SP (needs GLM53_MHC_SP=1; runs after it)
    # GLM53_SP_FP8AG (opt-kdamhc) is NOT shipped: the kit carries opt-dense-rev's GLM53_DENSE_W8A8_FP8AG -
    #   the same fp8 sequence-parallel all-gather behind the W8A8 install, pair-gated by boot_checks. Only
    #   that one ships; patch_mhc_sp/sp2 still tolerate an sp-fp8ag-edited tree.
    ("patch_moe_fused16.py", "GLM53_MOE_FUSED16"),  # opt-in (moe3): P16 prefill MoE, production arithmetic; integrate.py
    ("patch_moe_e4m3.py", "GLM53_MOE_E4M3"),  # opt-in (r16e4): e4m3 routed-MoE prefill; edits site-packages integrate.py
    ("patch_mhc_fused.py", "GLM53_MHC_FUSED"),  # opt-in (opt-kdamhc): fused mHC post+prenorm GEMM; arms integrate.py LAST
    ("patch_mla_exactlens.py", "GLM53_MLA_EXACT_LENS"),  # opt-in (opt-decodekit): exact FA2 sparse-MLA plan lengths
    ("patch_kpool_tail_positions.py", "GLM53_KPOOL_TAIL_POSITIONS"),  # opt-in (prefixhit): per-request kpool tail rings
    ("patch_mamba_align_seed.py", "GLM53_MAMBA_ALIGN_SEED"),  # opt-in (prefixhit): hit seed column in mamba blocks
    ("patch_nvme_spill.py", "GLM53_NVME_SPILL"),  # opt-in (nvme-spill): resume points on local NVMe (KV connector)
)


def install_site() -> None:
    src = OPT / "site"
    if not src.is_dir():
        raise SystemExit(f"{MARK} missing {src}")
    names = []
    for p in sorted(src.iterdir()):
        dst = SITEPKG / p.name
        if p.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(p, dst)
        else:
            shutil.copy2(p, dst)
        names.append(p.name)
    state = os.environ.get("TF_EXL3_MOE", "").strip() or "(unset: fork inert)"
    print(f"{MARK} installed {len(names)} entries into {SITEPKG}: {' '.join(names)}; TF_EXL3_MOE={state}")


def run_patch(fname: str, env: str) -> None:
    val = os.environ.get(env, "")
    if not val.strip():
        print(f"{MARK} {fname}: {env} unset -> skipped (stock)")
        return
    path = OPT / "overlay" / fname
    if not path.is_file():
        raise SystemExit(f"{MARK} {fname} missing but {env}={val!r} is set")
    spec = importlib.util.spec_from_file_location(fname[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect
    rc = mod.main([]) if inspect.signature(mod.main).parameters else mod.main()   # upstream patches take no argv
    if rc:
        raise SystemExit(f"{MARK} {fname} failed (rc={rc})")
    print(f"{MARK} {fname}: applied ({env}={val})")


def main() -> int:
    install_site()
    for fname, env in PATCHES:
        run_patch(fname, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
