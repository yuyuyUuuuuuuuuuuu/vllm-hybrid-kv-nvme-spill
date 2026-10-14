# vLLM KV connector: NVMe "resume points" for hybrid (MLA + Mamba / linear-attention) models, surviving engine restarts

A vLLM v1 KV connector (`KVConnectorBase_V1`, `SupportsHMA`) that saves each long conversation's **resume point** to
local NVMe right after its prompt is computed and, when the conversation's next turn arrives and its blocks have been
evicted from the GPU prefix cache, **restores the same bytes under the same prefix-cache keys** instead of re-running
prefill. It is built for a **hybrid model** whose prefix hit needs several KV groups at once: MLA attention + a sparse
indexer, four linear-attention (KDA, Mamba-style) state groups and a speculative-decoding (DFlash) draft window. It
runs with tensor parallelism 2 (every decision taken once by the scheduler, every read agreed by all ranks), contains
faults (deadlines, a latched breaker, quarantine of blocks whose late write could still land), prunes resume points of
superseded turns, accepts a per-request retention time through `kv_transfer_params`, and is designed to **reuse its
files across engine restarts** (a persisted index guarded by an identity signature, plus a fix for vLLM's per-process
random block-hash seed). It runs in production on GLM-5.3-Flash served by vLLM on two NVIDIA DGX Spark (GB10) machines.
Status of each part: save/restore and the per-request TTL path are live in production (the client's TTL field was
confirmed arriving on 2026-10-14: the first saved resume point landed in the 1-day class); pruning is deployed, but no
prune event on real traffic had been recorded yet. Reuse across restarts was
measured in the replay rig and in GPU round-trip tests across processes; adoption across a **production** restart had not
yet been observed when this was published (`docs/measurements.md` 8.6).

| fact | value | source |
|---|---|---|
| model / hardware | GLM-5.3-Flash (EXL3 4 bpw, DFlash2 drafter), vLLM `0.1.dev20051+g487ecf187` + launcher overlays, 2 x GB10, TP=2; KV cache 2,003,436 tokens, 582 usable blocks, prefix-hash unit 4,608 tokens | `docs/design.md`, companion repo |
| re-read tokens, replay rig, 4 h x 5 seeds | DSH conversations 6,372,224 -> 2,106,460 (**-66.9%**); all requests -54.0%; collapses (a continuation turn hitting <= 23,040 tokens) 7.2 -> **0** | `results/fix/aggregate_m_on.txt`, `docs/measurements.md` 5.1 |
| re-read after an engine restart, rig (restart at 2 h, 5 seeds) | 2,028,530 -> 174,139 (**-91%**); first turn of each conversation 1,903,966 -> 24,823 (-98.7%) | `results/ttl/aggregate.txt`, `docs/measurements.md` 7.3 |
| block hashes across restarts | stock behaviour of this vLLM: different in every process (`NONE_HASH = os.urandom(32)` without `PYTHONHASHSEED`); with the connector: identical | `docs/hash-seed-finding.md`, `tests/t_hash_stability.py` |
| NVMe speed, one GB10 rank on a test machine, 600k-token resume point (3.72 GB) | write 1.40 s, read 1.24 s (O_DIRECT, 32 MiB pinned staging, CRC-32 per 8 MiB), bytes identical; a bandwidth-bound kernel running alongside: 220.1 -> 217.1 / 216.5 GB/s | `results/fix/gpu_final.txt`, `docs/measurements.md` 5.4 |
| production decode / prefill, ON vs OFF | decode -1.0% to +0.3% (4 prompt types); 32k prefill median 3,021.5 -> 3,068 tok/s. OFF is an earlier run of the same configuration; ON was measured only in idle windows (prefill: runs that did not overlap other requests) | `docs/measurements.md` 8.3 |
| production output quality after a restore (4 probe runs, 81k-token conversation) | mean KL vs full prefill: restored 0.0090, GPU hit 0.0094, second full prefill 0.0061; drafter acceptance 0.423 / 0.436 / 0.426; judged on the 4-run mean: the single-run verdicts failed in 2 of 4 runs (KL and/or acceptance), and the GPU-hit control itself reached KL 0.0241 in run 4 | `docs/measurements.md` 8.2 |
| per-request TTL, rig (4 conversations, 4 h x 5 seeds) | total re-read -24.9% vs OFF; but the conversation sent with `ttl_s=0` (not saved) re-read +113% vs OFF, because restores for other conversations evict its GPU prefix earlier (unfixed B8, `docs/limitations.md`) | `results/ttl/final/aggregate.txt` (3), `docs/measurements.md` 7.5 |
| disk growth with pruning (`KEEP_PER_CHAIN=2`) | steady 1.31 -> 0.37 GiB/h/rank (-72%), re-reads unchanged | `results/prune/aggregate.txt`, `docs/measurements.md` 6.1 |
| file size per 4,608-token block, one rank | MLA+indexer 27,624,960 B; mamba states 21,233,664 / 18,874,368 B; DFlash window 11,796,480 B (+4 KiB header) | `docs/design.md` 2.4 |
| fault injection, rig (final version) | 27 runs incl. missing / corrupt files, write errors, read-only disk, hung or silent rank, corrupt index, signature mismatch: 0 unfinished requests, 0 content mismatches; 0 leaked pins except the two hung-rank runs (`h_load`, `k_load`), which end with 17 quarantined blocks by design (a late write must not land in a reused block). Plus 1 negative control (`n_noload`, restores skip the read) whose content check detects 1,238 mismatching blocks, as intended | `results/ttl/final_faults/aggregate.txt` |
| tests re-run from this tree | CPU 19 + 49 + 74 + 133 + 9 + 13 + 15 checks pass; GPU 18 + 9 + 5 pass; rig re-runs field-identical to the recorded runs | `REPRODUCE.md`, `results/published-tree/` |
| licence | AGPL-3.0-only | `LICENSE`, `THIRD_PARTY_NOTICES.md` |

Companion repository (the serving setup, kernels and measurements this builds on):
[glm-5.3-flash-dgx-spark-vllm-optimization](https://github.com/yuyuyUuuuuuuuuuuu/glm-5.3-flash-dgx-spark-vllm-optimization).

## Finding: vLLM's prefix-cache block hashes change on every restart

In vLLM builds without upstream PR #51875 (merged 2026-08-18), including the one we serve,
`vllm/v1/core/kv_cache_utils.py:init_none_hash` sets `NONE_HASH = os.urandom(32)` when `PYTHONHASHSEED` is unset
(lines 100-115 in our tree). Every block hash is chained from `NONE_HASH`, so **every prefix-cache block hash differs
after each engine restart**, and nothing stored under those hashes outside the process can be found again. With the
default `sha256` hash function this vLLM logs no warning (the warning is only for the CBOR hash functions). Upstream
`main` now derives `NONE_HASH` from a fixed seed for `sha256`/`sha256_cbor` but still from random bytes for `xxhash`.

The connector fixes it without touching the rest of the container: `EngineCore` constructs the scheduler (and the
connector) at `vllm/v1/engine/core.py:161`, **before** it calls `init_none_hash` at line 226, so the connector sets
`PYTHONHASHSEED` (default `5353`, an existing value wins) in the engine-core process environment only. It then checks
at start-up that `NONE_HASH == hash_fn(PYTHONHASHSEED)`; if not, it wipes its files and disables reuse across restarts.
Python's own `str` hash randomization is unaffected (fixed at interpreter start). Proof and details:
`docs/hash-seed-finding.md`, `tests/t_hash_stability.py` (9/9).

## How it works (short)

- **Resume point (RP)** = exactly the block set the engine's own `HybridKVCacheCoordinator.find_longest_cache_hit`
  returns for the request at the end of its prompt: MLA+indexer blocks 0..N, the block of each of the four mamba groups
  at N, the drafter window at N, each with every prefix-cache key it carries. Taken the step after the last prompt chunk
  (the boundary states are evicted first under pressure, often before the request finishes).
- **Files** = one per block, named by its primary key (content addressed): later turns of a conversation share the MLA
  files and write only new blocks plus one state set (~0.1-0.16 GB/rank per turn). 4 KiB header, O_DIRECT write to
  `.partial`, fsync, rename; CRC-32 per 8 MiB checked on every read.
- **Restore**: when a waiting request's local hit is shallower than a saved RP it extends (by `MIN_GAIN`, default
  16,384 tokens), the scheduler pins the RP blocks still cached, allocates the missing ones, every rank reads them, and
  only if every rank succeeds the original keys are registered on the new blocks. The request is then admitted through
  vLLM's normal lookup (mamba copy-on-write, drafter boundary lookup, retention all unchanged). Bytes are never
  transformed. Any failure falls back to normal prefill; failed blocks never get a key.
- **Pruning**: when a new RP is complete on every rank, older RPs of the same chain beyond `KEEP_PER_CHAIN` (default 2,
  so an edit of the last user message still hits) are deleted.
- **TTL**: `kv_transfer_params.glm53_spill = {ttl_s, session, epoch}` per request (client contract v1.1,
  `docs/client-contract.md`); `ttl_s=0` means "do not keep"; eviction order expired/old-epoch -> no TTL -> shorter TTL ->
  LRU. Requests without the field behave as before.
- **Restart reuse**: an index of complete RPs (sha256-framed, atomic rename) plus an identity signature (connector,
  Python/torch/vLLM versions, config, KV layout, site-packages digest, model files, relevant env, `NONE_HASH` digest);
  every rank reports its files and signature at start-up; adopted only if everything matches on every rank, otherwise
  every rank is wiped.

Full design: `docs/design.md`. Why vLLM's `OffloadingConnector` could not do this for this model (the CPU primary tier
caps a hit, a group without prefix caching makes every hit 0, the drafter window semantics differ): `docs/design.md`
section 1.

## Repository layout

| path | content |
|---|---|
| `code/glm53_nvme_spill_connector.py` | the connector (scheduler side `SpillScheduler`, worker side `SpillWorker` + block file store) |
| `code/patch_nvme_spill.py` | installs the module into vLLM's `kv_connector/v1/`, hands the connector the `KVCacheManager` (`scheduler.py`, 2 lines), exempts only this connector from the `expandable_segments` check (`config/vllm.py`); idempotent, fail-closed |
| `code/patch_tf_bundle.py`, `code/apply_start_sh.py`, `code/start.sh.nvmespill.patch` | integration with the GLM-5.3-Flash EXL3 launcher (bundle hook, `start.sh` edits forwarding every knob to both ranks) |
| `code/spill_quality_probe.py`, `code/conc_quality_probe.py` | output-quality probe of a restored hit (KL vs full prefill, drafter acceptance, proof of restore) |
| `tests/` | CPU unit tests, GPU tests, the replay rig (`tests/rig/`), aggregators |
| `docs/` | design, hash-seed finding, measurements, operations (knobs, abort criteria), client contract, review findings, limitations |
| `results/` | aggregated rig and test outputs (JSON/TXT; no traffic data) |

## Status and limits

The code targets one patched vLLM tree (the GLM-5.3-Flash EXL3 launcher image plus its overlays) and has not been tried
on stock vLLM; the design is general, the code is not. Unfixed: restores started for requests that are then not admitted
soon (B8). Open: a rig counter anomaly. See `docs/limitations.md`. Reproduction: `REPRODUCE.md`.
