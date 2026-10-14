# Measurements

All numbers below are copied from our test logs and production records without rounding. Where we derived a
number ourselves (a percentage, a sum) we say so. Every rig table names the results file it comes from.

Code stages referred to below (see docs/design.md for what each contains):

| Stage | What changed | Connector sha256 |
|---|---|---|
| initial version | first implementation | — |
| fix release | fixes for the independent review findings B1-B8 (docs/review-findings.md) | e8c218eb… |
| prune release | superseded-RP pruning (`KEEP_PER_CHAIN`) + capacity-cap LRU fix | d030f132… |
| TTL+persist first cut | per-session TTL + reuse across restarts | — |
| final version | epoch handling fixes on top of the first cut; in production until the finish release | 1762e23f… |
| published version | final version + a log-format fix (docs/limitations.md); the first publication of this repository | — |
| finish release | B8 policy gate + NVMe-backed blocks first for restores, `DEFAULT_TTL_S`, identity part "code", the log-format fix | eb2e1084… |
| u4 | run-time generated environment names excluded from the identity; a wipe reason names what differs | bd0f589e… |
| u5 | index written when the engine goes idle (+ timer, shutdown); runs in production; this repository's `code/` (comments and docstrings differ, same "code" digest) | 073a2e8c… |

## 1. Definitions

| Term | Meaning |
|---|---|
| RP (resume point) | The deepest set of KV blocks the engine's own coordinator would hit for the next turn of a conversation: MLA+indexer blocks, the four KDA (mamba) group states at the boundary, and the DFlash drafter window. Saved as one file per block per rank. |
| DSH re-read tokens | Σ (prompt tokens − admitted prefix-cache hit) over the requests of the long client conversations (DSH sessions). "All re-read" is the same sum over every request, including one-shot requests. |
| Collapse | A continuation turn of a conversation whose admitted hit is ≤ 23,040 tokens (the 13,824-token shared head + two 4,608-token sections). The production signature of a collapse is a hit of exactly 13,824 (the shared head only). |
| Stall estimate | Σ prefill tokens / 2,036 tok/s: the seconds of prefill that block decode of other sequences. |
| Finished requests | Requests completed within the 4 h simulated window. |
| /rank | Per tensor-parallel rank. With TP=2 each rank writes its own half of the KV to its own local NVMe, so totals for the deployment are 2× the per-rank figure. |

## 2. Unit tests

`tests/test_store.py` drives the real `SpillWorker` with the production tensor layout (22 KV tensors, production page
sizes), 12 blocks, ranks 0 and 1. CPU (production image, CPU only): 19/19. GPU (one GB10, pinned staging + CUDA
stream): 18/18 (the tmpfs ENOSPC case is CPU only). Results: `results/test_store_cpu.txt`, `results/test_store_gpu.txt`.

| Check | Result |
|---|---|
| Bytes per group (MLA+indexer 27,624,960 / mamba 21,233,664 and 18,874,368 / DFlash 11,796,480) | match |
| Save → read back into different block ids, all 6 groups byte-identical | PASS |
| No byte outside the destination rows changes (every tensor, every block compared) | PASS |
| Rank 1 restores its own bytes (different from rank 0's) | PASS |
| CRC mismatch (one data byte flipped) → failure, destination unchanged | PASS |
| Header mismatch (read as another group / another rank's file) → failure | PASS |
| Truncated file (partial write) / missing file → failure | PASS |
| Leftover `.partial` files removed at start-up | PASS |
| One rank fails → the aggregated job fails (ok 1, fail 1) | PASS |
| Free-space floor (`MIN_FREE_GIB`) → not written, failure | PASS |
| Real ENOSPC (40 MB tmpfs) / write stops mid-file (EFBIG via RLIMIT_FSIZE) → failure, no file left behind | PASS |
| `code/patch_nvme_spill.py`: 0/empty = no-op, 2 → SystemExit, second run no-op, drifted anchor or foreign file → SystemExit | PASS |
| `code/apply_start_sh.py`: 7 edits, `bash -n`, backwards replay equals the input, double application refused | PASS |

Unit counts at later stages:

| Stage | Tests |
|---|---|
| fix release | `test_store.py` 19/19 (CPU), 18/18 (GPU); `test_sched_faults.py` 49/49 (the initial version fails the B6 send-order check); `t_alloc_conf.py` (initial tree → ValueError, fixed tree → accepted, NixlConnector still refused); `test_start_sh.sh` 15/15 (23 forwarded names); `test_quality_probe.py` 10/10; a review test (not shipped): initial version = I/O thread dies and the second job is never answered, fix release = thread survives and answers (`results/fix/t_writer_death_old.txt`, `results/fix/t_writer_death_new.txt`); GPU under `expandable_segments`: `test_store.py` 18/18, `t_expandable_gpu.py` 9/9 (`results/fix/unit_final.txt`, `results/fix/gpu_final.txt`) |
| prune release | `test_prune.py` 74/74; 4 deliberate mutations (no in-use check, reference counts ignored, no branch protection, delete before ok) all detected (`results/prune/test_prune_unit.txt`, `results/prune/mutation_887e80c.txt`); `test_sched_faults.py` 49/49; `test_start_sh.sh` 15/15 (24 forwarded names) |
| final version | `test_ttl_persist.py` 133/133, `test_sched_faults.py` 49/49, `test_prune.py` 74/74, `test_store.py` 18/18 (CPU) and 18/18 (GPU), `test_quality_probe.py` 13/13, `test_start_sh.sh` 15/15 (31 forwarded names), `t_hash_stability.py` 9/9 (`results/ttl/unit_*.txt`, `results/ttl/t_hash_stability.txt`, `results/ttl/test_store_gpu.txt`) |
| finish release | `test_finish.py` 66/66 (in development, three deliberate mutations -- gate inverted, "NVMe-backed" always true, default TTL ignored -- were detected by 5 / 6 / 3 FAILs), `test_ttl_persist.py` 133/133, `test_sched_faults.py` 49/49, `test_prune.py` 74/74, `test_store.py` 18/18, `test_quality_probe.py` 22/22, `test_start_sh.sh` 15/15 (32 forwarded names), `t_hash_stability.py` 9/9 (`results/finish/unit/`). No GPU test: the worker's byte path did not change, only scheduler-side Python |
| u4 | `test_u4_env.py` 52/52, `test_finish.py` 66/66, `test_ttl_persist.py` 133/133, `test_sched_faults.py` 49/49, `test_prune.py` 74/74, `test_store.py` 18/18, `test_start_sh.sh` 15/15, `t_hash_stability.py` 9/9 (`results/u4/unit/`) |
| u5 | `test_u5_index.py` 25/25 (on the u4 connector with `--old`: 2/2 reproduce the bug), `test_u4_env.py` 52/52, `test_finish.py` 66/66, `test_ttl_persist.py` 133/133, `test_sched_faults.py` 49/49, `test_prune.py` 74/74, `test_store.py` 18/18, `test_start_sh.sh` 15/15, `t_hash_stability.py` 9/9 (`results/u5/unit/SUMMARY.txt`) |

## 3. The replay rig

`tests/rig/` replays a synthetic workload through real engine code on CPU. Nothing in the rig touches a GPU.

**Real code.** The vLLM `Scheduler`, `KVCacheManager`, hybrid coordinator and `BlockPool` of the production vLLM tree
(the image's vLLM after the launcher's start-up patches), with `code/patch_nvme_spill.py` applied the same way as in
production, and the connector started with the same `--kv-transfer-config`. Admission, prefix lookup, touch, free order
and preemption are the production code paths. The two worker ranks are two real `SpillWorker` instances operating on CPU
tensors with the production tensor structure, pages scaled to 1/576, reading and writing real files.

**Content model.** The content of every block is a function f(key, rank) of its primary key and rank. It is written
when the block gets its key through the normal path (from the fix release on: in the step whose forward pass writes it,
so a copy taken too early is detectable). A restore never writes it, so restored content can only come from the file.
At every admission the content of all hit blocks of all groups is compared on both ranks. In synchronous runs every
saved file is also compared against f(key, rank).

**Stand-ins.** Platform fixed to an unspecified CPU platform; the model architecture check is replaced (the model code
imports GPU kernels); structured output is a no-op stub; `ModelRunnerOutput` is synthesized (prefill chunks produce no
sample, decode rows produce 1 + accepted speculative tokens, calibrated to a mean of 2.4 tokens per step; drafter
proposals are K=7 dummy ids). Time is a simulated clock: prefill = tokens / 2,036 tok/s, one decode step = 0.5 s. In
synchronous mode restore I/O takes zero simulated time; asynchronous mode adds per-block I/O delays.

**Workload.** Fitted to the production admission log (4,758 admissions). Three long conversations of 430,000 / 600,000
/ 780,000 initial tokens sharing a 13,824-token head; each turn adds 2k-8k user tokens and 0.5k-3k output tokens, with
20 s - 5 min between turns. One-shot requests of 20k-100k tokens arrive with an exponential gap of mean 88 s. At most 8
concurrent sequences. A seed fixes the whole request stream, so OFF and ON runs of the same seed see identical requests.

**Run modes.** 4 h runs use `--no-drain`: the rig's drain phase hits a known assert (the `block_mask` length in
`cache_full_blocks` of the production code) with the connector OFF as well. 0.5 h correctness runs drain at the end and
check strict invariants at every step: ref_cnt == 0 (non-null) ⟷ block in the free queue in both directions, the free
queue walk count, no negative ref_cnt; at every admission the hit block hashes equal the request's block hashes; after
draining all 582 blocks are free with ref_cnt 0 and no connector pin.

The OFF runs reproduce, metric for metric, the baseline of our earlier re-read investigation (summarized in
docs/design.md section 'Problem'), so the rig changes do not alter engine behavior.

## 4. Rig results, initial version

### 4.1 4 h × 5 seeds

Source: `results/m_off_s*.json`, `results/m_on_s*.json`, `results/aggregate.txt`.

| Seed | DSH re-read OFF → ON | All re-read OFF → ON | Collapses OFF → ON | Stall estimate s | Finished requests | Restores (pinned until admission) | Blocks read | Written |
|---|---|---|---|---|---|---|---|---|
| 1 | 5,843,258 → 2,126,007 | 6,846,990 → 3,274,227 | 6 → 0 | 3,286 → 1,514 | 51 → 59 | 13 (9) | 258 | 14.6 GiB |
| 2 | 7,772,703 → 2,135,765 | 9,155,187 → 3,892,501 | 9 → 0 | 4,398 → 1,833 | 49 → 67 | 20 (12) | 395 | 14.9 GiB |
| 3 | 6,115,328 → 2,085,058 | 7,152,407 → 3,197,541 | 7 → 0 | 3,214 → 1,498 | 44 → 49 | 6 (6) | 119 | 13.9 GiB |
| 4 | 6,852,561 → 2,122,261 | 7,763,243 → 3,443,249 | 8 → 0 | 3,431 → 1,609 | 51 → 64 | 52 (9) | 608 | 14.7 GiB |
| 5 | 5,277,269 → 2,063,209 | 6,270,280 → 3,288,369 | 6 → 0 | 2,917 → 1,541 | 47 → 55 | 10 (8) | 217 | 13.9 GiB |
| Mean | **6,372,224 → 2,106,460 (−66.9%)** | 7,437,621 → 3,419,177 (−54.0%) | **7.2 → 0** | 3,449 → 1,599 (−54%) | 242 → 294 (+21.5%) | 20.2 | 319 | 3.6 GiB/h |

- The remaining ~2.1 M DSH re-read tokens are the first ingest of the three conversations (430k + 590k + 760k + head,
  as measured). No collapse after the first turn in any seed.
- Restore failures 0; `restore short` (restored, but the normal lookup does not reach the restored depth) 0; 4,700-6,600
  verified blocks per seed with 0 mismatches; every restored block (119-608 per seed) verified with 0 mismatches; no
  preemption (same as OFF).
- Every RP contained the four mamba group states and the DFlash window (2 blocks).
- Most of the writes are the first saves of the three conversations (≈10.9 GB). Later turns write ≈ 1-2 new MLA blocks
  + 4 mamba + 2 DFlash blocks (≈ 0.1-0.16 GB/rank).

### 4.2 0.5 h, drained, strict invariants

Source: `results/c_on_s2.json`, `results/f_missing_s2.json`, `results/f_crc_s3.json`, `results/f_save_s4.json`,
`results/c_hw_s5.json`.

| Run | Content | Result |
|---|---|---|
| c_on_s2 | normal | collapses 0, 11 restores succeeded, 5,408 blocks verified, 0 mismatches; after draining 582 free, ref 0, 0 pins, 0 violations |
| f_missing_s2 | every 3rd restore: delete one rank-1 file | 4 injected → 4 failures (fall back to normal prefill); every RP using the broken file dropped and re-saved; collapses 0; no leak |
| f_crc_s3 | same, flip one byte of a rank-1 file | 2 injected → 2 failures; collapses 0; no leak |
| f_save_s4 | every 3rd rank-1 block write fails with an I/O error | 31 save failures → 0 RPs (same behavior as OFF: 7 collapses); all pins released; no leak |
| c_hw_s5 | `HIGHWATER=0.97` (below the threshold saves are deferred, written once usage crosses it) | 7 deferred writes; collapses 0; no leak |

Defects the rig found during development (all fixed in the initial version): taking the RP when the request finishes is
too late (the boundary state is already gone; moved to right after the prompt is computed); two requests restoring the
same RP read the same blocks twice (merged into one load); delete and rewrite in the same step were ordered wrongly
(deletes first now); LRU deletion removed files shared with the new RP (RPs that share files with it are not evicted);
pins of requests waiting long for admission (MLA pins released after 5 s, state blocks, max 64, kept until admission).

### 4.3 Speed on GB10 (initial version)

`tests/bench_gpu.py`: real `SpillWorker`, production-shaped tensors on the GPU, 600k tokens' worth = 131 MLA + 4 mamba +
2 DFlash blocks = 3,723,575,296 B per rank, O_DIRECT, 32 MiB pinned staging, CRC-32 per 8 MiB. One GB10 with its
internal NVMe, one rank. Source: `results/bench_gpu.json` (one attached run: write 1.554 s, read 1.382 s, kernel 228.3 GB/s alone,
226.5 / 227.7 GB/s during write / read); the ranges below also include earlier review runs whose raw files are not attached.

| Item | Result |
|---|---|
| Write (GPU → NVMe, including fsync) | 1.51-1.55 s (2.4-2.5 GB/s) |
| Read (NVMe → GPU, including CRC check) | 1.38-1.45 s (2.6-2.7 GB/s), byte-identical |
| Bandwidth-bound kernel (back-to-back 2 GiB D2D copies on another stream), alone | 228-233 GB/s |
| Same kernel during a write / during a read | 226-231 GB/s (within −1%) |
| Write / read time while the kernel runs | 3.4 s / 3.0 s (the I/O side slows down through GIL contention between Python threads) |

In one earlier review run (raw file not attached) the kernel bandwidth went 232.8 → 230.6 / 231.2 GB/s. Actual restores in the rig average 12-30 blocks
(only what is missing: mainly mamba states and the trailing MLA blocks) = 0.3-0.8 GB, i.e. 0.1-0.4 s at these speeds. The
worst case of reading all 600k tokens is ~1.5 s (~3 s under load), against 5-15 minutes for re-reading 600k tokens.

## 5. Rig results after the review fixes (fix release)

Conditions: the rig builds `VllmConfig` with the `kv_transfer_config`, and **both arms run with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (that the fixed tree accepts it is itself the B1 check). The
content model writes in the step whose forward writes the block (detects B6). Synchronous runs compare every saved file
with f(key, rank). Experiment memory limit 8 GiB × 2 concurrent jobs.

### 5.1 4 h × 5 seeds, synchronous

Source: `results/fix/m_off_s*.json`, `results/fix/m_on_s*.json`, `results/fix/aggregate_m_on.txt`.

| Seed | DSH re-read OFF → ON | All re-read OFF → ON | Collapses OFF → ON | Stall estimate s | Admissions | Restores started/admitted | Blocks read | Saved files verified / mismatches | Max pinned (total / connector-only) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5,843,258 → 2,126,007 | 6,846,990 → 3,274,227 | 6 → 0 | 3,286 → 1,514 | 55 → 60 | 13/9 | 258 | 1,284 / 0 | 328 / 12 |
| 2 | 7,772,703 → 2,135,765 | 9,155,187 → 3,892,501 | 9 → 0 | 4,398 → 1,833 | 52 → 69 | 20/12 | 395 | 1,320 / 0 | 329 / 11 |
| 3 | 6,115,328 → 2,085,058 | 7,152,407 → 3,197,541 | 7 → 0 | 3,214 → 1,498 | 46 → 51 | 6/6 | 119 | 1,212 / 0 | 255 / 12 |
| 4 | 6,852,561 → 2,122,261 | 7,763,243 → 3,443,249 | 8 → 0 | 3,431 → 1,609 | 52 → 66 | 52/9 | 608 | 1,296 / 0 | 350 / 12 |
| 5 | 5,277,269 → 2,063,209 | 6,270,280 → 3,288,369 | 6 → 0 | 2,917 → 1,541 | 48 → 57 | 10/8 | 217 | 1,214 / 0 | 305 / 11 |
| Mean | **6,372,224 → 2,106,460 (−66.9%)** | −54.0% | **7.2 → 0** | 3,449 → 1,599 | | 20.2/8.8 | 319 | 0 mismatches | max 350 / 12 |

- The five OFF runs equal the initial version's OFF runs on every metric. The ON runs equal the initial version's ON
  runs down to DSH re-read, collapses, restore count and blocks read: the one-step save delay (B6), deadlines and the
  breaker do not change normal behavior.
- Content checks 4,710-6,578 per seed, 0 mismatches; breaker 0; deadline expiries 0; failed jobs 0; saves skipped by the
  pin cap 0.
- "Max pinned, total" includes blocks also referenced by running requests (max 350). The first cap design (half the
  pool = 291, counted over all pins) would have stopped restores in normal operation; the cap now counts only save pins
  held by the connector alone (max 12).

### 5.2 4 h × 5 seeds, asynchronous completion with rank skew

Per-block I/O delay rank 0: 0-80 ms, rank 1: 0-250 ms. Source: `results/fix/x_on_s*.json`,
`results/fix/aggregate_x_on.txt`.

- DSH re-read 6,372,224 → 2,108,750 (**−66.9%**); the independent review's own asynchronous run gave 2,108,028.
- Collapses 7.2 → 0 (0 in every seed). Restores 13.4 per seed, failures 0, short 0, content mismatches 0.
- Deadline expiries 0. Deadlines were raised to 900/600 s for this run only, because the delay model is 10-25× slower
  than the real device.
- Max connector-only save pins 311 (one save takes thousands of simulated seconds in this mode). The cap only stops new
  saves; skipped saves 0.

### 5.3 0.5 h correctness and fault injection

Drained, strict per-step invariants. Deadlines shortened to 3 s (load) / 5 s (save) for the fault runs only (defaults
30 / 120 s). Source: `results/fix/faults_summary.txt` and `results/fix/<run>.json`.

| Run | Injected | Collapses | Restores ok / failed / to prefill | Saves ok / failed | Expired save / load / late answer | Breaker | Content check (blocks / mismatches) | End: free / pinned / quarantined / unfinished |
|---|---|---|---|---|---|---|---|---|
| c_on_s2 normal | 0 | 0 | 12/0/0 | 32/0 | 0/0/0 | - | 5,381/0, files 1,236/0 | 582 / 0 / 0 / 0 |
| c_hw_s5 `HIGHWATER=0.97` | 0 | 0 | 10/0/0 | 29/0 | 0/0/0 | - | 5,225/0 | 582 / 0 / 0 / 0 |
| f_missing_s2 one rank's file deleted | 5 | 1 | 12/5/5 | 33/0 | 0/0/0 | - | 5,114/0 | 582 / 0 / 0 / 0 |
| f_crc_s3 CRC corruption | 2 | 0 | 6/2/2 | 29/0 | 0/0/0 | - | 4,519/0 | 582 / 0 / 0 / 0 |
| f_save_s4 every 3rd write of one rank EIO | 7 | 7 | 0/0/0 | 0/7 | 0/0/0 | open (6 consecutive failures) | 3,826/0 | 582 / 0 / 0 / 0 |
| a_plain_s2 asynchronous | 0 | 0 | 14/0/0 | 32/0 | 0/0/0 | - | 5,656/0 | 582 / 0 / 0 / 0 |
| a_abort_s4 asynchronous + 10 aborts during loads | 0 | 0 | 10/0/0 | 21/0 | 0/0/0 | - | 3,454/0 | 582 / 0 / 0 / 0 |
| a_crc_s3 asynchronous + CRC | 4 | 0 | 9/4/4 | 27/0 | 0/0/0 | - | 4,566/0 | 582 / 0 / 0 / 0 |
| n_noload_s2 negative control (pretends to read) | 261 | 0 | 12/0/0 | 32/0 | 0/0/0 | - | **5,381/1,238** (the check works) | 582 / 0 / 0 / 0 |
| **a_rofs_s2 the review's B2 injection, unchanged** | 41 exceptions | 11 | 0/0/0 | 0/1 | 0/0/0 | open (worker exception) | 3,470/0 | **582 / 0 / 0 / 0** (initial version: free 75, pinned 507, unfinished 27) |
| a_rofs_os_s2 EROFS at the OS layer | - | 11 | 0/0/0 | 0/11 | 0/0/0 | open (6 consecutive failures) | 3,564/0 | 582 / 0 / 0 / 0 |
| h_load_s2 rank 1 never answers a load | 1 | 14 | 1/0/1 | 6/0 | 0/1/0 | open (load deadline 3 s) | 3,125/0 | 565 / 17 / **17** / 0 |
| h_loadlate_s2 same, answers after 8 s | 1 | 11 | 1/0/1 | 6/0 | 0/1/1 | open | 3,465/0 | 582 / 0 / 0 / 0 |
| h_save_s2 rank 1 writes stall for 12 s | 1 | 11 | 1/0/0 | 1/1 | 1/0/1 | open (save deadline 5 s) | 3,470/0 | 582 / 0 / 0 / 0 |
| k_load_s2 rank 1's load answer is lost | 1 | 10 | 1/0/1 | 6/0 | 0/1/0 | open | 3,704/0 | 565 / 17 / 17 / 0 |
| k_save_s2 rank 1's save answer is lost | 1 | 11 | 1/0/0 | 2/1 | 1/0/0 | open | 3,470/0 | 582 / 0 / 0 / 0 |
| race_new_s2 force a same-step copy (fixed code) | 286 planned | 0 | 12/0/0 | | 0/0/0 | - | 5,381/0, **files 2,650/0** | 582 / 0 / 0 / 0 |
| race_oldcode_s2 same, initial version | 286 planned | 0 | 12/0/0 | | | | **5,381/1,238, files 2,650/2,650** | 582 |

- Runs where the breaker opened show 10-14 collapses, which is OFF behavior (OFF under the same conditions: c_off_s2 11,
  c_off_s4 7 = f_save_s4's 7; `results/fix/c_off_s*.json`).
- The 17 quarantined blocks are by design: a load that never answers keeps its new blocks out of circulation until the
  rank answers, so a late write cannot corrupt another request's KV. All requests finish; the other 565 blocks return.
- The single collapse in f_missing: a failed restore drops the RP and falls back to normal prefill; that conversation
  cannot be restored until its next save.
- The deadline, breaker and quarantine logic is also covered by the fake-pool unit test (49 items).

### 5.4 Speed rerun on GB10 (fix release, under `expandable_segments`)

Source: `results/fix/gpu_final.txt`. 600k tokens' worth (3.72 GB): write 1.40 s, read 1.24 s, byte-identical.
Bandwidth kernel alone 220.1 GB/s; during write / read 217.1 / 216.5 GB/s (−1.4% / −1.6%).
`t_expandable_gpu.py`: KV inside an expandable segment keeps its address and I/O stays byte-identical after allocator
churn (`empty_cache`), 9/9.

## 6. Pruning superseded resume points (prune release)

Pruning (`KEEP_PER_CHAIN`, default 2) plus the capacity-cap LRU fix. The rig changes are observation only: disk usage
over time, which RP each restore used, a cause split of DSH re-reads, and a branch-stress mode. 4 h runs used the 8 GiB
× 2 memory budget with the rig's per-step free-list log disabled (`NVMESPILL_RIG_FREE_LOG=0`; with 4-5 GiB they ran out of
memory). OFF and the pre-pruning ON numbers are those of section 5.1 (same seeds = same requests).
Source: `results/prune/aggregate.txt`.

### 6.1 Normal workload, 4 h × 5 seeds

Re-reads, collapses and restores do not depend on K; only disk usage drops.

| Condition | DSH re-read | Collapses | Restores | Disk/rank GiB: peak → after 4 h | Growth in the last 2 h | RPs at end | Pruned RPs / GiB |
|---|---|---|---|---|---|---|---|
| OFF (section 5.1) | 6,372,224 | 7.2 | 0 | — | — | — | — |
| ON K=0 (= pre-pruning behavior, equals the fix release on every metric) | 2,106,460 (−66.9%) | 0 | 20.2 | 14.39 (max 14.85) | **1.31 GiB/h** | 33.8 | 0 |
| **ON K=2 (default)** | **2,106,460 (−66.9%)** | **0** | 20.2 | **11.85** (11.98) → 11.73 | **0.37 GiB/h (−72%)** | 6.2 | 27.6 / 2.66 |
| ON K=2, asynchronous + rank skew (2 seeds) | same as the fix release's asynchronous runs | 0 | | 11.93 | | 6.5 | 29.5 |

In a 4 h run the first saves of the three conversations (≈10.3 GiB) dominate, so the peak drops by only 18%. The
steady growth of conversations that keep extending drops from 1.31 to 0.37 GiB/h (the remainder is the new MLA blocks of
the extension). In production logs, three turns of a 244k-token conversation that each grew by one block wrote 131 MB/rank
per turn, of which 104 MB (4 mamba + 2 DFlash blocks) is state that the next turn makes useless; K=2 keeps two such sets.

### 6.2 Which RP a restore hits

From the record of every restore (`restore_newer_alive_by_branch`: how many newer RPs of the same chain were alive when
the restore used an RP):

- A normal next turn, a regeneration (same prompt resent, also covers retries), and a DSH auto-compaction (whole
  conversation + instruction = an extension of the newest): **always the newest** (normal 101/101, regeneration 14/14).
  K=1 would suffice for these.
- Editing the last user message: the RP before the newest (when the newest RP boundary lies after the start of that
  message). Editing the message before that: the RP two back.
- A rig "turn" is one request. In a DSH tool-use loop one user message is followed by many tool-call requests and a new
  RP appears every 4,608 tokens, so editing a message from the middle of such a loop can reach back further than K
  (then: whatever is still on the GPU + normal prefill).

### 6.3 Branch stress, 4 h × 5 seeds

25% of DSH turns branch: regeneration 10%, edit of the last message 10%, edit of the message before it 5%.

| K | DSH re-read | Collapses (per seed) | Re-read on last-message edits (13) | On edits two back (6) | Disk peak / growth in the last 2 h | RPs at end |
|---|---|---|---|---|---|---|
| 1 | 2,935,954 | 1.4 [3,2,0,1,1] | 1,322,724 (2 collapses) | 3,168,352 (5 collapses) | 11.80 / 0.40 GiB/h | 8.2 |
| **2** | **2,236,635** | **0.2** [1,0,0,0,0] | **106,212 (0)** | 836,704 (1 collapse) | 12.27 / 0.54 | 12.8 |
| 3 | 2,077,876 | 0 | 106,212 (0) | 53,344 (0) | 12.54 / 0.55 | 16.0 |

Default K=2 covers regeneration, retries, auto-compaction and editing the last message (K=1 collapses on edits); only
edits two messages back remain (0.2 per 4 h under this 5% stress). K=3 costs one more state set per conversation
(≈104 MB/rank), which is also pruned eventually. Abandoned sibling branches keep "the RP before" of their own branch, so
RPs at the end rise from 6 to 13-16.

### 6.4 What the remaining re-reads are (K=0, 5 seeds)

Source: `results/prune/reread_causes.txt`. DSH 2,106,460 tokens per 4 h split by cause:

| Cause | Tokens | Share |
|---|---|---|
| (d) First ingest of a conversation (430k/590k/760k as measured; unavoidable in the rig; in production the first ingest **after a restart** falls here) | 1,781,561 | 84.6% |
| (f) Tokens new in that turn (after the previous turn's prompt + reply) | 178,284 | 8.5% |
| (e) RP granularity: the previous reply + the tail beyond the last state boundary (the RP is the deepest boundary right after the prompt is computed) | 129,105 | 6.1% |
| (e) Next turn arrived while the previous one was still computing (an artifact of the rig's fixed schedule; DSH sends after the reply) | 16,589 | 0.8% |
| (c) Not restored because the gain was below `MIN_GAIN` | 922 | 0.0% |
| (a) First turn not saved (`SAVE_POLICY=repeat`) / (b) pruned, LRU-evicted or unfinished RPs / (c) `MIN_TOKENS` | 0 | 0% |

### 6.5 SAVE_POLICY=all (K=2, 5 seeds)

Re-reads do not drop by a single token (2,106,460, identical split). Writes 3.60 → **4.65 GiB/h/rank (+29%)**, disk after
4 h 11.73 → 15.66 GiB (+34%). In the rig the first turn is never evicted from the GPU before the second turn (turn gaps
are 20 s - 5 min), so cause (a) is 0; the extra writes are one-shot requests (≥ 65,536 tokens) that are never used again.
Conversations with hours between turns could produce (a) in production, which this rig cannot measure. This analysis
pointed to reuse across restarts as the next improvement: in production most of (d) is the re-read of every
conversation after each restart, ~1.78 M tokens for three conversations ≈ 15 min of stall at 2,036 tok/s, ~28 min at
the measured cold prefill rate of 1,050 tok/s.

### 6.6 Capacity cap: a bug found and fixed

Cap `GIB=13`/rank (below K=0's 4 h peak of 14.4), 3 seeds. Before the fix, `_make_room` did not evict any RP that
shares files with the new RP. Every DSH conversation shares the 13,824-token head (3 blocks), so once the cap was reached
**no DSH RP could be evicted and new saves silently stopped** (`rp_skipped_cap`).

| | rp_skipped_cap | LRU evictions | DSH re-read | Disk peak |
|---|---|---|---|---|
| K=0, before the fix (c_k0) | 13 / 15 / 7 | 0 | 2,148,611 (+1.6%) | 12.92 |
| K=0, after the LRU fix (cn_k0) | 0 | 17 / 20 / 10 | 2,115,610 (same as without a cap) | 13.00 |
| K=2 (before and after the fix, c_k2 = cn_k2) | 0 | 0 | 2,115,610 | 11.85 |

Fix: references are first added to the existing files the new RP reuses, then room is made. An evicted RP loses only its
own files; shared files stay. Evictions are planned on a copy of the reference counts and executed only if they free
enough room (a save that cannot fit evicts nothing). Below the cap nothing changes (early return, identical reference
totals; c_k2 = cn_k2, m_k2 = mn_k2, c_on = cn_on on every metric). With `GIB=8` (below the ≈11 GiB of the three first
saves), strict invariants and draining: 25-27 evictions, content checks 4,672/0, files 5,050/0, 582 free, 0 pins.

### 6.7 Fault injection with pruning

0.5 h, drained, strict invariants, default K=2 (`results/prune/faults_summary.txt`): the same 17 runs as section 5.3,
all with 0 unfinished requests and 0 pins (the 17 quarantined blocks of h_load / k_load are the same by-design
remainder), 0 content mismatches (only the negative control n_noload detects 1,238), race probe files 2,650/0.
Synchronous runs (c_on, c_hw, f_missing, f_crc, f_save, k_save, …) equal section 5.3 in restores, saves and collapses.
Robustness against broken files versus K (`results/prune/aggregate_faults_vs_k.txt`): synchronous CRC / missing-file
runs × 3 seeds are identical for K=0 and K=2; asynchronous CRC gives 2.0 / 2.3 / 2.3 collapses for K=0/2/3 (3 seeds;
run-to-run spread at the same K=3: 5/0/2), so no K effect is visible.

**An open rig observation unrelated to pruning.** In asynchronous + CRC 0.5 h runs, `hit_hash_violations` occasionally
counts 1 (once in a K=2 run and once in a K=0 run, so independent of pruning). The detailed record
(`results/prune/acv_k0_r1_s2.viol.json`): the recorded hit was 456,192 (from the preceding lookup) while the request's
computed tokens were 446,976; the check tripped on the block one past the actual hit, which had no key yet. Full content
checks show 0 mismatches. We read it as the rig's record not following a hit that shrank while the request waited (the
same pattern as the lost-hit events in our earlier re-read investigation), not as a correctness problem; it is not yet
fully explained.

## 7. TTL and reuse across restarts (TTL+persist first cut, final version, finish release, u4, u5)

The 4 h matrices (sections 7.2 and 7.3; `results/ttl/*.json`, `results/ttl/aggregate.txt`) ran on the TTL+persist first
cut; the same seeds rerun on the final version are in `results/ttl/final/` (section 7.4). Rig: real Scheduler /
KVCacheManager / connector and two real `SpillWorker` ranks, as before. The client fields travel the vLLM path
(`SamplingParams.extra_args["kv_transfer_params"]` → `Request.kv_transfer_params`). A restart rebuilds Scheduler,
connector and both ranks as new objects (in the EngineCore order: Scheduler, then `init_none_hash`), with an empty GPU
cache; running and waiting requests are resent with the same content after a 10 min outage (the client's retry). Two
variants: clean stop (`connector.shutdown` writes the index) and crash (it does not).

### 7.1 Block-hash stability

`tests/t_hash_stability.py` 9/9 (`results/ttl/t_hash_stability.txt`). Two fresh interpreters without `PYTHONHASHSEED`
each build the rig's real Scheduler (with the connector), then `init_none_hash`, and hash the same token sequence:

| | NONE_HASH | Hash of the first block | Python `hash(str)` |
|---|---|---|---|
| PERSIST=0 (= behavior before the final version) | random, different per process | **different** | different |
| PERSIST=1 | identical in both processes (sha256 of "5353") | **identical** | still different (str hash randomization kept) |
| User sets PYTHONHASHSEED=77 | identical (from 77) | identical (differs from the 5353 case) | — |

The test also checks that in the production `v1/engine/core.py`, `Scheduler(` precedes `init_none_hash(caching_hash_fn)`
(character offset 5539 < 8733).

### 7.2 Normal workload with TTL fields, 4 h × 5 seeds

The three conversations carry 1h / 1d / 3d.

| Condition | DSH re-read | Collapses | All re-read | Written GiB/rank (4 h) | Disk peak GiB/rank |
|---|---|---|---|---|---|
| OFF (section 5.1) | 6,372,224 | 7.2 | 7,437,621 | — | — |
| ON, no fields (new code, `m_new`) | **2,106,460 (−66.9%)** | **0** | 3,419,177 | 14.39 | 11.85 |
| ON, TTL fields (`m_ttl3`, `TTL_SAVE_POLICY=all`) | **2,106,460 (−66.9%)** | **0** | 3,419,177 | 14.68 | 11.85 |
| ON, TTL fields, `TTL_SAVE_POLICY=repeat` (2 seeds) | identical to `all` for the same seeds | 0 | — | `all` − 0.29 for the same seeds | same |

Without fields the new code equals the prune release with K=2 on every metric. TTL fields (within their lifetime) do not
change re-reads. **Saving the first turn too (`all`) costs +0.29 GiB/rank per 4 h (+2.0%, about one state set ≈ 0.1 GB per
conversation)**: the MLA blocks are written at the second turn anyway, only the first turn's boundary state is extra. It
pays off when a restart or a long idle period follows the first turn (in the 0.5 h restart runs, conversation c0, which had
finished only one turn, restored 438k tokens).

### 7.3 Restart in the middle (2 h in, 10 min down), TTL fields on 3 conversations, 4 h × 5 seeds

| Condition | DSH re-read after the restart | of which: first turn of each conversation after it | Collapses after | Whole 4 h: DSH re-read / collapses | Written GiB/rank |
|---|---|---|---|---|---|
| OFF (no connector, 2 seeds) | 4,575,668 | 1,882,795 | 7.0 | 8,686,406 / 10.5 | — |
| ON, PERSIST=0 (= behavior before the final version) | **2,028,530** | **1,903,966** (all three conversations restart from a hit of 0 or 13,824) | 3.0 | 3,965,127 / 3.0 | 24.63 |
| **ON, PERSIST=1** | **174,139 (−91%)** | **24,823 (−98.7%)** | **0** | **2,110,736 / 0** (≈ the 2,106,460 without restart) | **14.46** |

- With PERSIST=1 all 5 seeds log `persist: adopted` (6.2 RPs, 11.0 GiB/rank adopted, 0 RPs dropped). The first turn after
  the restart is restored **up to the last state boundary**, e.g. c0 477,232 → hit 465,408, c1 630,779 → 626,688, c2
  810,605 → 801,792 (seed 1; the rest is the previous reply + the new message).
- Writes also drop (24.63 → 14.46 GiB/rank): without reuse every restart forces a re-read and a new first save of every
  conversation.
- Content checks: every restored block verified, 0 mismatches; saved files 0 mismatches (both conditions).

### 7.4 Rerun on the final version

Source: `results/ttl/final/aggregate.txt`.

- Normal workload `m_new_s1` 2,126,007, 0 collapses; restart `rs_p1_s1` 202,776 after the restart. Both identical to the
  first cut for the same seed.
- **Crash restart** (`rs_cr`, `connector.shutdown` not called, the index is whatever the last periodic write left, 3
  seeds): identical to the clean stop on every metric (after restart 202,776 / 179,391 / 171,704; 6 RPs adopted, 0
  orphans). The index was written 143-155 times per 4 h; building it took at most 8-10 ms; about 390 KB.
  (The first cut counted the index interval in wall-clock time; in the fast simulated clock the crash-time index then
  held 0 RPs. The interval is now counted on the engine clock.)

### 7.5 Mixed TTLs, 4 conversations, 4 h × 5 seeds

c0: 1h, idle 100 min after its third turn. c1: 1d, DSH auto-compaction at its seventh turn (epoch + 1). c2: 3d. c3: ttl 0
("do not save"), 150k tokens. Source: `results/ttl/final/aggregate.txt` (`mx2_off`, `mx2_on`, `mx2_nottl`).

| Conversation | TTL | OFF | ON, fields | vs OFF | ON, no fields (control) | Collapses OFF / ON / no fields |
|---|---|---|---|---|---|---|
| c0 | 1h (idle) | 877,065 | 539,760 | −38.5% | 539,760 | 1.0 / 0.2 / 0.2 |
| c1 | 1d (compaction) | 968,155 | 747,603 | −22.8% | 753,596 | 0.4 / 0 / 0 |
| c2 | 3d | 1,468,854 | 863,047 | −41.2% | 1,016,955 | 0.8 / 0 / 0.2 |
| c3 | 0 (do not save) | 245,654 | **523,321** | **+113%** | 221,737 | 0.2 / 1.8 / 0 |
| Total | | 3,559,729 | 2,673,731 | −24.9% | 2,532,047 | 2.4 / 2.0 / 0.4 |

- Per seed: expired 1.2, dropped by epoch 2.0, not written because of ttl 0 11.0 (every save opportunity of c3),
  first-turn saves 4.0. Content checks 0 mismatches.
- **c3 ("do not save") does worse than OFF.** It is never on NVMe, so it relies on the GPU cache alone, and with the
  connector ON other conversations' restores allocate and pin blocks before admission (review finding B8: in this
  workload 21.2 restores started per seed, 4.8 admitted, 21.0 pin timeouts), evicting c3's GPU prefix first. The same
  mechanism already affects one-shot requests in the earlier releases (section 5.1 runs: one-shot re-reads OFF 1,065,397
  → ON 1,312,717 = +23%). A ttl 0 conversation joins that "not saved" side and pays the same cost. The fix would be B8
  (do not restore ahead for a request unlikely to be admitted soon); see docs/limitations.md. **Fixed in the finish
  release: +2.5% (section 7.9); the "+23%" for one-shot requests turned out to be a count effect (more admissions), per
  admission it is −1.7%.**
- Two epoch defects this rig found during development: (1) a newer-epoch request only **waiting in the queue** already
  dropped older-epoch RPs, so an older-epoch request still queued behind it could not restore (c1 +88 to +134%); epochs
  are now recorded at **admission and at save**. (2) Saves of an older epoch were discarded; they are now saved with a
  "stale" mark, deleted after 30 minutes without use, and evicted first at the cap. DSH sends one session's requests in
  order, so production rarely hits either case; both fixes are on the safe side.
- Overloaded variant (c3 300k; idle and compaction cannot get through the queue, `tests/rig/env/ttlmix_overload.env`):
  OFF 7,785,404 → final version ON 4,668,625 (−40%; c0 −62%, c1 −55%, c2 −63%, c3 +103%; `results/ttl/final/mxo_on_s*.json`).
  The first cut gave ON 5,128,846 (c1 −8%) because of defects (1) and (2) (`results/ttl/aggregate.txt`).

### 7.6 Saving the first turn (`TTL_SAVE_POLICY`)

- 4 h × 2 seeds: `all` and `repeat` give **identical re-reads**; `all` writes +0.29 GiB/rank (+2.0%). The 0.5 h restart
  runs (`pr_clean` / `pr_cleanrep`, seeds 2 and 3) are also identical: a DSH conversation sends several requests per user
  message, so `repeat` saves at the second request anyway.
- `all` matters when a conversation the user explicitly asked to keep consists of a single request (≥ 65,536 tokens) and
  is then left idle or crosses a restart (paste a document, ask once, come back the next day). This workload has no such
  case, so there is no number for it. The cost is +2% writes, hence default `all`.
- `MIN_TOKENS` is not relaxed for TTL requests (`TTL_MIN_TOKENS=0` = same 65,536): re-reading less than 65.5k tokens
  costs under 32 s at 2,036 tok/s, and child sessions (delegated sub-tasks that the client runs as separate
  conversations, which inherit the TTL with their own session) would otherwise write many short requests.

### 7.7 Fault injection and restart faults (final version)

The 17 runs of sections 5.3 / 6.7 (without race_oldcode, incl. the negative control) plus 11 restart-fault runs (28 rows of table (4)), on the final version
(`results/ttl/final_faults/aggregate.txt`, `results/ttl/final/aggregate.txt`): restores, saves, collapses and content
checks equal section 6.7 (asynchronous runs within their spread); all runs end with 0 unfinished requests and 0 pins
(the 17 quarantined blocks of h_load / k_load are by design); only the negative control n_noload detects 1,238
mismatches; race probe files 2,650/0.

Restart faults (injected while the engine is down, seed 2):

| Injection | Decision | Result | End state |
|---|---|---|---|
| none (clean stop / crash) | adopt 2 RPs | c0's first turn after the restart: 438,254 → 437,760 hit | 582 free, 0 pins, 0 unfinished |
| PERSIST=0 | (previous behavior) | same turn: 0 hit (full re-read) | same |
| index with one flipped byte / index cut in half | wipe all (length / sha256 mismatch) | 0 hit | same |
| a patch flag in the environment changes | wipe all (`scheduler signature differs: env`) | 0 hit | same |
| only rank 1's signature differs (rank mismatch) | wipe all (`rank 1: salt`) | 0 hit | same |
| one indexed rank-1 file missing | adopt 1, drop 1, delete 15 orphans | 428,544 hit from the RP before | same |
| one indexed rank-1 file with a flipped byte | adopt 2 → CRC failure on load → that RP dropped, normal prefill | 428,544 hit from the RP before | same |

### 7.8 GPU round trip across a process restart

`tests/t_restart_gpu.py` on one GB10 with production page sizes (`results/ttl/gpu_restart_roundtrip.txt`): save → new
container = new process → adopt → restore. 6 blocks × 2 ranks restored, byte-identical to what was saved **and to an
in-process restore (the earlier path)**. The new process's GPU memory is filled with different random data, so the bytes
can only come from the files. Start-up scan 0.40 s, adoption 1.38 s (including the site-packages digest), restore 0.08 s.
Container 12 GiB, GPU capped at 8 GB, no concurrent GPU test.

Start-up cost (`tests/bench_persist.py`, production image; `results/ttl/bench_persist.json`,
`results/ttl/bench_persist_realistic.json`):

| Item | Time / size |
|---|---|
| site-packages digest (separate thread) | cold 4.4 s (one measurement in the production image, raw output not attached) / warm 0.9-1.0 s (`results/ttl/bench_persist*.json`) |
| Rank start-up scan | 6,000 files 16 ms, 12,000 files 37 ms |
| Index for 60 RPs / 5,220 files (≈132 GiB/rank) | 2.3 MB; build 14 ms the first time, 4.4 ms per write after that; read + verify 6 ms; adoption decision 152 ms (once at start-up) |
| Extreme case 300 RPs / 26,100 files | 11 MB; build 40 ms; adoption 314 ms |

### 7.9 Finish release: B8 fix, `DEFAULT_TTL_S`, fault suite

Source: `results/finish/aggregate.txt` (`tests/aggregate_finish.py`; runs `results/finish/fz_*`, `ff_*`, `dt_*`, `bf_*`,
`dg_*`, ablation `results/finish/gate_only/fm_*`). OFF and the previous ON (the final version, connector `1762e23f`) are
the existing runs of the same seeds (`results/fix`, `results/ttl`, `results/ttl/final`). The rig's added observations
(eviction attribution, preemption counts) do not change behaviour: `dg_mx_off_s1` / `dg_mx_on_s1` equal `ttl/final`
`mx2_off_s1` / `mx2_on_s1` on every field.

**Acceptance criterion** for the requests that are not saved (one-shot requests and the `ttl_s=0` conversation): not
worse than OFF, i.e. over the 5 seeds **re-read per admitted request within OFF + 10%** and **collapses within OFF + 1**.
Per admission, because with the connector more requests are admitted in the 4 hours (more throughput), so totals grow
with the count (this is how the earlier "+23%" for one-shot requests was a misreading). 10% is about one collapse over
the 5 seeds: one collapse of the 150k-token `ttl_s=0` conversation ≈ 150k tokens over its 53 admissions ≈ +2.8k per
admission ≈ +12% of OFF's 23.2k per admission; smaller differences cannot be told apart from seed-to-seed timing.

| 4 h × 5 seeds (`--no-drain`) | OFF | previous ON (final version) | **finish release** | Verdict |
|---|---|---|---|---|
| Normal load: DSH re-read / collapses | 6,372,224 / 7.2 | 2,106,460 (−66.9%) / 0 | **2,107,833 (−66.9%) / 0** | kept |
| Normal load: one-shot requests admitted / re-read per admission | 87 / 61,230 | 109 / 60,216 | **112 / 60,196 (−1.7%)** | pass (total +26.6% from +29% admissions) |
| TTL mix: `ttl_s=0` conversation (c3) total / per admission / collapses | 245,654 / 23,175 / 1 | 523,321 (+113%) / 46,725 / 9 | **251,743 (+2.5%) / 22,886 (−1.2%) / 1** | **pass** |
| TTL mix: one-shot re-read per admission | 59,949 | 59,917 | **60,421 (+0.8%)** | pass |
| TTL mix: all DSH | 3,559,729 | 2,673,731 (−24.9%) | **2,392,179 (−32.8%)** | better |
| Overloaded TTL mix: c3 total / per admission / collapses | 960,341 / 117,115 / 10 | 1,947,951 (+102.8%) / 221,358 / 25 | **1,255,162 (+30.7%) / 136,431 (+16.5%) / 14** | **fail (improved)** |
| Overloaded: one-shot re-read per admission | 58,835 | 60,084 | **61,025 (+3.7%)** | pass |
| Overloaded: all DSH / collapses | 7,785,404 / 51 | 4,668,625 (−40.0%) / 31 | **3,276,702 (−57.9%) / 14** | better |
| Restart at 2 h (PERSIST=1): DSH re-read after the restart | (PERSIST=0) 2,028,530 | 174,139 (−91.4%) | **173,362 (−91.5%)** | kept |
| Async + rank latency skew (2 seeds): DSH re-read | — | 2,132,851 / 2,144,981 | **2,142,829 / 2,123,858**, 0 collapses, 0 content mismatches | kept |

(The rows' percentages are as printed by `tests/aggregate_finish.py`; in the overloaded run "OFF" is the TTL+persist
first cut's OFF arm of the same seeds, `results/ttl/mx_off_s*.json`.)

- Restores started / admitted per seed: normal 20.2 / 8.8 → **9.8 / 9.4**, TTL mix 21.2 / 4.8 → **2.6 / 2.6**, overloaded
  448.2 / 7.2 → **11.6 / 11.0**; blocks read under overload 8,226 → 836. Restores that are not followed by an admission
  have almost disappeared. Content checks: 0 mismatches in every run.
- Ablation (seed 1, `gate_only/` = the policy gate alone): the gate alone takes the TTL mix's c3 from 921,498 to 377,754
  (OFF 358,219) and its restores from 14 to 3. Taking NVMe-backed blocks first then improves the overloaded run (c3
  1,697,442 → 1,343,609, DSH 3,735,074 → 3,381,241); normal load and the TTL mix are identical with and without it.
- Why the overloaded `ttl_s=0` conversation is still worse than OFF: `docs/limitations.md`, "B8 after the fix".

**`DEFAULT_TTL_S` (4 h × 2 seeds):** in a mix without client fields, 0 and 86400 are **identical on every field** (in 4
hours no resume point stays unused for a day). With an 8 GiB cap, c0 / c3 without fields and c1 1d / c2 3d (12 and 19
evictions in seeds 1 and 2), 0 and 86400 are again identical on every field (the eviction order is unchanged); the
normal load with 86400 equals `fz_m_on` (default 0) on every field. With 1800 s, the resume points of c0, idle for 100
minutes, expire (seed 1: 6, 12.06 GiB; seed 2: 5, 9.49 GiB); c0's turn after the idle period re-reads (444,473 →
868,409, the same as OFF), while the other conversations are unchanged in seed 1 (−1.3% in seed 2): the expiry works and
does not push out other classes.

**Fault suite (0.5 h, drained, strict per-step invariants), finish release:** 28 runs + 9 `hit_hash` re-runs (async + CRC,
seeds 2/3/4, 3 repetitions each), table (6) of the aggregate: 0 unfinished requests, 0 invariant / `hit_hash` /
accounting violations in every run; pins remain only in `h_load` / `k_load` (5 quarantined blocks by design, a late
write must not land in a reused block; 17 before, the different block choice changed the number); content mismatches
only in the negative control `n_noload` (2,121 detected, so the check works); race probe files 2,650/0. The restart
faults (corrupt / truncated index, env / rank signature mismatch, missing file, CRC, crash, `PERSIST=0`) give the same
decisions as in section 7.7. Preemptions occurred only in asynchronous runs (4 re-admissions in total, none with a
smaller hit), see `docs/limitations.md`, "Resolved: the rig's `hit_hash_violations` anomaly".

### 7.10 u4 in the rig

`tests/compare_u4.py` (`results/u4/compare_u4.txt`): the 12 restart-fault jobs re-run with u4 equal the finish-release
runs of the same jobs on every field except timings and the index size (the per-name breakdown adds about 11.5 KB, e.g.
387,374 → 398,861 bytes). The env-mismatch job now reports the name:
`scheduler signature differs: env` → `scheduler signature differs: env [added GLM53_RIG_PATCH_FLAG]`. (In one crash job
the index writes counted before the crash instant were 9 → 10, a timing difference of the writer thread.) The rig could
not reproduce the production bug itself: its simulated workers never read `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`
(`tests/test_u4_env.py` reproduces it with two real interpreters instead).

### 7.11 u5 in the rig

`tests/compare_u5.py` (`results/u5/compare_u5.txt`):

- Regression: the same 12 jobs with u5 equal the u4 runs on every field except timings, the index size and the u5 index
  counters (the rig's drain loop asks `has_pending_push_work`, so u5 also writes the index when the drain ends).
- Idle-then-stop restart (`pr_idle_*`, 0.5 h: at 15 min arrivals stop, the engine steps until idle, idles 20 s, then two
  probe-like requests of 81k tokens are run to idle, then the restart; the TTL fields of section 7.3):

| Run | Connector | RPs ok before the stop | In the index at the stop | Adopted | Expired during the downtime | Lost | Orphan files |
|---|---|---|---|---|---|---|---|
| `pr_idle_crash_s2` (SIGKILL) | u4 | 8 | 7 | 7 | 0 | **1** | 46 |
| `pr_idle_crash_s3` | u4 | 8 | 7 | 7 | 0 | **1** | 46 |
| `pr_idle_crash_s4` | u4 | 8 | 7 | 6 | 1 | **1** | 58 |
| `pr_idle_crash_s2` | u5 | 8 | 8 | 8 | 0 | 0 | 0 |
| `pr_idle_crash_s3` | u5 | 8 | 8 | 8 | 0 | 0 | 0 |
| `pr_idle_crash_s4` | u5 | 8 | 8 | 7 | 1 | 0 | 12 |
| `pr_idle_clean_s2` (clean stop) | u5 | 8 | 8 | 8 | 0 | 0 | 0 |

"Lost" = ok before the stop − adopted − expired (a resume point whose 1 h TTL ran out during the downtime is dropped by
the TTL rule, not lost; its files are the 12 orphans of the u5 seed-4 run). With u5 the index was written twice at idle in
each run; content checks 0 mismatches. The u4 rows ran on the u4 connector (`*_u4code` jobs, work tree not shipped).

## 8. Production A/B

Deployment: GLM-5.3-Flash on two GB10 nodes with TP=2 (the head node = rank 0, the worker node = rank 1), client DSH.
Timeline: ON from 2026-10-13 (fix release); prune release later on 2026-10-13; final version from 2026-10-14; finish
release, u4 and u5 on 2026-10-14 (u5 is what runs now). The record summarized here ends on 2026-10-14.

### 8.1 Configuration

| Knob | Value | Reason |
|---|---|---|
| `GLM53_NVME_SPILL` | 1 | enable |
| `GLM53_NVME_SPILL_DIR` | `$SPILL_DIR`, same absolute path on both nodes | the in-container path must match on both ranks |
| `GLM53_NVME_SPILL_GIB` | 200 per rank | the spill directory shares the root filesystem, so we required `2×(GIB+MIN_FREE_GIB) ≤ free space` on each node |
| `GLM53_NVME_SPILL_MIN_FREE_GIB` | 150 | leaves room for logs, Docker and the model cache |
| `GLM53_EXPOSE_CACHE_RESET` | 1 until the finish release, then removed (off) | needed by the in-process quality probe; exposes `/reset_prefix_cache` without the API key (anyone reaching the port can drop every conversation's GPU cache). Closed with the finish release; the probe across a restart does not need it |
| `GLM53_NVME_SPILL_DEFAULT_TTL_S` | 86400 (from the finish release) | resume points of requests without the client field expire after a day unused (`docs/design.md` 2.10) |
| other knobs | defaults | `MIN_TOKENS=65536 MIN_GAIN=16384 SAVE_POLICY=repeat PIN_TIMEOUT_S=5 SAVE_DEADLINE_S=120 LOAD_DEADLINE_S=30 BREAKER_FAILS=6` etc. (docs/operations.md) |

Both arms keep the launcher's allocator setting (`expandable_segments:True`); only this connector is exempted from
vLLM's compatibility check.

Start-up with the connector took about 5 minutes to `/health` 200; the torch.compile rebuild caused by the changed
config hash did not noticeably lengthen it. KV capacity was unchanged (2,003,436 tokens, 582 usable blocks, same as OFF).

### 8.2 Quality probe

`code/spill_quality_probe.py`, an 81k-token conversation, greedy. R = restored from NVMe, H = GPU prefix hit, F1 / F2 =
full prefill twice. KL is against F1; acceptance is the drafter acceptance rate.

| Run | KL R/F1 | KL H/F1 | KL F2/F1 | Accept R | H | F1 | F2 | Extra restore time R−H (full prefill F1−H) | Verdict (restore / kl / acceptance) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0126 | 0.0047 | 0.0079 | 0.382 | 0.424 | 0.373 | 0.353 | 0.93 s (25.9 s) | PASS / FAIL / FAIL |
| 2 | 0.0163 | 0.0045 | 0.0076 | 0.424 | 0.426 | 0.345 | 0.436 | 0.20 s (26.9 s) | PASS / FAIL / PASS |
| 3 | 0.0033 | 0.0041 | 0.0029 | 0.460 | 0.478 | 0.414 | 0.410 | 0.69 s (26.4 s) | PASS / PASS / PASS |
| 4 | 0.0036 | **0.0241** | 0.0059 | 0.427 | 0.415 | 0.462 | 0.503 | 0.26 s (24.5 s) | PASS / PASS / PASS |
| Mean | **0.0090** | **0.0094** | 0.0061 | **0.423** | 0.436 | 0.399 | 0.426 | 0.52 s (25.9 s) | |

- Run 1 was right after the restart (empty cache, so the probe's cache reset cost nothing); 78,336 tokens were restored.
  Its kl verdict failed by a small margin (0.0126 > 1.5 × max(0.0047, 0.0079) = 0.0119) and its acceptance verdict failed
  (0.382 < 0.424 − 0.03). But R's acceptance was above both full prefills (0.373 / 0.353), and R first diverged from F1
  at the same position as F2 (38; H at 303): R behaved like "one more full prefill".
- Restore was proven in 4/4 runs (log line plus timing).
- **Conclusion:** output restored from NVMe falls in the same band as output from a GPU hit. The single-run verdict
  thresholds (one KL value against a 1.5× band) are noisier than the effect: in run 4 the GPU-hit control H itself had
  KL 0.0241, worse than the largest R value (0.0163). We therefore judge on averages of at least 4 runs, not on one run.
  (Our abort criterion A7 says "roll back on one FAIL"; we did not, because the control fails the same way.)

### 8.3 Speed A/B

Idle periods only, same yardsticks as the OFF baseline. Decode: the launcher's decode benchmark, 10 runs per prompt,
temperature 0, one stream at a time; OFF = an earlier baseline with the same configuration.

| Prompt | OFF r1 / r2 (tok/s) | ON r1 / r2 | Difference (mean) |
|---|---|---|---|
| prose | 47.38 / 46.95 | 47.45 / 47.15 | +0.3% |
| structured (acceptance fixed at 7.0, the most stable) | 102.82 / 102.73 | 102.29 / 102.56 | −0.3% |
| coding | 63.74 / 63.49 | 65.33 / 62.08 | +0.14% |
| ja | 46.77 / 45.69 | 46.71 / 44.88 | −1.0% (r2 contains an outlier with min 11.1) |

Prefill: a 32k-token prefill gate. OFF = median 3,021.5 tok/s (n=6, seeds 71-76). ON, only runs that did not overlap with
other requests (seed 76 and 81-86): 3066 / 3077 / 3035 / 3072 / 3068 / 3075 / 3036 → **median 3,068 tok/s (+1.5%, within
noise)**. Running the same seed twice hits the prefix cache (TTFT 1.5-1.8 s) and is invalid; use a new seed each time.

**Conclusion:** the per-step wait for all ranks' connector output costs less than the measurement noise in both decode
and prefill.

Real-traffic OFF baseline (before the switch): pure decode tok/s at concurrency 1/2/3/4: 40.3 / 59.4 / 71.2 / 95.9. The
first ON windows (about 4 h of active time in total over three releases) gave 42-44 tok/s at concurrency 1 and too few
multi-request windows to compare; a comparison needs at least a day of real traffic.

### 8.4 Real-traffic observations

- After the four probe runs: 4 saves, 4 restores, 0 failures, breaker closed, NVMe 2.14 GiB/rank (81k tokens × 4 =
  0.53 GiB each; the estimate 0.49 + 0.09 = 0.58 GB agrees).
- **First real save:** a 244k-token conversation, first save 1.57 GB/rank (the formula 244k × 6.0 KB + 92 MB = 1.56 GB
  agrees). Each following turn wrote 0.131 GB (new = 7 blocks: one state set + 1 MLA block). No alert, no failure.
- Disk on both ranks at one point: head node 3.9 GB + worker node 3.9 GB = 7.8 GB in total.
- An early check of the prune release: no alert, breaker closed, `threads_alive=2/2` on both ranks, quarantined 0. No
  conversation of ≥ 65,536 tokens had arrived yet, so `saves_ok=0` and disk 0 GiB (by design with `MIN_TOKENS=65536` /
  `repeat`, not a fault).
- Host memory: we judge pressure by swap activity, not by swap usage, and saw no degradation attributable to the
  connector.
- Installing the final version: its first start found no index (the prune release writes none) and wiped as designed:
  `persist: wipe every rank (no usable index (no index file))`, removing 84 old files (2.05 GiB/rank). The check
  `block hashes reproducible` passed; the scheduler line showed `persist=1 hash_seed_from=connector ttl_save_policy=all`.
- A cosmetic defect in the final version: the rank's `dir=` start-up line formats `wiped=%d` with the string `deferred`,
  so Python logging drops that line. Behavior is unaffected; fixed in the published version (docs/limitations.md). We did
  not redeploy for it, because changing the connector bytes changes the persist signature and the next restart would
  wipe instead of demonstrating adoption. (The fix went into production with the finish release.)
- **TTL arrival confirmed (2026-10-14):** the first save after the client started sending the field landed in the 1d
  class (`ttl_rps=0/0/1/0/0 ttl_gib=0.00/0.00/1.54/...`; the client's default is 86400 s = 1d), 1.54 GiB/rank (du 1.6G on
  both nodes), and `index.json` (29 KB) was written. `write_failed=0`, breaker closed.
- **First restores on real traffic (2026-10-14, final version):** at one check `restores_ok=2 failed=0 fallback=0`,
  `saves_ok=40 failed=0`, 11 resume points, disk 8.04 GiB/rank; at a later check `restores_ok=7 failed=0`,
  `saves_ok=84 failed=0`, 12 resume points, disk 8.65 GiB/rank, `ttl_rps=2/0/10/0/0`; no alert, breaker closed.

### 8.5 Restart frequency

In our deployment a restart typically happens every 2-3 days, and many times a day while the serving stack itself is being
changed. The loss per restart without reuse ≈ the first re-read of 2-3 long DSH conversations (1.5-2.0 M tokens in total ≈ 12-17
min of generation stall). Further restarts during the A/B installed the prune release, the final version, the finish
release, u4 and u5, plus one test restart per version for section 8.6.

### 8.6 Reuse across production restarts (finish release, u4, u5; 2026-10-14)

**Production, not the rig.** Two nodes, TP=2, the deployment of section 8.1. Each new version's first start wipes once
by design (its code differs); the following restart shows whether adoption works. Between the two restarts the quality
probe's prepare phase saved resume points of an 81k-token conversation (`--phase prepare --runs 4`, 4 salts), with the
server otherwise idle. Restarts were done with our launcher's stop / start (`docker rm -f`, i.e. SIGKILL, then start).

| Start | Connector | `persist:` decision (log) | Files deleted |
|---|---|---|---|
| first start of the finish release | eb2e1084 | `wipe every rank (scheduler signature differs: code,env,site,version)` (expected: code and environment changed) | rank 0: 356 |
| second start of the finish release | eb2e1084 | **`wipe every rank (rank signature differs (rank 0: env; rank 1: env))`** although nothing changed; the scheduler signature matched. Cause: `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` (`docs/design.md` 2.8.2) | rank 0: 165 |
| first start of u4 | bd0f589e | `wipe every rank (scheduler signature differs: code, version)` (expected) | — |
| second start of u4 | bd0f589e | **`adopted 3 resume points (69 files, 1.60 GiB/rank) from the previous run; dropped 0 expired + 0 incomplete; orphan files deleted per rank {0: 23, 1: 23}; decision 4.95 s after the first step`** -- the first adoption across a production restart. The 4th resume point (the last one saved before the idle stop) was not in the index; its files were the orphans (`docs/design.md` 2.8.3) | 23 per rank |
| first start of u5 | 073a2e8c | `wipe` (code, version; expected) | — |
| second start of u5 | 073a2e8c | before the stop: `persist: index written (idle: 4 resume points, 92 files, ...)`; after: **`adopted 4 resume points (92 files, 2.14 GiB/rank) from the previous run; dropped 0 expired + 0 incomplete; orphan files deleted per rank {}; decision 2.10 s after the first step`** | none |

Our operator start-up check expecting adoption passed after the second start of u4 and of u5 (for u5 also the check
that every saved resume point was in the index). After the finish release was installed the scheduler line showed
`policy_gate=1 default_ttl_s=86400`, and `/reset_prefix_cache` answered 404 (the unauthenticated endpoint is closed).

**Quality probe across the restart** (`code/spill_quality_probe.py --phase restart`, 4 runs, the 81k-token conversation;
R = restored from the adopted files, H = GPU hit, F1 / F2 = full prefill; the verdict is on the 4-run means):

| Version | Restore proven | R−H per run (full prefill F1−H) | Mean KL R/F1 | H/F1 | F2/F1 | Mean acceptance R / H | Verdict |
|---|---|---|---|---|---|---|---|
| u4 | 3 of 4 (the 4th run: `cached_tokens=0`, its resume point lost to the index gap) | 0.81 / 0.45 / −0.13 s for the 3 restored runs (25.7 / 24.3 / 23.5 s) | 0.0014 | 0.0025 | 0.0039 | 0.419 / 0.447 | restore FAIL (3/4), kl PASS, acceptance PASS |
| **u5** | **4 of 4** | −0.14 / 1.02 / 0.55 / −1.46 s (25.2 / 26.6 / 26.5 / 23.8 s) | **0.0038** | 0.0069 | 0.0042 | **0.448 / 0.414** | **restore PASS (4/4), kl PASS, acceptance PASS** |

A restored 81k-token prefix came back in about the time of a GPU hit (R−H between −1.46 and +1.02 s) instead of the 24-27
s of a full prefill, and its output fell in the band of the GPU hit and of a second full prefill.

### 8.7 Not yet measured in production (as of the record's last entry, 2026-10-14)

- Adoption of real conversations' resume points across a restart, checked on its own (section 8.6 used the probe's
  resume points).
- Pruning on real traffic: the record contains no explicit prune event. Disk stayed at 8.04 → 8.65 GiB/rank while saves
  went from 40 to 84 (section 8.4), which is consistent with pruning but not a direct observation.
- The B8 counters (`restores_started` / `restores_admitted`) over a long period. With `GLM53_MIXED_PREFILL_CHUNK=0` the
  policy gate cannot fire (`docs/design.md` 2.9).
- The real-traffic ON values of the yardsticks in section 8.3 (stall share, ≥ 100k ingests, decode tok/s by
  concurrency; the few hours recorded so far give partial values with too few windows to compare, section 8.3), and restore counts, durations and failures over a long period.
- Real-traffic effect of `TTL_SAVE_POLICY=all`, of ttl 0 and of `DEFAULT_TTL_S`.

## 9. NVMe capacity and write volume

Per-rank figures; with TP=2 the totals for the deployment are 2× (e.g. head 3.9 GB + worker 3.9 GB = 7.8 GB observed).

### 9.1 File sizes per block (4,608 tokens, one rank)

| Content | Size |
|---|---|
| MLA + indexer (attention KV) | 27,624,960 B = 27.6 MB (≈ 6.0 KB per token per rank) |
| KDA (mamba) states, 4 groups | 21.2 + 21.2 + 18.9 + 18.9 = 80.2 MB (one set per RP) |
| DFlash drafter window | 11.8 MB per block |
| Header | 4 KiB per file |

### 9.2 First save of a conversation

**First save ≈ length × 6.0 KB + 92 MB per rank** (estimate; 92 MB = the 4 state groups + one DFlash block). Checked against
production: 244k tokens → 1.57 GB measured vs 1.56 GB from the formula. RPs actually carry two DFlash blocks (section 4.1),
and in production one per-turn state set measured 104 MB (4 mamba + 2 DFlash), so the formula slightly underestimates the
fixed part.

| Conversation length | First save, 1 rank (estimate) | Both ranks |
|---|---|---|
| 100k tokens | 0.7 GB | 1.4 GB |
| 380k (the client's auto-compaction limit in our setup) | 2.4 GB | 4.8 GB |
| 600k | 3.7 GB | 7.4 GB |
| 1M (model limit) | 6.1 GB | 12.2 GB |

### 9.3 Later turns

The next turn of the same conversation shares the earlier blocks (deduplication), so only the newly filled MLA blocks and
one state set are written: **≈ 0.1-0.16 GB/rank per turn** (rig); 0.131 GB measured in production.

### 9.4 Write volume per day (estimate)

Write volume ≈ (continuation turns per hour) × ≈ 0.13 GB + first saves of new conversations, per rank. For our load this
gave **~10 GB/h/rank ≈ 240 GB/day/rank** (estimate, made before the A/B, from our admission log). That is
≈ 88 TB/year/rank. Against a 2 TB drive rated around 1,200 TBW this is over 10 years; we have not checked the TBW rating of
our own drives. One-shot requests are not written (`SAVE_POLICY=repeat`). To reduce writes: `MAX_WRITE_GIB_PER_H`, or a
higher `MIN_TOKENS`. Pruning (section 6) reduces disk occupancy, not the write volume. In the rig, writes were 3.60
GiB/h/rank (section 6.4, K=0) for three conversations.

### 9.5 Capacity needed

- With 2-3 long DSH conversations at a time, the space actually needed is **10-20 GB per rank** (estimate). The 200 GB cap
  holds more than 50 conversations of 600k tokens.
- Above the cap, RPs are evicted (LRU, with the TTL ordering of docs/design.md); an evicted conversation is simply
  recomputed.
- Before pruning, disk grew ~10 GB/h/rank and would reach the 200 GiB cap in under a day, after which other conversations'
  RPs are evicted; with K=2 we expect disk to plateau at roughly conversations × (length × 6 KB + 2 × 92 MB).
- A 3-day TTL for one conversation at the 380k compaction limit ≈ 2.4 GB/rank (≈ 5 GB for both ranks).
- General formula: capacity per rank ≈ Σ(length of conversations to keep) × KV bytes per token per rank + number of
  conversations × size of one state set.

### 9.6 Speed versus recompute

600k tokens' worth: written in 1.40 s, restored in 1.24 s (fix release, section 5.4). Recomputing at about 2,000 tok/s
takes about 5 minutes, roughly 240× longer.
