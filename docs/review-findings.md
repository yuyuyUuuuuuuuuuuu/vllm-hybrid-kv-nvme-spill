# Independent review: findings and fixes

Before the first production A/B, the initial version of the connector was given to an independent review whose
task was to try to refute it. The review worked only from the production vLLM tree (the image's vLLM after the
launcher's start-up patches) and the launcher kit, on a separate test machine (one GB10); production was not touched.

**Verdict of the review: "fix first" (do not start the production A/B in this state).** The core design (restore
exactly the block set the engine's own coordinator returns, with the same keys and the same bytes, into the GPU
prefix cache, and leave admission to the ordinary path) was judged sound for correctness, and the review reproduced
the rig numbers exactly. The problems were local: the ON arm could not start (B1), a read-only file system on one
rank exhausted the KV pool (B2), a load that never returned made a conversation wait forever (B3), the start.sh patch
did not apply to the current kit (B4), and the quality probe could PASS when no restore had happened (B5).

All findings B1-B7 were fixed in the fix release, B8 was documented, and every test was re-run. Results of the
re-run are in `results/fix/` (`results/fix/faults_summary.txt`, `results/fix/aggregate_m_on.txt`,
`results/fix/aggregate_x_on.txt`, `results/fix/unit_final.txt`, `results/fix/gpu_final.txt`).

## What the review reproduced

| Item | Claimed (initial version) | Reproduced by the review |
|---|---|---|
| Replay rig 4 h x 5 seeds, same request sequence per seed, `--no-drain` | DSH re-read 6,372,224 -> 2,106,460 (-66.9%), collapses 7.2 -> 0 | **Every metric identical** to our JSON files, ON and OFF. Restore failures 0, `restore short` 0, content mismatches 0 |
| Unit tests | CPU 19/19, GPU 18/18 | CPU 19/19, GPU 18/18 |
| GPU bench, 600k tokens' worth (137 blocks, 3,723,575,296 B) | write 1.51 s, read 1.38-1.45 s | write 1.77 s, read 1.51 s, bytes identical |
| Bandwidth-bound kernel alone / during I/O | 228-233 GB/s / within -1% | 115.6 GB/s alone; 120.7/109.7 GB/s during I/O (+4%/-5%, noise range). Five CPU replay runs and another resident GPU process were running at the same time, so the absolute bandwidth was not reproduced and "within -1%" was **not reproduced under equal conditions** (not refuted either, because the conditions differed) |
| B6 frequency (a save planned in `build_meta` that includes a block keyed in the same step), instrumented 4 h x 5 seeds | - | **0 of 177 plans**. Instrumentation did not change behaviour (results identical to the plain ON run). Does not occur in this load (no concurrent recomputation of identical content) |
| Async completion 4 h x 5 seeds (workers not drained; I/O delay rank 0 0-80 ms, rank 1 0-250 ms per block; results counted in the step they arrive) | - | DSH re-read 6,372,224 -> 2,108,028 (**-66.9%**), collapses 7.2 -> 0, 92 restores with 0 failures, 0 short, 0 content mismatches. Fewer wasted re-restores than the synchronous run (seed 4: 17 vs 52). The rig's simulated clock does not advance during I/O, so real waiting time is not represented |

## Findings, fixes and regression tests

| # | Finding | Fix (fix release) | Regression test (result) |
|---|---|---|---|
| B1 blocking | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the kit's start.sh + this connector is rejected by `_verify_kv_transfer_compat` -> the ON arm does not start. The rig assigned the config after construction and never ran the check | `code/patch_nvme_spill.py` also patches `config/vllm.py` and exempts **only this connector's name** (it does no RDMA/IPC registration: only torch views and a pinned host staging buffer). Both arms keep the kit's allocator (no confound). The rig now **constructs** `VllmConfig` with `kv_transfer_config` and passes expandable_segments to both arms | `tests/t_alloc_conf.py`: unpatched tree -> ValueError, patched -> accepted, `NixlConnector` still rejected. `tests/t_expandable_gpu.py` (GB10): KV inside expandable segments, addresses unchanged and I/O bytes identical after allocator churn (`empty_cache`), 9/9. Every ON rig run is constructed under expandable_segments |
| B2 severe | The writer thread dies on EROFS/EACCES -> no answer -> pins accumulate without bound (507 of 582 blocks) | The I/O threads catch every exception per job and always answer; `delete` only logs an `OSError`. A save deadline (120 s) releases pins; consecutive failures (6), a deadline miss or a worker exception open a latched breaker (identical on both ranks); a cap on save pins held only by the connector (half the pool) | A review test (not shipped; its before/after output is `results/fix/t_writer_death_old.txt` / `_new.txt`): before = thread dead, job 2 unanswered / after = alive, job 2 answered. Rig `a_rofs` (the review's own injection, unchanged) and `a_rofs_os` (EROFS at the OS layer): 0 pins, 582 free, 0 unfinished requests, breaker open |
| B3 severe | Load jobs have no deadline (a conversation waits forever) | Deadline (30 s): the request is detached to the normal prefill, fresh blocks are quarantined until every rank answers (never freed early), breaker | `h_load` (rank 1's load never returns): all requests finish, only 17 quarantined blocks remain (by design) / `h_loadlate` (returns after 8 s): quarantine released, 582 free / `k_load` (rank 1's answer is lost): requests finish, 17 quarantined |
| B4 procedure | The start.sh patch does not apply to the current kit; 6 knobs do not reach the containers | `code/apply_start_sh.py`: anchors no longer depend on the last name of the `serve_env` loop; forwarded names are taken from the connector's `KNOBS` automatically (23 names at that time); idempotent, fail-closed | `tests/test_start_sh.sh` 15/15: applies to the kit's start.sh (matching its MANIFEST), a second run changes nothing, broken input ABORTs without writing anything; the patched start.sh's own validate/launch run with fake `docker`/`ssh` and all 23 names reach `docker run` on both ranks |
| B5 validity | The quality probe PASSes even if nothing was restored (production returns `cached=None`) | Positive evidence of a restore is required (a log line `restore ready ... rp=N`, or `cached_tokens`, plus R-H <= 0.25 x (F1-H)); otherwise FAIL | `tests/test_quality_probe.py` 10/10 |
| B6 latent | A save planned in `build_meta` may copy a block before the same step's forward writes it | Saves and deletes planned inside `build_meta` are shipped in the next step | Unit: the old module FAILs the send-order check, the new one PASSes. Rig race probe (forces copies of blocks keyed in the same step + checks the saved file contents): before 2,650/2,650 files with stale content and 1,238 restored blocks mismatched / after 0/0 |
| B7 low | If an RP is dropped while its save is running, its files are never accounted for | On a successful save, files no RP references are deleted | Unit `b7_no_disk_leak...` |
| B8 low | Restores before admission are wasted (seed 4: 52 restores / 9 admissions) | Not fixed (documented in `docs/limitations.md`). Not a correctness problem | The 4 h matrix records `restore_started`/`admitted` |
| Observability | No statistics for abort criteria | A periodic stats line (scheduler) and a per-rank `io` line | Seen in rig logs (`h_load`: `rank 1 io: r=load:6/107.7s`, `breaker=OPEN ... quarantined=17`) |

### B1 in detail

* The launcher kit's `<kit>/launcher/start.sh` (line 2974) passes
  `-e PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}` by default (the comment just above
  already says that the hidden-state KV connector rejects expandable_segments).
* vLLM `config/vllm.py:1001-1041` `_verify_kv_transfer_compat()` (always called from `__post_init__`, line 1799)
  raises `ValueError` when a `kv_connector` is configured **and** `expandable_segments:True` **and**
  `enable_cumem_allocator=False` (the default; start.sh does not set it). It does this for **any** KV connector.
  Reproduced with the review's `t_alloc_conf.py` on the rig's production `VllmConfig`:
  `'' -> accepted` / `'expandable_segments:True' -> ValueError: KV connector Glm53NvmeSpillConnector is incompatible
  with ...`.
* Why our rig missed it: it assigned `kv_transfer_config` **after** constructing `VllmConfig`, so the
  connector-related checks of `__post_init__` (this check, the PIECEWISE forcing decision and the automatic HMA
  disabling decision) never ran.
* Options considered by the review: (a) exempt only this connector, keep expandable_segments on both arms; (b) unset
  the variable on the ON arm only, which would confound the A/B unless the OFF arm is changed too, and on a GB10 with
  4 GiB free and a 2M-token KV pool a different allocator may change start-up OOM behaviour and bandwidth (the memory
  preparation step assumes the current allocator), which would need its own start-up test. We chose (a).
* Why the exemption is safe: the rule exists because connectors that register KV memory with a NIC or another process
  (NIXL `ibv_reg_mr`, Mooncake, CUDA IPC) hold physical-page registrations that a virtual-memory remap makes stale.
  This connector registers nothing. It reads and writes the KV tensors only through torch views inside the worker
  process (every access is a `cudaMemcpyAsync` on the tensor's virtual address, resolved through the current mapping;
  a live caching-allocator block is never remapped), and its staging buffers are pinned **host** memory from torch's
  host allocator, which expandable_segments does not govern. `tests/t_expandable_gpu.py` checks on a GB10 that KV
  addresses stay unchanged and I/O stays byte-identical after `empty_cache` churn. Every other connector name still
  raises (`tests/t_alloc_conf.py` checks `NixlConnector`).

### B2 in detail (as found)

`BlockFileStore.delete()` caught only `FileNotFoundError`. An EACCES/EROFS/EIO from the writer's delete path, or from
the cleanup `store.delete(fname)` inside the `except` of a failed save, ended the thread. From then on that rank never
answered a save job, and the scheduler kept the save pins (`block_pool.touch`) until every rank answered; there was no
global cap either. `has_pending_push_work()` also stayed true, so EngineCore kept running empty steps while idle.
Reproduced in the rig (async, 0.5 h, seed 2, `NVMESPILL_FAULT=rofs`: rank 1's writes from the 41st on and unlinks of
existing files fail with EROFS): `rp_saved 0`, 17 unfinished jobs, **120,000 drain steps without finishing, 75/582
free, 507 blocks pinned by the connector, 27 unfinished requests**. With the connector off, this fault does not affect
inference at all.

### B3, B5, B6, B7, B8 in detail (as found)

* B3: `get_num_new_matched_tokens` returned `None` for as long as the state was `loading`, with no deadline. A hung
  NVMe read (D state) or a missing answer from one rank left the request in the waiting queue forever; later requests
  for the same RP queued another load behind the stuck thread after 120 s. The fresh blocks must **not** be freed on a
  deadline, because the worker may still write them later.
* B5: the probe accepted `cached is None`. Production does not pass `--enable-prompt-tokens-details`, so vLLM's OpenAI
  serving code returns no `prompt_tokens_details` and `cached` was always None. If R was not restored and fully
  prefilled, its KL is almost the same as F1 and the probe PASSed. R's wall time was printed but not used.
* B6: saves planned by `_scan_prefill_done`/`sweep` inside `build_meta()` (the end of `schedule()`, after that step's
  `allocate_slots` attached keys) reached the worker in the same step's `SchedulerOutput` and were picked up in
  `pre_forward`, before that step's forward was queued. The writer's `torch.cuda.synchronize()` waits only for work
  queued so far, so a block written by that step's forward could be copied before or while it was written, and the
  check in `_finish_save` (did the key move?) cannot detect it. vLLM itself states for mamba that blocks created by
  other requests in the same step cannot be relied on (`cached_blocks_this_step`, `single_type_kv_cache_manager.py:1610-1617`).
* B7: an RP in state `saving` dropped by `_forget_bad_files`, whose save then succeeded, left files with state `ok`,
  refs 0, owned by no RP, counted in `disk_bytes` and never evicted by LRU (until restart).
* B8: in the 4 h rig `restore_timeouts == restore_started` for every seed. Restores happen before admission
  (decode floor etc.); after 5 s the MLA pins are released and blocks are read again. Seed 4: 52 restores, 9
  admissions, 608 blocks read. Each restore's `get_new_blocks` evicts other conversations' GPU cache.

### Also found by the review (documentation)

* The connector's module docstring still said "written at request finish" (the implementation writes right after the
  prompt is computed). Fixed.
* The capacity defaults of our deployment were justified by the free space of one disk, without a step to check
  whether the spill directory shares the root file system. Added to the operating procedure.

## Review verdicts per focus item

| # | Item | Verdict | Basis |
|---|---|---|---|
| 1a | Restored blocks = correct content for that range (hash / position / group) | **CONFIRMED** (bookkeeping) / PLAUSIBLE (real GPU) | The RP is exactly the return value of `coordinator.find_longest_cache_hit(block_hashes, num_tokens-1)`; file name = sha256 of the primary key (group id included); the header carries group, rank, row layout and CRC. The hit after a restore comes from the engine's ordinary lookup. 0 content mismatches in the rig (4 h x 5, 0.5 h x 5). **Negative control** (rank 1 pretends to read and writes nothing) detected 1,024 `verify_fail`, so the check works. Not verified with real forward-pass bytes |
| 1b | Byte positions (which page of which tensor) | **CONFIRMED** | `attn_utils._allocate_kv_cache` allocates each tensor with its own `torch.zeros` (storage offset 0); `_reshape_kv_cache` starts every layer at a page boundary (mamba `raw[:nb*page].view`, DFlash/kpool tail padded to the tensor's page, `_glm5_next_tensor_layout` enforces mla_page == mamba page). The connector's `group_rows` rejects packed tensors |
| 1c | MLA+indexer / 4 mamba groups / DFlash window | **CONFIRMED** | Every RP has one block of each of g2-g5 and g6 (mostly 2 blocks). The indexer's pooled keys are part of group 0's tensor rows |
| 1d | kpool tail (group 1) missing | **CONFIRMED (no problem)** | Group 1 is a per-request ring outside the prefix cache. Hit boundaries N are multiples of 4,608, so the in-progress pool of kpool=4 is empty. Same as an ordinary hit in GPU memory |
| 1e | Drafter state not restored (SGLang #31600 type) | PLAUSIBLE | The DFlash window blocks are part of the RP and come back with the same keys. If the window is missing, the coordinator's replay clamp applies as for a GPU hit. Real drafter acceptance not measured (only the A/B can) |
| 1f | Position and credit of the mamba state (N vs N-1) | **CONFIRMED (same as a GPU hit)** | Coordinator: block i = N/4608-1 gives hit_length = N. `MambaHybridModelState.add_request` seeds from (N-1)//4608 = the same block. The connector only puts the same bytes into the same block; the credit is the engine's existing code. An off-by-one would break GPU hits as well |
| 1g | Conflict with partial-hit CoW (`move_block_hashes`) | **CONFIRMED (does not occur in production)** | `KpoolTailManager` does not support fine-grained hits, so `enable_partial_hash_hits=False` (rig log "Disabling fine-grained prefix-cache hits ... KpoolTailManager") |
| 1h | Copy before write within one step | **REFUTED (can occur in principle)** | B6 |
| 2a | Both ranks decide the same | **CONFIRMED** | Decisions in one place (scheduler), the same `SpillMeta` to every rank; keys are attached only when ok count == world |
| 2b | Only one rank succeeds / is late / misses a file | **CONFIRMED (safe side)** | In the async rig (large I/O delay on rank 1) `job_acc` is summed across steps, 0 leaks. Missing file / CRC -> failure -> freed without keys -> normal prefill |
| 2c | Deadlock / paths that stop production | **REFUTED** | I/O runs on separate threads and does not block the forward or NCCL (no collective deadlock path), but B2/B3 lead to pool exhaustion and conversations that wait forever |
| 2d | Aggregation on the real two-node setup | UNVERIFIED | vLLM's `KVOutputAggregator` combines worker meta with `aggregate` and requires output from every rank when rank 0's is not None (`assert model_runner_output is not None`). If a production overlay omitted non-rank-0 output, the first step would fail. Not tested on real hardware |
| 3 | Safe side on failures | **REFUTED (partly)** | CRC / missing file / abort / preemption / ENOSPC / release races are on the safe side (table below). unlink failure, read-only FS and hung I/O are B2/B3 |
| 4a | Side effects of `--kv-transfer-config` | Partly CONFIRMED | Start-up rejection (B1). `defer_block_free=True` (kv_both = consumer + async) delays the return of aborted requests' blocks and **CoW-held blocks** by one step (`scheduler.py:3109-3127`). The config hash changes, so torch.compile rebuilds. No PIECEWISE forcing (`requires_piecewise_for_cudagraph` defaults to False); HMA stays on (`SupportsHMA`). **Every step now waits for every rank's output** (with an aggregator `output_rank=None`, `multiproc_executor.py:390-407`): the `ModelRunnerOutput` previously awaited only from rank 0 is now also received from the worker node every step. Not measured |
| 4b | Does the I/O thread slow decode through the GIL or bandwidth? | PLAUSIBLE (small) | On the test machine, a decode mock (600 small kernels + 6 GiB bandwidth read + sync, about 64 ms/step) measured while saving and loading a 300k-token RP: mean 63.9/63.2 -> 64.6/64.9 ms (+1-3%, similar to the idle-to-idle difference). Host-only mock: mean +1%, **p99 4.56 -> 6.94 ms**. I/O only during restores and saves |
| 4c | Write volume, endurance, free-space handling | PLAUSIBLE | Rig 3.6 GiB/h (3 conversations). The production estimate of ~10 GB/h/rank is arithmetically reasonable. Free-space floor, cap and LRU exist, but the throttling knobs did not reach the containers (B4) |
| 4d | Patch anchors | Partly REFUTED | The one `scheduler.py` anchor matches the production tree (the rig's working copy rebuilt from the production tree, applied, idempotent; no diff to ours). start.sh does not match the current kit (B4) |
| 5 | Validity of our own tests | Mostly valid, with holes | OFF/ON use the same seed = same request sequence (OFF identical to our earlier baseline). Bandwidth figures count read + write and are within physical limits. Holes: the rig's workers were **synchronous** (drained every step, I/O time 0); `VllmConfig` checks were skipped (B1); no fault injection for unlink failure / read-only FS / timeouts; the content check uses "content at key time" and cannot model when the real forward writes (B6) |

Individual failure cases run by the review (replay rig, strict invariants + drain):

| Fault | Result |
|---|---|
| Async completion (I/O delay rank 0 0-80 ms, rank 1 0-250 ms, drain) | 0 collapses, 11 restores all successful, 5,589 blocks checked with 0 mismatches, after drain 582 free, ref 0, 0 pins |
| Abort requests while loading (10) | All released through the orphan path, 582 free, 0 pins |
| Async + CRC corruption | 2 failures -> normal prefill, `bad_files` 2, 582 free |
| Rank 1 file system becomes read-only | **507 pinned, 75 free, drain never finishes** (B2) |
| Negative control (pretend to read) | `verify_fail` 1,024 (the check works) |

## Re-run after the fixes

From `results/fix/`. The 4 h x 5 seed synchronous runs match the initial version on every metric.

| Item | Initial version / review reproduction | Fix release |
|---|---|---|
| 4 h x 5 sync, DSH re-read OFF -> ON | 6,372,224 -> 2,106,460 (-66.9%) | 6,372,224 -> 2,106,460 (-66.9%; every seed, every metric identical) |
| Collapses OFF -> ON | 7.2 -> 0 | 7.2 -> 0 |
| All-request re-read / estimated stall | -54.0% / 3,449 -> 1,599 s | -54.0% / 3,449 -> 1,599 s |
| Restores (sync) started / admitted | 20.2 / 8.8 | 20.2 / 8.8; 0 failures, 0 short, 0 content mismatches; 6,326 saved files checked, 0 mismatches |
| 4 h x 5 async + rank delay difference | 2,108,028 (-66.9%) / 0 collapses / 18.4 restores | 2,108,750 (-66.9%) / 0 collapses / 13.4 restores, 0 failures |
| Read-only FS (rank 1) | 75 free, 507 pinned, 27 unfinished, drain never finishes | 582 free, 0 pinned, 0 unfinished, breaker open |
| Load never returns / answer lost | (not tested; waits forever) | All requests finish, 17 quarantined (by design) |
| Unit CPU / GPU | 19/19 / 18/18 | `test_store` 19/19, `test_sched_faults` 49/49, start.sh 15/15, probe 10/10 / GPU 18/18 + expandable 9/9 |
| GPU bench 600k tokens' worth (3.72 GB) | write 1.51 s, read 1.38-1.45 s, bandwidth within -1% | write 1.40 s, read 1.24 s, bytes identical, bandwidth 220.1 -> 217.1/216.5 GB/s (-1.4/-1.6%), under expandable_segments |

Found and fixed while fixing: the first pin cap counted every pin (including blocks also referenced by running
requests) and also blocked restores. In the async fault-free run this stopped 30,547 restore attempts, gave 25
give-ups and 5 collapses. The cap now counts only save pins held by the connector alone and never blocks restores;
its normal maximum is 12. At the end of an async rig run an I/O thread was still inside a torch copy and the process
printed `terminate called without an active exception` (the result JSON was already written); `SpillWorker.shutdown`
now joins the threads (up to 10 s) and the rig calls it at exit. That run was repeated (rc 0).

Not fixed in the fix release: B8 (see `docs/limitations.md`); aggregation on the real two-node setup (2d) and
real forward-pass bytes, which only the production A/B start-up checks and quality probe can show.

## Tests named here that are shipped

| Test | Finding |
|---|---|
| `tests/t_alloc_conf.py` | B1 |
| `tests/t_expandable_gpu.py` | B1 (GPU) |
| `tests/test_sched_faults.py` | B2, B3, B6, B7 and worker exception tolerance |
| `tests/test_start_sh.sh` | B4 |
| `tests/test_quality_probe.py` | B5 |

The review's writer-death test is a review test (not shipped); its outputs before and after the fix are in
`results/fix/t_writer_death_old.txt` and `results/fix/t_writer_death_new.txt`.
