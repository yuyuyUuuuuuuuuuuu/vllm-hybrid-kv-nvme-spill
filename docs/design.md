# Design

This document describes `Glm53NvmeSpillConnector` (`code/glm53_nvme_spill_connector.py`), a vLLM v1 KV connector
that saves the "resume point" of a long conversation on a hybrid model to local NVMe on every tensor-parallel rank
and brings it back into the GPU prefix cache before the conversation's next turn is admitted.

Measurements are in `docs/measurements.md`, operation in `docs/operations.md`, the client side in
`docs/client-contract.md`, the independent review in `docs/review-findings.md` and open risks in
`docs/limitations.md`.

## Problem

The served model is GLM-5.3-Flash (EXL3) on two GB10 machines with TP=2. Its KV cache is a hybrid of seven KV cache
groups that share one block pool:

| Group | Content | File bytes per block (one rank) |
|---|---|---|
| g0 | MLA latent cache + indexer keys | 27,624,960 |
| g1 | kpool tail (a per-request ring, no prefix caching) | not saved (1,672,704 in the pool) |
| g2, g3 | mamba (KDA) state, two groups | 21,233,664 each |
| g4, g5 | mamba (KDA) state, two groups | 18,874,368 each |
| g6 | DFlash drafter window | 11,796,480 |

(Per-group byte sizes: section 2.4 below; the `group_bytes` vector of the start-up log, as quoted by the
independent review, is `[27624960, 1672704, 21233664, 21233664, 18874368, 18874368, 11796480]`.)

The pool has 582 usable blocks; a prefix-cache block covers 4,608 tokens, so cache hits are multiples of 4,608
tokens. A deep prefix-cache hit on this model needs **all** of:

* the MLA+indexer blocks from the start up to the boundary,
* the boundary state of each of the four mamba (KDA) groups at that boundary,
* the DFlash drafter window at that boundary.

The boundary state of the last prompt chunk is released when the prefill finishes (it stays cached but becomes
evictable). Under a saturated pool the boundary states go first: during the minutes a request spends decoding, other
requests' prefills evict them. The next turn of the same conversation then finds MLA blocks but no matching state,
and its hit falls back to a shallow boundary (in our replay, for example 0 or 13,824 tokens = the 3-block head shared
by every conversation of the client). The turn re-reads 0.5-0.8M tokens, which takes 5-15 minutes during which every
other stream stalls. Our earlier re-read investigation also found that protecting boundary states inside the pool
is zero-sum: the root cause is pool capacity, not the eviction order. We call a continuing turn whose hit is at most
23,040 tokens a "collapse".

The connector therefore keeps a copy of exactly the blocks that make up the deepest resume point on local NVMe and
puts them back when the next turn arrives.

## 0. Summary

* vLLM's own `OffloadingConnector` with a file-system tier cannot do this on this model (section 1). We wrote our own
  connector, `Glm53NvmeSpillConnector` (`code/glm53_nvme_spill_connector.py`).
* It stores a conversation's deepest resume point (MLA+indexer block list + boundary state of the four mamba groups +
  DFlash window: exactly the block set the engine's own coordinator returns) as one file per block (content
  addressed, shared between turns) on each rank's local NVMe, and before the next turn is admitted puts it back into
  the GPU prefix cache **with the same bytes and the same keys**. The engine's ordinary lookup then finds the deep
  hit and counts it as an ordinary prefix-cache hit.
* In a replay rig that runs the real `Scheduler`/`KVCacheManager` of the production vLLM tree (the image's vLLM after
  the launcher's start-up patches) on CPU, 4 hours x 5 seeds with the same request sequence per seed: **DSH re-read
  tokens -66.9% (6.37M -> 2.11M; the rest is the first ingestion of the 3 conversations, which cannot be avoided),
  re-read tokens of all requests -54.0%, collapses (continuing-turn hit <= 23,040) 7.2 -> 0 (0 for every seed),
  estimated generation stall 3,449 s -> 1,599 s (-54%), requests finished within 4 hours 242 -> 294 summed over the 5 seeds (+21.5%)**. Every restored block
  was byte-checked on both ranks with 0 mismatches; 0 leaked blocks (after drain 582/582 free, ref 0, 0 connector
  pins); every injected fault (a file missing on one rank, CRC corruption, write failure) fell back to the normal
  prefill without leaks. (`results/aggregate.txt`, `results/m_*.json`; `docs/measurements.md`.)
* On the NVMe of a separate test machine (one GB10): 600k tokens' worth (3.72 GB for one rank) written in 1.51 s and
  read back in 1.45 s, bytes identical. With a bandwidth-bound kernel running concurrently, the kernel's bandwidth went
  from 232.8 to 230.6/231.2 GB/s (within -1%). (`results/bench_gpu.json`.)
* Off by default. Only `GLM53_NVME_SPILL=1` installs the patch and adds `--kv-transfer-config`.
* An independent review of the initial version found B1 (the ON arm could not start), B2 (pool exhaustion), B3
  (endless wait), B4 (start.sh patch), B5 (quality probe), B6 and B7. All were fixed in the fix release and every
  test was re-run (`docs/review-findings.md`). The 4 h x 5 seed numbers are identical to those before the fix (-66.9%,
  collapses 7.2 -> 0); every fault injection ends with all requests finished and 0 pins (only when a load never
  returns does a quarantined block set remain, by design).
* Prune release (made during the production A/B): when a newer turn of a conversation is saved, older resume points
  of the same conversation are deleted (`GLM53_NVME_SPILL_KEEP_PER_CHAIN`, default 2), and a bug was fixed where, at
  the capacity cap, resume points of DSH conversations (which all share the head blocks) could not be evicted and
  saving silently stopped (section 2.6). 4 h x 5 seeds: re-reads and collapses unchanged (-66.9%, 0); steady disk
  growth 1.31 -> 0.37 GiB/h/rank (-72%). Under branch stress K=1/2/3 give 1.4/0.2/0 collapses.
  (`results/prune/aggregate.txt`.)
* TTL + reuse across restarts (TTL+persist first cut, then final version): a per-session retention time sent by the
  client in `kv_transfer_params.glm53_spill` (off / 1 h / 1 d / 3 d), and reuse of the files across restarts (index +
  signature + a fixed block-hash seed). **In production the block hashes changed at every restart** (no
  `PYTHONHASHSEED`, so `NONE_HASH` was random); the connector sets the seed in the scheduler process only. In the rig
  with a restart in the middle, DSH re-reads after the restart went from 2,028,530 to 174,139 (-91%; first turn of each
  conversation after the restart 1,903,966 -> 24,823), collapses 3 -> 0, bytes written -41%. Without a restart the
  4 h x 5 seed result stays at -66.9% with 0 collapses. (`results/ttl/aggregate.txt`, `results/ttl/final/`.)
* Finish release, u4 and u5 (2026-10-14; this repository's `code/` is u5, the version running in production):
  * **B8 fixed** (section 2.9): no restore is started in a step in which the scheduler's decode-floor policy skips the
    request anyway, and a restore takes its fresh blocks from cached blocks whose bytes are on NVMe before blocks that
    exist only in GPU memory. In the rig's TTL mix the conversation sent with `ttl_s=0` went from +113% re-read vs OFF
    to +2.5% (-1.2% per admission); DSH -66.9% with 0 collapses and -91.5% after a restart are kept
    (`results/finish/aggregate.txt`). Under overload the `ttl_s=0` conversation is still +30.7% (+16.5% per
    admission), see `docs/limitations.md`.
  * `DEFAULT_TTL_S` (section 2.10): resume points of requests without the client field can expire too.
  * The identity's "code" part is the connector's syntax tree without docstrings (section 2.8.1), so a comment-only
    change no longer wipes the files at the next start.
  * **u4** (section 2.8.2): vLLM writes a random `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` into each process's environment
    on first read, so the worker's environment digest changed at every start and **every production restart wiped**
    the tier. The name is excluded (exact match), and a mismatch now names the variables that differ.
  * **u5** (section 2.8.3): the index is also written when the engine goes idle (plus a 2 s safety timer and a
    synchronous write at shutdown). Our production stop is `docker rm -f` (SIGKILL), so the connector's shutdown hook
    never ran and the resume point confirmed by the engine's last step before an idle stop was never indexed.
  * **Production:** with u5 a restart adopted all 4 saved resume points (92 files, 2.14 GiB/rank, 0 orphan files),
    and the quality probe across that restart proved the restore in 4/4 runs with KL and acceptance PASS
    (`docs/measurements.md` 8.6). With u4 the same procedure adopted 3 of 4 (the 4th was lost to the index gap that
    u5 closes).

## 1. Why not vLLM's OffloadingConnector

Judged from the code of the production vLLM tree (the image's vLLM after the launcher's start-up patches).

| Question | Evidence (production vLLM tree) | Result |
|---|---|---|
| Can a hit larger than the primary tier be restored piecewise? | `v1/kv_offload/tiering/manager.py` `lookup()`: a secondary-tier hit is always promoted through `_initiate_promotion()`, which first does `prepare_write` into the CPU primary tier, and only then goes to the GPU (`prepare_load` only loads from the primary tier). If the primary tier is full, `prepare_write` returns None and the lookup returns **MISS**. `offloading/scheduler.py::_maximal_prefix_lookup` stops at the first MISS. Moreover `Scheduler.schedule()` asks the connector only when `num_computed_tokens == 0` (scheduler.py:1465), so a request gets only one load | **No.** The hit is capped by the primary-tier capacity (with a 1-2 GB primary tier, 600k tokens = 3.6 GB/rank does not come back; the result is a partial hit plus recomputation) |
| Are KDA/mamba boundary states restored together and counted as a hit? | `get_sliding_window_size_in_chunks` treats `MambaSpec` as a sliding-window group with a one-chunk window, and mamba align is handled through `requires_cow_source` | The mechanism exists, but for the reason below it yields 0 in the production configuration |
| Does it work with the production 7-group layout? | Group 1, `KpoolTailSpec` (derived from `SlidingWindowSpec`, block 16, **no prefix caching, so no hashes**), is looked up upstream like an ordinary sliding-window group -> always MISS -> `_lookup_complete_chunks` returns 0 for the whole request if any group has 0 (`if num_hit_chunks == 0: return 0`) | **The hit is always 0** (group 1 would have to be excluded) |
| DFlash window | Upstream treats the last chunk of the eagle group as "volatile", excludes it from save and load, and asks for window+1. Production uses `[glm53-dflash-boundary-lookup-v1]` (the window ends exactly at the boundary), which has a different meaning | Does not match (would need changes) |
| GB10 memory | The primary tier is pinned CPU memory, i.e. the same DRAM as the GPU. The head node has 4 GiB free | The primary tier cannot be made large |

So the upstream route needs "a primary tier of several GB + excluding group 1 + changing the DFlash boundary
semantics + a new piecewise load", which is far more than a small patch. We wrote our own connector instead.

## 2. Design

### 2.1 Unit of saving: the resume point (RP)

* In the **step after** a request's last prompt chunk was computed (after the worker has finished that forward), the
  connector calls `HybridKVCacheCoordinator.find_longest_cache_hit(request.block_hashes, num_tokens-1)` and takes the
  returned block set (MLA+indexer blocks 0..N, the state of each of the four mamba groups at N, the DFlash window,
  every key each block carries) as the RP. It is exactly "what the next turn of this conversation would hit if it
  arrived now".
  * Why right after the prompt is computed and not at request finish: the boundary state of the last chunk is released
    when the ingestion finishes (cached and evictable) and is evicted by other ingestions during decoding (minutes).
    Measured in the replay rig: taking the RP at finish gave `rp_skipped_short`, and one rig conversation collapsed to
    13,824. This agrees with the earlier finding that protecting boundary states inside the pool is zero-sum and that
    capacity is the real problem.
  * Why not write at preemption/eviction time: if the boundary state is already gone at that point, the resume point
    is incomplete. In addition, the V2 runner runs the mamba-align pre-copy (`preprocess_state`) before
    `handle_preemptions`, so there is no hook to copy a preempted block before it is overwritten (it would need runner
    changes). Instead there is a **usage threshold (HIGHWATER)**: an RP that finishes while usage is below the
    threshold is remembered and written ahead once usage crosses it (`sweep`). Default HIGHWATER=0 (always write).
* If anything is missing, it is not used: an RP becomes `ok` only when every block was written on every rank. A load
  attaches keys only when it succeeded on every rank.
* One file = one block (named by the sha256 of the primary key, content addressed). The next turn of the same
  conversation shares the MLA blocks, so from the second turn on only the new MLA blocks + the 4 mamba states + the
  DFlash window are written (deduplication).
* Which conversations are written (`SAVE_POLICY=repeat`, default): only requests that extend the RP of an earlier
  finished request (= the second and later turns of a conversation). One-off requests are not written. RPs shorter than
  `MIN_TOKENS` (default 65,536) are not written. Above the capacity cap `GIB`, whole RPs are deleted in LRU order
  (shared blocks survive through reference counts). `MAX_WRITE_GIB_PER_H` can cap the bytes written per hour
  (default 0 = unlimited).

### 2.2 Restore path

* At admission, `get_num_new_matched_tokens(request, local_hit)` walks the request's hash chain from the deep end.
  If there is an `ok` RP deeper than `local_hit + MIN_GAIN` (default 16,384), it
  (1) pins (touches) the RP blocks still in GPU memory, (2) allocates fresh blocks only for the missing ones (ordinary
  LRU eviction), (3) sends both ranks a "file -> block" load job, and makes the request wait (returns `None` = ask again
  next step). If there is not enough free space it waits up to `SPACE_WAIT_S` (20 s), and otherwise takes the normal
  path.
* Worker: a load thread reads with O_DIRECT into a 32 MiB pinned staging buffer, checks the header (file name, group,
  rank, signature, row layout) and a CRC-32 per 8 MiB, then copies into the rows of each KV tensor on a separate CUDA
  stream. Because a freshly allocated block may still be written by a batch in flight (async scheduling, aborts), the
  device is synchronized before writing.
* Success on every rank -> the scheduler registers the **original keys** on the fresh blocks
  (`BlockPool._insert_block_hash`, primary + extra keys, and `block_hash_num_tokens`). The request's next lookup finds
  the deep hit and goes through ordinary admission (mamba CoW/seed, DFlash boundary lookup, retention: all ordinary
  code). Pins are released after admission (`update_state_after_alloc`). If the request is not admitted within
  `PIN_TIMEOUT_S` (5 s), only the MLA pins are released (the blocks stay in the cache); the mamba/DFlash state blocks
  stay pinned until admission (at most `MAX_STATE_PINS` = 64 blocks in total). Several requests asking for the same RP
  at the same time join one load.
* Failure on any rank (missing file, CRC, size, header mismatch) -> the fresh blocks are freed without keys, the RP is
  dropped, and the request takes the normal prefill.
* No answer (fix release): if a load job is not answered by every rank within `LOAD_DEADLINE_S` (30 s), the request is
  detached to the normal prefill (it is not restored again), pins on blocks that were already cached are released, and
  **the fresh blocks are quarantined until every rank has answered** (so a late write cannot corrupt another request's
  KV; once the answer arrives they are freed without keys). A save job releases its pins after `SAVE_DEADLINE_S`
  (120 s). (A save only reads the GPU. A broken file that a late copy might produce is always removed: the delete issued
  at the deadline is queued behind that job in the rank's write FIFO.) Either deadline miss, an unexpected exception on
  the worker side, or `BREAKER_FAILS` (6) consecutive failed jobs opens the **breaker**: no further save or restore is
  sent to any rank (the decision is made in one place, the scheduler, so both ranks agree automatically), unsent saves
  are dropped and all RPs are forgotten. It stays latched until restart. A new save is not started while the save pins
  held **only by the connector** (blocks that a running request also references are not counted) exceed
  `MAX_PINNED_BLOCKS` (default half the pool = 291). This bounds the path where pins accumulate when one rank stops
  answering; restore pins are bounded separately by `PIN_TIMEOUT_S`, `MAX_STATE_PINS` and the load deadline. The
  worker's I/O threads catch every exception per job and always answer (a thread can no longer die silently).
* (Fix release, B6) saves planned inside `build_meta` (the save right after a prompt is computed, and `sweep`) are
  shipped with **the next step's** metadata. The worker records its "wait for all work queued on the device" event in
  each step's `pre_forward` (before that step's forward), so without this delay a block keyed in the same step (written
  by that step's forward) could be copied before it was written.

### 2.3 Agreement across TP=2 ranks

Every decision is made once in the scheduler (EngineCore, one process), and the same command is sent to both ranks in
the connector metadata. Each rank returns per-job {success, failure} counts in `SpillWorkerMeta`; the executor's
aggregation (`aggregate` = sum) is added up by the scheduler until it reaches the world size. A job counts as
successful **only when the success count equals the world size**.

### 2.4 Byte positions of a block

`kv_cache_utils._glm5_next_tensor_layout`: 11 MLA tensors (page 2,359,296 B) are shared by MLA layer i, layer i of each
mamba group and DFlash layer i; 11 indexer tensors (page 152,064 B) are shared by indexer layer i and kpool tail layer
i. The runner (`v1/worker/gpu/attn_utils.py`) uses every tensor page by page for every layer (mamba as
`raw[:nb*page].view(nb,page)`; padded/KV-first attention layers also start at page boundaries), so "block b of group g"
is row b of every tensor that a layer of group g lives on.

File size of one block: MLA+indexer 27,624,960 B, mamba 21,233,664 / 18,874,368 B, DFlash window 11,796,480 B (plus a
4 KiB header). (These equal 11 x 2,359,296 + 11 x 152,064, 9 x 2,359,296, 8 x 2,359,296 and 5 x 2,359,296 B; our
arithmetic.) The kpool tail (group 1) is a per-request scratch area that is not prefix-cached, so it is not saved
(the same as an ordinary hit in GPU memory).

### 2.5 Precision

What is restored are the bytes that were written (no conversion). Restored blocks carry the same keys as before
eviction, and everything afterwards behaves as if "that block had never been evicted". If the RP contains the DFlash
window it comes back with the same bytes; if it does not, the coordinator's replay clamp works exactly as for an
ordinary hit in GPU memory (the SGLang #31600 type of bug, "only the drafter state is not restored", cannot happen by
construction).

### 2.6 Deleting resume points of older turns (prune release, `KEEP_PER_CHAIN`, default 2)

* Behaviour seen in our production A/B log (sizes measured; the growth rate below is an estimate): when the next turn of a conversation writes a new RP, the old RP stays
  until LRU removes it. MLA files are shared (reference counts), but each old RP keeps its 4 mamba states + DFlash
  window (about 92 MB/rank/turn). At the estimated ~10 GB/h/rank (measured total was 7.8 GB at the time pruning was added) disk usage would reach the
  200 GiB cap of our deployment in less than a day (extrapolated, not observed); after that, RPs of **other** conversations were pushed out by LRU (`_make_room` does not delete an
  RP that shares files with the new RP, i.e. older turns of the same conversation, so it is other conversations that
  disappear).
* When a new RP becomes `ok` **on every rank** (`_check_rp`), the true prefixes of its hash chain, i.e. the RPs of the
  earlier turns of the same conversation, are sorted by depth; counting the new RP itself, the newest
  `KEEP_PER_CHAIN` are kept and older ones are removed with `_drop_rp` (decided in one place, the scheduler ->
  `SpillMeta.deletes` -> each rank's write FIFO; shared files stay until their reference count is 0). Prefix test:
  the RP holds `request.block_hashes[:n/4608]` (`_RP.chain`), and X is a prefix if `chain[X.n/4608-1] == X.key`
  (a chained hash, so the whole prefix matches).
* Not deleted: RPs referenced by a restore (loading, pinned while waiting for admission, the admitted request still
  running, quarantined), RPs in a load job, and RPs still being saved (not `ok`). Also kept: an RP that is within
  `KEEP_PER_CHAIN-1` predecessors of **another branch** (a sibling RP created by editing or regenerating, which is not a
  prefix of the new RP); this protects "the one before" on that branch. The keys of deleted RPs stay in `seen` (with
  `SAVE_POLICY=repeat`, later turns that branched off before the kept RPs are still written as a "known
  conversation"). A new RP whose save failed deletes nothing.
* `KEEP_PER_CHAIN=0` = previous behaviour (LRU only). Why 2 (replay rig, `results/prune/`): a regeneration (the same
  prompt sent again) and a DSH automatic compaction (whole conversation + instruction = extension of the newest) hit
  the newest RP, while **editing the last user message** hits the one before (when the start of the last message lies
  before the boundary of the newest RP). With K=1 edits miss and re-read. Editing the message two back needs K=3.

### 2.7 Per-session retention time (TTL, final version, client contract v1.1)

* A request carries `kv_transfer_params.glm53_spill = {ttl_s, session, epoch}` (DSH attaches it to every
  `/v1/responses` request; it travels on vLLM's standard path `SamplingParams.extra_args["kv_transfer_params"]` ->
  `Request.kv_transfer_params`; see `docs/client-contract.md`). `parse_spill_params` reads it defensively **per
  field**: `ttl_s` must be a real int in 0..7 days, `session` a str of 1-64 characters, `epoch` an int >= 0 (bool,
  float, string or out-of-range values are **ignored for that field only**; no exception is raised and a request is
  never failed). Requests without the field (other clients) get the default = previous behaviour (no TTL, capacity
  LRU), or, since the finish release, the optional server default `DEFAULT_TTL_S` (section 2.10).
* An RP holds `ttl_s`, `last_used` (wall clock, persisted in the index), `session` and `epoch`. **Sliding**: the first
  lookup of a request that extends the RP, the start of a restore, and a repeated save of the same RP set `last_used`
  to now. A request with an explicit `ttl_s>0` updates the RP's `ttl_s` to its own value (changing 1 day to 3 days
  applies from the next request). Expired = `now - last_used >= ttl_s`. Expired RPs are **not used for restores**.
* `ttl_s=0` (do not keep): the request's RP is **not written**, and **every RP on that chain (on the request's hash
  chain, with the same session or no session) and every RP of the same session is expired** (deleted at once if not in
  use, otherwise marked and cleaned up after release). That request itself is not restored either. RPs of a different
  (explicit) session on the same prefix are not deleted.
* Cleanup (every `TTL_SWEEP_S` = 30 s, decided only by the scheduler, through the existing delete path
  `SpillMeta.deletes` -> each rank's write FIFO, so both ranks agree): expired RPs that are not in use are removed with
  `_drop_rp(..., "expired")`. Shared files stay until their reference count is 0.
* Eviction order at the capacity cap (`_make_room`): **expired and old-epoch RPs -> RPs without TTL (default) -> RPs
  with shorter TTL -> LRU within each class**. If no RP has a TTL the order is exactly the previous LRU. The free-space
  floor (`MIN_FREE_GIB`) is still always enforced at write time (the floor is never broken to keep an RP with a TTL).
* epoch: the maximum epoch per session is recorded (**at admission and at save**; not for a request that is only
  waiting in the queue). When a request with a **strictly larger** epoch is admitted, the session's RPs of older
  epochs are marked "stale" (`stale_at`): at the capacity cap they are evicted first, like expired ones, and they are
  deleted by the cleanup if unused for `EPOCH_GRACE_S` (30 minutes). During the grace period a late request of the old
  epoch can still use them for a restore (which extends them). Saves of old-epoch requests are still made, but are
  marked stale when created. "Keep only the newest RP per session" is **not** done: title-generation and summary
  requests arrive with the same session/epoch and would push out the conversation's RP. Older turns are still pruned
  **per chain** by `KEEP_PER_CHAIN`. Child sessions (delegated sub-tasks that the client runs as separate
  conversations) send their own session (contract v1.1), so they never touch the parent's RPs.
* First turn (`TTL_SAVE_POLICY`, default `all`): **only for requests with an explicit `ttl_s>0`**, the first turn is
  written even with `SAVE_POLICY=repeat`. Reason (`results/ttl/`): the extra writing happens only once per conversation
  (later turns are deduplicated to the increment); in the 4 h rig it was +0.29 GiB/rank (+2.0% of all writes, re-reads
  identical). On the other hand, after a restart or a long idle period (hours to days, which is exactly when a client
  sets a TTL) right after the first turn, a conversation that was not written has to be re-read in full (430k-780k
  tokens; measured in the restart rig). `MIN_TOKENS` is not relaxed (`TTL_MIN_TOKENS`=0 means the same as `MIN_TOKENS`,
  65,536): re-reading fewer than 65k tokens costs less than 32 s at 2,036 tok/s, and relaxing it would write many short
  requests of child sessions (which inherit the TTL and send their own session).
* Appended to the end of the stats line: `expired=` `epoch_dropped=` `ttl0_skipped=` `ttl_rps=none/1h/1d/3d/exp`
  (counts) `ttl_gib=...` (GiB of files referenced by each class) `ttl_classes=` `persist=` `adopted=` `adopted_gib=`
  `index_age_s=`, and since the finish release `restores_started=` `restores_admitted=` `restores_deferred_policy=`
  (section 2.9). The order of the existing fields is unchanged (monitoring reads `key=value`).

### 2.8 Reuse across restarts (final version, `PERSIST`, default 1)

**Block hashes change at restart** unless `PYTHONHASHSEED` is set: vLLM's prefix-cache hashes are chained from
`kv_cache_utils.NONE_HASH`, and `init_none_hash` uses `os.urandom(32)` when `PYTHONHASHSEED` is unset. Our production
start-up did not set it, so every restart changed every block hash. EngineCore calls `init_none_hash` **after** it
creates the Scheduler (and with it this connector), so the scheduler-side connector (with `PERSIST=1`) sets
`PYTHONHASHSEED=HASH_SEED` (default 5353; left alone if already set) in its own process environment only. Python's own
`str` hashing is fixed at interpreter start and is not affected. On the first step the connector checks
`NONE_HASH == hash_fn(PYTHONHASHSEED)` and disables reuse if it does not match (wipe everything, write no index =
previous behaviour). The `NONE_HASH` digest is part of the signature. The details, the measurements
(`tests/t_hash_stability.py`) and why we do not set the variable for the whole container are in
`docs/hash-seed-finding.md`.

* Index (scheduler, on the head node (rank 0) at `$SPILL_DIR/glm53-nvme-spill/index.json`): **only RPs that are `ok` on
  every rank** (key, n, chain hashes, and for each block its group / all keys / num_tokens / file name, plus
  TTL/last_used/session/epoch), a file table (file name -> group, **id of the save that wrote it**), the `seen` set of
  `SAVE_POLICY=repeat`, the epoch per session, and the signature. The first line holds the body length and sha256.
  Written to `.tmp`, fsync, rename, directory fsync (on a separate thread, every `INDEX_EVERY_S` = 5 s and only when
  something changed; immediately for ttl 0, the breaker and the start-up decision). Written synchronously on a clean
  shutdown (`EngineCore.shutdown` -> `connector.shutdown`). After a crash the last index remains (see "stale index"
  below). **Since u5 it is also written at once when the engine goes idle** (section 2.8.3): a stop by SIGKILL
  (`docker rm -f`) never runs the shutdown hook, so persistence must not rely on it.
* The id of the save that wrote a file (`<boot>-<job>`) is put into the file header (JSON key `w`) and passed to the
  load job as the expected value. On mismatch the load fails, the RPs that use that file are dropped, and the request
  takes the normal prefill. This prevents **mixing KV from two different computations across ranks**, for example when
  one rank deleted and rewrote a file after the last index was written.
* Start-up (everything decided in one place, the scheduler, with the same command to both ranks): each rank, **without
  deleting anything**, reports as the first job of its writer thread its file list (name -> size; `.partial` files are
  deleted) and its own **signature** (`SpillWorkerMeta.hello`, once). The scheduler adopts the files only if
  (1) `NONE_HASH` is deterministic, (2) the index is readable and its sha256 matches, (3) the scheduler signature
  matches the index exactly, (4) the world size is the same, and (5) reports from **every** rank arrived and each
  rank's signature matches the index exactly. If any of these fails, **every rank is wiped** (`SpillMeta.wipe`, queued
  in each rank's write FIFO ahead of any later save). Ranks that disagree with each other (one rank's signature differs,
  or a report carries an error) also cause a full wipe.
  On adoption: expired RPs are dropped, RPs using a file that is not present with the right size on every rank are
  dropped, the rest are adopted in their LRU order, and files no RP references are deleted **per rank**
  (`SpillMeta.rank_deletes`). If the capacity cap was lowered, eviction runs right after adoption in the normal order.
  Contents (CRC) are checked when read, as before (not all read at start-up). Until the decision, nothing is saved or
  restored (while an index exists, waiting requests get `None`; after `ADOPT_DEADLINE_S` = 60 s the connector gives up
  and wipes everything).
* **Signature** (a digest per part; the names of the differing parts are logged as the reason for a wipe): connector
  version and file/index format; Python, torch and vLLM versions; configuration (model / tokenizer / dtype /
  quantization / revision / hf_config_path / hf_overrides, cache_dtype / block_size / mamba_cache_mode /
  mamba_ssm_cache_dtype / prefix_caching_hash_algo / prefix_match_unit, speculative method / model / number of tokens,
  TP / PP / world); KV layout (page bytes and sharing layers of each tensor, spec repr and layer names of each group),
  file bytes per group, hash_block_size; **the content of every `*.py` and the size of every `.so` in site-packages**
  (the image's vLLM + every overlay patch applied at start-up; 27,914 files on the production image, 4.4 s cold /
  0.4-1.0 s warm, on a separate thread; since the finish release this connector's own file is counted by name only and
  covered by the part "code" = its syntax tree without docstrings, section 2.8.1); the model and drafter directories
  (name, size and mtime of every file, content for files of 1 MiB or less); part of the environment (a digest of names
  and values of `GLM53_` / `EXL3_` / `TF_` / `VLLM_` / `DFLASH` / `MTP_` / `SPEC_` / `KV_CACHE` / `QUANTIZATION` /
  `PREFIX_MATCH` / ... variables, excluding keys, tokens, passwords, logging, ports, hosts/IPs, this connector's own
  knobs and, since u4, the run-time generated names of section 2.8.2); the `NONE_HASH` digest and the hash function
  name. A rank's signature adds its rank number, row layout and header
  signature (each rank compares with its own previous value). The number of blocks in the KV pool is not included
  (a file holds only the page rows of one block, so it stays valid when the pool grows or shrinks by a few blocks; the
  header signature is likewise built from page bytes).
* Stale index (after a crash): a file listed in the index but missing -> its RP is dropped; a file not in the index ->
  deleted as an orphan; an RP that was deleted but is still in the index -> revived if its files exist (it is a valid
  RP), dropped otherwise; the same file name written by a different save -> the write-id check fails at load -> normal
  prefill. Each case costs at most one normal prefill.
* When the breaker opens, the index is rewritten empty (the next start adopts nothing and deletes every file as an
  orphan). `PERSIST=0` = previous behaviour (each rank deletes all its files at start-up, the index is neither read
  nor written, `PYTHONHASHSEED` is not touched).

#### 2.8.1 The identity's "code" part (finish release)

Before the finish release the site-packages digest hashed every `.py` byte for byte, so a one-character comment fix of
the connector wiped the NVMe tier at the next start. Comments, docstrings, blank lines and line breaks never reach the
bytecode, and the connector does not read its own `__doc__` (nor does anything else), so versions that differ only in
those behave identically. The connector's own file is therefore counted by name in the site digest, and a separate
part "code" holds the sha256 of the module's syntax tree with every module / class / function docstring removed,
dumped without positions (`_code_digest`). Any change of an executable token (a name, a constant -- log format
strings included --, an operator, a default, an import, an annotation) changes "code" and wipes, as before. Every other
`.py` in site-packages stays byte-strict (a comment change in another file still wipes).

`tests/test_finish.py`: 5 edits of comments, docstrings, blank lines and line breaks leave "code" unchanged; 7 kinds
of edit change it (log format, constant, name, operator, default, how a key is built, a string that is not
a docstring); with a fake site-packages only the connector is left to "code" and every other file stays strict.

The environment digest still covers every `GLM53_` etc. variable except secrets, logging, ports and this connector's own
knobs, so toggling an unrelated `GLM53_` variable (for example `GLM53_EXPOSE_CACHE_RESET`) also wipes (kept on the strict
side). This repository's `code/glm53_nvme_spill_connector.py` differs from the production u5 file only in comments and
docstrings, so its "code" digest is the same as production's (`results/provenance_connector.txt`).

#### 2.8.2 Run-time generated environment values; naming what differs (u4)

**Production symptom (2026-10-14):** the second restart after installing the finish release changed nothing, yet the
log said `persist: wipe every rank (rank signature differs (rank 0: env; rank 1: env))` (the scheduler's signature
matched).

**Cause:** vLLM's `envs.py` defines `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` with
`get_env_or_set_default(..., lambda: f"VLLM_OBJECT_STORAGE_SHM_BUFFER_{uuid.uuid4().hex}")`: **the first read writes a
random value into `os.environ`.** `enable_envs_cache()` (at the end of `EngineCore.__init__` and of `WorkerProc.__init__`)
reads every vLLM variable, so every process has a different value at every start, and an environment digest that covers
all `VLLM_` variables changes every time. `/proc/<pid>/environ` of the worker shows only the environment at exec time,
so the value is invisible from outside. The scheduler side matched only because its signature (computed on a separate
thread) happened to finish before `enable_envs_cache()` ran (a race). Our CPU rig did not reproduce it, because its
simulated workers never read the variable. Proof: `tests/test_u4_env.py` part `fresh`: two fresh interpreters each run
`enable_envs_cache()`; with the finish-release connector the environment digests differ, with u4 they are equal; a
control that does not call `enable_envs_cache()` gives the same value for both versions.

**Fix:** the names in `_ENV_VOLATILE` are excluded **by exact name**; every other variable under the prefixes stays
strict (similar names such as `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAMEX` too; unit-tested). The excluded names:

* `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` (above; it names the shared-memory segment of the multimodal processor cache and
  has nothing to do with the KV cache);
* `VLLM_DIST_IDENT` (`v1/worker/cpu_worker.py` writes the distributed-init file name / port at every start; CPU workers
  only, so not in production, listed for the CPU rigs).

Audit: a static scan of the production vLLM tree, the overlays and the image's site-packages for run-time writes into
`os.environ` of names under the prefixes. All other writes are constants or deterministic configuration values (for
example `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`); `_VLLM_INTERNAL_*` names start with `_`
and do not match the prefixes; Ray / XPU / ROCm / benchmark-only paths do not run here. Dynamically, two fresh
interpreters import the modules a scheduler and a worker import, run `enable_envs_cache()` and compare the prefixed
variables: the only name whose value differs between the two processes is `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`
(`test_u4_env.py` part `audit`).

**Diagnostics:** each identity part now carries a breakdown "name -> first 8 hex digits of the value's sha256" (the
environment variable names, each config field, python / torch / vllm, each model and drafter file, the top-level
entries of site-packages (vllm to its second level), each layout tensor and group). **Values are never stored**, and
the environment breakdown holds the same names as the digest, so variables with secret-like names are not in it
either. The scheduler stores its own breakdown and each rank sends its own in the start-up report; the index keeps them
as `sched_detail` / `rank_detail`. They are **not used for the decision** and are not signed. A wipe reason now names
what differs, for example `scheduler signature differs: env [changed VLLM_X / added GLM53_Y / removed EXL3_Z]` or
`rank signature differs (rank 1: env [added VLLM_U4_RANK_FLAG])`; an index without a breakdown (written by an older
version) gives `env [no per-name record in the index]`. Size: about 10 KB per process on the production image (312
site groups). The index format and `INDEX_VERSION` are unchanged (readers ignore the new keys). Installing u4 changes the
code, so its first start wipes once (`scheduler signature differs: code, version`); adoption works from the next restart.

**General lesson for vLLM users:** any identity, cache key or fingerprint derived from a vLLM process's environment must
exclude `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` (and any other value a process generates for itself), and must not be
validated by reading `/proc/<pid>/environ`, which does not show such values.

#### 2.8.3 Writing the index before the engine goes idle (u5)

**Production symptom (2026-10-14):** the quality probe's prepare phase saved 4 resume points; the last index write came
4 s before the 4th save, the engine then went idle, and the stop (`docker rm -f`) came 30 s after that save. The next
start logged `adopted 3 resume points ... orphan files deleted per rank {0: 23, 1: 23}`, and the probe's 4th run found
no restore (`cached_tokens=0`).

**Cause (from the code):** (1) the index was written only from `build_meta` (the start of an engine step), throttled to
`INDEX_EVERY_S`. The rank answers that confirmed the 4th resume point were processed in the last step's
`update_connector_output`; once `has_pending()` returns False, EngineCore blocks in `input_queue.get(block=True)`
(`while not self.has_work()` in `core.py`), and the next `build_meta` only comes with the next request. (2) The clean
shutdown write (`Scheduler.shutdown` -> `connector.shutdown` -> forced index write) runs on SIGTERM (signal handler ->
`_handle_shutdown` -> the `finally` of `run_engine_core` -> `EngineCore.shutdown`), but our launcher stops containers
with `docker rm -f` (SIGKILL), so **it had never run**. The newest resume point (the latest turn of the conversation
that was active right before a restart, i.e. the most valuable one) was missing from the index, and its files were
deleted as orphans at the next start.

**Fix:**

* When `has_pending()` returns False (nothing left for the workers; vLLM's `Scheduler.has_requests()` asks right before
  the engine blocks idle, and again before exiting on SIGTERM), a changed index is written **at once, without
  throttling** (`_flush_index("idle")`, handed to the existing writer thread: `.tmp` -> fsync -> rename -> directory
  fsync as before). While a save is in flight `has_pending()` is True, so only resume points confirmed by every rank are
  ever indexed (unchanged rule). Exceptions never reach vLLM (a warning and `index_write_failed`; retried at the next
  opportunity).
* Safety net: a timer thread on the scheduler side (0.5 s tick) writes a changed index when the engine has made no
  connector call (`build_meta` / answers) for `INDEX_IDLE_S` = 2 s (an engine whose idle path does not ask
  `has_pending`, or one long step). Every scheduler-side entry point (`get_num_new_matched_tokens`,
  `update_state_after_alloc`, `build_connector_meta`, `update_connector_output`, `request_finished(_all_groups)`,
  `has_pending_push_work`, bind) holds `SpillScheduler.lock` (an RLock), so the timer only sees consistent state between
  two calls. The CPU replay rig runs on a simulated clock, so the timer is off there (`NVMESPILL_RIG_INDEX_TIMER=1`
  turns it on).
* Clean shutdown: the timer is stopped (joined outside the lock), then the last index is written synchronously under the
  lock. EngineCore only gets there after `has_work()` turned False (no save in flight), so on SIGTERM every answer has
  been processed.
* Each idle / timer / shutdown write logs one line after the rename:
  `persist: index written (idle: N resume points, M files, B bytes on disk)`.
* No new knob (`KNOBS` unchanged, so the launcher patch is unchanged); index format and adoption rules unchanged.

**Tests:** `tests/test_u5_index.py` 25/25; run against the u4 connector with `--old` it reproduces the bug 2/2 (the
resume point confirmed by the last step is missing from the index after the idle period; with two real ranks and a
SIGKILL restart: 1 adopted, 10 orphans). Rig `pr_idle_*` (`NVMESPILL_RESTART_IDLE=1`: arrivals stop, the engine steps
while `has_requests()` exactly as EngineCore does, idles 20 s, then two probe-like requests of 81k tokens with
`max_tokens` 1, then SIGKILL): with u4 all 3 seeds lose one resume point (46-58 orphan files); with u5 the 3 seeds plus
a clean-stop variant lose none (`results/u5/compare_u5.txt`, `tests/compare_u5.py`).

**Operational lesson:** a container stopped with SIGKILL never runs `connector.shutdown()` (nor any other shutdown
hook), so anything that must survive a restart has to be persisted before the stop, not by the shutdown path.

### 2.9 No restore ahead of a request the scheduler will not admit (B8, finish release)

* Problem (review finding B8): a restore starts when the scheduler's waiting-queue loop asks the connector, but right
  after that the decode-floor policy of our launcher (`[glm53-decode-floor:v5]`) may skip the request (skip mode: no
  prefill is admitted while other requests decode; fair mode: it is not this request's prefill turn in this step). For
  the skipped request `get_new_blocks` evicted other requests' cached blocks, after 5 s (`PIN_TIMEOUT_S`) the MLA pins
  were released, the blocks were evicted while waiting, and the restore was read again -- repeatedly (rig seed 4: 52
  restores for 9 admissions). A conversation that is not on NVMe (`ttl_s=0`) relies on the GPU cache alone, so its
  prefix was evicted and it did worse than OFF (+113%, `docs/measurements.md` 7.5).
* Fix 1 (policy gate): right before starting a restore, `get_num_new_matched_tokens` calls
  `_policy_skips(request, rp.n_tokens)` and returns `None` **only when it is certain** that the policy skips this request
  in this step (the same outcome as the policy's own skip: the request just moves to the skipped-waiting list). The
  policy state is only read (the module variable `_GLM53_MIXED` of `vllm/v1/core/sched/scheduler.py`, the per-step
  record opened by `begin_step` at the top of `schedule()`): skip mode = "a request was decoding at the start of the
  step" (`had_decode`; conservative by one step only when the only decoding request is preempted in this step), fair
  mode = "a contention step and this request has no grant and is not selected", off / cap mode = never. **Our production
  runs with `GLM53_MIXED_PREFILL_CHUNK=0` = off, so the gate never fires there and production behaviour is unchanged.**
  Anything unexpected (no module, a missing attribute, an exception) -> False = restore as before. Not an operator knob;
  tests use `SpillConfig.policy_gate` / `NVMESPILL_TEST_NO_POLICY_GATE`.
* Fix 2 (**a restore takes NVMe-backed blocks first**, `_take_blocks`): `get_new_blocks` takes from the head of the free
  queue (never used, then LRU), so a restore pushed out prefixes that exist only in GPU memory (a `ttl_s=0`
  conversation, a one-shot request, a prefix not yet written). Fresh blocks for a restore are now taken (1) from free
  blocks that never had a key, (2) from cached free blocks **whose bytes are in an ok file on this tier** (the primary
  key names the file), in LRU order, (3) from the rest in plain LRU order if still short. A backed block evicted this
  way costs one read on a later hit, not a recompute. The mechanics equal `get_new_blocks` (removed from the queue, keys
  dropped, ref_cnt 0 -> 1), tested on the real `BlockPool` (`test_finish.py`). This applies to restores only; the
  normal allocation order (`allocate_slots`) stays vLLM's (making it NVMe-first globally would need a `BlockPool`
  change, not done). Not an operator knob (`NVMESPILL_TEST_NO_BACKED_FIRST` for tests).
* Stats line: `restores_started=` `restores_admitted=` `restores_deferred_policy=` appended (our monitor raises a WATCH
  when started minus admitted grows by more than 3 in 10 minutes).
* Correctness: `test_finish.py` drives the real policy class with 4,000 random states: whenever the gate says "skip", the
  policy returns 0 (sound); in skip mode they agree exactly; in mode 0 and cap mode the gate never fires. Rig results:
  `docs/measurements.md` 7.9.
* Rig versus production (**important**): the rig does not set `GLM53_MIXED_PREFILL_CHUNK`, so it runs the default skip
  mode (inherited from the rig of our earlier re-read investigation), while production uses 0 (off). The amplification of
  B8 (restores repeated for a request that is not admitted) comes from skip mode's "no prefill while decoding"; with off,
  only block shortage and the token budget can delay admission after a restore. The rig cannot produce mode-0 numbers:
  with mode 0 the OFF arm's queue overflows to 438 requests (the rig's time model is not calibrated for mode 0) and the ON
  arm stops on a rig-specific assertion (`results/finish/dg_mx0_off_s1.json`). In production the B8 counters
  (`restores_started` versus `restores_admitted`) are the check.

### 2.10 Expiry for resume points of requests without the client field (`DEFAULT_TTL_S`, finish release)

* Resume points of requests without `kv_transfer_params.glm53_spill` (batch jobs, headless clients: the "no TTL" class)
  used to expire only at the capacity cap (first after expired ones) or at a restart. With
  `GLM53_NVME_SPILL_DEFAULT_TTL_S` (seconds, 0-7 days; empty / 0 = no expiry, as before) they expire like an explicit TTL,
  **this long after their last use** (sliding: the first lookup of a request that extends the RP, the start of a
  restore, and a repeated save of the same RP extend it). Expired RPs are not used for restores, are deleted by the
  cleanup (`TTL_SWEEP_S`) and are dropped at adoption after a restart.
* Unchanged: (a) the **save policy** (still `SAVE_POLICY=repeat`; this is not an explicit "keep", so the first turn is not
  written); (b) the **eviction order at the cap**: expired / old epoch -> **this class (unexpired)** -> explicit TTLs,
  shortest first -> LRU (the position of the previous "no TTL" class, so batch requests do not push out the client's
  conversations). The RP's `ttl_s` stays `None` (index format unchanged); only the expiry test uses the default. An
  expired default-class RP that the conversation saves again is revived (explicit-TTL RPs are revived only by a request
  that asks to keep them, as before).
* The first class of the stats line's `ttl_classes=` is labelled `default` when a default is set (`none` otherwise); the
  scheduler start-up line shows `default_ttl_s=`.
* Our production value is 86400.
* Tests (`test_finish.py`): with default 0 an RP survives 7 days unused; default 100 s slides; expired RPs are not
  restored; the cleanup removes only that class; a repeated save revives; `SAVE_POLICY=repeat` writes no first turn; the
  cap eviction order is identical for default 0 and 86400; an expired default-class RP is evicted first; adoption after a
  restart drops expired ones; configuration checks. Rig: `docs/measurements.md` 7.9.

## 3. File map

| File | Content |
|---|---|
| `code/glm53_nvme_spill_connector.py` | The connector (scheduler side `SpillScheduler`, worker side `SpillWorker` + `BlockFileStore`, file format) |
| `code/patch_nvme_spill.py` | Installer in the style of the production overlays (idempotent, fails closed on an anchor mismatch, `GLM53_NVME_SPILL` 0/empty = does nothing): copies the module to `vllm/distributed/kv_transfer/kv_connector/v1/`, adds 2 lines to `scheduler.py` that hand the KVCacheManager to the connector, and exempts only this connector in `config/vllm.py` `_verify_kv_transfer_compat` (`docs/review-findings.md`, B1) |
| `code/patch_tf_bundle.py` | Launcher bundle hook: adds the line that runs the patch above when `GLM53_NVME_SPILL` is set |
| `code/apply_start_sh.py` + `code/start.sh.nvmespill.patch` | Additions to the launcher's `<kit>/launcher/start.sh` (validation, forwarding the env knobs to both ranks, a read-write mount of the spill directory, `--kv-transfer-config` in both inner scripts). Each anchor must match once; checked with `bash -n` and by reverse application |
| `code/spill_quality_probe.py` | Quality probe for the A/B (requires positive evidence of a restore; also a probe across a restart) |
| `code/conc_quality_probe.py` | Concurrency decode-vs-prefill quality probe used in the same A/B arms |
| `tests/test_store.py` / `tests/bench_gpu.py` / `tests/prodlayout.py` | Unit tests with production shapes; GPU speed |
| `tests/rig/` | Copy of the replay rig of our earlier re-read investigation + 2 simulated worker ranks (`sim/spill_world.py`) with content verification |
| `tests/run_gpu_tests.sh`, `tests/run_cpu_tests.sh` | test runners (one container at a time; the scratch directory is the only read-write mount besides the checkout) |
| `tests/test_sched_faults.py` | (Fix release) deadlines, quarantine, breaker, pin cap, B6 send order, B7, worker exception tolerance (fake pool + the real connector) |
| `tests/t_alloc_conf.py` / `tests/t_expandable_gpu.py` | (Fix release) B1: passes the check at `VllmConfig` construction / GPU I/O is byte-identical under expandable_segments |
| `tests/test_start_sh.sh` | (Fix release) B4: application to the kit's start.sh, idempotence, fail-closed, and tracing that every knob reaches `docker run` on both ranks |
| `tests/test_quality_probe.py` | (Fix release) B5: the probe FAILs when it cannot prove a restore |
| `tests/rig/env/*.env`, `tests/rig/jobs_*.txt`, `tests/rig/run_jobs.sh`, `tests/summarize_faults.py` | Re-run matrices and fault injection (2 at a time) / summaries |
| `tests/test_prune.py` | (Prune release) unit tests for deleting older RPs (fake pool + the real connector + the real `SpillWorker`) |
| `tests/aggregate_prune.py`, `tests/rig/jobs_prune*.txt`, `tests/rig/env/{k,bk,ck,sa}*.env` | (Prune release) K matrix, branch stress, capacity pressure, `SAVE_POLICY=all`. The rig records (observation only) `NVMESPILL_BRANCH` (regenerate, edit the last message, edit two back), disk usage over time, which RP each restore used, and a breakdown of DSH re-read causes |
| `tests/test_ttl_persist.py` | (Final version) TTL parsing, eviction order, sliding, ttl 0, epoch, first-turn policy, index round trip / partial / mismatch / corruption / expiry / deadline / non-deterministic hash / capacity / breaker / write rate; start-up scan, wipe, per-rank deletes and write ids of the real `SpillWorker`; 2 ranks + scheduler restart round trip (byte-identical) |
| `tests/t_hash_stability.py` | (Final version) block hashes change in a new process without `PYTHONHASHSEED` (previous behaviour), match with the connector, Python `str` hashes stay randomized, and the construction order in the production `core.py` |
| `tests/t_restart_gpu.py` | (Final version) on a GB10 with production page sizes: save -> adopt -> restore across a separate process (= restart) is byte-identical, and identical to an in-process restore |
| `tests/bench_persist.py` | (Final version) time to build / write / read the index, the adoption decision, the rank start-up scan and the site-packages digest |
| `tests/aggregate_ttl.py`, `tests/rig/jobs_ttl_*.txt`, `tests/rig/env/{ttl3,ttl3r,ttlmix,rsp0,rsp1,rscr,rsoff,plain,pr_*}.env` | (Final version) replay and summaries for mixed TTLs (`NVMESPILL_TTL`, `NVMESPILL_EXTRA_CONV`, `NVMESPILL_IDLE`, `NVMESPILL_COMPACT`) and a restart in the middle (`NVMESPILL_RESTART_AT`, `_DOWN_S`, `_CRASH`, `_INJECT`). The DSH field reaches the request through vLLM's path (`SamplingParams.extra_args`) |
| `tests/test_finish.py` | (Finish release) B8 policy gate against the real decode-floor policy class (4,000 random states), restore gating, NVMe-backed-first block choice on the real `BlockPool`, `DEFAULT_TTL_S`, the "code" identity part |
| `tests/test_u4_env.py` | (u4) run-time generated environment values: two fresh interpreters with `enable_envs_cache()`, an audit of every prefixed variable, exclusion by exact name, the per-name breakdown (names and short digests only), naming the differing variable in a wipe reason (scheduler and rank) |
| `tests/test_u5_index.py` | (u5) the resume point confirmed by the last step is in the index after an idle period, with the timer, under the lock, at shutdown, never raising; `--old` reproduces the bug on the u4 connector; 2 real ranks + SIGKILL restart |
| `tests/aggregate_finish.py`, `tests/compare_u4.py`, `tests/compare_u5.py`, `tests/rig/sim/diag.py`, `tests/rig/jobs_finish_*.txt`, `tests/rig/jobs_u4.txt`, `tests/rig/jobs_u5.txt`, `tests/rig/env/{dg,dt,fin,hv,pr_idle}*.env` | (Finish release, u4, u5) rig matrices and summaries; eviction attribution (`NVMESPILL_DIAG_EVICT`, observation only); preemption / re-admission counts; idle-then-SIGKILL restart (`NVMESPILL_RESTART_IDLE`, `_IDLE_S`, `_IDLE_PROBES`) |
| `results/` | Per-run summaries and aggregates (`results/fix/`, `results/prune/`, `results/ttl/`, `results/ttl/final/`, `results/ttl/final_faults/`, `results/finish/`, `results/u4/`, `results/u5/`, `results/published-tree/`, `results/published-tree-u5/`) |
| `THIRD_PARTY_NOTICES.md` | Design sources: TensorFold Spill-Tier (Apache-2.0) |
