# Limitations and known risks

This lists what we know is not fixed, not verified, or only verified under simulated conditions. Design references
are to `docs/design.md`, review findings to `docs/review-findings.md`, measurements to `docs/measurements.md`,
abort criteria and procedures to `docs/operations.md`.

## Scope of the code

* **Written for one specific patched vLLM tree.** The connector targets the vLLM inside the GLM-5.3-Flash EXL3
  launcher image plus the launcher's start-up overlays (the production vLLM tree). It depends on classes and helpers
  that exist only there or behave differently there, for example `HybridKVCacheCoordinator`, `KpoolTailSpec`,
  `kv_cache_utils._glm5_next_tensor_layout`, the 7-group KV layout, the DFlash boundary-lookup patch and the
  `scheduler.py` hook added by `code/patch_nvme_spill.py`. It has **not been tried on stock vLLM**. The design
  (resume point = the coordinator's own block set, restored with the same keys and bytes) is general; the code is not.
* **Not tested on real TP=2 hardware except in production.** All two-rank tests ran in the replay rig, which is
  CPU-only, uses two simulated worker ranks in one process and a simulated clock. GPU tests ran on a single GB10 with
  one rank (or two ranks in separate processes on the same machine). The real two-node setup was exercised only by the
  production A/B itself.

## Known risks (from the production A/B preparation, after the fix release)

1. **No real forward pass in the tests.** The KV byte positions were checked against the production layout
   (`_glm5_next_tensor_layout` and the V2 runner's per-page views) in code and verified with synthetic tensors.
   Correctness with the real engine is first confirmed by the A/B quality probe (the KL of R must be inside the band of
   H/F2).
2. **The replay rig is synchronous with a simulated clock.** During the real restore I/O time (0.1-1.5 s) the request
   waits (others keep running); the rig counts this as 0. Also, the production worker runs I/O and CRC on Python
   threads, which compete with the model-execution thread for the GIL (in the bench, I/O became about 2x slower; the
   other direction, the effect on decode, was within -1% of kernel bandwidth). Compare decode tok/s from the 10-second
   statistics during the A/B.
3. **Ordering with async scheduling.** A freshly allocated block may still be written by a batch in flight; the device
   is synchronized before a load and before a save. In addition, `kv_role=kv_both` turns on vLLM's `defer_block_free`
   (an upstream safety measure that delays returning an aborted request's blocks by one step). Normal finishes still
   return blocks immediately.
4. **What `--kv-transfer-config` changes:** vLLM's config hash (torch.compile is rebuilt on the first start),
   `requires_kv_delivery=False`, the `defer_block_free` above, and the hand-over of partial tails (empty in production,
   where fine-grained hits are disabled). Every connector hook runs only with `GLM53_NVME_SPILL=1`.
5. **Pins while waiting, and wasted restores (review B8).** Fixed in the finish release for the case where the
   scheduler's policy skips the request anyway; what remains is described in "B8 after the fix" below.
6. **No write at preemption/eviction time** (`docs/design.md` section 2.1). The RP is written right after the prompt
   is computed (can be deferred by HIGHWATER). So the first turn of a conversation (with `SAVE_POLICY=repeat`), and
   state evicted before the post-prompt write finishes (about 0.1-1.5 s), cannot be restored.
7. Reuse across restarts was not done in the first versions; it was added in the final version (`PERSIST=1`,
   `docs/design.md` section 2.8). Its own risks are listed in the next section.
8. **Agreement on the real two-node TP=2 setup** (aggregation of two processes' worker meta) was tested only with the
   rig's two simulated ranks (aggregation through an in-process `aggregate` call, the same function production's
   `KVOutputAggregator` uses). If a production overlay omitted the output of non-zero ranks, the first step would fail
   (review item 2d, not verified). Check that the stats line and the first `save rp` appear right after start-up.
9. **The breaker is latched.** One deadline miss, one worker exception or 6 consecutive failures stop saving and
   restoring until restart (behaviour falls back to the normal prefill, the same as OFF). Even a temporarily slow disk
   stops the feature: we chose the safe side. The deadlines have 2.6-3x headroom over our worst-case estimate.
10. **Quarantined blocks do not come back.** If a load misses its deadline and that rank never answers, the
    quarantined fresh blocks (the part that restore was missing; 17 blocks = 3% of the pool in the rig) stay unusable
    until restart. This is deliberate, so that a late write cannot corrupt another request's KV. `quarantined>0` is
    abort criterion A3.
11. **Deadlines are measured from send time** (including time queued on the rank). Situations with long queues (three
    first saves at once, etc.) are included in our estimate. The rig's async mode sets the I/O delay per block to
    10-25x real hardware, so only those runs used longer deadlines.
12. **The allocator exemption rests on code reading and a GB10 churn test.** torch's expandable segments do not
    relocate live allocations (`tests/t_expandable_gpu.py`: addresses unchanged and bytes identical across
    `empty_cache`). A production start with expandable_segments on both arms is confirmed at the first A/B start.

## Known risks of TTL and reuse across restarts (final version, finish release, u4, u5)

1. **A value that changes from start to start in any identity part makes every start wipe everything** (the safe side,
   the same as without reuse). This happened in production with the environment part: vLLM writes a random
   `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` into each process's environment (`docs/design.md` 2.8.2); fixed in u4 by
   excluding that name. Another, still unknown run-time generated value in the site / model / config / env parts would
   again wipe at every start. The wipe reason now names the differing variable or file; check that name before adding
   it to `_ENV_VOLATILE` (exclusion is by exact name only).
2. **A conversation with `ttl_s=0` (do not keep) can still get fewer GPU hits than with the connector off under
   overload.** After the B8 fix it is +2.5% in total (-1.2% per admission) in the normal TTL mix, but +30.7% in total
   and **+16.5% per admission** in the overloaded rig (`docs/measurements.md` 7.9). See "B8 after the fix".
3. **What a stop can still lose.** Since u5 the index is written as soon as the engine goes idle, by a timer after 2 s
   without engine calls, and at a clean shutdown. A resume point is indexed only once every rank has confirmed its
   save, so a stop (or crash) while a save is still unconfirmed loses that resume point and its files become orphans
   at the next start. This includes a restart procedure that waits only until no request is running and then sends
   SIGKILL: the save of the last prompt can still be in flight at that moment (writing a 600k-token resume point took 1.4
   s). Under load (the engine never idle) the index can be up to `INDEX_EVERY_S` (5 s) old at a crash. Each loss costs
   one normal prefill of that conversation.
4. `PYTHONHASHSEED` is set in the environment of the scheduler (EngineCore) process. Child processes that process starts
   afterwards get a fixed `str` hash. The `str` hash randomization of Python itself, the API server and the workers is
   unchanged (measured). (EngineCore currently starts no child processes after start-up.)
5. At start-up, while an index exists, waiting requests can be held for up to `ADOPT_DEADLINE_S` (60 s) (normally 1-2 s:
   the rank scan finishes during model warm-up; the scheduler's site digest takes about 1 s warm and about 4 s cold). In
   production the decision came 4.95 s (u4) and 2.10 s (u5) after the first step.
6. Adoption on the real two-node setup was confirmed in production with u4 and u5 (`docs/measurements.md` 8.6): both
   ranks reported, the decision was taken once and the per-rank orphan deletions ran on both nodes.
7. The benefit of writing the first turn (`TTL_SAVE_POLICY=all`) does not show up in the rig's load. Its cost is +2%
   bytes written.
8. **Every code change wipes once.** Installing a connector whose "code" part differs (any executable change), or
   changing any non-secret `GLM53_` / `VLLM_` / ... variable that is in the environment digest (for example toggling
   `GLM53_EXPOSE_CACHE_RESET`), wipes the tier at the next start. Only comment / docstring changes of the connector and
   changes of the connector's own knobs (`GLM53_NVME_SPILL_*`) keep the files.
9. With `DEFAULT_TTL_S` set (86400 in our deployment), a conversation of a client that does not send the field and is
   not used for that long is re-read at its next turn (before, it stayed until the capacity cap or a restart).
10. The in-process quality probe (`--phase both`) needs the unauthenticated `/reset_prefix_cache` endpoint
    (`GLM53_EXPOSE_CACHE_RESET=1`). Our production deployment closed it with the finish release; we use the restart
    variant (`--phase prepare` / `--phase restart`, `--runs 4`) instead.

## B8 after the fix (finish release)

Found by the independent review (`docs/review-findings.md`): a restore started for a request that is then not admitted
pins blocks (MLA 5 s, states until admission) and, more importantly, evicts other requests' GPU cache when it takes
fresh blocks. In the rig, seed 4 had 52 restores and 9 admissions; on average 20.2 restores started and 8.8 admitted per
seed (`results/fix/aggregate_m_on.txt`).

The finish release (`docs/design.md` 2.9) does not start a restore in a step in which the decode-floor policy skips the
request, and takes a restore's fresh blocks from NVMe-backed cached blocks first. Restores started / admitted per seed
went from 20.2 / 8.8 to 9.8 / 9.4 (normal), 21.2 / 4.8 to 2.6 / 2.6 (TTL mix) and 448.2 / 7.2 to 11.6 / 11.0 (overloaded)
(`results/finish/aggregate.txt`). What remains:

* **Overload.** The `ttl_s=0` conversation of the overloaded TTL mix is still +30.7% vs OFF (+16.5% per admission,
  collapses 10 -> 14). Eviction attribution (seed 1, `results/finish/bf_mxo_*_s1.json`): its blocks were evicted 200
  times with OFF and 345 times with the fix; 116 of those by restores (when no NVMe-backed block is free, the rest is
  taken in plain LRU order) and 71 (OFF: 22) by its own allocations. With the connector the rig admits 29% more DSH turns
  and 23% more one-shot requests in the same 4 hours (OFF is stuck in large re-reads), so there is more competition for
  the GPU cache. Removing this would need NVMe-backed blocks to be evicted first in **every** allocation
  (`allocate_slots`), i.e. a change of `BlockPool`'s eviction order, at the cost that the client's conversations would
  then be read back from NVMe at every turn. Not done.
* **The rig measures skip mode; production runs mode 0 (off).** The B8 amplification came from skip mode's "no prefill
  while another request decodes". In production the gate never fires (behaviour unchanged); restores that are not
  followed by an admission are visible only through the stats line (`restores_started` versus `restores_admitted`). The
  rig cannot run mode 0 (`docs/design.md` 2.9).
* **One-shot requests.** The earlier "+23% re-read for one-shot requests" was a misreading: with the connector more
  one-shot requests were admitted in the 4 hours (87 -> 109 over the 5 seeds); per admitted request the re-read was
  61,230 (OFF) -> 60,216 (before the fix) -> 60,196 (after the fix, -1.7%).

## Resolved: the rig's `hit_hash_violations` anomaly

In some async + CRC-injection 0.5 h runs of the prune-release matrix the rig's `hit_hash_violations` counter was 1
(1 of 9 runs with K=2, 1 of 3 with K=0; detail record `results/prune/acv_k0_r1_s2.viol.json`). Cause, found in the
finish release: **a bug in the rig's record, not in the connector.** The rig checked an admission's hit blocks before
updating the recorded hit with that admission's value. For a request that was preempted and re-admitted it therefore
compared the new block list against the hit of the previous admission (456,192 tokens = 99 blocks, while the
re-admission hit 442,368 tokens and block 97 was a fresh block for the first chunk, without a key yet). Preemptions
occurred only in asynchronous runs, and the rig's `preemptions` counter had never been incremented, so this was not
visible. The rig now updates the hit first and counts preemptions, re-admissions and re-admissions whose hit shrank.
Re-running the same conditions (async + CRC, seeds 2/3/4, 3 repetitions each) and the whole fault suite gave 0
violations and 4 re-admissions, none with a smaller hit (`results/finish/aggregate.txt`, table (6)). The content check
over all blocks had always found 0 mismatches.

## Cosmetic log bug (fixed; in production since the finish release)

The rank start-up log line `rank %d: dir=%s wiped=%d ...` formatted `wiped` with `%d`. With `PERSIST=1` the value is
the string `'deferred'` (the wipe is decided later by the scheduler), so Python's logging reported a formatting error
("--- Logging error ---") and the line was dropped. Nothing else was affected. The final version in production
(connector sha256 `1762e23f...`) kept the bug until the finish release, because changing the connector's bytes then
changed the restart signature and would have wiped the spill directory. Since the finish release the identity covers
the connector's syntax tree without docstrings instead of its bytes (`docs/design.md` 2.8.1).

## Other notes

* The capacity defaults in the code (`GIB`, `MIN_FREE_GIB`) are not those of our deployment; size them for your disk,
  and check whether the spill directory shares a file system with the root file system.
* Drafter acceptance after a restore was measured only with the quality probe (81k-token conversation, 4 runs per
  probe, in-process and across a restart; `docs/measurements.md` 8.2 and 8.6), not on real traffic. The cost of the
  executor waiting for every rank's output on every step was below the measurement noise in our idle-window speed A/B
  (`docs/measurements.md` 8.3); a comparison on real multi-request traffic is still missing.
