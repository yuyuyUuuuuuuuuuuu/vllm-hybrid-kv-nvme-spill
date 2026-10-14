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
5. **Pins while waiting, and wasted restores (review B8, not fixed).** See "Unfixed: B8" below.
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

## Known risks of TTL and reuse across restarts (final version)

1. **If any part of the site-packages digest changes from start to start, every start wipes everything** (the safe
   side, the same as without reuse). All production overlay patches should be deterministic (string replacement and
   copies), but this cannot be confirmed before the second real restart. A start-up check expecting adoption after the
   second restart shows it; on a mismatch the log names the part (site/env/model/...). The env digest was not computed
   from the real production container environment (it could not be read from the test machine), so it too is only
   confirmed at the second restart.
2. **A conversation with `ttl_s=0` (do not keep) can get fewer GPU hits than with the connector off** (+113% re-reads in
   the rig, see "Unfixed: B8" below). It is the same side effect of B8 that one-off requests already see (+23%).
3. The index is at most `INDEX_EVERY_S` (5 s) old. RPs saved in the last 5 s before a crash are deleted as orphans at
   the next start (the rig's three crash runs matched the clean-shutdown runs).
4. `PYTHONHASHSEED` is set in the environment of the scheduler (EngineCore) process. Child processes that process starts
   afterwards get a fixed `str` hash. The `str` hash randomization of Python itself, the API server and the workers is
   unchanged (measured). (EngineCore currently starts no child processes after start-up.)
5. At start-up, while an index exists, waiting requests can be held for up to `ADOPT_DEADLINE_S` (60 s) (normally 1-2 s:
   the rank scan finishes during model warm-up; the scheduler's site digest takes about 1 s warm and about 4 s cold).
6. Adoption on the real two-node setup (rank 1's report arriving from the worker node through the executor) is
   inferred from the rig's two simulated ranks and from production's existing worker-meta aggregation working (save
   and restore answers). Check the `persist:` decision line and both ranks' `start-up scan:` lines after the first
   restart.
7. The benefit of writing the first turn (`TTL_SAVE_POLICY=all`) does not show up in the rig's load. Its cost is +2%
   bytes written.

## Unfixed: B8 (wasted restores before admission)

Found by the independent review (`docs/review-findings.md`) and left unfixed.

* A request whose resume point was restored but that is not yet admitted keeps its MLA blocks pinned for 5 s and its
  state blocks pinned until admission (at most 64 blocks in total). A request that waits long behind the decode floor
  loses its MLA blocks and re-reads what is missing at admission.
* In the rig, seed 4 had 52 restores and 9 admissions; on average 20.2 restores started and 8.8 admitted per seed
  (`results/fix/aggregate_m_on.txt`).
* Correctness is unaffected, but in production it costs extra NVMe reads, and the fresh blocks allocated by
  `get_new_blocks` evict other conversations' GPU cache.
* We did not find a cheap fix: admission is decided by the scheduler after the restore, so it is not known in advance;
  a longer `PIN_TIMEOUT_S` would hold other requests' blocks longer. The likely fix is not to restore ahead of time for
  requests that are not expected to be admitted soon.
* **Side effect on requests that are not saved.** Because restores for other conversations allocate and pin blocks
  before admission, requests that rely only on the GPU cache lose their prefix earlier:
  * one-off requests (never saved with `SAVE_POLICY=repeat`): re-read 1,065,397 -> 1,312,717 tokens, **+23%**
    (4 h x 5 seeds of the fix-release re-run; computed by us from `results/fix/m_off_s*.json` and
    `results/fix/m_on_s*.json`);
  * a conversation sent with `ttl_s=0` in the mixed-TTL rig (4 conversations, 4 h x 5 seeds): re-read 245,654 (OFF)
    -> 523,321 (ON), **+113%**, collapses 0.2 -> 1.8. In that load there were 21.2 restores, 4.8 admissions and 21.0
    pin expiries per seed (final version; `results/ttl/final/aggregate.txt`, table "TTL mix", runs
    `results/ttl/final/mx2_{off,on,nottl}_s*.json`).
    The other three conversations improved by -38.5%, -22.8% and -41.2%; the total was -24.9%.

## Open rig anomaly: `hit_hash_violations`

In the async + CRC-injection 0.5 h runs of the prune-release matrix, the rig's `hit_hash_violations` counter (a
mismatch between the hit recorded at admission time and the actual admission) was occasionally 1: in 1 of 9 runs with
K=2 and in **1 of 3 runs with K=0**, so it does not depend on pruning.

The detail record (`results/prune/acv_k0_r1_s2.viol.json`): the recorded hit was 456,192 tokens (from the previous
lookup) while the request's computed tokens were 446,976; the check tripped on the block one past the actual hit,
which had no key yet. The content check over all blocks found 0 mismatches. Our reading is that the hit shrank while
the request was waiting (the same pattern as the "lost hit" of our earlier re-read investigation) and the rig's record
does not follow that. We judge it **not a correctness problem, but it is not explained** and remains open.

## Cosmetic log bug (fixed in the published version only)

The rank start-up log line `rank %d: dir=%s wiped=%d ...` formatted `wiped` with `%d`. With `PERSIST=1` the value is
the string `'deferred'` (the wipe is decided later by the scheduler), so Python's logging reported a formatting error
("--- Logging error ---") and the line was dropped. Nothing else is affected.

The published version (`code/glm53_nvme_spill_connector.py`) uses `%s`. The production deployment (connector sha256
`1762e23f...`, the final version) still has the bug **on purpose**: the connector file is part of the site-packages
digest in the restart signature (`docs/design.md` section 2.8), so changing its bytes would change the signature and
wipe the spill directory at the next restart. The bug is harmless, so we leave production as it is.

## Other notes

* The capacity defaults in the code (`GIB`, `MIN_FREE_GIB`) are not those of our deployment; size them for your disk,
  and check whether the spill directory shares a file system with the root file system.
* Not measured: real drafter acceptance after a restore, and the cost of the executor waiting for every rank's output
  on every step once a connector is configured (review item 4a). Both can only be measured in a real A/B.
