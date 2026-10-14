# Operations

How we deployed and operated the connector on a two-node TP=2 setup (`<head-host>` = rank 0, `<worker-host>` = rank 1)
running the GLM-5.3-Flash EXL3 launcher kit (`<kit>`, containers `glm53-exl3-head` / `glm53-exl3-worker`). Paths are
placeholders: `$SPILL_DIR` is the spill directory, `<kit>/launcher/start.sh` the kit's launcher script. Measured
numbers referred to here are in docs/measurements.md.

The connector is **off by default**. Nothing is patched and no `--kv-transfer-config` is passed unless
`GLM53_NVME_SPILL=1`.

## 1. Enabling

1. **Install the module through the bundle hook.** Put `code/glm53_nvme_spill_connector.py` and
   `code/patch_nvme_spill.py` into the kit's bundle overlay directory (the directory the launcher's bundle hook runs
   patches from; it must reach both containers: in our kit the head bind-mounts it and `start.sh` copies it to the worker
   on every start). Use `code/patch_tf_bundle.py` as the bundle hook, or add its one registration line
   `("patch_nvme_spill.py", "GLM53_NVME_SPILL")` to yours.
   `patch_nvme_spill.py` patches three places, and checks all three before writing any of them (each anchor exactly once;
   marker + exact patched text = no-op; anything else = `SystemExit`, nothing half-applied):
   - installs the connector module under `vllm/distributed/kv_transfer/kv_connector/v1/`;
   - two lines in `vllm/v1/core/sched/scheduler.py` that hand the `KVCacheManager` to the connector;
   - `vllm/config/vllm.py` `_verify_kv_transfer_compat`: exempts only `Glm53NvmeSpillConnector` from the "KV connector +
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → ValueError" rule (the connector registers no memory with a NIC
     or another process). Both A/B arms keep the kit's allocator setting.
2. **Patch the launcher.** `python3 code/apply_start_sh.py <kit>/launcher/start.sh <kit>/launcher/start.sh`. It adds
   validation, forwards `GLM53_NVME_SPILL` and every knob of the connector's `KNOBS` tuple to both ranks' `docker run -e`
   (31 names in the final version), bind-mounts `$SPILL_DIR` rw at the same path in both containers, and adds
   `--kv-transfer-config` naming `Glm53NvmeSpillConnector` to both inner scripts. It is idempotent ("already applied",
   no write), refuses any other input carrying its marker or missing an anchor (ABORT, nothing written), checks
   `bash -n`, and verifies that replaying the edits backwards gives the input. The diff for the kit we used is
   `code/start.sh.nvmespill.patch`. If you regenerate the kit, apply the tool again as the last step.
3. **Create `$SPILL_DIR` on both nodes at the same absolute path.** First check whether it is a separate filesystem:
   `findmnt -T $SPILL_DIR` and `df -h $SPILL_DIR` on both nodes. If it is on the root filesystem (shared with logs,
   Docker and the model cache), size `GIB + MIN_FREE_GIB` to at most half of the free space (our install script required
   `2×(GIB+MIN_FREE_GIB) ≤ free` on both nodes). Check that O_DIRECT works there (for example
   `dd if=/dev/zero of=$SPILL_DIR/.probe bs=1M count=4 oflag=direct`). Files under `$SPILL_DIR/glm53-nvme-spill/` are
   written by the containers and owned by root.
4. **Set the knobs in the kit's `.env`** (same values for both ranks; empty = default):
   ```
   GLM53_NVME_SPILL=1
   GLM53_NVME_SPILL_DIR=$SPILL_DIR
   GLM53_NVME_SPILL_GIB=200          # per rank; fit to the free space and to whether the FS is shared
   GLM53_NVME_SPILL_MIN_FREE_GIB=150 # never write below this much free space
   # everything else at the defaults of section 3
   ```
5. **Restart both ranks** (wait until no request is in flight). The first start after adding `--kv-transfer-config`
   rebuilds the torch.compile cache because the config hash changes (in our case the start still took about 5 minutes).
6. **Check the start-up logs** (both ranks unless noted):
   - `[glm53-tf-bundle] patch_nvme_spill.py: applied (GLM53_NVME_SPILL=1)`
   - `[glm53-nvme-spill] scheduler.py patched; config/vllm.py patched; module installed at ... (sha256 <12 hex>)`;
     the hash should match the connector file you deployed.
   - **no** `ValueError` mentioning `incompatible with PYTORCH_CUDA_ALLOC_CONF`.
   - `[glm53-nvme-spill] rank 0: dir=... group_bytes=[27624960, 1672704, 21233664, 21233664, 18874368, 18874368, 11796480]`
     (and `rank 1:` on the worker). The seven entries are the per-block bytes of the seven KV groups for this model; the
     1,672,704-byte group is the per-request kpool tail, which is never saved. (The final version loses this line to a
     logging bug when `PERSIST=1`; the published version prints it. See docs/limitations.md.)
   - `[glm53-nvme-spill] rank R: start-up scan: N block files (X GiB), P partial removed, T s`
   - head: `[glm53-nvme-spill] scheduler: dir=... world=2 keep_per_chain=2 persist=1 hash_seed_from=connector
     ttl_save_policy=all ttl_min_tokens=0 version=...`
   - head: `[glm53-nvme-spill] persist: index <path> (...)` and then a decision line, either
     `persist: adopted N resume points (...)` or `persist: wipe every rank (<reason>)`.
   - `Creating v1 connector with name: Glm53NvmeSpillConnector`, and a KV capacity log identical to OFF (582 usable
     blocks in our setup).
   - Once the engine runs steps: the first `[glm53-nvme-spill] stats breaker=closed ...` on the head and
     `rank 0 io:` / `rank 1 io:` on each rank, every `LOG_EVERY_S` (60 s). These periodic lines come from engine steps:
     **an idle engine prints none**, which is not a fault.
7. Side effects to know: with kv cache events enabled, restored blocks emit no `BlockStored` event; with a connector the
   executor waits for every rank's output each step (compare pure decode tok/s; we measured no difference,
   docs/measurements.md section 8.3); `kv_role=kv_both` enables vLLM's `defer_block_free` for aborted requests.

## 2. Rollback

Remove `GLM53_NVME_SPILL` from `.env` (or set it to 0) and restart both ranks once nothing is in flight. Neither the
patch nor `--kv-transfer-config` is applied any more. Confirm that no `[glm53-nvme-spill]` line appears. After the
containers have stopped using it, delete the files: `rm -rf $SPILL_DIR/glm53-nvme-spill` on both nodes (root-owned; this
also removes `index.json`).

Partial rollbacks at the next restart: `GLM53_NVME_SPILL_KEEP_PER_CHAIN=0` (no pruning, LRU only),
`GLM53_NVME_SPILL_PERSIST=0` (wipe at every start, no index, no `PYTHONHASHSEED`). There is no way to change either
without a restart (both are read at start-up).

If you run a watchdog that restarts a dead engine with the current configuration, roll back by hand whenever the engine
dies while the connector is ON, whatever the cause.

## 3. Knobs

All are `GLM53_NVME_SPILL_<name>`, forwarded to both ranks by the patched `start.sh`. Empty = default.

| Knob | Default | Meaning |
|---|---|---|
| `DIR` | (required) | spill directory (same absolute path on both nodes, mounted rw) |
| `GIB` | 64 | cap on the file total per rank (evicts whole RPs, see the eviction order in docs/design.md) |
| `MIN_FREE_GIB` | 50 | never write if the write would leave less free space (the save job fails) |
| `MIN_TOKENS` / `MIN_GAIN` | 65536 / 16384 | minimum RP length that is written / restore only when the RP is this much deeper than the GPU hit |
| `HIGHWATER` | 0 | below this pool usage, remember RPs instead of writing them (0 = always write) |
| `SAVE_POLICY` | repeat | repeat = write only from a conversation's second turn on / all |
| `MAX_WRITE_GIB_PER_H` | 0 | cap on written GiB per hour (0 = unlimited) |
| `PIN_TIMEOUT_S` | 5 | how long MLA blocks of a restored but not yet admitted request stay pinned |
| `MAX_STATE_PINS` | 64 | total mamba/DFlash state blocks kept pinned after that until admission |
| `RESERVE_BLOCKS` / `SPACE_WAIT_S` | 8 / 20 | free blocks kept back for restores / max seconds to wait for space |
| `MAX_RESTORE_FAILURES` / `MAX_RESTORES_PER_REQUEST` | 2 / 16 | per-request limits on restore failures / restores |
| `SAVE_DEADLINE_S` | 120 | a save job not answered by all ranks in time: pins released, breaker opens |
| `LOAD_DEADLINE_S` | 30 | a load job not answered in time: the request falls back to normal prefill, its new blocks are quarantined until the rank answers, breaker opens |
| `BREAKER_FAILS` | 6 | consecutive failed jobs that open the breaker |
| `MAX_PINNED_BLOCKS` | 0 | cap on save-job pins held only by the connector; above it no new save starts (0 = half the pool, 291 in our setup) |
| `LOG` / `LOG_EVERY_S` | 1 / 60 | per-save/restore log lines / interval of the periodic stats lines (0 = none) |
| `STAGE_MIB` / `DIRECT` | 32 / 1 | staging buffer size / O_DIRECT |
| `KEEP_PER_CHAIN` | 2 | when a new RP is ok on all ranks, keep the newest K RPs of the same conversation chain (including the new one) and delete older ones. 0 = no pruning |
| `PERSIST` | 1 | reuse files across restarts (index + signature + block-hash seed). 0 = wipe at start |
| `HASH_SEED` | 5353 | with PERSIST=1, the `PYTHONHASHSEED` set in the scheduler process only (an existing value is kept). Integer 0-4294967295 |
| `INDEX_EVERY_S` | 5 | minimum interval of index writes (only when something changed; ttl 0, breaker and start-up decisions are written at once) |
| `ADOPT_DEADLINE_S` | 60 | at start-up, stop waiting for the ranks' reports after this long and wipe. While an index exists, waiting requests also wait |
| `TTL_SWEEP_S` | 30 | interval for deleting expired RPs |
| `TTL_SAVE_POLICY` | all | requests with an explicit `ttl_s > 0` also write their first turn (all) / previous behavior (repeat) |
| `TTL_MIN_TOKENS` | 0 | `MIN_TOKENS` for requests with explicit `ttl_s > 0` (0 = same as `MIN_TOKENS`) |

The pinned staging buffers use 32 MiB × 2 (write and read) per rank. The KV pool is unchanged.

**Why these deadlines** (worst-case estimate): restoring a whole 780k-token conversation is 4.7 GB ÷ 2.6 GB/s ≈ 1.8 s,
×2 for GIL contention, three conversations at once ≈ 11 s; 30 s leaves almost 3× margin. Saving the first three
conversations is ≈ 14 GB; even if the NVMe drops to 1 GB/s after its SLC cache is exhausted that is ≈ 45 s; 120 s leaves
2.6× margin. An expiry opens the breaker at once (latched until restart, behaves like OFF), so the design errs towards
stopping when the disk is merely slow. Deadlines are measured from when the job is sent, including time queued on the
rank.

**Capacity:** see docs/measurements.md section 9 (first save ≈ length × 6.0 KB + 92 MB per rank; ≈ 0.1-0.16 GB per later
turn; ~10 GB/h/rank written, an estimate; totals are 2× for TP=2).

## 4. Quality checks

1. **Output quality after a restore** with `code/spill_quality_probe.py`: run on the ON arm, while idle, with
   `GLM53_EXPOSE_CACHE_RESET=1`, on the head host with read access to the head container's logs (the probe reads the
   API key from the kit's `.env` itself and does not print it). It sends an 80k-token conversation in two turns to
   create an RP, clears the GPU prefix cache with `/reset_prefix_cache`, and continues from NVMe with greedy decoding
   (R). Controls: the same request again (H = GPU hit) and full prefill twice with a new salt (F1, F2). Verdicts (all
   required; **no proof of restore = FAIL**):
   - restore: production does not return `prompt_tokens_details` (cached = None), so the head log in R's time window must
     contain `[glm53-nvme-spill] restore ready req=... rp=N` with N ≥ 0.9 × len(P+Q1), **and**
     R.wall − H.wall ≤ 0.25 × (F1.wall − H.wall). Unreadable logs or no such line = FAIL (so `LOG=0` always fails).
   - kl: KL(R‖F1) ≤ 1.5 × max(KL(H‖F1), KL(F2‖F1)).
   - acceptance: drafter acceptance R ≥ H − 0.03.

   Pass = the last line reads `SPILLQ verdict restore=PASS (...) kl=PASS acceptance=PASS` (exit 0). A stop at
   `/reset_prefix_cache -> HTTP ...` means the reset route is not exposed or the engine was not idle; wait and rerun
   (`--settle 30` helps). The probe clears the whole GPU prefix cache once, so every live conversation re-reads on its
   next turn: run it when traffic is low.

   In production the single-run kl/acceptance verdicts also failed for the GPU-hit control itself; we judge on the mean
   of at least 4 runs (docs/measurements.md section 8.2).
2. Also run the decode-vs-prefill probe `code/conc_quality_probe.py` on both arms.
3. **Drafter acceptance**: compare the per-position acceptance of the 10 s statistics ON vs OFF; the accepted length of
   requests right after a restore should match normal hits.
4. **Re-read volume and stalls**: compare before/after the switch time (1) the volume of ≥ 100k-token ingest windows,
   (2) the share of generation-stall windows while ≥ 2 sequences run, (3) pure decode tok/s by concurrency. (Our script
   for this is not shipped.) Save the OFF period's container logs before the first ON restart; recreating the containers
   discards them.
5. **Restart variant** (reuse across restarts): in addition to the in-process probe,
   ```
   python3 code/spill_quality_probe.py --phase prepare --state <state-file>
   # restart both ranks; run the boot check with EXPECT_ADOPT=1 (section 6); wait for idle
   python3 code/spill_quality_probe.py --phase restart --state <state-file> --out <result.json>
   ```
   Same pass line. R cannot come from the new engine's GPU, only from the files the previous engine wrote. Keep prepare →
   restart within an hour (the probe's requests carry no TTL field and survive until LRU eviction).

**`GLM53_EXPOSE_CACHE_RESET=1`** (needed only by the probe) exposes `/reset_prefix_cache` on the API port **without the
API key**: anyone who can reach the port can drop every conversation's GPU prefix cache and force re-reads. Turn it off
(remove it from `.env`, restart) once probing is done.

## 5. A/B procedure and abort criteria

**Lines to read** (every `LOG_EVERY_S` = 60 s while the engine is stepping):

- Head (scheduler):
  `[glm53-nvme-spill] stats breaker=closed pinned=P save=a save_excl=x restore=b quarantined=q peak=K excl_peak=X cap=C
  inflight_save=n oldest_save_s=x inflight_load=m oldest_load_s=y worker_exc=e jobs_failed=f save_timeouts=st
  load_timeouts=lt written_gib_1h=w disk_gib=d rps=r restores_ok=.. restores_failed=.. restores_fallback=..
  restore_short=.. saves_ok=.. saves_failed=.. rp_pruned=.. pruned_gib=.. rp_evicted=.. keep=K skipped_cap=..
  expired=.. epoch_dropped=.. ttl0_skipped=.. ttl_rps=none/1h/1d/3d/exp ttl_gib=.. ttl_classes=.. persist=..
  adopted=.. adopted_gib=.. index_age_s=..`
  New fields are only ever appended; the order of existing fields does not change, so parse as key=value. When the
  breaker opens: `breaker=OPEN ... reason='...'` at WARNING level.
- Each rank (head = rank 0, worker = rank 1):
  `[glm53-nvme-spill] rank R io: <w=save:<job>/<s>s r=load:<job>/<s>s | idle> queued w=.. r=.. written=.. (GiB)
  write_failed=.. loaded=.. load_failed=.. delete_failed=.. thread_exc=.. threads_alive=2/2`
- One-off lines: `BREAKER OPEN: <reason>`, `save job N unanswered after ...`, `load job N unanswered after ...`,
  `unexpected exception in ...`, `restore failed`, `restore short`, and with pruning
  `prune rp n=<old length> (superseded by n=<new length>) freed=... disk=... rps=...`.

**Collecting** (once a minute):
```
docker logs --since 70s glm53-exl3-head 2>&1 | grep -E '\[glm53-nvme-spill\] (stats|rank 0 io|BREAKER|.*unanswered|unexpected|restore (failed|short))'
ssh <worker-host> "docker logs --since 70s glm53-exl3-worker 2>&1 | grep -E '\[glm53-nvme-spill\] (rank 1 io|unexpected)'"
```

**Roll back at once (section 2) if any of these holds:**

| # | Condition (which line, which field) | Why |
|---|---|---|
| A1 | `breaker=OPEN` / a `BREAKER OPEN` line (also an `unanswered` job line) | a rank stopped answering, consecutive failures or an I/O thread exception. The feature has already stopped, but the cause needs investigating |
| A2 | `worker_exc>0`; on any rank `thread_exc>0` or `threads_alive` below 2/2; an `unexpected exception` line | unexpected exception (a bug) |
| A3 | `quarantined>0` | a load expired (quarantined blocks do not come back until restart) |
| A4 | `restore_short>0` / a `restore short` line | an invariant that is 0 by design broke (restored, but the normal lookup does not reach it) |
| A5 | `oldest_save_s>60` or `oldest_load_s>15` on 2 consecutive lines | half the deadline (120 / 30 s): the disk is clogging |
| A6 | `save_excl` > 0.8 × `cap` on 3 consecutive lines (3 minutes) | connector-only save pins are piling up (rig maximum in docs/measurements.md section 5.1) |
| A7 | quality probe FAIL (section 4) | output correctness (see the note on averaging in section 4) |
| A8 | pure decode tok/s (10 s statistics, at times without I/O) more than 5% below OFF, or more preemptions than OFF | performance regression |
| A9 | free space of the spill FS < `MIN_FREE_GIB` + 10 GiB on either node, or `written_gib_1h` > 40 (4× the ~10 estimate) | capacity / drive wear |

**Watch, and investigate when a threshold is crossed:** `restores_failed` +3 within 10 minutes (broken files; requests
fall back to prefill), growing `saves_failed` (free-space floor, I/O), `restores_fallback`, the trend of `rps` and
`disk_gib`, the count and duration of `restore start/ready` lines, `persist=disabled` (block hashes not reproducible:
only reuse across restarts is off for this run). With pruning: `rp_pruned` rising while `disk_gib` does not fall
(reference counts drifting) or a burst of `restores_failed` (a deleted file being read) are unexpected; roll back the
pruning (`KEEP_PER_CHAIN=0`) at the next restart. Missing stats/io lines only matter while requests are running or
waiting.

## 6. Restart reuse checks

**First restart after installing a persisting version over a non-persisting one.** There is no index yet, so expect
`persist: wipe every rank (no usable index (no index file))` and one round of normal re-reads, as before. Check:

- the 7 TTL/persist knobs (`PERSIST HASH_SEED INDEX_EVERY_S ADOPT_DEADLINE_S TTL_SWEEP_S TTL_SAVE_POLICY TTL_MIN_TOKENS`)
  are present in both containers (empty = default);
- head scheduler line `persist=1 hash_seed_from=...` and the connector version;
- `persist: index ...` line; a `start-up scan:` line on both ranks and no `start-up scan: ... ERROR`;
- the decision line, and that it does **not** say `reuse across restarts disabled` (block hashes reproducible);
- the stats line carries `expired / epoch_dropped / ttl0_skipped / ttl_rps / ttl_gib / persist / adopted`.

**Second restart and later.** The files the new code wrote must now be adopted: the decision line must read
`persist: adopted N resume points (...)`. Our boot check enforces this with `EXPECT_ADOPT=1`. If it reads
`persist: wipe every rank (... signature differs: <part>)`, the reason names what differed from the previous run: `site`
(the `.py` contents / `.so` sizes under site-packages), `env`, `model`, `config`, `layout`, `hash`, or `rank N: ...`. An
unintended difference (something that changes on every start, for example a start-up patch that writes different bytes
each time) means every restart wipes, which is safe (same as no reuse) but loses the benefit. Then run the restart
variant of the quality probe (section 4, item 5).

During start-up each rank scans its files in a background thread while the model warms up; the scheduler decides before
any save or restore. While an index exists, waiting requests can be held up to `ADOPT_DEADLINE_S` (normally 1-2 s; the
site-packages digest takes ~1 s warm, ~4 s cold).

## 7. Operator scripts we used (not shipped)

Our production change was driven by a set of shell scripts specific to our hosts. They are not part of this repository,
but the patterns were worth having; this is what each did, as guidance for writing your own.

**Common pattern.** A driver runs on an operator machine: it refuses to run if another experiment is marked as running
(a marker file, or a benchmark A/B process found in the process list), verifies the payload directory against its
`SHA256SUMS`, rsyncs the script directory to `<head-host>`, and runs the same script there with `--here` over ssh. The
`--here` part performs **all read-only checks first** and writes nothing unless all pass (`ABORT: <reason>`, exit 2). Every
target file must have one of exactly two sha256 values, "before" or "after" this step; anything else, or a mixed state
(some files before, some after), aborts. Writes go to a temporary file next to the target, get the target's mode
(`chmod --reference`), and are renamed into place (a running container's single-file bind mount keeps the old inode;
nothing is read before the next restart). After writing, the script re-verifies every sha256 and runs `bash -n` on
`start.sh`. A second run is a no-op ("already installed/upgraded ... nothing written"). **No script ever restarts the
engine**; restarts were a separate, deliberate step (our restart helper waited until no request was in flight, up to 900 s,
and otherwise refused).

| Script | What it did |
|---|---|
| `install.sh` | First install. Pinned the expected sha256 of the original `start.sh` (a8e81a10…) and bundle hook (13c6720d…) and of the four payload files ("reviewed bytes"). Checks: payload bytes; target files present; `.env` has no knob line outside our marked block, the block is absent or exactly as expected, and `.env` ends with a newline; `SPILL_DIR` is absolute with only safe characters; free space on both nodes ≥ 2×(GIB+MIN_FREE_GIB). Created `$SPILL_DIR` on both nodes (on the worker, whose parent directory was not writable by the operator user, through a one-off root container from the already-present image with `--network none --pull never`, then chown); probed O_DIRECT with `dd oflag=direct` on both. Backup directory (mode 700) with `start.sh`, the bundle hook, `.env` (mode 600) and a `STATE` file listing every original/new sha256; a `LATEST-BACKUP` pointer file. Wrote the 2 new overlay files, the bundle hook, `start.sh`, and `.env` = old `.env` + a marked block (built under `umask 077`). Verification printed only its own `.env` lines and the values the launcher would see after sourcing `.env` in a subshell (which catches a later duplicate assignment). `CACHE_RESET=0` left out `GLM53_EXPOSE_CACHE_RESET=1`. Environment overrides (target dir, backup root, worker shell command, ...) existed only for dry runs. |
| `upgrade.sh` | Prune release on top of the install: replaced only the connector (e8c218eb… → d030f132…) and `start.sh` (bad54d9a… → ee3208b8…, two more forwarded lines); required the bundle hook and `patch_nvme_spill.py` to be unchanged; verified the new knob is forwarded exactly twice (worker `serve_env` loop + head `-e`) and that the connector's `KNOBS` end with `KEEP_PER_CHAIN`. Own backup and `LATEST-BACKUP` pointer. `.env` untouched (empty knob = default). |
| `upgrade2.sh` | Final version on top of `upgrade.sh`: connector d030f132… → 1762e23f…, `start.sh` ee3208b8… → 7b8fcfe1… (7 more forwarded knobs). Verified every one of the 8 newer knobs is forwarded exactly twice and that the connector carries the expected version string. Refused the pre-`upgrade.sh` bytes. Also verified all earlier payloads. Guarded against a script whose sha constants were still placeholders. |
| `revert.sh` | Full rollback to the original launcher. Planned every item before writing: each file → restore / keep / remove; it accepted the "installed" bytes of `install.sh`, `upgrade.sh` and `upgrade2.sh`, so one revert works from any stage; anything else aborted. `.env`: exactly the installed `.env` → restored byte for byte from the backup; edited since → refused unless `FORCE_ENV_STRIP=1`, which removes only the marked block and keeps the other edits. Then verified sha256s, absence of the overlay files and of any knob line. A separate `--purge-spill` mode (after the OFF restart) refused while any running container mounted `$SPILL_DIR`, then removed `$SPILL_DIR/glm53-nvme-spill` on both nodes through a one-off root container (the files are root-owned). |
| `upgrade-revert.sh`, `upgrade2-revert.sh` | Undo one stage only (back to the previous stage's `start.sh` and connector) from that stage's backup, with the same plan-then-write logic. |
| `bootcheck.sh` | Read-only start-up check, modes `on` / `off`. `on`: `/health` 200; worker logs readable; `GLM53_NVME_SPILL=1` and identical knob values in both containers; spill dir mounted rw at the same path in both; the bundle and patch lines on both ranks; no allocator/connector `ValueError`; the `group_bytes` line (or, for the final version's known logging bug, the `start-up scan` line instead); scheduler `world=2`; the installed module's sha256 prefix in each rank's log equals the deployed file; knob presence and `keep_per_chain`; connector named in the engine args; KV capacity equal to OFF; no `BREAKER OPEN` / `unexpected exception` / `unanswered`; then it waits up to 180 s for the first stats line, both io lines and the persist decision; `breaker=closed`; the newer stats fields; the persist checks of section 6, and with `EXPECT_ADOPT=1` an `adopted` decision. `off`: no tagged line, connector not in the engine args, no non-empty knob in either container, `GLM53_EXPOSE_CACHE_RESET=0`, same KV capacity. **Pitfall we hit:** with `set -o pipefail`, `echo "$LOG" \| grep -q PATTERN` fails whenever grep matches early: grep exits, `echo` gets SIGPIPE (status 141), and the pipeline reports failure, giving a false MISS on a large log. The script runs without `pipefail` (alternatives: `grep -q PATTERN <<<"$LOG"`, or a temporary file). |
| `monitor.sh` | Long-running monitor on the head host, every 60 s: fetched new tagged lines from both containers (`docker logs --timestamps --since`, de-duplicated) and the free space of the spill FS on both nodes; appended one summary line to `monitor.log` and raw lines to `monitor-raw.log`; flagged A1-A6 and A9 automatically as `ALERT A<n> ...` lines (also appended to an `ALERT` file). A7 and A8 were left to the operator. WATCH lines: no stats/io line for > 3 min **only while** `/metrics` showed running or waiting requests (an idle engine logs nothing; the first version raised a false warning at idle), `/health` not 200, `restores_failed` +3 in 10 min, `persist=disabled`, unreadable free space. It never rolled anything back. It read only its own knob lines from `.env`. A `--replay <head log> <worker log>` mode judged saved logs offline: real logs gave 0 alerts, crafted abnormal lines fired each of A1-A6 and A9, and logs with newer stats fields gave the same judgments. |
| `make_payload_ttl.sh` | Built the `upgrade2.sh` payload from one git commit: extracted the connector, `apply_start_sh.py` and the probes with `git show`, applied `apply_start_sh.py` to the backed-up original `start.sh` (after checking its sha256 prefix), wrote a `SOURCE` file and `SHA256SUMS`, and a diff against the previous stage's `start.sh`. It then filled the new sha256 values into the placeholders of `upgrade2.sh` and `revert.sh`; a second run with the same commit was a no-op, a different commit after finalization aborted. Re-running produced identical bytes. |

**Dry runs before touching production** (also not shipped): each script was tested against copies of the production
launcher tree taken read-only, with every `.env` value masked (same keys, same order). install → no-op → revert left all
258 files with identical sha256 and mode; 9 refusal cases (wrong `start.sh`/bundle hash, knob already in `.env` (2
forms), a foreign connector file, the experiment marker, too little space, `.env` without a trailing newline, missing
worker directory) wrote nothing. A launch test ran `start.sh`'s validation, inner-script and launch functions with fake
`docker` / `ssh` / `scp` and recorded every call: with all knobs set, every name reached both ranks' `docker run -e` with
the `.env` value, the spill dir was mounted rw in both, and both inner scripts got `--kv-transfer-config`; with the
patch installed but the switch off, the only differences from the original were 23 empty `-e NAME=` entries and 3 inert
`if` lines. The upgrade tests (27/27 and 32/32) covered upgrade → no-op → stage revert → identical bytes, 8 refusal
cases each, and full chains install → upgrade → upgrade2 → `revert.sh` returning to the pre-install bytes. The new
`bootcheck.sh` was also run read-only against the live system before deployment and reported ALL OK.
