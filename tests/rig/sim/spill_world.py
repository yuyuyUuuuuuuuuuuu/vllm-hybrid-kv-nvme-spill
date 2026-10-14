# SPDX-License-Identifier: AGPL-3.0-only
"""Two simulated TP worker ranks for the spill connector inside the CPU replay rig.

Each rank owns CPU "GPU KV" tensors with the production tensor structure (11 MLA tensors shared with the mamba and
drafter layers, 11 indexer tensors shared with the kpool tail) but pages scaled down by SCALE, and a real
``SpillWorker`` (the production worker code) writing real files under the spill directory.

Content model (what the forward would have written): a block's bytes in the rows of its group are a pure function of
its prefix-cache key and the rank, ``f(key, rank)``. The engine's normal path gives a block its primary key
(BlockPool._insert_block_hash outside a restore) inside schedule(), BEFORE the step's forward computes it; the rig
therefore records the block and writes ``f`` only when the step's "forward" runs, i.e. in ``step()`` AFTER the workers
received that step's connector metadata (sync mode: after they finished it). A save copied before the forward of the
step that keyed its block holds stale bytes, and the restore check below catches it (the race of review finding B6).
A restore never writes ``f``: its bytes come only from the files. At every admission each hit block of every group is
checked against ``f(block.block_hash, rank)`` on both ranks (after the step's forward), so a restore that put wrong
bytes or the right bytes into the wrong block fails the check.

Modes (env):
  NVMESPILL_ASYNC=1           jobs complete asynchronously: per-rank random I/O latency (NVMESPILL_LAT0/1 s max),
                              results collected in whatever step they arrive (no drain)
  NVMESPILL_ABORT_P=p         async: per step, abort a request whose restore is loading with probability p
  NVMESPILL_FAULT=
    missing | crc             every 3rd restore: rank 1 loses one file / gets one flipped byte
    savefail                  rank 1 fails one block write in 3 (I/O error)
    rofs                      [review B2, exact] rank 1's store turns read-only after its 40th block write:
                              write_block AND store.delete raise EROFS (delete raising out of the method)
    rofs_os                   same, injected at the os level (os.open/unlink/rename in rank 1's dir -> EROFS),
                              so the production error handling runs as it would on a remounted-ro filesystem
    noload                    negative control: rank 1 reports every read ok but writes nothing (verify must fail)
    hang_load | hang_save     rank 1's NVMESPILL_HANG_NTH-th (default 2) read/write blocks NVMESPILL_HANG_S seconds
                              (default 1e9 = until shutdown), then completes normally (a late answer)
    noack_load | noack_save   rank 1 never reports the NVMESPILL_HANG_NTH-th load/save job (its answer is lost)
  NVMESPILL_RACE_PROBE=1      the prefill-done scan saves a request's resume point in the SAME build_meta its last
                              prompt chunk is scheduled (instead of one step later): the blocks are keyed in that
                              schedule() and written by that step's forward -- a connector that ships the copy in the
                              same step's metadata copies them unwritten (verify_fail > 0)
"""
from __future__ import annotations

import errno
import hashlib
import os
import threading
import time

import torch

SCALE = 576          # 2,359,296 / 576 = 4096 B ; 152,064 / 576 = 264 B


def content(key: bytes, rank: int, nbytes: int) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(bytes(key) + bytes([rank])).digest()[:8], "little")
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g)


class _OsProxy:
    """os module stand-in for the spill module: open/unlink/rename under one directory raise EROFS once armed."""

    def __init__(self, real, prefix: str):
        self._real = real
        self._prefix = prefix
        self.armed = False
        self.renames = 0
        self.hits = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _ro(self, path) -> bool:
        return self.armed and str(path).startswith(self._prefix)

    def open(self, path, flags, mode=0o777, *a, **k):
        if self._ro(path) and flags & (self._real.O_WRONLY | self._real.O_RDWR | self._real.O_CREAT):
            self.hits += 1
            raise OSError(errno.EROFS, "injected read-only fs", path)
        return self._real.open(path, flags, mode, *a, **k)

    def unlink(self, path, *a, **k):
        if self._ro(path):
            self.hits += 1
            raise OSError(errno.EROFS, "injected read-only fs (unlink)", path)
        return self._real.unlink(path, *a, **k)

    def rename(self, src, dst, *a, **k):
        if self._ro(dst):
            self.hits += 1
            raise OSError(errno.EROFS, "injected read-only fs (rename)", dst)
        r = self._real.rename(src, dst, *a, **k)
        if str(dst).startswith(self._prefix):
            self.renames += 1
            if self.renames >= 40:
                self.armed = True
        return r


class SpillWorld:
    def __init__(self, scheduler, spill_dir: str, fault: str = ""):
        # the module the engine loaded (installed into the work tree by patch_nvme_spill.py)
        from vllm.distributed.kv_transfer.kv_connector.v1 import glm53_nvme_spill_connector as spill
        import prodlayout
        from vllm.v1.core.kv_cache_utils import get_group_id

        self.spill = spill
        self.get_group_id = get_group_id
        self.scheduler = scheduler
        self.conn = scheduler.connector
        self.sched = self.conn._sched
        kvc_s = scheduler.kv_cache_config
        nb = kvc_s.num_blocks
        specs = [g.kv_cache_spec for g in kvc_s.kv_cache_groups]
        self.kvc = prodlayout.build_kv_cache_config(nb, scale=SCALE, specs=specs)
        self.pages, self.rows = spill.group_rows(self.kvc)
        cfg = spill.SpillConfig.from_env()
        self.os_proxy = None
        if fault == "rofs_os":
            self.os_proxy = _OsProxy(os, cfg.rank_dir(1))
            spill.os = self.os_proxy        # module-global: only paths under rank 1's dir are affected
        self.workers = []
        self.raws = []
        vc = getattr(scheduler, "vllm_config", None)
        kvc_w = self.kvc
        for rank in (0, 1):
            w = spill.SpillWorker(cfg, self.kvc, rank)
            raws = [torch.zeros(kvt.size, dtype=torch.uint8) for kvt in self.kvc.kv_cache_tensors]
            views = {}
            for t, kvt in enumerate(self.kvc.kv_cache_tensors):
                for name in kvt.shared_by:
                    views[name] = raws[t]
            if hasattr(spill, "_identity"):
                # [ttl rig] the rank identity of production (the rank's own scaled layout here); NVMESPILL_RANK1_SALT
                # (restart fault injection) makes rank 1's identity differ, as a different image/patch on the worker node would
                def ident(rank=rank):
                    parts = spill._identity(vc, kvc_w)
                    if rank == 1 and os.environ.get("NVMESPILL_RANK1_SALT"):
                        parts["salt"] = os.environ["NVMESPILL_RANK1_SALT"]
                    return parts
                w.register(views, sig=self.conn.sig, ident_fn=ident)
            else:
                w.register(views, sig=self.conn.sig)
            self.workers.append(w)
            self.raws.append(raws)
        # [ttl rig] production: each rank's start-up scan runs during model warm-up, long before the scheduler's first
        # step; the scheduler's identity thread likewise. Wait for both so the simulated clock does not run meanwhile.
        for w in self.workers:
            w.drain(600)
        th = getattr(self.sched, "_ident_thread", None)
        if th is not None:
            th.join(600)
        self.fault = fault
        # [nvme-spill prune rig] per-rank accounted disk (production file sizes): peak and a 5-min timeline
        self.disk_peak = 0
        self.disk_timeline: list[tuple[int, float, int]] = []    # (sim s, GiB, rps)
        self._next_disk_sample = 0.0
        # which RP each restore used: how many newer RPs of the same chain were alive (0 = the newest)
        self.restore_log: list[dict] = []
        self.in_restore = False
        self.content_writes = 0
        self.verify_blocks = 0
        self.verify_fail: list[tuple] = []
        self.restored_verify = 0
        self.faults_injected = 0
        self.io_bytes = 0
        self.pending_content: list[tuple[int, bytes]] = []     # (block id, key) keyed in this schedule()
        self.pending_checks: list[tuple] = []                   # admissions to verify after this step's forward
        self.file_verify = 0
        self.file_verify_fail: list[tuple] = []
        self.cur_step_hashed: set[int] = set()
        self.same_step = {"plans_in_build_meta": 0, "plans_hit": 0, "blocks_hit": 0, "by_group": {}}
        pool = scheduler.kv_cache_manager.block_pool
        orig_insert = pool._insert_block_hash
        world = self

        def insert(key, block, num_tokens):
            primary = block.block_hash is None
            orig_insert(key, block, num_tokens)
            if primary and not world.in_restore and block.block_hash == key:
                # the forward of this step computes it: written in step(), after the workers took this step's jobs
                world.pending_content.append((block.block_id, key))
                world.cur_step_hashed.add(block.block_id)

        pool._insert_block_hash = insert  # type: ignore[method-assign]

        # same-step probe (review): saves planned inside build_meta whose blocks were keyed in this schedule()
        orig_schedule = scheduler.schedule

        def schedule_wrapped(*a, **k):
            world.cur_step_hashed = set()
            return orig_schedule(*a, **k)

        scheduler.schedule = schedule_wrapped
        orig_build = self.sched.build_meta
        world._in_build = False

        def build_wrapped():
            world._in_build = True
            try:
                return orig_build()
            finally:
                world._in_build = False

        self.sched.build_meta = build_wrapped
        orig_save_plan = self.sched._save_plan

        def save_plan_wrapped(rp_key, n, entries, **kw):
            before = set(world.sched.save_jobs)
            r = orig_save_plan(rp_key, n, entries, **kw)
            if world._in_build:
                world.same_step["plans_in_build_meta"] += 1
                hit = 0
                for j in set(world.sched.save_jobs) - before:
                    items, pins, _rp = world.sched.save_jobs[j]
                    for (g, b, _f) in items:
                        if b.block_id in world.cur_step_hashed:
                            hit += 1
                            world.same_step["by_group"][g] = world.same_step["by_group"].get(g, 0) + 1
                if hit:
                    world.same_step["plans_hit"] += 1
                    world.same_step["blocks_hit"] += hit
            return r

        self.sched._save_plan = save_plan_wrapped
        if os.environ.get("NVMESPILL_RACE_PROBE") == "1":
            # build_connector_meta runs inside schedule() BEFORE _update_after_schedule, so num_computed_tokens is
            # still the pre-step value: the probe reads this step's num_scheduled_tokens to find requests whose LAST
            # prompt chunk is scheduled now (their last chunk's blocks were keyed by this schedule()'s allocate_slots
            # and are written by this step's forward) and saves their resume point right away.
            sched = self.sched
            world.cur_sched_out = None
            orig_bcm = self.conn.build_connector_meta

            def bcm(scheduler_output):
                world.cur_sched_out = scheduler_output
                return orig_bcm(scheduler_output)

            self.conn.build_connector_meta = bcm
            world.race_plans = 0
            world.race_blocks = 0
            world.race_max = int(os.environ.get("NVMESPILL_RACE_MAX", "300"))
            orig_scan = sched._scan_prefill_done
            coord = scheduler.kv_cache_manager.coordinator

            def scan_same_step():
                orig_scan()     # the normal resume-point saves still happen
                # The engine's own RP stops at the deepest boundary with a cached mamba state, and a request's last
                # prompt chunk is usually a sub-block tail, so a natural plan almost never holds a block keyed in the
                # same schedule() (review: 0 of 177). The probe plans, inside build_meta, a standalone copy (never
                # restored) of every block this schedule() keyed -- the hazard class itself -- and the file check in
                # step() compares every saved file with f(key, rank) before this step's forward writes f.
                pool_ = scheduler.kv_cache_manager.block_pool
                if world.race_plans >= world.race_max or not world.cur_step_hashed:
                    return
                blocks = [pool_.blocks[i] for i in sorted(world.cur_step_hashed)]
                blocks = [b for b in blocks if b.block_hash is not None and b.block_id in world.cur_step_hashed]
                if not blocks:
                    return
                entries = [(spill._RPBlock(get_group_id(b.block_hash), (bytes(b.block_hash),),
                                           b.block_hash_num_tokens, spill.block_file_name(bytes(b.block_hash))), b)
                           for b in blocks]
                world.race_plans += 1
                world.race_blocks += len(entries)
                sched._save_plan(b"raceprobe:%d" % world.race_plans, 0, entries)

            sched._scan_prefill_done = scan_same_step

        orig_start_r = self.sched._start_restore

        def start_restore_rec(request, rp, num_local, prev):
            r = orig_start_r(request, rp, num_local, prev)
            if r == "started":
                hbs = world.sched.hash_block_size
                newer = sum(1 for x in world.sched.rps.values()
                            if x is not rp and getattr(x, "chain", None) and x.state == "ok"
                            and hasattr(x, "extends") and x.extends(rp, hbs))
                world.restore_log.append({"req": request.request_id, "rp_n": rp.n_tokens, "local": num_local,
                                          "newer_alive": newer})
            return r

        self.sched._start_restore = start_restore_rec  # type: ignore[method-assign]
        orig_finish = self.sched._finish_load

        def finish_load(job, ok):
            world.in_restore = True
            try:
                st_before = {rid: st for rid, st in world.sched.restores.items() if st.job_id == job}
                orig_finish(job, ok)
                for st in st_before.values():
                    if st.state in ("ready", "done") and ok:
                        for nb_ in st.new_blocks:
                            world.check_block(nb_, tag="restored")
                            world.restored_verify += 1
            finally:
                world.in_restore = False

        self.sched._finish_load = finish_load  # type: ignore[method-assign]
        if fault in ("missing", "crc"):
            orig_start = self.sched._start_restore

            def start_restore(request, rp, num_local, prev):
                ok = orig_start(request, rp, num_local, prev)
                st = world.sched.restores.get(request.request_id)
                if ok == "started" and st is not None and st.entries and st.state == "loading":
                    world.inject(st)
                return ok

            self.sched._start_restore = start_restore  # type: ignore[method-assign]
        self._n_restore = 0
        store1 = self.workers[1].store
        if fault == "savefail":
            # rank 1 fails one block write in 3 (I/O error): those save jobs must fail as a whole on that rank,
            # the scheduler must forget the RP, unpin the blocks and never restore from it.
            orig_write = store1.write_block
            cnt = {"n": 0}

            def write_block(*a, **k):
                cnt["n"] += 1
                if cnt["n"] % 3 == 0:
                    world.faults_injected += 1
                    raise OSError(5, "injected I/O error")
                return orig_write(*a, **k)

            store1.write_block = write_block
        if fault == "rofs":
            # [review B2, exact injection] rank 1's filesystem turns read-only after its 40th block write (ext4
            # errors=remount-ro after an NVMe error): writes fail with EROFS, and unlinking an existing file fails
            # with EROFS too -- raised OUT of store.delete (harsher than the real method, which logs and returns).
            orig_write1 = store1.write_block
            orig_delete1 = store1.delete
            cnt1 = {"n": 0}

            def write_block_ro(*a, **k):
                cnt1["n"] += 1
                if cnt1["n"] > 40:
                    world.faults_injected += 1
                    raise OSError(errno.EROFS, "injected read-only fs")
                return orig_write1(*a, **k)

            def delete_ro(fname):
                if cnt1["n"] > 40 and os.path.exists(os.path.join(store1.dir, fname)):
                    raise OSError(errno.EROFS, "injected read-only fs (unlink)")
                return orig_delete1(fname)

            store1.write_block = write_block_ro
            store1.delete = delete_ro
        if fault == "noload":
            # negative control for the rig's verification: rank 1 reports every read as successful but writes
            # nothing into the GPU rows. The admission/restore content checks must catch it (verify_fail > 0).
            def read_noop(group, block_id, fname, stage, stream, *a, **k):
                world.faults_injected += 1
                return None

            store1.read_block = read_noop
        self.hang_release = threading.Event()
        self.hang_nth = int(os.environ.get("NVMESPILL_HANG_NTH", "2"))
        self.hang_s = float(os.environ.get("NVMESPILL_HANG_S", "1e9"))
        self.hang_events: list[dict] = []
        if fault in ("hang_load", "hang_save"):
            name = "read_block" if fault == "hang_load" else "write_block"
            orig_io = getattr(store1, name)
            seen_jobs: set[int] = set()
            cur_job = {"j": None}
            wk1 = self.workers[1]
            meth = "_load" if fault == "hang_load" else "_save"
            orig_job = getattr(wk1, meth)

            def job_wrapped(job_id, *a, **k):
                cur_job["j"] = job_id
                return orig_job(job_id, *a, **k)

            setattr(wk1, meth, job_wrapped)

            def io_hang(*a, **k):
                j = cur_job["j"]
                if j not in seen_jobs:
                    seen_jobs.add(j)
                    if len(seen_jobs) == world.hang_nth:
                        world.faults_injected += 1
                        t0 = time.monotonic()
                        world.hang_events.append({"job": j, "start": t0})
                        world.hang_release.wait(world.hang_s)
                        world.hang_events[-1]["held_s"] = round(time.monotonic() - t0, 2)
                return orig_io(*a, **k)

            setattr(store1, name, io_hang)
        self.noack_job = None
        if fault in ("noack_load", "noack_save"):
            kind = "loads" if fault == "noack_load" else "saves"
            seen: list[int] = []
            wk1 = self.workers[1]
            orig_handle = wk1.handle

            def handle_noack(meta):
                for j in getattr(meta, kind, {}) or {}:
                    if j not in seen:
                        seen.append(j)
                        if len(seen) == world.hang_nth:
                            world.noack_job = j
                            world.faults_injected += 1
                return orig_handle(meta)

            wk1.handle = handle_noack
            orig_take = wk1.take_meta

            def take_noack():
                m = orig_take()
                if m is not None and world.noack_job is not None and world.noack_job in m.done:
                    del m.done[world.noack_job]     # this rank's answer is lost
                return m

            wk1.take_meta = take_noack
        # asynchronous completion: per-rank random I/O latency, results collected without draining
        self.async_mode = os.environ.get("NVMESPILL_ASYNC", "") == "1"
        self.aborted: list[str] = []
        self.abort_p = float(os.environ.get("NVMESPILL_ABORT_P", "0"))
        if self.async_mode:
            import random as _r
            maxlat = [float(os.environ.get("NVMESPILL_LAT0", "0.08")), float(os.environ.get("NVMESPILL_LAT1", "0.25"))]
            lock = threading.Lock()
            rng = _r.Random(12345)
            for rank, w in enumerate(self.workers):
                st_ = w.store
                for name in ("read_block", "write_block"):
                    orig = getattr(st_, name)

                    def wrapped(*a, _o=orig, _m=maxlat[rank], **k):
                        with lock:
                            d = rng.uniform(0, _m)
                        time.sleep(d)
                        return _o(*a, **k)
                    setattr(st_, name, wrapped)
            self._arng = _r.Random(777)
        self._install_cause_tracking()

    # ---------------------------------------------------------------------------------------------------
    # [nvme-spill prune rig] observation only: why does a DSH turn still re-read? (observation added 2026-10-13)
    def _install_cause_tracking(self) -> None:
        sc = self.sched
        world = self
        self.rp_events: dict[bytes, dict] = {}       # RP key -> {n, created, ok, dropped, reason}
        self.skip_first: dict[bytes, float] = {}     # RP key a first turn would have had (SAVE_POLICY=repeat skip)
        self.skip_short: dict[bytes, float] = {}     # chain keys near the end of a request whose plan came out short
        self.dsh_adm: list[dict] = []
        orig_sp = sc._save_plan

        def sp(rp_key, n, entries, **kw):
            r = orig_sp(rp_key, n, entries, **kw)
            if rp_key in sc.rps and rp_key not in world.rp_events:
                world.rp_events[rp_key] = {"n": n, "created": sc.clock(), "ok": None, "dropped": None}
            return r

        sc._save_plan = sp
        orig_chk = sc._check_rp

        def chk(rp):
            r = orig_chk(rp)
            ev = world.rp_events.get(rp.key)
            if ev is not None and rp.state == "ok" and ev["ok"] is None:
                ev["ok"] = sc.clock()
            return r

        sc._check_rp = chk
        orig_drop = sc._drop_rp

        def drop(rp, reason):
            live = sc.rps.get(rp.key) is rp
            r = orig_drop(rp, reason)
            ev = world.rp_events.get(rp.key)
            if live and ev is not None and ev["dropped"] is None:
                ev["dropped"] = (sc.clock(), reason)
            return r

        sc._drop_rp = drop
        orig_forget = sc._forget_saving

        def forget(rp):
            live = sc.rps.get(rp.key) is rp
            r = orig_forget(rp)
            ev = world.rp_events.get(rp.key)
            if live and ev is not None and ev["dropped"] is None:
                ev["dropped"] = (sc.clock(), "save_failed")
            return r

        sc._forget_saving = forget
        orig_ms = sc._maybe_save

        def ms(block_hashes, num_tokens, *a, **k):
            f0, s0 = sc.stats["rp_skipped_first"], sc.stats["rp_skipped_short"]
            r = orig_ms(block_hashes, num_tokens, *a, **k)
            if sc.stats["rp_skipped_first"] != f0 and sc.seen:
                world.skip_first.setdefault(next(reversed(sc.seen)), sc.clock())
            if sc.stats["rp_skipped_short"] != s0:
                hbs = sc.hash_block_size
                top = min((num_tokens - 1) // hbs, len(block_hashes))
                for m in range(max(1, top - 2), top + 1):
                    world.skip_short.setdefault(bytes(block_hashes[m - 1]), sc.clock())
            return r

        sc._maybe_save = ms

    def note_admission(self, request, adm: dict, prev_total) -> None:
        if adm.get("kind") != "dsh":
            return
        sc = self.sched
        hbs = sc.hash_block_size
        chain = [bytes(h) for h in request.block_hashes[: (request.num_tokens - 1) // hbs]]
        snap = {"alive_ok": 0, "alive_saving": 0, "gone": (0, ""), "skip_first": 0, "skip_short": 0}
        now = sc.clock()
        # only resume points the previous turn could have left (at most its prompt + reply): deeper keys on this
        # request's chain are its own future RP
        top = len(chain) if prev_total is None else min(len(chain), prev_total // hbs)
        for m in range(top, 0, -1):
            k = chain[m - 1]
            rp = sc.rps.get(k)
            if rp is not None and rp.n_tokens == m * hbs:
                if rp.state == "ok" and not snap["alive_ok"]:
                    snap["alive_ok"] = m * hbs
                elif rp.state != "ok" and not snap["alive_saving"]:
                    snap["alive_saving"] = m * hbs
            ev = self.rp_events.get(k)
            if ev is not None and ev["dropped"] is not None and ev["dropped"][0] <= now and not snap["gone"][0]:
                snap["gone"] = (m * hbs, ev["dropped"][1])
            if k in self.skip_first and self.skip_first[k] <= now and not snap["skip_first"]:
                snap["skip_first"] = m * hbs
            if k in self.skip_short and self.skip_short[k] <= now and not snap["skip_short"]:
                snap["skip_short"] = m * hbs
        self.dsh_adm.append({"req": adm["req"], "turn": adm["turn"], "prompt": adm["prompt"], "hit": adm["hit"],
                             "t": now, "prev_total": prev_total, "chain": chain, **snap})

    def cause_breakdown(self) -> dict:
        """Split each DSH admission's re-read (prompt - hit) into causes. Turn 0: d_first_prefill. Later turns:
        f_new_turn_tokens (prompt - previous turn's prompt+reply: never computed before) + e_granularity (previous
        reply + the tail beyond the deepest state boundary an RP could hold) + the shortfall below that boundary,
        attributed to the deepest candidate on the request's chain at admission time."""
        hbs = self.sched.hash_block_size
        mg = self.sched.cfg.min_gain
        out: dict[str, int] = {}
        per_adm = []
        def add(k, v):
            if v:
                out[k] = out.get(k, 0) + int(v)
        for a in self.dsh_adm:
            rr = a["prompt"] - a["hit"]
            if a["turn"] == 0 or a["prev_total"] is None:
                add("d_first_prefill", rr)
                per_adm.append((a["req"], "d", rr))
                continue
            # an RP of this chain that only appeared after the admission (the previous turn was still computing)
            late = 0
            for m in range(min(len(a["chain"]), a["prev_total"] // hbs), 0, -1):
                ev = self.rp_events.get(a["chain"][m - 1])
                sk = self.skip_first.get(a["chain"][m - 1])
                if (ev is not None and ev["created"] > a["t"]) or (sk is not None and sk > a["t"]):
                    late = m * hbs
                    break
            cands = [(a["alive_ok"], "ok"), (a["alive_saving"], "b_rp_still_saving"), (a["gone"][0], "b_rp_" + (a["gone"][1] or "gone")),
                     (a["skip_first"], "a_first_turn_not_saved"), (a["skip_short"], "c_min_tokens_short_plan"),
                     (late, "e_previous_turn_still_computing")]
            target, why = max(cands, key=lambda x: x[0])
            new = max(0, a["prompt"] - a["prev_total"])
            add("f_new_turn_tokens", new)
            if target <= 0:
                add("e_no_rp_or_skip_on_chain", rr - new)
                per_adm.append((a["req"], "none", rr - new))
                continue
            top = max(target, a["hit"])
            add("e_granularity", max(0, a["prev_total"] - top))
            short = max(0, target - a["hit"])
            if short:
                if why == "ok":
                    why = "c_min_gain" if short < mg else "e_rp_ok_not_restored"
                add(why, short)
            rest = rr - new - max(0, a["prev_total"] - top) - short
            add("x_unattributed", rest)
            per_adm.append((a["req"], why if short else "-", short))
        out["total"] = sum(v for k, v in out.items())
        adm = [{k: v for k, v in a.items() if k != "chain"} for a in self.dsh_adm]
        return {"tokens": out, "shortfall_admissions": [p for p in per_adm if p[2]][:40], "admissions": adm}

    def bind_clock(self, fn) -> None:
        self.sched.clock = fn
        if hasattr(self.sched, "wall"):
            base = float(os.environ.get("NVMESPILL_WALL_BASE", "1760000000"))
            self.sched.wall = lambda: base + fn()       # TTL / last_used on the simulated clock (continuous over restarts)

    # -------------------------------------------------------------------------------------------------
    def write_content(self, block_id: int, key: bytes) -> None:
        g = self.get_group_id(key)
        for rank, raws in enumerate(self.raws):
            data = content(key, rank, sum(n for _, n in self.rows[g]))
            off = 0
            for t, page in self.rows[g]:
                raws[t].view(-1, page)[block_id].copy_(data[off:off + page])
                off += page
        self.content_writes += 1

    def check_block(self, block, tag: str) -> bool:
        if block.is_null or block.block_hash is None:
            return True
        key = block.block_hash
        g = self.get_group_id(key)
        ok = True
        for rank, raws in enumerate(self.raws):
            want = content(key, rank, sum(n for _, n in self.rows[g]))
            off = 0
            for t, page in self.rows[g]:
                if not torch.equal(raws[t].view(-1, page)[block.block_id], want[off:off + page]):
                    ok = False
                off += page
        self.verify_blocks += 1
        if not ok:
            self.verify_fail.append((tag, block.block_id, g))
        return ok

    def check_admission(self, request, hit: int) -> None:
        if hit <= 0:
            return
        # a hit may be on a block another request keyed in this same schedule() (allowed for full attention):
        # check after this step's forward wrote it
        self.pending_checks.append((request.request_id, request))

    def _run_checks(self) -> None:
        coord = self.scheduler.kv_cache_manager.coordinator
        for rid, request in self.pending_checks:
            for mgr in coord.single_type_managers:
                for b in mgr.req_to_blocks.get(rid, []):
                    if b.block_hash is not None and not b.is_null:
                        self.check_block(b, tag=f"admit:{rid}")
        self.pending_checks = []

    def _forward(self) -> None:
        """This step's forward: the blocks keyed in this schedule() get their content."""
        pool = self.scheduler.kv_cache_manager.block_pool
        for bid, key in self.pending_content:
            if pool.blocks[bid].block_hash == key:      # still that block's key (not evicted/reused meanwhile)
                self.write_content(bid, key)
        self.pending_content = []
        self._run_checks()

    def inject(self, st) -> None:
        """Fault injection on every 3rd restore: rank 1 loses one file (fault=missing) or gets a flipped byte
        (fault=crc). The job must fail on that rank, the restore must fall back to prefill, nothing may leak."""
        self._n_restore += 1
        if self._n_restore % 3:
            return
        e = st.entries[-1]
        path = os.path.join(self.workers[1].store.dir, e.fname)
        if not os.path.exists(path):
            return      # already gone (a shared file removed by an earlier injection)
        if self.fault == "missing":
            os.unlink(path)
        elif self.fault == "crc":
            with open(path, "r+b") as f:
                f.seek(4096 + 7)
                b = f.read(1)
                f.seek(4096 + 7)
                f.write(bytes([b[0] ^ 0x01]))
        self.faults_injected += 1

    # -------------------------------------------------------------------------------------------------
    def step(self, sched_out):
        """Hand the step's connector jobs to both ranks (each its own copy, as unpickled in production), run the
        step's forward, and return the KVConnectorOutput (sync: jobs finished; async: whatever has finished)."""
        import copy
        from vllm.v1.outputs import KVConnectorOutput

        meta = sched_out.kv_connector_metadata
        agg = None
        has_jobs = isinstance(meta, self.spill.SpillMeta) and (meta.saves or meta.loads or meta.deletes
                                                               or getattr(meta, "wipe", False)
                                                               or getattr(meta, "rank_deletes", None))
        if self.async_mode:
            if has_jobs:
                for w in self.workers:
                    w.handle(copy.deepcopy(meta))
            self._forward()
            if self.abort_p > 0 and self._arng.random() < self.abort_p:
                cands = sorted(rid for rid, st in self.sched.restores.items() if st.state == "loading")
                if cands:
                    from vllm.v1.request import RequestStatus
                    rid = cands[0]
                    self.scheduler.finish_requests([rid], RequestStatus.FINISHED_ABORTED)
                    self.aborted.append(rid)
            time.sleep(0.002)
        else:
            expect = []
            if has_jobs:
                for job in meta.saves:
                    items = self.sched.save_jobs.get(job)
                    if items is not None:
                        expect += [(fname, bytes(b.block_hash), g) for g, b, fname in items[0] if b.block_hash is not None]
                for w in self.workers:
                    w.handle(copy.deepcopy(meta))
                for w in self.workers:
                    if not w.drain(120):
                        raise TimeoutError("spill worker did not drain")
            # every file this step's saves wrote must hold f(key, rank) (sync mode: the copies are done, the forward
            # of this step has not run yet -- a copy of a block keyed in this schedule() would hold stale bytes)
            for fname, key, g in expect:
                for rank, w in enumerate(self.workers):
                    path = os.path.join(w.store.dir, fname)
                    if not os.path.exists(path):
                        continue        # the save failed on this rank (fault runs): nothing to compare
                    nbytes = sum(n for _, n in self.rows[g])
                    with open(path, "rb") as f:
                        f.seek(4096)
                        data = f.read(nbytes)
                    self.file_verify += 1
                    if data != content(key, rank, nbytes).numpy().tobytes():
                        self.file_verify_fail.append((fname[:12], rank, g))
            self._forward()
        for w in self.workers:
            m = w.take_meta()
            if m is not None:
                agg = m if agg is None else agg.aggregate(m)
        d = self.sched.disk_bytes
        if d > self.disk_peak:
            self.disk_peak = d
        t = self.sched.clock()
        if t >= self._next_disk_sample:
            self.disk_timeline.append((int(t), round(d / (1 << 30), 3), len(self.sched.rps)))
            self._next_disk_sample = t - (t % 300.0) + 300.0
        self.io_bytes = sum(w.stats["written_bytes"] + w.stats["loaded_bytes"] for w in self.workers)
        return KVConnectorOutput(kv_connector_worker_meta=agg)

    def summary(self) -> dict:
        s = dict(self.sched.stats)
        sc = self.sched
        s.update({
            "content_writes": self.content_writes,
            "verify_blocks": self.verify_blocks,
            "verify_fail": len(self.verify_fail),
            "verify_fail_first": self.verify_fail[:5],
            "restored_blocks_verified": self.restored_verify,
            "faults_injected": self.faults_injected,
            "connector_pinned_end": sc.pinned_blocks(),
            "disk_gib_accounted": round(sc.disk_bytes / (1 << 30), 3),
            "disk_gib_peak": round(self.disk_peak / (1 << 30), 3),
            "disk_timeline": self.disk_timeline,
            "restore_log": self.restore_log,
            "reread_causes": self.cause_breakdown() if hasattr(self, "dsh_adm") else None,
            "rps_end": len(sc.rps),
            "worker_stats": [dict(w.stats) for w in self.workers],
            "worker_threads_alive": [sum(th.is_alive() for th in w._threads) for w in self.workers],
            "aborted": list(self.aborted),
            "same_step": self.same_step,
            "orphans_end": len(sc.orphan_restores),
            "loading_end": sum(1 for st in sc.restores.values() if st.state == "loading"),
            "job_acc_end": len(sc.job_acc),
            "save_jobs_end": len(sc.save_jobs),
            "breaker": getattr(sc, "breaker", None),
            "quarantine_end": len(getattr(sc, "quarantine", {})),
            "quarantined_blocks_end": sum(len(st.new_blocks) for st in getattr(sc, "quarantine", {}).values()),
            "hang_events": self.hang_events,
            "noack_job": self.noack_job,
            "os_proxy_hits": self.os_proxy.hits if self.os_proxy else 0,
            "race_plans": getattr(self, "race_plans", 0),
            "race_blocks": getattr(self, "race_blocks", 0),
            "file_verify": self.file_verify,
            "file_verify_fail": len(self.file_verify_fail),
            "file_verify_fail_first": self.file_verify_fail[:5],
            "stats_line": sc.stats_line() if hasattr(sc, "stats_line") else "",
            "persist_state": getattr(sc, "persist_state", None),
            "persist_reason": getattr(sc, "persist_reason", None),
            "adopt_s": getattr(sc, "_adopt_s", None),
            "ident_s": getattr(sc, "_ident_s", None),
            "scan_s": [w.stats.get("scan_s") for w in self.workers],
            "files_on_disk": [len([f for f in os.listdir(w.store.dir) if f.endswith(".blk")]) for w in self.workers],
            "disk_files_accounted": len(sc.disk),
            "worker_stats_lines": [w.stats_line() for w in self.workers] if hasattr(self.workers[0], "stats_line")
            else [],
        })
        return s

    def shutdown(self):
        self.hang_release.set()
        for w in self.workers:
            w.shutdown()
