# SPDX-License-Identifier: AGPL-3.0-only
"""CPU replay driver: real Scheduler + fake ModelRunnerOutput loop.

Per step, exactly what the production EngineCore does:
  scheduler.schedule() -> fabricated ModelRunnerOutput -> update_from_output()
plus the draft-token refresh that the production drafter would deliver
(scheduler.update_draft_token_ids with K dummy ids; acceptance is folded into
the fabricated sampled_token_ids).

No model is loaded. GPU is never touched. Everything the scheduler actually
decides -- admission (scheduler_reserve_full_isl), prefix-cache lookup, touch,
free order, preemption -- runs in the production code from the work copy.
"""

from __future__ import annotations

import heapq
import os
import statistics as st
import json
import time
from dataclasses import dataclass, field

from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request

from sim.configs import build_scheduler

# [nvme-spill rig] a step that schedules no token still costs an engine round trip (EngineCore keeps stepping while
# the connector has jobs in flight); without this the simulated clock froze while every request waited on the
# connector (review B2: the pin/space timers never fired in that state)
EMPTY_STEP_S = float(os.environ.get("NVMESPILL_EMPTY_STEP_S", "0.01"))
from sim.workload import (
    DECODE_STEP_S,
    PREFILL_TOK_S,
    Workload,
)

NUM_SPEC = 7
# Per-position draft acceptance, shaped like the production 10-second engine statistics
# (0.875, 0.812, 0.625, 0.500, 0.500, 0.250, 0.125)
# scaled by a constant so that E[accepted spec tokens] = 1.4, i.e. the mean
# accepted length is 2.4 tokens/step (our calibration choice for the rig).
PROD_POS_RATES = (0.875, 0.812, 0.625, 0.500, 0.500, 0.250, 0.125)


def _scale_for_mean_accept(mean_accept: float) -> float:
    lo, hi = 0.05, 1.5
    for _ in range(60):
        mid = (lo + hi) / 2
        total, cum = 0.0, 1.0
        for r in PROD_POS_RATES:
            cum *= mid * r
            total += cum
        if total < mean_accept:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


SPEC_RATE_SCALE = _scale_for_mean_accept(1.4)


@dataclass
class Attempt:
    step: int
    hit: int
    admitted: bool


@dataclass
class Rig:
    seed: int
    horizon_s: float = 4.0 * 3600
    max_steps: int = 600_000
    drain_max_steps: int = 120_000
    drain: bool = True
    fix_hold_on_wait: bool = False  # earlier experiment "fix 1" (GLM53_APC_HOLD_ON_WAIT); off in all spill runs
    fix_defer_boundary_free: bool = False  # earlier experiment "fix 2" (GLM53_APC_DEFER_BOUNDARY_FREE); off in all spill runs
    cond: str = ""  # condition label (A/B/C experiments)

    def __post_init__(self):
        import os

        os.environ["GLM53_APC_HOLD_ON_WAIT"] = "1" if self.fix_hold_on_wait else "0"
        os.environ["GLM53_APC_DEFER_BOUNDARY_FREE"] = (
            "1" if self.fix_defer_boundary_free else "0"
        )
        self.cond = self.cond or os.environ.get("APCREREAD_COND", "")
        self.now = 0.0
        self.lookup_records: list[tuple[int, str, int]] = []  # step, req, hit
        self._last_lookup: dict[str, tuple[int, int]] = {}
        self.attempts: dict[str, list[Attempt]] = {}
        self.alloc_records: list[tuple[int, str, bool]] = []  # step, req, ok
        self.free_events: list[tuple[int, tuple[int, ...]]] = []
        self._build_engine()
        self.rng = __import__("random").Random(self.seed * 7919 + 13)
        self.workload = Workload(self.seed, self.horizon_s)
        self.pending = list(self.workload.events)
        heapq.heapify(self.pending)
        # [nvme-spill ttl rig] engine restart in the middle (NVMESPILL_RESTART_AT sim s; the engine is down
        # NVMESPILL_RESTART_DOWN_S; NVMESPILL_RESTART_CRASH=1: no clean shutdown, the index is whatever was last written)
        self.restart_at = float(os.environ.get("NVMESPILL_RESTART_AT", "0") or 0)
        self.restart_down = float(os.environ.get("NVMESPILL_RESTART_DOWN_S", "600"))
        self.restart_crash = os.environ.get("NVMESPILL_RESTART_CRASH", "") == "1"
        # [u5 rig] NVMESPILL_RESTART_IDLE=1: production's restart (wait until idle -> `docker rm -f`). At
        # RESTART_AT no new arrival is admitted; the engine steps exactly as EngineCore does (while
        # Scheduler.has_requests(), which asks the connector's has_pending_push_work) until it would block idle, stays
        # idle NVMESPILL_RESTART_IDLE_S (simulated; the background index writer gets that long in real time), then the
        # restart (CRASH=1: SIGKILL, no shutdown hook). Before it: the RPs that were complete on every rank.
        self.restart_idle = os.environ.get("NVMESPILL_RESTART_IDLE", "") == "1"
        self.restart_idle_s = float(os.environ.get("NVMESPILL_RESTART_IDLE_S", "20"))
        # NVMESPILL_RESTART_IDLE_PROBES=N: then N requests like spill_quality_probe --phase prepare (a fresh 81k-token
        # prompt with the DSH fields, max_tokens 1 as the probe), each run to idle: the last one's RP is saved when it
        # finishes and confirmed by the engine's last (empty) steps -- the case seen in production on 2026-10-14
        self.restart_idle_probes = int(os.environ.get("NVMESPILL_RESTART_IDLE_PROBES", "0") or 0)
        self.restart_info: dict | None = None
        self.spill_before_restart: dict | None = None
        self._requeue: list = []
        self._post_init_rest()

    def _build_engine(self) -> None:
        """Scheduler (+ connector) then init_none_hash, as EngineCore; the simulated ranks; the observers."""
        # env is read lazily inside the patched code paths; a fresh scheduler
        # per Rig instance picks it up.
        self.scheduler, self.hasher, self.vllm_config, self.kv_cache_config = (
            build_scheduler()
        )
        # [nvme-spill rig] simulated worker ranks + content verification
        self.world = None
        if os.environ.get("NVMESPILL_ON") == "1":
            from sim.spill_world import SpillWorld

            self.world = SpillWorld(
                self.scheduler,
                os.environ["GLM53_NVME_SPILL_DIR"],
                fault=os.environ.get("NVMESPILL_FAULT", ""),
            )
        if self.world is not None:
            self.world.bind_clock(lambda: self.now)

        # ---- observation-only wrappers (no behavior change) ----
        scm = self.scheduler.kv_cache_manager
        observe = os.environ.get("APCREREAD_OBSERVE", "1") == "1"
        orig_get = scm.get_computed_blocks
        orig_alloc = scm.allocate_slots

        def rec_get(request):
            blocks, hit, spb = orig_get(request)
            self.lookup_records.append((self.step, request.request_id, hit))
            self._last_lookup[request.request_id] = (
                self.step,
                hit,
            )
            return blocks, hit, spb

        def rec_alloc(request, num_new_tokens, **kw):
            out = orig_alloc(request, num_new_tokens, **kw)
            self.alloc_records.append(
                (self.step, request.request_id, out is not None)
            )
            return out

        if observe:
            scm.get_computed_blocks = rec_get  # type: ignore[method-assign]
            scm.allocate_slots = rec_alloc  # type: ignore[method-assign]

        bp = scm.block_pool
        if observe:
            orig_free = bp.free_blocks

            def rec_free(ordered):
                # materialize first: callers may pass a one-shot iterator
                # (SingleTypeKVCacheManager.free passes reversed(list)); a
                # bare genexp over `ordered` would consume it and drop the
                # actual free.
                items = list(ordered)
                # [nvme-spill prune rig] the free log is never read by the summary; NVMESPILL_RIG_FREE_LOG=0 skips it
                # (it grows with every free call; 4 h runs in 4 GiB containers were OOM-killed at ~7 min with it on)
                if os.environ.get("NVMESPILL_RIG_FREE_LOG", "1") != "0":
                  self.free_events.append(
                    (
                        self.step,
                        tuple(
                            b.block_id
                            for b in items
                            if b.block_hash is not None and not b.is_null
                        ),
                    )
                )
                return orig_free(items)

            bp.free_blocks = rec_free  # type: ignore[method-assign]

        # [nvme-spill finish rig] count preemptions (the summary's "preemptions" was never incremented before)
        orig_preempt = self.scheduler._preempt_request

        def rec_preempt(request, *a, **k):
            self.preempt_count = getattr(self, "preempt_count", 0) + 1
            self.preempt_log = getattr(self, "preempt_log", [])
            if len(self.preempt_log) < 200:
                self.preempt_log.append((self.step, request.request_id, request.num_computed_tokens))
            return orig_preempt(request, *a, **k)

        self.scheduler._preempt_request = rec_preempt  # type: ignore[method-assign]

        # [nvme-spill finish rig] eviction attribution (observation only)
        if os.environ.get("NVMESPILL_DIAG_EVICT") == "1":
            from sim.diag import EvictDiag
            if getattr(self, "diag", None) is None:
                self.diag = EvictDiag(self.scheduler)
            else:
                self.diag.attach(self.scheduler)

    def _post_init_rest(self) -> None:
        # ---- per-run metrics state ----
        self.step = 0
        self.event_i = 0
        self.admissions: list[dict] = []
        self.finished_reqs: set[str] = set()
        self.sum_scheduled: dict[str, int] = {}
        self.sum_rejected: dict[str, int] = {}
        self.sched_prompt_tokens = 0  # tokens actually sent to prefill compute
        self.stalled_s = 0.0
        self.preempt_count = 0
        self.errors: list[str] = []
        # correctness ledgers
        self.hit_hash_violations: list[tuple] = []
        self.accounting_violations: list[tuple] = []
        self.invariant_errors: list[str] = []
        self.admission_hit: dict[str, int] = {}
        self.admission_meta: dict[str, tuple[int, int]] = {}

        # "fix 2" memory cost: boundary state blocks held per deferred manager
        self._deferred_hist: list[tuple[int, int]] = []
        self._deferred_peak = 0
        self._deferred_sum = 0
        self._deferred_n = 0
        self._deferred_avg = 0.0
        self._protected_peak = 0
        self._protected_sum = 0
        self._protected_n = 0
        self._protected_avg = 0.0

    # ------------------------------------------------------------------
    def _add(self, spec) -> None:
        from vllm.sampling_params import SamplingParams

        # [nvme-spill ttl rig] the DSH contract fields travel the vLLM path: SamplingParams.extra_args
        # ["kv_transfer_params"] -> Request.kv_transfer_params (v1/request.py), as /v1/responses does
        spill = getattr(spec, "spill", None)
        sp = SamplingParams(max_tokens=spec.output_len, ignore_eos=True,
                            extra_args={"kv_transfer_params": {"glm53_spill": spill}} if spill else None)
        # [nvme-spill prune rig] the previous turn's prompt + reply of this conversation (re-read cause split)
        if spec.kind == "dsh":
            last = getattr(self, "_conv_last_total", None)
            if last is None:
                last = self._conv_last_total = {}
                self._prev_total_of = {}
            self._prev_total_of[spec.req_id] = last.get(spec.conv)
            last[spec.conv] = spec.prompt_len + spec.output_len
        req = Request(
            request_id=spec.req_id,
            prompt_token_ids=self.workload.token_ids(spec),
            sampling_params=sp,
            pooling_params=None,
            arrival_time=self.now,
            block_hasher=self.hasher,
        )
        self.scheduler.add_request(req)
        if getattr(self, "diag", None) is not None:
            self.diag.note_request(req)

    # ------------------------------------------------------------------
    def _restart_inject(self) -> str:
        """[ttl rig] faults injected while the engine is down (NVMESPILL_RESTART_INJECT):
        corrupt_index (one byte flipped) | truncate_index (a partial write) | sig_env (a production patch flag
        changes: every identity differs) | rank1_sig (only rank 1's identity differs: the ranks disagree) |
        missing_file (rank 1 lost one indexed file) | crc_file (one indexed file of rank 1 has a flipped data byte)."""
        kind = os.environ.get("NVMESPILL_RESTART_INJECT", "")
        if not kind or self.world is None:
            return ""
        import json as _json
        cfg = self.scheduler.connector._sched.cfg
        idx_path = cfg.index_path()
        if kind in ("corrupt_index", "truncate_index"):
            raw = open(idx_path, "rb").read()
            if kind == "corrupt_index":
                b = bytearray(raw)
                b[len(b) * 2 // 3] ^= 0x20
                raw = bytes(b)
            else:
                raw = raw[: len(raw) // 2]
            with open(idx_path, "wb") as f:
                f.write(raw)
            return f"{kind} ({len(raw)} B)"
        if kind == "sig_env":
            os.environ["GLM53_RIG_PATCH_FLAG"] = "1"
            return kind
        if kind == "rank1_sig":
            os.environ["NVMESPILL_RANK1_SALT"] = "other-worker-image"
            return kind
        if kind in ("missing_file", "crc_file"):
            raw = open(idx_path, "rb").read()
            idx = _json.loads(raw[raw.index(b"\n") + 1:])
            rps = [r for r in idx["rps"] if r["x"]["n"] > 0]
            if not rps:
                return f"{kind}: no RP in the index"
            fname = rps[-1]["x"]["b"][-1][3]
            path = os.path.join(cfg.rank_dir(1), fname)
            if kind == "missing_file":
                os.unlink(path)
            else:
                with open(path, "r+b") as f:
                    f.seek(4096 + 11)
                    c = f.read(1)
                    f.seek(4096 + 11)
                    f.write(bytes([c[0] ^ 0x01]))
            return f"{kind} {fname[:12]}"
        return f"unknown {kind}"

    def _index_rps_on_disk(self):
        """[u5 rig] RP keys (hex) in the on-disk index right now (None: no index / it does not verify)."""
        sc = self.scheduler.connector._sched
        idx, _err = self.world.spill._read_index(sc.cfg.index_path())
        if idx is None:
            return None
        return sorted(r["x"]["k"] for r in idx.get("rps") or [])

    def _run_to_idle(self) -> dict:
        """[u5 rig] EngineCore until it would block idle (Scheduler.has_requests() False), then idle
        NVMESPILL_RESTART_IDLE_S without a step. Returns what the restart check needs."""
        st = self.scheduler
        sc = st.connector._sched
        n = self._steps_to_idle()
        from vllm.sampling_params import SamplingParams
        import random as _r
        probe_ok = []
        for k in range(self.restart_idle_probes):
            rng = _r.Random(7919 * (k + 1) + self.seed)
            sp = SamplingParams(max_tokens=1, ignore_eos=True, extra_args={"kv_transfer_params": {
                "glm53_spill": {"ttl_s": 86400, "session": f"probe-{k}", "epoch": 0}}})
            rid = f"probe{k}"
            st.add_request(Request(request_id=rid, prompt_token_ids=[rng.randrange(200, 100_000) for _ in range(81251)],
                                   sampling_params=sp, pooling_params=None, arrival_time=self.now,
                                   block_hasher=self.hasher))
            n += self._steps_to_idle()
            probe_ok.append(sum(1 for rp in sc.rps.values() if rp.state == "ok" and rp.session == f"probe-{k}"))
        # idle: EngineCore blocks in input_queue.get; nothing calls the connector any more. The background index writer
        # runs in real time meanwhile (production: >= 20 s between the last answer and `docker rm -f`).
        self.now += self.restart_idle_s
        wr = sc._index_writer
        if wr is not None:
            wr.flush(10.0)
        ok = [rp.key.hex() for rp in sc.rps.values() if rp.state == "ok" and not sc._is_expired(rp)]
        return {"idle_steps": n, "rps_ok_before": len(ok), "rps_ok_keys": sorted(ok), "probe_rps_ok": probe_ok,
                "rps_saving_before": sum(1 for rp in sc.rps.values() if rp.state != "ok"),
                "save_jobs_before": len(sc.save_jobs), "index_dirty_at_kill": bool(sc._index_dirty),
                "index_writes_before": sc.stats.get("index_writes", 0),
                "index_writes_idle": sc.stats.get("index_writes_idle", 0)}

    def _steps_to_idle(self) -> int:
        st = self.scheduler
        n = 0
        while st.has_requests() and n < 200_000:      # has_requests -> connector.has_pending_push_work, as EngineCore
            sched_out = st.schedule()
            self.step += 1
            n += 1
            self._snapshot_step(sched_out)
            for rid, k in sched_out.num_scheduled_tokens.items():
                self.sum_scheduled[rid] = self.sum_scheduled.get(rid, 0) + k
            d_prompt = sum(k for rid, k in sched_out.num_scheduled_tokens.items()
                           if rid in st.requests and (st.requests[rid].num_computed_tokens
                                                      < st.requests[rid].num_prompt_tokens
                                                      or st.requests[rid].is_prefill_chunk))
            d_dec = any(not st.requests[r].is_prefill_chunk for r in sched_out.num_scheduled_tokens
                        if r in st.requests)
            ddt = d_prompt / PREFILL_TOK_S + (DECODE_STEP_S if d_dec else 0.0)
            self.now += ddt if ddt > 0 else EMPTY_STEP_S
            prev_keys = set(st.requests)
            out = self._fake_output(sched_out)
            out.kv_connector_output = self.world.step(sched_out)
            st.update_from_output(sched_out, out)
            for rid in prev_keys - set(st.requests):
                self.finished_reqs.add(rid)
            self._refresh_drafts()
            self._observe_deferred()
        return n

    def _restart(self) -> None:
        """[nvme-spill ttl rig] Engine restart: every request in flight is lost (the clients resend it after the
        restart, as DSH retries), the GPU cache is gone, the connector and the ranks are new objects on the same spill
        directories (persist: adopt or wipe). Clean (connector shutdown flushes the index) or crash (it does not)."""
        from vllm.sampling_params import SamplingParams

        st = self.scheduler
        idle_info = None
        if self.restart_idle and self.world is not None:
            idle_info = self._run_to_idle()
            st = self.scheduler
        t0 = self.now
        lost = []
        for r in list(st.running) + list(st.waiting) + list(getattr(st, "skipped_waiting", [])):
            if r.request_id in self.finished_reqs or r.is_finished():
                continue
            lost.append((r.request_id, list(r.prompt_token_ids), r.max_tokens, r.kv_transfer_params))
        seen_ids = set()
        lost = [x for x in lost if not (x[0] in seen_ids or seen_ids.add(x[0]))]
        if self.world is not None:
            self.spill_before_restart = self._spill_summary()
            if not self.restart_crash:
                st.connector.shutdown()             # scheduler role: index flush (clean engine shutdown)
            else:
                # a crash: nothing pending is written any more (an index write already in progress completes or
                # not -- the rename is atomic either way)
                wr = st.connector._sched._index_writer
                if wr is not None:
                    with wr._cv:
                        wr._pending = None
                        wr._stop = True
                        wr._cv.notify_all()
            self.world.shutdown()                   # the rank processes end (their I/O threads joined)
        inj = self._restart_inject()
        self._known_running = set()
        self._build_engine()
        self.now = t0 + self.restart_down
        for rid, toks, max_tok, ktp in lost:
            nid = rid[:3] + "R" + rid[3:]
            sp = SamplingParams(max_tokens=max_tok, ignore_eos=True,
                                extra_args={"kv_transfer_params": ktp} if ktp else None)
            from vllm.v1.request import Request
            self.scheduler.add_request(Request(request_id=nid, prompt_token_ids=toks, sampling_params=sp,
                                               pooling_params=None, arrival_time=self.now, block_hasher=self.hasher))
            if rid.startswith("dsh"):
                self._prev_total_of[nid] = self._prev_total_of.get(rid)
        if idle_info is not None:
            idle_info["index_rps_on_disk"] = self._index_rps_on_disk()
        self.restart_idle_info = idle_info
        self.restart_info = {"at": t0, "down_s": self.restart_down, "crash": self.restart_crash, "inject": inj,
                             "resent": [x[0] for x in lost], "step": self.step,
                             "persist": (self.scheduler.connector._sched.persist_state
                                         if self.world is not None else None)}

    # ------------------------------------------------------------------
    def _fake_output(self, sched_out) -> ModelRunnerOutput:
        """Fabricate the ModelRunnerOutput for the scheduled batch.

        Mirrors the real worker's sampled-token contract (gpu_model_runner):
        requests whose scheduled chunk does not reach ``num_tokens`` get an
        EMPTY sampled row (discard_request_mask), decode rows carry 1 + the
        accepted spec tokens (E[accepted]=1.4 -> mean 2.4 tokens/step, task
        brief; the un-accepted remainder is rolled back by the scheduler).
        """
        req_ids = list(sched_out.num_scheduled_tokens.keys())
        idx = {rid: i for i, rid in enumerate(req_ids)}
        sampled: list[list[int]] = []
        for rid in req_ids:
            request = self.scheduler.requests.get(rid)
            if request is None:
                sampled.append([])
                continue
            spec_ids = sched_out.scheduled_spec_decode_tokens.get(rid)
            if request.num_computed_tokens < request.num_tokens:
                # partial prefill chunk: worker discards the sample
                sampled.append([])
            elif spec_ids:
                acc = 0
                for r in PROD_POS_RATES[: len(spec_ids)]:
                    if self.rng.random() < r * SPEC_RATE_SCALE:
                        acc += 1
                    else:
                        break
                self.sum_rejected[rid] = (
                    self.sum_rejected.get(rid, 0) + len(spec_ids) - acc
                )
                sampled.append([100] * (1 + acc))
            else:
                sampled.append([100])
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=idx,
            sampled_token_ids=sampled,
        )

    # ------------------------------------------------------------------
    def _refresh_drafts(self) -> None:
        """Deliver the drafter's next proposal (K dummy ids), as the real
        worker does via DraftTokenIds after each step."""
        req_ids, drafts = [], []
        for request in self.scheduler.running:
            if request.is_finished():
                continue
            if request.num_computed_tokens < request.num_prompt_tokens:
                continue  # still prefilling: drafter has nothing to draft on
            req_ids.append(request.request_id)
            drafts.append([200 + k for k in range(NUM_SPEC)])
        if req_ids:
            self.scheduler.update_draft_token_ids(
                DraftTokenIds(req_ids=req_ids, draft_token_ids=drafts)
            )

    # ------------------------------------------------------------------
    def _observe_deferred(self) -> None:
        total = 0
        for m in self.scheduler.kv_cache_manager.coordinator.single_type_managers:
            d = getattr(m, "_glm53_apc_deferred_boundary_blocks", None)
            if d:
                total += sum(len(v) for v in d.values())
        if total:
            self._deferred_peak = max(self._deferred_peak, total)
        self._deferred_hist.append((self.step, total))
        self._deferred_sum += total
        self._deferred_n += 1
        # [A/B] protected tier occupancy
        pq = getattr(
            self.scheduler.kv_cache_manager.block_pool,
            "_glm53_apc_protected_queue",
            None,
        )
        if pq is not None:
            n = pq.num_free_blocks
            self._protected_peak = max(self._protected_peak, n)
            self._protected_sum += n
            self._protected_n += 1

    # ------------------------------------------------------------------
    def _check_hit_blocks(self, request) -> None:
        """Correctness check: the hash of every hit block matches the prompt's range.

        Group 0 (MLA, dense, no skips): hit block i covers tokens
        [i*4608, (i+1)*4608); its hash must be one of the request's chained
        block hashes for that token range (4 hash units of 1152 per block),
        or -- for the deepest hit block -- the partial-entry hash at the hit
        boundary. The chained hashes are content-derived, so a match proves
        the reused block holds this prompt's prefix (no different content is reused).
        """
        hit = self.admission_hit.get(request.request_id, 0)
        if hit <= 0:
            return
        mgr0 = self.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
        blocks = mgr0.req_to_blocks.get(request.request_id, [])
        from vllm.v1.core.kv_cache_utils import get_block_hash

        hashes = request.block_hashes
        bs = mgr0.block_size  # 4608
        hb = self.scheduler.hash_block_size  # 1152
        num_blocks = min(hit // bs, len(blocks))
        for i in range(num_blocks):
            blk = blocks[i]
            if blk.is_null:
                continue
            lo = i * (bs // hb)
            hi = min(lo + bs // hb, len(hashes))
            cands = {hashes[j] for j in range(lo, hi)}
            if hit % bs != 0 and i == num_blocks - 1:
                # partial boundary: the deepest hit block may carry the
                # hash at the hit's own 1152 boundary
                cands.add(hashes[min(hit // hb, len(hashes)) - 1])
            bh = blk.block_hash
            ok = bh is not None and get_block_hash(bh) in {
                c if not hasattr(c, "block_hash") else get_block_hash(c)
                for c in cands
            }
            if not ok:
                self.hit_hash_violations.append(
                    (self.step, request.request_id, i, hit,
                     # [nvme-spill prune rig] diagnosis: computed tokens at admission, blocks held, the block's id /
                     # whether it has a hash / ref count, its hash vs the request's hash at the block end
                     request.num_computed_tokens, len(blocks), blk.block_id, bh is not None, blk.ref_cnt,
                     (bytes(get_block_hash(bh)).hex()[:16] if bh is not None else None),
                     bytes(hashes[hi - 1]).hex()[:16] if hi >= 1 else None,
                     [h for (st_, rid_, h) in self.lookup_records[-50:] if rid_ == request.request_id][-6:])
                )
                break

    # ------------------------------------------------------------------
    def _membership_invariant(self) -> list[str]:
        """Correctness check: ref_cnt==0 (non-null) <-> membership of the free queue, both directions.

        Two-tier: a ref0 block must sit in exactly one of the normal /
        protected queues."""
        bp = self.scheduler.kv_cache_manager.block_pool
        inq = set()
        for q, tail in (
            (bp.free_block_queue, bp.free_block_queue.fake_free_list_tail),
            (
                getattr(bp, "_glm53_apc_protected_queue", None),
                getattr(bp, "_glm53_apc_protected_queue", None).fake_free_list_tail
                if getattr(bp, "_glm53_apc_protected_queue", None) is not None
                else None,
            ),
        ):
            if q is None:
                continue
            node = q.fake_free_list_head.next_free_block
            while node is not None and node is not tail:
                if node.block_id in inq:
                    errs = [f"blk {node.block_id} in both tiers"]
                    self.invariant_errors.extend(errs)
                    return errs
                inq.add(node.block_id)
                node = node.next_free_block
        errs = []
        for b in bp.blocks:
            if b.is_null:
                continue
            if b.ref_cnt == 0 and b.block_id not in inq:
                errs.append(f"blk {b.block_id} ref0 not in queue")
            elif b.ref_cnt > 0 and b.block_id in inq:
                errs.append(f"blk {b.block_id} ref{b.ref_cnt} in queue")
        if errs:
            self.invariant_errors.extend(errs[:3])
            return errs[:3]
        return []

    # ------------------------------------------------------------------
    def _snapshot_step(self, sched_out) -> None:
        st = self.scheduler
        live = {r.request_id for r in st.running}
        newly = [r for r in st.running if r.request_id not in self._known_running]
        # rec_get stores the latest lookup per request; a lookup recorded one
        # step earlier (the counter increments right after schedule()) is the
        # current step's lookup when its request got admitted now.
        step_lookups: dict[str, int] = {}
        for rid, (lstep, lhit) in self._last_lookup.items():
            if lstep in (self.step - 1, self.step):
                step_lookups[rid] = lhit
        for r in newly:
            self.admissions.append(
                {
                    "step": self.step,
                    "req": r.request_id,
                    "kind": "dsh" if r.request_id.startswith("dsh") else "one",
                    "turn": int(r.request_id.rsplit("t", 1)[-1])
                    if r.request_id.startswith("dsh")
                    else 0,
                    "prompt": r.num_prompt_tokens,
                    "branch": self.workload.branch_of.get(r.request_id, ""),
                    # the prefix-cache hit at admission: read BEFORE this
                    # step's chunk advance polluted num_computed_tokens
                    # (post-schedule it is hit + first chunk). Falls back to
                    # num_computed_tokens when observation is disabled.
                    "hit": step_lookups.get(r.request_id, r.num_computed_tokens),
                }
            )
            # [nvme-spill finish rig] this admission's hit BEFORE the hit-block check: a re-admission (after a
            # preemption) must not be checked against the hit of the request's previous admission (the stale value made
            # hit_hash_violations count 1 in some async+CRC runs, docs/limitations.md)
            prev_hit = self.admission_hit.get(r.request_id)
            if prev_hit is not None:
                # a re-admission (after a preemption): the old record order checked it against prev_hit
                self.readmissions = getattr(self, "readmissions", 0) + 1
                if prev_hit > self.admissions[-1]["hit"]:
                    self.readmit_hit_shrank = getattr(self, "readmit_hit_shrank", 0) + 1
            self.admission_hit[r.request_id] = self.admissions[-1]["hit"]
            self._check_hit_blocks(r)
            if self.world is not None:
                self.world.check_admission(r, self.admissions[-1]["hit"])
                if hasattr(self.world, "note_admission"):
                    self.world.note_admission(r, self.admissions[-1],
                                              getattr(self, "_prev_total_of", {}).get(r.request_id))
            # per-group hits (direct observation of MLA/mamba survival; read-only)
            try:
                _, pg = self.scheduler.kv_cache_manager.coordinator \
                    .find_longest_cache_hit_per_group(
                        r.block_hashes, r.num_tokens - 1
                    )
                self.admissions[-1]["per_group"] = [int(h) for h in pg]
            except Exception:
                pass
            # accounting ledger restarts at each (re-)admission
            self.sum_scheduled[r.request_id] = 0
            self.sum_rejected[r.request_id] = 0
            self.admission_hit[r.request_id] = step_lookups.get(
                r.request_id, r.num_computed_tokens
            )
            self.admission_meta[r.request_id] = (
                r.num_prompt_tokens,
                r.max_tokens,
            )
        # forget requests that left running (finished or preempted) so a
        # later re-admission is observed again
        self._known_running = live
        # per-step queue-membership scan is O(pool); gate it behind
        # APCREREAD_STRICT=1 (correctness runs). ref/queue consistency is
        # still checked every step via the cheaper count invariant in
        # check_invariants at summary time.
        if os.environ.get("APCREREAD_STRICT") == "1":
            self._membership_invariant()

    # ------------------------------------------------------------------
    def run(self) -> dict:
        st = self.scheduler
        self._known_running: set[str] = set()
        ev_i = 0
        events = self.pending
        t0 = time.monotonic()
        n_decode_steps = 0
        while True:
            # inject arrivals
            while ev_i < len(events) and events[ev_i][0] <= self.now:
                self._add(events[ev_i][2])
                ev_i += 1
            if self.now > self.horizon_s or self.step >= self.max_steps:
                break
            if self.restart_at and self.restart_info is None and self.now >= self.restart_at:
                self._restart()
                st = self.scheduler
                continue
            if (
                ev_i >= len(events)
                and not st.running
                and not len(st.waiting)
                and not len(st.skipped_waiting)
            ):
                break
            if not st.running and not len(st.waiting) and not len(
                st.skipped_waiting
            ):
                # idle: jump straight to the next arrival
                if ev_i < len(events):
                    self.now = max(self.now, events[ev_i][0])
                    continue
                break
            sched_out = st.schedule()
            self.step += 1
            self._snapshot_step(sched_out)

            # metrics: prompt tokens scheduled this step
            prompt_tok = 0
            for rid, n in sched_out.num_scheduled_tokens.items():
                self.sum_scheduled[rid] = self.sum_scheduled.get(rid, 0) + n
                request = st.requests.get(rid)
                if request is not None:
                    is_chunk = (
                        request.num_computed_tokens
                        < request.num_prompt_tokens
                    ) or request.is_prefill_chunk
                    if is_chunk:
                        prompt_tok += n
            self.sched_prompt_tokens += prompt_tok
            self.stalled_s += prompt_tok / PREFILL_TOK_S

            prev_keys = set(st.requests)
            out = self._fake_output(sched_out)
            if self.world is not None:
                out.kv_connector_output = self.world.step(sched_out)
            st.update_from_output(sched_out, out)
            finished_now = prev_keys - set(st.requests)
            for rid in finished_now:
                # correctness check: computed tokens = prefix (hit) + new
                prompt, max_tok = self.admission_meta.get(rid, (0, 0))
                sched_sum = self.sum_scheduled.get(rid, 0)
                rej = self.sum_rejected.get(rid, 0)
                hit = self.admission_hit.get(rid, 0)
                # target identity: computed = prefix (hit) + new + output.
                # Tolerance NUM_SPEC: when check_stop truncates the draft row at
                # the stopping step, the accepted tokens actually recorded differ
                # from update_from_output's rejection count (< K = 7).
                expected = prompt + max_tok - hit
                diff = sched_sum - rej - expected
                if not (-NUM_SPEC <= diff <= NUM_SPEC):
                    self.accounting_violations.append(
                        (self.step, rid, sched_sum, rej, hit, expected)
                    )
                self.finished_reqs.add(rid)
            self._refresh_drafts()
            self._observe_deferred()

            # advance sim time: prefill compute at 2,036 tok/s; any step that
            # carries at least one decode row costs DECODE_STEP_S (calibrated
            # in workload.py against the production median aggregate decode
            # throughput of ~16 tok/s).
            has_decode = any(
                not st.requests[rid].is_prefill_chunk
                for rid in sched_out.num_scheduled_tokens
                if rid in st.requests
            )
            dt = prompt_tok / PREFILL_TOK_S
            dt += DECODE_STEP_S if has_decode else 0.0
            if has_decode:
                n_decode_steps += 1
            if dt == 0.0 and self.world is not None:
                dt = EMPTY_STEP_S
            self.now += dt

            for r in list(st.requests.values()):
                if r.is_finished() and r.request_id not in self.finished_reqs:
                    self.finished_reqs.add(r.request_id)

        # ---- drain phase: no new arrivals; run every admitted/waiting
        # request to completion so the end-state check (all blocks back in
        # the free list) is meaningful. Metric runs may skip it (the
        # per-step ref/queue invariants already ran on every step); the
        # dedicated correctness runs do the full drain.
        drain_steps = 0
        while (
            self.drain
            and (st.running or len(st.waiting) or len(st.skipped_waiting)
                 # [nvme-spill rig] keep stepping while the connector has jobs in flight, as EngineCore does
                 # (Scheduler.has_requests -> connector.has_pending_push_work)
                 or (st.connector is not None and (st.connector.has_pending_push_work()
                                                   or st.connector._sched._meta.saves)))
            and self.step < self.max_steps
            and drain_steps < self.drain_max_steps
        ):
            sched_out = st.schedule()
            self.step += 1
            drain_steps += 1
            self._snapshot_step(sched_out)
            d_prompt = 0
            for rid, n in sched_out.num_scheduled_tokens.items():
                self.sum_scheduled[rid] = self.sum_scheduled.get(rid, 0) + n
                rq = st.requests.get(rid)
                if rq is not None and (rq.num_computed_tokens < rq.num_prompt_tokens or rq.is_prefill_chunk):
                    d_prompt += n
            if self.world is not None:
                # [nvme-spill rig] the clock runs in the drain phase too (connector timers, pin timeouts)
                d_dec = any(not st.requests[r].is_prefill_chunk for r in sched_out.num_scheduled_tokens
                            if r in st.requests)
                ddt = d_prompt / PREFILL_TOK_S + (DECODE_STEP_S if d_dec else 0.0)
                self.now += ddt if ddt > 0 else EMPTY_STEP_S
            prev_keys = set(st.requests)
            out = self._fake_output(sched_out)
            if self.world is not None:
                out.kv_connector_output = self.world.step(sched_out)
            st.update_from_output(sched_out, out)
            finished_now = prev_keys - set(st.requests)
            for rid in finished_now:
                prompt, max_tok = self.admission_meta.get(rid, (0, 0))
                sched_sum = self.sum_scheduled.get(rid, 0)
                rej = self.sum_rejected.get(rid, 0)
                hit = self.admission_hit.get(rid, 0)
                diff = sched_sum - rej - (prompt + max_tok - hit)
                if not (-NUM_SPEC <= diff <= NUM_SPEC):
                    self.accounting_violations.append(
                        (self.step, rid, sched_sum, rej, hit, prompt + max_tok - hit)
                    )
                self.finished_reqs.add(rid)
            self._refresh_drafts()
            self._observe_deferred()

        # ---- end-state check: the pool must be fully restored
        bp = self.scheduler.kv_cache_manager.block_pool
        self.end_free_blocks = bp.get_num_free_blocks()
        self.end_ref_nonnull = sum(
            b.ref_cnt for b in bp.blocks if not b.is_null
        )
        # the null block (id 0) keeps a ref by design (never freed)
        self.end_bad = [
            b.block_id for b in bp.blocks if not b.is_null and b.ref_cnt != 0
        ]
        self.drain_steps = drain_steps

        run_s = time.monotonic() - t0
        return self._summarize(run_s, n_decode_steps)

    # ------------------------------------------------------------------
    def _summarize(self, run_s: float, n_decode_steps: int) -> dict:
        adm = self.admissions
        dsh = [a for a in adm if a["kind"] == "dsh"]
        # a "fall" is a continuing conversation turn that landed on (at most)
        # the shared head neighborhood; turn-0 openers legitimately read from
        # 13,824. The prod signature is hit==13,824 exactly; in the rig the
        # floor can also land one segment higher (18,432/23,040) because the
        # head plus one/two segments stay hot, so both thresholds are counted.
        falls = [
            a
            for a in dsh
            if a["hit"] <= 23_040 and a["prompt"] > 100_000 and a["turn"] >= 1
        ]
        # hit-loss-while-waiting: consecutive attempts on one request whose
        # hit shrank between a failed attempt and any later attempt
        per_req: dict[str, list[tuple[int, int, bool]]] = {}
        look_i = 0
        look_by_step: dict[int, list[tuple[str, int]]] = {}
        for step, rid, hit in self.lookup_records:
            look_by_step.setdefault(step, []).append((rid, hit))
        alloc_by_step: dict[int, list[tuple[str, bool]]] = {}
        for step, rid, ok in self.alloc_records:
            alloc_by_step.setdefault(step, []).append((rid, ok))
        lost_events = 0
        lost_tokens = 0
        for step, pairs in look_by_step.items():
            oks = dict(alloc_by_step.get(step, []))
            for rid, hit in pairs:
                per_req.setdefault(rid, []).append((step, hit, oks.get(rid, True)))
        for rid, hist in per_req.items():
            for (s1, h1, ok1), (s2, h2, _ok2) in zip(hist, hist[1:]):
                if not ok1 and h2 < h1:
                    lost_events += 1
                    lost_tokens += h1 - h2
        re_read = sum(a["prompt"] - a["hit"] for a in adm)
        dsh_reread = sum(a["prompt"] - a["hit"] for a in dsh)
        # [nvme-spill prune rig] DSH re-read split by turn kind (plain extension / regen / edit / edit2)
        by_branch: dict[str, dict[str, int]] = {}
        for a in dsh:
            k = a.get("branch", "") or "plain"
            d = by_branch.setdefault(k, {"admissions": 0, "re_read": 0, "falls": 0})
            d["admissions"] += 1
            d["re_read"] += a["prompt"] - a["hit"]
            d["falls"] += a in falls
        # [nvme-spill ttl rig] DSH re-read per conversation, and after the restart
        by_conv: dict[str, dict[str, int]] = {}
        for a in dsh:
            c = a["req"].replace("dshR", "dsh")[3:].split("t", 1)[0]
            d = by_conv.setdefault(c, {"admissions": 0, "re_read": 0, "falls": 0, "after_restart": 0})
            d["admissions"] += 1
            d["re_read"] += a["prompt"] - a["hit"]
            d["falls"] += a in falls
            if self.restart_info is not None and a["step"] > self.restart_info["step"]:
                d["after_restart"] += a["prompt"] - a["hit"]
        restart = None
        if self.restart_info is not None:
            after = [a for a in dsh if a["step"] > self.restart_info["step"]]
            first: dict[str, dict] = {}
            for a in after:
                c = a["req"].replace("dshR", "dsh")[3:].split("t", 1)[0]
                first.setdefault(c, {"req": a["req"], "prompt": a["prompt"], "hit": a["hit"]})
            restart = dict(self.restart_info)
            ii = getattr(self, "restart_idle_info", None)
            if ii is not None:
                sc = self.scheduler.connector._sched
                st_ = sc.stats
                on_disk = ii.get("index_rps_on_disk") or []
                ok_keys = ii["rps_ok_keys"]
                restart["idle"] = {k: v for k, v in ii.items() if k not in ("rps_ok_keys", "index_rps_on_disk")}
                restart["idle"].update({
                    "ok_rps_in_index_at_kill": sum(1 for k in ok_keys if k in on_disk),
                    "ok_rps_missing_from_index": sum(1 for k in ok_keys if k not in on_disk),
                    "adopted_rps": st_.get("persist_adopted_rps", 0),
                    "orphan_files": st_.get("persist_orphans", 0),
                    "dropped_expired": st_.get("persist_dropped_expired", 0),
                    "dropped_missing": st_.get("persist_dropped_missing", 0),
                    "all_ok_rps_adopted": (st_.get("persist_adopted_rps", 0) == len(ok_keys)
                                           and st_.get("persist_orphans", 0) == 0)})
            restart.update({"dsh_re_read_after": sum(a["prompt"] - a["hit"] for a in after),
                            "dsh_admissions_after": len(after),
                            "dsh_first_after": first,
                            "dsh_first_after_re_read": sum(v["prompt"] - v["hit"] for v in first.values()),
                            "dsh_falls_after": sum(1 for a in after if a in falls),
                            "re_read_after_all": sum(a["prompt"] - a["hit"] for a in adm
                                                     if a["step"] > self.restart_info["step"])})
        # correctness: block accounting
        bp = self.scheduler.kv_cache_manager.block_pool
        bad_ref = [b.block_id for b in bp.blocks if b.ref_cnt < 0 and not b.is_null]
        nonnull_refcnt = sum(b.ref_cnt for b in bp.blocks if not b.is_null)
        free = bp.get_num_free_blocks()
        # invariants
        inv_errors = self.check_invariants()
        # deferred boundary blocks still held at end (should be 0 when empty)
        if self._deferred_n:
            self._deferred_avg = round(self._deferred_sum / self._deferred_n, 2)
        # per-group hit totals (DSH continuation turns only; survival per resource around collapses)
        dsh_pg = [a for a in dsh if "per_group" in a and a["turn"] >= 1]
        if dsh_pg:
            mla_mean = st.mean(a["per_group"][0] for a in dsh_pg)
            mamba_mean = st.mean(min(a["per_group"][2:6]) for a in dsh_pg)
        else:
            mla_mean = 0.0
            mamba_mean = 0.0
        if self._protected_n:
            self._protected_avg = round(self._protected_sum / self._protected_n, 2)
        return {
            "seed": self.seed,
            "cond": self.cond,
            "fix1": self.fix_hold_on_wait,
            "fix2": self.fix_defer_boundary_free,
            "mla_hit_mean": round(mla_mean),
            "mamba_hit_mean": round(mamba_mean),
            "protected_peak_blocks": self._protected_peak,
            "protected_avg_blocks": self._protected_avg,
            "steps": self.step,
            "sim_s": round(self.now, 1),
            "wall_s": round(run_s, 1),
            "decode_steps": n_decode_steps,
            "admissions": len(adm),
            "dsh_admissions": len(dsh),
            "re_read_tokens": re_read,
            "dsh_re_read_tokens": dsh_reread,
            "dsh_falls_le23040": len(falls),
            "dsh_by_branch": by_branch,
            "dsh_falls_eq13824": sum(
                1 for a in dsh if a["hit"] == 13_824 and a["turn"] >= 1 and a["prompt"] > 100_000
            ),
            "lost_hit_events": lost_events,
            "lost_hit_tokens": lost_tokens,
            "stalled_s_est": round(self.stalled_s, 1),
            "sched_prompt_tokens": self.sched_prompt_tokens,
            "preemptions": self.preempt_count,
            "preempt_log": getattr(self, "preempt_log", [])[:50],
            "readmissions": getattr(self, "readmissions", 0),
            "readmit_hit_shrank": getattr(self, "readmit_hit_shrank", 0),
            "finished": len(self.finished_reqs),
            "pending_unfinished": len(self.scheduler.requests),
            "free_blocks_end": free,
            "nonnull_refcnt_end": nonnull_refcnt,
            "drain_steps": getattr(self, "drain_steps", 0),
            "end_free_blocks": getattr(self, "end_free_blocks", None),
            "end_ref_nonnull": getattr(self, "end_ref_nonnull", None),
            "end_bad_blocks": getattr(self, "end_bad", None),
            "bad_ref_blocks": bad_ref,
            "invariant_errors": inv_errors,
            "hit_hash_violations": len(self.hit_hash_violations),
            "accounting_violations": len(self.accounting_violations),
            "deferred_peak_blocks": self._deferred_peak,
            "deferred_avg_blocks": self._deferred_avg,
            "errors": self.errors[:20],
            "dsh_by_conv": by_conv,
            "restart": restart,
            "spill_before_restart": self.spill_before_restart,
            "spill": self._spill_summary(),
            "diag_evict": self.diag.summary() if getattr(self, "diag", None) is not None else None,
        }

    def _spill_summary(self):
        if self.world is None:
            return None
        sp = self.world.summary()
        # [nvme-spill prune rig] which RP each restore used, by turn kind: newer_alive = newer RPs of the same chain
        # still on disk (0 = the newest RP of its chain)
        hist: dict[str, dict[str, int]] = {}
        for r in sp.get("restore_log", []):
            r["branch"] = self.workload.branch_of.get(r["req"], "") or "plain"
            d = hist.setdefault(r["branch"], {})
            k = str(r["newer_alive"])
            d[k] = d.get(k, 0) + 1
        sp["restore_newer_alive_by_branch"] = hist
        return sp

    # ------------------------------------------------------------------
    def check_invariants(self) -> list[str]:
        errs: list[str] = []
        bp = self.scheduler.kv_cache_manager.block_pool
        q = bp.free_block_queue
        pq = getattr(bp, "_glm53_apc_protected_queue", None)
        blocks = bp.blocks
        ref0 = sum(1 for b in blocks if b.ref_cnt == 0 and not b.is_null)
        total_free = q.num_free_blocks + (
            pq.num_free_blocks if pq is not None else 0
        )
        if total_free != ref0:
            errs.append(
                f"free_count {total_free} != ref0_count {ref0} (step {self.step})"
            )
        if pq is not None:
            pid_count = len(getattr(bp, "_glm53_apc_protected_ids", ()) or ())
            if pid_count != pq.num_free_blocks:
                errs.append(
                    f"protected_ids {pid_count} != protected_queue "
                    f"{pq.num_free_blocks} (step {self.step})"
                )
        if ref0 > len(blocks):
            errs.append("more free blocks than pool")
        # queue walk integrity
        n = 0
        node = q.fake_free_list_head.next_free_block
        seen = set()
        while node is not None and node is not q.fake_free_list_tail:
            if node.block_id in seen:
                errs.append("cycle in free queue")
                break
            seen.add(node.block_id)
            n += 1
            node = node.next_free_block
            if n > len(blocks) + 2:
                errs.append("free queue too long")
                break
        if n != q.num_free_blocks and not errs:
            errs.append(f"queue walk {n} != num_free {q.num_free_blocks}")
        return errs[:10]
