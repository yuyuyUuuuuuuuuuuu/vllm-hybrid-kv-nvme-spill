# SPDX-License-Identifier: AGPL-3.0-only
"""[nvme-spill finish rig, 2026-10-14] Observation-only eviction attribution (NVMESPILL_DIAG_EVICT=1).

Who evicts whose cached blocks: every eviction of a cached block (BlockPool._maybe_evict_cached_block that removed a
hash) is attributed to (a) the owner of the evicted block (the conversation whose prompt carries that block hash:
"c<k>", "head" for the shared head, "one" for one-shot requests, "?" unknown) and (b) the cause: the allocation that
popped it from the free queue -- "restore:<conv>" (the spill connector's _start_restore for a request of <conv>),
"alloc:<conv>" (KVCacheManager.allocate_slots for a request of <conv>), "other". Nothing here changes behaviour.
"""
from __future__ import annotations

from collections import defaultdict


def conv_of(rid: str) -> str:
    if rid.startswith("dsh"):
        return "c" + rid.replace("dshR", "dsh")[3:].split("t", 1)[0]
    if rid.startswith("one"):
        return "one"
    return "?"


class EvictDiag:
    def __init__(self, scheduler):
        from vllm.v1.core.kv_cache_utils import get_block_hash

        self.get_block_hash = get_block_hash
        self.owner: dict[bytes, str] = {}
        self.cause = "other"
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.restore_blocks: dict[str, int] = defaultdict(int)
        self.attach(scheduler)

    def note_request(self, request) -> None:
        c = conv_of(request.request_id)
        for h in request.block_hashes:
            k = bytes(h)
            prev = self.owner.get(k)
            if prev is None:
                self.owner[k] = c
            elif prev != c:
                self.owner[k] = "head"

    def attach(self, scheduler) -> None:
        bp = scheduler.kv_cache_manager.block_pool
        orig_evict = bp._maybe_evict_cached_block
        diag = self

        def evict(block):
            bh = block.block_hash
            r = orig_evict(block)
            if r and bh is not None:
                own = diag.owner.get(bytes(diag.get_block_hash(bh)), "?")
                diag.counts[own][diag.cause] += 1
            return r

        bp._maybe_evict_cached_block = evict
        kvm = scheduler.kv_cache_manager
        orig_alloc = kvm.allocate_slots

        def alloc(request, *a, **k):
            prev = diag.cause
            diag.cause = "alloc:" + conv_of(request.request_id)
            try:
                return orig_alloc(request, *a, **k)
            finally:
                diag.cause = prev

        kvm.allocate_slots = alloc
        conn = getattr(scheduler, "connector", None)
        sched = getattr(conn, "_sched", None) if conn is not None else None
        if sched is not None:
            orig_sr = sched._start_restore

            def start_restore(request, *a, **k):
                prev = diag.cause
                diag.cause = "restore:" + conv_of(request.request_id)
                try:
                    res = orig_sr(request, *a, **k)
                    if res == "started":
                        diag.restore_blocks[conv_of(request.request_id)] += 1
                    return res
                finally:
                    diag.cause = prev

            sched._start_restore = start_restore

    def summary(self) -> dict:
        return {"evicted_by_owner_cause": {o: dict(c) for o, c in self.counts.items()},
                "restores_started_by_conv": dict(self.restore_blocks)}
