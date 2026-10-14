# SPDX-License-Identifier: AGPL-3.0-only
"""GPU byte identity of a restart round trip (GB10, production page sizes, two ranks in one process).

phase 1 (process A): production-shaped KV tensors on the GPU (random bytes), real SpillWorkers (rank 0/1, O_DIRECT,
  pinned staging, CUDA streams) and the real SpillScheduler (fake block pool, persist on) save two turns of one
  conversation; an IN-PROCESS restore of the deeper resume point into other block ids is done too. The sha256 of every
  block row set (per rank) is recorded: original blocks and in-process restored blocks. Clean shutdown (index flush).
phase 2 (process B, a new interpreter = a restart): new GPU tensors with DIFFERENT random bytes, new workers + new
  scheduler on the same directories -> start-up scan -> adopt -> restore the same resume point into a fresh pool ->
  every restored block must hold exactly the phase-1 bytes on both ranks (= the in-process restore of phase 1).

usage: python3 t_restart_gpu.py --dir /spill/gpu_rt --phase 1|2   (GPU container, the image's vLLM; tests/run_gpu_tests.sh restart)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prodlayout  # noqa: E402
import test_sched_faults as T  # noqa: E402
import torch  # noqa: E402

HBS = T.HBS
NB = 24


def load_spill():
    spec = importlib.util.spec_from_file_location(
        "spill_gpu", os.path.join(HERE, "..", "code", "glm53_nvme_spill_connector.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["spill_gpu"] = m
    spec.loader.exec_module(m)
    return m


def world(spill, root, seed):
    out = []
    for rank in (0, 1):
        cfg = spill.SpillConfig(root=root, cap_bytes=64 << 30, min_free_bytes=1 << 30, direct=True, log_every_s=0,
                                persist=True, log=False)
        kvc = prodlayout.build_kv_cache_config(NB)          # production page sizes (MLA 2,359,296 B / idx 152,064 B)
        w = spill.SpillWorker(cfg, kvc, rank)
        g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
        raws = [torch.randint(0, 256, (kvt.size,), dtype=torch.uint8, device="cuda", generator=g)
                for kvt in kvc.kv_cache_tensors]
        views = {n: raws[t] for t, kvt in enumerate(kvc.kv_cache_tensors) for n in kvt.shared_by}
        w.register(views, sig="gpu-layout", ident_fn=lambda: {"node": "test-gb10", "layout": "prod"})
        w.drain(120)
        out.append((w, raws))
    return [x[0] for x in out], [x[1] for x in out]


def sched(spill, root, wall):
    cfg = spill.SpillConfig(root=root, cap_bytes=64 << 30, min_tokens=2 * HBS, min_gain=HBS, save_policy="all",
                            log=False, log_every_s=0, persist=True)
    kvc = prodlayout.build_kv_cache_config(NB)
    s = spill.SpillScheduler(cfg, SimpleNamespace(parallel_config=SimpleNamespace(world_size=2)), kvc)
    pool = T.FakePool(NB)
    s.bind(block_pool=pool, kv_cache_manager=SimpleNamespace(coordinator=T.FakeCoord(pool), block_pool=pool))
    s.clock = T.Clock()
    s.io_clock = T.Clock()
    s.wall = lambda: wall
    s._ident_thread.join(120)
    return s, pool


def step(s, workers):
    m = s.build_meta()
    for w in workers:
        w.handle(copy.deepcopy(m))
    for w in workers:
        w.drain(120)
    agg = None
    for w in workers:
        mm = w.take_meta()
        if mm is not None:
            agg = mm if agg is None else agg.aggregate(mm)
    s.update_connector_output(agg)


def block_sha(w, raws, g, bid):
    torch.cuda.synchronize()
    h = hashlib.sha256()
    for t, page in w.store.rows[g]:
        h.update(raws[t].view(-1, page)[bid].cpu().numpy().tobytes())
    return h.hexdigest()


def req(rid, hashes):
    n = (len(hashes) - 1) * HBS + 7
    return T.Req(request_id=rid, block_hashes=list(hashes), num_tokens=n, num_computed_tokens=n, num_prompt_tokens=n,
                 skip_reading_prefix_cache=False,
                 kv_transfer_params={"glm53_spill": {"ttl_s": 86400, "session": "gpu", "epoch": 0}})


def restored_shas(s, workers, raws, rid):
    st = s.restores[rid]
    out = {}
    for b in list(st.new_blocks) + list(st.pins):
        key = bytes(b.block_hash)
        g = int.from_bytes(key[-4:], "big")
        out[key.hex()] = [block_sha(workers[k], raws[k], g, b.block_id) for k in (0, 1)]
    return out, len(st.new_blocks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--phase", type=int, required=True)
    a = ap.parse_args()
    spill = load_spill()
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    init_none_hash(sha256)
    rec_path = os.path.join(a.dir, "phase1.json")
    c = [b"G" + i.to_bytes(4, "big") for i in range(12)]
    wall = 1_760_000_000.0
    if a.phase == 1:
        os.makedirs(a.dir, exist_ok=True)
        workers, raws = world(spill, a.dir, seed=1)
        s, pool = sched(spill, a.dir, wall)
        step(s, workers)
        orig = {}
        t0 = time.perf_counter()
        for rid, nb in (("t0", 3), ("t1", 5)):
            r = req(rid, c[:nb + 1])
            blocks = T.compute(pool, r, nb)
            for b in blocks:
                key = bytes(b.block_hash)
                g = int.from_bytes(key[-4:], "big")
                orig.setdefault(key.hex(), [block_sha(workers[k], raws[k], g, b.block_id) for k in (0, 1)])
            s.save_request(r)
            step(s, workers)
            step(s, workers)
        save_s = time.perf_counter() - t0
        assert s.stats["rp_saved"] == 2, s.stats
        # in-process restore (the existing path) of the deeper RP into fresh blocks
        pool.evict_all_cached()
        r2 = req("inproc", c[:7])
        s.get_num_new_matched_tokens(r2, 0)
        step(s, workers)
        step(s, workers)
        inproc, nnew = restored_shas(s, workers, raws, "inproc")
        s.update_state_after_alloc(r2)
        s.request_finished(r2)
        s.shutdown()
        for w in workers:
            w.shutdown()
        rec = {"orig": orig, "inproc": inproc, "inproc_new_blocks": nnew, "save_s": save_s,
               "files": sorted(os.listdir(workers[0].store.dir))}
        json.dump(rec, open(rec_path, "w"))
        ok = all(inproc[k] == orig[k] for k in inproc)
        print(f"{'PASS' if ok else 'FAIL'} gpu_phase1_inprocess_restore_byte_identical blocks={len(inproc)} "
              f"new={nnew} files={len(rec['files'])}")
        print(f"SUMMARY {int(ok)}/1 passed")
        return 0 if ok else 1
    rec = json.load(open(rec_path))
    t_start = time.perf_counter()
    workers, raws = world(spill, a.dir, seed=2)          # new process, new GPU bytes
    t_scan = time.perf_counter() - t_start
    s, pool = sched(spill, a.dir, wall + 3600)
    step(s, workers)
    step(s, workers)
    t_adopt = time.perf_counter() - t_start
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)

    check("gpu_restart_adopted", s.persist_reason == "adopted" and len(s.rps) == 2,
          f"{s.persist_reason} rps={len(s.rps)} scan+start {t_scan:.2f}s adopt {t_adopt:.2f}s")
    r = req("after", c[:7])
    t0 = time.perf_counter()
    s.get_num_new_matched_tokens(r, 0)
    step(s, workers)
    step(s, workers)
    load_s = time.perf_counter() - t0
    st = s.restores.get("after")
    check("gpu_restart_restore_ready", st is not None and st.state == "ready", f"{st and st.state} {load_s:.2f}s")
    got, nnew = restored_shas(s, workers, raws, "after")
    same_orig = all(got[k] == rec["orig"][k] for k in got)
    same_inproc = got.keys() == rec["inproc"].keys() and all(got[k] == rec["inproc"][k] for k in got)
    nbytes = sum(len(workers[0].store.rows[int.from_bytes(bytes.fromhex(k)[-4:], 'big')]) for k in got)
    check("gpu_restart_restore_byte_identical_to_saved", same_orig and len(got) > 0,
          f"blocks={len(got)} new={nnew} ranks=2")
    check("gpu_restart_restore_equals_inprocess_restore", same_inproc, f"{len(got)} vs {len(rec['inproc'])}")
    s.update_state_after_alloc(r)
    s.request_finished(r)
    s.shutdown()
    for w in workers:
        w.shutdown()
    print(f"SUMMARY {sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
