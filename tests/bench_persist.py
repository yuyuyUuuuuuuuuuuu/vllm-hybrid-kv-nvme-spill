# SPDX-License-Identifier: AGPL-3.0-only
"""Start-up / hot-path cost of reuse across restarts at production scale (CPU, real connector code, fake pool).

* index build (scheduler thread, every INDEX_EVERY_S while dirty) and write (index thread) for N resume points of
  B blocks each with a chain of C hashes (production: 4608-token hash blocks, 780k-token conversations ~ 170 blocks)
* index read + adoption decision at start-up (scheduler)
* a rank's start-up scan of F block files (listdir + stat; the identity digest is measured separately)

usage: python3 bench_persist.py --dir /spill/bp [--rps 300 --blocks 170 --files 12000]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prodlayout  # noqa: E402
import test_sched_faults as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rps", type=int, default=300)
    ap.add_argument("--blocks", type=int, default=170)
    ap.add_argument("--files", type=int, default=12000)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("spill_b", os.path.join(HERE, "..", "code",
                                                                         "glm53_nvme_spill_connector.py"))
    spill = importlib.util.module_from_spec(spec)
    sys.modules["spill_b"] = spill
    spec.loader.exec_module(spill)
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    init_none_hash(sha256)
    shutil.rmtree(a.dir, ignore_errors=True)
    os.makedirs(a.dir)
    out = {}

    def mk():
        cfg = spill.SpillConfig(root=a.dir, cap_bytes=1 << 50, min_tokens=2 * T.HBS, min_gain=T.HBS, persist=True,
                                log=False, log_every_s=0)
        s = spill.SpillScheduler(cfg, SimpleNamespace(parallel_config=SimpleNamespace(world_size=2)),
                                 prodlayout.build_kv_cache_config(64))
        pool = T.FakePool(64)
        s.bind(block_pool=pool, kv_cache_manager=SimpleNamespace(coordinator=T.FakeCoord(pool), block_pool=pool))
        t0 = time.perf_counter()
        s._ident_thread.join(600)
        return s, time.perf_counter() - t0

    s, ident_s = mk()
    out["sched_identity_s"] = round(s._ident_s, 3)
    s.build_meta()
    # synthetic state: N RPs, each B blocks (B-4 MLA + 4 states), chains of B hashes; neighbouring RPs share 80% MLA
    files = {}
    for i in range(a.rps):
        conv = i // 2
        chain = tuple(bytes([conv % 256, conv // 256]) + j.to_bytes(4, "big") + b"\0" * 26 for j in range(a.blocks))
        blocks = []
        for j in range(a.blocks - 4):
            key = chain[j] + (0).to_bytes(4, "big")
            blocks.append(spill._RPBlock(0, (key,), (j + 1) * T.HBS, spill.block_file_name(key)))
        for g in (2, 3, 4, 5):
            key = chain[-1] + bytes([i % 256]) + g.to_bytes(4, "big")
            blocks.append(spill._RPBlock(g, (key,), None, spill.block_file_name(key)))
        rp = spill._RP(key=chain[-1] + bytes([i % 256]), n_tokens=a.blocks * T.HBS, blocks=blocks, state="ok",
                       last_used=s.wall(), chain=chain, ttl_s=86400, session=f"sess-{conv}", epoch=0)
        s.rps[rp.key] = rp
        for e in blocks:
            if e.fname not in s.disk:
                s.disk[e.fname] = spill._DiskBlock(e.fname, e.group, s.file_bytes[e.group], state="ok", wid="b-1")
                s.disk_bytes += s.file_bytes[e.group]
                files[e.fname] = s.file_bytes[e.group]
            s.disk[e.fname].refs += 1
    s.hello = {r: {"rank": r, "sig": "S", "files": files} for r in (0, 1)}
    s.hash_info = s._check_hash()
    t0 = time.perf_counter()
    payload = s._index_payload()
    out["index_build_first_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    payload = s._index_payload()
    out["index_build_cached_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["index_mib"] = round(len(payload) / (1 << 20), 2)
    t0 = time.perf_counter()
    s._index_writer._write(payload)
    out["index_write_fsync_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["rps"], out["files"] = len(s.rps), len(files)
    out["disk_gib"] = round(s.disk_bytes / (1 << 30), 1)
    # start-up: read + verify + adopt in a new scheduler (same identity)
    s2, _ = mk()
    t0 = time.perf_counter()
    idx, err = spill._read_index(s2.cfg.index_path())
    out["index_read_verify_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    s2._index = idx
    s2.build_meta()
    t0 = time.perf_counter()
    s2._note_hello({r: {"rank": r, "sig": "S", "files": files} for r in (0, 1)})
    out["adopt_decision_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["adopted"] = s2.stats["persist_adopted_rps"]
    out["adopt_reason"] = s2.persist_reason
    # a rank's start-up scan: F files (sparse, real names), listdir + stat
    d = os.path.join(a.dir, "scan")
    os.makedirs(d)
    for k in range(a.files):
        with open(os.path.join(d, f"{k:040x}.blk"), "wb") as f:
            f.truncate(27631616)
    cfg = spill.SpillConfig(root=os.path.join(a.dir, "scanroot"), cap_bytes=1 << 40, direct=False, persist=True)
    store = SimpleNamespace(dir=d, rows=[[(0, 1)]], sig="x", file_bytes=lambda g: 0)
    w = spill.SpillWorker(cfg, prodlayout.build_kv_cache_config(4), 0)
    t0 = time.perf_counter()
    w._scan(store, None)
    out["rank_scan_files"] = len(w._hello["files"])
    out["rank_scan_listdir_stat_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    spill._site_digest()
    out["site_digest_s"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(out, indent=1))
    shutil.rmtree(a.dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
