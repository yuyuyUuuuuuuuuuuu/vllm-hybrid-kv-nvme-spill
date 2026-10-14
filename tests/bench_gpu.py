# SPDX-License-Identifier: AGPL-3.0-only
"""GB10 GPU bench of the spill worker: one rank's share of a 600k-token resume point (131 MLA+indexer blocks,
4 mamba states, 2 drafter blocks = production pages) written to and read back from the local NVMe through the real
SpillWorker (pinned 32 MiB staging, O_DIRECT, CRC-32/8 MiB), byte-identity check on the GPU, and the interference on
a concurrent bandwidth-bound kernel (a large device-to-device copy loop standing in for decode).

Usage (inside tests/run_gpu_tests.sh, spill dir mounted rw): python3 tests/bench_gpu.py --dir /spill/gpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))

import torch  # noqa: E402

import prodlayout  # noqa: E402


class BwLoad:
    """A bandwidth-bound loop on its own stream: copies a big buffer back and forth, counts bytes moved."""

    def __init__(self, nbytes: int):
        self.a = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        self.b = torch.empty_like(self.a)
        self.stream = torch.cuda.Stream()
        self.stop = threading.Event()
        self.moved = 0
        self.t = 0.0

    def run(self):
        torch.cuda.set_device(0)
        t0 = time.perf_counter()
        n = 0
        with torch.cuda.stream(self.stream):
            while not self.stop.is_set():
                for _ in range(8):
                    self.b.copy_(self.a)
                    n += 1
                self.stream.synchronize()
        self.t = time.perf_counter() - t0
        self.moved = 2 * n * self.a.numel()   # read + write

    def measure(self, seconds: float) -> float:
        self.stop.clear()
        th = threading.Thread(target=self.run)
        th.start()
        time.sleep(seconds)
        self.stop.set()
        th.join()
        return self.moved / self.t / 1e9


def run_job(w, meta, timeout=600):
    w.handle(meta)
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        m = w.take_meta()
        if m is not None:
            return m.done
        time.sleep(0.001)
    raise TimeoutError


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mla-blocks", type=int, default=131)   # 600k tokens / 4608
    ap.add_argument("--bw-gib", type=float, default=2.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import glm53_nvme_spill_connector as spill

    torch.cuda.set_device(0)
    nb = args.mla_blocks + 4 + 2 + 2          # + mamba x4 + drafter x2 + null + spare
    kvc = prodlayout.build_kv_cache_config(nb)
    pages, rows = spill.group_rows(kvc)
    g = torch.Generator(device="cuda").manual_seed(7)
    raws = []
    views = {}
    for kvt in kvc.kv_cache_tensors:
        t = torch.randint(0, 256, (kvt.size,), dtype=torch.uint8, device="cuda", generator=g)
        raws.append(t)
        for name in kvt.shared_by:
            views[name] = t
    torch.cuda.synchronize()
    cfg = spill.SpillConfig(root=args.dir, cap_bytes=64 << 30, min_free_bytes=10 << 30)
    w = spill.SpillWorker(cfg, kvc, 0)
    w.register(views, sig="bench")
    # entries: MLA blocks 1..n, mamba groups 2-5 in the next 4 blocks, drafter 2 blocks
    ent = [(0, 1 + i, f"mla{i}.blk") for i in range(args.mla_blocks)]
    base = 1 + args.mla_blocks
    ent += [(2 + j, base + j, f"mamba{j}.blk") for j in range(4)]
    ent += [(6, base + 4 + j, f"draft{j}.blk") for j in range(2)]
    nbytes = sum(w.store.file_bytes(gg) for gg, _, _ in ent)
    ref = [r.clone() for r in raws]
    res = {"entries": len(ent), "bytes": nbytes, "tokens_equiv": args.mla_blocks * 4608}

    bw = BwLoad(int(args.bw_gib * (1 << 30)))
    res["bw_alone_GBps"] = round(bw.measure(3.0), 1)

    # ---- write alone ----
    t0 = time.perf_counter()
    done = run_job(w, spill.SpillMeta(saves={1: ent}))
    res["write_s"] = round(time.perf_counter() - t0, 3)
    res["write_ok"] = done == {1: [1, 0]}
    res["write_GBps"] = round(nbytes / res["write_s"] / 1e9, 2)

    # ---- read alone (scramble first) ----
    for r in raws:
        r.fill_(0xA5)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    done = run_job(w, spill.SpillMeta(loads={2: ent}))
    res["read_s"] = round(time.perf_counter() - t0, 3)
    res["read_ok"] = done == {2: [1, 0]}
    res["read_GBps"] = round(nbytes / res["read_s"] / 1e9, 2)
    torch.cuda.synchronize()
    touched = {(t, b) for gg, b, _ in ent for t, _ in rows[gg]}
    ident = True
    for t, (cur, old) in enumerate(zip(raws, ref)):
        cv, ov = cur.view(nb, -1), old.view(nb, -1)
        for b in range(nb):
            if (t, b) in touched and not torch.equal(cv[b], ov[b]):
                ident = False
    res["read_byte_identical"] = ident

    # ---- interference: bandwidth kernel during write and during read ----
    for name, meta in (("write", spill.SpillMeta(saves={3: [(gg, b, "x" + f) for gg, b, f in ent]})),
                       ("read", spill.SpillMeta(loads={4: ent}))):
        bw.stop.clear()
        th = threading.Thread(target=bw.run)
        th.start()
        time.sleep(0.3)
        t0 = time.perf_counter()
        done = run_job(w, meta)
        dt = time.perf_counter() - t0
        bw.stop.set()
        th.join()
        res[f"{name}_with_bw_s"] = round(dt, 3)
        res[f"bw_during_{name}_GBps"] = round(bw.moved / bw.t / 1e9, 1)
        res[f"{name}_with_bw_ok"] = list(done.values()) == [[1, 0]]
    res["worker_stats"] = w.stats
    # cleanup files
    for name in os.listdir(w.store.dir):
        os.unlink(os.path.join(w.store.dir, name))
    w.shutdown()
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
