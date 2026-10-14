# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests of the spill tier's worker side on production-shaped pages (CPU, or GPU with --device cuda).

Checks: save -> load into other block ids is byte-identical for every saved group (MLA+indexer, mamba x4, drafter);
no other byte of any KV tensor changes; CRC mismatch, header/layout mismatch, truncated (partial) file, a stray
.partial at start, the free-disk floor and a real ENOSPC all fail closed (the job reports failure, the destination
rows are untouched, no file is left behind); two ranks report through SpillWorkerMeta.aggregate and one rank failing
fails the job.

Usage: python3 tests/test_store.py --dir <spill dir> [--full-dir <small tmpfs>] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))

import torch  # noqa: E402

import prodlayout  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def make_worker(spill, cfg_dir, rank, kvc, device, seed, **cfg_kw):
    cfg = spill.SpillConfig(root=cfg_dir, cap_bytes=8 << 30, min_free_bytes=cfg_kw.pop("min_free", 0), **cfg_kw)
    w = spill.SpillWorker(cfg, kvc, rank)
    kv, raws = prodlayout.allocate(kvc, device=device, seed=seed)
    # mimic the runner's per-layer views (mamba: (nb,1,1,page) int8 view of the raw tensor)
    nb = kvc.num_blocks
    views = {}
    for t, kvt in enumerate(kvc.kv_cache_tensors):
        page = kvt.size // nb
        for name in kvt.shared_by:
            views[name] = raws[t].view(nb, 1, 1, page) if name.startswith("m") else raws[t].view(nb, page)
    w.register(views, sig="test-sig")
    return w, raws


def run_job(w, meta, timeout=120):
    w.handle(meta)
    t_end = time.monotonic() + timeout
    got = {}
    while time.monotonic() < t_end:
        m = w.take_meta()
        if m is not None:
            for k, v in m.done.items():
                got[k] = v
        if all(k in got for k in list(meta.saves) + list(meta.loads)):
            return got
        time.sleep(0.002)
    raise TimeoutError("job did not finish")


def rows_of(raws, rows, g, bid):
    return [raws[t].view(-1, page)[bid].clone() for t, page in rows[g]]


def digest_all(raws):
    import hashlib
    h = hashlib.sha256()
    for r in raws:
        h.update(r.view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--full-dir", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--blocks", type=int, default=12)
    args = ap.parse_args()
    import glm53_nvme_spill_connector as spill

    kvc = prodlayout.build_kv_cache_config(args.blocks)
    pages, rows = spill.group_rows(kvc)
    gb = [sum(n for _, n in r) for r in rows]
    check("geometry", gb[0] == 11 * 2359296 + 11 * 152064 and gb[2] == 9 * 2359296 and gb[4] == 8 * 2359296
          and gb[6] == 5 * 2359296 and gb[1] == 11 * 152064, f"group_bytes={gb}")

    # stray partial is wiped at start
    rd = os.path.join(args.dir, spill.SUBDIR, "rank0")
    os.makedirs(rd, exist_ok=True)
    open(os.path.join(rd, "junk.blk.123.partial"), "wb").write(b"x" * 100)
    w0, raws0 = make_worker(spill, args.dir, 0, kvc, args.device, seed=1)
    w1, raws1 = make_worker(spill, args.dir, 1, kvc, args.device, seed=2)
    check("wipe_partial_at_start", not any(n.endswith(".partial") for n in os.listdir(rd)))

    # ---- save groups 0, 2, 3, 4, 5, 6 from blocks 1..6 ----
    src = {0: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    names = {g: f"g{g}_b{b}.blk" for g, b in src.items()}
    saves = {1: [(g, b, names[g]) for g, b in src.items()]}
    ref0 = {g: rows_of(raws0, rows, g, b) for g, b in src.items()}
    ref1 = {g: rows_of(raws1, rows, g, b) for g, b in src.items()}
    t0 = time.perf_counter()
    r0 = run_job(w0, spill.SpillMeta(saves=saves))
    r1 = run_job(w1, spill.SpillMeta(saves=saves))
    agg = spill.SpillWorkerMeta(done=r0).aggregate(spill.SpillWorkerMeta(done=r1))
    check("save_both_ranks", agg.done.get(1) == [2, 0], f"agg={agg.done} {time.perf_counter() - t0:.2f}s")
    sizes = sorted(os.path.getsize(os.path.join(rd, n)) for n in names.values())
    check("file_sizes_4k_aligned", all(s % 4096 == 0 for s in sizes), f"{sizes}")

    # ---- load into blocks 7..11 (+0 is null; use distinct) ----
    dst = {0: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 0 if args.blocks <= 12 else 12}
    dst[6] = 11 if False else dst[6]
    # avoid the null block 0 for realism: reuse block 1 for drafter (already saved, overwrite is fine)
    dst[6] = 1
    before_other = None
    # scramble destination rows first so a no-op load cannot pass
    for g, b in dst.items():
        for t, page in rows[g]:
            raws0[t].view(-1, page)[b].fill_(0x5A)
    keep_rows = {(t, b) for g, b in dst.items() for t, _ in rows[g]}
    snap = [r.clone() for r in raws0]
    loads = {2: [(g, b, names[g]) for g, b in dst.items()]}
    t0 = time.perf_counter()
    r = run_job(w0, spill.SpillMeta(loads=loads))
    dt = time.perf_counter() - t0
    check("load_ok", r.get(2) == [1, 0], f"{r} {dt:.2f}s")
    ident = all(torch.equal(a, bb) for g, b in dst.items() for a, bb in zip(rows_of(raws0, rows, g, b), ref0[g]))
    check("load_byte_identical_all_groups", ident)
    # nothing else changed
    other_ok = True
    for t, (cur, old) in enumerate(zip(raws0, snap)):
        page = pages[t]
        cv, ov = cur.view(-1, page), old.view(-1, page)
        for b in range(args.blocks):
            if (t, b) in keep_rows:
                continue
            if not torch.equal(cv[b], ov[b]):
                other_ok = False
    check("no_other_bytes_changed", other_ok)
    # rank 1 loads its own (different) bytes
    r = run_job(w1, spill.SpillMeta(loads={3: [(0, 9, names[0])]}))
    check("rank1_load_own_bytes", r.get(3) == [1, 0] and all(
        torch.equal(a, b) for a, b in zip(rows_of(raws1, rows, 0, 9), ref1[0])))
    check("ranks_differ", not all(torch.equal(a, b) for a, b in zip(ref0[0], ref1[0])))

    # ---- corruption: flip one data byte in the mamba file on rank 0 ----
    p = os.path.join(rd, names[2])
    with open(p, "r+b") as f:
        f.seek(4096 + 12345)
        b = f.read(1)
        f.seek(4096 + 12345)
        f.write(bytes([b[0] ^ 0xFF]))
    for t, page in rows[2]:
        raws0[t].view(-1, page)[8].fill_(0x33)
    r = run_job(w0, spill.SpillMeta(loads={4: [(2, 8, names[2])]}))
    untouched = all(bool((raws0[t].view(-1, page)[8] == 0x33).all()) for t, page in rows[2])
    check("crc_mismatch_fails_closed", r.get(4) == [0, 1] and untouched, f"{r}")

    # ---- header/layout mismatch: ask for the MLA file as group 6 ----
    r = run_job(w0, spill.SpillMeta(loads={5: [(6, 1, names[0])]}))
    check("wrong_group_fails", r.get(5) == [0, 1])
    # ---- rank mismatch: copy rank1's file into rank0's name ----
    rd1 = os.path.join(args.dir, spill.SUBDIR, "rank1")
    import shutil
    shutil.copyfile(os.path.join(rd1, names[3]), os.path.join(rd, "swapped.blk"))
    r = run_job(w0, spill.SpillMeta(loads={6: [(3, 9, "swapped.blk")]}))
    check("foreign_rank_file_fails", r.get(6) == [0, 1])
    # ---- truncated file (a partial write that got renamed) ----
    p = os.path.join(rd, names[4])
    os.truncate(p, os.path.getsize(p) - 4096)
    r = run_job(w0, spill.SpillMeta(loads={7: [(4, 10, names[4])]}))
    check("truncated_fails", r.get(7) == [0, 1])
    # ---- missing file ----
    r = run_job(w0, spill.SpillMeta(loads={8: [(5, 11, "nope.blk")]}))
    check("missing_fails", r.get(8) == [0, 1])
    # ---- one rank fails -> the aggregated job fails ----
    os.unlink(os.path.join(rd1, names[6]))
    ra = run_job(w0, spill.SpillMeta(loads={9: [(6, 1, names[6])]}))
    rb = run_job(w1, spill.SpillMeta(loads={9: [(6, 1, names[6])]}))
    agg = spill.SpillWorkerMeta(done=ra).aggregate(spill.SpillWorkerMeta(done=rb))
    check("one_rank_failure_fails_job", agg.done.get(9) == [1, 1], f"{agg.done}")

    # ---- free-disk floor ----
    wf, _ = make_worker(spill, args.dir, 5, kvc, args.device, seed=3, min_free=1 << 60)
    r = run_job(wf, spill.SpillMeta(saves={10: [(0, 2, "floor.blk")]}))
    check("free_disk_floor_fails", r.get(10) == [0, 1] and not os.path.exists(
        os.path.join(args.dir, spill.SUBDIR, "rank5", "floor.blk")))
    wf.shutdown()

    # ---- delete ----
    w0.handle(spill.SpillMeta(deletes=[names[0]]))
    w0.drain()
    time.sleep(0.05)
    check("delete", not os.path.exists(os.path.join(rd, names[0])))

    # ---- real ENOSPC on a small tmpfs ----
    if args.full_dir:
        wfull, _ = make_worker(spill, args.full_dir, 0, kvc, args.device, seed=4)
        r = run_job(wfull, spill.SpillMeta(saves={11: [(0, 1, "a.blk"), (0, 2, "b.blk"), (0, 3, "c.blk")]}))
        fd = os.path.join(args.full_dir, spill.SUBDIR, "rank0")
        left = os.listdir(fd)
        check("enospc_fails_and_cleans", r.get(11) == [0, 1] and not left, f"{r} left={left}")
        wfull.shutdown()

    # ---- a write that fails half-way (file-size rlimit -> EFBIG on the 2nd block): job fails, nothing left ----
    import resource
    import signal
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (20 << 20, hard))
    try:
        r = run_job(w0, spill.SpillMeta(saves={12: [(6, 2, "half_a.blk"), (0, 3, "half_b.blk")]}))
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    left = [n for n in os.listdir(rd) if n.startswith("half_")]
    check("write_fails_midway_cleans", r.get(12) == [0, 1] and not left, f"{r} left={left}")

    w0.shutdown()
    w1.shutdown()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"SUMMARY {len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
