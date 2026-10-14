# SPDX-License-Identifier: AGPL-3.0-only
"""B1 evidence on the GPU: the spill worker under the kit's PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

The KV tensors are allocated by the caching allocator inside expandable segments; the worker registers its torch views;
the allocator is then churned hard (large allocations freed and torch.cuda.empty_cache(), which unmaps the freed
physical pages of expandable segments and maps new ones on the next growth). Checks: the KV tensors keep their address,
blocks saved before the churn load back byte-identical after it, blocks saved after it give byte-identical files,
and no KV byte outside the load targets changed.
usage (GPU container, env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True): python3 t_expandable_gpu.py <dir>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import torch  # noqa: E402

import glm53_nvme_spill_connector as spill  # noqa: E402
import prodlayout  # noqa: E402

R = []


def check(name, ok, d=""):
    R.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name} {d}", flush=True)


root = sys.argv[1]
check("env_expandable", "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
      os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
torch.cuda.set_device(0)
NB = 24
kvc = prodlayout.build_kv_cache_config(NB)
g = torch.Generator(device="cuda").manual_seed(7)
raws = [torch.randint(0, 256, (kvt.size,), dtype=torch.uint8, device="cuda", generator=g) for kvt in kvc.kv_cache_tensors]
segs = torch.cuda.memory_snapshot()
check("kv_in_expandable_segments", any(s.get("is_expandable") for s in segs),
      f"segments={len(segs)} expandable={sum(1 for s in segs if s.get('is_expandable'))}")
ptrs = [r.data_ptr() for r in raws]
views = {n: raws[t] for t, kvt in enumerate(kvc.kv_cache_tensors) for n in kvt.shared_by}
cfg = spill.SpillConfig(root=root, cap_bytes=64 << 30, min_free_bytes=5 << 30, log_every_s=0)
w = spill.SpillWorker(cfg, kvc, 0)
w.register(views, sig="exp")
pages, rows = spill.group_rows(kvc)


def rows_of(g_, b):
    return torch.cat([raws[t].view(-1, p)[b] for t, p in rows[g_]]).cpu()


def run(meta, jobs):
    w.handle(meta)
    w.drain(300)
    got = {}
    m = w.take_meta()
    if m is not None:
        got.update(m.done)
    return all(got.get(j) == [1, 0] for j in jobs), got


src = {0: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
ref = {gg: rows_of(gg, b) for gg, b in src.items()}
ok, got = run(spill.SpillMeta(saves={1: [(gg, b, f"pre_g{gg}.blk") for gg, b in src.items()]}), [1])
check("save_before_churn", ok, f"{got}")
# churn: grow/shrink the expandable segments repeatedly, release freed pages to the driver
for i in range(12):
    big = torch.empty((768 + 64 * i) << 20, dtype=torch.uint8, device="cuda")
    big.fill_(i)
    small = [torch.empty(3 << 20, dtype=torch.uint8, device="cuda") for _ in range(50)]
    del big, small
    torch.cuda.empty_cache()
torch.cuda.synchronize()
check("kv_addresses_unchanged_after_churn", [r.data_ptr() for r in raws] == ptrs)
check("source_rows_intact_after_churn", all(torch.equal(rows_of(gg, b), ref[gg]) for gg, b in src.items()))
snap = [r.clone() for r in raws]
dst = {0: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18}
for gg, b in dst.items():
    for t, p in rows[gg]:
        raws[t].view(-1, p)[b].fill_(0x5A)
        snap[t].view(-1, p)[b].fill_(0x5A)
ok, got = run(spill.SpillMeta(loads={2: [(gg, dst[gg], f"pre_g{gg}.blk") for gg in src]}), [2])
check("load_after_churn_ok", ok, f"{got}")
check("load_after_churn_byte_identical", all(torch.equal(rows_of(gg, dst[gg]), ref[gg]) for gg in src))
keep = {(t, dst[gg]) for gg in dst for t, _ in rows[gg]}
other = all(torch.equal(raws[t].view(-1, pages[t])[b], snap[t].view(-1, pages[t])[b])
            for t in range(len(raws)) for b in range(NB) if (t, b) not in keep)
check("no_other_kv_byte_changed", other)
ok, got = run(spill.SpillMeta(saves={3: [(gg, b, f"post_g{gg}.blk") for gg, b in src.items()]}), [3])
same = all(open(os.path.join(w.store.dir, f"pre_g{gg}.blk"), "rb").read()[4096:] ==
           open(os.path.join(w.store.dir, f"post_g{gg}.blk"), "rb").read()[4096:] for gg in src)
check("save_after_churn_same_payload", ok and same)
w.shutdown()
print(f"SUMMARY {sum(R)}/{len(R)} passed")
sys.exit(0 if all(R) else 1)
