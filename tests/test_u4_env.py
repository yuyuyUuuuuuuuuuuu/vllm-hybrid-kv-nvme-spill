# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests of the u4 change (2026-10-14): the reuse-across-restarts identity must not depend on env values a process
generates for itself, and a mismatch must name what differs.

A production restart on 2026-10-14 (connector eb2e1084): "persist: wipe every rank (rank signature differs (rank 0: env; rank 1:
env))" although nothing was changed. vLLM's envs.py defines VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME with
get_env_or_set_default(..., lambda: f"VLLM_OBJECT_STORAGE_SHM_BUFFER_{uuid.uuid4().hex}"): the first read WRITES a random
value into os.environ, and enable_envs_cache() (end of EngineCore.__init__ and of WorkerProc.__init__) reads every vLLM
env var. The connector's env digest hashed every VLLM_* item -> a different digest in every process at every start.

  fresh   two fresh interpreters each run the production trigger (vllm.envs.enable_envs_cache()) and compute the
          connector's env digest: with --old-module (eb2e1084) they differ (the bug), with the module under test they
          are equal; a control without the trigger is equal for both. Static: the trigger and the generator are in the
          production tree (rig work copy).
  audit   two fresh interpreters import the production modules a scheduler / worker imports, run the trigger, and
          report every env item under _ENV_PREFIXES that appeared or changed; any item whose value differs between the
          two processes must be in _ENV_VOLATILE (otherwise: an unknown volatile name -> FAIL, name printed).
  unit    exclusion by exact name only (everything else under the prefixes stays strict; secret-named variables stay
          out, also of the breakdown); the breakdown holds names + 8-hex digests only, never values; _detail_diff /
          _explain wording.
  sched   real SpillScheduler + the real _identity: an unknown variable that differs between runs -> wipe, its NAME in
          the reason (changed / added / removed); the volatile names differing -> adopted; the index carries
          sched_detail / rank_detail and no env value; an index without breakdown (eb2e1084) -> "no per-name record".
  rank    real SpillWorker with the real _identity: the hello carries the breakdown (not signed), the volatile names
          do not change the rank signature, another variable does; a rank mismatch names the variable per rank.

usage (production image, PYTHONPATH = rig work tree, PYTHONHASHSEED set):
  python3 test_u4_env.py --dir <scratch> [--module <connector .py>] [--old-module <eb2e1084 copy>] [--only a,b]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def load(path: str, name: str = "spill_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ fresh processes
AUDIT_MODULES = ("vllm.v1.engine.core", "vllm.v1.executor.multiproc_executor", "vllm.v1.core.sched.scheduler",
                 "vllm.v1.worker.gpu_worker", "vllm.v1.worker.gpu_model_runner", "vllm.config",
                 "vllm.distributed.kv_transfer.kv_connector.v1.glm53_nvme_spill_connector",
                 "vllm.multimodal.cache", "exllamav3", "flashinfer", "triton", "tilelang", "deep_gemm", "deep_ep")


def child(module: str, trigger: bool, audit: bool) -> None:
    before = dict(os.environ)
    imported = {}
    if audit:
        import importlib
        for m in AUDIT_MODULES:
            try:
                importlib.import_module(m)
                imported[m] = "ok"
            except BaseException as e:  # noqa: BLE001 - CPU container: some GPU-only imports fail; report them
                imported[m] = type(e).__name__
    spill = load(module)
    if trigger:
        import vllm.envs as envs
        envs.enable_envs_cache()
    pfx = spill._ENV_PREFIXES
    seen = {k: v for k, v in os.environ.items() if k.startswith(pfx)}
    changed = {k: v for k, v in seen.items() if before.get(k) != v}
    print("RESULT " + json.dumps({"digest": spill._env_digest(), "has_shm": "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME" in os.environ,
                                  "changed": changed, "imported": imported,
                                  "volatile": sorted(getattr(spill, "_ENV_VOLATILE", ()))}), flush=True)


def run_child(module: str, trigger: bool, audit: bool = False) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"}
    args = [sys.executable, os.path.abspath(__file__), "--child", "--module", module]
    if trigger:
        args.append("--trigger")
    if audit:
        args.append("--audit")
    out = subprocess.run(args, env=env, capture_output=True, text=True, timeout=900)
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
    if not line:
        raise RuntimeError(f"child failed rc={out.returncode}: {out.stderr[-1500:]}")
    return json.loads(line[-1][7:])


def t_fresh(spill, a):
    tree = os.path.dirname(os.path.abspath(sys.modules["vllm"].__file__))
    envs_src = open(os.path.join(tree, "envs.py")).read()
    check("fresh_static_generator_in_prod_tree",
          re.search(r'"VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": get_env_or_set_default\(\s*"VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",'
                    r'\s*lambda: f"VLLM_OBJECT_STORAGE_SHM_BUFFER_\{uuid\.uuid4\(\)\.hex\}"', envs_src) is not None
          and "os.environ[env_name] = default_value" in envs_src, tree)
    for rel in ("v1/engine/core.py", "v1/executor/multiproc_executor.py"):
        check(f"fresh_static_trigger_{rel.split('/')[-1]}", "enable_envs_cache()" in open(os.path.join(tree, rel)).read())
    cases = [("new", a.module)] + ([("old", a.old_module)] if a.old_module else [])
    for tag, mod in cases:
        x, y = run_child(mod, True), run_child(mod, True)
        if tag == "old":
            check("fresh_old_connector_digest_differs_between_processes (the production bug)",
                  x["digest"] != y["digest"] and x["has_shm"] and y["has_shm"], f"{x['digest']} {y['digest']}")
        else:
            check("fresh_new_connector_digest_equal_between_processes", x["digest"] == y["digest"] and x["has_shm"],
                  f"{x['digest']} {y['digest']}")
        c1, c2 = run_child(mod, False), run_child(mod, False)
        check(f"fresh_{tag}_control_without_trigger_equal", c1["digest"] == c2["digest"] and not c1["has_shm"],
              f"{c1['digest']} {c2['digest']}")


def t_audit(spill, a):
    x, y = run_child(a.module, True, audit=True), run_child(a.module, True, audit=True)
    print(f"     imported: {x['imported']}")
    print(f"     env items under the prefixes written at run time (names): {sorted(x['changed'])}")
    vol = set(spill._ENV_VOLATILE)
    names = set(x["changed"]) | set(y["changed"])
    differ = sorted(k for k in names if x["changed"].get(k) != y["changed"].get(k))
    unknown = [k for k in differ if k not in vol]
    check("audit_every_item_differing_between_processes_is_excluded", not unknown,
          f"differ={differ} unknown={unknown}")
    check("audit_new_digest_equal_after_imports", x["digest"] == y["digest"], f"{x['digest']} {y['digest']}")


# ------------------------------------------------------------------------------------------------ unit
def t_unit(spill, a):
    keep = dict(os.environ)
    try:
        base = spill._env_digest()
        for k in sorted(spill._ENV_VOLATILE):
            os.environ[k] = "v-" + uuid.uuid4().hex
        check("unit_volatile_names_do_not_change_digest", spill._env_digest() == base)
        check("unit_volatile_exactly_two_names", sorted(spill._ENV_VOLATILE) ==
              ["VLLM_DIST_IDENT", "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"], f"{sorted(spill._ENV_VOLATILE)}")
        # exact names only: a near name is still strict
        for k in ("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAMEX", "VLLM_OBJECT_STORAGE_SHM_BUFFER", "VLLM_DIST_IDENT2",
                  "VLLM_SOMETHING_NEW_U4", "GLM53_KPOOL_TEST_FLAG", "EXL3_U4_TEST", "TF_EXL3_U4", "MAX_MODEL_LEN"):
            d0 = spill._env_digest()
            os.environ[k] = "1" if os.environ.get(k) != "1" else "2"
            check(f"unit_strict_{k}", spill._env_digest() != d0)
        d0 = spill._env_digest()
        os.environ["VLLM_API_KEY"] = "sekrit-u4-value"
        os.environ["GLM53_NVME_SPILL_GIB"] = "77"
        check("unit_secret_and_own_knobs_still_ignored", spill._env_digest() == d0)
        os.environ["GLM53_U4_MARKER"] = "zz-u4-marker-value"
        dig, det = spill._env_part()
        names = [k for k, _ in spill._env_items()]
        check("unit_env_part_digest_equals_env_digest", dig == spill._env_digest())
        check("unit_breakdown_names_equal_digest_names", sorted(det) == names)
        check("unit_breakdown_no_secret_volatile_or_own_names",
              "VLLM_API_KEY" not in det and "GLM53_NVME_SPILL_GIB" not in det
              and not (set(det) & set(spill._ENV_VOLATILE)))
        check("unit_breakdown_values_are_8hex_not_values",
              all(re.fullmatch(r"[0-9a-f]{8}", v) for v in det.values())
              and "zz-u4-marker-value" not in json.dumps(det))
    finally:
        os.environ.clear()
        os.environ.update(keep)
    dd = spill._detail_diff
    check("unit_diff_changed_added_removed",
          dd({"A": "1", "B": "2", "C": "3"}, {"A": "1", "B": "9", "D": "4"}) == "changed B / added D / removed C",
          dd({"A": "1", "B": "2", "C": "3"}, {"A": "1", "B": "9", "D": "4"}))
    check("unit_diff_same", dd({"A": "1"}, {"A": "1"}) == "same names and short digests")
    check("unit_diff_no_index_record", dd(None, {"A": "1"}) == "no per-name record in the index")
    check("unit_diff_no_run_record", dd({"A": "1"}, None) == "no per-name record in this run")
    big = dd({"Z": "0"}, {"Z": "0", **{f"N{i:02d}": "x" for i in range(20)}})
    check("unit_diff_limit", big.startswith("added N00,N01") and big.endswith("(+4 more)"), big)
    check("unit_explain", spill._explain(["code", "env"], {"env": {"X": "1"}}, {"env": {"X": "2"}}) ==
          "code, env [changed X]", spill._explain(["code", "env"], {"env": {"X": "1"}}, {"env": {"X": "2"}}))
    ident = spill._identity(SimpleNamespace(), SimpleNamespace(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[]))
    det = ident.get(spill._DETAIL)
    check("unit_identity_has_breakdown_key", isinstance(det, dict) and {"env", "python", "site"} <= set(det),
          f"{sorted(det) if isinstance(det, dict) else det}")
    check("unit_identity_parts_are_strings", all(isinstance(v, str) for k, v in ident.items() if k != spill._DETAIL))
    check("unit_identity_site_breakdown_has_vllm_groups", any(k.startswith("vllm/") for k in det.get("site", {})),
          f"{len(det.get('site', {}))} groups")
    # u4 introduced finish.2; u5 (idle index write) is finish.3 -- any finish.N >= 2 carries the u4 identity rules
    check("unit_version_finish2", ident["version"].startswith(("2026-10-14.finish.2/", "2026-10-14.finish.3/")),
          ident["version"])


# ------------------------------------------------------------------------------------------------ scheduler
def t_sched(spill, a):
    import test_ttl_persist as tp
    root = os.path.join(a.dir, "sched")
    w = tp.Wall()
    keep = dict(os.environ)

    def run2(name, env1, env2, files_fn=tp.disk_files):
        os.environ.clear(); os.environ.update(keep); os.environ.update(env1)     # noqa: E702
        s1, _ = tp.first_run(spill, f"{root}/{name}", w)
        files = files_fn(s1)
        os.environ.clear(); os.environ.update(keep)                              # noqa: E702
        for k in env1:
            os.environ.pop(k, None)
        os.environ.update(env2)
        w.t += 60
        s2, *_r, m1, m2 = tp.boot(spill, f"{root}/{name}", w, (files, files))
        return s1, s2, (m1.wipe or m2.wipe)

    try:
        s1, s2, wiped = run2("vol", {"VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": "VLLM_OBJECT_STORAGE_SHM_BUFFER_" + uuid.uuid4().hex,
                                     "VLLM_DIST_IDENT": "a1"},
                             {"VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": "VLLM_OBJECT_STORAGE_SHM_BUFFER_" + uuid.uuid4().hex,
                              "VLLM_DIST_IDENT": "b2"})
        check("sched_volatile_names_differ_still_adopts", not wiped and s2.persist_reason == "adopted"
              and len(s2.rps) == len(s1.rps) > 0, s2.persist_reason)
        s1, s2, wiped = run2("chg", {"VLLM_U4_UNKNOWN_VOLATILE": uuid.uuid4().hex},
                             {"VLLM_U4_UNKNOWN_VOLATILE": uuid.uuid4().hex})
        check("sched_unknown_volatile_wipes_and_is_named", wiped and not s2.rps
              and "scheduler signature differs: env [changed VLLM_U4_UNKNOWN_VOLATILE]" in s2.persist_reason,
              s2.persist_reason)
        s1, s2, wiped = run2("add", {}, {"GLM53_U4_NEW_FLAG": "1"})
        check("sched_added_var_named", wiped and "env [added GLM53_U4_NEW_FLAG]" in s2.persist_reason, s2.persist_reason)
        s1, s2, wiped = run2("rem", {"EXL3_U4_GONE": "1"}, {})
        check("sched_removed_var_named", wiped and "env [removed EXL3_U4_GONE]" in s2.persist_reason, s2.persist_reason)
        # the index carries the breakdown, never a value (not the marker, not the secret)
        os.environ.clear(); os.environ.update(keep)                              # noqa: E702
        os.environ.update({"GLM53_U4_MARKER": "zz-u4-marker-value", "VLLM_API_KEY": "sekrit-u4-value"})
        s1, _ = tp.first_run(spill, f"{root}/idx", w)
        raw = open(s1.cfg.index_path(), "rb").read()
        body = json.loads(raw[raw.index(b"\n") + 1:])
        check("sched_index_has_sched_detail_env", "GLM53_U4_MARKER" in (body.get("sched_detail") or {}).get("env", {}),
              f"{sorted((body.get('sched_detail') or {}).keys())}")
        check("sched_index_has_rank_detail_slots", sorted((body.get("rank_detail") or {}).keys()) == ["0", "1"])
        check("sched_index_holds_no_env_value", b"zz-u4-marker-value" not in raw and b"sekrit-u4-value" not in raw
              and b"VLLM_API_KEY" not in raw)
        # an index without breakdown (written by eb2e1084) + an env change -> wipe, says so
        os.environ.clear(); os.environ.update(keep)                              # noqa: E702
        s1, _ = tp.first_run(spill, f"{root}/old", w)
        files = tp.disk_files(s1)
        p = s1.cfg.index_path()
        raw = open(p, "rb").read()
        body = json.loads(raw[raw.index(b"\n") + 1:])
        body.pop("sched_detail", None)
        body.pop("rank_detail", None)
        nb = json.dumps(body, separators=(",", ":")).encode()
        import hashlib
        top = json.dumps({"magic": spill.INDEX_MAGIC, "version": spill.INDEX_VERSION, "len": len(nb),
                          "sha256": hashlib.sha256(nb).hexdigest()}).encode()
        open(p, "wb").write(top + b"\n" + nb)
        os.environ["VLLM_U4_X"] = "1"
        s2, *_r, m1, m2 = tp.boot(spill, f"{root}/old", w, (files, files))
        check("sched_old_index_without_breakdown_says_so", (m1.wipe or m2.wipe)
              and "env [no per-name record in the index]" in s2.persist_reason, s2.persist_reason)
    finally:
        os.environ.clear()
        os.environ.update(keep)


# ------------------------------------------------------------------------------------------------ rank
def t_rank(spill, a):
    import prodlayout
    import test_ttl_persist as tp
    import torch
    keep = dict(os.environ)
    root = os.path.join(a.dir, "rank")

    def worker(sub, rank=0):
        cfg = spill.SpillConfig(root=f"{root}/{sub}", cap_bytes=8 << 30, min_free_bytes=0, direct=False,
                                log_every_s=0, persist=True)
        kvc = prodlayout.build_kv_cache_config(16, scale=576)
        wk = spill.SpillWorker(cfg, kvc, rank)
        raws = [torch.zeros(kvt.size, dtype=torch.uint8) for kvt in kvc.kv_cache_tensors]
        views = {n: raws[t] for t, kvt in enumerate(kvc.kv_cache_tensors) for n in kvt.shared_by}
        vc = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))
        wk.register(views, sig="layoutsig", ident_fn=lambda: spill._identity(vc, kvc))
        h = tp.take_hello(wk, 120)
        wk.shutdown()
        return h

    try:
        os.environ["VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"] = "VLLM_OBJECT_STORAGE_SHM_BUFFER_" + uuid.uuid4().hex
        h1 = worker("a")
        os.environ["VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME"] = "VLLM_OBJECT_STORAGE_SHM_BUFFER_" + uuid.uuid4().hex
        h2 = worker("a")
        check("rank_hello_carries_breakdown_not_in_parts", isinstance(h1.get("detail"), dict)
              and "env" in h1["detail"] and spill._DETAIL not in h1["parts"], f"{sorted(h1.get('detail') or {})}")
        check("rank_volatile_name_does_not_change_rank_sig", h1["sig"] == h2["sig"] and h1["sig"])
        os.environ["VLLM_U4_RANK_FLAG"] = "1"
        h3 = worker("a")
        check("rank_other_var_changes_rank_sig", h3["sig"] != h1["sig"])
        # scheduler decision with these real rank reports: rank 1 differs by VLLM_U4_RANK_FLAG -> named per rank
        w = tp.Wall()
        s1, *_r = tp.boot(spill, f"{root}/dec", w, send_hello=False)
        s1.update_connector_output(spill.SpillWorkerMeta(hello={0: dict(h1, rank=0, files={}),
                                                               1: dict(h1, rank=1, files={})}))
        s1.build_meta()
        s1.shutdown()
        s2, *_r = tp.boot(spill, f"{root}/dec", w, send_hello=False)
        s2.update_connector_output(spill.SpillWorkerMeta(hello={0: dict(h2, rank=0, files={}),
                                                               1: dict(h3, rank=1, files={})}))
        s2.build_meta()
        check("rank_mismatch_names_var_per_rank", s2.persist_reason.startswith("rank signature differs (rank 1: env "
                                                                                "[added VLLM_U4_RANK_FLAG])"),
              s2.persist_reason)
    finally:
        os.environ.clear()
        os.environ.update(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/u4")
    ap.add_argument("--module", default=os.path.join(HERE, "..", "code", "glm53_nvme_spill_connector.py"))
    ap.add_argument("--old-module", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--trigger", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    a.module = os.path.abspath(a.module)
    if a.child:
        child(a.module, a.trigger, a.audit)
        return 0
    assert os.environ.get("PYTHONHASHSEED"), "run with PYTHONHASHSEED set"
    spill = load(a.module)
    import test_sched_faults as T
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    init_none_hash(sha256)
    a.dir = os.path.abspath(a.dir)
    shutil.rmtree(a.dir, ignore_errors=True)
    os.makedirs(a.dir, exist_ok=True)
    for name, fn in (("unit", t_unit), ("sched", t_sched), ("rank", t_rank), ("fresh", t_fresh), ("audit", t_audit)):
        if a.only and name not in a.only.split(","):
            continue
        try:
            fn(spill, a)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{name}_raised", False, repr(e))
    res = RESULTS + T.RESULTS
    n_fail = sum(1 for _, ok, _ in res if not ok)
    print(f"SUMMARY {len(res) - n_fail}/{len(res)} passed", flush=True)
    shutil.rmtree(a.dir, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
