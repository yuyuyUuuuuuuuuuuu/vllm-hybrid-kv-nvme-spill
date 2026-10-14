# Reproducing

## Hardware and software we used

| item | value |
|---|---|
| production | 2 x NVIDIA DGX Spark class machines (GB10, 128 GB unified memory each), tensor parallelism 2 over ConnectX-7 RoCE; head node = rank 0, worker node = rank 1; spill directory on each node's internal NVMe (same absolute path on both) |
| tests | a separate machine with one GB10 and its internal NVMe: CPU-only containers for the unit tests and the replay rig, one GPU container at a time for the GPU tests |
| image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` (`sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d`), vLLM `0.1.dev20051+g487ecf187` at `/usr/local/lib/python3.12/dist-packages/vllm` |
| serving setup | the GLM-5.3-Flash EXL3 launcher with our overlays (deploy kit r16z8p), described in the companion repository [glm-5.3-flash-dgx-spark-vllm-optimization](https://github.com/yuyuyUuuuuuuuuuuu/glm-5.3-flash-dgx-spark-vllm-optimization) |
| model | GLM-5.3-Flash EXL3 4 bpw + DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2` @ dc77ff1c) |

Everything below runs from a checkout of this repository. Outputs go to `test-out/`, `rig-out/` and `rig-scratch/`
(git-ignored), never into `results/`.

## 1. Inputs you need

1. **The production vLLM tree** (for the CPU tests and the rig): the image's `vllm` package *after* the launcher's
   start-up overlays have patched it. The overlays patch site-packages inside the container when it starts, so start
   the serving container once with the companion repository's kit and copy the package out, Python files only:

   ```sh
   mkdir -p prod_src && docker cp glm53-exl3-head:/usr/local/lib/python3.12/dist-packages/vllm prod_src/vllm
   find prod_src/vllm -type f ! -name '*.py' -delete
   # our tree: 2,423 .py files; fingerprint
   (cd prod_src && find vllm -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum)
   # -> b6c8f7fd356af10ac990f6277e24e7b8ac7061ef9875e0202b240bd21578f765
   ```

   A different overlay set gives a different fingerprint; the scheduler then behaves differently and the rig numbers
   will not match ours exactly (the connector itself does not depend on the overlays beyond what `docs/limitations.md`
   lists). Then build the rig's work tree (a copy with the connector installed by `code/patch_nvme_spill.py`):

   ```sh
   PROD_SRC=$PWD/prod_src/vllm tests/rig/setup_work.sh      # -> tests/rig/work/vllm
   ```

2. **Model config directories** (for the two tests that build a `VllmConfig` and for the rig): a directory
   `$MODELS` with `GLM-5.3-Flash-EXL3-TR3-4bpw-partial/` (`config.json`, `model.safetensors.index.json`) and
   `GLM-5.3-Flash-DFlash2-dc77ff1c/` (`config.json`). No weights are read. (The TR3 weight repository was the one in
   production when the rig was built; we expect the `config.json` of the newer EXL3 TensorFold weights to work too, but
   did not test it.)

3. **A launcher kit** for `tests/test_start_sh.sh`: the companion repository's `code/kit/` is enough for part 1;
   part 2 also needs a deployed launcher tree (its `start.sh`, `overlay/` and `env_nonsecret.txt` = the non-secret lines
   of its `.env`).

## 2. CPU unit tests (serving image, CPU only, one container at a time, 4 GiB; `u4_env` 6 GiB)

```sh
RIG_MODELS_DIR=$MODELS tests/run_cpu_tests.sh            # all; or name tests: store prune ttl_persist finish ...
tests/test_start_sh.sh <scratch dir> <kit dir> [<launcher dir>]
```

### 2.1 This update (code = u5, re-run from this tree on 2026-10-14, outputs in `results/published-tree-u5/`)

| test | what | from this tree | recorded in development |
|---|---|---|---|
| `tests/test_store.py --device cpu` | worker byte path (unchanged since the final version) | **19/19** | 18/18 without the 40 MB tmpfs case (`results/u5/unit/test_store.txt`) |
| `tests/test_sched_faults.py` | deadlines, quarantine, breaker, pin cap, B6, B7, worker exception safety | **49/49** | 49/49 (`results/u5/unit/`) |
| `tests/test_prune.py` | `KEEP_PER_CHAIN` pruning, LRU at the cap | **74/74** | 74/74 |
| `tests/test_ttl_persist.py` | TTL, epochs, index, adoption, restart round trip on two ranks | **133/133** | 133/133 |
| `tests/t_hash_stability.py` | block hashes across processes with / without the connector | **9/9** | 9/9 |
| `tests/t_alloc_conf.py` | B1: this connector accepted under `expandable_segments:True`, `NixlConnector` still rejected | **3/3 lines as expected** | same |
| `tests/test_finish.py` | B8 policy gate vs the real policy class, restore gating, NVMe-backed-first blocks on the real `BlockPool`, `DEFAULT_TTL_S`, the "code" identity part | **66/66** | 66/66 |
| `tests/test_u4_env.py` | run-time generated env values (two fresh interpreters + `enable_envs_cache()`), audit, exact-name exclusion, per-name breakdown, names in wipe reasons | **50/50** (the 2 checks that compare against the previous connector need `--old-module <finish-release connector>`, which is not shipped) | 52/52 with `--old-module` |
| `tests/test_u5_index.py` | the resume point confirmed by the last step is indexed after an idle period / by the timer / at shutdown; lock; 2 ranks + SIGKILL restart | **25/25** (`--old` needs the u4 connector, not shipped) | 25/25; 2/2 reproduce the bug with `--old` on the u4 connector |
| `tests/test_quality_probe.py` (host Python) | B5 + `--runs N` (averaged verdict, restore proven in every run) | **22/22** | 22/22 (`results/finish/unit/test_quality_probe.txt`) |
| `tests/test_start_sh.sh`, r16z8p kit + launcher | apply, idempotence, fail-closed, every knob reaches both ranks' `docker run` | **15/15** (32 forwarded names; the generated diff equals `code/start.sh.nvmespill.patch` apart from its two header lines) | 15/15 |
| `tests/test_start_sh.sh`, companion repository's public kit, part 1 | same, on the public `start.sh` | **7/7** (part 2 skipped: no launcher tree) | - |

All of these ran on the final bytes of `code/`. Every `.py` file passes `python3 -m py_compile`, every `.sh` file passes `bash -n` (counts in section 6). `tests/bench_persist.py`
also ran (`results/published-tree-u5/bench_persist.txt`). The GPU tests (section 3) were not re-run for this update: the
worker's byte path they exercise did not change (the changes are scheduler-side Python), and their results from the
first publication stand.
The aggregates of `results/finish/`, `results/u4/` and `results/u5/` regenerate byte-identically from this tree:
`python3 tests/aggregate_finish.py`, `python3 tests/compare_u4.py`, `python3 tests/compare_u5.py`.

### 2.2 First publication (code = published version)

| test | what | from this tree (2026-10-14) | recorded in development |
|---|---|---|---|
| `tests/test_store.py --device cpu` | worker byte path on production page sizes, CRC / header / truncation / ENOSPC / free-space floor | **19/19** | 19/19 (`results/test_store_cpu.txt`) |
| `tests/test_sched_faults.py` | deadlines, quarantine, breaker, pin cap, B6 ordering, B7, worker exception safety | **49/49** | 49/49 (`results/ttl/unit_test_sched_faults.txt`) |
| `tests/test_prune.py` | `KEEP_PER_CHAIN` pruning, LRU at the cap | **74/74** | 74/74 (`results/ttl/unit_test_prune.txt`) |
| `tests/test_ttl_persist.py` | TTL parsing / eviction order / epochs, index round trip / corruption / mismatch, start-up scan, restart round trip on two ranks | **133/133** | 133/133 (`results/ttl/unit_test_ttl_persist.txt`) |
| `tests/t_hash_stability.py` | block hashes across processes with / without the connector (no `PYTHONHASHSEED`) | **9/9** | 9/9 (`results/ttl/t_hash_stability.txt`) |
| `tests/t_alloc_conf.py` | B1: `VllmConfig` with this connector accepted under `expandable_segments:True`, `NixlConnector` still rejected | **3/3 lines as expected** | same (`docs/review-findings.md`) |
| `tests/test_quality_probe.py` (host Python) | B5: the probe fails unless a restore is proven | **13/13** | 13/13 (`results/ttl/unit_test_quality_probe.txt`) |
| `tests/test_start_sh.sh`, r16z8p kit + launcher | `apply_start_sh.py`: apply, idempotence, fail-closed, every knob reaches both ranks' `docker run` | **15/15** (31 forwarded names; the generated diff equals `code/start.sh.nvmespill.patch`) | 15/15 (`results/ttl/unit_test_start_sh.txt`) |
| `tests/test_start_sh.sh`, companion repository's public kit, part 1 | same, on the public `start.sh` | **7/7** (part 2 skipped: no launcher tree) | - |
| `tests/bench_persist.py` (60 RPs, 6,000 files) | index build / write / read, adoption decision, rank scan | ran (`results/published-tree/bench_persist.txt`) | `results/ttl/bench_persist*.json` |

All outputs of the first publication: `results/published-tree/*.txt` (paths masked).

## 3. GPU tests (one GB10, one container at a time)

```sh
GPU_TEST_MEM=8g tests/run_gpu_tests.sh <scratch dir on the NVMe> store restart [bench]
```

| test | from this tree (container memory 8 GiB, torch CUDA allocations capped at 8 GiB) | recorded in development |
|---|---|---|
| `test_store.py --device cuda` (pinned staging, CUDA streams, O_DIRECT) | **18/18** | 18/18 (`results/test_store_gpu.txt`, `results/ttl/test_store_gpu.txt`) |
| `t_expandable_gpu.py` (B1: KV in expandable segments survives allocator churn, bytes identical) | **9/9** | 9/9 (`results/fix/gpu_final.txt`) |
| `t_restart_gpu.py` (save in process A, adopt + restore in process B, bytes identical to the saved bytes and to the in-process restore) | **1/1 + 4/4** | same (`results/ttl/gpu_restart_roundtrip.txt`) |
| `bench_gpu.py` (600k-token resume point, 3.72 GB; bandwidth kernel alongside) | **not re-run from this tree**: it needs about 10 GB of GPU memory and stopped with a CUDA OOM under our 8 GiB per-process cap | `results/bench_gpu.json` (initial version), `results/fix/gpu_final.txt` (fix release) |

## 4. Replay rig (CPU)

See `tests/rig/README.md` for what is real and what is simulated.

```sh
RIG_MODELS_DIR=$MODELS RIG_PAR=1 tests/rig/run_jobs.sh tests/rig/jobs_ttl_final.txt rig-out/done
python3 tests/aggregate_ttl.py rig-out          # or aggregate.py / aggregate_prune.py / summarize_faults.py
```

**This update:** the work tree was rebuilt from the same production vLLM tree with `tests/rig/setup_work.sh` (identical
to our development work tree except the connector's comments), then five 0.5 h jobs were re-run from this tree (57-147 s
each, one 8 GiB container at a time): `u5_pr_idle_crash_s2` (idle, then SIGKILL restart), `u5_pr_idle_clean_s2`,
`u5_c_on_s2`, `u5_pr_clean_s2` and the asynchronous `ff_hv_r1_s4`. The four synchronous jobs equal the recorded runs
in `results/u5/` on every field except timings; the asynchronous one differs as asynchronous runs do (thread timing)
and has the same correctness result (0 unfinished, 0 violations, 0 content mismatches, all blocks free at the end):
`results/published-tree-u5/rig_rerun_vs_recorded.txt`, outputs in `results/published-tree-u5/rig/`. Job lists for the new
results: `tests/rig/jobs_finish_final.txt`, `jobs_u4.txt`, `jobs_u5.txt` (`*_u4code` jobs need the u4 connector's work
tree, not shipped); summaries: `tests/aggregate_finish.py`, `tests/compare_u4.py`, `tests/compare_u5.py`.

**First publication:**

From the first publication's tree we re-ran four 0.5 h jobs of the final-version fault suite (`c_on_s2`, `pr_clean_s2` (restart with
reuse), `f_crc_s3`, `n_noload_s2`; 88-101 s each). Every field of their summary JSON equals the recorded run in
`results/ttl/final_faults/` except timing fields and, in `pr_clean_s2`, the number of index writes before the restart
(9 vs 10; the index writer thread coalesces writes): `results/published-tree/rig_rerun_vs_recorded.txt`, outputs in
`results/published-tree/rig/`. The 4 h matrices (hours per job, two 8 GiB containers at a time in our runs) were not
re-run from this tree; the `*_oldcode*` jobs need the pre-fix connector, which is not shipped.

## 5. Which code produced which results

| results | connector version |
|---|---|
| `results/*.json`, `results/aggregate.txt`, `results/bench_gpu.json`, `results/test_store_*.txt` | initial version |
| `results/fix/` | fix release (production sha256 `e8c218eb...`) |
| `results/prune/` | prune release (production sha256 `d030f132...`) |
| `results/ttl/*.json`, `results/ttl/aggregate.txt` | TTL+persist first cut |
| `results/ttl/final/`, `results/ttl/final_faults/`, `results/ttl/*.txt` | final version (production sha256 `1762e23f...`) |
| `results/published-tree/` | the first publication's `code/` (published version) |
| `results/finish/` (`gate_only/` and `g1_*`: intermediate connectors, not shipped), `results/finish/unit/` | finish release (production sha256 `eb2e1084...`) |
| `results/u4/` (`*.json`, `unit/`) | u4 (production sha256 `bd0f589e...`) |
| `results/u5/` (`*_u4code*.json`: u4) | u5 (production sha256 `073a2e8c...`) |
| `results/published-tree-u5/` | this repository's `code/` |

`code/glm53_nvme_spill_connector.py` has the same syntax tree (docstrings aside) as the u5 file running in production,
and the same restart identity part "code"; `results/provenance_connector.txt` shows the exact differences and hashes.

## 6. Syntax checks of this tree

All 35 `.py` files of this tree pass `python3 -m py_compile` and all 7 `.sh` files pass `bash -n` (checked after this
update).
