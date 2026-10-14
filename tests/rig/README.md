# Replay rig (CPU)

A deterministic replay of a long-conversation workload through the **real vLLM scheduler code of the production
tree**, with the spill connector installed exactly as in production and two real `SpillWorker` ranks writing and
reading real files. No GPU and no model weights are used. Every number in `docs/measurements.md` marked "rig" comes
from this directory.

## What is real and what is simulated

Real (the production vLLM tree, i.e. the image's vLLM package after the launcher's start-up patches, copied into
`work/vllm` by `setup_work.sh`, with `code/patch_nvme_spill.py` applied):

- `Scheduler.schedule()` / `update_from_output()` in full (chunked prefill, the decode-floor prefill policy of the
  production overlays, mamba block-aligned splits, full-ISL admission, adaptive K);
- `KVCacheManager`, `HybridKVCacheCoordinator`, `BlockPool` (mamba align, prefix retention, DFlash boundary lookup);
- the request block-hash chain (sha256);
- `VllmConfig` built **with** the `kv_transfer_config`, so `__post_init__` runs every connector check production runs;
- the connector (`code/glm53_nvme_spill_connector.py`): scheduler side in the engine process, worker side as two
  `SpillWorker` ranks on CPU tensors with the production tensor structure and pages scaled down 576x; files are real.

Simulated (`sim/`):

| part | substitution | why |
|---|---|---|
| platform | `UnspecifiedPlatform`, `DeviceConfig(device="cpu")` | no GPU in the container |
| architecture check | `hf_overrides` maps the model / drafter architectures to plain classes | the model modules import GPU kernels; the scheduler never reads the architecture |
| structured output | no-op stub | no grammar requests |
| model output | fake `ModelRunnerOutput`: prefill chunks sample nothing, decode rows emit 1 + accepted draft tokens (acceptance calibrated to 2.4 tokens/step), draft proposals K=7 | no model |
| KV content | each block's bytes are a function f(primary key, rank), written when the normal path keys the block (in the step whose forward writes it), never written by a restore | restored blocks can only get their bytes from the files; every admission checks the bytes of every hit block on both ranks |
| clock | simulated: prefill 2,036 tok/s, decode step 0.5 s | |
| workload (`sim/workload.py`) | synthetic, from `random.Random(seed)`: 3 DSH conversations (430k / 600k / 780k initial tokens, a 13,824-token common head, user turns +2k..8k, outputs 0.5k..3k, 20 s..5 min between turns) plus one-shot requests (20k..100k, mean gap 88 s); options for branches (regenerate / edit), TTL fields, compaction, idle sessions, a 4th conversation, restarts | parameters fitted to production admission statistics; no production prompt, response or token id is used |

The same seed gives the same request stream in every condition, so ON/OFF differences are caused by the connector.

## Correctness checks (every run)

- (a) ref_cnt == 0 (non-null) <-> membership of the free queue, both directions; (b) queue walk length; (c) no
  negative ref_cnt (per step with `APCREREAD_STRICT=1`);
- (d) at admission, the hash of each hit MLA block equals the request's chained hash for that range
  (`hit_hash_violations`), and the content of every hit block matches f(key, rank) on both ranks;
- (e) per finished request: scheduled - rejected = prompt + max_tokens - hit (tolerance K=7);
- (f) after the drain phase (runs without `--no-drain`): all 582 blocks free, every ref_cnt 0, no connector pins.

## Running

```sh
# 1. the work tree (once): PROD_SRC = the patched vllm package directory (REPRODUCE.md explains how to get it)
PROD_SRC=/path/to/vllm tests/rig/setup_work.sh
# 2. one job:  <env file or -> <on|off> <seed> <hours> <name> [run_sim args]
printf 'strict.env on 2 0.5 c_on_s2\n' > /tmp/jobs.txt
RIG_MODELS_DIR=/path/to/models RIG_PAR=1 tests/rig/run_jobs.sh /tmp/jobs.txt /tmp/jobs.done
# results: rig-out/c_on_s2.json (+ .log, .rc)
```

`RIG_MODELS_DIR` must contain the two config directories named in `sim/configs.py`
(`GLM-5.3-Flash-EXL3-TR3-4bpw-partial`: `config.json` + `model.safetensors.index.json`;
`GLM-5.3-Flash-DFlash2-dc77ff1c`: `config.json`), or set `APCREREAD_MODEL_DIR` / `APCREREAD_DRAFTER_DIR` in an env
file. No weights are read.

- `jobs_*.txt` are the job lists behind `results/`: `jobs_main.txt` / `jobs_faults.txt` (results/fix),
  `jobs_prune*.txt` (results/prune), `jobs_ttl_*.txt` (results/ttl, results/ttl/final, results/ttl/final_faults).
- `env/*.env` select the condition: fault injection (`NVMESPILL_FAULT=missing|crc|savefail|rofs|...`), hangs,
  async completion with per-rank latency (`NVMESPILL_ASYNC`, `NVMESPILL_LAT0/1`), branch stress
  (`NVMESPILL_BRANCH`), TTL fields (`NVMESPILL_TTL`), restarts (`NVMESPILL_RESTART_AT`, `_DOWN_S`, `_CRASH`,
  `_INJECT`), connector knobs (`GLM53_NVME_SPILL_*`).
- A 4 h run needs more than 5 GiB of memory; we ran two 8 GiB containers at a time. A 0.5 h run takes 1.5 minutes.
- Jobs named `*_oldcode*` need a work tree of the initial (pre-fix) connector at `work_old`, which is not shipped.
- `../aggregate.py`, `../aggregate_prune.py`, `../aggregate_ttl.py` and `../summarize_faults.py` produce the
  `aggregate*.txt` / `faults_summary.txt` tables from a results directory.
