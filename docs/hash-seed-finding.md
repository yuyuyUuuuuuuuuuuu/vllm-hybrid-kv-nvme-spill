# Finding: vLLM prefix-cache block hashes change on every engine restart unless PYTHONHASHSEED is set

**Short version.** In the vLLM tree we serve (image vLLM `0.1.dev20051+g487ecf187` plus the launcher's overlays), and in
any vLLM whose `vllm/v1/core/kv_cache_utils.py:init_none_hash` predates upstream PR #51875, the root of every
prefix-cache block hash, `NONE_HASH`, is `os.urandom(32)` when the environment variable `PYTHONHASHSEED` is not set.
Every block hash is chained from `NONE_HASH`, so **every prefix-cache block hash is different after each engine
restart**, even for identical tokens, model and configuration. Anything that stores KV blocks under their hash outside
the process (an NVMe/CPU tier, a shared cache, an index) cannot be found again after a restart. With the default hash
function (`sha256`, pickle-based) this vLLM logs no warning; its warning is only emitted for the CBOR hash functions.

Our production start-up (launcher `start.sh` and the image's `ENV`) had no `PYTHONHASHSEED`, so until 2026-10-14 every
restart silently changed every block hash. The connector now fixes the seed in the engine-core process only, verifies
the result at start-up, and disables reuse across restarts if hashes are not reproducible.

## Where it is in vLLM (production tree)

| file:line (production vLLM tree) | what |
|---|---|
| `vllm/v1/core/kv_cache_utils.py:88-96` | comment and declaration: "We use a random value to avoid hash collisions or PYTHONHASHSEED environment variable if set ..."; `NONE_HASH: BlockHash` |
| `vllm/v1/core/kv_cache_utils.py:100-115` | `init_none_hash(hash_fn)`: `hash_seed = os.getenv("PYTHONHASHSEED")`; if `None` -> `NONE_HASH = BlockHash(os.urandom(32))` (line 113), else `BlockHash(hash_fn(hash_seed))` (line 115) |
| `vllm/v1/core/kv_cache_utils.py:97, 104-110` | the "PYTHONHASHSEED is not set" warning is logged only when `hash_fn in _CBOR_HASH_FUNCTIONS` (`sha256_cbor`, `xxhash_cbor`); the default `sha256` is silent |
| `vllm/v1/core/kv_cache_utils.py:599` | the first block of every request starts its chain at `parent_block_hash = NONE_HASH` |
| `vllm/v1/engine/core.py:161` | `EngineCore.__init__` constructs the scheduler (and with it the KV connector, scheduler role) ... |
| `vllm/v1/engine/core.py:222-226` | ... and only afterwards calls `init_none_hash(caching_hash_fn)` (when prefix caching or a KV connector is enabled) |

The hash function itself (`prefix_caching_hash_algo`, default `sha256` = pickle + SHA-256) and the block extra keys
(LoRA, multimodal, `cache_salt`) are deterministic; only the seed is random.

### Upstream status (checked 2026-10-14)

Upstream vLLM changed this in **PR #51875, "[Core] Make prefix-cache NONE_HASH deterministic by default"** (merged
2026-08-18, commit `ef47a89`). On upstream `main` as of 2026-10-14, `resolve_none_hash_seed()` in
`vllm/v1/core/kv_cache_utils.py` returns `PYTHONHASHSEED` if set; otherwise the fixed seed
`DEFAULT_NONE_HASH_SEED = "vllm-none-hash"` for cryptographic hash functions (`sha256`, `sha256_cbor`) and fresh random
bytes for the non-cryptographic `xxhash` / `xxhash_cbor` (with a warning). The vLLM in our image does not contain that
change: its `init_none_hash` is the older code quoted above (line numbers in the table refer to our tree). So:

- vLLM builds without PR #51875 (such as the tree we serve): hashes are random per process unless `PYTHONHASHSEED`
  is set, whatever the hash function;
- vLLM with PR #51875 and `sha256`/`sha256_cbor`: deterministic by default;
- any vLLM with `xxhash`/`xxhash_cbor`: random per process unless `PYTHONHASHSEED` is set.

## How the connector fixes it

`code/glm53_nvme_spill_connector.py`:

1. **Seed before vLLM uses it.** In the connector's constructor, scheduler role, with `GLM53_NVME_SPILL_PERSIST=1`
   (default), lines 2689-2699: if `PYTHONHASHSEED` is not set in the process environment, set it to
   `GLM53_NVME_SPILL_HASH_SEED` (default `5353`). Because `EngineCore` constructs the scheduler (and the connector)
   **before** it calls `init_none_hash` (`core.py:161` < `core.py:226`), vLLM then computes
   `NONE_HASH = hash_fn("5353")`, identical in every process. An operator-set `PYTHONHASHSEED` is kept as is.
2. **Only that process.** The variable is set in the engine-core process environment only, not for the container.
   Python's own `str` hash randomization is fixed at interpreter start and is not affected (measured: `hash(str)`
   still differs between processes). Setting `PYTHONHASHSEED` for the whole container would also disable `str` hash
   randomization in the API server; we did not do that. Child processes started later by that process inherit the
   variable.
3. **Verify at start-up.** `SpillScheduler._check_hash` (line 2328) checks on the first steps that
   `NONE_HASH == hash_fn(PYTHONHASHSEED)` for the configured `prefix_caching_hash_algo`. If not (or if `NONE_HASH` is
   never initialised within `ADOPT_DEADLINE_S`), `_persist_poll` (line 2358) wipes every rank's files and disables
   reuse across restarts for this run (the connector then behaves as with `PERSIST=0`: files are only used within the
   process lifetime). The `NONE_HASH` digest and the hash function name are part of the identity signature written to
   the index, so a different seed or hash function on the next start also leads to a full wipe, never to a wrong hit.
4. The scheduler start-up log line reports `persist=1 hash_seed_from=connector|env`; a failed check logs
   `block hashes are not reproducible across restarts (...)`.

## Evidence

`tests/t_hash_stability.py` (9 checks): fresh interpreters **without** `PYTHONHASHSEED` each build the real scheduler of
the replay rig (production vLLM tree, connector installed by `code/patch_nvme_spill.py`, the EngineCore order: scheduler,
then `init_none_hash`) and hash the same token list (3 blocks of 4,608 tokens + 5).

| condition | `NONE_HASH` (first 16 bytes) | first block hash | Python `hash(str)` |
|---|---|---|---|
| `PERSIST=0` (= stock behaviour), process 1 | `f0dedd7822fce0562008db1ba09a8742` | `6762f18fb29c746d75999aa27c1d0bf3` | random |
| `PERSIST=0`, process 2 | `18ea7c952da340828124a01e3cad555d` | `e1c55bc39575fa809afc9bcc6907afeb` | random |
| `PERSIST=1`, process 1 | `1f6065cfcc6f0c410597cad23ddd5644` | `d8dd5cb43c1f00e81e9d1a3f60be1c51` | random |
| `PERSIST=1`, process 2 | `1f6065cfcc6f0c410597cad23ddd5644` | `d8dd5cb43c1f00e81e9d1a3f60be1c51` | random (differs from process 1) |
| operator `PYTHONHASHSEED=77`, 2 processes | identical to each other, different from seed 5353 | identical | - |

The test also checks statically that in the production `core.py` the scheduler construction comes before
`init_none_hash` (character offsets 5539 < 8733). Values above are from the run on this published tree
(`results/published-tree/hash_stability.txt`, 9/9); the original run (`results/ttl/t_hash_stability.txt`, 9/9) gave the
same `PERSIST=1` values, i.e. the hashes are also identical across days, and different random values for `PERSIST=0`.

Effect in the replay rig (restart 2 h into a 4 h run, 10 min down, 5 seeds; `docs/measurements.md` section 7.3,
`results/ttl/aggregate.txt`): DSH re-read after the restart 2,028,530 tokens with `PERSIST=0` vs 174,139 with
`PERSIST=1` (-91%); the first turn of each conversation after the restart 1,903,966 vs 24,823 (-98.7%).

## If you run your own KV tier

- Set `PYTHONHASHSEED` for the engine process (or use a vLLM with PR #51875 and a cryptographic hash function), and
  check at start-up that the resulting `NONE_HASH` is what you expect; do not assume it.
- Put the seed / `NONE_HASH` digest and the hash function into whatever signature guards your persisted index, so a
  change invalidates the index instead of producing misses or, worse, hits on different content.
