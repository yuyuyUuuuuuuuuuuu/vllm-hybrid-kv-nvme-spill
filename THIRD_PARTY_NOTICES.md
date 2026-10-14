# Third-party notices

This repository is licensed under the GNU Affero General Public License v3.0 only (`LICENSE`), the licence of the
launcher it plugs into and of our companion repository. It contains code derived from, or written against, the projects
below.

## Code in this repository

| component | where it appears here | licence |
|---|---|---|
| vLLM (github.com/vllm-project/vllm) | `code/glm53_nvme_spill_connector.py` implements vLLM's `KVConnectorBase_V1` API; `code/patch_nvme_spill.py` carries short fragments of vLLM source text (`scheduler.py`, `config/vllm.py`) as patch anchors; the tests import vLLM at run time (not shipped) | Apache-2.0 |
| MiaAI-Lab "GLM-5.3-Flash-EXL3-2x-DGX-Sparks" launcher (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | `code/patch_tf_bundle.py` is a modified copy of the bundle hook of our launcher fork (published in the companion repository, `code/fork/overlay/patch_tf_bundle.py`) with the `GLM53_NVME_SPILL` entry added; `code/start.sh.nvmespill.patch` is a diff against the launcher kit's `start.sh` and contains context lines of it; `code/apply_start_sh.py` contains anchor fragments of that `start.sh` | AGPL-3.0; contributions made before 2026-09-07 were MIT (notice below) |
| companion repository glm-5.3-flash-dgx-spark-vllm-optimization (github.com/yuyuyUuuuuuuuuuuu/glm-5.3-flash-dgx-spark-vllm-optimization) | `code/conc_quality_probe.py` is a copy of its `code/fork/tools/prodcheck/conc_quality_probe.py` (code unchanged; docstrings, comments and one `--help` string adapted) | AGPL-3.0 |

## Design credit (no code copied)

`code/glm53_nvme_spill_connector.py` re-implements, for vLLM's KV connector API, the design of the CUDA spill tier of
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (patch 0088, `tensorfold/cuda/spill.py`, PR #78, written by
wojo (Robert Wojciechowski), Apache-2.0): one set of files per resume point on local NVMe per rank, decisions taken once
and shipped to every rank with all-rank agreement on every read, header-padded files written with O_DIRECT to
`.partial`, fsynced and renamed, a CRC-32 per 8 MiB checked on every read, small fixed staging buffers between NVMe and
GPU. That module's multi-rank design credits patch 0250 of jayleaton/glm53-tensorfold-spark (Jay Leaton, Apache-2.0);
its file conventions TensorFold's MLX spill (PR #68, gilby) and issue #155 (raymondkpwong). No source text was copied;
the file format, the content-addressed block files, the scheduler-side resume-point selection, the restore into vLLM's
prefix cache, pruning, TTL and reuse across restarts are this repository's own. The attribution is also kept in the
module's docstring.

## Not included

| component | relation | licence |
|---|---|---|
| DeepSeek Harness (DSH, github.com/deepseek-ai/deepseek-harness) | the client we used with the contract in `docs/client-contract.md`; our client-side change is described there, no DSH code is included | MIT |
| Docker image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` | serving image and test runtime (vLLM, PyTorch, ...) | a mix of the licences of its components, the launcher layers AGPL-3.0 |
| `zai-org/GLM-5.3-Flash`, its EXL3 quantizations, `incoai/GLM-5.3-Flash-DFlash2` | served model and drafter; the replay rig reads only their `config.json` / index files, nothing is shipped | the model MIT; the DFlash2 drafter CC BY-NC-ND 4.0 |

## AGPL-3.0 section 13

The connector runs inside a network service. If you run a modified version for other users, section 13 requires you to
offer them the corresponding source.

## Notice of the launcher's pre-2026-09-07 MIT licence


```
This project is licensed under the GNU Affero General Public License v3.0
(see LICENSE). Before 2026-09-07 it was distributed under the MIT license
reproduced below, which is retained for the contributions made under it.

----------------------------------------------------------------------

MIT License

Copyright (c) 2026 Mia's AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The full text of the Apache License 2.0, which covers the vLLM source fragments above, is in `LICENSES/Apache-2.0.txt`.
