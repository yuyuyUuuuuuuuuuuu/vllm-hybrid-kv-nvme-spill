#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Origin: copy of code/fork/tools/prodcheck/conc_quality_probe.py of the companion repository
# github.com/yuyuyUuuuuuuuuuuu/glm-5.3-flash-dgx-spark-vllm-optimization (code unchanged except docstrings, comments and one --help string adapted).
# docs/*.md references below are files under code/fork/docs/ of that repository. Used here only as the client library
# of code/spill_quality_probe.py (Client, api_key, ids_of, kl_top, text_prompts).
"""Concurrency decode-vs-prefill quality probe (deploy kit r16z7; the kpool tail-ring defect of
docs/PREFIX_HIT_TAIL.md). Operator-run on the head node inside an A/B arm (the server otherwise idle), read-only requests.

DEFAULT --mode trace (r16z7 review, after measuring production with light requests on 2026-10-05):
  S  solo   each of STREAMS (default 4) prompts (~5-6.6k tokens of the licence/stdlib corpus) decoded ALONE first
            (fresh salt, greedy, --gen 768 tokens, top-5 logprobs): the paired control
  C  conc   per round (--rounds warm,cold) the same prompts decoded together: warm = from a prefix hit (warmed with
            max_tokens 1 under their own salts), cold = new salts, prefill + decode concurrently (the stale-id path)
  R  rescore per stream ONE solo teacher-forced prefill of prompt + generated (prompt_logprobs 5, fresh salt): every
            generated position is scored, exactly as tools/kpool_decode_consistency.py does in every A/B arm
  Per position KL(decode || prefill) over the top-5 union (missing-mass floor: production's dvp statistic), top-1 and
  dlpc; per stream the mean; every concurrent stream paired with the SOLO decode of the SAME prompt:
    CONCQ mode=trace conc kl=<> solo kl=<> conc/solo=<> worst/solo=<> paired gm=<geo-mean of the 8 paired ratios>
          z=<mean log ratio / its SE> streams=8/4 | rounds warm=<> cold=<> | top1 .. | dlpc .. | per-stream kl=[..] |
          positions=<n> ctx=[..] gen=768 wall=..s
  Why not the sampled design below: production's decode logprobs carry batch-coupled numerical noise. The same prompt
  decoded solo twice from the same prefix hit diverges at token ~119 and differs by dlpc 0.25 (KL 0.004) at the
  matched positions; decode-vs-prefill dlpc is ~0.32 per stream - 10-50x the rig's stock-vs-=2 difference
  (0.005-0.015), so 12 positions x dlpc cannot see the defect there. KL over all ~768 positions per stream, paired
  per prompt, is the most sensitive statistic production allows (stream-to-stream log-ratio SD ~0.2-0.35 measured on
  stock production with 2 streams). Contexts stay <= ~7.4k + 768: prompt_logprobs materialises full-vocab logits for
  every prompt position (kpool_decode_consistency runs 6.3k + 1536 on production every arm); the defect needs
  concurrency, not long context (the single-GPU engine smoke shows it at 6-7k). ctx + gen > 9000 is refused (--force).
  Load: 4 warm + 4 solo + 4 cold prefills of ~5-6.6k, 4 solo + 2 x 4 concurrent decodes of 768, 12 prompt_logprobs
  prefills of ~6-7.4k: ~3.5-4 min on production (2-stream runs measured 112-121 s).

--mode sampled (the first design, kept for the rig record):
What it measures
  The stock kpool tail ring (GLM53_KPOOL_TAIL_POSITIONS unset) is ONE ring shared by every running request, and stale
  warm-up block ids send prefill seeds into pooled indexer keys of other requests / cached prefixes. A solo request is
  barely affected; CONCURRENT decodes corrupt each other's decode-written pooled keys. So:
    W  warm   every warm-round prompt + the solo prompt once (max_tokens 1, own cache_salt) -> those decodes start from
              a prefix hit
    C  conc   per round (--rounds warm,cold), STREAMS (default 4) long-context greedy decodes launched together, --gen
              tokens each, top-K logprobs per generated position: warm = short suffix prefill then all decode
              concurrently; cold = new salts, the full prompts prefill concurrently (first-chunk tail seeds of one
              request land while the others' blocks are live - the stale-id path) and then decode. The stock damage is
              a random event per run (which block-table row a request gets decides where its stale seeds go); two
              rounds double the chances to catch it within the time budget.
    S  solo   one more stream decoded ALONE from its prefix hit (the control: no cross-request traffic)
    R  rescore per stream, a SOLO teacher-forced prefill of prompt + generated[:j] (max_tokens 1, top-K logprobs, a fresh
              cache_salt per stream) at --positions sampled positions j; the first request of a stream prefills the
              whole context fresh, the later ones hit its prefix cache (prefill-written, solo)
  Per position: KL(decode_j || prefill_j) over the union of the two top-K sets (missing-mass floor, as
  kpool_decode_consistency.py / quality_long.py) and top-1 agreement (argmax of both distributions). The summary line
  compares the concurrent streams with the solo control:
    CONCQ conc dlp=<> solo dlp=<> conc/solo=<> worst/solo=<> n=<conc>/<solo> | rounds warm=<> cold=<> | conc kl ... |
          per-stream dlp=[w0.. k0.. solo]   (worst/solo = the most affected concurrent stream: a stock damage event hits
          single streams)
  dlp = mean |logprob_decode - logprob_prefill| over the tokens in both top-K sets (the primary statistic: continuous
  in the logits; KL / top-1 over truncated top-K sets are dominated by membership flips at near-ties and stay secondary).
  Stock: conc dlp > solo dlp. With GLM53_KPOOL_TAIL_POSITIONS=2: conc ~ solo (and solo unchanged against stock).
  dlpc (r16z7 review, the PRIMARY statistic of the summary line) = dlp after removing each position's mean difference:
  mean |d_t - mean(d)| with d_t = logprob_decode(t) - logprob_prefill(t) over the common tokens. A constant offset
  between the two logprob vectors (a normalisation difference, the same distribution) is not a context error, and on
  production it is large: one decode position vs its teacher-forced prefill gave KL 1.2e-6 over the top-20 union but
  raw dlp 0.34 (2026-10-05, 1 light request) - 13x the whole rig signal. dlpc removes it; on the single-GPU test rig the offsets
  are ~0 and dlpc separates stock / =2 exactly like dlp. Raw dlp and the
  mean offset stay in the line as secondary columns.

Load (defaults): 5 warm prefills + 4 cold concurrent prefills of 30-48k tokens, 2 x 4 concurrent + 1 solo decodes of
256 tokens, 9 x 12 max_tokens-1 rescore requests (all but the first per stream are prefix hits). ~6-8 min on production; --deadline
(default 570 s) stops the rescore early and summarises what was measured (the line then says partial).
Waits up to --wait-idle seconds (default 180) for the server to report no running/waiting request, then refuses
(rc 3) if it is still busy (--force overrides); the --deadline clock starts after the wait. Never prints a secret: the
API key is read from the launcher .env (--env; absent file or key = no Authorization header, e.g. a single-GPU test rig).

  run:      conc_quality_probe.py --out <json> [--mode trace|sampled] [--streams 4] [--ctx ...] [--gen ...]
                                  [--logprobs ...] [--url ...] [--model ...] [--wait-idle 180] [--deadline 570] [--force]
            (trace defaults: --ctx 5600,6200,6800,7400 --gen 768 --logprobs 5; sampled: --ctx 30000,36000,42000,48000
             --solo-ctx 39000 --gen 256 --positions 12 --logprobs 20)
  compare:  conc_quality_probe.py --compare off.json on.json [more.json ...]
  ids mode (single-GPU test rig, a server with --skip-tokenizer-init): --prompt-ids-file <json list of token-id lists, one per
            stream; sampled mode: the solo stream LAST> instead of text prompts.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request

DEF_URL = os.environ.get("CONCQ_URL", "http://127.0.0.1:8888")
DEF_ENV = os.environ.get("CONCQ_ENV", os.path.expanduser("~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/.env"))
DEF_MODEL = os.environ.get("CONCQ_MODEL", "GLM-5.3-Flash-EXL3")
CHARS_PER_TOKEN = 3.8          # the corpus windows tokenize at 3.67-3.88 chars/token (GLM tokenizer, r16z7 review;
                               # 3.1 gave 25-38k instead of 30-48k); the real count comes back as prompt_token_ids


def api_key(path: str) -> str | None:
    try:
        for line in open(path):
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'") or None
    except OSError:
        return None
    return None


class Client:
    def __init__(self, url: str, key: str | None, model: str):
        self.url, self.key, self.model = url.rstrip("/"), key, model

    def _req(self, path: str, body=None, timeout=1800):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:     # the body may echo the request: print status + a short reason only
            msg = e.read()[:300].decode(errors="replace").replace(self.key or "\0", "***")
            raise SystemExit(f"concq: HTTP {e.code} from {path}: {msg}")
        return json.loads(raw) if body is not None else raw.decode()

    def metrics_busy(self) -> tuple[float, float]:
        text = self._req("/metrics")
        run = wait = 0.0
        for line in text.splitlines():
            if line.startswith("vllm:num_requests_running"):
                run += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:num_requests_waiting"):
                wait += float(line.rsplit(" ", 1)[1])
        return run, wait

    def complete(self, prompt, max_tokens: int, salt: str, logprobs: int):
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
                "logprobs": logprobs, "return_tokens_as_token_ids": True, "return_token_ids": True, "cache_salt": salt}
        t0 = time.time()
        out = self._req("/v1/completions", body)
        c = out["choices"][0]
        return {"prompt_ids": c.get("prompt_token_ids"), "ids": c.get("token_ids") or [],
                "top": (c.get("logprobs") or {}).get("top_logprobs") or [], "secs": time.time() - t0,
                "usage": out.get("usage") or {}}

    def prompt_lp(self, prompt_ids: list, k: int, salt: str):
        """teacher-forced prefill of prompt_ids: prompt_logprobs k for EVERY position (one request, max_tokens 1),
        as tools/kpool_decode_consistency.py does on production in every A/B arm."""
        body = {"model": self.model, "prompt": prompt_ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": k,
                "cache_salt": salt}
        t0 = time.time()
        out = self._req("/v1/completions", body)
        return {"pl": out["choices"][0].get("prompt_logprobs") or [], "secs": time.time() - t0}


def ids_of(top) -> dict[int, float]:
    """{'token_id:123': lp} (return_tokens_as_token_ids) -> {123: lp}."""
    out = {}
    for t, v in (top or {}).items():
        tid = int(t.split(":", 1)[1]) if isinstance(t, str) and t.startswith("token_id:") else int(t)
        out[tid] = float(v["logprob"] if isinstance(v, dict) else v)
    return out


def kl_top(pa: dict, pb: dict) -> float:
    keys = set(pa) | set(pb)
    floor = min(min(pa.values()), min(pb.values())) - 1.0
    la = {x: pa.get(x, floor) for x in keys}
    lb = {x: pb.get(x, floor) for x in keys}
    za = math.log(sum(math.exp(v) for v in la.values()))
    zb = math.log(sum(math.exp(v) for v in lb.values()))
    return sum(math.exp(la[x] - za) * ((la[x] - za) - (lb[x] - zb)) for x in keys)


def dlp(pa: dict, pb: dict) -> float:
    """mean |logprob_decode - logprob_prefill| over the tokens in BOTH top-K sets (full-vocab logprobs, directly
    comparable). Continuous in the logits: a small perturbation of the context (a wrong pooled indexer key) moves it,
    while top-K membership flips at near-ties (what dominates KL/top-1 on flat distributions) do not."""
    common = set(pa) & set(pb)
    if not common:
        return float("nan")
    return sum(abs(pa[t] - pb[t]) for t in common) / len(common)


def dlpc(pa: dict, pb: dict) -> float:
    """dlp with the position's constant offset removed: mean |d_t - mean(d)|, d_t = logprob_decode - logprob_prefill
    over the common top-K tokens. Invariant to a per-position additive shift of either logprob vector (a different
    normaliser of the same distribution), still continuous in the logits."""
    common = set(pa) & set(pb)
    if not common:
        return float("nan")
    d = [pa[t] - pb[t] for t in common]
    m = sum(d) / len(d)
    return sum(abs(x - m) for x in d) / len(d)


def shift(pa: dict, pb: dict) -> float:
    common = set(pa) & set(pb)
    return sum(pa[t] - pb[t] for t in common) / len(common) if common else float("nan")


def corpus() -> str:
    files = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob("/usr/lib/python3.12/*.py"))
    return "".join("\n\n### %s\n%s" % (os.path.basename(f), open(f, errors="ignore").read()) for f in files)


def text_prompts(ctxs: list[int]) -> list[str]:
    t = corpus()
    out = []
    for i, n in enumerate(ctxs):
        nch = int(n * CHARS_PER_TOKEN)
        if len(t) < nch + 1000:
            raise SystemExit(f"concq: corpus has {len(t)} chars, stream {i} needs ~{nch}")
        s = random.Random("concq-%d-%d" % (i, n)).randrange(0, len(t) - nch)
        out.append(t[s:s + nch])
    return out


def positions(gen: int, n: int) -> list[int]:
    lo = min(16, max(0, gen - 1))
    if n <= 0 or gen <= lo:
        return []
    step = (gen - lo) / n
    return sorted({min(gen - 1, lo + int(k * step)) for k in range(n)})


def run(a) -> int:
    cli = Client(a.url, None if a.no_auth else api_key(a.env), a.model)
    t_wait = time.time()
    run_, wait_ = cli.metrics_busy()
    while (run_ or wait_) and time.time() - t_wait < a.wait_idle:
        time.sleep(5)
        run_, wait_ = cli.metrics_busy()
    t_start = time.time()
    if (run_ or wait_) and not a.force:
        print(f"concq: server busy (running={run_:.0f} waiting={wait_:.0f}); retry when idle or pass --force")
        return 3
    tag = "concq-%d" % time.time_ns()
    if a.mode == "trace":
        return run_trace(a, cli, t_start, tag)
    if a.prompt_ids_file:
        prompts = json.load(open(a.prompt_ids_file))
        if len(prompts) < 2 or not all(isinstance(p, list) and p for p in prompts):
            raise SystemExit("concq: --prompt-ids-file must hold >= 2 non-empty token-id lists (solo stream last)")
        streams = len(prompts) - 1
    else:
        streams = a.streams
        ctxs = [int(x) for x in a.ctx.split(",")]
        if len(ctxs) < streams:
            ctxs = (ctxs * streams)[:streams]
        prompts = text_prompts(ctxs[:streams] + [a.solo_ctx])
    rounds = [x for x in a.rounds.split(",") if x]
    if not rounds or any(x not in ("warm", "cold") for x in rounds):
        raise SystemExit("concq: --rounds must be a comma list of warm / cold")
    names, src = [], {}
    for rd in rounds:                         # w<i>: from a prefix hit (warmed); k<i>: cold (prefill + decode together)
        for i in range(streams):
            nm = ("w" if rd == "warm" else "k") + str(i)
            names.append(nm); src[nm] = i
    names.append("solo"); src["solo"] = streams
    salt = {n: f"{tag}-dec-{n}" for n in names}
    rec = {n: {"round": "solo" if n == "solo" else ("warm" if n[0] == "w" else "cold")} for n in names}
    ids = {}

    def decode_all(group):
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=max(1, len(group))) as ex:
            futs = {n: ex.submit(cli.complete, ids[src[n]], a.gen, salt[n], a.logprobs) for n in group}
            for n, f in futs.items():
                r = f.result()
                rec[n].update(prompt_ids=r["prompt_ids"] or ids[src[n]], gen=r["ids"], dec_top=r["top"],
                              dec_s=round(r["secs"], 2),
                              dec_cached=(r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"))
        return time.time() - t0

    # ---- W: warm the warm-round prompts + the solo prompt (own salts; their decodes then start from a prefix hit).
    #      Text prompts are tokenized here once (prompt_token_ids); every later request sends token ids.
    wset = [n for n in names if rec[n]["round"] in ("warm", "solo")] if "warm" in rounds else ["solo"]
    for n in wset:
        w = cli.complete(prompts[src[n]], 1, salt[n], 1)
        ids[src[n]] = w["prompt_ids"] or prompts[src[n]]
    for i, p in enumerate(prompts):
        if i not in ids:                      # a cold-only run: tokenize with a 1-token request on a throwaway salt
            ids[i] = cli.complete(p, 1, f"{tag}-tok-{i}", 1)["prompt_ids"] or p
    print("concq: warm %s tokens in %.0fs" % ([len(ids[i]) for i in sorted(ids)], time.time() - t_start), flush=True)
    # ---- C: each round's decodes launched together (warm: short suffix prefill, then all decode concurrently;
    #      cold: the full prompts prefill concurrently, so first-chunk seeds and decodes overlap other requests)
    times = {}
    for rd in rounds:
        times[rd] = decode_all([n for n in names if rec[n]["round"] == rd])
    # ---- S: the solo control (prefix hit, alone)
    times["solo"] = decode_all(["solo"])
    for n in names:
        if len(rec[n]["gen"]) != a.gen or len(rec[n]["dec_top"]) != a.gen:
            raise SystemExit(f"concq: stream {n}: {len(rec[n]['gen'])} tokens / {len(rec[n]['dec_top'])} logprob rows "
                             f"for --gen {a.gen} (ignore_eos/logprobs not honoured?)")
    print("concq: decodes %s (cached tokens %s)" % (" ".join("%s %.0fs" % kv for kv in times.items()),
                                                    [rec[n]["dec_cached"] for n in names]), flush=True)
    conc_s = sum(v for k, v in times.items() if k != "solo")
    # ---- R: solo teacher-forced prefill at sampled positions (fresh salt per stream; the stream's first request
    #      prefills the whole context, the later ones hit its cache). Position-major, solo first: a deadline cut
    #      leaves every stream (and the control) with the same early positions.
    pos = positions(a.gen, a.positions)
    partial = False
    order = ["solo"] + [n for n in names if n != "solo"]
    for n in order:
        rec[n]["pos"] = []
    for j in pos:
        for n in order:
            if time.time() - t_start > a.deadline:
                partial = True
                break
            r = cli.complete(rec[n]["prompt_ids"] + rec[n]["gen"][:j], 1, f"{tag}-ref-{n}", a.logprobs)
            pa, pb = ids_of(rec[n]["dec_top"][j]), ids_of(r["top"][0] if r["top"] else {})
            if not pa or not pb:
                continue
            rec[n]["pos"].append({"j": j, "kl": kl_top(pa, pb), "dlp": dlp(pa, pb), "dlpc": dlpc(pa, pb), "shift": shift(pa, pb),
                                  "top1": max(pa, key=pa.get) == max(pb, key=pb.get),
                                  "dec": pa, "pre": pb, "secs": round(r["secs"], 2)})
        if partial:
            break
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "sampled", "args": {k: v for k, v in vars(a).items() if k != "env"},
           "streams": streams, "partial": partial, "wall_s": round(time.time() - t_start, 1), "conc_decode_s": round(conc_s, 1),
           "rec": {n: {k: v for k, v in rec[n].items() if k != "dec_top"} for n in names}}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f)
    print(summary_line(out))
    return 0


def run_trace(a, cli, t_start: float, tag: str) -> int:
    """--mode trace (default, r16z7 review): every generated position of every stream is scored against ONE
    teacher-forced prefill (prompt_logprobs), and every concurrent stream is paired with a SOLO decode of the SAME prompt.
    Why: production's decode logprobs carry batch-coupled numerical noise far above the rig's defect signal (2026-10-05,
    light requests: decode-vs-decode of the same prompt solo twice dlpc 0.25, decode-vs-prefill dlpc 0.32 vs the rig's
    stock/=2 difference of ~0.005-0.015), so the sampled mode's 12 positions x dlpc cannot see the defect there. KL
    (probability-weighted, top-K union, the production dvp statistic: floor ~0.017 at top-5 over 4608 positions) over
    hundreds of positions per stream, paired per prompt, is what this mode reports. Contexts stay <= ~6.8k + gen tokens:
    prompt_logprobs materialises full-vocab logits for every prompt position (kpool_decode_consistency.py runs 6.3k +
    1536 on production in every arm; the defect needs concurrency, not long context: the single-GPU engine smoke shows it at
    6-7k)."""
    if a.prompt_ids_file:
        prompts = json.load(open(a.prompt_ids_file))
        if len(prompts) < 1 or not all(isinstance(p, list) and p for p in prompts):
            raise SystemExit("concq: --prompt-ids-file must hold >= 1 non-empty token-id lists")
        streams = len(prompts)
    else:
        streams = a.streams
        ctxs = [int(x) for x in a.ctx.split(",")]
        if len(ctxs) < streams:
            ctxs = (ctxs * streams)[:streams]
        prompts = text_prompts(ctxs[:streams])
    rounds = [x for x in a.rounds.split(",") if x]
    if not rounds or any(x not in ("warm", "cold") for x in rounds):
        raise SystemExit("concq: --rounds must be a comma list of warm / cold")
    names, src = [], {}
    for i in range(streams):
        names.append("s%d" % i); src["s%d" % i] = i
    for rd in rounds:
        for i in range(streams):
            nm = ("w" if rd == "warm" else "k") + str(i)
            names.append(nm); src[nm] = i
    salt = {n: f"{tag}-dec-{n}" for n in names}
    rec = {n: {"round": {"s": "solo", "w": "warm", "k": "cold"}[n[0]], "prompt": src[n]} for n in names}
    ids = {}

    def decode_all(group):
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=max(1, len(group))) as ex:
            futs = {n: ex.submit(cli.complete, ids[src[n]], a.gen, salt[n], a.logprobs) for n in group}
            for n, f in futs.items():
                r = f.result()
                rec[n].update(prompt_ids=r["prompt_ids"] or ids[src[n]], gen=r["ids"], dec_top=r["top"],
                              dec_s=round(r["secs"], 2),
                              dec_cached=(r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"))
        return time.time() - t0

    # ---- W: tokenize + warm the warm-round prompts (own salts: those decodes start from a prefix hit)
    for i in range(streams):
        w = cli.complete(prompts[i], 1, salt["w%d" % i] if "warm" in rounds else f"{tag}-tok-{i}", 1)
        ids[i] = w["prompt_ids"] or prompts[i]
    print("concq: trace, prompts %s tokens (tokenize/warm %.0fs)" % ([len(ids[i]) for i in range(streams)],
                                                                    time.time() - t_start), flush=True)
    # ---- S: the controls FIRST, each prompt decoded ALONE (fresh salt: prefill + decode, nothing else running)
    times = {"solo": sum(decode_all(["s%d" % i]) for i in range(streams))}
    # ---- C: per round the streams launched together (warm: from the prefix hit; cold: prefill + decode together)
    for rd in rounds:
        times[rd] = decode_all([n for n in names if rec[n]["round"] == rd])
    for n in names:
        if len(rec[n]["gen"]) != a.gen or len(rec[n]["dec_top"]) != a.gen:
            raise SystemExit(f"concq: stream {n}: {len(rec[n]['gen'])} tokens / {len(rec[n]['dec_top'])} logprob rows "
                             f"for --gen {a.gen} (ignore_eos/logprobs not honoured?)")
    print("concq: decodes %s (cached tokens %s)" % (" ".join("%s %.0fs" % kv for kv in times.items()),
                                                    [rec[n]["dec_cached"] for n in names]), flush=True)
    # ---- R: per stream ONE solo teacher-forced prefill of prompt + generated (fresh salt), prompt-major (the
    #      control first), so a --deadline cut keeps complete prompt groups
    partial = False
    for i in range(streams):
        for n in [x for x in names if src[x] == i]:
            if time.time() - t_start > a.deadline:
                partial = True
                break
            P, G = rec[n]["prompt_ids"], rec[n]["gen"]
            r = cli.prompt_lp(P + G, a.logprobs, f"{tag}-ref-{n}")
            pl = r["pl"]
            if len(pl) != len(P) + len(G):
                raise SystemExit(f"concq: stream {n}: {len(pl)} prompt_logprobs rows for {len(P) + len(G)} tokens")
            kl, d1, dc = [], [], []
            for j in range(len(G)):
                pa, pb = ids_of(rec[n]["dec_top"][j]), ids_of(pl[len(P) + j])
                if not pa or not pb:
                    kl.append(None); d1.append(None); dc.append(None)
                    continue
                kl.append(round(kl_top(pa, pb), 7)); d1.append(max(pa, key=pa.get) == max(pb, key=pb.get))
                dc.append(round(dlpc(pa, pb), 5))
            rec[n].update(kl=kl, top1=d1, dlpc=dc, ref_s=round(r["secs"], 2))
        if partial:
            break
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "trace",
           "args": {k: v for k, v in vars(a).items() if k != "env"}, "streams": streams, "partial": partial,
           "wall_s": round(time.time() - t_start, 1), "times": {k: round(v, 1) for k, v in times.items()},
           "rec": {n: {k: v for k, v in rec[n].items() if k not in ("dec_top", "prompt_ids")} | {"ctx": len(rec[n]["prompt_ids"])}
                   for n in names}}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f)
    print(summary_line(out))
    return 0


KL_EPS = 0.002   # additive floor of the paired ratio: a greedy stream that falls into a repetition loop scores KL
                 # ~0.001 (production 2026-10-05: 0.0006-0.0017 on one prompt vs 0.012-0.040 otherwise) and its ratio
                 # would be pure noise; 0.002 is ~1/8 of production's typical stream KL and ~2 % of the rig's


def trace_stats(d: dict):
    """per stream: mean KL / top-1 / dlpc over its scored positions; per concurrent stream: KL ratio to the SOLO
    decode of the same prompt (paired)."""
    per = {}
    for n, r in d["rec"].items():
        kl = [x for x in r.get("kl") or [] if x is not None]
        if not kl:
            continue
        t1 = [x for x in r.get("top1") or [] if x is not None]
        dc = [x for x in r.get("dlpc") or [] if x is not None and x == x]
        per[n] = {"kl": _m(kl), "top1": _m(t1), "dlpc": _m(dc), "n": len(kl), "round": r["round"], "prompt": r["prompt"]}
    solo = {v["prompt"]: v["kl"] for n, v in per.items() if v["round"] == "solo"}
    for n, v in per.items():
        if v["round"] != "solo" and v["prompt"] in solo and solo[v["prompt"]] + KL_EPS > 1e-12:
            v["ratio"] = (v["kl"] + KL_EPS) / (solo[v["prompt"]] + KL_EPS)
    return per


def trace_line(d: dict) -> str:
    per = trace_stats(d)
    conc = [v for v in per.values() if v["round"] != "solo"]
    solo = [v for v in per.values() if v["round"] == "solo"]
    ck, sk = _m([v["kl"] for v in conc]), _m([v["kl"] for v in solo])
    lr = [math.log(v["ratio"]) for v in conc if v.get("ratio", 0) > 0]
    gm = math.exp(_m(lr)) if lr else float("nan")
    se = (math.sqrt(sum((x - _m(lr)) ** 2 for x in lr) / (len(lr) - 1)) / math.sqrt(len(lr))) if len(lr) > 1 else float("nan")
    worst = max((v["ratio"] for v in conc if "ratio" in v), default=float("nan"))
    rd = {}
    for v in conc:
        rd.setdefault(v["round"], []).append(v.get("ratio", float("nan")))
    npos = sum(v["n"] for v in per.values())
    ctx = sorted({r.get("ctx", 0) for r in d["rec"].values()})
    return ("CONCQ mode=trace conc kl=%.5f solo kl=%.5f conc/solo=%.2f worst/solo=%.2f paired gm=%.2f z=%.1f streams=%d/%d | "
            "rounds %s | top1 conc=%.1f%% solo=%.1f%% | dlpc conc=%.4f solo=%.4f | per-stream kl=[%s] | positions=%d "
            "ctx=%s gen=%d wall=%.0fs%s" % (
                ck, sk, ck / sk if sk > 1e-12 else float("nan"), worst, gm, (_m(lr) / se) if se and se == se and se > 0 else float("nan"),
                len(conc), len(solo), " ".join("%s=%.2f" % (k, _m(v)) for k, v in rd.items()),
                100 * _m([v["top1"] for v in conc]), 100 * _m([v["top1"] for v in solo]),
                _m([v["dlpc"] for v in conc]), _m([v["dlpc"] for v in solo]),
                " ".join("%s:%.4f" % (n, per[n]["kl"]) for n in sorted(per, key=lambda x: (int(x[1:]), "swk".index(x[0])))),
                npos, ctx, d["args"].get("gen", 0), d["wall_s"], " PARTIAL" if d.get("partial") else ""))


def stats(d: dict):
    for r in d["rec"].values():          # dlp / dlpc / shift from the stored top-K rows (older files lack the fields)
        for p in r.get("pos", []):
            pa, pb = {int(k): v for k, v in p["dec"].items()}, {int(k): v for k, v in p["pre"].items()}
            for k, f in (("dlp", dlp), ("dlpc", dlpc), ("shift", shift)):
                if k not in p:
                    p[k] = f(pa, pb)
    conc = [p for n, r in d["rec"].items() if n != "solo" for p in r.get("pos", [])]
    solo = d["rec"].get("solo", {}).get("pos", [])
    per = {n: _m([p["dlpc"] for p in r.get("pos", []) if p["dlpc"] == p["dlpc"]]) for n, r in d["rec"].items()}
    return conc, solo, per


def _m(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _p90(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(0.9 * len(s))) - 1)]


def summary_line(d: dict) -> str:
    if d.get("mode") == "trace":
        return trace_line(d)
    conc, solo, per = stats(d)
    cd = [p["dlpc"] for p in conc if p["dlpc"] == p["dlpc"]]
    sd = [p["dlpc"] for p in solo if p["dlpc"] == p["dlpc"]]
    rcd = [p["dlp"] for p in conc if p["dlp"] == p["dlp"]]
    rsd = [p["dlp"] for p in solo if p["dlp"] == p["dlp"]]
    sh = [abs(p["shift"]) for p in conc + solo if p["shift"] == p["shift"]]
    ck, sk = [p["kl"] for p in conc], [p["kl"] for p in solo]
    ratio = _m(cd) / _m(sd) if sd and _m(sd) > 1e-9 else float("nan")   # no ratio of rounding noise
    ctx = sorted({len(r.get("prompt_ids") or []) for r in d["rec"].values()})
    rd = {}
    for n, r in d["rec"].items():
        if n != "solo":
            rd.setdefault(r.get("round", "warm"), []).extend(p["dlpc"] for p in r.get("pos", []) if p["dlpc"] == p["dlpc"])
    worst = max((v for n, v in per.items() if n != "solo" and v == v), default=float("nan"))
    wratio = worst / _m(sd) if sd and _m(sd) > 1e-9 else float("nan")
    return ("CONCQ conc dlpc=%.5f solo dlpc=%.5f conc/solo=%.2f worst/solo=%.2f n=%d/%d | rounds %s | raw conc dlp=%.5f "
            "solo dlp=%.5f |shift| mean=%.4f max=%.4f | conc kl_mean=%.5f top1=%.1f%% | "
            "solo kl_mean=%.5f top1=%.1f%% | per-stream dlpc=[%s] ctx=%s gen=%d wall=%.0fs%s" % (
                _m(cd), _m(sd), ratio, wratio, len(conc), len(solo), " ".join("%s=%.5f" % (k, _m(v)) for k, v in rd.items()),
                _m(rcd), _m(rsd), _m(sh), max(sh, default=float("nan")),
                _m(ck), 100.0 * _m([p["top1"] for p in conc]), _m(sk), 100.0 * _m([p["top1"] for p in solo]),
                " ".join("%s:%.4f" % (n, v) for n, v in per.items()), ctx, d["args"].get("gen", 0), d["wall_s"],
                " PARTIAL" if d.get("partial") else ""))


def compare(paths: list[str]) -> int:
    rows = []
    for p in paths:
        d = json.load(open(p))
        if d.get("mode") == "trace":
            per = trace_stats(d)
            c = [v for v in per.values() if v["round"] != "solo"]
            so = [v for v in per.values() if v["round"] == "solo"]
            rows.append((p, _m([v["kl"] for v in c]), _m([v["kl"] for v in so]), _m([v["top1"] for v in c]),
                         _m([v["top1"] for v in so]), len(c), len(so)))
            print(summary_line(d) + "   <- " + p)
            continue
        conc, solo, _ = stats(d)
        rows.append((p, _m([x["dlpc"] for x in conc]), _m([x["dlpc"] for x in solo]), _m([x["top1"] for x in conc]),
                     _m([x["top1"] for x in solo]), len(conc), len(solo)))
        print(summary_line(d) + "   <- " + p)
    modes = {json.load(open(p)).get("mode", "sampled") for p in (paths[0], paths[-1])}
    if len(rows) >= 2 and len(modes) > 1:
        print("CONCQ-CMP skipped: %s and %s were run in different modes (trace KL vs sampled dlpc)" % (
            os.path.basename(paths[0]), os.path.basename(paths[-1])))
    elif len(rows) >= 2:
        a, b = rows[0], rows[-1]
        print("CONCQ-CMP %s -> %s: conc %s %.5f -> %.5f (x%.2f), solo %.5f -> %.5f (x%.2f), conc top1 %.1f%% -> %.1f%%" % (
            os.path.basename(a[0]), os.path.basename(b[0]), "kl" if json.load(open(a[0])).get("mode") == "trace" else "dlpc",
            a[1], b[1], (b[1] / a[1]) if a[1] else float("nan"),
            a[2], b[2], (b[2] / a[2]) if a[2] else float("nan"), 100 * a[3], 100 * b[3]))
    return 0


def ab(off_paths: list[str], on_paths: list[str]) -> int:
    """--ab OFF.json [..] --on ON.json [..] (trace files of the A/B arms): the paired log KL ratio (concurrent stream /
    solo decode of the same prompt), averaged over the two rounds per (run, prompt) - streams sharing a solo control are
    not independent - then ON minus OFF with a Welch SE. verdict BETTER (z < -2) / WORSE (z > +2) / NO-DIFF."""
    def units(paths):
        u, solo = [], []
        for p in paths:
            d = json.load(open(p))
            if d.get("mode") != "trace":
                raise SystemExit(f"concq: --ab needs trace-mode files ({p} is {d.get('mode', 'sampled')})")
            if d.get("partial"):
                print(f"concq: note: {os.path.basename(p)} is PARTIAL")
            per = trace_stats(d)
            byp = {}
            for v in per.values():
                if v["round"] == "solo":
                    solo.append(v["kl"])
                elif v.get("ratio", 0) > 0:
                    byp.setdefault(v["prompt"], []).append(math.log(v["ratio"]))
            u += [_m(x) for x in byp.values()]
        return u, solo

    def var(x):
        return sum((y - _m(x)) ** 2 for y in x) / (len(x) - 1) if len(x) > 1 else float("nan")

    lo, so = units(off_paths)
    ln, sn = units(on_paths)
    if len(lo) < 2 or len(ln) < 2:
        raise SystemExit("concq: --ab needs >= 2 (run, prompt) units per side")
    dlt = _m(ln) - _m(lo)
    se = math.sqrt(var(ln) / len(ln) + var(lo) / len(lo))
    z = dlt / se if se > 0 else float("nan")
    verdict = "BETTER" if z < -2 else ("WORSE" if z > 2 else "NO-DIFF")
    print("CONCQ-AB off gm=%.3f (units %d) on gm=%.3f (units %d) on/off=%.3f z=%.2f solo kl on/off=%.3f verdict=%s" % (
        math.exp(_m(lo)), len(lo), math.exp(_m(ln)), len(ln), math.exp(dlt), z, _m(sn) / _m(so) if _m(so) else float("nan"),
        verdict))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="result JSON (token ids + top-K logprobs per sampled position; no text)")
    ap.add_argument("--compare", nargs="+", metavar="JSON", help="summarise result files (first vs last compared)")
    ap.add_argument("--ab", nargs="+", metavar="OFF_JSON", help="trace files of the OFF arms (with --on): CONCQ-AB verdict")
    ap.add_argument("--on", nargs="+", metavar="ON_JSON", help="trace files of the ON arms (with --ab)")
    ap.add_argument("--url", default=DEF_URL)
    ap.add_argument("--env", default=DEF_ENV, help="launcher .env with VLLM_API_KEY (never printed)")
    ap.add_argument("--no-auth", action="store_true", help="send no Authorization header (single-GPU test rig)")
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--mode", choices=("trace", "sampled"), default="trace",
                    help="trace (default): all generated positions vs one prompt_logprobs prefill, paired solo per prompt; "
                         "sampled: the first design (long contexts, --positions max_tokens-1 rescores, dlpc)")
    ap.add_argument("--streams", type=int, default=4, help="concurrent streams per round")
    ap.add_argument("--rounds", default="warm,cold", help="warm (decodes from prefix hits) and/or cold (prefill+decode)")
    ap.add_argument("--ctx", default=None, help="approx. prompt tokens per concurrent stream (trace 5600,6200,6800,7400; "
                                                "sampled 30000,36000,42000,48000)")
    ap.add_argument("--solo-ctx", type=int, default=39000, help="sampled mode only")
    ap.add_argument("--gen", type=int, default=None, help="tokens per stream (trace 768, sampled 256)")
    ap.add_argument("--positions", type=int, default=12, help="rescored positions per stream")
    ap.add_argument("--logprobs", type=int, default=None, help="top-K (trace 5 = the production dvp statistic, sampled 20)")
    ap.add_argument("--deadline", type=float, default=570.0, help="seconds; the rescore stops after it (partial)")
    ap.add_argument("--prompt-ids-file", help="JSON list of token-id prompts (concurrent streams, solo LAST)")
    ap.add_argument("--wait-idle", type=float, default=180.0,
                    help="seconds to wait for running=waiting=0 before refusing (rc 3); the deadline starts after it")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    tr = a.mode == "trace"
    a.ctx = a.ctx or ("5600,6200,6800,7400" if tr else "30000,36000,42000,48000")
    a.gen = a.gen or (768 if tr else 256)
    a.logprobs = a.logprobs or (5 if tr else 20)
    if tr and not a.prompt_ids_file and max(int(x) for x in a.ctx.split(",")) + a.gen > 9000 and not a.force:
        ap.error("trace mode: ctx + gen > 9000 tokens materialises full-vocab prompt logits beyond what production has "
                 "run (kpool_decode_consistency: ~7.9k); pass --force to override")
    if a.compare:
        return compare(a.compare)
    if a.ab or a.on:
        if not (a.ab and a.on):
            ap.error("--ab OFF.json [..] needs --on ON.json [..]")
        return ab(a.ab, a.on)
    if not a.out:
        ap.error("--out is required for a run")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
