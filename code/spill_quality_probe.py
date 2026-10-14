#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Quality probe of an NVMe-restored prefix hit (docs/operations.md). Operator-run on the head node in the GLM53_NVME_SPILL=1
arm with the server otherwise idle and GLM53_EXPOSE_CACHE_RESET=1 (the probe flushes the GPU prefix cache once).

Sequence (one long prompt P of --ctx tokens, a turn suffix Q1 >= 4608 tokens, a short final suffix Q2):
  W1  P, max_tokens 1, salt S                      (first turn: the connector only remembers its boundary)
  W2  P+Q1, max_tokens 1, salt S                   (a later turn of the same conversation: its resume point is saved)
      wait --settle s, POST /reset_prefix_cache    (every GPU block of S is gone; the files stay)
  R   P+Q1+Q2, greedy --gen, top-5 logprobs, salt S     -> restored from NVMe (expect cached_tokens ~ len(P+Q1))
  H   P+Q1+Q2 again, salt S                              -> ordinary GPU prefix hit (control: hit path, no NVMe)
  F1  P+Q1+Q2, fresh salt                                -> full prefill (reference)
  F2  P+Q1+Q2, another fresh salt                        -> full prefill (noise floor of decode-vs-decode)
Per run: TTFT-ish wall of the request, cached tokens from usage.prompt_tokens_details (when the server reports it),
and the drafter acceptance from /metrics counter deltas (server idle -> attributable).
Per pair (X vs F1) over the common greedy prefix (positions before the first token mismatch): mean KL over the
top-5 union (missing-mass floor), top-1 agreement, first divergence index.
  SPILLQ ctx=<> R: wall=<>s cached=<> acc=<> | H: ... | F1 ... | F2 ... | KL R/F1=<> H/F1=<> F2/F1=<> | div R=<> H=<> F2=<>
Restart variant (reuse across restarts, 2026-10-14; --phase prepare, then restart the engine, then --phase restart):
  prepare  W1, W2 with salt S, wait --settle, write --state (the salt, sizes); no cache reset
  restart  (after an engine restart, idle) R, H, F1, F2 exactly as above -- R's prefix can only come back from the files the
           previous engine wrote (the GPU cache is new), so the same restore/kl/acceptance criteria prove the adopted
           files give the same output quality as the in-process restore.
Several runs (2026-10-14 finish): --phase prepare --runs N prepares N salts; --phase restart measures each and judges
the AVERAGED KL / acceptance (single runs are noisy); restore must be proven in every run. The in-process variant
(--phase both) needs POST /reset_prefix_cache, i.e. GLM53_EXPOSE_CACHE_RESET=1 (unauthenticated on the LAN) -- our
production deployment turns it off; the restart variant needs no cache reset.
Pass criteria (docs/operations.md), ALL required -- the probe FAILS when it can not prove that R was restored:
  restore   positive evidence that R's prefix came back from NVMe:
              usage.prompt_tokens_details.cached_tokens >= 0.9 * len(P+Q1) when the server reports it (production does
              not: no --enable-prompt-tokens-details), otherwise the connector's own log line
              "[glm53-nvme-spill] restore ready req=<id> rp=<N>" with N >= 0.9 * len(P+Q1) inside R's time window
              (--log-cmd, default `docker logs --since <t0> --until <t1> glm53-exl3-head`; needs GLM53_NVME_SPILL_LOG
              unset/1); no evidence (cached None AND no log line, or the log unreadable) = FAIL
            AND timing: R.wall - H.wall <= 0.25 * (F1.wall - H.wall) (all three decode --gen tokens, so the difference
              is the prefill R skipped; a restore costs ~0.1-1.5 s, re-reading P+Q1 tens of seconds)
  kl        KL(R,F1) within the band of KL(H,F1) and KL(F2,F1) (not above max(H,F2) * 1.5)
  acceptance  drafter acceptance of R not below H by more than 0.03
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conc_quality_probe as cq  # noqa: E402  (Client, api_key, ids_of, kl_top, text_prompts)

RESTORE_RE = re.compile(r"\[glm53-nvme-spill\] restore ready req=(\S+) rp=(\d+) in ([0-9.]+) s")


def restore_events(log_cmd: str, t0: float, t1: float) -> list[tuple[str, int, float]] | None:
    """The connector's "restore ready" lines logged in [t0, t1] (unix seconds); None when the log can't be read."""
    cmd = shlex.split(log_cmd.format(since=f"{t0:.3f}", until=f"{t1:.3f}"))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"spillq: log command failed: {e!r}", file=sys.stderr)
        return None
    if r.returncode != 0:
        print(f"spillq: log command rc={r.returncode}: {r.stderr.strip()[:200]}", file=sys.stderr)
        return None
    return [(m.group(1), int(m.group(2)), float(m.group(3))) for m in RESTORE_RE.finditer(r.stdout + r.stderr)]


def restore_verdict(r_cached, pq1: int, events, wall_r: float, wall_h: float, wall_f1: float) -> tuple[bool, str]:
    """PASS only with positive evidence that R's prefix was restored (see the module doc)."""
    need = 0.9 * pq1
    if r_cached is not None:
        ev_ok, ev = r_cached >= need, f"cached_tokens={r_cached} (need >= {need:.0f})"
    elif events is None:
        return False, "no evidence: the server reports no cached_tokens and the connector log could not be read"
    else:
        deep = [e for e in events if e[1] >= need]
        ev_ok = bool(deep)
        ev = (f"log: restore ready rp={deep[0][1]} in {deep[0][2]:.3f}s" if deep else
              f"no 'restore ready' line with rp >= {need:.0f} in R's window ({len(events)} shallower)")
    skipped = wall_f1 - wall_h
    excess = wall_r - wall_h
    t_ok = skipped > 0 and excess <= 0.25 * skipped
    tm = f"R-H={excess:.2f}s vs F1-H={skipped:.2f}s (limit 0.25x)"
    return ev_ok and t_ok, f"{ev}; {tm}"


def spec_counters(cli) -> tuple[float, float]:
    acc = drafted = 0.0
    for line in cli._req("/metrics").splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            acc += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            drafted += float(line.rsplit(" ", 1)[1])
    return acc, drafted


def gen(cli, prompt, n, salt):
    a0, d0 = spec_counters(cli)
    out = cli.complete(prompt, n, salt, 5)
    a1, d1 = spec_counters(cli)
    det = (out["usage"].get("prompt_tokens_details") or {})
    out["cached"] = det.get("cached_tokens")
    out["acc"] = (a1 - a0) / (d1 - d0) if d1 > d0 else float("nan")
    return out


def compare(x, ref):
    n = min(len(x["ids"]), len(ref["ids"]), len(x["top"]), len(ref["top"]))
    div = next((i for i in range(n) if x["ids"][i] != ref["ids"][i]), n)
    kls, top1 = [], 0
    for i in range(div):
        pa, pb = cq.ids_of(x["top"][i]), cq.ids_of(ref["top"][i])
        if pa and pb:
            kls.append(cq.kl_top(pa, pb))
            top1 += max(pa, key=pa.get) == max(pb, key=pb.get)
    return {"kl": sum(kls) / len(kls) if kls else float("nan"), "div": div, "n": len(kls),
            "top1": top1 / len(kls) if kls else float("nan")}


def reset(cli):
    req = urllib.request.Request(cli.url + "/reset_prefix_cache", data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status
    except urllib.error.HTTPError as e:
        raise SystemExit(f"spillq: /reset_prefix_cache -> HTTP {e.code} (needs GLM53_EXPOSE_CACHE_RESET=1 and an idle "
                         f"server; the connector also needs its save jobs finished: retry after --settle)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=80000)
    ap.add_argument("--q1", type=int, default=6000)
    ap.add_argument("--q2", type=int, default=3000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--settle", type=float, default=10.0)
    ap.add_argument("--url", default=cq.DEF_URL)
    ap.add_argument("--env", default=cq.DEF_ENV)
    ap.add_argument("--model", default=cq.DEF_MODEL)
    ap.add_argument("--no-auth", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--log-cmd", default="docker logs --since {since} --until {until} glm53-exl3-head",
                    help="command printing the head container's log for [since, until] (unix seconds)")
    ap.add_argument("--phase", choices=("both", "prepare", "restart"), default="both",
                    help="both = the in-process probe; prepare / restart = around an engine restart (--state)")
    ap.add_argument("--state", default="", help="state file written by --phase prepare, read by --phase restart")
    ap.add_argument("--runs", type=int, default=1,
                    help="prepare: resume points of this many independent salts; restart then measures each and "
                         "judges the AVERAGED KL / acceptance (single runs are noisy); restore must pass in every run")
    a = ap.parse_args()
    cli = cq.Client(a.url, None if a.no_auth else cq.api_key(a.env), a.model)
    run_, wait_ = cli.metrics_busy()
    if run_ or wait_:
        raise SystemExit(f"spillq: server busy (running={run_} waiting={wait_}); run it on an idle arm")
    if a.phase != "both" and not a.state:
        raise SystemExit("spillq: --phase prepare/restart needs --state <file>")
    if a.runs < 1 or (a.runs > 1 and a.phase == "both"):
        raise SystemExit("spillq: --runs N (N >= 1) is for --phase prepare (the in-process variant measures one run)")
    st = {}
    if a.phase == "restart":
        st = json.load(open(a.state))
        a.ctx, a.q1, a.q2 = st["ctx"], st["q1"], st["q2"]
    P, Q1, Q2 = cq.text_prompts([a.ctx, a.q1, a.q2])
    t0 = time.time()
    if a.phase == "restart":
        tags = list(st.get("tags") or [st["tag"]])
    else:
        base = "spillq-%d" % int(time.time())
        tags = [base] if a.runs == 1 else [f"{base}-{i}" for i in range(a.runs)]
    if a.phase != "restart":
        for tag in tags:
            cli.complete(P, 1, tag + "-S", 0)
            cli.complete(P + Q1, 1, tag + "-S", 0)
        time.sleep(a.settle)
    if a.phase == "prepare":
        with open(a.state, "w") as f:
            json.dump({"tag": tags[0], "tags": tags, "ctx": a.ctx, "q1": a.q1, "q2": a.q2, "t": time.time()}, f)
        print(f"SPILLQ prepared {len(tags)} run(s) tags={','.join(tags)} (resume point of P+Q1 saved per salt); "
              f"restart the engine, then --phase restart --state {a.state}")
        return 0
    if a.phase == "both":
        reset(cli)
    full = P + Q1 + Q2
    pq1 = None
    runs = []
    for tag in tags:
        S = tag + "-S"
        res = {}
        t_r0 = time.time() - 1.0
        res["R"] = gen(cli, full, a.gen, S)
        t_r1 = time.time() + 1.0
        res["H"] = gen(cli, full, a.gen, S)
        res["F1"] = gen(cli, full, a.gen, tag + "-F1")
        res["F2"] = gen(cli, full, a.gen, tag + "-F2")
        if pq1 is None:
            pq1 = len(cli.complete(P + Q1, 1, tag + "-len", 0)["prompt_ids"] or [])
        cmp_ = {k: compare(res[k], res["F1"]) for k in ("R", "H", "F2")}
        line = (f"SPILLQ tag={tag} ctx={len(res['R']['prompt_ids'] or [])} pq1={pq1} " +
                " | ".join(f"{k}: wall={res[k]['secs']:.1f}s cached={res[k]['cached']} acc={res[k]['acc']:.3f}"
                           for k in ("R", "H", "F1", "F2")) +
                " | KL " + " ".join(f"{k}/F1={cmp_[k]['kl']:.4g}" for k in ("R", "H", "F2")) +
                " | div " + " ".join(f"{k}={cmp_[k]['div']}" for k in ("R", "H", "F2")) +
                f" | wall={time.time() - t0:.0f}s")
        print(line, flush=True)
        events = restore_events(a.log_cmd, t_r0, t_r1) if res["R"]["cached"] is None else []
        ok_r, why = restore_verdict(res["R"]["cached"], pq1, events, res["R"]["secs"], res["H"]["secs"],
                                    res["F1"]["secs"])
        print(f"SPILLQ run {tag} restore={'PASS' if ok_r else 'FAIL'} ({why})", flush=True)
        runs.append({"tag": tag, "line": line, "res": res, "cmp": cmp_, "ok_restore": ok_r, "why": why,
                     "events": events})
    ok_restore, why, ok_kl, ok_acc, summary = aggregate_verdict(runs)
    print(f"SPILLQ verdict runs={len(runs)} restore={'PASS' if ok_restore else 'FAIL'} ({why}) "
          f"kl={'PASS' if ok_kl else 'FAIL'} acceptance={'PASS' if ok_acc else 'FAIL'} | {summary}")
    if a.out:
        out = {"runs": [{"tag": r["tag"], "line": r["line"], "restore": r["why"], "events": r["events"], "cmp": r["cmp"],
                         "runs": {k: {kk: v[kk] for kk in ("secs", "cached", "acc")} for k, v in r["res"].items()}}
                        for r in runs], "summary": summary,
               "verdict": {"restore": ok_restore, "kl": ok_kl, "acceptance": ok_acc}}
        if len(runs) == 1:      # the single-run layout of before (line / runs / cmp / restore / events) stays readable
            out.update({"line": runs[0]["line"], "cmp": runs[0]["cmp"], "restore": runs[0]["why"],
                        "events": runs[0]["events"]})
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if (ok_restore and ok_kl and ok_acc) else 1


def _mean(xs):
    xs = [x for x in xs if x == x]          # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate_verdict(runs):
    """restore: every run proven restored; kl: mean KL(R,F1) <= 1.5 x max(mean KL(H,F1), mean KL(F2,F1)); acceptance:
    mean acceptance of R not below the mean of H by more than 0.03 (one run = the single-run criteria of before)."""
    ok_restore = all(r["ok_restore"] for r in runs)
    fails = [r["tag"] for r in runs if not r["ok_restore"]]
    why = runs[0]["why"] if len(runs) == 1 else (f"{len(runs) - len(fails)}/{len(runs)} runs proven restored"
                                                 + (f"; not proven: {','.join(fails)}" if fails else ""))
    kl = {k: _mean([r["cmp"][k]["kl"] for r in runs]) for k in ("R", "H", "F2")}
    band = max(kl["H"], kl["F2"]) * 1.5
    ok_kl = kl["R"] <= band
    acc_r = _mean([r["res"]["R"]["acc"] for r in runs])
    acc_h = _mean([r["res"]["H"]["acc"] for r in runs])
    ok_acc = not (acc_r < acc_h - 0.03)
    summary = (f"mean KL R/F1={kl['R']:.4g} H/F1={kl['H']:.4g} F2/F1={kl['F2']:.4g} (band {band:.4g}) "
               f"mean acc R={acc_r:.3f} H={acc_h:.3f}")
    return ok_restore, why, ok_kl, ok_acc, summary

if __name__ == "__main__":
    sys.exit(main())
