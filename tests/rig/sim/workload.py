# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic replay workload fitted to production admission statistics.

Sources (aggregate statistics from our earlier re-read investigation; the files themselves are not shipped):
- the admission log (4,758 admissions): DSH-session totals decile range 200,027..
  742,497 tokens; one-shot requests deciles 14,097..137,383 (n=1,745); hit
  collapses to <=13,824 on 223/4,758 (205 exactly 13,824 == the shared head).
- the investigation report: 3 DSH conversations, common head 13,824, per-turn user +2k..8k,
  output 0.5k..3k, turn interval 20s..5min; one-shot 20k..100k occasionally;
  max 8 concurrent (MAX_NUM_SEQS=8, production environment).
- one-shot arrivals: exponential inter-arrival with a mean gap of 88 s (fitted).

Everything is drawn from a `random.Random(seed)` so all conditions share the
identical request stream for a given seed.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

COMMON_HEAD_LEN = 13_824
COMMON_HEAD_TOKEN = 100  # single repeated id: content identical across sessions

DSH_INIT_LENS = (430_000, 600_000, 780_000)
USER_TURN_RANGE = (2_000, 8_000)
OUTPUT_RANGE = (500, 3_000)
TURN_INTERVAL_RANGE = (20.0, 300.0)  # 20s..5min (observed in production)
ONESHOT_RANGE = (20_000, 100_000)
ONESHOT_MEAN_GAP_S = 88.0  # fitted mean gap between one-shot requests

PREFILL_TOK_S = 2036.0  # prefill rate assumed for the simulated clock (tok/s)
DECODE_STEP_S = 0.5  # decode step wall estimate (recorded assumption)


@dataclass
class ReqSpec:
    req_id: str
    kind: str  # "dsh" | "one"
    conv: int | None
    turn: int
    prompt_len: int
    output_len: int
    arrival_s: float
    content_salt: int  # mixes token ids beyond the shared head
    # [nvme-spill prune rig] a branch turn rewrites the conversation from this token position (the stream is cut there
    # and continues with fresh tokens); -1 = a plain extension. branch = "" | regen | edit | edit2 (see Conv.next_turn)
    branch_at: int = -1
    branch: str = ""
    # [nvme-spill ttl rig, 2026-10-14] DSH contract v1.1 fields this request carries (None = no field, like other API clients)
    spill: dict | None = None


def _ttl_spec() -> dict[str, int]:
    """NVMESPILL_TTL="c0=3600,c1=86400,c2=259200,c3=0": per DSH conversation, the ttl_s its requests carry (with
    session "sess-c<k>" and the conversation's epoch). Unset = no field on any request (byte-identical workload)."""
    out: dict[str, int] = {}
    for part in filter(None, os.environ.get("NVMESPILL_TTL", "").split(",")):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def _compactions() -> dict[int, float]:
    """NVMESPILL_COMPACT="1@7200": conversation 1's first turn at or after 7200 s is a DSH compaction: the context is
    rewritten from the shared head (a ~40k summary + the user turn) and its epoch goes +1."""
    out: dict[int, float] = {}
    for part in filter(None, os.environ.get("NVMESPILL_COMPACT", "").split(",")):
        c, t = part.split("@")
        out[int(c)] = float(t)
    return out


def _idles() -> dict[int, tuple[float, float]]:
    """NVMESPILL_IDLE="0@3600+4500": conversation 0's turns after 3600 s arrive 4500 s later (an idle session)."""
    out: dict[int, tuple[float, float]] = {}
    for part in filter(None, os.environ.get("NVMESPILL_IDLE", "").split(",")):
        c, rest = part.split("@")
        t, g = rest.split("+")
        out[int(c)] = (float(t), float(g))
    return out


def _branch_probs() -> dict[str, float]:
    """NVMESPILL_BRANCH="regen=0.1,edit=0.1,edit2=0.05": per DSH turn (turn >= 2), the probability that the turn is
    a regenerate of the previous turn (the same prompt re-sent: the previous reply is discarded), an edit of the last
    user message (the conversation is rewritten from where that message started), or an edit of the user message
    before it (two turns back). Unset = no branches (the workload is byte-identical to the original)."""
    raw = os.environ.get("NVMESPILL_BRANCH", "").strip()
    out: dict[str, float] = {}
    for part in filter(None, raw.split(",")):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


@dataclass
class Conv:
    conv: int
    total: int  # current committed conversation length (prompt-side)
    next_arrival: float
    turn: int = 0
    rng: random.Random = field(repr=False, default=None)  # type: ignore[assignment]
    branch_rng: random.Random | None = field(repr=False, default=None)
    branch_p: dict = field(repr=False, default_factory=dict)
    hist: list = field(repr=False, default_factory=list)   # [(prompt_len, user_len)] of the live branch
    epoch: int = 0
    compact_at: float | None = None

    def next_turn(self, now: float) -> tuple[ReqSpec, float]:
        self.turn += 1
        user = self.rng.randrange(*USER_TURN_RANGE)
        out = self.rng.randrange(*OUTPUT_RANGE)
        prompt_len = self.total + user
        branch, branch_at = "", -1
        if self.compact_at is not None and now >= self.compact_at:
            # DSH compaction: the next request carries the summary instead of the history (new epoch)
            self.compact_at = None
            self.epoch += 1
            branch, branch_at = "compact", COMMON_HEAD_LEN
            prompt_len = COMMON_HEAD_LEN + 40_000 + user
            self.hist = []
        if self.branch_rng is not None and len(self.hist) >= 2:
            u = self.branch_rng.random()
            pr, pe, pe2 = (self.branch_p.get(k, 0.0) for k in ("regen", "edit", "edit2"))
            if u < pr:                      # regenerate: the previous prompt again (its reply is discarded)
                branch, (prompt_len, user) = "regen", self.hist.pop()
                branch_at = prompt_len
            elif u < pr + pe:               # edit the last user message: rewritten from where it started
                p_prev, u_prev = self.hist.pop()
                branch, branch_at = "edit", p_prev - u_prev
                prompt_len = branch_at + user
            elif u < pr + pe + pe2 and len(self.hist) >= 3:   # edit the user message one turn further back
                self.hist.pop()
                p_prev, u_prev = self.hist.pop()
                branch, branch_at = "edit2", p_prev - u_prev
                prompt_len = branch_at + user
        if self.branch_rng is not None:
            self.hist.append((prompt_len, user))
        spec = ReqSpec(
            req_id=f"dsh{self.conv}t{self.turn}",
            kind="dsh",
            conv=self.conv,
            turn=self.turn,
            prompt_len=prompt_len,
            output_len=out,
            arrival_s=now,
            content_salt=self.conv * 1_000_003 + self.turn,
            branch_at=branch_at,
            branch=branch,
        )
        spec.spill = {"epoch": self.epoch}     # completed with ttl/session by Workload (only if NVMESPILL_TTL is set)
        gap = self.rng.uniform(*TURN_INTERVAL_RANGE)
        self.next_arrival = now + gap
        # the assistant reply (out) plus the next user turn join the context
        self.total = prompt_len + out
        return spec, self.next_arrival


class Workload:
    """Event stream of DSH turns + one-shot requests.

    Each DSH conversation owns ONE position-stable token stream: turn N+1's
    prompt is a strict extension of turn N's prompt (previous prompt + the
    assistant reply + the new user turn), so cross-turn prefix hits are
    meaningful — as in the real DSH session resend.
    """

    def __init__(self, seed: int, horizon_s: float, n_conv: int = 3):
        extra = int(os.environ.get("NVMESPILL_EXTRA_CONV", "0") or 0)   # [ttl rig] a 4th conversation of this length
        init_lens = DSH_INIT_LENS + ((extra,) if extra else ())
        n_conv = len(init_lens) if extra else n_conv
        ttl = _ttl_spec()
        comp = _compactions()
        idle = _idles()
        self.rng = random.Random(seed)
        self.horizon_s = horizon_s
        self.events: list[tuple[float, int, ReqSpec]] = []
        self.streams: dict[int, list[int]] = {}
        self._stream_rng: dict[int, random.Random] = {}
        seq = 0
        self.convs: list[Conv] = []
        bp = _branch_probs()
        self.branch_of: dict[str, str] = {}
        for c, init_len in enumerate(init_lens[:n_conv]):
            rng = random.Random(seed * 1_000_003 + c)
            conv = Conv(conv=c, total=init_len, next_arrival=1.0 + c * 37.0, rng=rng)
            conv.compact_at = comp.get(c)
            if bp:
                conv.branch_rng = random.Random(seed * 31_337 + c)
                conv.branch_p = bp
            self.convs.append(conv)
            self._stream_rng[c] = random.Random(seed * 7_777_777 + c)
            self.streams[c] = [COMMON_HEAD_TOKEN] * COMMON_HEAD_LEN
            # first turn of each conversation opens the session
            user = rng.randrange(*USER_TURN_RANGE)
            out = rng.randrange(*OUTPUT_RANGE)
            prompt_len = init_len - (out + user) if c else init_len
            prompt_len = max(prompt_len, COMMON_HEAD_LEN + 4608)
            spec = ReqSpec(
                req_id=f"dsh{c}t0",
                kind="dsh",
                conv=c,
                turn=0,
                prompt_len=prompt_len,
                output_len=out,
                arrival_s=conv.next_arrival,
                content_salt=c * 1_000_003,
            )
            conv.total = prompt_len + out
            spec.spill = {"epoch": 0}
            if bp:
                conv.hist.append((prompt_len, user))
            self.events.append((conv.next_arrival, seq, spec))
            seq += 1
            # schedule the following turns eagerly up to the horizon
            now = conv.next_arrival
            while True:
                nxt, nnext_arr = conv.next_turn(now)
                if nnext_arr > horizon_s:
                    break
                seq += 1
                self.events.append((nxt.arrival_s, seq, nxt))
                if nxt.branch:
                    self.branch_of[nxt.req_id] = nxt.branch
                now = nnext_arr
        # one-shot requests, ~one per ONESHOT_MEAN_GAP_S
        t = 30.0
        while t < horizon_s:
            seq += 1
            spec = ReqSpec(
                req_id=f"one{seq}",
                kind="one",
                conv=None,
                turn=0,
                prompt_len=self.rng.randrange(*ONESHOT_RANGE),
                output_len=self.rng.randrange(64, 512),
                arrival_s=t,
                content_salt=7_000_000 + seq,
            )
            self.events.append((t, seq, spec))
            t += self.rng.expovariate(1.0 / ONESHOT_MEAN_GAP_S)
        # [ttl rig] contract fields on the DSH requests of conversations listed in NVMESPILL_TTL (others: no field)
        for _t, _q, sp in self.events:
            if sp.kind == "dsh" and f"c{sp.conv}" in ttl:
                sp.spill = {"ttl_s": ttl[f"c{sp.conv}"], "session": f"sess-c{sp.conv}",
                            "epoch": (sp.spill or {}).get("epoch", 0)}
            else:
                sp.spill = None
        # [ttl rig] idle sessions: shift that conversation's later turns
        if idle:
            ev = []
            for t, q, sp in self.events:
                if sp.kind == "dsh" and sp.conv in idle and t > idle[sp.conv][0]:
                    t = t + idle[sp.conv][1]
                    if t > horizon_s:
                        continue
                    sp.arrival_s = t
                ev.append((t, q, sp))
            self.events = ev
        self.events.sort(key=lambda e: (e[0], e[1]))

    def __iter__(self):
        return iter(self.events)

    def _extend(self, conv: int, upto: int) -> None:
        stream = self.streams[conv]
        rng = self._stream_rng[conv]
        while len(stream) < upto:
            stream.append(rng.randrange(200, 100_000))

    def token_ids(self, spec: ReqSpec) -> list[int]:
        if spec.kind != "dsh":
            # one-shots share nothing (no shared prefix, as observed)
            rng = random.Random(spec.content_salt)
            return [rng.randrange(200, 100_000) for _ in range(spec.prompt_len)]
        if spec.branch_at >= 0:
            # a branch turn: the conversation is rewritten from branch_at (fresh tokens from the stream rng)
            del self.streams[spec.conv][spec.branch_at:]
        # extend through this turn's prompt AND its reply: the next turn's
        # prompt covers both.
        self._extend(spec.conv, spec.prompt_len + spec.output_len)
        return self.streams[spec.conv][: spec.prompt_len]


def token_ids_for(spec: ReqSpec, workload: "Workload | None" = None) -> list[int]:
    """Backward-compatible helper; the rig uses Workload.token_ids."""
    if workload is not None:
        return workload.token_ids(spec)
    n = spec.prompt_len
    head = COMMON_HEAD_LEN if spec.kind == "dsh" else 0
    rng = random.Random(spec.content_salt)
    ids = [COMMON_HEAD_TOKEN] * head
    ids.extend(rng.randrange(200, 100_000) for _ in range(n - head))
    return ids
