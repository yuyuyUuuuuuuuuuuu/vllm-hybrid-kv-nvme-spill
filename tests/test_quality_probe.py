# SPDX-License-Identifier: AGPL-3.0-only
"""B5 regression: spill_quality_probe's restore verdict FAILS unless a restore is proven (no network, no server).
usage: python3 tests/test_quality_probe.py"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.modules["conc_quality_probe"] = types.ModuleType("conc_quality_probe")   # the probe's client (not used here)
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import spill_quality_probe as q  # noqa: E402

R = []


def check(name, ok, d=""):
    R.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {d}")


pq1 = 86000
walls = dict(wall_r=14.0, wall_h=12.5, wall_f1=58.0)      # restore ~1.5 s extra, re-read ~45 s
ok, why = q.restore_verdict(None, pq1, [], **walls)
check("no_cached_no_log_line_fails", not ok, why)
ok, why = q.restore_verdict(None, pq1, None, **walls)
check("log_unreadable_fails", not ok, why)
ok, why = q.restore_verdict(None, pq1, [("r1", 82944, 0.6)], **walls)
check("log_line_with_deep_rp_and_fast_R_passes", ok, why)
ok, why = q.restore_verdict(None, pq1, [("r1", 13824, 0.1)], **walls)
check("shallow_restore_fails", not ok, why)
ok, why = q.restore_verdict(None, pq1, [("r1", 82944, 0.6)], wall_r=57.0, wall_h=12.5, wall_f1=58.0)
check("log_line_but_R_as_slow_as_full_prefill_fails", not ok, why)
ok, why = q.restore_verdict(86100, pq1, None, **walls)
check("cached_tokens_reported_and_fast_passes", ok, why)
ok, why = q.restore_verdict(4608, pq1, None, **walls)
check("cached_tokens_shallow_fails", not ok, why)
line = "INFO 10-13 [glm53-nvme-spill] restore ready req=cmpl-abc-0 rp=82944 in 0.612 s"
m = q.RESTORE_RE.search(line)
check("log_regex_matches_connector_line", m is not None and m.group(2) == "82944", line)
import subprocess  # noqa: E402
ev = q.restore_events("printf '%s\\n' '" + line + "' x{since}x{until}", 1.0, 2.0)
check("restore_events_parses_command_output", ev == [("cmpl-abc-0", 82944, 0.612)], f"{ev}")
check("restore_events_none_on_failing_command", q.restore_events("false {since} {until}", 1.0, 2.0) is None)
# --- restart variant (2026-10-14): prepare writes the salt/sizes, restart re-uses them and never resets the cache
import json as _json  # noqa: E402
import tempfile  # noqa: E402

calls = []


class FakeCli:
    url = "http://x"

    def __init__(self, *a):
        pass

    def metrics_busy(self):
        return 0, 0

    def _req(self, path):
        return "vllm:spec_decode_num_accepted_tokens_total 10\nvllm:spec_decode_num_draft_tokens_total 20\n"

    def complete(self, prompt, n, salt, lp):
        calls.append((len(prompt), n, salt))
        return {"usage": {}, "ids": [1, 2, 3], "top": [], "prompt_ids": list(range(len(prompt))), "secs": 1.0}


cq = sys.modules["conc_quality_probe"]
cq.Client, cq.api_key, cq.DEF_URL, cq.DEF_ENV, cq.DEF_MODEL = FakeCli, (lambda e: "k"), "u", "e", "m"
cq.text_prompts = lambda ns: ["x" * n for n in ns]
cq.ids_of, cq.kl_top = (lambda t: {}), (lambda a, b: 0.0)
resets = []
q.reset = lambda cli: resets.append(1)
st = os.path.join(tempfile.mkdtemp(), "st.json")
sys.argv = ["p", "--phase", "prepare", "--state", st, "--ctx", "100", "--q1", "20", "--q2", "5", "--settle", "0"]
rc = q.main()
tag = _json.load(open(st))["tag"]
check("prepare_runs_W1_W2_writes_state_no_reset", rc == 0 and len(calls) == 2 and not resets
      and all(c[2] == tag + "-S" for c in calls), f"{calls}")
calls.clear()
sys.argv = ["p", "--phase", "restart", "--state", st, "--log-cmd", "true {since} {until}"]
rc = q.main()
check("restart_reuses_salt_and_sizes_no_W_no_reset", not resets and calls[0] == (125, 512, tag + "-S")
      and calls[1][2] == tag + "-S" and calls[2][2] == tag + "-F1", f"{calls[:3]}")
calls.clear()
sys.argv = ["p", "--phase", "restart"]
try:
    q.main()
    check("restart_without_state_refused", False)
except SystemExit:
    check("restart_without_state_refused", True)
# --- several runs (2026-10-14 finish): prepare N salts, restart measures each, averaged verdict
calls.clear()
st3 = os.path.join(tempfile.mkdtemp(), "st3.json")
sys.argv = ["p", "--phase", "prepare", "--state", st3, "--ctx", "100", "--q1", "20", "--q2", "5", "--settle", "0",
            "--runs", "3"]
rc = q.main()
s3 = _json.load(open(st3))
check("prepare_runs3_writes_3_tags_6_requests", rc == 0 and len(s3["tags"]) == 3 and len(set(s3["tags"])) == 3
      and len(calls) == 6 and [c[2] for c in calls] == [t + "-S" for t in s3["tags"] for _ in (0, 1)] and not resets,
      f"{s3['tags']}")
calls.clear()
sys.argv = ["p", "--phase", "restart", "--state", st3, "--log-cmd", "true {since} {until}"]
rc = q.main()
rs = [c[2] for c in calls if c[1] == 512 and c[2].endswith("-S")]
check("restart_runs3_measures_every_tag_no_reset", not resets and rs == [t + "-S" for t in s3["tags"] for _ in (0, 1)],
      f"{rs}")
check("restart_runs3_fails_without_restore_evidence", rc == 1)
sys.argv = ["p", "--phase", "both", "--runs", "2"]
try:
    q.main()
    check("runs_with_phase_both_refused", False)
except SystemExit:
    check("runs_with_phase_both_refused", True)


def run(kl_r, kl_h, kl_f2, ok=True, acc_r=0.7, acc_h=0.7):
    return {"tag": "t", "ok_restore": ok, "why": "w",
            "cmp": {"R": {"kl": kl_r}, "H": {"kl": kl_h}, "F2": {"kl": kl_f2}},
            "res": {"R": {"acc": acc_r}, "H": {"acc": acc_h}}}


v = q.aggregate_verdict([run(0.02, 0.01, 0.02), run(0.05, 0.02, 0.03), run(0.01, 0.02, 0.02), run(0.02, 0.01, 0.01)])
check("aggregate_one_noisy_run_mean_in_band_passes", v[0] and v[2] and v[3], v[4])
v = q.aggregate_verdict([run(0.05, 0.01, 0.01)] * 4)
check("aggregate_mean_above_band_fails", v[0] and not v[2], v[4])
v = q.aggregate_verdict([run(0.01, 0.01, 0.01), run(0.01, 0.01, 0.01, ok=False)])
check("aggregate_one_unproven_restore_fails", not v[0], v[1])
v = q.aggregate_verdict([run(0.01, 0.01, 0.01, acc_r=0.60, acc_h=0.70)] * 4)
check("aggregate_acceptance_drop_fails", not v[3], v[4])
v = q.aggregate_verdict([run(0.01, 0.01, 0.01, acc_r=float("nan"), acc_h=0.7), run(0.01, 0.01, 0.01, acc_r=0.69)])
check("aggregate_nan_acceptance_ignored", v[3], v[4])
print(f"SUMMARY {sum(R)}/{len(R)} passed")
sys.exit(0 if all(R) else 1)
