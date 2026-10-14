# SPDX-License-Identifier: AGPL-3.0-only
"""Aggregate the 4h replay matrix (same seed = same request stream; off = production, on = spill connector)."""
import glob
import json
import os
import statistics as st
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ON = sys.argv[2] if len(sys.argv) > 2 else "m_on"      # m_on (sync) | x_on (async completion, rank skew)


def load(p):
    with open(p) as f:
        return json.load(f)


rows = []
for off_p in sorted(glob.glob(os.path.join(R, "m_off_s*.json"))):
    s = off_p.rsplit("_s", 1)[1].split(".")[0]
    on_p = os.path.join(R, f"{ON}_s{s}.json")
    if not os.path.exists(on_p):
        continue
    off, on = load(off_p), load(on_p)
    sp = on["spill"] or {}
    hours = on["sim_s"] / 3600
    rows.append({
        "seed": int(s),
        "dsh_reread_off": off["dsh_re_read_tokens"], "dsh_reread_on": on["dsh_re_read_tokens"],
        "reread_off": off["re_read_tokens"], "reread_on": on["re_read_tokens"],
        "falls_off": off["dsh_falls_le23040"], "falls_on": on["dsh_falls_le23040"],
        "stall_off": off["stalled_s_est"], "stall_on": on["stalled_s_est"],
        "adm_off": off["admissions"], "adm_on": on["admissions"],
        "restores": sp.get("restore_ok"), "restore_failed": sp.get("restore_failed"),
        "restore_short": sp.get("restore_short"), "restored_tokens": sp.get("restored_tokens"),
        "loaded_blocks": sp.get("restore_loaded_blocks"), "pinned_blocks": sp.get("restore_pinned_blocks"),
        "write_gib_per_h": round(sp.get("save_bytes", 0) / (1 << 30) / hours, 1),
        "rp_saved": sp.get("rp_saved"), "skipped_first": sp.get("rp_skipped_first"),
        "disk_gib": sp.get("disk_gib_accounted"), "verify_blocks": sp.get("verify_blocks"),
        "verify_fail": sp.get("verify_fail"), "giveups": sp.get("restore_space_giveup"),
        "inv_on": len(on["invariant_errors"]), "hash_viol_on": on["hit_hash_violations"],
        "acct_viol_on": on["accounting_violations"],
        "restore_started": sp.get("restore_started"), "restore_admitted": sp.get("restore_admitted"),
        "pinned_peak": sp.get("pinned_peak"), "breaker": sp.get("breaker"),
        "file_verify": sp.get("file_verify"), "file_verify_fail": sp.get("file_verify_fail"),
        "rp_skipped_pins": sp.get("rp_skipped_pins"), "pinned_excl_peak": sp.get("pinned_excl_peak"),
        "jobs_failed": sp.get("jobs_failed"), "save_timeouts": sp.get("save_timeouts"),
        "load_timeouts": sp.get("load_timeouts"), "preempt_on": on.get("preemptions", on.get("preempt_count")),
    })
for r in rows:
    print(json.dumps(r))
if rows:
    def m(k):
        return st.mean(r[k] for r in rows)
    print("MEAN dsh_reread off=%.0f on=%.0f (%.1f%%) | all reread off=%.0f on=%.0f (%.1f%%) | falls off=%.1f on=%.1f |"
          " stall_s off=%.0f on=%.0f | write %.1f GiB/h | restores %.1f (failed %d, short %d) | verify_fail %d" % (
              m("dsh_reread_off"), m("dsh_reread_on"), 100 * (m("dsh_reread_on") / m("dsh_reread_off") - 1),
              m("reread_off"), m("reread_on"), 100 * (m("reread_on") / m("reread_off") - 1),
              m("falls_off"), m("falls_on"), m("stall_off"), m("stall_on"), m("write_gib_per_h"), m("restores"),
              sum(r["restore_failed"] for r in rows), sum(r["restore_short"] for r in rows),
              sum(r["verify_fail"] for r in rows)))
    print("MEAN restores started %.1f admitted %.1f | pinned_peak max %s | breaker trips %d | file_verify_fail %s | "
          "excl_peak max %s | skipped(pins) %s | jobs_failed %s | timeouts save %s load %s" % (
              m("restore_started"), m("restore_admitted"), max(r["pinned_peak"] or 0 for r in rows),
              sum(1 for r in rows if r["breaker"]), sum(r["file_verify_fail"] or 0 for r in rows),
              max(r["pinned_excl_peak"] or 0 for r in rows),
              sum(r["rp_skipped_pins"] or 0 for r in rows), sum(r["jobs_failed"] or 0 for r in rows),
              sum(r["save_timeouts"] or 0 for r in rows), sum(r["load_timeouts"] or 0 for r in rows)))
