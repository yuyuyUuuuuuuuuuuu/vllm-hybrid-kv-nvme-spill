# SPDX-License-Identifier: AGPL-3.0-only
"""One line per correctness/fault run of the rig (results/fix/*.json): what was injected, what the connector did, and
the end state (drain: every block back in the free list except quarantined ones; requests finished; no pins)."""
import glob
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "fix")
names = sys.argv[2:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(R, "*.json"))
                                if not os.path.basename(p).startswith(("m_", "x_")))
for n in names:
    p = os.path.join(R, n + ".json")
    if not os.path.exists(p):
        print(f"{n}: MISSING")
        continue
    d = json.load(open(p))
    s = d.get("spill") or {}
    ws = s.get("worker_stats") or [{}, {}]
    print(f"{n}: faults={s.get('faults_injected')} falls={d['dsh_falls_le23040']} "
          f"restores ok={s.get('restore_ok')} failed={s.get('restore_failed')} fallback={s.get('restore_fallback')} "
          f"short={s.get('restore_short')} | saves ok={s.get('rp_saved')} failed={s.get('rp_save_failed')} "
          f"jobs_failed={s.get('jobs_failed')} | timeouts save={s.get('save_timeouts')} load={s.get('load_timeouts')} "
          f"late={s.get('late_answers')} | breaker={'OPEN' if s.get('breaker') else 'closed'} "
          f"worker_exc={s.get('worker_exc')} thread_exc={[w.get('thread_exc') for w in ws]} "
          f"threads_alive={s.get('worker_threads_alive')} | verify {s.get('verify_blocks')}/{s.get('verify_fail')} "
          f"restored_verified={s.get('restored_blocks_verified')} files {s.get('file_verify')}/{s.get('file_verify_fail')} "
          f"| END free={d.get('end_free_blocks')}/582 ref={d.get('end_ref_nonnull')} pinned={s.get('connector_pinned_end')} "
          f"quarantined={s.get('quarantined_blocks_end')} unfinished={d.get('pending_unfinished')} "
          f"save_jobs={s.get('save_jobs_end')} loading={s.get('loading_end')} drain_steps={d.get('drain_steps')} "
          f"inv={len(d.get('invariant_errors', []))} hashviol={d.get('hit_hash_violations')} "
          f"acct={d.get('accounting_violations')} errors={d.get('errors', [])[:1]} aborted={len(s.get('aborted') or [])}")
    if s.get("breaker"):
        print(f"    breaker reason: {s['breaker']}")
