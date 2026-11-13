#!/usr/bin/env python
"""Export an adaptive attacker's best-shot payload per case, for replay against another stack.

    python scripts/export_payloads.py runs/adapt1_adaptive --stack E --out runs/adapt1_adaptive/payloads_E.jsonl
    # then: every layer's verdict on those payloads (the adaptive verdict matrix)
    EXTRA_ARGS="--payload-file runs/adapt1_adaptive/payloads_E.jsonl" RUN_ID=adapt1_replayE STACKS=shadow_all ... run_stack.sbatch
    python scripts/analyze.py --records runs/adapt1_replayE/shadow_all --out results/adapt1_replayE/analysis

An enforced adaptive run cannot give pairwise redundancy (a layer that ends the session hides the
later layers' verdicts). Replaying the payloads against the shadow-all stack, where every layer
observes and none acts, gives the full layer x case matrix under the adaptive attack, so the whole
static analysis (MBR for any order, rho, Shapley, subset selection) applies unchanged.

One payload per case: the attacker's first successful attempt against --stack, else the attempt
that got furthest (guardstack.analysis.adaptive.representative, mode "best"). --mode static
exports attempt 1 instead (a check: the replay must then reproduce the static shadow-all run).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guardstack.analysis import adaptive as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--stack", required=True, help="stack the payloads were adapted against, e.g. E")
ap.add_argument("--tag", default=None, help="attack tag folder (needed only if the stack has several)")
ap.add_argument("--mode", choices=A.MODES, default="best")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

folders = [f for (s, t), f in A.find_runs(args.run_dir).items() if s == args.stack and args.tag in (None, t)]
if len(folders) != 1:
    sys.exit(f"expected one result folder for stack {args.stack}, found {len(folders)}: {[str(f) for f in folders]}")
cases = A.load_attempts(folders[0])
rows, n_missing = [], 0
for cid in sorted(cases):
    a = A.representative(cases[cid], args.mode)
    if a is None or not a.get("payload"):
        n_missing += 1
        continue
    rows.append({"case_id": cid, "payload": a["payload"], "source_stack": args.stack, "source_tag": folders[0].name,
                 "attempt": a["k"], "success_on_source": a["success"], "outcome_on_source": A.outcome(a)})
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"wrote {len(rows)} payloads to {args.out} ({sum(r['success_on_source'] for r in rows)} succeeded against "
      f"{args.stack}; {n_missing} cases skipped: no usable attempt)")
