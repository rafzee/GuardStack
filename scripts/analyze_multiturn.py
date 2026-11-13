#!/usr/bin/env python
"""Multi-turn success rates per stack (bootstrap over goals), turns-to-success, and which layers acted.
    python scripts/analyze_multiturn.py runs/<RUN_ID> --out results/<RUN_ID>/multiturn
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guardstack.analysis.metrics import cluster_bootstrap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for f in a.run_dir.rglob("multiturn/*.jsonl") for l in f.read_text().splitlines() if l.strip()]
df = pd.DataFrame([r for r in rows if r.get("error") is None and r.get("success") is not None])
summ = []
for sid, g in df.groupby("stack_id"):
    y = g.success.astype(float).to_numpy()
    p, lo, hi = cluster_bootstrap(lambda ix: float(y[ix].mean()), g.goal_id.to_numpy(), n_boot=2000)
    fires = Counter(l for f in g.fires for l in f)
    summ.append({"stack": sid, "n": len(g), "goals": g.goal_id.nunique(), "success": p, "lo": lo, "hi": hi,
                 "median_turns_success": float(g[g.success].turns.median()) if g.success.any() else np.nan,
                 "mean_backtracks": float(g.backtracks.mean()),
                 "layers_acting": ", ".join(f"{k}={v}" for k, v in fires.most_common(6))})
out = pd.DataFrame(summ)
out.to_csv(a.out / "multiturn_summary.csv", index=False)
df.groupby(["stack_id", "goal_id"]).success.mean().unstack(0).to_csv(a.out / "multiturn_by_goal.csv")
print(out.round(3).to_string(index=False))
