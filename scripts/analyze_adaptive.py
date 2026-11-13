#!/usr/bin/env python
"""ASR@k curves for adaptive runs (static attack = attempt 1, so ASR@1 is the static number).

    python scripts/analyze_adaptive.py runs/<RUN_ID> --out results/<RUN_ID>/adaptive

Writes asr_at_k.csv (stack, attack, k, asr, lo, hi), overstatement.csv (ASR@B - ASR@1 with CI),
and fig_asr_at_k.{pdf,png}. Intervals: two-way bootstrap over user tasks x injection tasks.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guardstack.analysis.metrics import twoway_bootstrap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

groups = defaultdict(dict)   # (stack, attack_tag) -> case_id -> first_success_k
budget = {}
for f in args.run_dir.rglob("*.histories.jsonl"):
    stack, tag = f.parent.parent.name, f.parent.name
    cfg = json.loads((f.parent / "run_config.json").read_text())
    budget[(stack, tag)] = cfg["budget"]
    for line in f.read_text().splitlines():
        if line.strip():
            h = json.loads(line)
            groups[(stack, tag)][h["case_id"]] = h["first_success_k"]

rows, over = [], []
for (stack, tag), d in sorted(groups.items()):
    ids = sorted(d)
    suite = np.array([i.split("/")[0] for i in ids])
    ut = np.array(["/".join(i.split("/")[:2]) for i in ids])
    it = np.array([i.split("/")[0] + "/" + i.split("/")[2] for i in ids])
    first = np.array([d[i] if d[i] is not None else 10**9 for i in ids])
    B = budget[(stack, tag)]
    for k in range(1, B + 1):
        p, lo, hi = twoway_bootstrap(lambda ix, k=k: float((first[ix] <= k).mean()), ut, it, suite, n_boot=args.n_boot)
        rows.append({"stack": stack, "attack": tag, "k": k, "asr": p, "lo": lo, "hi": hi, "n": len(ids)})
    p, lo, hi = twoway_bootstrap(lambda ix: float((first[ix] <= B).mean() - (first[ix] <= 1).mean()), ut, it, suite,
                                 n_boot=args.n_boot)
    over.append({"stack": stack, "attack": tag, "asr_static": float((first <= 1).mean()),
                 "asr_adaptive": float((first <= B).mean()), "overstatement": p, "lo": lo, "hi": hi, "n": len(ids)})

df, ov = pd.DataFrame(rows), pd.DataFrame(over)
df.to_csv(args.out / "asr_at_k.csv", index=False)
ov.to_csv(args.out / "overstatement.csv", index=False)
print(ov.round(3).to_string(index=False))

if len(df):
    from guardstack.analysis import report  # noqa: E402  (sets the figure style)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for (key, g), color in zip(df.groupby(["stack", "attack"]), report.SERIES * 4):
        ax.plot(g.k, g.asr, marker="o", ms=3, lw=2, color=color, label=f"{key[0]} / {key[1].replace('adaptive_', '')}")
        ax.fill_between(g.k, g.lo, g.hi, color=color, alpha=0.12, lw=0)
    ax.set_xlabel("attempts k (k = 1 is the static attack)")
    ax.set_ylabel("ASR@k")
    ax.grid(color=report.GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6)
    report._save(fig, args.out, "fig_asr_at_k", None)
