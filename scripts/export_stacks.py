#!/usr/bin/env python
"""Turn the exact subset search on a shadow-all run into enforceable stacks (run 6).

    python scripts/export_stacks.py --records runs/<RUN_ID>/shadow_all --fpr-max 0.10 \
        --sizes 2 3 4 5 --out configs/stacks_optimized.yaml

For each size k it picks the stack with the lowest predicted residual ASR under the FPR cap (ties:
lower cost), plus the cheapest stack whose predicted ASR is within --tolerance of the best overall.
The output keeps the original `layers:` block, so run_stack.py / run_adaptive.py can use it via
--stack-config. Predicted numbers are written next to each stack so measured-vs-predicted
(the interaction effect) can be computed later.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import load_records  # noqa: E402
from guardstack.analysis import selection, structure  # noqa: E402
from guardstack.stacks import load_raw, make_layer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--records", type=Path, nargs="+", required=True)
ap.add_argument("--stack-config", type=Path, default=ROOT / "configs" / "stacks.yaml")
ap.add_argument("--fpr-max", type=float, default=0.10)
ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5])
ap.add_argument("--tolerance", type=float, default=0.01, help="ASR slack for the 'cheapest near-best' stack")
ap.add_argument("--latency-records", type=Path, nargs="*")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

raw = load_raw(args.stack_config)
name_to_key = {}
for key, spec in raw["layers"].items():
    try:
        name_to_key[make_layer(dict(spec)).name] = key
    except Exception as e:   # layer libs missing here: fall back to the conventional name
        name_to_key.setdefault(f'{spec["type"]}@{spec.get("stage", "")}', key)

lat = load_records(args.latency_records) if args.latency_records else None
V = structure.VerdictMatrix.from_records(load_records(args.records), cost_records=lat)
subs = selection.enumerate_subsets(V)
ok = subs[subs.fpr.fillna(0) <= args.fpr_max]
picked = []
for k in args.sizes:
    c = ok[ok.k == k].sort_values(["residual_asr", "cost_ms"])
    if len(c):
        picked.append((f"opt_k{k}", c.iloc[0]))
best = ok.residual_asr.min()
near = ok[ok.residual_asr <= best + args.tolerance].sort_values(["cost_ms", "k"])
if len(near):
    picked.append(("opt_cheapest_near_best", near.iloc[0]))

out = {"layers": raw["layers"], "stacks": []}
for sid, row in picked:
    keys = [name_to_key[n] for n in row.layers if n in name_to_key]
    missing = [n for n in row.layers if n not in name_to_key]
    if missing:
        raise SystemExit(f"{sid}: no config key for layers {missing}")
    out["stacks"].append({"id": sid, "layers": keys,
                          "description": f"predicted residual ASR {row.residual_asr:.3f}, FPR {row.fpr:.3f}, "
                                         f"cost {row.cost_ms:.0f} ms (union model, {V.notes.get('cost_source')})"})
args.out.write_text(yaml.safe_dump(out, sort_keys=False))
for s in out["stacks"]:
    print(f'{s["id"]:24s} {s["layers"]}  -- {s["description"]}')
