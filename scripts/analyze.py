#!/usr/bin/env python
"""Run the full structural analysis on a verdict matrix and write tables + figures.

    # real data: a shadow-all run on the bare agent
    python scripts/analyze.py --records runs/shadow_all.jsonl --out results/analysis
    # synthetic preview (figures watermarked SYNTHETIC)
    python scripts/analyze.py --synthetic --out results/synthetic_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardstack.analysis import assignment, report, selection, structure  # noqa: E402
from guardstack.analysis.metrics import twoway_bootstrap  # noqa: E402

# Nested stack B->E order; real layer names (configs/stacks.yaml) and synthetic ones, filtered by presence.
NESTED_E = ["promptguard2@input", "promptguard2@tool_output", "llamaguard4@output", "llamaguard@output",
            "tool_gate@action", "provenance_gate@action", "alignment_check@trace"]


def load_records(paths) -> list[dict]:
    """Accepts files or directories; skips *.events.jsonl and any line that is not a SessionRecord."""
    out = []
    files = []
    for p in ([paths] if isinstance(paths, Path) else paths):
        p = Path(p)
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    for f in files:
        if f.name.endswith(".events.jsonl"):
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "session_id" in r and "stack_layers" in r:
                    out.append(r)
    return out


def nested_order(stack_config: Path, stack_id: str, layers: list[str]) -> list[str]:
    """Layer order of a configured stack (transform layers dropped: they have no verdicts)."""
    from guardstack.stacks import load_stacks
    st = load_stacks(stack_config)[stack_id]
    names = [l.name for l in st.build() if getattr(l, "family", "") != "transform"]
    missing = [n for n in names if n not in layers]
    if missing:
        raise SystemExit(f"stack {stack_id} layers missing from the verdict matrix: {missing}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, nargs="*", help="shadow-all SessionRecord files or directories")
    ap.add_argument("--latency-records", type=Path, nargs="*", help="latency-mode records: per-layer costs")
    ap.add_argument("--stack-config", type=Path, default=Path(__file__).resolve().parent.parent / "configs" / "stacks.yaml")
    ap.add_argument("--nested-stack", default="E", help="stack whose layer order defines the MBR curve")
    ap.add_argument("--layer-meta", type=Path, help="YAML/JSON {layer: {stage, family}}")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--order", nargs="*", help="layer order for the MBR curve (default: nested stack B..E)")
    ap.add_argument("--prevalence", type=float, default=0.05, help="attack share of traffic for cost ordering")
    args = ap.parse_args()

    if args.synthetic:
        from guardstack.analysis.synthetic import CATEGORIES, make_synthetic
        V = structure.VerdictMatrix  # for type checkers
        V = make_synthetic()
        cats = CATEGORIES
        wm = "SYNTHETIC DATA - NOT A RESULT"
    else:
        meta = None
        if args.layer_meta:
            import yaml
            meta = yaml.safe_load(args.layer_meta.read_text())
        else:  # stage and family straight from the layer classes in the stack config
            from guardstack.stacks import load_stacks
            meta = {}
            for st in load_stacks(args.stack_config).values():
                try:
                    for l in st.build():
                        meta[l.name] = {"stage": l.stage.value, "family": l.family}
                except Exception as e:  # a layer whose deps are missing here; its family stays "other"
                    print(f"note: could not build stack {st.id} for metadata: {type(e).__name__}")
        lat = load_records(args.latency_records) if args.latency_records else None
        V = structure.VerdictMatrix.from_records(load_records(args.records), layer_meta=meta, cost_records=lat)
        cats = sorted(set(V.category))
        wm = None if lat else "COSTS FROM THROUGHPUT RUN - NOT FOR REPORTING"
        print("verdict matrix notes:", V.notes)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tables = {}

    # 1. per-layer table and activity ("which layer is active where")
    lt = structure.layer_table(V)
    act = structure.activity_matrix(V, cats)
    tables["layers"] = lt
    tables["activity"] = act
    report.heatmap(act, out, "fig_activity", "Recall by layer and attack category", watermark=wm)

    # 2. redundancy
    rho = structure.redundancy_rho(V)
    lift = structure.miss_lift(V)
    rho_s = structure.stratified_rho(V)
    tables["rho"], tables["rho_within_category"], tables["lift"] = rho, rho_s, lift
    report.heatmap(rho_s, out, "fig_rho_within", "Scope-adjusted redundancy (within-category rho)", diverging=True,
                   vmax=1.0, watermark=wm)
    report.heatmap(rho, out, "fig_rho", "Redundancy rho(i,j) = corr(miss_i, miss_j)", diverging=True, vmax=1.0,
                   watermark=wm)
    report.heatmap(np.log2(lift.clip(lower=1e-3)), out, "fig_lift",
                   "log2 miss lift (0 = independent holes, >0 = holes line up)", diverging=True, vmax=3, fmt="{:.1f}",
                   watermark=wm)
    tables["dominated"] = pd.DataFrame(structure.dominated_layers(V, tol=0.02), columns=["layer", "dominated_by"])

    # 3. contribution: unique (leave-one-out) and Shapley
    report.bars(lt["shapley"], out, "fig_shapley", "Shapley share of attacks stopped", "share of effective attacks",
                watermark=wm)

    # 4. MBR along the nested stack (and its reverse, to show order dependence)
    if args.order:
        order = args.order
    elif args.synthetic:
        order = [l for l in NESTED_E if l in V.layers]
    else:
        order = nested_order(args.stack_config, args.nested_stack, V.layers)
    if order:
        c1 = structure.mbr_curve(V, order)
        c2 = structure.mbr_curve(V, order[::-1])
        tables["mbr_nested"], tables["mbr_reversed"] = c1, c2
        report.mbr_plot({"nested B to E": c1, "reversed": c2}, out, "fig_mbr", watermark=wm)
        tables["independence_gap"] = pd.DataFrame([structure.independence_gap(V, order)])

        # bootstrap CI for MBR of the last layer in the nested order
        last = V.idx([order[-1]])[0]
        prev = V.idx(order[:-1])

        M = V.stop_or_catch

        def mbr_last(ix):
            alive = ~np.any(M[prev][:, ix], axis=0)
            return float(M[last, ix][alive].mean()) if alive.any() else np.nan

        inj = V.inj_cluster if V.inj_cluster is not None else np.zeros(V.n_eff)
        p, lo, hi = twoway_bootstrap(mbr_last, V.cluster, inj, V.suite, n_boot=1000)
        tables["mbr_last_ci"] = pd.DataFrame([{"layer": order[-1], "mbr": p, "lo": lo, "hi": hi}])

    # 5. exact subset search + Pareto front
    subs = selection.enumerate_subsets(V)
    front = selection.pareto_front(subs)
    tables["pareto_front"] = front.drop(columns=["mask"])
    tables["best_by_k"] = selection.best_subsets(subs).drop(columns=["mask"])
    tables["best_by_k_fpr10"] = selection.best_subsets(subs, fpr_max=0.10).drop(columns=["mask"])
    greedy = selection.greedy_select(V, k=5)
    tables["greedy5"] = pd.DataFrame({"layer": greedy})

    # 6. Hungarian: ownership, and what it costs to build a stack from it
    own = assignment.ownership(V, cats)
    tables["hungarian_ownership"] = own
    hv = assignment.hungarian_vs_optimal(V, cats, subsets=subs)
    tables["hungarian_vs_optimal"] = hv
    lf = [l for l in V.layers if l.split("@")[0] in ("promptguard2", "alignment_check", "codeshield", "regex")]
    nemo = [l for l in V.layers if l.startswith("nemo_")]
    if lf and nemo:
        tables["counterparts_lf_vs_nemo"] = assignment.counterpart_matching(V, lf, nemo)

    hl = {f"Hungarian owners (FPR {hv.loc['hungarian_owners', 'fpr']:.0%})":
          (hv.loc["hungarian_owners", "cost_ms"], hv.loc["hungarian_owners", "residual_asr"]),
          f"cheapest stack at least as good (FPR {hv.loc['cheapest_at_least_as_good', 'fpr']:.0%})":
          (hv.loc["cheapest_at_least_as_good", "cost_ms"], hv.loc["cheapest_at_least_as_good", "residual_asr"])}
    report.pareto_plot(subs, front, out, "fig_pareto", highlight=hl, watermark=wm)
    report.pareto_plot(subs, front, out, "fig_pareto_fpr15", fpr_cap=0.15, watermark=wm)

    # 7. cost-optimal ordering of the best 5-layer stack
    best5 = list(tables["best_by_k"].query("k == 5").layers.iloc[0]) if (tables["best_by_k"].k == 5).any() else greedy
    o, ec = selection.optimal_order(best5, V, attack_prevalence=args.prevalence)
    tables["best5_order"] = pd.DataFrame({"order": o, "expected_cost_ms": ec})

    for k, t in tables.items():
        t.to_csv(out / f"{k}.csv")
    with open(out / "summary.md", "w") as f:
        if wm:
            f.write(f"# {wm}\n\n")
        f.write(f"Verdict-matrix notes: {V.notes}\n\n")
        for k, t in tables.items():
            f.write(f"## {k}\n\n{t.to_markdown(floatfmt='.3f')}\n\n")
    print(f"wrote {len(tables)} tables and figures to {out}")


if __name__ == "__main__":
    main()
