#!/usr/bin/env python
"""Static vs adaptive marginal block rate, MBR(k), for the enforced stacks of an adaptive run.

    python scripts/analyze_adaptive_mbr.py runs/<RUN_ID>_adaptive --out results/<RUN_ID>/adaptive_mbr
    python scripts/analyze_adaptive_mbr.py runs/adapt1_adaptive --bare A_bare --tag adaptive_edit_staged_trace_b10 \
        --out results/adapt1/adaptive_mbr

Attempt 1 of every case is the static attack against the same enforced stack, so one run gives
both columns. Headline collapse: "best" (the payload the attacker would deploy; see
guardstack/analysis/adaptive.py). Ablations written next to it: "last" and pooled.

Writes, per attack tag:
  mbr_shift.csv        stack, layer, pos, survivors/MBR/CI for static and adaptive, shift
  summary.csv          stack, mode: n_cases, ASR, effective cases, where they ended up
  mbr_ablation.csv     stack, layer: MBR with the best / last / pooled collapse
  cofire.csv           what an enforced run shows about pairs of layers (rho is not identified)
  summary.md, fig_mbr_shift.{pdf,png}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guardstack.analysis import adaptive as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bare", default="A_bare", help="stack id of the bare-agent run (defines the effective attacks)")
    ap.add_argument("--tag", default=None, help="attack tag folder, e.g. adaptive_edit_staged_trace_b10 (default: all)")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    runs = A.find_runs(args.run_dir)
    tags = sorted({t for _, t in runs if args.tag in (None, t)})
    if not tags:
        sys.exit(f"no adaptive result folders under {args.run_dir}" + (f" with tag {args.tag}" if args.tag else ""))
    for tag in tags:
        out = args.out / tag if len(tags) > 1 else args.out
        out.mkdir(parents=True, exist_ok=True)
        bare = A.load_attempts(runs[(args.bare, tag)]) if (args.bare, tag) in runs else None
        if bare is None:
            print(f"[{tag}] no {args.bare} run: MBR is over ALL cases, not over effective attacks")
        shifts, sums, abl, cof = [], [], [], []
        for (stack, t), folder in sorted(runs.items()):
            if t != tag:
                continue
            cases = A.load_attempts(folder)
            table, summary = A.mbr_shift(cases, bare, n_boot=args.n_boot)
            sums.append(summary.assign(stack=stack).reset_index())
            if stack == args.bare or table.empty:
                continue
            shifts.append(table.assign(stack=stack).reset_index())
            eff = A.effective_ids(bare, "best") if bare is not None else None
            last, _ = A.sequential_mbr(cases, "last", effective=eff, n_boot=0)
            pooled = A.pooled_mbr(cases, effective=eff)
            abl.append(pd.DataFrame({"stack": stack, "layer": table.index, "mbr_static": table.mbr_static.values,
                                     "mbr_best": table.mbr_adaptive.values, "mbr_last": last.mbr.values,
                                     "mbr_pooled": pooled.mbr.reindex(table.index).values}))
            cof.append(A.cofire_table(cases).assign(stack=stack))

        summary = pd.concat(sums)[["stack", "mode", "n_cases", "asr", "n_effective", "success_effective",
                                   "stopped_by_a_layer", "unstopped_fail", "residual_effective"]]
        summary.to_csv(out / "summary.csv", index=False)
        md = [f"# Static vs adaptive MBR(k): {args.run_dir.name} / {tag}\n",
              "MBR(k) = P(layer k is the first to stop the attack | layers 1..k-1 did not), over effective "
              "attacks (cases the same attacker breaks on the bare agent). static = attempt 1; adaptive = the "
              "attacker's best-shot payload per case.\n", "## summary\n", summary.to_markdown(index=False, floatfmt=".3f"), ""]
        if shifts:
            shift = pd.concat(shifts)[["stack", "layer", "pos", "survivors_static", "mbr_static", "lo_static",
                                       "hi_static", "survivors_adaptive", "mbr_adaptive", "lo_adaptive",
                                       "hi_adaptive", "shift"]]
            shift.to_csv(out / "mbr_shift.csv", index=False)
            pd.concat(abl).to_csv(out / "mbr_ablation.csv", index=False)
            pd.concat(cof).to_csv(out / "cofire.csv", index=False)
            md += ["## mbr_shift\n", shift.to_markdown(index=False, floatfmt=".3f"), "",
                   "## collapse ablation (adaptive MBR under each way of choosing one attempt per case)\n",
                   pd.concat(abl).to_markdown(index=False, floatfmt=".3f"), "",
                   "## cofire (observed pairs only; rho is not identified from an enforced run)\n",
                   pd.concat(cof).to_markdown(index=False), ""]
            figure(shift, out)
        (out / "summary.md").write_text("\n".join(md))
        print("\n".join(md))


def figure(shift: pd.DataFrame, out: Path):
    """One panel per stack: MBR(k) of each layer, static next to adaptive, with intervals."""
    from guardstack.analysis import report
    import matplotlib.pyplot as plt
    stacks = list(dict.fromkeys(shift["stack"]))
    fig, axes = plt.subplots(1, len(stacks), figsize=(3.4 * len(stacks), 2.8), squeeze=False, sharey=True)
    for ax, stack in zip(axes[0], stacks):
        g = shift[shift["stack"] == stack].sort_values("pos")
        x = np.arange(len(g))
        for off, col, lo, hi, color, label in ((-0.19, "mbr_static", "lo_static", "hi_static", report.SERIES[0], "static"),
                                               (0.19, "mbr_adaptive", "lo_adaptive", "hi_adaptive", report.SERIES[1], "adaptive")):
            y = g[col].fillna(0).values
            err = np.nan_to_num(np.vstack([y - g[lo].fillna(0).values, g[hi].fillna(0).values - y])).clip(min=0)
            ax.bar(x + off, y, width=0.36, color=color, label=label, edgecolor=report.SURFACE, linewidth=2)
            ax.errorbar(x + off, y, yerr=err, fmt="none", ecolor=report.INK2, elinewidth=0.8, capsize=2)
            for xi, yi, n in zip(x + off, y, g[col.replace("mbr", "survivors")].values):
                ax.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=6, color=report.INK2)
        ax.set_xticks(x, [report._short(l) for l in g.layer], rotation=30, ha="right")
        ax.set_title(f"stack {stack}", fontsize=8, color=report.INK, loc="left")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", color=report.GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("MBR(k)")
    axes[0][0].legend(frameon=False, fontsize=7)
    report._save(fig, out, "fig_mbr_shift", None)


if __name__ == "__main__":
    main()
