"""Paper figures (matplotlib, static PDF + PNG).

Color roles follow one rule each: recall/activity = one-hue sequential (blue, light->dark);
correlation = diverging blue<->red around a gray zero; series = fixed categorical order.
Text stays in ink colors; every heatmap cell carries its value so color is never the only
channel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_br", ["#1c5cab", "#86b6ef", "#f0efec", "#f19c9b", "#b42b2b"])

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
})


def _save(fig, out: Path, name: str, watermark: str | None):
    if watermark:
        fig.text(0.99, 0.995, watermark, ha="right", va="top", fontsize=7, color="#b42b2b", alpha=0.9)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def heatmap(df: pd.DataFrame, out: Path, name: str, title: str, diverging: bool = False,
            vmin=None, vmax=None, fmt="{:.2f}", watermark: str | None = None):
    h, w = df.shape
    fig, ax = plt.subplots(figsize=(1.2 + 0.55 * w, 0.9 + 0.32 * h))
    cmap = DIV if diverging else SEQ
    if diverging:
        lim = vmax if vmax is not None else np.nanmax(np.abs(df.to_numpy()))
        vmin, vmax = -lim, lim
    im = ax.imshow(df.to_numpy(float), cmap=cmap, vmin=vmin if vmin is not None else 0,
                   vmax=vmax if vmax is not None else 1, aspect="auto")
    ax.set_xticks(range(w), df.columns, rotation=40, ha="right")
    ax.set_yticks(range(h), df.index)
    for (i, j), v in np.ndenumerate(df.to_numpy(float)):
        if np.isnan(v):
            continue
        rgba = im.cmap(im.norm(v))
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6, color="white" if lum < 0.5 else INK)
    ax.set_title(title, loc="left", color=INK, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)
    _save(fig, out, name, watermark)


def _short(layer: str) -> str:
    name, _, stage = layer.partition("@")
    abbrev = {"input": "in", "tool_output": "tool", "action": "act", "trace": "trace", "output": "out"}
    return f"{name}@{abbrev.get(stage, stage)}" if stage else name


def mbr_plot(curves: dict[str, pd.DataFrame], out: Path, name: str, watermark: str | None = None):
    """MBR(k) and residual ASR against position k; every point is labelled with its layer,
    since different orders put different layers at the same position."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for (label, c), color in zip(curves.items(), SERIES):
        x = np.arange(1, len(c) + 1)
        for ax, col in ((axes[0], "mbr"), (axes[1], "residual_asr")):
            ax.plot(x, c[col], color=color, lw=2, marker="o", ms=4, label=label)
            for xi, yi, lay in zip(x, c[col], c.index):
                ax.annotate(_short(lay), (xi, yi), xytext=(3, 4 if color == SERIES[0] else -9),
                            textcoords="offset points", fontsize=5.5, color=INK2)
    for ax in axes:
        ax.set_xticks(range(1, len(next(iter(curves.values()))) + 1))
        ax.set_xlabel("position k in the stack")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("MBR(k)")
    axes[0].set_title("Marginal block rate of the k-th layer", loc="left", color=INK, fontsize=9)
    axes[1].set_ylabel("Residual ASR after k layers")
    axes[1].set_title("Residual attack success", loc="left", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, out, name, watermark)


def pareto_plot(subsets: pd.DataFrame, front: pd.DataFrame, out: Path, name: str,
                highlight: dict[str, tuple] | None = None, fpr_cap: float | None = None,
                watermark: str | None = None):
    """Residual ASR vs cost. The line is the 2-D front (cheapest stack reaching each ASR)
    among stacks with FPR <= fpr_cap; grey dots are all subsets."""
    d = subsets if fpr_cap is None else subsets[subsets.fpr.fillna(0) <= fpr_cap]
    d = d.sort_values(["cost_ms", "residual_asr"])
    best = np.minimum.accumulate(d.residual_asr.to_numpy())
    keep = np.r_[True, best[1:] < best[:-1]]
    f2 = d[keep]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.scatter(subsets.cost_ms, subsets.residual_asr, s=3, color=GRID, label="all subsets", rasterized=True)
    lab = "cheapest stack per ASR level" + (f" (FPR <= {fpr_cap:.0%})" if fpr_cap is not None else "")
    ax.step(f2.cost_ms, f2.residual_asr, where="post", color=SERIES[0], lw=2, label=lab)
    ax.scatter(f2.cost_ms, f2.residual_asr, s=10, color=SERIES[0], zorder=4)
    offsets = [(8, 10), (8, -14), (8, 22)]
    for (labh, (x, y)), color, off in zip((highlight or {}).items(), SERIES[1:], offsets):
        ax.scatter([x], [y], s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax.annotate(labh, (x, y), xytext=off, textcoords="offset points", fontsize=7, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Guard cost per session (ms, symlog)")
    ax.set_ylabel("Residual ASR (union model)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _save(fig, out, name, watermark)


def bars(series: pd.Series, out: Path, name: str, title: str, xlabel: str, watermark: str | None = None):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(4.2, 0.3 * len(s) + 0.8))
    ax.barh(range(len(s)), s.to_numpy(), color=SERIES[0], height=0.6)
    ax.set_yticks(range(len(s)), s.index)
    for i, v in enumerate(s.to_numpy()):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=6, color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", color=INK, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6)
    _save(fig, out, name, watermark)
