"""Rates with honest uncertainty.

AgentDojo cases are not independent: 949 attack cases come from 97 user tasks x 35
injection tasks, crossed within each suite, so outcomes are correlated within a user task
AND within an injection task. Reported intervals use the two-way (pigeonhole) bootstrap;
the one-way cluster bootstrap is kept for single-factor data (benign cases, by user task).
Wilson is provided for quick looks and for datasets whose items are independent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: Sequence,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile CI for ``stat(idx)`` where idx are row indices; rows are resampled by
    cluster. Returns (point, lo, hi)."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(clusters)))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        vals[b] = stat(idx)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def twoway_bootstrap(
    stat: Callable[[np.ndarray], float],
    row_cluster: Sequence,
    col_cluster: Sequence,
    stratum: Sequence | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Pigeonhole (two-way) bootstrap for crossed designs (Owen 2007): within each stratum
    (suite), resample row clusters (user tasks) and column clusters (injection tasks)
    independently; a case is repeated w_row * w_col times. Needed because AgentDojo outcomes
    correlate within injection tasks as well as within user tasks; a one-way user-task
    bootstrap under-covers badly (about 43% for a nominal 95% in our simulation)."""
    r = np.asarray(row_cluster)
    c = np.asarray(col_cluster)
    st = np.asarray(stratum) if stratum is not None else np.zeros(len(r), dtype=int)
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(r)))
    groups = []
    for sv in np.unique(st):
        m = np.flatnonzero(st == sv)
        ur, ri = np.unique(r[m], return_inverse=True)
        uc, ci = np.unique(c[m], return_inverse=True)
        groups.append((m, len(ur), ri, len(uc), ci))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for m, nr, ri, nc, ci in groups:
            wr = rng.multinomial(nr, np.full(nr, 1 / nr))
            wc = rng.multinomial(nc, np.full(nc, 1 / nc))
            w = wr[ri] * wc[ci]
            parts.append(np.repeat(m, w))
        vals[b] = stat(np.concatenate(parts)) if parts else np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def percentiles(x: Sequence[float], qs=(50, 95, 99)) -> dict[str, float]:
    x = np.asarray(list(x), dtype=float)
    if len(x) == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def gpu_cost_usd(gpu_ms: float, gpus: float = 1.0, usd_per_gpu_hour: float = 3.5) -> float:
    """Price GPU time at a stated cloud-equivalent rate so open-weight layers can be compared
    with API-priced ones (BELLS-O reports dollars). The rate is a reported assumption."""
    return gpu_ms / 3.6e6 * gpus * usd_per_gpu_hour
