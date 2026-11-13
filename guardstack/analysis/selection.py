"""Which combination of layers is best, and in what order.

Two separate problems, with different correct tools:

1. *Selection* (which layers): with detection-only layers that block on fire, the attacks a
   stack stops are the union of what its layers stop on the shared trajectory. Coverage of a
   union is a monotone submodular function, so (a) greedy selection has the classic
   (1 - 1/e) guarantee, and (b) for the pool sizes we have (<= ~20 layers) we can simply
   enumerate every subset exactly and report the full Pareto front of
   (residual ASR, false-positive rate, cost). No heuristic is needed.

2. *Ordering* (in what order): ordering does not change which attacks a detector-only stack
   catches; it changes expected cost when a block ends the session early. Minimising
   expected cost of a filter pipeline is the pipelined-filter-ordering problem (Garey 1973
   for independent filters: sort by cost / P(block); Babu et al. 2004 / Munagala et al. 2005
   greedy for correlated filters). In an agent, stages fix most of the order, so only
   layers sharing a stage are permuted.

The union assumption is what enforced stack runs test: ``interaction()`` compares the
predicted residual ASR against the measured one. Any gap is a real interaction effect
(a redaction that changes later behaviour, an adaptive attacker routing around the stack,
a transform layer that changes what later detectors see).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .structure import VerdictMatrix


def _pack(rows: np.ndarray) -> np.ndarray:
    return np.packbits(rows, axis=1)


def _popcount(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=-1).sum(axis=-1)


def enumerate_subsets(V: VerdictMatrix, max_layers: int = 20, cost: dict[str, float] | None = None) -> pd.DataFrame:
    """Exact evaluation of every non-empty subset (plus the empty one) under the union model.

    residual_asr = |effective attacks missed by every layer in S| / n_attacks
    fpr          = |benign cases where some layer in S fires| / n_benign
    cost_ms      = sum of member costs (every layer runs; upper bound when blocks abort early)
    """
    L = V.L
    if L > max_layers:
        raise ValueError(f"{L} layers: use greedy_select instead (2^{L} subsets)")
    cost = cost or V.cost_ms
    c = np.array([cost.get(l, 0.0) for l in V.layers])
    C = _pack(V.stop_or_catch) if V.n_eff else np.zeros((L, 0), np.uint8)
    F = _pack(V.fp) if V.fp.shape[1] else np.zeros((L, 0), np.uint8)
    n_sub = 1 << L
    uc = np.zeros((n_sub, C.shape[1]), np.uint8)
    uf = np.zeros((n_sub, F.shape[1]), np.uint8)
    csum = np.zeros(n_sub)
    size = np.zeros(n_sub, dtype=int)
    for m in range(1, n_sub):
        low = (m & -m).bit_length() - 1
        prev = m & (m - 1)
        uc[m] = uc[prev] | C[low]
        uf[m] = uf[prev] | F[low]
        csum[m] = csum[prev] + c[low]
        size[m] = size[prev] + 1
    caught = _popcount(uc) if C.shape[1] else np.zeros(n_sub)
    fired_b = _popcount(uf) if F.shape[1] else np.zeros(n_sub)
    n_b = V.fp.shape[1]
    df = pd.DataFrame({
        "mask": np.arange(n_sub),
        "k": size,
        "residual_asr": (V.n_eff - caught) / max(1, V.n_attacks),
        "fpr": fired_b / n_b if n_b else np.nan,
        "cost_ms": csum,
    })
    df["layers"] = [tuple(V.layers[i] for i in range(L) if m >> i & 1) for m in range(n_sub)]
    return df


def pareto_front(df: pd.DataFrame, cols: Sequence[str] = ("residual_asr", "fpr", "cost_ms")) -> pd.DataFrame:
    """Non-dominated rows when minimising every column in ``cols``."""
    X = df[list(cols)].to_numpy(float)
    X = np.where(np.isnan(X), 0.0, X)
    order = np.lexsort(X.T[::-1])
    keep = []
    best: list[np.ndarray] = []
    for i in order:
        x = X[i]
        if any(np.all(b <= x) and np.any(b < x) for b in best):
            continue
        if any(np.all(b == x) for b in best):
            continue
        best.append(x)
        keep.append(i)
    return df.iloc[keep].sort_values(list(cols))


def best_subsets(df: pd.DataFrame, fpr_max: float | None = None, cost_max: float | None = None,
                 by_k: bool = True) -> pd.DataFrame:
    d = df
    if fpr_max is not None:
        d = d[d.fpr.fillna(0) <= fpr_max]
    if cost_max is not None:
        d = d[d.cost_ms <= cost_max]
    if by_k:
        return d.sort_values(["k", "residual_asr", "fpr", "cost_ms"]).groupby("k").head(1)
    return d.sort_values(["residual_asr", "fpr", "cost_ms"]).head(10)


def greedy_select(V: VerdictMatrix, k: int, fp_weight: float = 0.0, cost_weight: float = 0.0) -> list[str]:
    """Greedy on marginal (coverage - fp_weight*added FP - cost_weight*cost). With weights 0
    this is the (1 - 1/e)-approximate max-coverage greedy."""
    chosen: list[int] = []
    caught = np.zeros(V.n_eff, bool)
    fired = np.zeros(V.fp.shape[1], bool)
    for _ in range(min(k, V.L)):
        best, best_gain = None, -np.inf
        for i in range(V.L):
            if i in chosen:
                continue
            g = (V.stop_or_catch[i] & ~caught).sum() / max(1, V.n_attacks)
            if V.fp.shape[1]:
                g -= fp_weight * (V.fp[i] & ~fired).sum() / V.fp.shape[1]
            g -= cost_weight * V.cost_ms.get(V.layers[i], 0.0)
            if g > best_gain:
                best, best_gain = i, g
        if best is None:
            break
        chosen.append(best)
        caught |= V.stop_or_catch[best]
        if V.fp.shape[1]:
            fired |= V.fp[best]
    return [V.layers[i] for i in chosen]


# ---------------------------------------------------------------------- ordering
def expected_cost(order: Sequence[str], V: VerdictMatrix, attack_prevalence: float = 0.05,
                  cost: dict[str, float] | None = None) -> float:
    """Expected guard cost per session when each block aborts the session, under a traffic
    mix with the given attack prevalence (benign cases use the FP matrix).

    Approximation (stated in the paper): each layer is a per-session cost lump run in stage
    order, and the attack share is represented by effective attacks. The exact version
    replays the shadow event logs in seq order (interleaved stages); see
    scripts/analyze.py --events for the replay estimate."""
    cost = cost or V.cost_ms
    idx = V.idx(order)
    tot = 0.0
    for rows, w in ((V.stop_or_catch, attack_prevalence), (V.fp, 1 - attack_prevalence)):
        if rows.shape[1] == 0:
            continue
        alive = np.ones(rows.shape[1], bool)
        for name, i in zip(order, idx):
            tot += w * alive.mean() * cost.get(name, 0.0)
            alive &= ~rows[i]
    return tot


def optimal_order(layers: Sequence[str], V: VerdictMatrix, attack_prevalence: float = 0.05,
                  respect_stages: bool = True, exhaustive_max: int = 8) -> tuple[list[str], float]:
    """Cheapest order. Stages are kept in time order; within each stage, exhaustive search
    for <= exhaustive_max layers, else the greedy conditional cost/P(block) rule."""
    from ..types import Stage

    def stage_key(l):
        try:
            return Stage(V.stage.get(l, "input")).order
        except ValueError:
            return 0

    groups: list[list[str]] = []
    if respect_stages:
        for _, g in itertools.groupby(sorted(layers, key=stage_key), key=stage_key):
            groups.append(list(g))
    else:
        groups = [list(layers)]

    order: list[str] = []
    for g in groups:
        if len(g) <= exhaustive_max:
            best = min(itertools.permutations(g), key=lambda p: expected_cost(order + list(p), V, attack_prevalence))
            order += list(best)
        else:
            # greedy conditional rule: next = argmin cost / P(block | still alive)
            remaining = list(g)
            while remaining:
                nxt = min(remaining, key=lambda l: _cond_ratio(order, l, V, attack_prevalence))
                order.append(nxt)
                remaining.remove(nxt)
    return order, expected_cost(order, V, attack_prevalence)


def _cond_ratio(prefix: Sequence[str], l: str, V: VerdictMatrix, prev: float) -> float:
    i = V.layers.index(l)
    p_block = 0.0
    for rows, w in ((V.stop_or_catch, prev), (V.fp, 1 - prev)):
        if rows.shape[1] == 0:
            continue
        alive = np.ones(rows.shape[1], bool)
        for j in V.idx(prefix):
            alive &= ~rows[j]
        if alive.any():
            p_block += w * (rows[i] & alive).sum() / alive.sum()
    return V.cost_ms.get(l, 0.0) / max(p_block, 1e-12)


# ---------------------------------------------------------------------- interaction
def predicted_residual_asr(V: VerdictMatrix, subset: Sequence[str]) -> float:
    if not subset:
        return V.n_eff / max(1, V.n_attacks)
    idx = V.idx(subset)
    return float((~np.any(V.stop_or_catch[idx], axis=0)).sum() / max(1, V.n_attacks))


def interaction(V: VerdictMatrix, subset: Sequence[str], measured_asr: float) -> dict[str, float]:
    """Positive = the stack does worse than the union model predicts (sub-additive, e.g. an
    adaptive attacker or a redaction that changes later behaviour); negative = better
    (super-additive, e.g. a transform layer making later detectors more effective)."""
    p = predicted_residual_asr(V, subset)
    return {"predicted_asr": p, "measured_asr": measured_asr, "interaction": measured_asr - p}
