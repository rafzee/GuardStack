"""Where the Hungarian algorithm fits, and where it does not.

The Hungarian method (Kuhn 1955; scipy.optimize.linear_sum_assignment) solves the linear
assignment problem: given a matrix M[a, b], choose a one-to-one matching between rows and
columns that maximises the SUM of the matched entries. Two properties matter here:

* one-to-one: each row gets at most one column (or `capacity` columns if rows are
  duplicated), and each column exactly one row;
* separable: the objective is a sum of independent cell values. It cannot represent
  "layer A plus layer B together catch more than either", or "B adds nothing once A is
  present" -- interactions between layers are invisible to it.

Because redundancy and marginal value ARE interactions, Hungarian cannot pick the best
stack (use selection.enumerate_subsets for that). It is still the right tool for two
one-to-one questions the paper asks:

1. ownership(): which layer is the primary owner of each threat category (one layer per
   category, maximising total recall) -- a compact, readable summary of the activity matrix.
2. counterpart_matching(): pair each layer of framework A with its closest counterpart in
   framework B by overlap of what they catch -- matched pairs with high overlap are
   redundant across frameworks (e.g. a NeMo rail duplicating a LlamaFirewall scanner).

hungarian_vs_optimal() quantifies what you lose if you build a stack from the ownership
assignment instead of from the exact subset search.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .selection import enumerate_subsets
from .structure import VerdictMatrix, activity_matrix, catch_jaccard, redundancy_rho


def ownership(V: VerdictMatrix, categories: Sequence[str] | None = None, capacity: int = 1,
              cost_weight: float = 0.0) -> pd.DataFrame:
    """Assign one owner layer per category maximising sum of recall - cost_weight * cost_ms.
    capacity > 1 lets one layer own up to that many categories (rows duplicated)."""
    A = activity_matrix(V, categories).fillna(0.0)
    cats = list(A.columns)
    cost = np.array([V.cost_ms.get(l, 0.0) for l in A.index])
    M = A.to_numpy() - cost_weight * cost[:, None]
    M_rep = np.repeat(M, capacity, axis=0)
    rows, cols = linear_sum_assignment(M_rep, maximize=True)
    out = []
    for r, c in zip(rows, cols):
        layer = A.index[r // capacity]
        best_layer = A.index[int(np.argmax(A.iloc[:, c].to_numpy()))]
        out.append({"category": cats[c], "owner": layer, "owner_recall": A.iloc[r // capacity, c],
                    "best_single_layer": best_layer, "best_recall": A.iloc[:, c].max()})
    return pd.DataFrame(out).set_index("category").loc[cats]


def hungarian_vs_optimal(V: VerdictMatrix, categories: Sequence[str] | None = None, capacity: int = 1,
                         subsets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the stack = set of owners; compare against the exact best subset with the same
    number of layers, and the exact cheapest subset that is at least as secure."""
    own = ownership(V, categories, capacity)
    chosen = tuple(sorted(set(own.owner), key=V.layers.index))
    df = subsets if subsets is not None else enumerate_subsets(V)
    mask = sum(1 << V.layers.index(l) for l in chosen)
    h = df.loc[df["mask"] == mask].iloc[0]
    same_k = df[df.k == len(chosen)].sort_values(["residual_asr", "fpr", "cost_ms"]).iloc[0]
    as_good = df[(df.residual_asr <= h.residual_asr + 1e-12) & (df.fpr.fillna(0) <= (h.fpr if h.fpr == h.fpr else 0) + 1e-12)]
    cheapest = as_good.sort_values(["cost_ms", "k"]).iloc[0]
    rows = []
    for name, r in (("hungarian_owners", h), ("best_same_size", same_k), ("cheapest_at_least_as_good", cheapest)):
        rows.append({"stack": name, "k": int(r.k), "residual_asr": r.residual_asr, "fpr": r.fpr,
                     "cost_ms": r.cost_ms, "layers": ", ".join(r.layers)})
    return pd.DataFrame(rows).set_index("stack")


def counterpart_matching(V: VerdictMatrix, group_a: Sequence[str], group_b: Sequence[str]) -> pd.DataFrame:
    """Max-overlap one-to-one matching of group_a layers to group_b layers (Jaccard of catch
    sets). Returns pairs with their Jaccard and redundancy rho."""
    J = catch_jaccard(V).loc[list(group_a), list(group_b)].fillna(0.0)
    R = redundancy_rho(V)
    rows, cols = linear_sum_assignment(J.to_numpy(), maximize=True)
    out = []
    for r, c in zip(rows, cols):
        a, b = J.index[r], J.columns[c]
        out.append({"layer_a": a, "layer_b": b, "jaccard": J.iloc[r, c], "rho": R.loc[a, b]})
    return pd.DataFrame(out).sort_values("jaccard", ascending=False)
