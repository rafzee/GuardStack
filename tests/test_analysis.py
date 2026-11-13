"""Known-answer tests for the analysis code."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardstack.analysis import assignment, selection, structure  # noqa: E402
from guardstack.analysis.synthetic import make_synthetic  # noqa: E402


def small():
    # 3 layers, 6 effective attacks, 4 benign; A and B overlap, C is unique
    catch = np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
    ], bool)
    fp = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], bool)
    return structure.VerdictMatrix(["A@input", "B@input", "C@output"], catch, fp,
                                   np.array(list("xxxyyy")), np.array(list("ababab")), n_attacks=10,
                                   stage={"A@input": "input", "B@input": "input", "C@output": "output"},
                                   cost_ms={"A@input": 10, "B@input": 1, "C@output": 5})


def test_mbr_and_unique():
    V = small()
    c = structure.mbr_curve(V, ["A@input", "B@input", "C@output"])
    assert np.isclose(c.loc["A@input", "mbr"], 3 / 6)
    assert np.isclose(c.loc["B@input", "mbr"], 0.0)          # B adds nothing after A
    assert np.isclose(c.loc["C@output", "mbr"], 2 / 3)
    assert np.isclose(c.loc["C@output", "residual_asr"], 1 / 10)
    u = structure.unique_catch(V)
    assert np.allclose(u, [1 / 6, 0, 2 / 6])


def test_shapley_matches_bruteforce():
    V = make_synthetic(n_per_cat=15, n_benign=20)
    V = structure.VerdictMatrix(V.layers[:6], V.catch[:6], V.fp[:6], V.category, V.cluster, V.n_attacks)
    n = V.L

    def v(S):
        return np.any(V.catch[list(S)], axis=0).mean() if S else 0.0

    from math import factorial
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for S in itertools.combinations(others, r):
                w = factorial(r) * factorial(n - r - 1) / factorial(n)
                phi[i] += w * (v(S + (i,)) - v(S))
    assert np.allclose(phi, structure.shapley_coverage(V), atol=1e-9)
    assert np.isclose(phi.sum(), v(tuple(range(n))))


def test_subset_enumeration_matches_naive():
    V = small()
    df = selection.enumerate_subsets(V).set_index("mask")
    for m in range(8):
        idx = [i for i in range(3) if m >> i & 1]
        caught = np.any(V.catch[idx], axis=0).sum() if idx else 0
        assert np.isclose(df.loc[m, "residual_asr"], (6 - caught) / 10)
        fpr = np.any(V.fp[idx], axis=0).mean() if idx else 0
        assert np.isclose(df.loc[m, "fpr"], fpr)


def test_ordering_puts_cheap_high_block_first():
    V = small()
    V.fp = np.zeros_like(V.fp)  # with FPs, an expensive layer that aborts benign sessions can "save" cost
    order, _ = selection.optimal_order(["A@input", "B@input"], V)
    # B is 10x cheaper and catches a subset of A: run B first, A only on survivors
    assert order == ["B@input", "A@input"]


def test_hungarian_one_to_one_and_capacity():
    V = small()
    own = assignment.ownership(V, ["x", "y"])
    assert own.loc["x", "owner"] == "A@input" and own.loc["y", "owner"] == "C@output"
    own2 = assignment.ownership(V, ["x", "y"], capacity=2)
    assert set(own2.owner) <= set(V.layers)


def test_stratified_rho_removes_scope_effect():
    # a and d both cover only category x and are independent inside it; both miss all of y.
    # Pooled rho is positive from shared scope alone; within-category rho is ~0.
    rng = np.random.default_rng(0)
    n = 400
    cat = np.array(["x"] * n + ["y"] * n)
    a = np.r_[rng.random(n) < 0.8, np.zeros(n, bool)]
    d = np.r_[rng.random(n) < 0.8, np.zeros(n, bool)]
    V = structure.VerdictMatrix(["a", "d"], np.vstack([a, d]), np.zeros((2, 0), bool), cat,
                                np.zeros(2 * n).astype(str), 2 * n)
    pooled = structure.redundancy_rho(V).loc["a", "d"]
    strat = structure.stratified_rho(V).loc["a", "d"]
    assert pooled > 0.3
    assert abs(strat) < 0.1


def test_threshold_rederivation():
    recs = []
    for i in range(10):
        recs.append({"session_id": f"a{i}", "config": {"case_id": f"a{i}"}, "is_attack": True,
                     "attack_success": True, "error": None, "fired": {"L": True}, "injection_task": "i0",
                     "max_score": {"L": i / 10}, "category": "c", "suite": "s", "user_task": f"u{i}",
                     "stack_layers": ["L"], "guard_latency_ms": {"L": 1.0}})
        recs.append({"session_id": f"b{i}", "config": {"case_id": f"b{i}"}, "is_attack": False,
                     "attack_success": None, "error": None, "fired": {"L": False}, "injection_task": None,
                     "max_score": {"L": i / 20}, "category": "benign", "suite": "s", "user_task": f"u{i}",
                     "stack_layers": ["L"], "guard_latency_ms": {"L": 1.0}})
    V = structure.VerdictMatrix.from_records(recs, thresholds={"L": 0.5})
    assert V.catch.sum() == 5 and V.fp.sum() == 0
    t = structure.threshold_at_fpr(recs, "L", 0.0)
    assert t > 0.45
