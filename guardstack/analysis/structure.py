"""Structural quantities of a guardrail pool: who catches what, how much each layer adds,
and how much the layers overlap.

Input is a :class:`VerdictMatrix` -- for every layer and every case, did the layer fire?
The cheapest way to get it is one *shadow-all* run (every layer observes the bare agent,
none acts). Isolated enforced runs validate it; enforced stacks measure what the shadow
matrix cannot (trajectory changes, adaptive attackers).

Definitions (E = effective attacks: attack cases that succeed against the bare agent):

    catch_i(x)   layer i fires on x                         x in E
    miss_i(x)    1 - catch_i(x)
    recall_i     mean_x catch_i(x)
    MBR(k | pi)  P(catch_{pi_k} | miss_{pi_1..pi_{k-1}})    marginal block rate along order pi
    rho(i,j)     Pearson (= phi) correlation of miss_i, miss_j over E   (one-pager definition)
    lift(i,j)    P(miss_i and miss_j) / (P(miss_i) P(miss_j))  -- 1 = independent holes,
                 >1 = holes line up (Swiss-cheese failure). Easier to read than phi,
                 whose range is capped by the marginals.
    unique_i     |catch_i minus union of all other layers' catches| / |E|  (leave-one-out value)
    shapley_i    Shapley value of i in the coverage game v(S) = |union_{j in S} catch_j| / |E|.
                 Closed form: sum over x caught by i of 1 / #layers catching x, divided by |E|.
    indep_gap(S) P(all of S miss) / prod_i P(miss_i) -- how far "multiply the miss rates"
                 defense-in-depth arithmetic is off for stack S.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class VerdictMatrix:
    layers: list[str]
    catch: np.ndarray                 # (L, n_eff) bool, effective attacks only
    fp: np.ndarray                    # (L, n_benign) bool
    category: np.ndarray              # (n_eff,) category label per effective attack
    cluster: np.ndarray               # (n_eff,) cluster id (user task) for bootstrap
    n_attacks: int                    # all attack cases, effective or not (ASR denominator)
    benign_cluster: np.ndarray | None = None
    stage: dict[str, str] = field(default_factory=dict)
    family: dict[str, str] = field(default_factory=dict)
    cost_ms: dict[str, float] = field(default_factory=dict)   # mean latency per session
    stop: np.ndarray | None = None    # (L, n_eff) bool: layer fired (on anything) before the harm -> would have stopped it
    inj_cluster: np.ndarray | None = None   # (n_eff,) suite/injection_task, for the two-way bootstrap
    suite: np.ndarray | None = None
    notes: dict = field(default_factory=dict)

    @property
    def stop_or_catch(self) -> np.ndarray:
        return self.stop if self.stop is not None else self.catch

    @property
    def L(self) -> int:
        return len(self.layers)

    @property
    def n_eff(self) -> int:
        return self.catch.shape[1]

    def idx(self, names: Iterable[str]) -> list[int]:
        m = {n: i for i, n in enumerate(self.layers)}
        return [m[n] for n in names]

    def subset_cases(self, cols: np.ndarray) -> "VerdictMatrix":
        sub = (lambda a: a[cols] if a is not None else None)
        return VerdictMatrix(self.layers, self.catch[:, cols], self.fp, self.category[cols], self.cluster[cols],
                             self.n_attacks, self.benign_cluster, self.stage, self.family, self.cost_ms,
                             self.stop[:, cols] if self.stop is not None else None, sub(self.inj_cluster),
                             sub(self.suite), self.notes)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_records(cls, records: Sequence[dict], layers: Sequence[str] | None = None,
                     layer_meta: dict[str, dict] | None = None,
                     thresholds: dict[str, float] | None = None,
                     cost_records: Sequence[dict] | None = None) -> "VerdictMatrix":
        """Build from SessionRecord dicts of a shadow-all run on the bare agent.

        A layer *catches* an effective attack only if it fired on attacker material
        (payload text, attacker entity, injected call, or a contaminated trajectory) BEFORE the
        harm happened (the first executed attacker-signature tool call; for objectives with no
        such call, e.g. a manipulated answer, the end of the session). That is the detection
        matrix used for recall, activity, rho, Shapley and unique catch. A layer *stops* the
        attack if it fired on anything before the harm (an input block on AgentDojo's benign
        user prompt also ends the session); that matrix drives MBR and the union-model ASR.

        ``thresholds`` re-derives a layer's verdict offline as max_score >= tau (shadow runs
        only; timing then falls back to "any time", so use it for operating curves, not MBR).
        ``cost_records``: latency-mode records to take per-layer costs from (throughput-run
        latencies are contended and must not be reported)."""
        seen: dict[str, dict] = {}
        for r in records:
            if "session_id" not in r:
                continue
            seen[r.get("config", {}).get("case_id", r["session_id"])] = r   # dedupe: last record wins
        records = list(seen.values())
        layers = list(layers or records[0]["stack_layers"])
        guard_err = [r for r in records if r.get("guard_errors")]
        ok = [r for r in records if r.get("error") is None and not r.get("guard_errors")]
        att = [r for r in ok if r["is_attack"]]
        eff = [r for r in att if r["attack_success"]]
        ben = [r for r in ok if not r["is_attack"]]
        th = thresholds or {}
        INF = float("inf")

        def harm(r):
            h = r.get("harm_seq")
            return INF if h is None else h

        def fired_any(r, l):
            if l in th:
                return r.get("max_score", {}).get(l, float("-inf")) >= th[l]
            return bool(r["fired"].get(l, False))

        def stop(r, l):
            if l in th or "first_fire_seq" not in r:
                return fired_any(r, l)
            s_ = r["first_fire_seq"].get(l)
            return s_ is not None and s_ < harm(r)

        def catch(r, l):
            if l in th or "attributed_fire_seq" not in r:
                return stop(r, l)
            s_ = r["attributed_fire_seq"].get(l)
            return s_ is not None and s_ < harm(r)

        mk = (lambda f, rows: np.array([[f(r, l) for r in rows] for l in layers], dtype=bool).reshape(
            len(layers), len(rows)))
        src = cost_records if cost_records is not None else records
        cost = {l: float(np.mean([r["guard_latency_ms"].get(l, 0.0) for r in src])) for l in layers}
        meta = layer_meta or {}
        notes = {"n_records": len(records), "excluded_guard_errors": len(guard_err),
                 "excluded_errors": sum(1 for r in records if r.get("error") and not r.get("guard_errors")),
                 "skipped_scored_as_success": sum(1 for r in att if r.get("skipped_reason")),
                 "cost_source": "latency-mode records" if cost_records is not None else
                 "THROUGHPUT RUN (contended; not for reporting)"}
        return cls(
            layers=layers, catch=mk(catch, eff), fp=mk(fired_any, ben),
            category=np.array([r["category"] for r in eff]),
            cluster=np.array([f'{r["suite"]}/{r["user_task"]}' for r in eff]),
            benign_cluster=np.array([f'{r["suite"]}/{r["user_task"]}' for r in ben]),
            n_attacks=len(att),
            stage={l: meta.get(l, {}).get("stage", l.split("@")[-1]) for l in layers},
            family={l: meta.get(l, {}).get("family", "other") for l in layers},
            cost_ms=cost, stop=mk(stop, eff),
            inj_cluster=np.array([f'{r["suite"]}/{r["injection_task"]}' for r in eff]),
            suite=np.array([r["suite"] for r in eff]), notes=notes,
        )


# ---------------------------------------------------------------------- per-layer tables
def layer_table(V: VerdictMatrix) -> pd.DataFrame:
    rec = V.catch.mean(axis=1) if V.n_eff else np.zeros(V.L)
    fpr = V.fp.mean(axis=1) if V.fp.shape[1] else np.full(V.L, np.nan)
    uniq = unique_catch(V)
    shap = shapley_coverage(V)
    return pd.DataFrame({
        "layer": V.layers,
        "stage": [V.stage.get(l, "") for l in V.layers],
        "family": [V.family.get(l, "") for l in V.layers],
        "recall": rec,
        "fpr": fpr,
        "unique": uniq,
        "shapley": shap,
        "redundant_share": np.where(rec > 0, 1 - uniq / np.maximum(rec, 1e-12), np.nan),
        "cost_ms": [V.cost_ms.get(l, np.nan) for l in V.layers],
    }).set_index("layer")


def activity_matrix(V: VerdictMatrix, categories: Sequence[str] | None = None) -> pd.DataFrame:
    """Recall of every layer on every attack category: the "which layer is active where" view."""
    cats = list(categories or sorted(set(V.category)))
    data = {c: (V.catch[:, V.category == c].mean(axis=1) if (V.category == c).any() else np.full(V.L, np.nan))
            for c in cats}
    return pd.DataFrame(data, index=V.layers)


def first_actor_share(records: Sequence[dict], categories: Sequence[str] | None = None,
                      effective_case_ids: set[str] | None = None) -> pd.DataFrame:
    """From an ENFORCED stack: for each category, the share of stopped attacks credited to each
    layer, crediting the first enforcing layer that fired (time order). This is the
    'which layer is active in the deployed stack' view, as opposed to the shadow view."""
    rows = []
    for r in records:
        if not r["is_attack"] or r.get("error"):
            continue
        if effective_case_ids is not None and r.get("config", {}).get("case_id") not in effective_case_ids:
            continue   # only attacks that succeed on the bare agent can be credited
        enforcing = [l for l, m in r["layer_modes"].items() if m == "enforce"]
        fires = {l: s for l, s in r["first_fire_seq"].items() if l in enforcing}
        first = min(fires, key=fires.get) if fires else None
        rows.append({"category": r["category"], "first": first or "(none)", "success": bool(r["attack_success"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    stopped = df[~df.success & (df["first"] != "(none)")]
    tab = pd.crosstab(stopped["first"], stopped["category"], normalize="columns")
    if categories is not None:
        tab = tab.reindex(columns=list(categories), fill_value=0.0)
    return tab


# ---------------------------------------------------------------------- MBR
def mbr_curve(V: VerdictMatrix, order: Sequence[str], use: str = "stop") -> pd.DataFrame:
    """use="stop": deployment MBR (any fire before the harm counts, incl. false-positive blocks);
    use="catch": detection MBR (only fires on attacker material count)."""
    M = V.stop_or_catch if use == "stop" else V.catch
    idx = V.idx(order)
    alive = np.ones(V.n_eff, dtype=bool)
    rows = []
    for name, i in zip(order, idx):
        n_alive = int(alive.sum())
        caught = M[i] & alive
        mbr = caught.sum() / n_alive if n_alive else np.nan
        alive &= ~M[i]
        rows.append({"layer": name, "survivors_in": n_alive, "mbr": mbr,
                     "residual_asr": alive.sum() / max(1, V.n_attacks)})
    return pd.DataFrame(rows).set_index("layer")


# ---------------------------------------------------------------------- redundancy
def _pairwise(V: VerdictMatrix):
    miss = (~V.catch).astype(float)
    n = V.n_eff
    pm = miss.mean(axis=1)
    joint = miss @ miss.T / max(1, n)
    return miss, pm, joint


def redundancy_rho(V: VerdictMatrix) -> pd.DataFrame:
    miss, _, _ = _pairwise(V)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(miss) if V.n_eff > 1 else np.full((V.L, V.L), np.nan)
    r = np.where(np.isfinite(r), r, np.nan)
    return pd.DataFrame(r, index=V.layers, columns=V.layers)


def miss_lift(V: VerdictMatrix) -> pd.DataFrame:
    _, pm, joint = _pairwise(V)
    with np.errstate(invalid="ignore", divide="ignore"):
        lift = joint / np.outer(pm, pm)
    return pd.DataFrame(lift, index=V.layers, columns=V.layers)


def catch_jaccard(V: VerdictMatrix) -> pd.DataFrame:
    c = V.catch.astype(float)
    inter = c @ c.T
    sz = c.sum(axis=1)
    union = sz[:, None] + sz[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.where(union > 0, inter / union, np.nan)
    return pd.DataFrame(j, index=V.layers, columns=V.layers)


def conditional_miss(V: VerdictMatrix) -> pd.DataFrame:
    """Entry (i, j) = P(miss_j | miss_i): of what layer i lets through, how much does j also miss."""
    _, pm, joint = _pairwise(V)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = joint / pm[:, None]
    return pd.DataFrame(cm, index=V.layers, columns=V.layers)


def unique_catch(V: VerdictMatrix) -> np.ndarray:
    tot = V.catch.sum(axis=0)
    return (V.catch & (tot == 1)[None, :]).sum(axis=1) / max(1, V.n_eff)


def shapley_coverage(V: VerdictMatrix) -> np.ndarray:
    k = V.catch.sum(axis=0).astype(float)
    w = np.where(k > 0, 1.0 / np.maximum(k, 1), 0.0)
    return (V.catch * w[None, :]).sum(axis=1) / max(1, V.n_eff)


def independence_gap(V: VerdictMatrix, subset: Sequence[str]) -> dict[str, float]:
    idx = V.idx(subset)
    miss = ~V.catch[idx]
    joint = float(np.all(miss, axis=0).mean()) if V.n_eff else np.nan
    prod = float(np.prod(miss.mean(axis=1)))
    return {"joint_miss": joint, "product_of_misses": prod,
            "gap_ratio": joint / prod if prod > 0 else np.inf if joint > 0 else 1.0}


def dominated_layers(V: VerdictMatrix, tol: float = 0.0) -> list[tuple[str, list[str]]]:
    """Layers whose catches are (almost) a subset of a single other layer's catches with no
    lower FPR -- candidates to drop outright."""
    out = []
    for i, a in enumerate(V.layers):
        ci = V.catch[i]
        if not ci.any():
            out.append((a, ["<catches nothing>"]))
            continue
        by = []
        for j, b in enumerate(V.layers):
            if i == j:
                continue
            missed = (ci & ~V.catch[j]).sum() / ci.sum()
            fpr_ok = (V.fp[j].mean() <= V.fp[i].mean()) if V.fp.shape[1] else True
            if missed <= tol and fpr_ok:
                by.append(b)
        if by:
            out.append((a, by))
    return out


def stratified_rho(V: VerdictMatrix, min_n: int = 20, strata: np.ndarray | None = None) -> pd.DataFrame:
    """Scope-adjusted redundancy: within-stratum miss correlation, averaged over strata
    (weighted by stratum size; undefined strata skipped). Strata default to attack category;
    pass e.g. difficulty bins x category to answer the "correlation is just input difficulty"
    objection (Eckhardt-Lee common-cause; Alotaibi et al. 2026 use a CMH-style version).

    Pooled rho conflates "both layers ignore this category" (a scope fact, visible in the
    activity matrix) with "both layers miss the same attacks inside a category they both
    try to cover" (a shared blind spot). Only the second is the redundancy hypothesis (3)
    is about, so this is the version to report."""
    L = V.L
    num = np.zeros((L, L))
    den = np.zeros((L, L))
    strata = V.category if strata is None else np.asarray(strata)
    for c in np.unique(strata):
        cols = strata == c
        n = int(cols.sum())
        if n < min_n:
            continue
        miss = (~V.catch[:, cols]).astype(float)
        sd = miss.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(miss)
        ok = (sd[:, None] > 0) & (sd[None, :] > 0) & np.isfinite(r)
        num += np.where(ok, r * n, 0.0)
        den += np.where(ok, n, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(out, index=V.layers, columns=V.layers)


def operating_points(records: Sequence[dict], layer: str, taus: Sequence[float] | None = None) -> pd.DataFrame:
    """Recall on effective attacks vs FPR on benign cases as the layer's threshold moves
    (shadow runs only). Use it to pick each layer's threshold at a common FPR budget."""
    att = [r for r in records if r["is_attack"] and r.get("error") is None and r["attack_success"]]
    ben = [r for r in records if not r["is_attack"] and r.get("error") is None]
    sa = np.array([r.get("max_score", {}).get(layer, -np.inf) for r in att])
    sb = np.array([r.get("max_score", {}).get(layer, -np.inf) for r in ben])
    if taus is None:
        finite = np.concatenate([sa[np.isfinite(sa)], sb[np.isfinite(sb)]])
        taus = np.unique(finite)
    rows = [{"tau": t, "recall": float((sa >= t).mean()) if len(sa) else np.nan,
             "fpr": float((sb >= t).mean()) if len(sb) else np.nan} for t in taus]
    return pd.DataFrame(rows)


def threshold_at_fpr(records: Sequence[dict], layer: str, fpr_budget: float) -> float:
    """Smallest threshold whose benign FPR is <= the budget."""
    op = operating_points(records, layer)
    ok = op[op.fpr <= fpr_budget]
    return float(ok.tau.min()) if len(ok) else float("inf")


def difficulty_strata(V: VerdictMatrix, holdout: Sequence[str] | None = None, bins: int = 3) -> np.ndarray:
    """Difficulty proxy per effective attack: how many layers of a held-out reference set catch it
    (0 = hardest), crossed with category. Use a holdout disjoint from the pair being tested,
    or the proxy leaks the outcome into the stratum."""
    ref = V.idx(holdout) if holdout else list(range(V.L))
    k = V.catch[ref].sum(axis=0)
    edges = np.unique(np.quantile(k, np.linspace(0, 1, bins + 1)))
    d = np.digitize(k, edges[1:-1], right=True)
    return np.array([f"{c}|d{x}" for c, x in zip(V.category, d)])
