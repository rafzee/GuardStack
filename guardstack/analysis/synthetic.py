"""SYNTHETIC verdict data for testing the analysis code and previewing figures.

Nothing produced from this module is a result. It mirrors the one-pager's design-intent
coverage table (designed-for = high base rate, partial = medium, blank = near zero) and adds
two sources of correlated failure so redundancy is non-trivial:

    P(catch_i(x)) = sigmoid(alpha_i[c(x)] - beta * z_x - gamma * u_{family(i), x})

z_x is a stealth score shared by every layer (stealthy attacks are hard for everyone) and
u_{f,x} is shared within a layer family (same-type classifiers share blind spots), which
is the mechanism hypothesis (3) predicts.
"""

from __future__ import annotations

import numpy as np

from .structure import VerdictMatrix

CATEGORIES = [
    "direct_jailbreak", "multiturn_escalation", "indirect_injection", "goal_hijack",
    "unsafe_tool_call", "harmful_output", "insecure_code", "pii_leak",
]

H, P = 2.2, 0.2   # designed-for, partial  (logit); anything else -> -4.5

# name: (stage, family, mean ms per session, base FPR, {category: level})
POOL = {
    "promptguard2@input":       ("input", "pi_classifier", 12, 0.04, {"direct_jailbreak": H, "multiturn_escalation": P}),
    "promptguard2@tool_output": ("tool_output", "pi_classifier", 25, 0.05, {"indirect_injection": H, "goal_hijack": P}),
    "alignment_check@trace":    ("trace", "llm_judge", 950, 0.06, {"indirect_injection": P, "goal_hijack": H, "unsafe_tool_call": H, "pii_leak": P}),
    "codeshield@output":        ("output", "static_analysis", 60, 0.01, {"insecure_code": H}),
    "regex@tool_output":        ("tool_output", "rule", 1, 0.02, {"direct_jailbreak": P, "indirect_injection": P, "goal_hijack": P, "pii_leak": P}),
    "nemo_input@input":         ("input", "llm_judge", 420, 0.07, {"direct_jailbreak": H, "multiturn_escalation": P, "harmful_output": P}),
    "nemo_dialog@input":        ("input", "llm_judge", 380, 0.05, {"multiturn_escalation": P, "goal_hijack": P, "direct_jailbreak": P}),
    "nemo_retrieval@tool_output": ("tool_output", "llm_judge", 410, 0.06, {"indirect_injection": H}),
    "nemo_execution@action":    ("action", "policy", 40, 0.03, {"unsafe_tool_call": H}),
    "nemo_output@output":       ("output", "llm_judge", 400, 0.05, {"harmful_output": H, "pii_leak": P}),
    "presidio_pii@output":      ("output", "pii", 18, 0.08, {"pii_leak": H}),
    "llamaguard4@output":       ("output", "safety_classifier", 240, 0.03, {"harmful_output": H, "direct_jailbreak": P, "multiturn_escalation": P}),
    "pi_detector@tool_output":  ("tool_output", "pi_classifier", 30, 0.05, {"indirect_injection": H}),
    "tool_gate@action":         ("action", "policy", 4, 0.04, {"unsafe_tool_call": H, "goal_hijack": P, "indirect_injection": P, "pii_leak": P}),
}


def make_synthetic(n_per_cat: int = 120, n_benign: int = 600, beta: float = 0.9, gamma: float = 1.6,
                   eff_rate: float = 0.6, seed: int = 7) -> VerdictMatrix:
    rng = np.random.default_rng(seed)
    layers = list(POOL)
    fams = sorted({v[1] for v in POOL.values()})
    n_att = n_per_cat * len(CATEGORIES)
    n_eff = int(round(n_att * eff_rate))
    cats = np.repeat(CATEGORIES, n_per_cat)
    eff = rng.choice(n_att, size=n_eff, replace=False)
    eff.sort()
    cat_eff = cats[eff]
    z = rng.normal(size=n_eff)
    u = {f: rng.normal(size=n_eff) for f in fams}
    catch = np.zeros((len(layers), n_eff), bool)
    fp = np.zeros((len(layers), n_benign), bool)
    ub = {f: rng.normal(size=n_benign) for f in fams}
    for i, l in enumerate(layers):
        stage, fam, cost, base_fpr, lv = POOL[l]
        alpha = np.array([lv.get(c, -4.5) for c in cat_eff])
        p = 1 / (1 + np.exp(-(alpha - beta * z - gamma * u[fam])))
        catch[i] = rng.random(n_eff) < p
        # benign FPs also correlate within family (e.g. both PI classifiers trip on imperative text)
        logit0 = np.log(base_fpr / (1 - base_fpr))
        pb = 1 / (1 + np.exp(-(logit0 + 0.8 * ub[fam])))
        fp[i] = rng.random(n_benign) < pb
    n_users = 97
    cluster = rng.integers(0, n_users, size=n_eff).astype(str)
    return VerdictMatrix(
        layers=layers, catch=catch, fp=fp, category=cat_eff, cluster=cluster, n_attacks=n_att,
        benign_cluster=rng.integers(0, n_users, size=n_benign).astype(str),
        stage={l: POOL[l][0] for l in layers}, family={l: POOL[l][1] for l in layers},
        cost_ms={l: float(POOL[l][2]) for l in layers},
    )
