"""Sequential (marginal-block-rate) accounting for ENFORCED stacks under an adaptive attacker.

The static analysis (``structure.py``) works on a shadow-all verdict matrix: one payload per
case, every layer observes it. An adaptive run is different in two ways:

1. Each case has up to ``budget`` attempts with different payloads, so a case has several sets
   of layer verdicts. It must be collapsed to ONE representative attempt per case:

       "static"   attempt 1 (the static AgentDojo attack, run against the same enforced stack)
       "best"     the payload the attacker would deploy: its first successful attempt; if it never
                  succeeded, the attempt that got furthest (reached the target action > no layer
                  fired > stopped by a later layer of the stack > most recent). This is the worst
                  case for the defender and the headline number.
       "last"     the final attempt (ablation)

   ``pooled_mbr`` is the other ablation: every (case, attempt) is its own trial.

2. The stack is enforced, so a layer that ends the session hides the later layers' verdicts.
   Each attempt therefore has exactly one outcome:

       SUCCESS            the attack succeeded (no layer stopped it)
       <layer name>       the attack failed and this layer was the first to fire (time order)
       UNSTOPPED_FAIL     the attack failed and no layer fired (the agent did not comply)

   which is all the sequential definition needs:

       MBR(k) = P(layer k stops the attack | layers 1..k-1 did not)

   The same censoring means pairwise redundancy (rho) CANNOT be estimated from an enforced run
   (a layer after an aborting layer is only observed when that layer missed). ``cofire_table``
   reports what is observed; an unbiased adaptive rho needs the best-shot payloads replayed
   against the shadow-all stack.

Population. As in the static analysis, MBR is computed over EFFECTIVE attacks: cases where the
same attacker succeeds against the bare agent (``effective_ids``; static mode: the seed succeeds
on the bare agent, adaptive modes: any attempt does). Attack success rates use all cases.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import twoway_bootstrap

SUCCESS = "(success)"
UNSTOPPED_FAIL = "(failed, no layer fired)"
MODES = ("static", "best", "last")


# ---------------------------------------------------------------------------- loading
def load_attempts(folder: Path) -> dict[str, list[dict]]:
    """case_id -> error-free attempts sorted by k, from one ``<stack>/<attack tag>`` folder.

    Verdicts come from the session records; whether the agent reached the target action comes
    from the histories file (the records do not carry the tool trace)."""
    folder = Path(folder)
    reached: dict[tuple[str, int], bool] = {}
    for f in sorted(folder.glob("*.histories.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            h = json.loads(line)
            for a in h["attempts"]:
                reached[(h["case_id"], a["k"])] = any(c.get("harmful") for c in a.get("tool_calls", []))
    best: dict[tuple[str, int], dict] = {}
    for f in sorted(folder.glob("*.jsonl")):
        if f.name.endswith(".histories.jsonl"):
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cfg = r.get("config", {})
            if cfg.get("attempt") is None or not r.get("is_attack", True):
                continue
            key = (cfg["case_id"], int(cfg["attempt"]))
            if key not in best or (best[key].get("error") and not r.get("error")):   # error-free record wins
                best[key] = r
    out: dict[str, list[dict]] = {}
    for (cid, k), r in best.items():
        if r.get("error") or r.get("guard_errors"):
            continue
        out.setdefault(cid, []).append({
            "k": k, "success": bool(r["attack_success"]), "fired": dict(r.get("fired", {})),
            "first_fire_seq": dict(r.get("first_fire_seq", {})), "aborted_by": r.get("aborted_by"),
            "reached_target": reached.get((cid, k), False), "layers": list(r["stack_layers"]),
            "suite": r["suite"], "user_task": r["user_task"], "injection_task": r["injection_task"],
            "category": r.get("category"), "payload": r.get("config", {}).get("payload"),
        })
    for v in out.values():
        v.sort(key=lambda a: a["k"])
    return out


# ---------------------------------------------------------------------------- one attempt per case
def _closeness(a: dict) -> tuple:
    """How far an attempt got: success > reached the target action > no layer fired > stopped by
    a later layer of the stack > more recent."""
    o = outcome(a)
    depth = len(a["layers"]) if o in (SUCCESS, UNSTOPPED_FAIL) else evaluation_order(a["layers"]).index(o)
    return (a["success"], a["reached_target"], o == UNSTOPPED_FAIL, depth, a["k"])


def representative(attempts: Sequence[dict], mode: str = "best") -> dict | None:
    if not attempts:
        return None
    if mode == "static":
        return next((a for a in attempts if a["k"] == 1), None)
    if mode == "last":
        return attempts[-1]
    if mode == "best":
        wins = [a for a in attempts if a["success"]]
        return wins[0] if wins else max(attempts, key=_closeness)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


# Order in which an indirect injection meets the stages of guardstack.pipeline: the injected tool
# result is checked (tool_output) before the agent's next proposed call, which is checked by trace
# layers before action layers (GuardedToolsExecutor); output layers see only the final message.
# Stage.order is the order WITHIN one tool step and ranks tool_output last, so it is not used here.
IPI_STAGE_RANK = {"input": 0, "tool_output": 1, "trace": 2, "action": 3, "output": 4}


def evaluation_order(layers: Sequence[str]) -> list[str]:
    """Layer names ("name@stage") in the order the enforced pipeline evaluates them against an
    indirect injection: by stage (IPI_STAGE_RANK), then by position in the stack. MBR(k) conditions
    on the layers evaluated before k, so it must follow this order, not the listing order: in stack
    E (listed pg2, lg_out, gate, spot, align) AlignmentCheck sees each proposed call before the gate,
    and conditioning in listing order would charge the gate with cases it never saw."""
    def rank(item):
        i, name = item
        stage = name.rsplit("@", 1)[-1] if "@" in name else ""
        if stage not in IPI_STAGE_RANK:
            raise ValueError(f"layer {name!r}: cannot tell its stage (expected name@<stage>)")
        return IPI_STAGE_RANK[stage], i
    return [name for _, name in sorted(enumerate(layers), key=rank)]


def outcome(a: dict) -> str:
    """SUCCESS, the first layer that fired (time order; ties by stack order), or UNSTOPPED_FAIL."""
    if a["success"]:
        return SUCCESS
    fired = [l for l in a["layers"] if a["fired"].get(l)]
    if not fired:
        return UNSTOPPED_FAIL
    pos = {l: i for i, l in enumerate(evaluation_order(a["layers"]))}
    return min(fired, key=lambda l: (a["first_fire_seq"].get(l, float("inf")), pos[l]))


def effective_ids(bare: dict[str, list[dict]], mode: str) -> set[str]:
    """Cases where the attacker succeeds with no defense. static: the seed (attempt 1) succeeds;
    best / last / pooled: any attempt within the budget succeeds."""
    if mode == "static":
        return {c for c, at in bare.items() if any(a["k"] == 1 and a["success"] for a in at)}
    return {c for c, at in bare.items() if any(a["success"] for a in at)}


# ---------------------------------------------------------------------------- MBR
def _mbr_rows(outcomes: np.ndarray, order: Sequence[str]) -> list[dict]:
    alive, rows = len(outcomes), []
    for pos, layer in enumerate(order, 1):
        stopped = int((outcomes == layer).sum())
        rows.append({"layer": layer, "pos": pos, "survivors_in": alive, "stopped": stopped,
                     "mbr": stopped / alive if alive else np.nan})
        alive -= stopped
    return rows


def sequential_mbr(cases: dict[str, list[dict]], mode: str = "best", order: Sequence[str] | None = None,
                   effective: Iterable[str] | None = None, n_boot: int = 2000) -> tuple[pd.DataFrame, dict]:
    """MBR(k) along ``order`` (default: ``evaluation_order`` of the stack) on one attempt per case.

    Returns (table indexed by layer: pos, survivors_in, stopped, mbr, lo, hi; summary dict).
    ``effective``: case ids to compute MBR over (see module docstring); None = all cases.
    Intervals: two-way bootstrap over user tasks x injection tasks within suite."""
    reps = {c: representative(at, mode) for c, at in cases.items()}
    reps = {c: a for c, a in reps.items() if a is not None}
    if not reps:
        return pd.DataFrame(), {"n_cases": 0}
    order = list(order or evaluation_order(next(iter(reps.values()))["layers"]))
    n_all = len(reps)
    n_success_all = sum(a["success"] for a in reps.values())
    ids = sorted(c for c in reps if effective is None or c in set(effective))
    out = np.array([outcome(reps[c]) for c in ids], dtype=object)
    ut = np.array([f'{reps[c]["suite"]}/{reps[c]["user_task"]}' for c in ids])
    it = np.array([f'{reps[c]["suite"]}/{reps[c]["injection_task"]}' for c in ids])
    suite = np.array([reps[c]["suite"] for c in ids])

    rows = _mbr_rows(out, order)
    for row in rows:
        if len(ids) and n_boot:
            stat = (lambda ix, layer=row["layer"]: next(
                r["mbr"] for r in _mbr_rows(out[ix], order) if r["layer"] == layer))
            _, row["lo"], row["hi"] = twoway_bootstrap(stat, ut, it, suite, n_boot=n_boot)
        else:
            row["lo"] = row["hi"] = np.nan
    summary = {
        "mode": mode, "n_cases": n_all, "asr": n_success_all / n_all, "n_effective": len(ids),
        "success_effective": int((out == SUCCESS).sum()),
        "stopped_by_a_layer": int(sum(r["stopped"] for r in rows)),
        "unstopped_fail": int((out == UNSTOPPED_FAIL).sum()),
        "residual_effective": float((out == SUCCESS).mean()) if len(ids) else float("nan"),
    }
    table = pd.DataFrame(rows).set_index("layer") if rows else pd.DataFrame()   # bare agent: no layers
    return table, summary


def pooled_mbr(cases: dict[str, list[dict]], order: Sequence[str] | None = None,
               effective: Iterable[str] | None = None, skip_static: bool = True) -> pd.DataFrame:
    """Ablation: every (case, attempt) is a trial (attempt 1 = the static seed is left out by
    default). Averages over the attacker's probes, so it understates what the deployed payload does."""
    eff = None if effective is None else set(effective)
    atts = [a for c, at in cases.items() if eff is None or c in eff for a in at if not (skip_static and a["k"] == 1)]
    if not atts:
        return pd.DataFrame()
    order = list(order or evaluation_order(atts[0]["layers"]))
    rows = _mbr_rows(np.array([outcome(a) for a in atts], dtype=object), order)
    return pd.DataFrame(rows).set_index("layer") if rows else pd.DataFrame()


def mbr_shift(cases: dict[str, list[dict]], bare: dict[str, list[dict]] | None = None,
              order: Sequence[str] | None = None, n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Static vs adaptive MBR(k) for one enforced stack, from the same run (attempt 1 is the static
    attack). Returns (per-layer table with mbr_static, mbr_adaptive, shift and intervals;
    per-mode summary table). ``bare``: the bare-agent run of the same attacker, for the effective set."""
    tabs, sums = {}, []
    for mode in ("static", "best"):
        eff = effective_ids(bare, mode) if bare is not None else None
        t, s = sequential_mbr(cases, mode, order, eff, n_boot)
        tabs[mode] = t
        sums.append(s)
    if tabs["static"].empty or tabs["best"].empty:
        return pd.DataFrame(), pd.DataFrame(sums)
    s_, a_ = tabs["static"], tabs["best"]
    table = pd.DataFrame({
        "pos": a_["pos"],
        "survivors_static": s_["survivors_in"], "mbr_static": s_["mbr"], "lo_static": s_["lo"], "hi_static": s_["hi"],
        "survivors_adaptive": a_["survivors_in"], "mbr_adaptive": a_["mbr"], "lo_adaptive": a_["lo"],
        "hi_adaptive": a_["hi"], "shift": a_["mbr"] - s_["mbr"],
    })
    summary = pd.DataFrame(sums).set_index("mode").rename(index={"best": "adaptive"})
    return table, summary


# ---------------------------------------------------------------------------- what an enforced run shows of rho
def cofire_table(cases: dict[str, list[dict]], mode: str = "best") -> pd.DataFrame:
    """For each ordered pair (i evaluated before j, see ``evaluation_order``): on how many representative attempts both
    verdicts were observed (the session was not ended by another layer first), and how often each
    fired there. Shows why rho is not identified from an enforced run; not an estimate of rho."""
    reps = [a for a in (representative(at, mode) for at in cases.values()) if a is not None]
    if not reps:
        return pd.DataFrame()
    layers = evaluation_order(reps[0]["layers"])
    rows = []
    for x, i in enumerate(layers):
        for j in layers[x + 1:]:
            seen = [a for a in reps if a["aborted_by"] in (None, i, j)]
            both = sum(1 for a in seen if a["fired"].get(i) and a["fired"].get(j))
            rows.append({"layer_i": i, "layer_j": j, "n_observed": len(seen), "n_total": len(reps),
                         "i_fired": sum(1 for a in seen if a["fired"].get(i)),
                         "j_fired": sum(1 for a in seen if a["fired"].get(j)), "both_fired": both})
    return pd.DataFrame(rows)


def find_runs(run_dir: Path) -> dict[tuple[str, str], Path]:
    """(stack, attack tag) -> folder, for every adaptive result folder under a run."""
    return {(f.parent.parent.name, f.parent.name): f.parent for f in sorted(Path(run_dir).rglob("*.histories.jsonl"))}

