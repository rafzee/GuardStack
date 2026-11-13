#!/usr/bin/env python
"""Guard-only sweep: score every layer directly on labelled datasets (no agent in the loop).

    python scripts/guard_sweep.py --data-root ~/work/guardstack_data --stack shadow_all \
        --out results/<RUN_ID>/guard_sweep [--max-per-dataset 500]

For each (layer, dataset) pair whose role matches the layer's stage it records block, score and
latency per item, then reports recall (label 1), FPR (label 0), ROC AUC from scores, and recall at
FPR 1% / 5%. Thresholds for the recall@FPR columns are fit on a held-out half of the negatives and
evaluated on the other half, so they aren't optimistically tuned on the test items.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardstack.layers.base import CheckContext  # noqa: E402
from guardstack.stacks import load_stacks  # noqa: E402
from guardstack.types import Stage  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STAGE_ROLE = {Stage.INPUT: "prompt", Stage.TOOL_OUTPUT: "tooltext", Stage.OUTPUT: "response"}


class DataError(RuntimeError):
    pass


def _read(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    if suf == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    obj = v
                    break
        if obj and isinstance(obj[0], str):
            return pd.DataFrame({"text": obj})
        return pd.DataFrame(obj)
    raise DataError(f"unsupported file type {path}")


def _col(df: pd.DataFrame, cands, what: str, name: str, required=True):
    for c in cands or []:
        if c in df.columns:
            return c
    if required:
        raise DataError(f"[{name}] no {what} column among {cands}; columns are {list(df.columns)}")
    return None


def load_dataset(name: str, spec: dict, root: Path) -> pd.DataFrame:
    if "builtin" in spec:
        return BUILTINS[spec["builtin"]]()
    files = []
    for pat in spec["files"]:
        files += sorted(glob.glob(str(root / pat), recursive=True))
    if not files:
        raise DataError(f"[{name}] no files match {spec['files']} under {root} "
                        f"(present: {sorted(p.name for p in (root / spec['files'][0].split('/')[0]).glob('*'))[:20] if (root / spec['files'][0].split('/')[0]).exists() else 'dataset dir missing'})")
    df = pd.concat([_read(Path(f)) for f in files], ignore_index=True)
    for col, keep in (spec.get("filter") or {}).items():
        if col not in df.columns:
            raise DataError(f"[{name}] filter column {col!r} missing; columns are {list(df.columns)}")
        df = df[df[col].isin(keep)]
    tcol = _col(df, spec["text"], "text", name)
    out = pd.DataFrame({"text": df[tcol].astype(str)})
    ccol = _col(df, spec.get("context"), "context", name, required=False)
    out["context"] = df[ccol].fillna("").astype(str) if ccol else ""
    if "label" in spec:
        out["label"] = int(spec["label"])
    else:
        lf = spec.get("label_from")
        lcol = _col(df, lf["column"], "label", name, required="label_from_type_prefix" not in spec) if lf else None
        if lcol:
            mapped = df[lcol].astype(str).str.lower().map({k.lower(): v for k, v in lf["map"].items()})
            if mapped.isna().any():
                bad = sorted(df[lcol][mapped.isna()].astype(str).unique())[:10]
                raise DataError(f"[{name}] unmapped label values in {lcol!r}: {bad}")
            out["label"] = mapped.astype(int).values
        else:
            tp = spec["label_from_type_prefix"]
            tc = _col(df, tp["column"], "type", name)
            out["label"] = np.where(df[tc].astype(str).str.startswith(tp["prefix"]), tp["positive"], tp["negative"])
    out = out[out.text.str.strip().str.len() > 0].reset_index(drop=True)
    return out


def _payloads() -> pd.DataFrame:
    """Static AgentDojo attack payloads (several attack templates x every injection task)."""
    from agentdojo.agent_pipeline import AgentPipeline
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.task_suite.load_suites import get_suites
    import agentdojo.attacks  # noqa: F401
    p = AgentPipeline([])
    p.name = "local"
    rows = []
    for sname, suite in get_suites("v1.2.2").items():
        ut = next(iter(suite.user_tasks.values()))
        for atk_name in ("important_instructions", "tool_knowledge", "ignore_previous", "system_message", "direct"):
            try:
                atk = load_attack(atk_name, suite, p)
            except Exception:
                continue
            for it in suite.injection_tasks.values():
                try:
                    inj = atk.attack(ut, it)
                except Exception:
                    continue
                for v in list(inj.values())[:1]:
                    rows.append({"text": v, "context": "", "label": 1})
    return pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)


def _benign_tooltext() -> pd.DataFrame:
    """Tool outputs from AgentDojo ground-truth runs with the default (non-injected) data."""
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suites
    rows = []
    for sname, suite in get_suites("v1.2.2").items():
        for ut in suite.user_tasks.values():
            env = ut.init_environment(suite.load_and_inject_default_environment({}))
            rt = FunctionsRuntime(suite.tools)
            try:
                for call in ut.ground_truth(env.model_copy(deep=True)):
                    out, err = rt.run_function(env, call.function, call.args)
                    if err is None:
                        rows.append({"text": tool_result_to_str(out)[:4000], "context": "", "label": 0})
            except Exception:
                continue
    return pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)


BUILTINS = {"agentdojo_payloads": _payloads, "agentdojo_benign_tooltext": _benign_tooltext}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _recall_at_fpr(pos, neg, fpr, rng):
    if len(pos) == 0 or len(neg) < 20:
        return float("nan")
    idx = rng.permutation(len(neg))
    fit, ev = neg[idx[: len(neg) // 2]], neg[idx[len(neg) // 2:]]
    tau = np.quantile(fit, 1 - fpr) if len(fit) else np.inf
    tau = np.nextafter(tau, np.inf)
    return float((pos >= tau).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--stack-config", type=Path, default=ROOT / "configs" / "stacks.yaml")
    ap.add_argument("--stack", default="shadow_all", help="stack whose layers are swept")
    ap.add_argument("--sweep-config", type=Path, default=ROOT / "configs" / "guard_sweep.yaml")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--max-per-dataset", type=int, default=0, help="stratified cap per dataset (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    spec = yaml.safe_load(args.sweep_config.read_text())["datasets"]
    names = args.datasets or list(spec)
    data, problems = {}, {}
    for n in names:
        try:
            d = load_dataset(n, spec[n], args.data_root.expanduser())
            if args.max_per_dataset and len(d) > args.max_per_dataset:
                d = d.groupby("label", group_keys=False).apply(
                    lambda g: g.sample(min(len(g), max(1, args.max_per_dataset * len(g) // len(d))),
                                       random_state=args.seed))
            data[n] = d.reset_index(drop=True)
            print(f"loaded {n}: {len(d)} items ({int(d.label.sum())} positive)")
        except Exception as e:
            problems[n] = f"{type(e).__name__}: {e}"
            print(f"SKIPPED {n}: {problems[n]}")

    layers = [l for l in load_stacks(args.stack_config)[args.stack].build() if l.stage in STAGE_ROLE]
    for l in layers:
        l.warmup()
    items_path = args.out / "items.jsonl"
    rows = []
    with open(items_path, "w") as fi:
        for layer in layers:
            role = STAGE_ROLE[layer.stage]
            for n, d in data.items():
                roles = spec[n]["role"] if isinstance(spec[n]["role"], list) else [spec[n]["role"]]
                if role not in roles:
                    continue
                scores, blocks, lat, errs = [], [], [], 0
                for t, c, y in zip(d.text, d.context, d.label):
                    if role == "response":
                        msgs = [{"role": "user", "content": [{"type": "text", "content": c}]},
                                {"role": "assistant", "content": [{"type": "text", "content": t}], "tool_calls": None}]
                        ctx = CheckContext(stage=layer.stage, text=t, messages=msgs, user_goal=c)
                    else:
                        text = (c + "\n\n" + t) if (c and role == "prompt") else t
                        msgs = [{"role": "user", "content": [{"type": "text", "content": text}]}]
                        ctx = CheckContext(stage=layer.stage, text=text, messages=msgs if role == "prompt" else [])
                    t0 = time.perf_counter()
                    try:
                        v = layer.check(ctx)
                        b, s = bool(v.block), (float(v.score) if v.score is not None else float(v.block))
                    except Exception:
                        errs += 1
                        b, s = False, float("nan")
                    lat.append((time.perf_counter() - t0) * 1000)
                    blocks.append(b); scores.append(s)
                    fi.write(json.dumps({"layer": layer.name, "dataset": n, "label": int(y), "block": b,
                                         "score": s}) + "\n")
                y = d.label.to_numpy(); sc = np.array(scores); bl = np.array(blocks)
                ok = ~np.isnan(sc)
                pos, neg = sc[(y == 1) & ok], sc[(y == 0) & ok]
                rows.append({"layer": layer.name, "stage": layer.stage.value, "family": layer.family, "dataset": n,
                             "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "errors": errs,
                             "recall": float(bl[y == 1].mean()) if (y == 1).any() else np.nan,
                             "fpr": float(bl[y == 0].mean()) if (y == 0).any() else np.nan,
                             "auc": _auc(pos, neg),
                             "recall_at_fpr1": _recall_at_fpr(pos, neg, 0.01, rng),
                             "recall_at_fpr5": _recall_at_fpr(pos, neg, 0.05, rng),
                             "latency_ms_p50": float(np.median(lat)) if lat else np.nan})
                print(f"{layer.name:45s} {n:28s} recall={rows[-1]['recall']:.3f} fpr={rows[-1]['fpr']:.3f} "
                      f"errors={errs}")
    pd.DataFrame(rows).to_csv(args.out / "guard_sweep.csv", index=False)
    # pooled per layer: positives and negatives from every dataset the layer saw (most datasets are
    # single-class, so AUC and recall@FPR are only defined on the pool)
    it = pd.read_json(items_path, lines=True) if items_path.stat().st_size else pd.DataFrame()
    pooled = []
    for lname, g in (it.groupby("layer") if len(it) else []):
        g = g[~g.score.isna()]
        pos, neg = g.score[g.label == 1].to_numpy(), g.score[g.label == 0].to_numpy()
        pooled.append({"layer": lname, "n_pos": len(pos), "n_neg": len(neg),
                       "recall": float(g.block[g.label == 1].mean()) if len(pos) else np.nan,
                       "fpr": float(g.block[g.label == 0].mean()) if len(neg) else np.nan,
                       "auc": _auc(pos, neg), "recall_at_fpr1": _recall_at_fpr(pos, neg, 0.01, rng),
                       "recall_at_fpr5": _recall_at_fpr(pos, neg, 0.05, rng),
                       "datasets": ",".join(sorted(g.dataset.unique()))})
    pd.DataFrame(pooled).to_csv(args.out / "guard_sweep_pooled.csv", index=False)
    (args.out / "problems.json").write_text(json.dumps(problems, indent=2))
    print(f"wrote {args.out}/guard_sweep.csv ({len(rows)} rows); dataset problems: {list(problems) or 'none'}")


if __name__ == "__main__":
    main()
