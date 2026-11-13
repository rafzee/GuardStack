#!/usr/bin/env python
"""One-screen summary of a run directory (no torch; safe on a login node).

    python scripts/summarize_run.py runs/<RUN_ID>
    python scripts/summarize_run.py runs/<RUN_ID> --errored-stacks     # for a repair pass:
        STACKS="$(python scripts/summarize_run.py runs/pathA_gptoss --errored-stacks)" \
            bash slurm/submit_path_a.sh gptoss static
    python scripts/summarize_run.py runs/<RUN_ID> --check --max-error-rate 0.05   # exit 1 if not
        # every result folder has records and at most 5% errored cases (gate for dependent jobs)

Results are grouped by folder under the run (a stack, or a stack's adaptive tag folder). Each
session counts once: a resumed run appends a new record for a session that errored, so the latest
error-free record wins (otherwise the latest record). A session is a case in a static folder and a
(case, attempt) pair in an adaptive folder, so the gate's error rate counts errored attempts: an
adaptive case can look fine while most of its attempts errored (the Nov 9 YAML bug).
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(root: Path) -> dict[str, dict[str, dict]]:
    """result folder (relative to the run, e.g. 'shadow_all' or 'E/adaptive_rewrite_guard_b5')
    -> session key -> record (deduplicated). The key is the case id, plus ':a<k>' for attempt k of
    an adaptive attack."""
    by_stack: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in sorted(root.rglob("shard*.jsonl")):
        if f.name.endswith((".events.jsonl", ".histories.jsonl")):
            continue
        key = f.parent.relative_to(root).as_posix()
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # a line cut off by a killed job
                continue
            if "session_id" not in r:
                continue
            cid = r["config"]["case_id"]
            if r["config"].get("attempt") is not None:
                cid = f"{cid}:a{r['config']['attempt']}"
            prev = by_stack[key].get(cid)
            if prev is None or prev.get("error") or not r.get("error"):
                by_stack[key][cid] = r
    return by_stack


def stack_token(folder: str) -> str:
    """Result folder -> the STACKS token that reruns it (iso_x -> iso:x; a tag subfolder -> its stack)."""
    stack_id = folder.split("/")[0]
    return "iso:" + stack_id[4:] if stack_id.startswith("iso_") else stack_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--errored-stacks", action="store_true",
                    help="print only the STACKS tokens of stacks with errored cases, space-separated")
    ap.add_argument("--check", action="store_true", help="exit 1 unless every folder passes --max-error-rate")
    ap.add_argument("--max-error-rate", type=float, default=0.05)
    a = ap.parse_args()
    by_stack = load(a.run_dir)
    if a.errored_stacks:
        toks = dict.fromkeys(stack_token(s) for s, cases in sorted(by_stack.items())
                             if any(r.get("error") for r in cases.values()))
        print(" ".join(toks))
        return
    if a.check:
        bad = []
        for s_, cases in sorted(by_stack.items()):
            n = len(cases)
            e = sum(1 for r in cases.values() if r.get("error"))
            unit = "attempts" if any(r["config"].get("attempt") is not None for r in cases.values()) else "cases"
            print(f"{s_}: {n} {unit}, {e} errored ({e / n:.1%})" if n else f"{s_}: no records")
            if not n or e / n > a.max_error_rate:
                bad.append(s_)
        if not by_stack:
            print(f"no records under {a.run_dir}")
        ok = bool(by_stack) and not bad
        print("CHECK", "PASSED" if ok else f"FAILED: {bad or 'no records'}")
        sys.exit(0 if ok else 1)
    for sid, cases in sorted(by_stack.items()):
        rs = list(cases.values())
        if any(r["config"].get("attempt") is not None for r in rs):
            by_case: dict[str, list] = defaultdict(list)
            for r in rs:
                by_case[r["config"]["case_id"]].append(r)
            err = Counter((r.get("error") or "").split(":")[0] for r in rs if r.get("error"))
            broken = sum(1 for v in by_case.values() if any(r.get("attack_success") for r in v))
            first = sum(1 for v in by_case.values() if any(r.get("attack_success") and r["config"]["attempt"] == 1
                                                           for r in v))
            n = len(by_case)
            print(f"== {sid}: {n} cases, {len(rs)} attempts, errored attempts: {dict(err) or 0}")
            print(f"   broken at attempt 1: {first}/{n} | within the budget: {broken}/{n}")
            continue
        ok = [r for r in rs if not r.get("error")]
        att = [r for r in ok if r["is_attack"]]
        ben = [r for r in ok if not r["is_attack"]]
        err = Counter((r.get("error") or "").split(":")[0] for r in rs if r.get("error"))
        asr = sum(bool(r["attack_success"]) for r in att) / len(att) if att else float("nan")
        util = sum(bool(r["utility"]) for r in ben) / len(ben) if ben else float("nan")
        util_att = sum(bool(r["utility"]) for r in att) / len(att) if att else float("nan")
        wall = [r["wall_ms"] for r in ok]
        calls = [r["n_llm_calls"] for r in ok]
        fired = Counter(l for r in att for l, f in r["fired"].items() if f)
        print(f"== {sid}: {len(rs)} cases ({len(att)} attack, {len(ben)} benign ok), errored: {dict(err) or 0}")
        print(f"   ASR {asr:.3f} | utility benign {util:.3f} | utility under attack {util_att:.3f}")
        if wall:
            print(f"   session wall ms median {st.median(wall):.0f}, LLM calls/session median {st.median(calls):.0f}, "
                  f"retries {sum(r.get('n_retries', 0) for r in ok)}, skipped {sum(1 for r in ok if r.get('skipped_reason'))}")
        if fired:
            print("   fired on attacks: " + ", ".join(f"{l}={n}" for l, n in fired.most_common()))


if __name__ == "__main__":
    main()
