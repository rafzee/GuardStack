"""Known-answer tests for the adaptive (enforced-stack) MBR analysis."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from guardstack.analysis import adaptive as A  # noqa: E402

L1, L2, L3 = "cheap@tool_output", "gate@action", "monitor@output"   # listed in evaluation order
LAYERS = [L1, L2, L3]


def rec(case, k, success=False, fired=(), seqs=None, aborted=None, layers=LAYERS, error=None):
    suite, ut, it = case.split("/")
    return {"session_id": f"{case}:a{k}", "stack_layers": list(layers), "is_attack": True, "suite": suite,
            "user_task": ut, "injection_task": it, "category": "indirect:x", "attack_success": success,
            "fired": {l: l in fired for l in layers},
            "first_fire_seq": {l: (seqs or {}).get(l, 5 + i) for i, l in enumerate(layers) if l in fired},
            "aborted_by": aborted, "error": error, "guard_errors": {},
            "config": {"case_id": case, "attempt": k}}


def write(folder, recs, reached=()):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "shard000-of-001.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    hist = {}
    for r in recs:
        c, k = r["config"]["case_id"], r["config"]["attempt"]
        hist.setdefault(c, []).append({"k": k, "success": r["attack_success"], "fired": r["fired"],
                                       "tool_calls": [{"function": "f", "harmful": (c, k) in reached}]})
    (folder / "shard000-of-001.histories.jsonl").write_text("\n".join(
        json.dumps({"case_id": c, "attempts": a, "first_success_k": None}) for c, a in hist.items()) + "\n")
    (folder / "run_config.json").write_text(json.dumps({"budget": 3}))


def stack_run(tmp):
    """6 cases. Static: the cheap layer stops all 6. Adaptive: the cheap layer is evaded;
    c1 c2 succeed, c3 c4 are stopped by the monitor, c5 by the gate, c6 evades everything but fails."""
    r = []
    for i in range(1, 7):
        c = f"s/u{i}/i{i % 2}"
        r.append(rec(c, 1, fired=[L1], aborted=L1))
    r += [rec("s/u1/i1", 2, success=True), rec("s/u2/i0", 2, fired=[L3], aborted=L3), rec("s/u2/i0", 3, success=True),
          rec("s/u3/i1", 2, fired=[L3], aborted=L3), rec("s/u3/i1", 3, fired=[L1], aborted=L1),
          rec("s/u4/i0", 2, fired=[L1], aborted=L1), rec("s/u4/i0", 3, fired=[L3], aborted=L3),
          rec("s/u5/i1", 2, fired=[L2, L3], seqs={L2: 4, L3: 9}), rec("s/u6/i0", 2), rec("s/u6/i0", 3, fired=[L1], aborted=L1)]
    write(tmp / "S" / "tag", r, reached={("s/u4/i0", 3)})
    bare = [rec(f"s/u{i}/i{i % 2}", 1, success=i <= 5, layers=[]) for i in range(1, 7)]
    bare.append(rec("s/u6/i0", 2, success=True, layers=[]))
    write(tmp / "A_bare" / "tag", bare)
    return A.load_attempts(tmp / "S" / "tag"), A.load_attempts(tmp / "A_bare" / "tag")


def test_representative_and_outcome(tmp_path):
    cases, _ = stack_run(tmp_path)
    best = {c: A.representative(a, "best") for c, a in cases.items()}
    assert best["s/u2/i0"]["k"] == 3 and best["s/u2/i0"]["success"]          # first success
    assert best["s/u3/i1"]["k"] == 2                                         # got past the cheap layer to the monitor
    assert best["s/u4/i0"]["k"] == 3                                         # reached the target action
    assert best["s/u6/i0"]["k"] == 2                                         # evaded every layer beats a flagged one
    assert A.representative(cases["s/u6/i0"], "static")["k"] == 1 and A.representative(cases["s/u6/i0"], "last")["k"] == 3
    assert A.outcome(best["s/u1/i1"]) == A.SUCCESS
    assert A.outcome(best["s/u5/i1"]) == L2                                  # earliest fire in time gets the credit
    assert A.outcome(best["s/u6/i0"]) == A.UNSTOPPED_FAIL


def test_static_vs_adaptive_mbr(tmp_path):
    cases, bare = stack_run(tmp_path)
    assert A.effective_ids(bare, "static") == {f"s/u{i}/i{i % 2}" for i in range(1, 6)}
    assert len(A.effective_ids(bare, "best")) == 6
    table, summary = A.mbr_shift(cases, bare, n_boot=200)
    # static: 5 effective attacks, all stopped by the cheap layer
    assert table.loc[L1, "mbr_static"] == 1.0 and table.loc[L1, "survivors_static"] == 5
    assert np.isnan(table.loc[L3, "mbr_static"])                             # nothing left for the monitor
    # adaptive (6 effective): every best shot gets past the cheap layer; gate stops u5; monitor stops u3, u4
    assert table.loc[L1, "mbr_adaptive"] == 0.0
    assert np.isclose(table.loc[L2, "mbr_adaptive"], 1 / 6)
    assert np.isclose(table.loc[L3, "mbr_adaptive"], 2 / 5) and table.loc[L3, "survivors_adaptive"] == 5
    assert np.isclose(table.loc[L1, "shift"], -1.0)
    assert table.loc[L1, "lo_adaptive"] <= table.loc[L1, "mbr_adaptive"] <= table.loc[L1, "hi_adaptive"]
    ad = summary.loc["adaptive"]
    assert ad.n_cases == 6 and np.isclose(ad.asr, 2 / 6) and ad.success_effective == 2
    assert ad.stopped_by_a_layer == 3 and ad.unstopped_fail == 1
    assert summary.loc["static", "asr"] == 0.0


def test_pooled_and_last_differ_from_best(tmp_path):
    cases, bare = stack_run(tmp_path)
    eff = A.effective_ids(bare, "best")
    pooled = A.pooled_mbr(cases, effective=eff)                              # 10 probe attempts
    assert pooled.loc[L1, "survivors_in"] == 10 and np.isclose(pooled.loc[L1, "mbr"], 3 / 10)
    last, _ = A.sequential_mbr(cases, "last", effective=eff, n_boot=0)
    assert np.isclose(last.loc[L1, "mbr"], 2 / 6)                            # u3 and u6 end on a flagged attempt


def test_errored_attempts_are_dropped_and_error_free_wins(tmp_path):
    f = tmp_path / "S" / "tag"
    write(f, [rec("s/u1/i1", 1, error="ParserError: x"), rec("s/u1/i1", 1, fired=[L1], aborted=L1),
              rec("s/u1/i1", 2, error="boom")])
    cases = A.load_attempts(f)
    assert [a["k"] for a in cases["s/u1/i1"]] == [1] and cases["s/u1/i1"][0]["fired"][L1]


def test_cofire_counts_only_observed_pairs(tmp_path):
    cases, _ = stack_run(tmp_path)
    t = A.cofire_table(cases, "static").set_index(["layer_i", "layer_j"])
    assert t.loc[(L2, L3), "n_observed"] == 0 and t.loc[(L2, L3), "n_total"] == 6   # the cheap layer ended every session
    assert t.loc[(L1, L2), "n_observed"] == 6 and t.loc[(L1, L2), "i_fired"] == 6


def test_cli_writes_tables(tmp_path):
    stack_run(tmp_path / "run")
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_adaptive_mbr.py"), str(tmp_path / "run"),
                        "--out", str(out), "--n-boot", "50"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    for name in ("mbr_shift.csv", "summary.csv", "mbr_ablation.csv", "cofire.csv", "summary.md", "fig_mbr_shift.png"):
        assert (out / name).exists(), name
    assert "mbr_static" in (out / "mbr_shift.csv").read_text()


def test_export_payloads_and_replay_through_the_runner(tmp_path):
    """Best-shot payloads are exported per case and the runner uses them in place of the static text."""
    sys.path.insert(0, str(ROOT / "tests"))
    from agentdojo.task_suite.load_suites import get_suites
    from fakes import OracleVictimLLM
    from guardstack.runner import AgentDojoRunner, enumerate_agentdojo_cases
    from guardstack.stacks import StackConfig
    suite = get_suites("v1.2.2")["banking"]
    case = [c for c in enumerate_agentdojo_cases(suites=["banking"], include_benign=False) if c.user_task == "user_task_0"][0]
    goal = suite.get_injection_task_by_id(case.injection_task).GOAL
    # an adaptive run folder: the seed fails, attempt 2 (the bare goal) succeeds
    r1, r2 = rec(case.case_id.rsplit("/", 1)[0], 1, fired=[L1], aborted=L1), rec(case.case_id.rsplit("/", 1)[0], 2, success=True)
    for r, payload in ((r1, "<INFORMATION> seed"), (r2, goal)):
        r["config"].update(case_id=case.case_id, payload=payload)
    write(tmp_path / "run" / "E" / "tag", [r1, r2])
    out = tmp_path / "payloads.jsonl"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_payloads.py"), str(tmp_path / "run"),
                        "--stack", "E", "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["payload"] == goal and rows[0]["attempt"] == 2 and rows[0]["success_on_source"]

    shadow = StackConfig("shadow", [{"type": "regex", "stage": "tool_output", "mode": "shadow"}])
    mk = lambda payloads: AgentDojoRunner(shadow, lambda: OracleVictimLLM(suite), "sys", "oracle",  # noqa: E731
                                          run_config={"run_id": "t"}, payloads=payloads)
    replay = mk({case.case_id: goal}).run_case(case)
    static = mk(None).run_case(case)
    assert static.fired["regex@tool_output"] and not replay.fired["regex@tool_output"]   # the layer sees the new text
    assert replay.attack_success and replay.error is None


def test_each_attempt_keeps_its_own_payload(tmp_path):
    """Regression: the payload was read from a leaked loop variable, so every attempt of every case
    got the last record's payload and the exported replay files held one payload repeated."""
    f = tmp_path / "S" / "tag"
    recs = []
    for i in (1, 2):
        for k in (1, 2):
            r = rec(f"s/u{i}/i{i}", k, success=(k == 2))
            r["config"]["payload"] = f"payload-u{i}-k{k}"
            recs.append(r)
    write(f, recs)
    cases = A.load_attempts(f)
    assert {a["payload"] for c in cases.values() for a in c} == {f"payload-u{i}-k{k}" for i in (1, 2) for k in (1, 2)}
    assert A.representative(cases["s/u1/i1"], "best")["payload"] == "payload-u1-k2"   # the attempt that worked
    assert A.representative(cases["s/u2/i2"], "static")["payload"] == "payload-u2-k1"


def test_mbr_conditions_in_evaluation_order_not_listing_order(tmp_path):
    """Stack E lists the gate (action) before AlignmentCheck (trace), but the pipeline runs trace
    layers first on each proposed call. MBR must condition in evaluation order: the gate's
    denominator is what AlignmentCheck let through, not every case the earlier filter passed."""
    pg, lg, gate, spot, align = ("pg2@tool_output", "lg@output", "gate@action",
                                 "spot@tool_output", "align@trace")
    listed = [pg, lg, gate, spot, align]
    assert A.evaluation_order(listed) == [pg, spot, align, gate, lg]
    with pytest.raises(ValueError):
        A.evaluation_order(["no_stage_suffix"])
    # 10 effective cases: pg2 stops 2, align stops 6, the gate stops 1 of the 2 align passed, 1 succeeds
    plan = [pg] * 2 + [align] * 6 + [gate] + [None]
    recs = [rec(f"s/u{i}/i{i % 3}", 1, success=who is None, fired=[who] if who else [], aborted=who,
                layers=listed) for i, who in enumerate(plan)]
    write(tmp_path / "E" / "tag", recs)
    cases = A.load_attempts(tmp_path / "E" / "tag")
    table, summary = A.sequential_mbr(cases, "static", n_boot=0)
    assert list(table.index) == [pg, spot, align, gate, lg]
    assert np.isclose(table.loc[align, "mbr"], 6 / 8) and table.loc[gate, "survivors_in"] == 2
    assert np.isclose(table.loc[gate, "mbr"], 1 / 2)                 # listing order would give 1/8
    assert summary["success_effective"] == 1


def test_find_runs_rejects_a_stack_found_twice(tmp_path):
    write(tmp_path / "C" / "tag", [rec("s/u1/i1", 1, success=True)])
    assert set(A.find_runs(tmp_path)) == {("C", "tag")}
    write(tmp_path / "A_bare" / "C" / "tag", [rec("s/u1/i1", 1, success=True)])
    with pytest.raises(ValueError, match="found twice"):
        A.find_runs(tmp_path)
