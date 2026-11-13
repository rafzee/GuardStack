"""scripts/summarize_run.py: dedupe of resumed cases, grouping by folder, repair list, gate."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_run.py"


def _rec(stack, case, err=None, attack=True):
    return {"session_id": f"{stack}:{case}", "stack_id": stack, "config": {"case_id": case}, "error": err,
            "is_attack": attack, "attack_success": False, "utility": True, "wall_ms": 1.0, "n_llm_calls": 1,
            "fired": {}}


def _write(p: Path, recs):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs) + '{"cut off by a killed job')


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_repair_list_and_gate(tmp_path):
    run = tmp_path / "pathA_x"
    _write(run / "A_bare" / "shard000-of-002.jsonl", [_rec("A_bare", "c1"), _rec("A_bare", "c2")])
    # c1 errored, then the repair pass fixed it; c2 still errored
    _write(run / "iso_nemo_in" / "shard000-of-002.jsonl",
           [_rec("iso_nemo_in", "c1", "guard_error: x"), _rec("iso_nemo_in", "c2", "guard_error: x"),
            _rec("iso_nemo_in", "c1")])
    # adaptive records under a tag folder must not mix with the static ones of the same stack
    _write(run / "A_bare" / "adaptive_rewrite_guard_b5" / "shard000-of-002.jsonl", [_rec("A_bare", "c1", "boom")])
    assert _run(run, "--errored-stacks").stdout.split() == ["A_bare", "iso:nemo_in"]
    out = _run(run).stdout
    assert "== A_bare: 2 cases" in out and "== iso_nemo_in: 2 cases" in out
    g = _run(run, "--check", "--max-error-rate", "0.6")
    assert g.returncode == 1 and "A_bare/adaptive_rewrite_guard_b5" in g.stdout   # 1/1 errored
    assert _run(run / "A_bare", "--check").returncode == 1                        # tag folder fails
    assert _run(tmp_path / "empty", "--check").returncode == 1
    _write(tmp_path / "ok" / "B" / "shard000-of-001.jsonl", [_rec("B", "c1"), _rec("B", "c2", "x"), _rec("B", "c3")])
    assert _run(tmp_path / "ok", "--check", "--max-error-rate", "0.5").returncode == 0


def test_adaptive_errors_count_per_attempt(tmp_path):
    def att(case, k, err=None, success=False):
        r = _rec("E", case, err)
        r["config"]["attempt"] = k
        r["attack_success"] = success
        return r
    run = tmp_path / "smoke_adaptive"
    # one case, 4 attempts, 3 of them errored: the case itself has an error-free record
    _write(run / "E" / "adaptive_rewrite_trace_b10" / "shard000-of-001.jsonl",
           [att("c1", 1), att("c1", 2, "ParserError: x"), att("c1", 3, "ParserError: x"), att("c1", 4, "ParserError: x"),
            att("c2", 1), att("c2", 2, success=True)])
    g = _run(run, "--check", "--max-error-rate", "0.3")
    assert g.returncode == 1 and "6 attempts, 3 errored" in g.stdout
    out = _run(run).stdout
    assert "2 cases, 6 attempts" in out and "within the budget: 1/2" in out and "attempt 1: 0/2" in out
