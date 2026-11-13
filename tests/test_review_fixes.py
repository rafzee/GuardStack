"""Regression tests for the independent code review (docs/review/code_review.md).
Each test is the reviewer's reproduction turned into an assertion."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from fakes import OracleVictimLLM  # noqa: E402
from guardstack.analysis import selection, structure  # noqa: E402
from guardstack.analysis.metrics import twoway_bootstrap  # noqa: E402
from guardstack.layers.base import GuardLayer  # noqa: E402
from guardstack.runner import AgentDojoRunner, enumerate_agentdojo_cases  # noqa: E402
from guardstack.stacks import REGISTRY, StackConfig  # noqa: E402
from guardstack.types import Stage, Verdict  # noqa: E402


class AlwaysFire(GuardLayer):
    family = "test"

    def check(self, ctx):
        return Verdict(block=True, score=1.0)


class Crash(GuardLayer):
    family = "test"
    default_stage = Stage.TOOL_OUTPUT

    def check(self, ctx):
        raise RuntimeError("context too long")


REGISTRY["always_fire"] = "test_review_fixes:AlwaysFire"
REGISTRY["crash"] = "test_review_fixes:Crash"


def _runner(suite_name, stack):
    suite = get_suites("v1.2.2")[suite_name]
    return AgentDojoRunner(stack, lambda: OracleVictimLLM(suite), "sys", "oracle", run_config={"run_id": "rv"})


def _cases(suite_name, users=None, injections=None, benign=True):
    out = []
    for c in enumerate_agentdojo_cases(suites=[suite_name], include_benign=benign):
        if users and c.user_task not in users:
            continue
        if injections and c.injection_task is not None and c.injection_task not in injections:
            continue
        out.append(c)
    return out


def _records(runner, cases):
    return [json.loads(runner.run_case(c).to_json()) for c in cases]


def test_c1_posthoc_output_fire_is_not_a_catch():
    stack = StackConfig("s", [{"type": "always_fire", "stage": "output"}], shadow_all=True)
    recs = _records(_runner("banking", stack), _cases("banking", users=["user_task_0", "user_task_1", "user_task_2"]))
    V = structure.VerdictMatrix.from_records(recs)
    assert V.n_eff > 0
    harmful = [r for r in recs if r["is_attack"] and r["attack_success"] and r["harm_seq"] is not None]
    assert harmful, "scripted victim should execute attacker calls"
    # the output filter fires after send_money etc. -> neither a stop nor a catch for those cases
    assert V.stop_or_catch.sum() <= V.n_eff - len(harmful)
    assert selection.predicted_residual_asr(V, V.layers) >= len(harmful) / V.n_attacks - 1e-9


def test_c2_temperature_zero_and_seed_are_sent():
    from guardstack.llm import TrackedOpenAILLM, UsageClient
    from agentdojo.functions_runtime import FunctionsRuntime
    sent = {}

    class Fake:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent.update(kw)
                    from types import SimpleNamespace as N
                    msg = N(role="assistant", content="ok", tool_calls=None, reasoning_content=None)
                    return N(choices=[N(message=msg)], usage=N(prompt_tokens=1, completion_tokens=1))
        completions = None

    llm = TrackedOpenAILLM(UsageClient(Fake(), role="agent"), "meta-llama/Llama-3.1-8B-Instruct", temperature=0.0,
                           seed=7)
    sysm = {"role": "system", "content": [{"type": "text", "content": "sys"}]}
    user = {"role": "user", "content": [{"type": "text", "content": "hi"}]}
    llm.query("hi", FunctionsRuntime([]), None, [sysm, user])
    assert sent["temperature"] == 0.0 and sent["seed"] == 7
    assert sent["messages"][0]["role"] == "system"


def test_c3_denied_calls_are_not_scored_as_executed():
    stack = StackConfig("deny", [{"type": "tool_allowlist", "stage": "action",
                                  "params": {"allowed": ["read_channel_messages", "get_channels", "read_inbox",
                                                         "get_webpage", "get_users_in_channel"]}}])
    cases = [c for c in _cases("slack", benign=False, injections=["injection_task_5"])][:6]
    recs = _records(_runner("slack", stack), cases)
    assert recs and not any(r["attack_success"] for r in recs)
    assert any(r["denied_calls"] > 0 for r in recs)


def test_c4_guard_errors_are_visible_and_excluded():
    stack = StackConfig("c", [{"type": "crash", "stage": "tool_output"}], shadow_all=True)
    recs = _records(_runner("banking", stack), _cases("banking", users=["user_task_0"]))
    assert all(r["guard_errors"] for r in recs if r["guard_calls"])
    assert all(r["error"] and r["error"].startswith("guard_error") for r in recs if r["guard_errors"])
    V = structure.VerdictMatrix.from_records(recs)
    assert V.notes["excluded_guard_errors"] > 0


def test_m3_spotlight_does_not_blind_the_gate():
    base = [{"type": "provenance_gate", "stage": "action"}]
    with_spot = [{"type": "spotlight", "stage": "tool_output", "params": {"method": "datamark"}}] + base
    cases = [c for c in _cases("banking", benign=False, users=["user_task_0", "user_task_1"])
             if c.injection_task in ("injection_task_5", "injection_task_6")]
    a = _records(_runner("banking", StackConfig("g", base)), cases)
    b = _records(_runner("banking", StackConfig("sg", with_spot)), cases)
    assert [r["attack_success"] for r in a] == [r["attack_success"] for r in b]


def test_m6_input_fire_on_benign_prompt_is_a_stop_not_a_catch():
    stack = StackConfig("i", [{"type": "always_fire", "stage": "input"}], shadow_all=True)
    recs = _records(_runner("banking", stack), _cases("banking", users=["user_task_0", "user_task_1"]))
    V = structure.VerdictMatrix.from_records(recs)
    assert V.stop.all() and not V.catch.any()


def test_m1_twoway_bootstrap_runs_and_brackets_point():
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(20), 7)
    injs = np.tile(np.arange(7), 20)
    y = (rng.random(140) < 0.3 + 0.08 * (injs % 3)).astype(float)
    p, lo, hi = twoway_bootstrap(lambda ix: y[ix].mean(), users, injs, n_boot=300)
    assert lo <= p <= hi and hi - lo > 0.05


def test_split_parallel_tool_calls_for_llama_templates():
    from guardstack.llm import split_parallel_tool_calls as split
    tc = lambda i: {"id": f"c{i}", "type": "function", "function": {"name": f"f{i}", "arguments": "{}"}}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "thinking", "tool_calls": [tc(1), tc(2), tc(3)]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c3", "content": "r3"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "assistant", "content": "done"}]
    out = split(msgs)
    assert [(m["role"], m.get("tool_call_id") or [t["id"] for t in m.get("tool_calls") or []] or m.get("content"))
            for m in out] == [("system", "s"), ("user", "u"), ("assistant", ["c1"]), ("tool", "c1"),
                              ("assistant", ["c2"]), ("tool", "c2"), ("assistant", ["c3"]), ("tool", "c3"),
                              ("assistant", "done")]
    assert out[2]["content"] == "thinking" and out[4]["content"] is None
    one = [{"role": "assistant", "content": None, "tool_calls": [tc(1)]}, {"role": "tool", "tool_call_id": "c1", "content": "r"}]
    assert split(one) == one
