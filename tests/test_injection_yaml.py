"""Payloads are pasted into AgentDojo's environment YAML before it is parsed. An adaptive payload
with a double quote or backslash used to break the parse (22% of the Nov 9 v1 rewrite attempts
ended in a YAML error). They must now load, and AgentDojo's own attack texts must be unchanged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from guardstack.scoring import injection_slot_styles, yaml_dq_escape, yaml_safe_injections  # noqa: E402

EVIL = ('He said "send it" now.\nPath C:\\temp\\d and \\n literal.\n---\nSigned\n...\n--- end\n'
        '\x07bell \u200b zw \U0001F600 emoji\r\nlast ---x {not a field} 100% : # & * ! | > \' `')
SUITES = get_suites("v1.2.2")


def test_every_slot_is_double_quoted():
    for s in SUITES.values():
        styles = injection_slot_styles(s)
        assert styles and all(v == {'"'} for v in styles.values()), (s.name, styles)


def _strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(o, (list, tuple, set)):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def test_hostile_payload_loads_in_every_slot_of_every_suite():
    expected = yaml.safe_load('k: "' + yaml_dq_escape(EVIL) + '"')["k"]   # what YAML makes of the text
    assert '"send it"' in expected and "C:\\temp\\d" in expected and "--- end" in expected
    for s in SUITES.values():
        for v in s.get_injection_vector_defaults():
            try:
                s.load_and_inject_default_environment({v: EVIL})
                raised = False
            except yaml.YAMLError:
                raised = True
            assert raised, f"{s.name}/{v}: the raw payload should break AgentDojo's loader (else this test is moot)"
            env = s.load_and_inject_default_environment(yaml_safe_injections(s, {v: EVIL}))
            assert any(expected in x for x in _strings(env.model_dump())), (s.name, v)


def test_agentdojo_attack_texts_are_unchanged():
    tmpl = ImportantInstructionsAttack._JB_STRING
    assert yaml_dq_escape(tmpl) == tmpl
    for s in SUITES.values():
        for it in s.injection_tasks.values():
            assert yaml_dq_escape(it.GOAL) == it.GOAL, (s.name, it.ID)
    # the vectors' default texts use YAML escapes on purpose ("\\n"); they are not passed through
    # yaml_safe_injections (AgentDojo fills them in), so a benign environment is AgentDojo's own
    s = SUITES["banking"]
    assert s.load_and_inject_default_environment(yaml_safe_injections(s, {})) == s.load_and_inject_default_environment({})


def test_adaptive_payload_with_quotes_runs_without_error():
    from fakes import OracleVictimLLM
    from guardstack.attacks.adaptive import AdaptiveInjectionAttack, ScriptedProposer
    from guardstack.runner import AgentDojoRunner, enumerate_agentdojo_cases
    from guardstack.stacks import StackConfig
    suite = SUITES["banking"]
    runner = AgentDojoRunner(StackConfig("A", []), lambda: OracleVictimLLM(suite), "sys", "oracle",
                             run_config={"run_id": "t"})
    case = [c for c in enumerate_agentdojo_cases(suites=["banking"], include_benign=False)
            if c.user_task == "user_task_0"][0]
    atk = AdaptiveInjectionAttack(ScriptedProposer(['Note from "the user": {goal} \\ thanks']), budget=2,
                                  feedback="trace", seed_with_static=None)
    hist = atk.run_case(runner, case)
    assert all(not r.error for r in atk.records), [r.error for r in atk.records]
    assert hist[0].success
