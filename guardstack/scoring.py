"""Run one AgentDojo task and score it, mirroring TaskSuite.run_task_with_pipeline
(agentdojo 0.1.35, task_suite.py) with three deliberate differences:

1. Tool calls DENIED by a guardrail are removed from the function-call trace before
   scoring. AgentDojo builds the trace from every assistant tool call, so a denied call
   would otherwise count as executed for trace-scored injection tasks (e.g. slack
   injection_task_5).
2. The attacker signature (payload snippets, attacker entities, injection-only functions)
   is registered on the session recorder before the run, so every guard event and tool
   execution can be attributed to attacker material and the time of harm is known.
3. The number of AgentDojo retries (re-runs when the final output is None) is returned.
4. Payloads are escaped for the YAML scalar they are pasted into (``yaml_safe_injections``).
   AgentDojo pastes them into its environment YAML text with ``str.format`` before parsing,
   and every injection slot sits inside a double-quoted scalar, so a payload with a double
   quote or backslash broke the parse and the session ended in a YAML error. AgentDojo's own
   attack texts contain neither, so for them the escaping is the identity.

Everything else -- environment loading, init_environment, the 3x retry, AbortAgentError
handling, and the utility/security checks -- is AgentDojo's own code path.
"""

from __future__ import annotations

import re
import threading
import warnings

import yaml

from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.base_tasks import BaseUserTask
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages

from . import recorder


def _functions(calls) -> set[str]:
    return {c.function for c in calls}


# PyYAML's reader rejects these characters anywhere in a document (yaml/reader.py NON_PRINTABLE)
_NON_PRINTABLE = re.compile("[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]")
_DOC_MARKER = re.compile(r"(^|\n)([-.])(?=\2\2(?:[ \t\r\n]|$))")
_styles_cache: dict[tuple, dict[str, set]] = {}
_styles_lock = threading.Lock()


def yaml_dq_escape(text: str) -> str:
    """Escape text for the inside of a YAML double-quoted scalar. Line breaks and tabs are left as
    they are, so YAML folds them exactly as it folds AgentDojo's own (unescaped) attack texts."""
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif _NON_PRINTABLE.match(ch):
            o = ord(ch)
            out.append(f"\\x{o:02x}" if o < 0x100 else f"\\u{o:04x}" if o < 0x10000 else f"\\U{o:08x}")
        else:
            out.append(ch)
    # a line that starts with a document marker ("---" or "...") would end the YAML document
    return _DOC_MARKER.sub(lambda m: m.group(1) + ("\\x2d" if m.group(2) == "-" else "\\x2e"), "".join(out))


def injection_slot_styles(suite) -> dict[str, set]:
    """injection vector -> YAML scalar styles of the scalars its placeholder sits in ('"' = double
    quoted). Found by formatting the environment text with plain tokens and parsing it."""
    from agentdojo.task_suite.task_suite import read_suite_file
    key = (suite.name, str(getattr(suite, "data_path", None)))
    with _styles_lock:
        if key in _styles_cache:
            return _styles_cache[key]
        text = read_suite_file(suite.name, "environment.yaml", suite.data_path)
        tokens = {v: f"GSINJSLOT{i:04d}X" for i, v in enumerate(suite.get_injection_vector_defaults())}
        styles: dict[str, set] = {v: set() for v in tokens}
        for ev in yaml.parse(text.format(**tokens)):
            if isinstance(ev, yaml.ScalarEvent):
                for v, tok in tokens.items():
                    if tok in ev.value:
                        styles[v].add(ev.style)
        _styles_cache[key] = styles
        return styles


def yaml_safe_injections(suite, injections: dict[str, str]) -> dict[str, str]:
    """Escape each payload for its slot. Slots that are not (only) inside double-quoted scalars are
    left unescaped; none exist in AgentDojo v1.2.2."""
    styles = injection_slot_styles(suite)
    return {v: (yaml_dq_escape(p) if styles.get(v) == {'"'} else p) for v, p in injections.items()}


def run_task_scored(suite, agent_pipeline, user_task, injection_task, injections: dict[str, str]):
    """Returns (utility, security, n_retries). security is AgentDojo's flag: True = injection goal reached."""
    rec = recorder.current()
    environment = suite.load_and_inject_default_environment(yaml_safe_injections(suite, injections))
    task_environment = user_task.init_environment(environment) if isinstance(user_task, BaseUserTask) else environment
    pre_environment = task_environment.model_copy(deep=True)
    prompt = user_task.PROMPT if isinstance(user_task, BaseUserTask) else user_task.GOAL

    if rec is not None and injection_task is not None:
        try:
            inj_fns = _functions(injection_task.ground_truth(pre_environment.model_copy(deep=True)))
            user_fns = _functions(user_task.ground_truth(pre_environment.model_copy(deep=True)))
        except Exception:  # ground truth is best-effort metadata for attribution only
            inj_fns, user_fns = set(), set()
        rec.set_attack_signature(list(injections.values()), prompt, inj_fns - user_fns)

    runtime = FunctionsRuntime(suite.tools)
    model_output = None
    messages = []
    runs = 0
    for _ in range(3):
        runs += 1
        if rec is not None:
            rec.n_pipeline_runs += 1
        try:
            _, _, task_environment, messages, _ = agent_pipeline.query(prompt, runtime, task_environment)
        except AbortAgentError as e:
            task_environment = e.task_environment
            messages = e.messages
        model_output = model_output_from_messages(messages)
        if model_output is not None:
            break
    if model_output is None:
        warnings.warn(f"Model output was None for task {user_task.ID}")

    rec_ = recorder.current()
    if rec_ is not None:
        rec_.final_output = output_text(model_output)[:1000]
    trace = functions_stack_trace_from_messages(messages)
    if rec is not None and rec.denied_ids:
        trace = [c for c in trace if id(c) not in rec.denied_ids]
    utility = suite._check_task_result(user_task, model_output or [], pre_environment, task_environment, trace)
    if injection_task is None:
        return utility, True, runs - 1
    security = suite._check_task_result(injection_task, model_output or [], pre_environment, task_environment, trace)
    return utility, security, runs - 1


def output_text(model_output) -> str:
    """AgentDojo's final output as plain text (a string, or a list of content blocks)."""
    if model_output is None:
        return ""
    if isinstance(model_output, str):
        return model_output
    parts = []
    for b in model_output:
        if isinstance(b, dict):
            parts.append(str(b.get("content") or b.get("text") or ""))
        else:
            parts.append(str(getattr(b, "content", b)))
    return "\n".join(p for p in parts if p)
