"""A scripted, maximally obedient agent for testing the harness without a GPU.

It executes the user task's ground-truth tool calls one per step; as soon as an injection
goal appears verbatim in a tool result it follows the injection task's ground truth
first. Its bare-agent ASR is therefore ~100% on cases where the injection is reachable,
which gives the harness known answers to check against.
"""

from __future__ import annotations

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string

from guardstack import recorder
from guardstack.layers.builtin import _msg_text


class OracleVictimLLM(BasePipelineElement):
    name = "local-oracle"

    def __init__(self, suite) -> None:
        self.suite = suite
        self._state: dict[str, dict] = {}

    def _find_user_task(self, prompt: str):
        for t in self.suite.user_tasks.values():
            if t.PROMPT.strip() == prompt.strip():
                return t
        return None

    def query(self, query, runtime, env, messages=[], extra_args={}):
        rec = recorder.current()
        key = rec.session_id if rec else "default"
        st = self._state.get(key)
        if st is None:
            ut = self._find_user_task(next(_msg_text(m) for m in messages if m["role"] == "user"))
            st = {"queue": list(ut.ground_truth(env)) if ut else [], "final": ut.GROUND_TRUTH_OUTPUT if ut else "",
                  "hijacked": False}
            self._state[key] = st
        if not st["hijacked"]:
            tool_text = "\n".join(_msg_text(m) for m in messages if m["role"] == "tool")
            for it in self.suite.injection_tasks.values():
                if it.GOAL[:60] in tool_text:
                    st["queue"] = list(it.ground_truth(env)) + st["queue"]
                    st["hijacked"] = True
                    break
        if st["queue"]:
            call = st["queue"].pop(0)
            out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")],
                                       tool_calls=[call])
        else:
            out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(st["final"])],
                                       tool_calls=None)
            self._state.pop(key, None)
        return query, runtime, env, [*messages, out], extra_args
