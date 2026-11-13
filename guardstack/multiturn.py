"""Turn-by-turn guarded sessions on an AgentDojo environment, for the direct and multi-turn
vectors (the attacker writes the user turns).

Each send() runs one full guarded turn: input guard -> agent + tool loop (action/trace/
tool-output guards) -> output guard. rollback() restores the conversation AND the environment
to before the last turn, which is what Crescendo-style backtracking needs: the target must
forget the refused exchange, and any tool side effects of that turn must be undone.
"""

from __future__ import annotations

import copy
import warnings

from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.functions_runtime import FunctionsRuntime

from . import recorder
from .attacks.adaptive import TurnSession
from .layers.builtin import _msg_text
from .pipeline import build_pipeline


class AgentDojoTurnSession(TurnSession):
    def __init__(self, suite, llm, layers, system_message: str, env=None, model_name: str = "",
                 max_iters: int = 15) -> None:
        self.suite = suite
        self.env = env if env is not None else suite.load_and_inject_default_environment({})
        self.runtime = FunctionsRuntime(suite.tools)
        self.messages: list = []
        self._snapshots: list[tuple[list, object]] = []
        kw = dict(max_iters=max_iters, name=f"local|{model_name}|multiturn", model_name=model_name)
        self._first = build_pipeline(llm, system_message, layers, first_turn=True, **kw)
        self._next = build_pipeline(llm, system_message, layers, first_turn=False, **kw)

    def send(self, user_msg: str) -> tuple[str, str | None]:
        self._snapshots.append((copy.deepcopy(self.messages), self.env.model_copy(deep=True)))
        rec = recorder.current()
        before = rec.aborted_by if rec else None
        if rec is not None:
            rec.aborted_by = None
        pipe = self._first if not self.messages else self._next
        blocked_by = None
        try:
            with warnings.catch_warnings():  # InitQuery warns when a user turn is not the first message
                warnings.simplefilter("ignore", UserWarning)
                _, _, self.env, self.messages, _ = pipe.query(user_msg, self.runtime, self.env, self.messages)
        except AbortAgentError as e:
            self.env, self.messages = e.task_environment, list(e.messages)
            blocked_by = rec.aborted_by if rec else "guard"
        if rec is not None:
            if blocked_by is None:   # deny / redact / refuse also count as "the guard acted" for backtracking
                acted = [e for e in rec.layer_events if e.turn == rec.turn and e.acted]
                blocked_by = acted[0].layer if acted else None
                rec.aborted_by = before
            rec.turn += 1
        reply = _msg_text(self.messages[-1]) if self.messages and self.messages[-1]["role"] == "assistant" else ""
        return reply, blocked_by

    def rollback(self) -> None:
        if self._snapshots:
            self.messages, self.env = self._snapshots.pop()

    def transcript(self) -> list[dict]:
        return [{"role": m["role"], "content": _msg_text(m)} for m in self.messages if m["role"] in ("user", "assistant")]
