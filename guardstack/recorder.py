"""Per-session event recorder.

Layers and LLM wrappers call :func:`current()` to log events, so no pipeline element
needs a reference to the runner. One recorder is active per session (context variable),
which also keeps concurrent sessions in threads separate.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from collections import defaultdict
from collections.abc import Iterator

import json
import re

from .types import LayerEvent, LLMEvent, ToolEvent

_current: contextvars.ContextVar["SessionRecorder | None"] = contextvars.ContextVar("guardstack_recorder", default=None)


class SessionRecorder:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.layer_events: list[LayerEvent] = []
        self.llm_events: list[LLMEvent] = []
        self.turn = 0
        self._seq = 0
        self.aborted_by: str | None = None
        self.t0 = time.perf_counter()
        self.tool_events: list[ToolEvent] = []
        self.denied_ids: set[str] = set()
        self.n_pipeline_runs = 0
        # attacker signature, set by the runner before the session runs (empty for benign cases)
        self.payload_snippets: list[str] = []
        self.final_output: str | None = None
        self.attacker_entities: set[str] = set()
        self.injection_only_functions: set[str] = set()

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def log_layer(self, **kw) -> LayerEvent:
        ev = LayerEvent(session_id=self.session_id, seq=self.next_seq(), turn=self.turn, **kw)
        self.layer_events.append(ev)
        return ev

    def set_attack_signature(self, payloads, user_prompt: str, injection_only_functions=()) -> None:
        from .layers.builtin import _entities
        snippets = []
        for p in payloads:
            t = re.sub(r"\s+", " ", p or "").strip()
            # several short windows so redaction of one part or light reformatting still matches
            for i in range(0, max(1, len(t) - 40), 80):
                w = t[i:i + 40]
                if len(w) >= 20:
                    snippets.append(w.lower())
        self.payload_snippets = snippets[:50]
        user_ents = _entities(user_prompt or "")
        self.attacker_entities = {e for p in payloads for e in _entities(p or "")} - user_ents
        self.injection_only_functions = set(injection_only_functions)

    def is_attacker_text(self, text: str | None) -> bool:
        if not text or not (self.payload_snippets or self.attacker_entities):
            return False
        t = re.sub(r"\s+", " ", text).lower()
        if any(sn in t for sn in self.payload_snippets):
            return True
        from .layers.builtin import _entities
        return bool(_entities(text) & self.attacker_entities)

    def is_attacker_call(self, call) -> bool:
        if call is None:
            return False
        if call.function in self.injection_only_functions:
            return True
        try:
            args = json.dumps(call.args, default=str)
        except TypeError:
            args = str(call.args)
        from .layers.builtin import _entities
        return bool(_entities(args) & self.attacker_entities)

    def log_tool(self, function: str, call_id, denied: bool, harmful: bool, call=None) -> ToolEvent:
        ev = ToolEvent(session_id=self.session_id, seq=self.next_seq(), turn=self.turn, function=function,
                       call_id=call_id, denied=denied, harmful=harmful)
        self.tool_events.append(ev)
        if denied and call is not None:
            self.denied_ids.add(id(call))   # identity: local models / scripted agents may give no call id
        return ev

    def harm_seq(self) -> int | None:
        for e in self.tool_events:
            if e.harmful and not e.denied:
                return e.seq
        return None

    def attributed_fire_seq(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.layer_events:
            if e.block and e.payload_hit and e.layer not in out:
                out[e.layer] = e.seq
        return out

    def guard_errors(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in self.layer_events:
            if e.error:
                out[e.layer] += 1
        return dict(out)

    def log_llm(self, **kw) -> LLMEvent:
        ev = LLMEvent(session_id=self.session_id, seq=self.next_seq(), turn=self.turn, **kw)
        self.llm_events.append(ev)
        return ev

    # ---- summaries used to build the SessionRecord ----
    def fired(self) -> dict[str, bool]:
        out: dict[str, bool] = defaultdict(bool)
        for e in self.layer_events:
            out[e.layer] = out[e.layer] or e.block
        return dict(out)

    def first_fire_seq(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.layer_events:
            if e.block and e.layer not in out:
                out[e.layer] = e.seq
        return out

    def max_score(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.layer_events:
            if e.score is not None:
                out[e.layer] = max(out.get(e.layer, float("-inf")), float(e.score))
        return out

    def guard_latency(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for e in self.layer_events:
            out[e.layer] += e.latency_ms
        return dict(out)

    def guard_calls(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in self.layer_events:
            out[e.layer] += 1
        return dict(out)

    def guard_tokens(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in self.layer_events:
            out[e.layer] += e.tokens_in + e.tokens_out
        return dict(out)

    def agent_llm(self) -> tuple[int, float, int, int]:
        ev = [e for e in self.llm_events if e.role == "agent"]
        return (
            len(ev),
            sum(e.latency_ms for e in ev),
            sum(e.prompt_tokens for e in ev),
            sum(e.completion_tokens for e in ev),
        )

    def wall_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0


def current() -> SessionRecorder | None:
    return _current.get()


@contextlib.contextmanager
def session(session_id: str) -> Iterator[SessionRecorder]:
    rec = SessionRecorder(session_id)
    token = _current.set(rec)
    try:
        yield rec
    finally:
        _current.reset(token)
