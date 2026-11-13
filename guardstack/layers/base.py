"""The guardrail-layer interface.

Every guardrail (LlamaFirewall scanner, NeMo rail, Llama Guard, Presidio, a regex, a
tool allowlist ...) is wrapped as a :class:`GuardLayer`. A layer only has to implement
``check(ctx) -> Verdict``. Timing, logging, shadow/enforce semantics and the action taken
on a block are handled here and in :mod:`guardstack.pipeline`, so every layer is measured
the same way.

Adapter authors: subclass GuardLayer, set ``family`` (used to group same-type layers in
the redundancy analysis) and ``default_stage``, implement ``check``. Count the tokens
your model consumed in ``Verdict.tokens_in`` / ``tokens_out`` if you can.
"""

from __future__ import annotations

import abc
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from agentdojo.functions_runtime import FunctionCall

from .. import recorder
from ..types import Mode, OnBlock, Stage, Verdict


@dataclass
class CheckContext:
    """What a layer gets to look at. Which fields are set depends on the stage."""

    stage: Stage
    text: str | None = None                    # input / tool_output / output text
    tool_call: FunctionCall | None = None      # action stage
    tool_name: str | None = None               # tool_output stage: which tool produced the text
    messages: list[dict] = field(default_factory=list)  # conversation so far (agentdojo messages)
    user_goal: str | None = None               # first user message, for alignment-style checks
    extra: dict[str, Any] = field(default_factory=dict)


class GuardLayer(abc.ABC):
    family: str = "other"            # e.g. "pi_classifier", "safety_classifier", "llm_judge", "rule", "policy", "pii", "transform"
    default_stage: Stage = Stage.INPUT
    default_on_block: OnBlock | None = None

    def __init__(
        self,
        name: str | None = None,
        stage: Stage | str | None = None,
        mode: Mode | str = Mode.ENFORCE,
        on_block: OnBlock | str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.stage = Stage(stage) if stage is not None else self.default_stage
        base = name or self.__class__.__name__.lower()
        self.name = base if "@" in base else f"{base}@{self.stage.value}"
        self.mode = Mode(mode)
        self.threshold = threshold
        ob = on_block or self.default_on_block or _default_on_block(self.stage)
        self.on_block = OnBlock(ob)

    # ---- to implement ----
    @abc.abstractmethod
    def check(self, ctx: CheckContext) -> Verdict: ...

    def transform(self, ctx: CheckContext) -> str | None:
        """Optional: rewrite content even when not blocking (spotlighting, PII masking).
        Return the new text or None for no change. Only applied in ENFORCE mode."""
        return None

    def warmup(self) -> None:
        """Load weights / open clients before timing starts."""

    def reset(self) -> None:
        """Clear per-session state (called at the start of every session)."""

    def close(self) -> None:
        """Release resources (background threads, clients). Called once at the end of a run."""

    # ---- harness side ----
    def run(self, ctx: CheckContext) -> Verdict:
        rec = recorder.current()
        t0 = time.perf_counter()
        err = None
        try:
            v = self.check(ctx)
        except Exception as e:  # a crashing guard must not crash the run; it is logged and fails open
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
            v = Verdict(block=False, reason="layer_error")
        dt = (time.perf_counter() - t0) * 1000.0
        if rec is not None:
            rec.log_layer(
                layer=self.name,
                stage=self.stage.value,
                mode=self.mode.value,
                block=bool(v.block),
                acted=bool(v.block) and self.mode == Mode.ENFORCE,
                score=v.score,
                latency_ms=dt,
                tokens_in=v.tokens_in,
                tokens_out=v.tokens_out,
                chars_in=len(ctx.text or "") + len(str(ctx.tool_call.args) if ctx.tool_call else ""),
                reason=v.reason,
                error=err,
                payload_hit=_payload_hit(rec, ctx),
            )
        return v

    @property
    def enforcing(self) -> bool:
        return self.mode == Mode.ENFORCE

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name} {self.mode.value} on_block={self.on_block.value}>"


def _payload_hit(rec, ctx: CheckContext) -> bool:
    """Did this check look at attacker material? Input/tool-output: the inspected text. Action:
    the proposed call carries an attacker entity or is an injection-only function. Trace/output:
    either of those, or the context the layer judges already contains attacker text (the
    trajectory is contaminated). Detection metrics only credit fires with payload_hit=True,
    so a layer that fires on AgentDojo's (benign) user prompt is not counted as detecting an
    injection that arrives later in a tool result."""
    if ctx.stage in (Stage.INPUT, Stage.TOOL_OUTPUT):
        return rec.is_attacker_text(ctx.text)
    if ctx.stage == Stage.ACTION:
        return rec.is_attacker_call(ctx.tool_call)
    if rec.is_attacker_call(ctx.tool_call) or rec.is_attacker_text(ctx.text):
        return True
    from .builtin import _msg_text
    return any(m.get("role") == "tool" and rec.is_attacker_text(m.get("_raw_text") or _msg_text(m))
               for m in ctx.messages)


def _default_on_block(stage: Stage) -> OnBlock:
    return {
        Stage.INPUT: OnBlock.ABORT,
        Stage.TOOL_OUTPUT: OnBlock.REDACT,
        Stage.ACTION: OnBlock.DENY,
        Stage.TRACE: OnBlock.ABORT,
        Stage.OUTPUT: OnBlock.REFUSE,
    }[stage]
