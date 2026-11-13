"""Core data types shared by the harness and the analysis code.

Vocabulary used throughout the code and the paper:

* layer      -- one guardrail deployed at one stage, e.g. ``promptguard2@tool_output``.
                The same model at two stages is two layers.
* stage      -- where in the agent loop a layer looks: the user input, a tool output,
                a proposed tool call (action), the full trace, or the final answer.
* stack      -- an ordered list of layers. Stages impose a partial order in time
                (input -> [action -> tool_output]* -> output); only layers that share a
                stage can be reordered freely.
* case       -- one evaluation unit, e.g. (suite, user_task, injection_task, attack).
* session    -- one execution of one case against one stack.
* fired      -- the layer's verdict was "block" for at least one event in the session.
* enforce /  -- an enforcing layer acts on its verdict; a shadow layer only records it.
  shadow        Running every layer in shadow on the bare agent gives the full
                layer x case verdict matrix from a single run.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from typing import Any


class Stage(str, enum.Enum):
    INPUT = "input"            # user turn, before the model sees it
    ACTION = "action"          # a proposed tool call, before it executes
    TOOL_OUTPUT = "tool_output"  # tool result, before the model sees it
    TRACE = "trace"            # whole conversation, checked at each action (AlignmentCheck-style)
    OUTPUT = "output"          # final assistant message, before the user sees it

    @property
    def order(self) -> int:
        return _STAGE_ORDER[self]


_STAGE_ORDER = {
    Stage.INPUT: 0,
    Stage.TRACE: 1,
    Stage.ACTION: 2,
    Stage.TOOL_OUTPUT: 3,
    Stage.OUTPUT: 4,
}


class OnBlock(str, enum.Enum):
    """What an enforcing layer does when it fires."""

    ABORT = "abort"      # end the session with a refusal
    REDACT = "redact"    # replace the inspected content, continue
    DENY = "deny"        # (action stage) refuse this tool call, continue
    REFUSE = "refuse"    # (output stage) replace final answer with a refusal


class Mode(str, enum.Enum):
    ENFORCE = "enforce"
    SHADOW = "shadow"


@dataclass
class Verdict:
    block: bool
    score: float | None = None          # detector confidence, if any (higher = more malicious)
    reason: str | None = None
    tokens_in: int = 0                  # tokens the layer itself consumed (LLM/classifier input)
    tokens_out: int = 0                 # tokens the layer generated (LLM judges)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerEvent:
    session_id: str
    layer: str
    stage: str
    mode: str
    seq: int                 # global event index within the session (time order)
    turn: int                # user turn index (0 for single-turn AgentDojo tasks)
    block: bool
    acted: bool              # True if an enforcing layer actually changed the trajectory
    score: float | None
    latency_ms: float
    tokens_in: int
    tokens_out: int
    chars_in: int
    reason: str | None = None
    error: str | None = None
    payload_hit: bool = False   # inspected content carried attacker material (payload text, attacker entity, injected call)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ToolEvent:
    session_id: str
    seq: int
    turn: int
    function: str
    call_id: str | None
    denied: bool
    harmful: bool            # matches the injection's attacker signature (entity or injection-only function)


@dataclass
class LLMEvent:
    session_id: str
    role: str                # "agent", "attacker", "judge", or a layer name for LLM-based layers
    seq: int
    turn: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    model: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class SessionRecord:
    """One row per (case, stack). Written as JSONL by the runner."""

    session_id: str
    run_id: str
    stack_id: str
    stack_layers: list[str]          # in configured order
    layer_modes: dict[str, str]      # layer -> enforce | shadow
    suite: str
    user_task: str
    injection_task: str | None
    attack: str | None
    category: str                    # attack category (one-pager taxonomy) or "benign"
    is_attack: bool
    utility: bool | None             # AgentDojo user-task success
    attack_success: bool | None      # AgentDojo "security" flag: True = injection goal achieved
    aborted_by: str | None           # first enforcing layer that ended the session, if any
    fired: dict[str, bool]           # layer -> fired at least once (shadow or enforce)
    first_fire_seq: dict[str, int]   # layer -> seq of first block verdict
    max_score: dict[str, float]      # layer -> max detector score in the session (offline threshold sweeps)
    n_turns: int
    n_llm_calls: int
    wall_ms: float
    agent_latency_ms: float
    guard_latency_ms: dict[str, float]   # layer -> total ms in this session
    guard_calls: dict[str, int]
    agent_prompt_tokens: int
    agent_completion_tokens: int
    guard_tokens: dict[str, int]
    config: dict[str, Any]           # pinned configuration tuple (models, seeds, budgets, versions)
    error: str | None = None
    # --- timing of harm vs. verdicts (a "catch" must precede the harm; see analysis/structure.py)
    harm_seq: int | None = None              # seq of first executed attacker-signature tool call (None: no such call)
    attributed_fire_seq: dict[str, int] = field(default_factory=dict)  # first block on attacker content, per layer
    guard_errors: dict[str, int] = field(default_factory=dict)         # layer -> number of failed checks (failed open)
    skipped_reason: str | None = None        # AgentDojo-style scoring of context overflow / server errors
    n_retries: int = 0                       # AgentDojo re-runs a task up to 3x when the final output is None
    denied_calls: int = 0
    final_output: str | None = None          # agent's final answer, first 1000 chars (attacker feedback, debugging)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)
