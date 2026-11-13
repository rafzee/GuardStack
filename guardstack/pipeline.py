"""Build an AgentDojo pipeline with guardrail layers inserted at their stages.

Resulting element order (one user turn):

    SystemMessage            (first turn only; transform layers append their notes)
    InitQuery                (adds the user turn)
    InputGuard               <- Stage.INPUT layers, in stack order
    LLM (instrumented)
    ToolsExecutionLoop:
        GuardedToolsExecutor <- Stage.TRACE then Stage.ACTION layers, per proposed call
        ToolOutputGuard      <- Stage.TOOL_OUTPUT layers, per tool result
        LLM (instrumented)
    OutputGuard              <- Stage.OUTPUT layers on the final answer

Within a stage, layers run in the order given in the stack; that is the only ordering
freedom the agent loop allows, and it is what the ordering ablation permutes.
"""

from __future__ import annotations

import time
from ast import literal_eval
from collections.abc import Sequence

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from . import recorder
from .layers.base import CheckContext, GuardLayer
from .layers.builtin import Spotlight, _msg_text
from .types import OnBlock, Stage

REFUSAL = "I can't continue with this request because a safety check blocked it."
REDACTED = "<Content removed by a guardrail>"
DENIED = "This tool call was blocked by a security policy."


def _user_goal(messages: Sequence[ChatMessage]) -> str | None:
    for m in messages:
        if m["role"] == "user":
            return _msg_text(m)
    return None


def _abort(layer: GuardLayer, messages: Sequence[ChatMessage], env: Env) -> None:
    rec = recorder.current()
    if rec is not None and rec.aborted_by is None:
        rec.aborted_by = layer.name
    raise AbortAgentError(REFUSAL, list(messages), env)


class _StageGuard(BasePipelineElement):
    def __init__(self, layers: Sequence[GuardLayer]) -> None:
        self.layers = list(layers)


class InputGuard(_StageGuard):
    name = "input_guard"

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if not self.layers or not messages or messages[-1]["role"] != "user":
            return query, runtime, env, messages, extra_args
        msg = messages[-1]
        for layer in self.layers:
            ctx = CheckContext(stage=Stage.INPUT, text=_msg_text(msg), messages=list(messages),
                               user_goal=_user_goal(messages))
            v = layer.run(ctx)
            if layer.enforcing:
                new = layer.transform(ctx)
                if new is not None:
                    msg["content"] = [text_content_block_from_string(new)]
                if v.block:
                    _abort(layer, messages, env)  # input blocks always end the turn
        return query, runtime, env, messages, extra_args


class GuardedToolsExecutor(BasePipelineElement):
    """AgentDojo's ToolsExecutor plus per-call TRACE and ACTION checks."""

    name = "guarded_tools_executor"

    def __init__(self, layers: Sequence[GuardLayer], tool_output_formatter=tool_result_to_str) -> None:
        # TRACE (alignment-style) before ACTION (policy) -- both see the proposed call
        self.layers = sorted(layers, key=lambda l: l.stage.order)
        self.output_formatter = tool_output_formatter

    def query(self, query, runtime: FunctionsRuntime, env: Env = EmptyEnv(), messages: Sequence[ChatMessage] = [],
              extra_args: dict = {}):
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args
        results = []
        tool_names = {t.name for t in runtime.functions.values()}
        for call in messages[-1]["tool_calls"]:
            denied_by = None
            for layer in self.layers:
                ctx = CheckContext(stage=layer.stage, tool_call=call, messages=list(messages),
                                   user_goal=_user_goal(messages))
                v = layer.run(ctx)
                if v.block and layer.enforcing:
                    if layer.on_block == OnBlock.ABORT:
                        _abort(layer, [*messages, *results], env)
                    denied_by = layer.name
                    break
            rec = recorder.current()
            if denied_by is not None:
                if rec is not None:
                    rec.log_tool(call.function, call.id, denied=True, harmful=rec.is_attacker_call(call), call=call)
                results.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string("")],
                                                     tool_call_id=call.id, tool_call=call, error=DENIED))
                continue
            if call.function not in tool_names:
                results.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string("")],
                                                     tool_call_id=call.id, tool_call=call,
                                                     error=f"Invalid tool {call.function} provided."))
                continue
            for k, val in call.args.items():
                if isinstance(val, str) and is_string_list(val):
                    call.args[k] = literal_eval(val)
            out, error = runtime.run_function(env, call.function, call.args)
            if rec is not None:
                rec.log_tool(call.function, call.id, denied=False, harmful=rec.is_attacker_call(call), call=call)
            results.append(ChatToolResultMessage(role="tool",
                                                 content=[text_content_block_from_string(self.output_formatter(out))],
                                                 tool_call_id=call.id, tool_call=call, error=error))
        return query, runtime, env, [*messages, *results], extra_args


class ToolOutputGuard(_StageGuard):
    name = "tool_output_guard"

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if not self.layers or not messages or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args
        # the trailing block of tool messages produced by this step
        i = len(messages) - 1
        while i >= 0 and messages[i]["role"] == "tool":
            i -= 1
        for msg in messages[i + 1:]:
            if msg.get("error") == DENIED:
                continue
            # keep the untransformed text: provenance-style layers and attribution must see the
            # data as the tool returned it, not after spotlighting / redaction
            msg.setdefault("_raw_text", _msg_text(msg))
            for layer in self.layers:
                ctx = CheckContext(stage=Stage.TOOL_OUTPUT, text=_msg_text(msg), messages=list(messages),
                                   tool_name=msg["tool_call"].function if msg.get("tool_call") else None,
                                   user_goal=_user_goal(messages))
                v = layer.run(ctx)
                if not layer.enforcing:
                    continue
                if v.block:
                    if layer.on_block == OnBlock.ABORT:
                        _abort(layer, messages, env)
                    msg["content"] = [text_content_block_from_string(REDACTED)]
                    break  # content is gone; later layers would only see the placeholder
                new = layer.transform(ctx)
                if new is not None:
                    msg["content"] = [text_content_block_from_string(new)]
        return query, runtime, env, messages, extra_args


class OutputGuard(_StageGuard):
    name = "output_guard"

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if not self.layers or not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        msg = messages[-1]
        for layer in self.layers:
            ctx = CheckContext(stage=Stage.OUTPUT, text=_msg_text(msg), messages=list(messages),
                               user_goal=_user_goal(messages))
            v = layer.run(ctx)
            if not layer.enforcing:
                continue
            if v.block:
                if layer.on_block == OnBlock.ABORT:
                    _abort(layer, messages[:-1], env)
                msg["content"] = [text_content_block_from_string(REFUSAL)]
                break
            new = layer.transform(ctx)
            if new is not None:
                msg["content"] = [text_content_block_from_string(new)]
        return query, runtime, env, messages, extra_args


class InstrumentedLLM(BasePipelineElement):
    """Times every model call. Token usage is recorded by the OpenAI client wrapper in
    guardstack.llm (vLLM returns usage), so this element only adds wall time."""

    def __init__(self, llm: BasePipelineElement, role: str = "agent", model: str = "") -> None:
        self.llm = llm
        self.role = role
        self.model = model or getattr(llm, "model", "") or ""
        self.name = getattr(llm, "name", None) or self.model

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        t0 = time.perf_counter()
        out = self.llm.query(query, runtime, env, messages, extra_args)
        dt = (time.perf_counter() - t0) * 1000.0
        rec = recorder.current()
        if rec is not None:
            usage = extra_args.pop("_last_usage", None) or getattr(self.llm, "_last_usage", None) or (0, 0)
            rec.log_llm(role=self.role, latency_ms=dt, prompt_tokens=usage[0], completion_tokens=usage[1],
                        model=self.model)
            if hasattr(self.llm, "_last_usage"):
                self.llm._last_usage = None
        return out


def build_pipeline(
    llm: BasePipelineElement,
    system_message: str,
    layers: Sequence[GuardLayer],
    *,
    first_turn: bool = True,
    max_iters: int = 15,
    tool_output_formatter=tool_result_to_str,
    name: str = "guarded",
    model_name: str = "",
) -> AgentPipeline:
    by_stage: dict[Stage, list[GuardLayer]] = {s: [] for s in Stage}
    for layer in layers:
        by_stage[layer.stage].append(layer)

    notes = [l.system_note for l in layers if isinstance(l, Spotlight) and l.enforcing]
    sys_msg = system_message + ("\n\n" + "\n".join(notes) if notes else "")
    agent = InstrumentedLLM(llm, role="agent", model=model_name)

    elements: list[BasePipelineElement] = []
    if first_turn:
        elements.append(SystemMessage(sys_msg))
    elements += [
        InitQuery(),
        InputGuard(by_stage[Stage.INPUT]),
        agent,
        ToolsExecutionLoop(
            [
                GuardedToolsExecutor(by_stage[Stage.TRACE] + by_stage[Stage.ACTION], tool_output_formatter),
                ToolOutputGuard(by_stage[Stage.TOOL_OUTPUT]),
                agent,
            ],
            max_iters=max_iters,
        ),
        OutputGuard(by_stage[Stage.OUTPUT]),
    ]
    p = AgentPipeline(elements)
    p.name = name
    return p
