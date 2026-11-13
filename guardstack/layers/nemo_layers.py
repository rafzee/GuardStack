"""NVIDIA NeMo Guardrails, one rail type at one stage per layer.

Verified against ``nemoguardrails==0.24.1`` (installed and exercised against a local fake
OpenAI-compatible server; see tests/test_adapters.py). How each rail type is run without
NeMo's main LLM generating anything:

* input / output / tool_input / tool_output: ``LLMRails.generate_async(messages, options)``
  with ``options.rails`` enabling only the chosen rail type (optionally only some of its
  ``flows``) and dialog rails off -- the mechanism ``LLMRails.check()`` itself uses
  (``rails/llm/llmrails.py``) -- plus ``log={"activated_rails", "llm_calls"}`` so we get
  the blocking rail and token counts (``check()`` returns neither). With dialog disabled
  the colang flows (``rails/llm/llm_flows.co``) echo the checked content instead of calling
  the main LLM. As a safety net every response's ``llm_calls`` is checked and the layer
  raises if a generation task (general / generate_user_intent / generate_next_steps /
  generate_bot_message) shows up.
  NeMo naming: *tool_output* rails run on tool **calls** before execution (our ACTION
  stage); *tool_input* rails run on tool **results** (our TOOL_OUTPUT stage).
  ``rail="execution"`` picks the one matching the layer's stage.
* retrieval: NeMo only runs retrieval rails inside its "generate bot message" flow, after
  ``retrieve_relevant_chunks``; there is no option to run them alone. We use the public
  event API: ``generate_events_async([ContextUpdate(relevant_chunks=<tool output>),
  UserMessage, BotIntent])`` with the ``generate_bot_message`` action replaced by a stub
  (``register_action``), so the retrieval flows run on the tool output and the main LLM is
  never called. A retrieval rail that blocks emits its own BotMessage; one that filters
  rewrites ``$relevant_chunks``. Token counts are not available on this path (0).
* dialog: not supported -- dialog rails need the main LLM (user-intent and next-step
  generation) by construction, so they cannot be measured as a stand-alone guard.

Any LLM a rail needs (self-check, content-safety model) is pointed at the local vLLM
server: an explicit ``base_url`` param is written into every model's ``parameters.base_url``;
otherwise ``$GUARDSTACK_GUARD_URL`` fills models whose config has no base_url (and replaces
the config's value when ``override_base_url=True``). Every model id goes through
``assert_allowed``.
"""

from __future__ import annotations

import threading

import json
import logging
from collections.abc import Sequence

from .. import llm as _llm
from ..types import Stage, Verdict
from . import _adapter_common as C
from .base import CheckContext, GuardLayer

LOG = logging.getLogger(__name__)

RAIL_TYPES = ("input", "output", "retrieval", "tool_input", "tool_output")
_RAIL_STAGES = {
    "input": (Stage.INPUT, Stage.TOOL_OUTPUT),
    "output": (Stage.OUTPUT, Stage.ACTION),
    "retrieval": (Stage.TOOL_OUTPUT,),
    "tool_input": (Stage.TOOL_OUTPUT,),
    "tool_output": (Stage.ACTION,),
}
_DEFAULT_STAGE = {"input": Stage.INPUT, "output": Stage.OUTPUT, "retrieval": Stage.TOOL_OUTPUT,
                  "tool_input": Stage.TOOL_OUTPUT, "tool_output": Stage.ACTION}
# NeMo LLM task names that mean "the main LLM is generating" (llm/types.py Task enum values)
GENERATION_TASKS = {"general", "generate_user_intent", "generate_next_steps", "generate_bot_message",
                    "generate_intent_steps_message", "generate_value", "generate_user_intent_from_user_action",
                    "generate_flow_continuation"}
NO_GENERATION = "__guardstack_no_generation__"
PLACEHOLDER_GOAL = "(user request not available)"


_NEMO_LOOP: "C.LoopThread | None" = None
_NEMO_LOOP_LOCK = threading.Lock()


def nemo_loop() -> "C.LoopThread":
    """The one event loop all NeMo layers of this process run on (whatever their class/config)."""
    global _NEMO_LOOP
    with _NEMO_LOOP_LOCK:
        if _NEMO_LOOP is None:
            _NEMO_LOOP = C.LoopThread(name="nemo")
        return _NEMO_LOOP


class NemoRailsLayer(GuardLayer):
    """params: rail in {input, output, retrieval, tool_input, tool_output, execution},
    config_path (a NeMo config dir, e.g. configs/nemo/self_check), flows (subset of that
    rail's configured flows; default all), base_url (vLLM endpoint for the rail's LLMs),
    models (model type -> served model id, overriding the config, e.g. ``{main: ${GUARD_MODEL}}``).

    block = a rail of this type stopped the conversation (NeMo's ``check()`` definition:
    an activated rail with ``stop=True``), or -- for retrieval, where library rails filter
    rather than stop -- the content was modified (``modified_as_block``, default True only
    for retrieval). Otherwise a modification is exposed through ``transform()``.
    score = 1.0 / 0.0 (NeMo rails are binary)."""

    family = "rails"
    default_stage = Stage.INPUT

    def __init__(self, rail: str = "input", config_path: str | None = None, flows: Sequence[str] | None = None,
                 base_url: str | None = None, override_base_url: bool = False,
                 modified_as_block: bool | None = None, check_served: bool = False,
                 models: dict | None = None, timeout_s: float = 300.0, **kw) -> None:
        stage = kw.get("stage")
        if rail == "dialog":
            raise NotImplementedError("NeMo dialog rails need the main LLM (user-intent / next-step generation); "
                                      "they cannot run as a stand-alone guard layer")
        if rail == "execution":
            st = C.coerce_stage(stage, Stage.ACTION)
            rail = "tool_output" if st == Stage.ACTION else "tool_input"
        if rail not in RAIL_TYPES:
            raise ValueError(f"rail must be one of {RAIL_TYPES + ('execution',)}, got {rail!r}")
        kw["stage"] = C.coerce_stage(stage, _DEFAULT_STAGE[rail])
        kw.setdefault("name", f"nemo_{rail}")
        super().__init__(**kw)
        C.require_stage(f"NemoRailsLayer(rail={rail})", self.stage, _RAIL_STAGES[rail])
        if not config_path:
            raise ValueError("NemoRailsLayer needs config_path (a NeMo Guardrails config directory)")
        self.rail = rail
        self.config_path = str(C.resolve_path(config_path))
        self.flows = list(flows) if flows else None
        self.base_url, self.override_base_url = base_url, override_base_url
        self.models = dict(models or {})     # model type -> model id override, e.g. {main: ${GUARD_MODEL}}
        self.modified_as_block = (rail == "retrieval") if modified_as_block is None else modified_as_block
        self.check_served = check_served
        self.timeout_s = timeout_s
        self._rails = None
        self._loop: C.LoopThread | None = None

    # ---------------------------------------------------------------- setup
    def _load_config(self):
        from nemoguardrails import RailsConfig

        cfg = RailsConfig.from_path(self.config_path)
        explicit = C.resolve_str(self.base_url)
        env = C.resolve_str("${GUARDSTACK_GUARD_URL}")
        for m in cfg.models:
            override = C.resolve_str(self.models.get(m.type)) if m.type in self.models else None
            if override:
                m.model = override
            if m.model:
                _llm.assert_allowed(m.model)
            params = m.parameters if m.parameters is not None else {}
            has = C.resolve_str(params.get("base_url"))
            if explicit:
                params["base_url"] = explicit
            elif not has:
                params["base_url"] = env or C.DEFAULT_GUARD_URL
            elif env and self.override_base_url:
                params["base_url"] = env
            params.setdefault("api_key", "EMPTY")
            m.parameters = params
        configured = list(getattr(cfg.rails, self.rail).flows)
        if not configured:
            raise ValueError(f"{self.config_path}: no {self.rail} rails configured")
        if self.flows:
            missing = [f for f in self.flows if f not in configured]
            if missing:
                raise ValueError(f"{self.config_path}: flows {missing} are not configured {self.rail} rails {configured}")
        return cfg

    def warmup(self) -> None:
        # One event loop for every NeMo layer in the process, and one LLMRails per rails config,
        # shared by all worker threads. With a loop per layer instance (one per thread), NeMo's
        # async HTTP client ended up used from several loops: "Stale event loop: <asyncio.locks.Event>
        # is bound to a different event loop" and "list.remove(x): x not in list" on ~100-150 calls
        # per static-phase job, and one job hung (Nov 8, jobs 502284/502285).
        if self._rails is not None:
            return
        self._loop = nemo_loop()
        key = (self.config_path, self.rail, tuple(self.flows or ()), tuple(sorted(self.models.items())),
               C.resolve_str(self.base_url), self.override_base_url, self.check_served)
        self._rails, _ = C.shared_load(self, self._build_rails, *key)

    def _build_rails(self):
        from nemoguardrails import LLMRails

        cfg = self._load_config()
        if self.check_served:
            for m in cfg.models:
                C.check_served(C.make_guard_client(m.parameters["base_url"], role=self.name), m.model)
        rails = LLMRails(cfg)
        if self.rail == "retrieval":
            from nemoguardrails.actions.actions import ActionResult
            from nemoguardrails.utils import new_event_dict

            async def _no_generation(**kwargs):
                return ActionResult(events=[new_event_dict("BotMessage", text=NO_GENERATION)])

            rails.register_action(_no_generation, name="generate_bot_message")
        return rails

    # ---------------------------------------------------------------- inputs
    def close(self) -> None:
        # the loop and rails are shared by every NeMo layer in the process: leave them running
        # (daemon thread); they end with the process
        self._loop = self._rails = None

    def _messages(self, ctx: CheckContext) -> tuple[list[dict], str]:
        """(NeMo messages, content under check)."""
        goal = ctx.user_goal or PLACEHOLDER_GOAL   # NeMo skips tool rails if there is no user turn
        if self.rail == "input":
            text = ctx.text or ""
            return [{"role": "user", "content": text}], text
        if self.rail == "output":
            text = C.tool_call_text(ctx.tool_call) if ctx.stage == Stage.ACTION and ctx.tool_call else (ctx.text or "")
            return [{"role": "user", "content": ctx.user_goal or ""}, {"role": "assistant", "content": text}], text
        if self.rail == "tool_output":
            call = ctx.tool_call
            if call is None:
                return [], ""
            tc = {"id": getattr(call, "id", None) or "call_0", "type": "function",
                  "function": {"name": call.function, "arguments": json.dumps(dict(call.args), default=str)}}
            return [{"role": "user", "content": goal}, {"role": "assistant", "content": "", "tool_calls": [tc]}], \
                C.tool_call_text(call)
        # tool_input
        text = ctx.text or ""
        name = ctx.tool_name or "tool"
        tc = {"id": "call_0", "type": "function", "function": {"name": name, "arguments": "{}"}}
        return [{"role": "user", "content": goal},
                {"role": "assistant", "content": "", "tool_calls": [tc]},
                {"role": "tool", "content": text, "name": name, "tool_call_id": "call_0"}], text

    def _options(self) -> dict:
        rails = {k: False for k in ("input", "output", "retrieval", "dialog", "tool_input", "tool_output")}
        rails[self.rail] = self.flows or True
        return {"rails": rails, "log": {"activated_rails": True, "llm_calls": True}}

    # ---------------------------------------------------------------- check
    def check(self, ctx: CheckContext) -> Verdict:
        self.warmup()
        if self.rail == "retrieval":
            return self._check_retrieval(ctx)
        msgs, original = self._messages(ctx)
        if not msgs:
            return Verdict(block=False, score=0.0, reason="nothing to check")
        res = self._loop.run(self._rails.generate_async(messages=msgs, options=self._options()), self.timeout_s)
        log = res.log
        calls = list(getattr(log, "llm_calls", None) or [])
        bad = sorted({c.task for c in calls if c.task in GENERATION_TASKS})
        if bad:
            raise RuntimeError(f"NeMo called its main LLM for generation ({bad}); this config is not a pure guard")
        empty = sorted({c.task for c in calls if not (c.completion or "").strip()})
        if empty:  # NeMo's parsers read an empty answer as "safe": fail loudly instead of open
            raise ValueError(f"NeMo judge returned an empty completion for {empty} (a reasoning model may have "
                             "spent max_tokens before answering); counted as a guard error, not a pass")
        tin = sum(int(c.prompt_tokens or 0) for c in calls)
        tout = sum(int(c.completion_tokens or 0) for c in calls)
        activated = [a for a in (getattr(log, "activated_rails", None) or []) if a.type == self.rail]
        blocking = next((a.name for a in activated if a.stop), None)
        content = ""
        if isinstance(res.response, list) and res.response:
            content = res.response[-1].get("content", "") or ""
        modified = blocking is None and self.rail in ("input", "output") and content != original
        if modified and not self.modified_as_block:
            ctx.extra[f"{self.name}:transformed"] = content
        block = blocking is not None or (modified and self.modified_as_block)
        reason = blocking or ("modified" if modified else None)
        return Verdict(block=block, score=1.0 if block else 0.0, tokens_in=tin, tokens_out=tout, reason=reason,
                       extra={"activated": [a.name for a in activated], "llm_tasks": [c.task for c in calls]})

    def _check_retrieval(self, ctx: CheckContext) -> Verdict:
        text = ctx.text or ""
        events = [{"type": "ContextUpdate", "data": {"relevant_chunks": text}},
                  {"type": "UserMessage", "text": ctx.user_goal or PLACEHOLDER_GOAL},
                  {"type": "BotIntent", "intent": "respond to user"}]
        out = self._loop.run(self._rails.generate_events_async(events), self.timeout_s)
        chunks = None
        bot_msgs = []
        for e in out:
            if e.get("type") == "ContextUpdate" and "relevant_chunks" in (e.get("data") or {}):
                chunks = e["data"]["relevant_chunks"]
            elif e.get("type") == "BotMessage":
                bot_msgs.append(e.get("text"))
        blocked = any(t != NO_GENERATION for t in bot_msgs)
        modified = chunks is not None and str(chunks).strip() != text.strip()
        if modified and not self.modified_as_block and not blocked:
            ctx.extra[f"{self.name}:transformed"] = str(chunks)
        block = blocked or (modified and self.modified_as_block)
        reason = "blocked" if blocked else ("chunks_modified" if modified else None)
        return Verdict(block=block, score=1.0 if block else 0.0, reason=reason)

    def transform(self, ctx: CheckContext) -> str | None:
        return ctx.extra.get(f"{self.name}:transformed")
