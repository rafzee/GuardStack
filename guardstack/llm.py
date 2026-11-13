"""Model endpoints.

All models (agent, attacker, LLM-based guards, judges) are served with vLLM's
OpenAI-compatible server *inside the SLURM job* (the cluster does not allow long-lived services).
The cluster also bans models from DeepSeek, Alibaba/Qwen, Moonshot/Kimi, Z.ai/GLM,
ByteDance/Doubao and Xiaomi/MiMo -- :func:`assert_allowed` enforces that on every
model id we load.
"""

from __future__ import annotations

import os
import re
import time

import openai
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

from . import recorder

# Substrings that identify covered-entity models (the cluster's restricted-vendor policy). Case-insensitive.
# Covered entities (the cluster's restricted-vendor policy) and their known HF orgs / model families.
# Short tokens are anchored to separators to avoid false positives (glmnet, mimosa, ...).
_B = r"(?:^|[/_.\-\s])"
BANNED_PATTERNS = [
    r"qwen", r"qwq", r"deepseek", r"moonshot", r"zhipu", r"chatglm", r"alibaba", r"tongyi", r"bytedance",
    r"doubao", r"xiaomi", r"thudm", r"aidc-ai", r"wan-ai", r"zai-org", r"seed-oss",
    _B + r"kimi", _B + r"glm-?\d", _B + r"glm4", _B + r"mimo(?:$|[/_.\-\s])", _B + r"z-ai", _B + r"damo",
    _B + r"iic/",
]
# Known derivatives whose names hide the base model (Qwen/DeepSeek fine-tunes, Qwen-based embedders).
BANNED_DERIVATIVES = [
    "deepscaler", "openthinker", "sky-t1", "openreasoning-nemotron", "bespoke-stratos", "skywork-or1",
    "jina-embeddings-v4", "gte-qwen", "stella_en", "nvembed-qwen",
]
# config.json model_type / architectures of banned families
BANNED_ARCH = ["qwen", "deepseek", "chatglm", "glm4", "glm", "kimi", "mimo", "seed_oss", "minicpm-qwen"]


class BannedModelError(RuntimeError):
    pass


def assert_allowed(model_id: str, check_local_config: bool = True) -> None:
    """Refuse covered-entity models by id, by known derivative name, and (when the weights are
    on disk or in the HF cache) by config.json architecture and model-card base_model.
    Names alone cannot catch every fine-tune: when provenance is unclear, check the model card."""
    low = model_id.lower()
    for p in BANNED_PATTERNS:
        if re.search(p, low):
            raise BannedModelError(
                f"{model_id!r} matches banned pattern {p!r}: models from covered entities may not be "
                "installed or run on this cluster. Pick a Llama/Mistral/Gemma/Phi/OLMo/gpt-oss/Granite model."
            )
    for d in BANNED_DERIVATIVES:
        if d in low:
            raise BannedModelError(f"{model_id!r} is a known derivative of a covered-entity model ({d!r}).")
    if check_local_config:
        _check_config(model_id)


def _check_config(model_id: str) -> None:
    import json
    from pathlib import Path
    cands = []
    p = Path(model_id)
    if p.exists():
        cands.append(p)
    else:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        snap = home / ("models--" + model_id.replace("/", "--")) / "snapshots"
        if snap.exists():
            cands += sorted(snap.iterdir())
    for d in cands:
        cfg = d / "config.json"
        if cfg.exists():
            try:
                c = json.loads(cfg.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            fields = [str(c.get("model_type", ""))] + [str(a) for a in c.get("architectures", []) or []]
            for f in fields:
                if any(b in f.lower() for b in BANNED_ARCH):
                    raise BannedModelError(f"{model_id!r}: config.json architecture {f!r} is a covered-entity family.")
        card = d / "README.md"
        if card.exists():
            head = card.read_text(errors="ignore")[:4000].lower()
            m = re.search(r"base_model:\s*(.+?)(?:\n[a-z_]+:|\Z)", head, re.S)
            if m:
                for p_ in BANNED_PATTERNS:
                    if re.search(p_, m.group(1)):
                        raise BannedModelError(f"{model_id!r}: model card base_model {m.group(1).strip()[:80]!r} is banned.")
    if p.exists() and not cands[0].joinpath("config.json").exists():
        raise BannedModelError(f"local path {model_id!r} has no config.json; cannot verify provenance.")


class _UsageCompletions:
    def __init__(self, inner, sink, role: str) -> None:
        self._inner, self._sink, self._role = inner, sink, role

    def create(self, *args, **kwargs):
        t0 = time.perf_counter()
        resp = self._inner.create(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        u = getattr(resp, "usage", None)
        pt, ct = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
        self._sink._last_usage = (pt, ct)
        if self._role != "agent":  # agent calls are logged by InstrumentedLLM with its own timing
            rec = recorder.current()
            if rec is not None:
                rec.log_llm(role=self._role, latency_ms=dt, prompt_tokens=pt, completion_tokens=ct,
                            model=kwargs.get("model", ""))
        return resp


class _UsageChat:
    def __init__(self, inner, sink, role):
        self.completions = _UsageCompletions(inner.completions, sink, role)


class UsageClient:
    """Thin proxy over openai.OpenAI that records token usage of each chat completion."""

    def __init__(self, client: openai.OpenAI, role: str = "agent") -> None:
        self._client = client
        self._last_usage: tuple[int, int] | None = None
        self.chat = _UsageChat(client.chat, self, role)
        self.completions = _UsageCompletions(client.completions, self, role)  # raw-prompt guards (vLLM /v1/completions)

    def __getattr__(self, item):
        return getattr(self._client, item)


def openai_client(base_url: str | None = None, api_key: str | None = None, role: str = "agent") -> UsageClient:
    base_url = base_url or os.environ.get("GUARDSTACK_AGENT_URL", "http://127.0.0.1:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return UsageClient(openai.OpenAI(base_url=base_url, api_key=api_key, timeout=600, max_retries=3), role=role)


def split_parallel_tool_calls(msgs: list[dict]) -> list[dict]:
    """Rewrite an assistant turn with several tool calls as consecutive one-call turns, each followed
    by its tool result. The Llama 3.1/3.3 chat templates raise ("This model only supports single
    tool-calls at once!") when the history holds a multi-call turn, which killed 26/57 Llama-8B and
    1/57 Llama-70B smoke sessions; the calls themselves were executed in this order, so the rewrite
    only changes how the history is spelled for the template."""
    out: list[dict] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        calls = m.get("tool_calls") or []
        if m.get("role") != "assistant" or len(calls) < 2:
            out.append(m)
            i += 1
            continue
        j = i + 1
        results: dict[str, dict] = {}
        while j < len(msgs) and msgs[j].get("role") == "tool":
            results.setdefault(msgs[j].get("tool_call_id"), msgs[j])
            j += 1
        used = set()
        for k, tc in enumerate(calls):
            out.append({**m, "content": m.get("content") if k == 0 else None, "tool_calls": [tc]})
            r = results.get(tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None))
            if r is not None:
                out.append(r)
                used.add(id(r))
        out.extend(t for t in msgs[i + 1:j] if id(t) not in used)   # results we could not pair stay
        i = j
    return out


class TrackedOpenAILLM(OpenAILLM):
    """OpenAILLM whose client records usage; exposes ``_last_usage`` for InstrumentedLLM.

    query() is overridden because AgentDojo's request helper sends ``temperature or
    NOT_GIVEN``, which silently drops temperature=0.0 (vLLM then samples with the model's
    generation_config defaults), and sends the system prompt with role "developer", which
    open-model chat templates may not render. Here temperature and seed are always sent, and
    the system prompt goes out as role "system"."""

    def __init__(self, client: UsageClient, model: str, temperature: float | None = 0.0, seed: int | None = 0,
                 max_tokens: int | None = None, **kw) -> None:
        assert_allowed(model)
        super().__init__(client, model, temperature=temperature, **kw)  # type: ignore[arg-type]
        self.name = model
        self.seed = seed
        self.max_tokens = max_tokens

    def sampling_params(self) -> dict:
        return {"temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens}

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        from agentdojo.agent_pipeline.llms.openai_llm import (_function_to_openai, _message_to_openai,
                                                              _openai_to_assistant_message)
        from agentdojo.functions_runtime import EmptyEnv
        env = env if env is not None else EmptyEnv()
        msgs = []
        for m in messages:
            om = dict(_message_to_openai(m, self.model))
            if om.get("role") == "developer":
                om["role"] = "system"
            msgs.append(om)
        if "llama" in self.model.lower():
            msgs = split_parallel_tool_calls(msgs)
        tools = [_function_to_openai(t) for t in runtime.functions.values()]
        kw = {"model": self.model, "messages": msgs, "temperature": float(self.temperature or 0.0)}
        if tools:
            kw.update(tools=tools, tool_choice="auto")
        if self.seed is not None:
            kw["seed"] = self.seed
        if self.max_tokens:
            kw["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        completion = self.client.chat.completions.create(**kw)
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args if extra_args is not None else {}

    @property
    def _last_usage(self):
        return self.client._last_usage  # type: ignore[attr-defined]

    @_last_usage.setter
    def _last_usage(self, v):
        self.client._last_usage = v  # type: ignore[attr-defined]


def make_agent_llm(model: str, base_url: str | None = None, temperature: float = 0.0, seed: int | None = 0
                   ) -> TrackedOpenAILLM:
    return TrackedOpenAILLM(openai_client(base_url, role="agent"), model, temperature=temperature, seed=seed)
