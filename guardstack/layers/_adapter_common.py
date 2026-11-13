"""Shared plumbing for the third-party guardrail adapters.

Nothing here imports a heavy library at module level: ``import guardstack.layers.*``
must work in an environment that has none of torch / transformers / llamafirewall /
nemoguardrails / presidio / gliner installed.

Conventions every adapter follows (so all layers are measured the same way):

* ``__init__`` validates parameters and calls :func:`guardstack.llm.assert_allowed` on every
  model id, but loads nothing; ``warmup()`` loads weights / opens clients; ``check()``
  lazily calls ``warmup()`` if the harness did not.
* ``check()`` never catches exceptions from the guard itself. A crash, an unparsable
  model answer or an unreachable endpoint propagates to ``GuardLayer.run``, which logs it
  as ``layer_error`` and fails open. Upstream libraries that fail *closed* or *open*
  silently are noted in docs/guardrails/adapter_notes.md.
* ``Verdict.score`` is "higher = more malicious", in [0, 1] when the guard exposes a
  probability; binary guards report 0.0 / 1.0.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import concurrent.futures
import contextlib
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..types import Stage
from .builtin import _msg_text

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GUARD_URL = "http://127.0.0.1:8001/v1"


# --------------------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------------------


def resolve_str(value: str | None) -> str | None:
    """Expand ``$VAR`` / ``${VAR}`` in a config value. Unresolved or empty -> None, so stack
    YAML can say ``base_url: "${GUARD_URL}"`` and fall back to defaults when unset."""
    if value is None:
        return None
    v = os.path.expandvars(str(value)).strip()
    if not v or "$" in v:
        return None
    return v


def guard_base_url(base_url: str | None) -> str:
    return resolve_str(base_url) or resolve_str(os.environ.get("GUARDSTACK_GUARD_URL")) or DEFAULT_GUARD_URL


def resolve_path(p: str | os.PathLike) -> Path:
    """Absolute path; relative paths are tried against CWD, then the repo root."""
    path = Path(os.path.expandvars(str(p))).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    alt = REPO_ROOT / path
    return alt.resolve() if alt.exists() else path.resolve()


def coerce_stage(stage: Stage | str | None, default: Stage) -> Stage:
    return Stage(stage) if stage is not None else default


def require_stage(layer_cls: str, stage: Stage, allowed: Iterable[Stage]) -> None:
    allowed = tuple(allowed)
    if stage not in allowed:
        raise ValueError(f"{layer_cls} does not support stage {stage.value!r}; allowed: "
                         f"{[s.value for s in allowed]}")


# --------------------------------------------------------------------------------------
# content extraction
# --------------------------------------------------------------------------------------


def tool_call_text(call) -> str:
    """Canonical text form of an AgentDojo FunctionCall: ``name({json args})``."""
    return f"{call.function}({json.dumps(dict(call.args), default=str, ensure_ascii=False)})"


def tool_call_string_args(call, arg_names: Sequence[str] | None = None) -> str:
    """Concatenated string-valued arguments (what a content scanner should look at when the
    call carries free text, e.g. an email body or file content)."""
    parts = []
    for k, v in dict(call.args).items():
        if arg_names is not None and k not in arg_names:
            continue
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (list, tuple)):
            parts.extend(x for x in v if isinstance(x, str))
    return "\n".join(parts)


def stage_text(ctx, *, action_mode: str = "call") -> str:
    """The text a content classifier inspects at ``ctx.stage``.

    action_mode="call": ``name({args})``; "args": string argument values only.
    """
    if ctx.stage == Stage.ACTION:
        if ctx.tool_call is None:
            return ""
        return tool_call_string_args(ctx.tool_call) if action_mode == "args" else tool_call_text(ctx.tool_call)
    return ctx.text or ""


def msg_text(m: Mapping) -> str:
    return _msg_text(m)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# OpenAI-compatible endpoints (vLLM inside the SLURM job)
# --------------------------------------------------------------------------------------


def make_guard_client(base_url: str | None, role: str):
    """UsageClient from guardstack.llm (records token usage under ``role``)."""
    from .. import llm as _llm

    return _llm.openai_client(base_url=guard_base_url(base_url), role=role)


class _TrackedCompletions:
    """Usage-recording wrapper for the legacy /v1/completions endpoint.

    guardstack.llm.UsageClient only wraps ``chat.completions``; raw-prompt guards
    (WildGuard, ShieldGemma via vLLM) need ``completions``. We reuse llm.py's own wrapper
    class so the accounting is identical. See the interface note in adapter_notes.md.
    """

    def __init__(self, client, role: str) -> None:
        from .. import llm as _llm

        inner = getattr(client, "_client", client)
        self._impl = _llm._UsageCompletions(inner.completions, client, role)

    def create(self, *args, **kwargs):
        return self._impl.create(*args, **kwargs)


def completions_api(client, role: str):
    c = getattr(client, "completions", None)
    # once llm.UsageClient wraps completions itself, use that
    if c is not None and c.__class__.__name__ == "_UsageCompletions":
        return c
    return _TrackedCompletions(client, role)


def usage_of(resp) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    return int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)


def check_served(client, model: str) -> None:
    """Fail fast (in warmup, before timing) if the endpoint does not serve ``model``."""
    ids = [m.id for m in client.models.list().data]
    if model not in ids:
        raise RuntimeError(f"endpoint {client.base_url} does not serve {model!r}; it serves {ids}")


def chat_logprob_positions(choice) -> list[tuple[str, list[tuple[str, float]]]]:
    """[(generated_token, [(alt_token, logprob), ...]), ...] from a chat completion choice."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    out = []
    for pos in content or []:
        alts = [(t.token, float(t.logprob)) for t in (getattr(pos, "top_logprobs", None) or [])]
        if not alts:
            alts = [(pos.token, float(pos.logprob))]
        out.append((pos.token, alts))
    return out


def completion_logprob_positions(choice) -> list[tuple[str, list[tuple[str, float]]]]:
    """Same shape from a legacy /v1/completions choice (``logprobs.top_logprobs`` dicts)."""
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return []
    tokens = list(getattr(lp, "tokens", None) or [])
    tops = list(getattr(lp, "top_logprobs", None) or [])
    tlp = list(getattr(lp, "token_logprobs", None) or [])
    out = []
    for i, tok in enumerate(tokens):
        top = tops[i] if i < len(tops) and tops[i] else {}
        alts = [(k, float(v)) for k, v in dict(top).items()]
        if not alts and i < len(tlp) and tlp[i] is not None:
            alts = [(tok, float(tlp[i]))]
        out.append((tok, alts))
    return out


def prob_of(alts: Iterable[tuple[str, float]], word: str) -> float:
    """Total probability mass of tokens whose stripped, lower-cased text equals ``word``."""
    w = word.lower()
    return sum(math.exp(lp) for t, lp in alts if t.strip().lower() == w)


def binary_prob(alts: Sequence[tuple[str, float]], pos_word: str, neg_word: str, floor: float = 1e-50) -> float | None:
    """P(pos) / (P(pos) + P(neg)) from the top-k alternatives at one position; None if
    neither word is among them."""
    p, n = prob_of(alts, pos_word), prob_of(alts, neg_word)
    if p == 0.0 and n == 0.0:
        return None
    p, n = p + floor, n + floor
    return p / (p + n)


def first_content_position(positions, skip=lambda t: not t.strip()):
    for i, (tok, alts) in enumerate(positions):
        if not skip(tok):
            return i
    return None


# --------------------------------------------------------------------------------------
# async + local models
# --------------------------------------------------------------------------------------


def run_coro(coro):
    """Run a coroutine from sync code; works whether or not this thread has a running loop
    (the harness calls layers from worker threads, but notebooks run inside a loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def _target():
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:  # re-raised in the caller's thread
            box["e"] = e

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


class LoopThread:
    """A private asyncio loop on a daemon thread. Libraries that keep loop-bound state
    (NeMo's async HTTP client, semaphores) stay on one loop no matter which harness worker
    thread calls the layer."""

    def __init__(self, name: str = "guardstack-loop") -> None:
        self.loop = asyncio.new_event_loop()
        self._t = threading.Thread(target=self.loop.run_forever, name=name, daemon=True)
        self._t.start()

    def run(self, coro, timeout: float | None = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()      # a hung call becomes a guard error instead of a stuck worker
            raise TimeoutError(f"no answer within {timeout}s") from None

    def close(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._t.join(timeout=5)


# -- one copy of each in-process model per process --------------------------------------------
# Runner threads each build their own layers. Loading a model per thread put 16 copies of every
# classifier on the guard GPU (CUDA out of memory in the first shadow_all run) and raced
# transformers' lazy imports ("cannot import name 'AutoModelForSequenceClassification'").
# Models are therefore loaded once, under one lock, and shared; callers hold the returned
# per-model lock around tokenize + forward (HF fast tokenizers are not safe for concurrent use).
_SHARED: dict = {}
_SHARED_LOCK = threading.Lock()
NOLOCK = contextlib.nullcontext()


def shared_load(owner, loader, *key):
    """-> (loader() result, per-model lock), built once per (layer class, loader, *key)."""
    k = (type(owner).__name__, getattr(loader, "__func__", loader), *key)
    with _SHARED_LOCK:
        if k not in _SHARED:
            _SHARED[k] = (loader(), threading.Lock())
        return _SHARED[k]


def torch_device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def hf_local_only() -> bool:
    """Respect HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE (compute nodes have no internet)."""
    return any(os.environ.get(k, "").lower() in ("1", "true", "yes") for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def parse_json_object(text: str) -> dict:
    """First JSON object in ``text`` (models sometimes wrap it in prose or ``` fences)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError(f"model output is not a JSON object: {text[:200]!r}")
    return obj
