"""Run AgentDojo cases against a guardrail stack and write one SessionRecord per case.

Two run modes, because they need different conditions:

* ``throughput`` (default): many concurrent sessions against the vLLM server. Use for
  ASR / utility / verdict matrices. Latencies recorded here are contaminated by
  contention and must not be reported.
* ``latency``: one session at a time on a dedicated GPU, warm caches, fixed hardware. Use
  on a stratified subsample to report p50/p95 per layer and per stack.

Resumable: cases already present in the output JSONL are skipped. Shardable for SLURM
job arrays with --shard/--num-shards.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suites

import agentdojo.attacks  # noqa: F401  (registers the built-in attacks)

import openai

from . import recorder
from .pipeline import build_pipeline
from .scoring import run_task_scored
from .stacks import StackConfig
from .types import SessionRecord

log = logging.getLogger("guardstack.runner")

OBJECTIVES_PATH = Path(__file__).resolve().parent.parent / "configs" / "agentdojo_objectives.yaml"


@dataclass(frozen=True)
class Case:
    suite: str
    user_task: str
    injection_task: str | None
    attack: str | None
    vector: str          # direct | multiturn | indirect | none
    objective: str       # harmful_content | unsafe_action | exfiltration | insecure_code | benign

    @property
    def case_id(self) -> str:
        return f"{self.suite}/{self.user_task}/{self.injection_task or '-'}/{self.attack or '-'}"

    @property
    def category(self) -> str:
        return "benign" if self.objective == "benign" else f"{self.vector}:{self.objective}"


def enumerate_agentdojo_cases(
    benchmark_version: str = "v1.2.2",
    suites: list[str] | None = None,
    attack: str | None = "important_instructions",
    include_benign: bool = True,
    include_attacks: bool = True,
) -> list[Case]:
    objectives = yaml.safe_load(OBJECTIVES_PATH.read_text())
    out: list[Case] = []
    for sname, suite in get_suites(benchmark_version).items():
        if suites and sname not in suites:
            continue
        for ut in suite.user_tasks:
            if include_benign:
                out.append(Case(sname, ut, None, None, "none", "benign"))
            if include_attacks and attack:
                for it in suite.injection_tasks:
                    out.append(Case(sname, ut, it, attack, "indirect", objectives[sname][it]))
    return out


def shard(cases: list[Case], i: int, n: int) -> list[Case]:
    # stable hash sharding so shards don't change when the case list is filtered
    return [c for c in cases if int(hashlib.md5(c.case_id.encode()).hexdigest(), 16) % n == i]


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("error") is None:
                done.add(r["config"]["case_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


class AgentDojoRunner:
    def __init__(
        self,
        stack: StackConfig,
        llm_factory,                    # () -> pipeline LLM element (one per worker thread)
        system_message: str,
        model_name: str,
        benchmark_version: str = "v1.2.2",
        run_config: dict | None = None,
        max_iters: int = 15,
        strict_guards: bool = True,
        payloads: dict[str, str] | None = None,
    ) -> None:
        self.stack = stack
        self.llm_factory = llm_factory
        self.system_message = system_message
        self.model_name = model_name
        self.version = benchmark_version
        self.suites = get_suites(benchmark_version)
        self.run_config = run_config or {}
        self.max_iters = max_iters
        # case_id -> fixed payload that replaces the static attack text for that case (replay of an
        # adaptive attacker's best-shot payloads, e.g. against the shadow-all stack)
        self.payloads = payloads or {}
        self.strict_guards = strict_guards   # a guard error marks the session errored (retried on resume)
        self.run_id = self.run_config.get("run_id") or uuid.uuid4().hex[:10]
        self._local = threading.local()
        self._all_layers: list = []
        self._layers_lock = threading.Lock()

    def _pipeline(self):
        # Layers and LLM clients are built per thread: HF pipelines and HTTP clients are not
        # guaranteed thread-safe, and per-thread objects keep latency attribution clean.
        if not hasattr(self._local, "pipeline"):
            layers = self.stack.build()
            for l in layers:
                l.warmup()
            with self._layers_lock:
                self._all_layers.extend(layers)
            self._local.layers = layers
            p = build_pipeline(self.llm_factory(), self.system_message, layers, max_iters=self.max_iters,
                               name=f"local|{self.model_name}|{self.stack.id}", model_name=self.model_name)
            self._local.pipeline = p
        return self._local.pipeline, self._local.layers

    def run_case(self, case: Case) -> SessionRecord:
        rec_obj, events = self.run_case_with_events(case)
        return rec_obj

    def static_injections(self, case: Case) -> dict[str, str]:
        suite = self.suites[case.suite]
        pipeline, _ = self._pipeline()
        if case.injection_task is None or not case.attack:
            return {}
        attack = load_attack(case.attack, suite, pipeline)
        return attack.attack(suite.get_user_task_by_id(case.user_task),
                             suite.get_injection_task_by_id(case.injection_task))

    def run_case_with_events(self, case: Case, injections: dict[str, str] | None = None,
                             attempt: int | None = None) -> tuple[SessionRecord, list[str]]:
        """Run one case. ``injections`` overrides the static attack (adaptive attackers pass
        their candidate payloads here; ``attempt`` is recorded in the config)."""
        suite = self.suites[case.suite]
        pipeline, layers = self._pipeline()
        user_task = suite.get_user_task_by_id(case.user_task)
        injection_task = suite.get_injection_task_by_id(case.injection_task) if case.injection_task else None
        if injections is None:
            injections = self.static_injections(case)
            if case.case_id in self.payloads and injections:
                injections = {v: self.payloads[case.case_id] for v in injections}

        sid = f"{self.run_id}:{self.stack.id}:{case.case_id}" + (f":a{attempt}" if attempt is not None else "")
        err = None
        skipped = None
        utility = security = None
        n_retries = 0
        for l in layers:
            l.reset()
        with recorder.session(sid) as rec:
            try:
                utility, security, n_retries = run_task_scored(suite, pipeline, user_task, injection_task, injections)
            except openai.BadRequestError as e:
                # AgentDojo (benchmark.py) scores context overflow as utility=False, security=True
                if "context_length_exceeded" in repr(e) or "maximum context length" in str(e) \
                        or "reduce the length of the messages" in repr(e) or getattr(e, "param", None) == "max_tokens":
                    utility, security, skipped = False, True, "context_length"
                else:
                    err = f"{type(e).__name__}: {e}"
            except (openai.InternalServerError, openai.APIStatusError) as e:
                if getattr(e, "status_code", 0) and e.status_code >= 500:
                    utility, security, skipped = False, True, "server_error"
                else:
                    err = f"{type(e).__name__}: {e}"
            except Exception as e:  # keep going; the record carries the error and is retried on resume
                err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
            gerr = rec.guard_errors()
            if gerr and err is None and self.strict_guards:
                err = f"guard_error: {gerr}"
            n_llm, agent_ms, pt, ct = rec.agent_llm()
            events = ([e.to_json() for e in rec.layer_events] + [e.to_json() for e in rec.llm_events]
                      + [json.dumps(e.__dict__) for e in rec.tool_events])
            return SessionRecord(
                session_id=sid,
                run_id=self.run_id,
                stack_id=self.stack.id,
                stack_layers=[l.name for l in layers],
                layer_modes={l.name: l.mode.value for l in layers},
                suite=case.suite,
                user_task=case.user_task,
                injection_task=case.injection_task,
                attack=case.attack,
                category=case.category,
                is_attack=case.injection_task is not None,
                utility=utility,
                attack_success=(security if injection_task is not None else None),
                aborted_by=rec.aborted_by,
                fired={l.name: rec.fired().get(l.name, False) for l in layers},
                first_fire_seq=rec.first_fire_seq(),
                max_score=rec.max_score(),
                n_turns=rec.turn + 1,
                n_llm_calls=n_llm,
                wall_ms=rec.wall_ms(),
                agent_latency_ms=agent_ms,
                guard_latency_ms=rec.guard_latency(),
                guard_calls=rec.guard_calls(),
                agent_prompt_tokens=pt,
                agent_completion_tokens=ct,
                guard_tokens=rec.guard_tokens(),
                config={**self.run_config, "case_id": case.case_id, "vector": case.vector,
                        "objective": case.objective, "benchmark_version": self.version,
                        "model": self.model_name, "stack_specs": self.stack.layer_specs,
                        "attempt": attempt},
                error=err,
                final_output=rec.final_output,
                harm_seq=rec.harm_seq(),
                attributed_fire_seq=rec.attributed_fire_seq(),
                guard_errors=gerr,
                skipped_reason=skipped,
                n_retries=n_retries,
                denied_calls=sum(1 for e in rec.tool_events if e.denied),
            ), events

    def run(self, cases: list[Case], out_path: str | Path, workers: int = 1,
            events_path: str | Path | None = None) -> int:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = _done_ids(out_path)
        todo = [c for c in cases if c.case_id not in done]
        log.info("stack=%s cases=%d done=%d todo=%d workers=%d", self.stack.id, len(cases), len(done), len(todo),
                 workers)
        if todo:
            # warm every layer once here, serially: model loads and lazy imports then happen in one
            # thread, and the workers reuse the shared models (see _adapter_common.shared_load)
            self._pipeline()
        lock = threading.Lock()
        n = 0
        t0 = time.monotonic()
        every = max(5, min(25, len(todo) // 10))
        ev_f = open(events_path, "a") if events_path else None
        with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.run_case_with_events, c): c for c in todo}
            for fut in as_completed(futs):
                rec, events = fut.result()
                with lock:
                    f.write(rec.to_json() + "\n")
                    f.flush()
                    if ev_f is not None:
                        ev_f.write("".join(e + "\n" for e in events))
                        ev_f.flush()
                n += 1
                if n % every == 0 or n == len(todo):
                    mins = (time.monotonic() - t0) / 60
                    rate = n / mins if mins > 0 else 0.0
                    eta = (len(todo) - n) / rate if rate else 0.0
                    log.info("%d/%d  stack=%s  %.1f cases/min  elapsed %s  ETA %s", n, len(todo), self.stack.id,
                             rate, _hm(mins), _hm(eta))
        if ev_f is not None:
            ev_f.close()
        self.close()
        return n

    def close(self) -> None:
        with self._layers_lock:
            for l in self._all_layers:
                try:
                    l.close()
                except Exception:  # best effort at shutdown
                    log.exception("close() failed for %s", l.name)
            self._all_layers.clear()


def _hm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60}h{m % 60:02d}m"
