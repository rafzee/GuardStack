"""PII detectors for content leaving the agent: Microsoft Presidio and GLiNER.

Verified against presidio-analyzer / presidio-anonymizer 2.2.366 and gliner 0.2.30.

Typical placement: ``stage: action`` (PII in the arguments of an outgoing tool call, e.g. an
email body -- exfiltration) or ``stage: output`` (PII in the final answer). At the action
stage only string-valued arguments are scanned (``arg_names`` narrows that, ``tools``
restricts which tools are checked). ``ignore_user_supplied=True`` ignores entities whose
text occurs verbatim in a user turn (the user may legitimately ask to email their own
contact); default False = plain detector behaviour.

score = highest entity score (Presidio recognizer score / GLiNER span probability);
block = any entity with score >= threshold. With ``anonymize=True`` the layer becomes a
transform (output/tool_output/input stages only): it never blocks, records the score, and
replaces the entities in ENFORCE mode.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .. import llm as _llm
from ..types import Stage, Verdict
from . import _adapter_common as C
from .base import CheckContext, GuardLayer

_PII_STAGES = (Stage.ACTION, Stage.OUTPUT, Stage.TOOL_OUTPUT, Stage.INPUT)


class _PIIBase(GuardLayer):
    family = "pii"
    default_stage = Stage.ACTION

    def __init__(self, threshold: float = 0.5, arg_names: Sequence[str] | None = None,
                 tools: Sequence[str] | None = None, ignore_user_supplied: bool = False,
                 anonymize: bool = False, **kw) -> None:
        super().__init__(threshold=threshold, **kw)
        C.require_stage(type(self).__name__, self.stage, _PII_STAGES)
        if anonymize and self.stage == Stage.ACTION:
            raise ValueError("anonymize=True needs a text stage (output/tool_output/input): the pipeline "
                             "does not apply transforms to tool calls")
        self.arg_names = list(arg_names) if arg_names else None
        self.tools = set(tools) if tools else None
        self.ignore_user_supplied, self.anonymize = ignore_user_supplied, anonymize

    def _content(self, ctx: CheckContext) -> str | None:
        if ctx.stage == Stage.ACTION:
            if ctx.tool_call is None or (self.tools is not None and ctx.tool_call.function not in self.tools):
                return None
            return C.tool_call_string_args(ctx.tool_call, self.arg_names)
        return ctx.text or ""

    @staticmethod
    def _user_text(ctx: CheckContext) -> str:
        parts = [C.msg_text(m) for m in ctx.messages if m.get("role") == "user"]
        if ctx.user_goal:
            parts.append(ctx.user_goal)
        return "\n".join(parts).lower()

    def _filter(self, ctx, ents: list[dict]) -> list[dict]:
        if not self.ignore_user_supplied:
            return ents
        user = self._user_text(ctx)
        return [e for e in ents if e["text"].strip().lower() not in user]

    def _verdict(self, ctx: CheckContext, text: str, ents: list[dict], tokens_in: int = 0) -> Verdict:
        ents = self._filter(ctx, ents)
        hits = [e for e in ents if e["score"] >= self.threshold]
        score = max((e["score"] for e in ents), default=0.0)
        kinds = sorted({e["label"] for e in hits})
        if self.anonymize:
            ctx.extra[f"{self.name}:masked"] = self._mask(text, hits) if hits else None
            return Verdict(block=False, score=score, tokens_in=tokens_in,
                           reason=("pii_masked:" + ",".join(kinds)) if hits else None, extra={"entities": kinds})
        return Verdict(block=bool(hits), score=score, tokens_in=tokens_in,
                       reason=("pii:" + ",".join(kinds)) if hits else None, extra={"entities": kinds})

    def _mask(self, text: str, hits: list[dict]) -> str:
        out, last = [], 0
        for e in sorted(hits, key=lambda e: e["start"]):
            if e["start"] < last:
                continue
            out.append(text[last:e["start"]])
            out.append(f"<{e['label']}>")
            last = e["end"]
        out.append(text[last:])
        return "".join(out)

    def transform(self, ctx: CheckContext) -> str | None:
        if not self.anonymize:
            return None
        return ctx.extra.get(f"{self.name}:masked")


class PresidioPIILayer(_PIIBase):
    """Presidio AnalyzerEngine (spaCy NLP engine). ``entities=None`` = all supported entity
    types. ``spacy_model`` must be installed (``python -m spacy download en_core_web_lg`` on a
    login node) or be a path; ``"blank:en"`` builds an empty pipeline so only the pattern
    recognizers (email, phone, credit card, IBAN, IP, URL, SSN, ...) run -- no PERSON /
    LOCATION NER. warmup() refuses to let Presidio ``spacy download`` at run time. Presidio's
    EmailRecognizer calls ``tldextract.extract``, which fetches the Public Suffix List over
    the network on first use (falling back to a bundled snapshot after the request fails --
    a stall on offline compute nodes); ``offline_psl=True`` (default) swaps tldextract's
    process-wide extractor for one that only uses the bundled snapshot (deterministic,
    no network). With ``anonymize=True`` presidio-anonymizer's default ``<ENTITY_TYPE>``
    replacement is used."""

    def __init__(self, entities: Sequence[str] | None = None, threshold: float = 0.5, language: str = "en",
                 spacy_model: str = "en_core_web_lg", allow_list: Sequence[str] | None = None,
                 offline_psl: bool = True, **kw) -> None:
        kw.setdefault("name", "presidio_pii")
        super().__init__(threshold=threshold, **kw)
        self.entities = list(entities) if entities else None
        self.language, self.spacy_model = language, spacy_model
        self.allow_list = list(allow_list) if allow_list else None
        self.offline_psl = offline_psl
        self._analyzer = self._anonymizer = None

    def _model_path(self) -> str:
        if self.spacy_model.startswith("blank:"):
            import spacy

            p = Path(tempfile.gettempdir()) / f"guardstack_spacy_blank_{self.spacy_model[6:]}_{os.getpid()}"
            if not p.exists():
                spacy.blank(self.spacy_model[6:]).to_disk(p)
            return str(p)
        import spacy.util

        if not (spacy.util.is_package(self.spacy_model) or Path(self.spacy_model).exists()):
            raise RuntimeError(f"spaCy model {self.spacy_model!r} is not installed; install it on a node with "
                               "internet (python -m spacy download ...) or use spacy_model='blank:en'")
        return self.spacy_model

    @staticmethod
    def _offline_tldextract() -> None:
        import tldextract.tldextract as tx

        if tuple(getattr(tx.TLD_EXTRACTOR, "suffix_list_urls", ()) or ()):
            tx.TLD_EXTRACTOR = tx.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)

    def _build(self):
        if self.offline_psl:
            self._offline_tldextract()
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        prov = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy", "models": [{"lang_code": self.language, "model_name": self._model_path()}]})
        return AnalyzerEngine(nlp_engine=prov.create_engine(), supported_languages=[self.language])

    def warmup(self) -> None:
        if self._analyzer is not None:
            return
        analyzer = self._build()
        if self.entities:
            unknown = set(self.entities) - set(analyzer.get_supported_entities(self.language))
            if unknown:
                raise ValueError(f"Presidio does not support entities {sorted(unknown)}")
        analyzer.analyze("contact jane.doe@example.com", language=self.language)  # one-off init costs
        if self.anonymize:
            from presidio_anonymizer import AnonymizerEngine

            self._anonymizer = AnonymizerEngine()
        self._analyzer = analyzer

    def _mask(self, text: str, hits: list[dict]) -> str:
        if self._anonymizer is None:
            return super()._mask(text, hits)
        from presidio_anonymizer.entities import RecognizerResult

        res = [RecognizerResult(e["label"], e["start"], e["end"], e["score"]) for e in hits]
        return self._anonymizer.anonymize(text, res).text

    def check(self, ctx: CheckContext) -> Verdict:
        self.warmup()
        text = self._content(ctx)
        if not text:
            return Verdict(block=False, score=0.0, reason="nothing to scan" if text is None else "empty")
        results = self._analyzer.analyze(text, language=self.language, entities=self.entities,
                                         score_threshold=0.0, allow_list=self.allow_list)
        ents = [{"label": r.entity_type, "start": r.start, "end": r.end, "score": float(r.score),
                 "text": text[r.start:r.end]} for r in results]
        return self._verdict(ctx, text, ents)


GLINER_DEFAULT_LABELS = [
    "person", "email", "phone number", "address", "credit card number", "bank account number", "iban",
    "social security number", "passport number", "date of birth", "ip address", "username", "password",
]


class GlinerPIILayer(_PIIBase):
    """GLiNER zero-shot NER for PII. Default model urchade/gliner_multi_pii-v1 (Apache-2.0,
    individual author; fine-tune of urchade/gliner_multi-v2.1). nvidia/gliner-PII (NVIDIA
    Open Model License; built on urchade/gliner_large-v2.1) also works. ``labels`` are
    free-text entity names; score = span probability from ``predict_entities``. On offline
    nodes the GLiNER checkpoint *and* its encoder's tokenizer must be in HF_HOME."""

    def __init__(self, model: str = "urchade/gliner_multi_pii-v1", labels: Sequence[str] | None = None,
                 threshold: float = 0.5, device: str | None = None, flat_ner: bool = True, **kw) -> None:
        kw.setdefault("name", "gliner_pii")
        super().__init__(threshold=threshold, **kw)
        self.model_id = C.resolve_str(model) or "urchade/gliner_multi_pii-v1"
        _llm.assert_allowed(self.model_id)
        self.labels = list(labels or GLINER_DEFAULT_LABELS)
        self.device, self.flat_ner = device, flat_ner
        self._model = None
        self._infer_lock = C.NOLOCK

    def _load(self):
        from gliner import GLiNER

        model = GLiNER.from_pretrained(self.model_id, local_files_only=C.hf_local_only())
        return model.to(C.torch_device(self.device))

    def warmup(self) -> None:
        if self._model is None:
            self._model, self._infer_lock = C.shared_load(self, self._load, self.model_id, self.device)

    def check(self, ctx: CheckContext) -> Verdict:
        self.warmup()
        text = self._content(ctx)
        if not text:
            return Verdict(block=False, score=0.0, reason="nothing to scan" if text is None else "empty")
        # predict below the decision threshold (floor 0.3) so `score` also reflects near misses
        with self._infer_lock:     # one GLiNER model shared by all worker threads
            preds = self._model.predict_entities(text, self.labels, flat_ner=self.flat_ner,
                                                 threshold=min(self.threshold, 0.3))
        ents = [{"label": p["label"], "start": int(p["start"]), "end": int(p["end"]), "score": float(p["score"]),
                 "text": p.get("text", text[int(p["start"]):int(p["end"])])} for p in preds]
        return self._verdict(ctx, text, ents)
