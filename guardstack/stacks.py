"""Layer registry and stack configs.

A stack config (YAML) looks like::

    stacks:
      - id: C
        layers:
          - {type: promptguard2, stage: tool_output}
          - {type: llamaguard4, stage: output, params: {model: meta-llama/Llama-Guard-4-12B}}
      - id: shadow_all          # every layer observes, none acts -> verdict matrix in one run
        shadow_all: true
        layers: [...]

``type`` is a key of REGISTRY (lazy import path) so heavy dependencies are only imported
when a stack uses them.
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .layers.base import GuardLayer
from .types import Mode

REGISTRY: dict[str, str] = {
    # dependency-free
    "regex": "guardstack.layers.builtin:RegexLayer",
    "provenance_gate": "guardstack.layers.builtin:ProvenanceGate",
    "tool_allowlist": "guardstack.layers.builtin:ToolAllowlist",
    "spotlight": "guardstack.layers.builtin:Spotlight",
    "substring": "guardstack.layers.builtin:SubstringLayer",
    # adapters (guardstack/layers/*.py, written against the real libraries)
    "promptguard2": "guardstack.layers.llamafirewall_layers:PromptGuard2Layer",
    "alignment_check": "guardstack.layers.llamafirewall_layers:AlignmentCheckLayer",
    "codeshield": "guardstack.layers.llamafirewall_layers:CodeShieldLayer",
    "nemo_rails": "guardstack.layers.nemo_layers:NemoRailsLayer",
    "presidio_pii": "guardstack.layers.pii_layers:PresidioPIILayer",
    "gliner_pii": "guardstack.layers.pii_layers:GlinerPIILayer",
    "llamaguard": "guardstack.layers.safety_classifiers:LlamaGuardLayer",
    "shieldgemma": "guardstack.layers.safety_classifiers:ShieldGemmaLayer",
    "wildguard": "guardstack.layers.safety_classifiers:WildGuardLayer",
    "granite_guardian": "guardstack.layers.safety_classifiers:GraniteGuardianLayer",
    "gpt_oss_safeguard": "guardstack.layers.safety_classifiers:GptOssSafeguardLayer",
    "hf_pi_classifier": "guardstack.layers.safety_classifiers:HFPromptInjectionClassifier",
}


def _resolve(path: str):
    mod, _, attr = path.partition(":")
    return getattr(importlib.import_module(mod), attr)


def make_layer(spec: dict, force_mode: Mode | None = None) -> GuardLayer:
    spec = dict(spec)
    cls = _resolve(REGISTRY[spec.pop("type")])
    params = spec.pop("params", {}) or {}
    if force_mode is not None:
        spec["mode"] = force_mode
    return cls(**spec, **params)


@dataclass
class StackConfig:
    id: str
    layer_specs: list[dict] = field(default_factory=list)
    shadow_all: bool = False
    description: str = ""

    def build(self) -> list[GuardLayer]:
        mode = Mode.SHADOW if self.shadow_all else None
        layers = [make_layer(s, force_mode=mode) for s in self.layer_specs]
        names = [l.name for l in layers]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"stack {self.id}: duplicate layer names {dup}; set distinct `name:` fields")
        return layers


def expand_env(obj):
    """Replace ${VAR} with environment values (left as-is when unset)."""
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    return obj


def load_raw(path: str | Path) -> dict:
    return expand_env(yaml.safe_load(Path(path).read_text()))


def load_stacks(path: str | Path) -> dict[str, StackConfig]:
    raw = load_raw(path)
    anchors = raw.get("layers", {})  # optional named layer specs: reference with {ref: name}
    out = {}
    for s in raw["stacks"]:
        specs = []
        for l in s.get("layers", []):
            if isinstance(l, str):
                l = {"ref": l}
            if "ref" in l:
                base = dict(anchors[l["ref"]])
                base.update({k: v for k, v in l.items() if k != "ref"})
                l = base
            specs.append(l)
        out[s["id"]] = StackConfig(s["id"], specs, bool(s.get("shadow_all", False)), s.get("description", ""))
    return out
