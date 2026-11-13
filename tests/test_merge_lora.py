"""scripts/merge_lora.py on a hand-built base + adapter (no peft needed). The script was also
checked against peft 0.21.2's merge_and_unload on a tiny Llama (bitwise-equal weights)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_lora.py"


def _base(d: Path):
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    w = {"model.layers.0.self_attn.q_proj.weight": torch.randn(8, 6, generator=g).to(torch.bfloat16),
         "model.layers.0.mlp.up_proj.weight": torch.randn(10, 6, generator=g).to(torch.bfloat16),
         "lm_head.weight": torch.randn(12, 6, generator=g).to(torch.bfloat16)}
    st.save_file({k: w[k] for k in list(w)[:2]}, str(d / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    st.save_file({"lm_head.weight": w["lm_head.weight"]}, str(d / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors"}}))
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}))
    (d / "tokenizer.json").write_text("{}")
    return w


def _adapter(d: Path, cfg: dict, tensors: dict):
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", **cfg}))
    st.save_file(tensors, str(d / "adapter_model.safetensors"))


def _run(base, adapter, out):
    return subprocess.run([sys.executable, str(SCRIPT), "--base", str(base), "--adapter", str(adapter), "--out", str(out)],
                          capture_output=True, text=True)


@pytest.mark.parametrize("rslora", [False, True])
def test_merge_matches_formula(tmp_path, rslora):
    w = _base(tmp_path / "base")
    g = torch.Generator().manual_seed(1)
    A, B = torch.randn(4, 6, generator=g), torch.randn(8, 4, generator=g)
    Ah, Bh = torch.randn(4, 6, generator=g), torch.randn(12, 4, generator=g)
    _adapter(tmp_path / "repo" / "nested-lora-adapter", {"r": 4, "lora_alpha": 8, "use_rslora": rslora}, {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B,
        "base_model.model.lm_head.lora_A.weight": Ah, "base_model.model.lm_head.lora_B.weight": Bh,
        "base_model.model.lm_head.base_layer.weight": w["lm_head.weight"]})   # peft saves this for lm_head
    r = _run(tmp_path / "base", tmp_path / "repo", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    s = 8 / math.sqrt(4) if rslora else 8 / 4
    m1 = st.load_file(str(tmp_path / "out" / "model-00001-of-00002.safetensors"))
    m2 = st.load_file(str(tmp_path / "out" / "model-00002-of-00002.safetensors"))
    q = "model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(m1[q], (w[q].float() + s * B @ A).to(torch.bfloat16))
    assert torch.equal(m1["model.layers.0.mlp.up_proj.weight"], w["model.layers.0.mlp.up_proj.weight"])
    assert torch.equal(m2["lm_head.weight"], (w["lm_head.weight"].float() + s * Bh @ Ah).to(torch.bfloat16))
    info = json.loads((tmp_path / "out" / "MERGE_INFO.json").read_text())
    assert info["lora_modules_merged"] == 2 and info["scale"] == s
    assert (tmp_path / "out" / "config.json").exists() and (tmp_path / "out" / "tokenizer.json").exists()


def test_merge_refuses_what_it_does_not_understand(tmp_path):
    _base(tmp_path / "base")
    A, B = torch.ones(4, 6), torch.ones(8, 4)
    _adapter(tmp_path / "dora", {"r": 4, "lora_alpha": 8, "use_dora": True},
             {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
              "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B})
    assert "DoRA" in _run(tmp_path / "base", tmp_path / "dora", tmp_path / "o1").stderr
    _adapter(tmp_path / "unmatched", {"r": 4, "lora_alpha": 8},
             {"base_model.model.model.layers.9.self_attn.q_proj.lora_A.weight": A,
              "base_model.model.model.layers.9.self_attn.q_proj.lora_B.weight": B})
    r = _run(tmp_path / "base", tmp_path / "unmatched", tmp_path / "o2")
    assert r.returncode != 0 and "no matching base tensor" in r.stderr and not (tmp_path / "o2").exists()


def test_repo_id_resolves_from_cache_without_snapshot_download(tmp_path, monkeypatch):
    """hub >= 1.3 raises IncompleteSnapshotError offline when original/* was skipped; we read refs/main."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("merge_lora", SCRIPT)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    from huggingface_hub import constants
    d = tmp_path / "hub" / "models--org--base"
    (d / "refs").mkdir(parents=True)
    (d / "refs" / "main").write_text("c0ffee")
    (d / "snapshots" / "c0ffee").mkdir(parents=True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    assert M.local_dir("org/base") == d / "snapshots" / "c0ffee"
    with pytest.raises(SystemExit, match="not in the HF cache"):
        M.local_dir("org/absent")
