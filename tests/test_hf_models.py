"""scripts/hf_models.py check: weights, not config.json, decide 'complete'."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hf_models as H  # noqa: E402


def _mk(root: Path, repo: str, files: dict, partial: int = 0):
    d = root / "hub" / ("models--" + repo.replace("/", "--"))
    (d / "refs").mkdir(parents=True)
    (d / "refs" / "main").write_text("abc")
    snap = d / "snapshots" / "abc"
    (d / "blobs").mkdir()
    snap.mkdir(parents=True)
    for rel, content in files.items():
        p = snap / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            os.symlink(d / "blobs" / "dangling", p)
            continue
        b = d / "blobs" / hashlib.md5(rel.encode()).hexdigest()
        b.write_bytes(content)
        os.symlink(b, p)
    for i in range(partial):
        (d / "blobs" / f"p{i}.incomplete").write_bytes(b"x")


def test_check_statuses(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    idx = json.dumps({"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors"}}).encode()
    _mk(tmp_path, "o/ok", {"config.json": b"{}", "model.safetensors.index.json": idx,
                           "m-1.safetensors": b"w" * 10, "m-2.safetensors": b"w" * 10})
    _mk(tmp_path, "o/half", {"model.safetensors.index.json": idx, "m-1.safetensors": b"w"}, partial=1)
    _mk(tmp_path, "o/configs", {"config.json": b"{}", "tokenizer.json": b"t" * 100, "original/c.pth": b"w"})
    _mk(tmp_path, "o/gliner", {"gliner_config.json": b"{}", "pytorch_model.bin": b"w"})
    _mk(tmp_path, "o/lora", {"README.md": b"r", "sub/adapter_config.json": b"{}", "sub/adapter_model.safetensors": b"w"})
    _mk(tmp_path, "o/dangling", {"config.json": b"{}", "model.safetensors": None})
    bidx = json.dumps({"weight_map": {"a": "pytorch_model-1.bin", "b": "pytorch_model-2.bin"}}).encode()
    _mk(tmp_path, "o/twofmt", {"model.safetensors.index.json": idx, "pytorch_model.bin.index.json": bidx,
                               "m-1.safetensors": b"w", "m-2.safetensors": b"w"})       # WildGuard layout
    _mk(tmp_path, "o/twofmt-half", {"model.safetensors.index.json": idx, "pytorch_model.bin.index.json": bidx,
                                    "m-1.safetensors": b"w", "pytorch_model-2.bin": b"w"})
    names = ("o/ok", "o/half", "o/configs", "o/gliner", "o/lora", "o/dangling", "o/none", "o/twofmt", "o/twofmt-half")
    got = {r: H.inspect(r)[0] for r in names}
    assert got == {"o/ok": "complete", "o/half": "INCOMPLETE", "o/configs": "NO-WEIGHTS", "o/gliner": "complete",
                   "o/lora": "complete", "o/dangling": "NO-WEIGHTS", "o/none": "MISSING", "o/twofmt": "complete",
                   "o/twofmt-half": "INCOMPLETE"}
    assert "1/2 shards" in H.inspect("o/half")[2] and "partial" in H.inspect("o/half")[2]


def test_plan_ignores_skips_duplicate_bin():
    class Api:
        def list_repo_files(self, r):
            return ["model.safetensors", "pytorch_model.bin", "sub/pytorch_model.bin", "metal/model.bin"]
    extra = [i for i in H.plan_ignores(Api(), "x") if i not in H.IGNORE]
    assert extra == ["pytorch_model.bin"]
