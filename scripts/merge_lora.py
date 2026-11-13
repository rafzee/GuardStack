#!/usr/bin/env python
"""Merge a PEFT LoRA adapter into its base model's weights, shard by shard, without peft.

Used for NemoGuard 8B content safety (nvidia/llama-3.1-nemoguard-8b-content-safety), whose LoRA
also targets lm_head and sits in a subfolder of its repo; serving the merged weights avoids
depending on vLLM's LoRA support for lm_head.

    python scripts/merge_lora.py --base meta-llama/Llama-3.1-8B-Instruct \
        --adapter nvidia/llama-3.1-nemoguard-8b-content-safety \
        --out ~/work/models/llama-3.1-nemoguard-8b-content-safety-merged

W' = W + s * (B @ A) for every LoRA module, with s = lora_alpha / r (alpha / sqrt(r) under
rsLoRA), computed in float32 and cast back to the base dtype. Full tensors saved by the
adapter (modules_to_save, or a LoRA'd layer's base_layer) replace the base tensor first. Anything the script does not understand
(DoRA, per-module rank/alpha patterns, embedding LoRA, unmatched keys) stops it with an error.
Runs on CPU; peak RAM is about one base shard plus one fp32 copy of the largest tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

LORA_KEY = re.compile(r"^(?:base_model\.model\.)?(?P<mod>.+)\.lora_(?P<ab>[AB])(?:\.[A-Za-z0-9_]+)?\.weight$")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja", "chat_template.json", "tokenizer.model")


def local_dir(repo_or_path: str) -> Path:
    """A path, or the cached snapshot of a repo id. Reads the cache layout directly:
    huggingface_hub >= 1.3 refuses snapshot_download(local_files_only=True) when files were
    deliberately skipped at download time (we skip original/*), even though the weights are there."""
    p = Path(os.path.expanduser(repo_or_path))
    if p.exists():
        return p
    from huggingface_hub import constants
    d = Path(constants.HF_HUB_CACHE) / ("models--" + repo_or_path.replace("/", "--"))
    ref = d / "refs" / "main"
    if ref.exists() and (d / "snapshots" / ref.read_text().strip()).is_dir():
        return d / "snapshots" / ref.read_text().strip()
    snaps = sorted((d / "snapshots").glob("*"), key=lambda q: q.stat().st_mtime) if (d / "snapshots").is_dir() else []
    if not snaps:
        raise SystemExit(f"{repo_or_path}: not in the HF cache ({d})")
    return snaps[-1]


def find_adapter(root: Path) -> Path:
    hits = sorted(root.rglob("adapter_config.json"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one adapter_config.json under {root}, found {[str(h) for h in hits]}")
    return hits[0].parent


def load_tensors(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(adapter_dir: Path):
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("peft_type", "LORA").upper() != "LORA":
        raise SystemExit(f"not a LoRA adapter: peft_type={cfg.get('peft_type')}")
    if cfg.get("use_dora"):
        raise SystemExit("DoRA adapters are not supported by this merge")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit("per-module rank_pattern / alpha_pattern are not supported by this merge")
    if cfg.get("fan_in_fan_out"):
        raise SystemExit("fan_in_fan_out=True (Conv1D-style) adapters are not supported by this merge")
    r, alpha = int(cfg["r"]), float(cfg.get("lora_alpha", cfg["r"]))
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    files = [adapter_dir / n for n in ("adapter_model.safetensors", "adapter_model.bin") if (adapter_dir / n).exists()]
    if not files:
        raise SystemExit(f"no adapter_model.safetensors/.bin in {adapter_dir}")
    sd = load_tensors(files[0])
    pairs: dict[str, dict] = {}
    full: dict[str, object] = {}
    for k, t in sd.items():
        if "lora_embedding" in k or "lora_magnitude" in k:
            raise SystemExit(f"unsupported adapter tensor {k}")
        m = LORA_KEY.match(k)
        if m:
            pairs.setdefault(m["mod"] + ".weight", {})[m["ab"]] = t
            continue
        key = re.sub(r"^base_model\.model\.", "", k)
        key = re.sub(r"\.modules_to_save\.[A-Za-z0-9_]+\.", ".", key)
        key = key.replace(".base_layer.", ".")   # base weight saved alongside a LoRA'd lm_head / embedding
        full[key] = t
    for target, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise SystemExit(f"{target}: has only lora_{''.join(ab)}")
    return cfg, scale, pairs, full, files[0]


def merge(base: str, adapter: str, out: Path, dtype: str | None = None) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = local_dir(base)
    adapter_dir = find_adapter(local_dir(adapter))
    cfg, scale, pairs, full, afile = plan(adapter_dir)
    idx_path = base_dir / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif (base_dir / "model.safetensors").exists():
        shards, weight_map = ["model.safetensors"], None
    else:
        raise SystemExit(f"{base_dir}: no safetensors weights (the merge reads safetensors only)")
    out.mkdir(parents=True, exist_ok=True)
    done_lora, done_full = set(), set()
    for shard in shards:
        tensors = {}
        with safe_open(str(base_dir / shard), framework="pt") as f:
            meta = f.metadata()
            for key in f.keys():
                w = f.get_tensor(key)
                if key in full:      # replacement first: a LoRA delta applies on top of a saved base_layer
                    t = full[key]
                    if tuple(t.shape) != tuple(w.shape):
                        raise SystemExit(f"{key}: adapter full tensor {tuple(t.shape)} vs base {tuple(w.shape)}")
                    w = t.to(w.dtype)
                    done_full.add(key)
                if key in pairs:
                    A, B = pairs[key]["A"].float(), pairs[key]["B"].float()
                    if (B.shape[0], A.shape[1]) != tuple(w.shape) or A.shape[0] != B.shape[1]:
                        raise SystemExit(f"{key}: base {tuple(w.shape)} vs B{tuple(B.shape)} @ A{tuple(A.shape)}")
                    w = (w.float() + scale * (B @ A)).to(w.dtype)
                    done_lora.add(key)
                if dtype:
                    w = w.to(getattr(torch, dtype))
                tensors[key] = w.contiguous()
        save_file(tensors, str(out / shard), metadata=meta or {"format": "pt"})
        print(f"  wrote {shard}: {len(tensors)} tensors", flush=True)
        del tensors
    missing = sorted(set(pairs) - done_lora) + sorted(set(full) - done_full)
    if missing:
        shutil.rmtree(out)
        raise SystemExit(f"adapter tensors with no matching base tensor (merge removed): {missing[:5]} ...")
    if weight_map is not None:
        shutil.copy2(idx_path, out / idx_path.name)
    for name in ("config.json", "generation_config.json") + TOKENIZER_FILES:
        if (base_dir / name).exists():
            shutil.copy2(base_dir / name, out / name)
    for name in TOKENIZER_FILES:          # the adapter's own tokenizer / chat template wins if it ships one
        if (adapter_dir / name).exists():
            shutil.copy2(adapter_dir / name, out / name)
    if dtype:
        c = json.loads((out / "config.json").read_text())
        c["torch_dtype"] = c["dtype"] = dtype
        (out / "config.json").write_text(json.dumps(c, indent=2))
    info = {"base": base, "base_dir": str(base_dir), "adapter": adapter, "adapter_dir": str(adapter_dir),
            "adapter_file_sha256": sha256(afile), "r": cfg["r"], "lora_alpha": cfg.get("lora_alpha"),
            "use_rslora": bool(cfg.get("use_rslora")), "scale": scale, "lora_modules_merged": len(done_lora),
            "full_tensors_replaced": sorted(done_full), "target_modules": cfg.get("target_modules")}
    (out / "MERGE_INFO.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, help="repo id or path; the adapter may sit in a subfolder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default=None, help="optional output dtype, e.g. bfloat16 (default: keep base)")
    a = ap.parse_args(argv)
    out = Path(os.path.expanduser(a.out))
    if out.exists() and any(out.iterdir()):
        print(f"{out} exists and is not empty; delete it first to re-merge", file=sys.stderr)
        return 1
    info = merge(a.base, a.adapter, out, a.dtype)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
