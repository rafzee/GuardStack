#!/usr/bin/env python
"""Download, verify and diagnose Hugging Face model weights in $HF_HOME.

    python scripts/hf_models.py fetch  <repo> [<repo> ...]   # download (errors are printed, not hidden)
    python scripts/hf_models.py check  <repo> [<repo> ...]   # offline: are the weight files really there?
    python scripts/hf_models.py diag                         # why downloads fail: versions, token, CDN, test file

`check` looks at the weight files themselves, not config.json: every shard named in a
*.index.json must resolve to a non-empty blob, and a repo with no index must have at least
one non-empty weight file. Nested folders (e.g. a LoRA adapter in a subfolder) are included.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Never pulled: duplicate checkpoints in other formats.
IGNORE = ["original/*", "metal/*", "*.gguf", "*.onnx", "onnx/*", "*.msgpack", "*.h5",
          "flax_model*", "tf_model*", "*.ot", "coreml/*", "openvino/*"]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
NON_WEIGHT_BIN = {"training_args.bin", "optimizer.bin", "scheduler.bin"}


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def repo_dir(repo: str) -> Path:
    return hf_home() / "hub" / ("models--" + repo.replace("/", "--"))


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit in ("B", "K") else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def is_weight(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith(("original/", "metal/")) or name in NON_WEIGHT_BIN:
        return False
    return name.endswith(WEIGHT_SUFFIXES)


def snapshot_of(d: Path) -> Path | None:
    ref = d / "refs" / "main"
    if ref.exists():
        s = d / "snapshots" / ref.read_text().strip()
        if s.is_dir():
            return s
    snaps = sorted((d / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime) if (d / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


def inspect(repo: str) -> tuple[str, str, str]:
    """-> (status, size, detail). status: complete | INCOMPLETE | NO-WEIGHTS | MISSING."""
    d = repo_dir(repo)
    snap = snapshot_of(d) if d.is_dir() else None
    if snap is None:
        return "MISSING", "-", "nothing in the cache"
    present: dict[str, int] = {}          # rel path -> size of the resolved blob (0 = dangling/empty)
    total = 0
    for p in snap.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(snap).as_posix()
        try:
            size = p.resolve(strict=True).stat().st_size
        except (OSError, RuntimeError):
            size = 0
        present[rel] = size
        total += size
    partial = len(list((d / "blobs").glob("*.incomplete"))) if (d / "blobs").is_dir() else 0
    note = f"; {partial} partial download(s) in blobs/" if partial else ""

    have = {r for r, sz in present.items() if is_weight(r) and sz > 0}
    groups: dict[str, list[tuple[str, set[str]]]] = {}   # folder -> [(index file, shards it names)]
    for rel in present:
        if rel.endswith(".index.json") and not rel.startswith(("original/", "metal/")):
            try:
                wm = json.loads((snap / rel).read_text()).get("weight_map", {})
            except (OSError, json.JSONDecodeError):
                return "INCOMPLETE", human(total), f"unreadable {rel}{note}"
            base = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
            groups.setdefault(base, []).append((rel, {base + v for v in wm.values()}))
    if groups:
        # every folder needs one complete index; a .bin index beside a complete safetensors one
        # is the same checkpoint in another format (we skip .bin duplicates on purpose)
        done, worst = [], None
        for base, idxs in sorted(groups.items()):
            scored = sorted(((sorted(exp - have), rel, len(exp)) for rel, exp in idxs), key=lambda x: len(x[0]))
            missing, rel, n = scored[0]
            if missing:
                worst = worst or (missing, rel, n)
            else:
                done.append((rel, n))
        if worst:
            missing, rel, n = worst
            return ("INCOMPLETE", human(total),
                    f"{n - len(missing)}/{n} shards of {rel}; first missing: {missing[0]}{note}")
        return "complete", human(total), ", ".join(f"{n} shards" for _, n in done) + note
    if have:
        return "complete", human(total), ", ".join(sorted(have))[:120] + note
    return "NO-WEIGHTS", human(total), f"only small files ({len(present)} files){note}"


def cmd_check(repos: list[str]) -> int:
    bad = 0
    for r in repos:
        status, size, detail = inspect(r)
        bad += status != "complete"
        print(f"{status:<11} {size:>7}  {r}  [{detail}]")
    print(f"-- {len(repos) - bad}/{len(repos)} complete")
    return 0


def plan_ignores(api, repo: str) -> list[str]:
    """Skip .bin/.pt duplicates when the repo also has safetensors at the same level."""
    ign = list(IGNORE)
    try:
        files = api.list_repo_files(repo)
    except Exception:
        return ign
    dirs_with_st = {f.rsplit("/", 1)[0] if "/" in f else "" for f in files
                    if f.endswith(".safetensors") and not f.startswith(("original/", "metal/"))}
    for f in files:
        if f.endswith((".bin", ".pt", ".pth")) and is_weight(f):
            parent = f.rsplit("/", 1)[0] if "/" in f else ""
            if parent in dirs_with_st:
                ign.append(f)
    return ign


def cmd_fetch(repos: list[str]) -> int:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    failed = []
    for r in repos:
        t = time.time()
        print(f"== {r}  {time.strftime('%H:%M:%S')}", flush=True)
        try:
            snapshot_download(r, ignore_patterns=plan_ignores(api, r), max_workers=8)
        except GatedRepoError:
            print(f"GATED: {r} -- your token has no access yet. Request it at https://huggingface.co/{r}")
            failed.append(r)
            continue
        except RepositoryNotFoundError:
            print(f"NOT FOUND: {r} -- wrong id, or gated and the token is missing/invalid")
            failed.append(r)
            continue
        except Exception as e:  # noqa: BLE001 -- print everything; this is the diagnosis
            print(f"FAILED: {r}: {type(e).__name__}: {e}")
            failed.append(r)
            continue
        status, size, detail = inspect(r)
        print(f"{status}: {r}  {size} in {time.time() - t:.0f}s  [{detail}]", flush=True)
        if status != "complete":
            failed.append(r)
    if failed:
        print("FAILED/INCOMPLETE:", " ".join(failed))
    return 1 if failed else 0


def cmd_diag() -> int:
    import importlib.metadata as md
    import subprocess

    print("== versions")
    for pkg in ("huggingface_hub", "hf_xet", "hf_transfer", "requests", "httpx"):
        try:
            print(f"  {pkg} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg} not installed")
    print("== env")
    for k in sorted(os.environ):
        if k.startswith(("HF_", "HUGGINGFACE")) and "TOKEN" not in k:
            print(f"  {k}={os.environ[k]}")
    print(f"  token file: {'yes' if (hf_home() / 'token').exists() else 'no'}; "
          f"HF_TOKEN set: {'yes' if os.environ.get('HF_TOKEN') else 'no'}")
    print("== token")
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami()
        print(f"  user={who.get('name')}  token role={who.get('auth', {}).get('accessToken', {}).get('role')}")
    except Exception as e:  # noqa: BLE001
        print(f"  whoami failed: {type(e).__name__}: {e}")
    print("== reachability (any HTTP code = reachable; 'fail' = blocked/DNS)")
    for u in ("https://huggingface.co", "https://cdn-lfs.hf.co", "https://cdn-lfs-us-1.hf.co",
              "https://cas-bridge.xethub.hf.co", "https://cas-server.xethub.hf.co",
              "https://transfer.xethub.hf.co"):
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-m", "10", "-w", "%{http_code}", u],
                           capture_output=True, text=True)
        c = r.stdout.strip()
        print(f"  {u} -> {c if c and c != '000' else 'fail (no connection)'}")
    print("== test download: one ~0.7 GB weight file (protectai deberta, ungated)")
    repo, fname = "protectai/deberta-v3-base-prompt-injection-v2", "model.safetensors"
    from huggingface_hub import try_to_load_from_cache
    if isinstance(try_to_load_from_cache(repo, fname), str):
        print("  (already in the cache, so the timing below is not a network speed)")
    code = ("import sys,time; from huggingface_hub import hf_hub_download; t=time.time(); "
            f"p=hf_hub_download('{repo}','{fname}'); import os; s=os.path.getsize(p); "
            "print(f'  OK {s/1e6:.0f} MB in {time.time()-t:.0f}s = {s/1e6/max(time.time()-t,1e-3):.0f} MB/s')")
    for label, extra in (("default (Xet if available)", {}), ("Xet disabled (classic LFS)", {"HF_HUB_DISABLE_XET": "1"})):
        print(f"  -- {label}")
        env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", **extra)
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1800)
        out = (r.stdout + r.stderr).strip().splitlines()
        print("\n".join("  " + line for line in out[-3:]))
        if r.returncode == 0:
            break
    print("== cache size")
    subprocess.run(["du", "-sh", str(hf_home())])
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("fetch", "check", "diag"):
        print(__doc__)
        return 2
    if argv[0] == "diag":
        return cmd_diag()
    return {"fetch": cmd_fetch, "check": cmd_check}[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
