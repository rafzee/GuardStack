#!/usr/bin/env python3
"""Download the pinned evaluation datasets for the guardstack study.

Meant to run on the the cluster *login node*: compute-node internet access is
unverified, so everything the SLURM jobs need is fetched here first. The script
only downloads files. It runs no model, hashes files in 1 MiB chunks and uses
few download workers, which keeps it well inside the ~8 GB per-user RAM cap on
the login nodes.

Sources
  * GitHub: shallow fetch of one pinned commit (`git fetch --depth 1 <sha>`),
    with sparse checkout of only the paths we need, then a check that HEAD equals
    the pin.
  * Hugging Face: `huggingface_hub.snapshot_download(revision=<sha>)`. Where we
    could not find a commit SHA ahead of time (revision: null below), the
    script resolves `main` to a SHA once, downloads exactly that SHA and writes
    it to revisions.lock.json. Later runs reuse the lock (commit it to git).
    Gated sets need a token (HF_TOKEN, or the one saved by `hf auth login`), and their terms must already
    have been accepted in the web UI.

Every run updates <out>/manifest.json with per-file SHA-256 checksums, a tree
hash per dataset, the resolved revision and licence or gating notes.

Examples
  python data/download_datasets.py --list
  python data/download_datasets.py --out ~/work/guardstack/datasets
  HF_TOKEN=hf_xxx python data/download_datasets.py --only mhj,wildjailbreak
  python data/download_datasets.py --only safemtdata --allow-gray-area
  python data/download_datasets.py --verify            # re-hash and compare

the cluster tips: put this on ~/work or /groups/<pi> (home is 50 GB / 300k inodes),
and export HF_HOME=~/work/.cache/huggingface before running.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOCK = SCRIPT_DIR / "revisions.lock.json"
DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "datasets.yaml"
MANIFEST_NAME = "manifest.json"
CHUNK = 1 << 20  # 1 MiB

# --------------------------------------------------------------------------------------
# the cluster prohibited-technology interlock (Texas covered entities). Datasets only: we never
# download models here, but refuse any repo owned by a covered entity, and refuse the
# Qwen-based Agent-SafetyBench scorer by name in case someone adds it.
# --------------------------------------------------------------------------------------
BANNED_OWNER_PREFIXES = (
    "qwen", "alibaba", "deepseek", "moonshot", "kimi", "zai-org", "thudm", "zhipu",
    "bytedance", "doubao", "xiaomi",
)
BANNED_REPOS = {"thu-coai/shieldagent"}

# --------------------------------------------------------------------------------------
# Registry. Pins were verified on 2026-11-03: GitHub SHAs via `git ls-remote`/checkout,
# HF SHAs from inspect_evals pins or the HF commit/tree pages. revision=None means that
# no SHA could be verified: it is resolved once and locked (see module docstring).
# gated: True / False / None (= unknown, try anonymously and report GatedRepoError).
# default=False entries are fetched only when named in --only (or with --all).
# gray_area: entries are skipped unless --allow-gray-area is given.
# --------------------------------------------------------------------------------------
REGISTRY: list[dict[str, Any]] = [
    # ---- agentic, AgentDojo-native ---------------------------------------------------
    dict(name="agentdojo", kind="git", repo="ethz-spylab/agentdojo",
         revision="a75aba7631d3ca5fb7ab938965c97ead2f9ff84b", ref="v0.1.35 (benchmark v1.2.2)",
         paths=["src/agentdojo/data/", "src/agentdojo/default_suites/", "LICENSE", "README.md"],
         license="MIT", gated=False, default=True, categories=[3, 4, 5, 8, 9],
         size="~0.6 MB", notes="Provenance copy of suites; the runner uses the pip package agentdojo==0.1.35."),
    # ---- indirect prompt injection / goal hijack -------------------------------------
    dict(name="injecagent", kind="git", repo="uiuc-kang-lab/InjecAgent",
         revision="f19c9f2c79a41046eb13c03c51a24c567a8ffa07",
         paths=["data/", "src/", "LICENCE", "README.md"],
         license="MIT", gated=False, default=True, categories=[3],
         size="~5 MB", notes="1,054 cases (510 DH + 544 DS) per setting (base/enhanced)."),
    dict(name="asb", kind="git", repo="agiresearch/ASB",
         revision="544540ff0788998072f7df463056fd17e62cc9f4",
         paths=["data/", "config/", "LICENSE", "README.md"],
         license="MIT", gated=False, default=True, categories=[3, 4],
         size="<1 MB data", notes="10 agents / 51 tasks / 400 attacker tools; refusal judge gpt-4o-mini (swap)."),
    dict(name="bipia", kind="git", repo="microsoft/BIPIA",
         revision="a004b69ec0dd446e0afd461d98cb5e96e120a5d0",
         paths=["benchmark/", "bipia/", "LICENSE", "README.md", "NOTICE.md"],
         license="MIT", gated=False, default=True, categories=[3],
         size="~2 MB data", notes="WebQA and Summarization contexts must be rebuilt from NewsQA/XSum (licence)."),
    dict(name="llmail_inject", kind="hf", repo="microsoft/llmail-inject-challenge",
         revision="1063bdf01ec8762b812d5e06ee768a06faa5a6f7",
         license="MIT", gated=None, default=False, categories=[3],
         size="2.45 GB (whole repo)", notes="Optional: large. Human adaptive injection payloads."),
    dict(name="wasp", kind="git", repo="facebookresearch/wasp",
         revision="ffee6f41fde76acd14bd792db442479c506260c2",
         paths=["webarena_prompt_injections/configs/", "webarena_prompt_injections/constants.py",
                "LICENSE", "README.md"],
         license="CC-BY-NC-4.0", gated=False, default=False, categories=[3, 4],
         size="<1 MB", notes="Optional: needs dockerised GitLab/Reddit (VisualWebArena); not runnable on the cluster as-is."),
    # ---- direct jailbreak / harmful output -------------------------------------------
    dict(name="harmbench", kind="git", repo="centerforaisafety/HarmBench",
         revision="8e1604d1171fe8a48d8febecd22f600e462bdcdd",
         paths=["data/behavior_datasets/", "eval_utils.py", "LICENSE", "README.md"],
         license="MIT", gated=False, default=True, categories=[1, 2, 6],
         size="~0.6 MB", notes="Text test split: 159 standard + 81 contextual + 80 copyright."),
    dict(name="jbb_behaviors", kind="hf", repo="JailbreakBench/JBB-Behaviors",
         revision="886acc352a31533ffbcf4ef22c744658688086fc",
         license="MIT (per GitHub README; HF card licence field not re-checked)", gated=None,
         default=True, categories=[1, 9], size="small",
         notes="100 harmful + 100 benign behaviors + judge_comparison (300)."),
    dict(name="jbb_artifacts", kind="git", repo="JailbreakBench/artifacts",
         revision="909e68c01d94222b8ad2e397a017e2e12e2adb73",
         paths=["attack-artifacts/", "LICENSE", "README.md"],
         license="MIT", gated=False, default=True, categories=[1],
         size="~4.5 MB", notes="Static jailbreak strings (PAIR, GCG, JBC, prompt_with_random_search, DSN)."),
    dict(name="wildjailbreak", kind="hf", repo="allenai/wildjailbreak", revision=None,
         allow_patterns=["eval/*", "README.md"],
         license="ODC-BY", gated=True, default=True, categories=[1, 9],
         size="eval split only (full repo 541 MB)",
         notes="Eval: 2,000 adversarial harmful + 210 adversarial benign. allow_patterns UNVERIFIED."),
    dict(name="strongreject", kind="git", repo="alexandrasouly/strongreject",
         revision="f7cad6c17e624e21d8df2278e918ae1dddb4cb56",
         paths=["strongreject_dataset/", "strongreject/", "LICENSE", "README.md"],
         license="MIT (custom data); sourced prompts keep original licences", gated=False,
         default=True, categories=[6], size="<1 MB",
         notes="313 forbidden prompts (+60 small); rubric judge prompt included."),
    dict(name="wildguardmix", kind="hf", repo="allenai/wildguardmix", revision=None,
         license="ODC-BY", gated=True, default=True, categories=[6, 9],
         size="UNVERIFIED", notes="wildguardtest (1,725) used for guard-only offline P/R."),
    # ---- multi-turn -------------------------------------------------------------------
    dict(name="mhj", kind="hf", repo="ScaleAI/mhj", revision=None,
         license="CC-BY-NC-4.0", gated=True, default=True, categories=[2],
         size="1.3 MB", notes="537 conversations / 2,912 prompts; some behaviors and WMDP-bio responses withheld."),
    dict(name="mt_jailbench", kind="git", repo="SafetyArena/mt-jailbench",
         revision="cb8184e0a9ad7a65c86a00e2a38c57a5f58043f0",
         paths=None, license="MIT", gated=False, default=True, categories=[2],
         size="~13 MB", notes="Adaptive multi-turn harness (Crescendo, CoA, FITD, ActorBreaker, X-Teaming). "
                               "Default attacker is Qwen2.5-32B: MUST be reconfigured before use on the cluster."),
    dict(name="safemtdata", kind="hf", repo="SafeMTData/SafeMTData", revision=None,
         license="UNVERIFIED", gated=None, default=True, categories=[2],
         gray_area="dialogues collected against deepseek-chat; refusals generated by deepseek-chat",
         size="UNVERIFIED", notes="ActorAttack data; 600 seeds from Circuit-Breaker train set."),
    dict(name="xguard_train", kind="hf", repo="marslabucla/XGuard-Train", revision=None,
         license="UNVERIFIED (card states none)", gated=True, default=False, categories=[2],
         gray_area="attacker Qwen-2.5-32B-IT; DeepSeek-V3 among target models",
         size="UNVERIFIED", notes="30,695 training conversations; not an eval set."),
    dict(name="cosafe", kind="git", repo="ErxinYu/CoSafe-Dataset",
         revision="a298d9c742203b00eaeca58dfca7100c7e2ff360", paths=None,
         license="NONE in repo (all rights reserved by default)", gated=False, default=False,
         categories=[2], size="~2 MB", notes="1,400 coreference multi-turn prompts, 14 categories."),
    # ---- unsafe tool call -------------------------------------------------------------
    dict(name="agentharm", kind="hf", repo="ai-safety-institute/AgentHarm",
         revision="e23b3fe60a0da9037314b88e5ee3a0c054970dad",
         license="MIT + use restriction (safety research only)", gated=None, default=True,
         categories=[5, 9], size="<1 MB",
         notes="Pinned as in UKGovernmentBEIS/inspect_evals. Canary GUID; do not train on it."),
    dict(name="agent_safetybench", kind="git", repo="thu-coai/Agent-SafetyBench",
         revision="74feea8de601b3a1449a93fcf70017fe61556f73",
         paths=["data/", "environments/", "LICENSE", "README.md"],
         license="MIT", gated=False, default=True, categories=[5],
         size="~4 MB, ~700 files", notes="2,000 cases. Official scorer ShieldAgent is Qwen-based (banned): use substitute judge."),
    dict(name="rjudge", kind="git", repo="Lordog/R-Judge",
         revision="83ce301da3ad50dd8b397e772863f5411c3d3dc2", paths=["data/", "README.md"],
         license="NONE in repo", gated=False, default=False, categories=[5],
         size="~2 MB", notes="571 labelled agent trajectories at this commit (paper: 569); guard-only eval."),
    dict(name="toolemu", kind="git", repo="ryoungj/ToolEmu",
         revision="ac4a7ab7ed8c7985d96231e214bd6b54304b7ddb", paths=["assets/", "LICENSE", "README.md"],
         license="Apache-2.0", gated=False, default=False, categories=[5],
         size="~4 MB", notes="144 cases; needs a GPT-4-class emulator and evaluator."),
    # ---- insecure code ----------------------------------------------------------------
    dict(name="seccodeplt", kind="hf", repo="Virtue-AI-HUB/SecCodePLT",
         revision="e847141569d113dea5348dcf834e15d21027b5bd",
         license="MIT", gated=None, default=True, categories=[7], size="UNVERIFIED",
         notes="insecure_coding: 1,345 samples with functionality and security unit tests. "
               "Pin taken from an HF tree URL; may not be the latest commit."),
    dict(name="cweval", kind="git", repo="Co1lin/CWEval",
         revision="e9a2a124c8c53679b6d8d27adfd2f6c40e7576d7", paths=None,
         license="Apache-2.0", gated=False, default=True, categories=[7],
         size="~1 MB benchmark", notes="119 tasks, 31 CWEs, C/C++/Go/JS/Py. Docker image: convert to Apptainer on the cluster."),
    dict(name="cyberseceval", kind="git", repo="meta-llama/PurpleLlama",
         revision="f64eebd53003a87d9c2d2a3a712133534c214c8d",
         paths=["CybersecurityBenchmarks/datasets/instruct/", "CybersecurityBenchmarks/datasets/autocomplete/",
                "CybersecurityBenchmarks/datasets/interpreter/", "CybersecurityBenchmarks/datasets/prompt_injection/",
                "CybersecurityBenchmarks/datasets/mitre_frr/", "CybersecurityBenchmarks/LICENSE",
                "CybersecurityBenchmarks/README.md"],
         license="MIT (CybersecurityBenchmarks/LICENSE)", gated=False, default=True, categories=[7, 5, 9],
         size="~13 MB", notes="instruct 1,916 / autocomplete 1,916 / interpreter 500 / prompt_injection 251 / mitre_frr 750."),
    # ---- PII / privacy ----------------------------------------------------------------
    dict(name="privacylens", kind="git", repo="SALT-NLP/PrivacyLens",
         revision="9c2ee07b080dc54ed4924af11d9751e81753c94d",
         paths=["data/", "evaluation/", "LICENSE", "README.md"],
         license="MIT", gated=False, default=True, categories=[8],
         size="~3 MB", notes="493 seeds, vignettes and trajectories; default leak judge Mistral-7B-Instruct-v0.2."),
    dict(name="ai4privacy_openpii", kind="hf", repo="ai4privacy/open-pii-masking-500k-ai4privacy",
         revision="144ca06b62d78f173df83e0f47feb97524921714",
         license="card says cc-by-4.0 but adds custom terms (commercial use needs permission)",
         gated=None, default=False, categories=[8], size="UNVERIFIED (580,227 rows)",
         notes="Guard-only PII span detection (Presidio/GLiNER). Pin taken from an HF README URL."),
    dict(name="confaide", kind="git", repo="skywalker023/confAIde",
         revision="3174c63e689226189675a740b7bf9053f10bae50",
         paths=["benchmark/", "eval.py", "LICENSE", "README.md"],
         license="MIT", gated=False, default=False, categories=[8], size="<1 MB",
         notes="Tiers 1/2a/2b/3/4 (10/98/98/1955 lines/1150 lines)."),
    dict(name="agentdam", kind="git", repo="facebookresearch/ai-agent-privacy",
         revision="5d4068a404b624ed24ebbe5fb49ba5f644557912",
         paths=["agentdam/data/", "agentdam/configs/", "LICENSE", "README.md"],
         license="CC-BY-NC-4.0", gated=False, default=False, categories=[8], size="~3 MB",
         notes="246 tasks (48 GitLab / 114 Reddit / 84 shopping); needs WebArena Docker; GPT-4o judge."),
    # ---- benign / over-refusal --------------------------------------------------------
    dict(name="xstest", kind="git", repo="paul-rottger/exaggerated-safety",
         revision="d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
         paths=["xstest_prompts.csv", "evaluation/classify_completions_strmatch.py",
                "evaluation/classify_completions_gpt.py", "LICENSE", "readme.md"],
         license="CC-BY-4.0", gated=False, default=True, categories=[9], size="<0.2 MB",
         notes="250 safe + 200 unsafe prompts."),
    dict(name="or_bench", kind="hf", repo="bench-llm/or-bench",
         revision="e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
         license="CC-BY-4.0", gated=None, default=True, categories=[9], size="UNVERIFIED",
         notes="Configs or-bench-80k / or-bench-hard-1k / or-bench-toxic (row counts UNVERIFIED)."),
    dict(name="notinject", kind="git", repo="leolee99/InjecGuard",
         revision="1b5751e88bf7475acbedfc8eda795ce060307c84",
         paths=["datasets/NotInject_one.json", "datasets/NotInject_two.json",
                "datasets/NotInject_three.json", "LICENSE", "README.md"],
         license="MIT", gated=False, default=True, categories=[9], size="<1 MB",
         notes="339 benign prompts with injection trigger words (3 x 113). Measures over-defense of PI detectors."),
]
REG = {d["name"]: d for d in REGISTRY}


# ----------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def banned(repo: str) -> bool:
    r = repo.lower()
    owner = r.split("/", 1)[0]
    return r in BANNED_REPOS or owner.startswith(BANNED_OWNER_PREFIXES)


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return default


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def iter_files(root: Path):
    """Yield data files under root, skipping VCS and HF-cache metadata."""
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if rel == Path("."):
            dirnames[:] = [d for d in dirnames if d not in (".git", ".cache")]
        dirnames.sort()
        for fn in sorted(filenames):
            yield Path(dirpath) / fn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checksum_tree(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    total = 0
    for p in iter_files(root):
        if p.is_symlink() and not p.exists():
            continue
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        files[rel] = {"sha256": sha256_file(p), "bytes": size}
        total += size
    tree = hashlib.sha256("\n".join(f"{v['sha256']}  {k}" for k, v in sorted(files.items())).encode()).hexdigest()
    return {"n_files": len(files), "total_bytes": total, "tree_sha256": tree, "files": files}


def run_git(args: list[str], cwd: Path | None = None) -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    res = subprocess.run(["git", "-c", "advice.detachedHead=false", *args], cwd=cwd, env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()[:500]}")
    return res.stdout.strip()


# ----------------------------------------------------------------------------- fetchers
def fetch_git(entry: dict[str, Any], dest: Path) -> str:
    sha = entry["revision"]
    url = f"https://github.com/{entry['repo']}.git"
    tmp = dest.parent / f".{dest.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    run_git(["init", "-q"], cwd=tmp)
    run_git(["remote", "add", "origin", url], cwd=tmp)
    paths = entry.get("paths")
    if paths:
        # Classic (non-cone) sparse checkout via the info file: works on old and new git.
        run_git(["config", "core.sparseCheckout", "true"], cwd=tmp)
        pats = ["/" + p.lstrip("/") for p in paths]
        (tmp / ".git" / "info").mkdir(parents=True, exist_ok=True)
        (tmp / ".git" / "info" / "sparse-checkout").write_text("\n".join(pats) + "\n")
    fetched = False
    for extra in (["--filter=blob:none"], []):  # partial clone first, plain fetch as fallback
        try:
            run_git(["fetch", "-q", "--depth", "1", *extra, "origin", sha], cwd=tmp)
            fetched = True
            break
        except RuntimeError as e:
            log(f"    shallow fetch {'with filter ' if extra else ''}failed, retrying: {e}")
    if not fetched:
        run_git(["fetch", "-q", "origin"], cwd=tmp)
    run_git(["checkout", "-q", "--detach", sha], cwd=tmp)
    head = run_git(["rev-parse", "HEAD"], cwd=tmp)
    if head != sha:
        raise RuntimeError(f"HEAD {head} != pinned {sha}")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return head


def _hf_errors():
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
    except ImportError:  # huggingface_hub < 0.23
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
    return GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError


def resolve_hf_revision(entry: dict[str, Any], lock: dict[str, Any], token: str | None,
                        refresh: bool) -> tuple[str, bool, str]:
    """Return (sha, pinned_in_registry, how)."""
    if entry.get("revision"):
        return entry["revision"], True, "registry"
    locked = lock.get(entry["name"])
    if locked and locked.get("repo") == entry["repo"] and not refresh:
        return locked["revision"], False, "lockfile"
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(entry["repo"], revision="main", token=token)
    lock[entry["name"]] = {"repo": entry["repo"], "revision": info.sha, "resolved_at": now_iso()}
    return info.sha, False, "resolved-main-now"


def fetch_hf(entry: dict[str, Any], dest: Path, sha: str, token: str | None, workers: int) -> str:
    from huggingface_hub import snapshot_download
    tmp = dest.parent / f".{dest.name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    snapshot_download(repo_id=entry["repo"], repo_type="dataset", revision=sha, local_dir=str(tmp),
                      allow_patterns=entry.get("allow_patterns"), token=token, max_workers=workers)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return sha


# ----------------------------------------------------------------------------- commands
def print_list() -> None:
    hdr = f"{'name':<20} {'src':<4} {'repo@revision':<58} {'gated':<6} {'set':<8} {'cat':<10} licence"
    log(hdr)
    log("-" * len(hdr))
    for d in REGISTRY:
        rev = (d["revision"] or "unpinned")[:12]
        flag = "GRAY" if d.get("gray_area") else ("default" if d["default"] else "optional")
        g = {True: "yes", False: "no", None: "?"}[d["gated"]]
        cats = ",".join(str(c) for c in d["categories"])
        log(f"{d['name']:<20} {d['kind']:<4} {d['repo'] + '@' + rev:<58} {g:<6} {flag:<8} {cats:<10} {d['license']}")
    log("\nCategories: 1 jailbreak, 2 multi-turn, 3 indirect PI, 4 goal hijack, 5 unsafe tool call,"
        " 6 harmful output, 7 insecure code, 8 PII leak, 9 benign/over-refusal")
    log("GRAY = generated with a the cluster-covered entity's model; downloaded only with --allow-gray-area.")


def check_config(path: Path) -> int:
    """Warn if configs/datasets.yaml pins a different revision than this registry."""
    try:
        import yaml  # type: ignore
    except ImportError:
        log("[config] PyYAML not installed; skipping consistency check")
        return 0
    if not path.exists():
        log(f"[config] {path} not found; skipping consistency check")
        return 0
    cfg = yaml.safe_load(path.read_text()) or {}
    problems = 0
    for cat in (cfg.get("categories") or {}).values():
        for role in ("primary", "secondary", "extra"):
            for e in cat.get(role) or []:
                name = e.get("download")
                if not name:
                    continue
                if name not in REG:
                    log(f"[config] WARNING {e.get('name')}: download '{name}' not in registry")
                    problems += 1
                elif (e.get("revision") or None) != REG[name]["revision"]:
                    log(f"[config] WARNING {e.get('name')}: yaml revision {e.get('revision')} "
                        f"!= registry {REG[name]['revision']}")
                    problems += 1
    log(f"[config] consistency check: {problems} problem(s)")
    return problems


def select(args) -> list[dict[str, Any]]:
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in REG]
        if unknown:
            sys.exit(f"unknown dataset(s): {', '.join(unknown)} (see --list)")
        return [REG[n] for n in names]
    return [d for d in REGISTRY if d["default"] or args.all]


def verify(out: Path, manifest: dict[str, Any], names: list[str]) -> int:
    bad = 0
    for name in names:
        m = manifest.get("datasets", {}).get(name)
        if not m or m.get("status") != "ok":
            continue
        cur = checksum_tree(out / name)
        ok = cur["tree_sha256"] == m["tree_sha256"]
        bad += not ok
        log(f"[verify] {name:<20} {'OK' if ok else 'MISMATCH'} ({cur['n_files']} files)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = os.environ.get("GUARDSTACK_DATA_DIR") or str(Path.home() / "work" / "guardstack" / "datasets")
    ap.add_argument("--out", default=default_out, help=f"output directory (default: {default_out})")
    ap.add_argument("--only", help="comma-separated dataset names (see --list)")
    ap.add_argument("--list", action="store_true", help="list registry and exit")
    ap.add_argument("--all", action="store_true", help="include optional (non-default) datasets")
    ap.add_argument("--allow-gray-area", action="store_true",
                    help="also fetch datasets generated with the cluster-covered entities' models (flagged)")
    ap.add_argument("--skip-gated", action="store_true",
                    help="treat gated sets without HF_TOKEN as a soft skip (default: counts as failure)")
    ap.add_argument("--force", action="store_true", help="re-download even if present at the same revision")
    ap.add_argument("--verify", action="store_true", help="re-hash existing downloads against manifest and exit")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK), help="revision lock file for unpinned HF sets")
    ap.add_argument("--refresh-lock", action="store_true", help="re-resolve unpinned HF sets to current main")
    ap.add_argument("--workers", type=int, default=4, help="HF download workers (keep low on login nodes)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="datasets.yaml for the consistency check")
    args = ap.parse_args()

    if args.list:
        print_list()
        check_config(Path(args.config))
        return 0

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / MANIFEST_NAME
    manifest = load_json(manifest_path, {"datasets": {}})
    entries = select(args)

    if args.verify:
        return 1 if verify(out, manifest, [e["name"] for e in entries]) else 0

    if os.environ.get("SLURM_JOB_ID"):
        log("NOTE: running inside a SLURM job; compute-node internet is unverified on the cluster. "
            "Prefer the login node for downloads.")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:   # the token saved by `hf auth login` (in $HF_HOME/token)
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:  # noqa: BLE001
            token = None
    lock_path = Path(args.lock).expanduser()
    lock = load_json(lock_path, {})
    failures = 0
    soft_skips = 0

    for e in entries:
        name, dest = e["name"], out / e["name"]
        log(f"==> {name} ({e['kind']}:{e['repo']})")
        rec: dict[str, Any] = {k: e.get(k) for k in ("kind", "repo", "license", "gated", "gray_area",
                                                    "categories", "notes", "ref")}
        rec["requested_revision"] = e.get("revision")
        rec["local_path"] = str(dest)
        if banned(e["repo"]):
            log(f"    REFUSED: {e['repo']} belongs to a the cluster-prohibited entity")
            failures += 1
            continue
        if e.get("gray_area") and not args.allow_gray_area:
            log(f"    SKIPPED (gray area: {e['gray_area']}); pass --allow-gray-area to fetch")
            soft_skips += 1
            continue
        if e.get("gray_area"):
            log(f"    WARNING gray area: {e['gray_area']}")
        if e["kind"] == "hf" and e["gated"] is True and not token:
            log(f"    SKIPPED: gated. Accept the terms at https://huggingface.co/datasets/{e['repo']} "
                "and export HF_TOKEN")
            if args.skip_gated:
                soft_skips += 1
            else:
                failures += 1
            continue
        try:
            if e["kind"] == "git":
                sha, pinned, how = e["revision"], True, "registry"
            else:
                sha, pinned, how = resolve_hf_revision(e, lock, token, args.refresh_lock)
                if not pinned:
                    log(f"    revision not pinned in registry; using {sha} ({how})")
            prev = manifest["datasets"].get(name, {})
            if (dest.exists() and not args.force and prev.get("status") == "ok"
                    and prev.get("resolved_revision") == sha):
                log(f"    present at {sha[:12]}; skipping (use --force to refetch, --verify to re-hash)")
                continue
            if e["kind"] == "git":
                fetch_git(e, dest)
            else:
                fetch_hf(e, dest, sha, token, args.workers)
            sums = checksum_tree(dest)
            if sums["n_files"] == 0:
                raise RuntimeError("download produced no files (allow_patterns/paths matched nothing?)")
            rec.update(status="ok", resolved_revision=sha, revision_pinned_in_registry=pinned,
                       revision_source=how, downloaded_at=now_iso(), **sums)
            log(f"    ok: {sums['n_files']} files, {sums['total_bytes'] / 1e6:.2f} MB, tree {sums['tree_sha256'][:16]}")
        except Exception as exc:  # keep going; report at the end
            GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError = (
                _hf_errors() if e["kind"] == "hf" else (type(None),) * 3)
            kind = ("gated (accept terms / set HF_TOKEN)" if isinstance(exc, GatedRepoError) else
                    "repo not found" if isinstance(exc, RepositoryNotFoundError) else
                    "revision not found" if isinstance(exc, RevisionNotFoundError) else type(exc).__name__)
            log(f"    FAILED [{kind}]: {str(exc)[:400]}")
            shutil.rmtree(out / f".{name}.partial", ignore_errors=True)
            failures += 1
            prev = manifest["datasets"].get(name, {})
            if prev.get("status") == "ok" and dest.exists():  # keep the good copy's record
                prev["last_failed_attempt"] = {"at": now_iso(), "error": f"{kind}: {str(exc)[:400]}"}
                continue
            rec.update(status="failed", error=f"{kind}: {str(exc)[:400]}", attempted_at=now_iso())
        manifest["datasets"][name] = rec

    try:
        import huggingface_hub
        hf_ver = huggingface_hub.__version__
    except Exception:
        hf_ver = None
    try:
        from importlib.metadata import version
        adojo = version("agentdojo")
    except Exception:
        adojo = None
    manifest["environment"] = {"host": socket.gethostname(), "python": platform.python_version(),
                               "huggingface_hub": hf_ver, "agentdojo_pip": adojo,
                               "git": run_git(["--version"]), "updated_at": now_iso()}
    write_json_atomic(manifest_path, manifest)
    if lock:
        write_json_atomic(lock_path, lock)
    log(f"\nmanifest: {manifest_path}" + (f"\nlock:     {lock_path}" if lock else ""))
    log(f"failures: {failures}  soft-skips: {soft_skips}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
