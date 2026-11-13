#!/usr/bin/env python
"""Fail if anything that would identify the authors is about to be published.

    python scripts/check_anonymity.py            # scan what git would track (or the whole tree)
    python scripts/check_anonymity.py --staged   # only files staged for the next commit

Double-blind review and the anonymising mirror only hide the repository's OWNER. Everything inside
the files is served as written, so a cluster name, a NetID in an example command, or an institutional
policy reference deanonymises the submission just as effectively as a byline.

Run it before every push. Add a term the moment it appears; the check is only worth as much as the
list. Exit status 1 means something matched.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# term -> why it identifies us / what to write instead
DENY: dict[str, str] = {
    r"mxb\d{6}": "cluster NetID; use $USER or <netid>",
    r"\bMuhaimin\b": "author name",
    r"\bMunir\b": "author name",
    r"muhaiminbinmunir": "author email",
    r"\bThuraisingham\b": "author name",
    r"\bLatifur\b": "author name",
    r"\bKhan\b(?!\s*Academy)": "author name",
    r"0000-000[\d]-[\d]{4}-[\dX]{4}": "ORCID",
    r"\bUTD\b": "institution; say 'our institution'",
    r"UT Dallas|University of Texas": "institution",
    r"\bJuno\b|\bjuno\b": "the cluster's name identifies the institution; say 'the cluster'",
    r"Texas DIR": "state procurement list identifies the institution; say 'the cluster's model policy'",
    r"/Users/[a-z]+": "local home path leaks a username",
    r"Desktop/UTD|IEEE S&P/guardstack": "local working path",
    r"claude\.ai/(artifact|code)/": "private workspace link",
}
# Files the scan skips: not published, or legitimately about the submission venue.
SKIP = ("scripts/check_anonymity.py", ".gitignore", ".git/")
COMPILED = [(re.compile(p), why) for p, why in DENY.items()]


def candidate_files(staged: bool) -> list[Path]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged else \
          ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / p for p in out.split("\n") if p.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):   # not a git repo yet: scan the tree
        if staged:
            sys.exit("--staged needs a git repository")
        files = [p for p in ROOT.rglob("*") if p.is_file()
                 and not any(part in {".git", "__pycache__", "runs", "results", "logs", "payloads",
                                      "juno_out", ".pytest_cache"} for part in p.parts)]
    return [p for p in files if p.is_file() and not any(s in str(p.relative_to(ROOT)) for s in SKIP)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only the per-file counts")
    args = ap.parse_args()

    hits: list[tuple[str, int, str, str, str]] = []
    for path in candidate_files(args.staged):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                                              # binary or unreadable: nothing to leak as text
        for n, line in enumerate(text.splitlines(), 1):
            for rx, why in COMPILED:
                m = rx.search(line)
                if m:
                    hits.append((str(path.relative_to(ROOT)), n, m.group(0), why, line.strip()[:90]))
    if not hits:
        print("anonymity check: clean")
        return 0
    by_file: dict[str, int] = {}
    for f, *_ in hits:
        by_file[f] = by_file.get(f, 0) + 1
    print(f"anonymity check: {len(hits)} match(es) in {len(by_file)} file(s)\n")
    for f, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {f}")
    if not args.quiet:
        print()
        for f, n, term, why, line in hits:
            print(f"{f}:{n}: {term!r} ({why})\n      {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
