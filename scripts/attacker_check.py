#!/usr/bin/env python
"""Will this attacker model write injections? Measures refusal rate on N AgentDojo goals.

    python scripts/attacker_check.py --attacker-model <id> --attacker-url http://127.0.0.1:8003/v1 --n 40
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from guardstack.attacks.adaptive import LLMProposer, looks_like_refusal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--attacker-model", required=True)
ap.add_argument("--attacker-url", required=True)
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()

pairs = []
for sname, suite in get_suites("v1.2.2").items():
    for it in suite.injection_tasks.values():
        for ut in suite.user_tasks.values():
            pairs.append((it.GOAL, ut.PROMPT))
random.Random(0).shuffle(pairs)
prop = LLMProposer(args.attacker_model, args.attacker_url, refusal_retries=0)
outs = [prop.propose(g, u, [], "guard") for g, u in pairs[:args.n]]
ref = [looks_like_refusal(o) for o in outs]
print(json.dumps({"attacker": args.attacker_model, "n": len(outs), "refusal_rate": sum(ref) / len(outs),
                  "median_chars": st.median(len(o) for o in outs)}, indent=2))
for o in [o for o, r in zip(outs, ref) if not r][:3]:
    print("--- sample payload ---\n" + o[:500])
