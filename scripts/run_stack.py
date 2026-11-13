#!/usr/bin/env python
"""Run one stack (or one isolated layer) over AgentDojo cases.

Examples
  # verdict matrix: every layer in shadow on the bare agent, shard 3 of 8
  python scripts/run_stack.py --stack shadow_all --shard 3 --num-shards 8 --workers 16
  # one layer alone, enforced
  python scripts/run_stack.py --isolated pg2_tool
  # nested stack E, latency mode on a stratified 10% sample
  python scripts/run_stack.py --stack E --mode latency --sample 0.1 --workers 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardstack.logsetup import describe_logging, setup_logging
from guardstack.llm import make_agent_llm  # noqa: E402
from guardstack.runner import AgentDojoRunner, enumerate_agentdojo_cases, shard  # noqa: E402
from guardstack.stacks import StackConfig, load_raw, load_stacks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# AgentDojo's own default system message, so bare-agent numbers are comparable to published results.
from agentdojo.agent_pipeline.agent_pipeline import load_system_message  # noqa: E402

SYSTEM_MESSAGE = load_system_message(None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-config", default=str(ROOT / "configs" / "stacks.yaml"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stack")
    g.add_argument("--isolated", help="layer key from the config's `layers:` block, run alone and enforced")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "meta-llama/Llama-3.3-70B-Instruct"))
    ap.add_argument("--agent-url", default=os.environ.get("GUARDSTACK_AGENT_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--no-benign", action="store_true")
    ap.add_argument("--no-attacks", action="store_true")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--sample", type=float, default=1.0, help="stratified fraction of cases (by suite x objective)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mode", choices=["throughput", "latency"], default="throughput")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M")))
    ap.add_argument("--out-dir", default=str(ROOT / "runs"))
    ap.add_argument("--payload-file", help="JSONL from scripts/export_payloads.py ({case_id, payload}): run only "
                    "these attack cases, with the given payload in place of the static attack text (replay of an "
                    "adaptive attacker's best-shot payloads). Benign cases are kept unless --no-benign.")
    args = ap.parse_args()
    setup_logging()

    if args.mode == "latency" and args.workers != 1:
        logging.warning("latency mode: forcing --workers 1 so timings are not contaminated by contention")
        args.workers = 1

    raw = load_raw(args.stack_config)
    stacks = load_stacks(args.stack_config)
    if args.isolated:
        spec = dict(raw["layers"][args.isolated])
        stack = StackConfig(f"iso_{args.isolated}", [spec])
    else:
        stack = stacks[args.stack]

    cases = enumerate_agentdojo_cases(args.benchmark_version, args.suites, args.attack,
                                      include_benign=not args.no_benign, include_attacks=not args.no_attacks)
    if args.sample < 1.0:
        rng = random.Random(args.seed)
        strata: dict[tuple, list] = {}
        for c in cases:
            strata.setdefault((c.suite, c.objective), []).append(c)
        cases = [c for s in strata.values() for c in rng.sample(s, max(1, round(len(s) * args.sample)))]
    payloads = None
    if args.payload_file:
        payloads = {}
        for line in Path(args.payload_file).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                payloads[d["case_id"]] = d["payload"]
        cases = [c for c in cases if c.injection_task is None or c.case_id in payloads]
        missing = set(payloads) - {c.case_id for c in cases}
        if missing:
            raise SystemExit(f"{len(missing)} case ids in {args.payload_file} are not in the benchmark, e.g. {sorted(missing)[:2]}")
    cases = shard(cases, args.shard, args.num_shards)

    run_config = {
        "run_id": args.run_id, "mode": args.mode, "workers": args.workers, "temperature": args.temperature,
        "attack": args.attack, "seed": args.seed, "agent_url": args.agent_url,
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "guard_model": os.environ.get("GUARD_MODEL"),
    }
    if args.payload_file:
        run_config["payload_file"] = str(args.payload_file)
    runner = AgentDojoRunner(stack, lambda: make_agent_llm(args.model, args.agent_url, args.temperature),
                             SYSTEM_MESSAGE, model_name=args.model, benchmark_version=args.benchmark_version,
                             run_config=run_config, payloads=payloads)
    out = Path(args.out_dir) / args.run_id / stack.id / f"shard{args.shard:03d}-of-{args.num_shards:03d}.jsonl"
    ev = out.with_suffix(".events.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "run_config.json").write_text(json.dumps({**run_config, "model": args.model,
                                                            "stack": stack.layer_specs}, indent=2))
    n = runner.run(cases, out, workers=args.workers, events_path=ev)
    logging.info("wrote %d records to %s", n, out)
    logging.info(describe_logging())


if __name__ == "__main__":
    main()
