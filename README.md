# guardstack — measuring what each guardrail layer buys for an LLM agent

Research artifact. `guardstack` inserts off-the-shelf guardrail layers at every stage of an
[AgentDojo](https://github.com/ethz-spylab/agentdojo) agent loop and measures, per layer and per
stack, under static **and adaptive** attack:

| quantity | what it answers |
|---|---|
| **MBR(k)**, marginal block rate | P(layer *k* stops the attack \| layers 1..*k*-1 did not) — what the *k*-th layer adds in deployed order |
| **rho** | pairwise redundancy: do two layers fail on the same attacks? |
| ASR, utility | attack success and benign task completion |
| latency, tokens, GPU-seconds | per turn and cumulative, measured rather than estimated |
| Pareto front, ordering | every layer subset ranked on security vs cost; pipelined-filter ordering and submodular selection |

Layers wrap published components — LlamaFirewall (PromptGuard 2, AlignmentCheck, CodeShield),
NVIDIA NeMo Guardrails rails, the Llama Guard family, prompt-injection classifiers, a PII filter,
a tool-provenance gate and spotlighting. Each runs at a defined stage (`input`, `tool_output`,
`action`, `trace`, `output`) in either `enforce` or `shadow` mode. In `shadow_all`, every layer
observes and none blocks, which yields the full layer x case verdict matrix in a single run.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt        # harness + AgentDojo + analysis
pip install -r requirements/adapters.txt    # guardrail adapters (torch, transformers, nemoguardrails...)
pip install -r requirements/serve.txt       # only if you serve models locally with vLLM
```

Agents and guard models are reached over an OpenAI-compatible HTTP API, so any server works
(vLLM, TGI, a hosted endpoint). Set `GUARDSTACK_AGENT_URL` and the per-layer `base_url`.

## Data

**No datasets or model weights are committed.** Everything is public and fetched on demand:

```bash
python data/download_datasets.py --out $DATA_ROOT        # needs a Hugging Face token for gated sets
```

`configs/datasets.yaml` lists every source and `docs/datasets/dataset_matrix.md` records what each
one is used for, with licences and gating. The main benchmark (AgentDojo v1.2.2: 4 suites, 97 user
tasks, 35 injection tasks, 949 attack cases) installs with the Python package.

## Quick start — no cluster needed

```bash
# one stack, one suite, against a model you already serve
python scripts/run_stack.py --stack E --suites banking --model <id> --agent-url http://127.0.0.1:8000/v1

# every detector observes, none blocks -> the verdict matrix the analysis consumes
python scripts/run_stack.py --stack shadow_all --sample 0.1
python scripts/analyze.py --records runs/<RUN_ID>/shadow_all --out results/<RUN_ID>/analysis

pytest -q        # the harness, adapters, attacks and analysis have known-answer tests
```

`python scripts/analyze.py --synthetic --out /tmp/demo` runs the whole analysis on generated data,
so the metrics can be checked without a GPU.

## Reproducing the full study

`docs/RUNBOOK_CLUSTER.md` is a step-by-step guide for a SLURM cluster with 2 GPUs per node
(the study used 2 x H200 NVL, 143 GB each; the agent occupies one GPU and the guard and attacker
models share the other). It covers environment setup, model download and validation, each
experiment phase, monitoring, resume and repair, and a troubleshooting table.

`slurm/` holds the job scripts; `slurm/submit_path_a.sh <model> <phase>` submits one phase.
Paths come from environment variables, so nothing is specific to one site. The `scripts/` entry
points run standalone if you are not on SLURM.

## Attacks

| attack | what it does |
|---|---|
| `important_instructions` | AgentDojo's static injection (the baseline, and attempt 1 of every adaptive run) |
| `rewrite` | an attacker LLM writes a fresh payload each attempt from feedback |
| `edit` | minimal revisions of the best payload so far, keeping what already persuades the agent |
| `staged` | payloads must evade the stack's text detectors locally before an agent run is spent |
| `edit_staged` *(default)* | both: minimal edits that clear the local detectors first |
| multi-turn | escalation across turns toward a target action |

Feedback levels (`binary`, `guard`, `score`, `trace`) set how much the attacker learns between
attempts, from "did it work" up to the agent's tool trace. `--budget` caps attempts per case.

## Layout

```
guardstack/
  types.py, recorder.py       session records; per-event verdict, score, latency, tokens
  layers/base.py              GuardLayer: stage, enforce/shadow, action on block
  layers/builtin.py           regex, provenance gate, tool allowlist, spotlighting
  layers/*_layers.py          adapters: LlamaFirewall, NeMo Guardrails, PII, safety classifiers
  pipeline.py                 builds the guarded AgentDojo pipeline
  runner.py                   case enumeration, sharding, resume, JSONL output, payload replay
  llm.py                      OpenAI-compatible client with usage tracking
  attacks/adaptive.py         adaptive injection attackers and the feedback levels
  analysis/structure.py       verdict matrix, MBR, rho, Shapley, independence gap
  analysis/adaptive.py        static vs adaptive MBR(k) for enforced stacks
  analysis/selection.py       subsets, Pareto front, ordering, submodular selection
  analysis/assignment.py      Hungarian layer-to-category assignment
  analysis/report.py          figures
scripts/run_stack.py          one stack or one isolated layer
scripts/run_adaptive.py       adaptive injection attacks
scripts/run_multiturn.py      multi-turn escalation
scripts/export_payloads.py    best-shot payload per case, for replay against another stack
scripts/analyze.py            the static analysis (verdict matrix in, tables and figures out)
scripts/analyze_adaptive_mbr.py  static vs adaptive MBR(k)
scripts/check_anonymity.py    pre-publication scan for identifying strings
configs/stacks.yaml           layers and the stacks built from them
docs/RUNBOOK_CLUSTER.md       full reproduction guide
```

## Results and trajectories

Run outputs (`runs/`, `results/`) are not committed: they are large and fully regenerable from the
commands above. **Attack trajectories are not included in this snapshot and will be released
separately.**

## Licence

TBD — a licence file will be added before release.
