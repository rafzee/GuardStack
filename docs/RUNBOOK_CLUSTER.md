# Reproducing the experiments on the cluster (SLURM cluster)

This runbook takes a new user from a the cluster account to every result in the paper. Each step gives
the command, how long it takes, and what a good result looks like. The numbers in "expected" come
from our own runs in November 2026 (logged in `docs/experiment_log.md`). Choices that affect the
results are listed in `docs/experiment_notes.md`.

**Contents:** 0 Requirements · 1 Conventions · 2 One-time setup · 3 Validation runs ·
4 The experiments · 5 Monitoring, resuming, collecting results · 6 Troubleshooting ·
7 Reference (scripts, variables, GPU layout, the cluster rules)

---

## 0. Requirements

- A the cluster account (our institution NetID) and the our institution network or campus VPN. All `ssh`/`rsync` below need it.
- A Hugging Face account with access to the gated repos in step 2.1.
- Storage under `~/work` (1 TB quota): about 480 GB of model weights, 0.5 GB of datasets, about 25 GB
  for the two Python environments plus a 9.5 GB packed copy, and tens of GB of results.
- Time: about 1 day of setup (most of it downloads and access approvals), then about 2 days of GPU jobs.

the cluster policy that the scripts follow (do not change these): vLLM servers run only inside batch jobs
and stop with them; at most 4 running jobs per user; nothing heavy on login nodes; and models from
DeepSeek, Alibaba (Qwen), Moonshot (Kimi), Z.ai (GLM), ByteDance (Doubao), Xiaomi (MiMo) and their
derivatives are forbidden on the cluster (Texas prohibited-technology rule). `guardstack/llm.py`
refuses to load such models by name, by architecture and by declared base model.

## 1. Conventions

**Paths on the cluster**

| What | Where |
|---|---|
| Code (this repo) | `~/work/guardstack` |
| Python envs | `~/work/envs/guardstack` (harness + guards), `~/work/envs/vllm` (model server) |
| Packed vLLM env | `~/work/envs/vllm.tar` |
| HF cache (weights) | `~/work/.cache/huggingface` (`HF_HOME`) |
| Merged NemoGuard | `~/work/models/llama-3.1-nemoguard-8b-content-safety-merged` |
| Datasets | `~/work/guardstack_data` |
| Job logs | `~/work/guardstack/logs/` (`<name>_<jobid>.out/.err`, vLLM logs in `logs/<jobid>/`) |
| Results | `~/work/guardstack/runs/<RUN_ID>/<stack>/shardNNN-of-NNN.jsonl` (+ `.events.jsonl`) |

**Every the cluster command starts in the repo directory** (`cd ~/work/guardstack`). The `slurm/...` paths
are relative, and `sbatch` writes logs relative to the directory it is run from.

**Working from your laptop (recommended).** Add this to `~/.ssh/config` on the laptop, run
`ssh <cluster>` once (password, and Duo if prompted), and every later `ssh`/`rsync` reuses that login for 8 hours:
```
Host <cluster>
  HostName <cluster>.utdallas.edu
  User <NetID>
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
```
Then a step can run from the laptop with its output saved locally, e.g.
`ssh <cluster> 'cd ~/work/guardstack && bash slurm/status.sh' > status.txt`. Never put your NetID password in
scripts, chats or files.

**Copying code to the cluster.** Edit on the laptop and push with
`rsync -av <local>/guardstack/ <cluster>:~/work/guardstack/` (no `--delete`). Running jobs are not disturbed:
each experiment stack runs as its own Python process and reads the code when it starts.

**Watching jobs.** `bash slurm/status.sh` shows running jobs, SLURM's start estimates for waiting
ones, and the last progress line of each running job. Progress lines look like
`250/949 stack=shadow_all 21.4 cases/min elapsed 0h12m ETA 0h33m`; the ETA is for the current stack.

## 2. One-time setup

### 2.1 Hugging Face access (do this first: approvals can take hours)

Create a token at https://huggingface.co/settings/tokens. A classic **read** token works. A
fine-grained token works only if "Read access to contents of all public gated repos you can access"
is ticked. While logged in, request access on each page:

| Used for | Repo |
|---|---|
| Smoke agent, NemoGuard base | `meta-llama/Llama-3.1-8B-Instruct` |
| Agent 2 | `meta-llama/Llama-3.3-70B-Instruct` |
| Llama Guard layers | `meta-llama/Llama-Guard-4-12B` |
| PromptGuard 2 layers | `meta-llama/Llama-Prompt-Guard-2-86M` |
| Attacker (accept the terms if asked) | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| Guard-only sweep | `google/shieldgemma-2b`, `allenai/wildguard` |
| Datasets | `allenai/wildjailbreak`, `allenai/wildguardmix`, `ScaleAI/mhj`, `Virtue-AI-HUB/SecCodePLT` |

The rest are not gated: `openai/gpt-oss-120b` (agent 1), `openai/gpt-oss-20b` (judge), the
NemoGuard adapter, GLiNER, the ProtectAI DeBERTa classifier, HarmBench, Granite Guardian and
gpt-oss-safeguard.

### 2.2 Copy the repo and probe a GPU node (1 min)

```bash
rsync -av <local>/guardstack/ <cluster>:~/work/guardstack/
ssh <cluster>
cd ~/work/guardstack && mkdir -p logs
sbatch slurm/probe.sbatch            # wait until `squeue --me` no longer lists it
cat logs/probe_*.out
```
Expected: `NVIDIA H200 NVL ... 143771 MiB`, driver 580.x with CUDA 13.0, and HTTP 200 for
huggingface.co, pypi.org and github.com (compute nodes have internet). If the driver differs, the
torch/vLLM builds in `requirements/` may need changing.

### 2.3 Python environments (15–20 min on the login node, then a 10-min GPU check)

```bash
cd ~/work/guardstack
SKIP_MODELS=1 SKIP_DATA=1 bash slurm/setup_env.sh 2>&1 | tee logs/setup.log
sbatch slurm/verify_env.sbatch       # GPU check of both envs + the test suite
cat logs/verify_env_*.out
```
There are two envs because vLLM 0.33 pins older torch/transformers than the guard libraries:
- `guardstack`: torch 2.15.0+cu132, transformers 5.22.1, llamafirewall 1.0.3, nemoguardrails 0.24.1,
  presidio 2.2.366, gliner 0.2.30, semgrep, agentdojo 0.1.35, plus spaCy `en_core_web_lg`;
- `vllm`: vLLM 0.33.0, torch 2.14.0+cu130.

`setup_env.sh` never imports torch, because torch cannot load on a login node under its memory cap.

Expected in `verify_env`: `gpu ok True` for both envs, and the tests end in `passed, 1 skipped` with
no failures. (We had 89 passed then; the suite has grown since, so compare against the current count.)

### 2.4 Model weights and datasets (batch job on a CPU node, several hours)

Save the token once, into the cache the jobs read:
```bash
cd ~/work/guardstack
module load miniconda; eval "$(conda shell.bash hook)"; conda activate ~/work/envs/guardstack
HF_HOME=~/work/.cache/huggingface hf auth login      # paste the token; answer "n" to git credential
sbatch slurm/download.sbatch                         # tiers 1-3 + datasets, then a check
```
Options: `TIERS="1" sbatch slurm/download.sbatch` gets only the smoke-test models (about 20 GB);
`TIERS="" sbatch slurm/download.sbatch` gets datasets only. Reruns skip whatever is already complete,
so after an access approval simply submit again.

Check the result at any time. This takes seconds, reads only the local cache, and is safe on the login node:
```bash
bash slurm/download_models.sh check
```
Expected: `complete` for all 15 models, `-- 15/15 complete`, about 480 GB. `check` verifies that every
shard named in each `*.index.json` is on disk (one complete index per folder is enough). Ignore
`du` sizes; on the cluster's file system they lag behind right after a download.

If downloads fail, diagnose on a compute node:
```bash
srun -p normal -c 4 --mem=16G -t 0:30:00 bash slurm/download_models.sh diag
```
It prints the token's user and type, whether each HF file host is reachable, and the result of a 0.7 GB
test download done twice: with HF's Xet transport and with Xet disabled. If only the Xet-disabled
download works, use `HF_HUB_DISABLE_XET=1 sbatch slurm/download.sbatch`.

Datasets land in `~/work/guardstack_data` (`python data/download_datasets.py --list` shows them).
SafeMTData is skipped on purpose: it was collected against deepseek-chat, a grey area under the
the cluster rule. AgentDojo itself ships inside the `agentdojo` package.

### 2.5 Merge the NemoGuard adapter (CPU job, about 20 min)

The `nemo_out` layer uses NVIDIA's NemoGuard 8B content-safety model, published as a LoRA adapter on
Llama-3.1-8B-Instruct (it also adapts `lm_head`). `scripts/merge_lora.py` merges it into full weights.
On a test model its output is bitwise identical to peft's `merge_and_unload`, and it refuses adapters
it does not understand.
```bash
sbatch slurm/merge_nemoguard.sbatch
cat logs/merge_*.out
```
Expected: ends with `MERGE_INFO` (`lora_modules_merged`, `full_tensors_replaced: ["lm_head.weight"]`)
and a 15 GB folder with 4 shards. Jobs serve that folder under the name
`llama-3.1-nemoguard-8b-content-safety`. If the folder is missing, stacks with `nemo_out` stop at warmup.

### 2.6 Pack the vLLM env (CPU job, about 5 min; repeat after any change to `~/work/envs/vllm`)

Python imports from `~/work` are slow on the cluster's file system: `import vllm, torch` took 45 s there
versus 3 s from a node's RAM disk, and every model server imports vLLM twice. The pack job writes a
plain tar of the env (`~/work/envs/vllm.tar`, 9.5 GB). `serve_models.sh` unpacks it to `/tmp` at the
start of each GPU job (12–15 s) and checks it with `vllm serve --help` before using it. If the tar is
missing or broken, it falls back to `~/work`.
```bash
sbatch slurm/pack_envs.sbatch
cat logs/pack_*.out                 # "packed in ...s" and the import time from both places
```
Use plain `tar`. conda-pack produced a broken env here (pip-upgraded setuptools came out mixed).

## 3. Validation runs (about 2 GPU-hours in total; run them in parallel)

These check, in increasing order of cost, that every part works before the long runs. All four
can be submitted at once; the cluster runs 4 of your jobs at a time.

```bash
cd ~/work/guardstack
# 3.1 small agent, in-process guards only (1 GPU, ~15 min)
sbatch slurm/smoke.sbatch
# 3.2 the two real agents, same stacks (1 GPU each, ~20-30 min)
AGENT_MODEL=meta-llama/Llama-3.3-70B-Instruct \
AGENT_VLLM_ARGS="--enable-auto-tool-choice --tool-call-parser llama3_json --max-model-len 32768 --quantization fp8" \
AGENT_MEM=0.60 sbatch slurm/smoke.sbatch
AGENT_MODEL=openai/gpt-oss-120b \
AGENT_VLLM_ARGS="--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768" \
AGENT_MEM=0.55 sbatch slurm/smoke.sbatch
# 3.3 every guard layer and guard server, gpt-oss-120b agent, 3% sample (2 GPUs, ~15 min)
RUN_ID=guardsmoke_gptoss SAMPLE=0.03 NUM_SHARDS=1 STACKS=shadow_all \
  sbatch --array=0 --time=06:00:00 slurm/run_stack.sbatch
# 3.4 attacker check: does the attacker model write injections or refuse? (1 GPU, ~20 min each)
sbatch slurm/attacker_check.sbatch
ATTACKER_MODEL=meta-llama/Llama-3.3-70B-Instruct ATTACKER_ARGS="--quantization fp8" sbatch slurm/attacker_check.sbatch
```
Summaries: `python scripts/summarize_run.py runs/<RUN_ID>` (smoke runs are named `smoke_<jobid>`).

Expected:
- **3.1/3.2**: three stacks (`A_bare`, `smoke_shadow`, `smoke_guards`), 57 sessions each on the 20%
  sample, no session errors, no guard errors. Server ready in about 50 s (8B), 150 s (70B FP8) and
  2–3 min (gpt-oss-120b). On our runs the unguarded agents had these attack success rates on this sample:
  gpt-oss-120b 38/50, Llama 3.3 70B 28–29/50. With guards enforcing, attack success was 0/50.
- **3.3**: all three guard servers start on GPU1 (Llama Guard 4, gpt-oss-20b, NemoGuard) in under
  4 min in total. Every one of the 13 layers is called and logs scores. Guard errors stay at or near
  zero (`grep guard_error runs/guardsmoke_gptoss/shadow_all/*.jsonl`).
  `grep -h "KV cache size" logs/<jobid>/vllm_*.log` shows each server's cache.
- **3.4**: JSON with `refusal_rate`. Mistral Small 3.2 refused 0/40 and Llama 3.3 70B 8/40, so the
  attacker is Mistral Small 3.2 (FP8) in every adaptive and multi-turn job.

## 4. The experiments

Every phase is submitted with one command per agent model:
```bash
bash slurm/submit_path_a.sh <gptoss|llama70> <phase>
```
`RUN_ID` defaults to `pathA_<model>`. Static results go to `runs/pathA_<model>/<stack>/`; the other
phases write to their own run folders so their records never mix with the static ones:
`pathA_<model>_adaptive`, `_staged`, `_reattack`, `_reattack_adaptive`, `_multiturn`, `_lat`.
Phases, in order:

| # | Phase | Runs in the paper | What it does | Jobs | Time (wall) |
|---|---|---|---|---|---|
| 1 | `static` | 1–4 | 19 stacks on all 1,046 AgentDojo cases (949 attack + 97 benign): `A_bare`, `shadow_all` (13 layers watching), each layer alone enforcing (`iso_*`), nested stacks B–E | 4 shards, 2 at a time | ~8–10 h per model |
| 2 | `analyze` | — | verdict matrix, MBR, redundancy, selection; then `python scripts/export_stacks.py` writes `configs/stacks_optimized_<model>.yaml` for phase 5 | 1 CPU job | < 1 h |
| 3 | `sweep` | guard-only sweep | every guard on the direct-attack and benign datasets (no agent; run once, not per model) | 1 GPU | < 12 h |
| 4 | `adaptive`, `staged` | 7 + staged attacker | LLM rewriter with trace feedback, budget 10, 30% sample; staged attacker on stacks C–E | 4 shards each | ~7 h per model (estimate from v1) |
| 5 | `reattack` | 6 | the cost-optimal stacks from phase 2, static and adaptive | 4 shards | < 1 day |
| 6 | `latency` | 9 | 10% sample, one session at a time (clean timings) | 1 job | < 1 day |
| 7 | `multiturn` | 8 | Crescendo-style escalation with backtracking against 6 stacks | 2 jobs | < 1 day |

Both models can run side by side. With 2 running shards each, they use all 4 of your job slots.
Examples:
```bash
bash slurm/submit_path_a.sh gptoss static;  bash slurm/submit_path_a.sh llama70 static
TASKS=1 bash slurm/submit_path_a.sh gptoss static      # (re)submit only shard 1
SHARDS=8 bash slurm/submit_path_a.sh gptoss static     # more, smaller shards
```

**After each phase, run a repair pass.** Resume skips every session that already has an error-free
record and reruns only the missing or errored ones (guard errors, server errors, timeouts). To skip
stacks with nothing to redo, pass only the stacks that have errored cases:
```bash
module load miniconda; eval "$(conda shell.bash hook)"; conda activate ~/work/envs/guardstack
STACKS="$(python scripts/summarize_run.py runs/pathA_gptoss --errored-stacks)" bash slurm/submit_path_a.sh gptoss static
```
Sessions that fail the same way twice are agent-side failures (e.g. the agent wrote invalid JSON in
a tool call), and the analysis reports them as such.

**Gating a phase on a smoke run (useful overnight).** `slurm/gate.sbatch` fails unless every result
folder of a run has records and at most `MAX_ERR` (default 5%) errored cases. Jobs submitted with
`DEPEND=afterok:<gate job>` start only if the gate passed:
```bash
A=$(RUN_ID=smoke_gptoss SAMPLE=0.02 SHARDS=1 TASKS=0 STACKS="A_bare iso:pg2_tool E" \
      bash slurm/submit_path_a.sh gptoss adaptive | awk '/Submitted batch job/{print $4}')
G=$(RUN_ID=smoke_gptoss_adaptive sbatch --parsable --dependency=afterany:$A slurm/gate.sbatch)
DEPEND=afterok:$G bash slurm/submit_path_a.sh gptoss adaptive
```
If the gate fails, the dependent jobs stay pending with reason `DependencyNeverSatisfied`; read
`logs/gate_<id>.out`, then `scancel` them.

**Adaptive attacker settings.** `FEEDBACK` and `BUDGET` set what the attacker sees between attempts
and how many attempts it gets; results go to `<stack>/adaptive_<attack>_<FEEDBACK>_b<BUDGET>/`, so
settings never mix. Default: `FEEDBACK=trace BUDGET=10` (after each attempt: did it succeed, which
guards flagged it, which tools the agent called and which were blocked, how many calls served the
target action, how the final answer began). The weak-attacker ablation in the paper is
`FEEDBACK=guard BUDGET=5` (only success and which guards flagged it):
```bash
FEEDBACK=guard BUDGET=5 bash slurm/submit_path_a.sh gptoss adaptive
```
Attacker token counts per shard are in `attacker_stats_shardNNN.json`.

## 5. Monitoring, resuming, collecting results

- `bash slurm/status.sh`: what runs, what waits and why, and per-stack progress with ETA.
- `squeue --me --start`: SLURM's start estimates. They are only meaningful for reason `Priority` or
  `Resources`; `Dependency` and `QOSMaxJobsPerUserLimit` jobs start when another of your jobs ends.
- A job that stops printing progress for much longer than its ETA is stuck. Cancel it with
  `scancel <jobid>_<task>` and resubmit that task with `TASKS=<task>`. Nothing finished is lost.
- Errors per stack: `python scripts/summarize_run.py runs/pathA_gptoss`.
- Copy results to a laptop without the large per-event files:
  `rsync -av --exclude '*.events.jsonl' <cluster>:~/work/guardstack/runs <local>/`.
  The `.events.jsonl` files hold the per-layer verdicts and error messages, and analysis needs them,
  so keep them on the cluster.

## 6. Troubleshooting (every failure we hit, and its fix, already in the code)

| Symptom | Cause | Fix / what to do |
|---|---|---|
| `sbatch: error: Unable to open file slurm/...` | Not in the repo directory | `cd ~/work/guardstack` first |
| Adaptive records with `ParserError` / `ScannerError` ("while parsing a block mapping") | Payload broke AgentDojo's environment YAML (a `"` or `\` inside a double-quoted slot); fixed Nov 9 by `scoring.yaml_safe_injections` | Rerun on current code with a new `RUN_ID` (resume skips cases already in the histories file). The gate now counts errored attempts, not cases. |
| `Intel oneMKL FATAL ERROR: Cannot load libtorch_cpu.so` on the login node | Per-user memory cap on login nodes | Run anything that imports torch in a job (`verify_env.sbatch`, `srun`) |
| Model looks downloaded but has only small files | Old size check | `bash slurm/download_models.sh check` (counts shards); `diag` for causes |
| `IncompleteSnapshotError` when loading offline | huggingface_hub ≥ 1.3 refuses snapshots with skipped files (`original/*`) | Code resolves the snapshot folder directly; models are served from that folder |
| vLLM dies at startup: `Could not find nvcc` (FlashInfer) | FlashInfer JIT-compiles its sampler; nodes have no nvcc | `VLLM_USE_FLASHINFER_SAMPLER=0` is exported by `serve_models.sh` |
| `vLLM on <port> exited during startup ... errors in <log>:` | Any server crash; the job stops within ~10 s and prints the cause | Read the printed error and the log named there |
| `No module named 'distutils.compilers.errors'` | Env packed with conda-pack | Re-run `pack_envs.sbatch` (plain tar) |
| `cannot import name 'AutoModelForSequenceClassification'`, or CUDA OOM on GPU1 at warmup | 16+ worker threads each loading their own classifier copies | Models are loaded once per process and shared (`shared_load`); a pipeline is warmed before workers start |
| `This model only supports single tool-calls at once!` (Llama agents) | Llama 3.x chat template rejects multi-call turns in the history | Such turns are re-sent as one call per turn (`split_parallel_tool_calls`) |
| AlignmentCheck / NeMo self-check: empty answer or cut-off JSON | The judge (gpt-oss-20b) reasons before answering and ran out of tokens | Budgets raised to 4096 tokens; AlignmentCheck retries once at 8192; repeated failure = guard error |
| NeMo: `Stale event loop ... bound to a different event loop`, `list.remove(x): x not in list`, or a job that hangs | NeMo's async client used from several event loops | All NeMo layers share one loop and one LLMRails per config; every call has a 300 s timeout |
| CodeShield: `UnicodeDecodeError ... 0xe2` | CodeShield cannot decode some non-ASCII text | One retry with typographic punctuation mapped to ASCII (`ascii_fallback` in the verdict) |
| `Event loop is closed` tracebacks | CodeShield subprocesses outliving a per-call loop | CodeShield runs on one long-lived loop |
| Logs of many MB per job (NeMo events, `HTTP Request` lines) | Library INFO logging | `guardstack/logsetup.py` filters them at the handler |
| Jobs wait with `QOSMaxJobsPerUserLimit` | the cluster allows 4 running jobs per user | Expected; they start as others finish |

## 7. Reference

### Scripts
| Script | Purpose |
|---|---|
| `slurm/probe.sbatch` | GPU/driver/network/storage probe |
| `slurm/setup_env.sh` | builds both conda envs (login node) |
| `slurm/verify_env.sbatch` | GPU check of both envs + tests |
| `slurm/download.sbatch` → `slurm/download_models.sh` → `scripts/hf_models.py` | weights (tiers 1–3), datasets, `check`, `diag` |
| `slurm/merge_nemoguard.sbatch` → `scripts/merge_lora.py` | NemoGuard LoRA merge |
| `slurm/pack_envs.sbatch` | tar of the vLLM env for the RAM disk |
| `slurm/serve_models.sh` | starts the agent and guard vLLM servers inside a job (sourced by every GPU job) |
| `slurm/smoke.sbatch`, `slurm/attacker_check.sbatch` | validation |
| `slurm/run_stack.sbatch` → `scripts/run_stack.py` | static stacks (also latency mode) |
| `slurm/run_adaptive.sbatch` → `scripts/run_adaptive.py` | adaptive and staged attacks |
| `slurm/run_multiturn.sbatch` → `scripts/run_multiturn.py` | multi-turn escalation |
| `slurm/guard_sweep.sbatch` → `scripts/guard_sweep.py` | guard-only sweep |
| `slurm/analyze.sbatch` → `scripts/analyze.py`, `analyze_adaptive.py`, `analyze_multiturn.py` | analysis |
| `slurm/submit_path_a.sh` | one command per phase and model |
| `slurm/status.sh` | job overview with progress |
| `slurm/gate.sbatch` | pass/fail check of a run, for `afterok` dependencies |
| `scripts/summarize_run.py` | per-stack summary; `--errored-stacks` (repair list); `--check` (gate) |

### Environment variables (set before `sbatch`, or exported by `submit_path_a.sh`)
| Variable | Meaning (default) |
|---|---|
| `RUN_ID` | results folder under `runs/` (`pathA_<model>`; smoke: `smoke_<jobid>`) |
| `STACKS` | stacks to run in order; `iso:<layer>` = that layer alone, enforcing |
| `STACK_CONFIG` | stack/layer definitions (`configs/stacks.yaml`) |
| `NUM_SHARDS`, `TASKS` | number of case shards; which array tasks to submit (`submit_path_a.sh`) |
| `DEPEND` | SLURM dependency for the submitted jobs, e.g. `afterok:<jobid>` (`submit_path_a.sh`) |
| `WORKERS` | concurrent sessions per job (static 24, adaptive 8, multi-turn 6) |
| `SAMPLE` | fraction of cases (stratified; 1.0 static, 0.3 adaptive, 0.1 latency) |
| `MODE` | `throughput` or `latency` (latency forces 1 worker) |
| `AGENT_MODEL`, `AGENT_VLLM_ARGS`, `AGENT_MEM`, `AGENT_GPUS` | agent server |
| `GUARD_MODELS` | extra servers: `"model|gpu|port|mem_fraction|extra args; ..."` |
| `GUARD_MODEL`, `LLAMAGUARD_MODEL`, `NEMOGUARD_MODEL` | judge (gpt-oss-20b), Llama Guard 4, merged NemoGuard |
| `ATTACKER_MODEL`, `ATTACKER_ARGS` | attacker server (Mistral Small 3.2, FP8) |
| `VLLM_PACK`, `VLLM_BIN` | packed env to unpack (`~/work/envs/vllm.tar`); setting `VLLM_BIN` skips unpacking |
| `HF_HOME` | HF cache (`~/work/.cache/huggingface`); jobs run with `HF_HUB_OFFLINE=1` |

### GPU layout of a static-phase job (`run_stack.sbatch`, 2× H200 NVL)
| GPU | Server (port) | Memory fraction | Measured KV cache |
|---|---|---|---|
| 0 | agent: gpt-oss-120b (8000) | 0.90 | 61 GiB, 1.19M tokens |
| 0 | agent: Llama 3.3 70B FP8 (8000) | 0.90 | — |
| 1 | Llama Guard 4 12B (8001) | 0.35 | 13 GiB at 0.30 (raised to 0.35) |
| 1 | gpt-oss-20b judge for AlignmentCheck and NeMo self-check (8002) | 0.30 | 33 GiB at 0.35 |
| 1 | NemoGuard 8B, merged (8005) | 0.18 | 7.8 GiB, 63k tokens |
| 1 | in-process: PromptGuard 2, DeBERTa, GLiNER, Presidio, CodeShield | ~24 GiB free | — |

Adaptive and multi-turn jobs put the attacker (port 8003) and, for multi-turn, the HarmBench judge
(port 8004) on GPU1 instead of NemoGuard. Their memory split is in those `.sbatch` files.
