#!/bin/bash
# One stack (or one isolated layer) over AgentDojo, sharded as a job array.
# Layout on one h200 node: GPU0 = agent model, GPU1 = guard LLMs + in-process classifiers.
#
#   RUN_ID=gptoss_v1 STACK=shadow_all sbatch slurm/run_stack.sbatch
#   RUN_ID=gptoss_v1 STACKS="A_bare shadow_all iso:pg2_tool B C D E" sbatch slurm/run_stack.sbatch
#     (several stacks in one job: vLLM loads once; "iso:<key>" = that layer alone, enforced)
#   RUN_ID=gptoss_v1 ISOLATED=pg2_tool sbatch slurm/run_stack.sbatch
#   RUN_ID=gptoss_lat STACK=E MODE=latency SAMPLE=0.1 NUM_SHARDS=1 sbatch --array=0 slurm/run_stack.sbatch
# RUN_ID is required: resubmitting with the same RUN_ID resumes (finished cases are skipped).
#
# Cluster limits respected: <=4 running jobs per user (array throttle %4), 2-day cap (we ask 1 day;
# runs are resumable, so a timed-out shard just reruns and skips finished cases).
#SBATCH --job-name=gs_run
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=B:USR1@600
set -eo pipefail

PROJ=${PROJ:-$HOME/work/guardstack}
cd "$PROJ"
module load miniconda
eval "$(conda shell.bash hook)"
conda activate "${ENV:-$HOME/work/envs/guardstack}"
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PROJ:$PYTHONPATH

# ---- models (override from the sbatch command line with --export=ALL,AGENT_MODEL=...)
export AGENT_MODEL=${AGENT_MODEL:-openai/gpt-oss-120b}
export AGENT_VLLM_ARGS=${AGENT_VLLM_ARGS:-"--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768"}
export AGENT_GPUS=${AGENT_GPUS:-0}
export LLAMAGUARD_MODEL=${LLAMAGUARD_MODEL:-meta-llama/Llama-Guard-4-12B}
export GUARD_MODEL=${GUARD_MODEL:-openai/gpt-oss-20b}          # AlignmentCheck / NeMo self-check judge
# NemoGuard 8B content safety (nemo_out): merged LoRA from slurm/merge_nemoguard.sbatch, served under
# the name NeMo's config expects. Skipped if not merged yet; nemo_out then fails its warmup check.
export NEMOGUARD_MODEL=${NEMOGUARD_MODEL:-$HOME/work/models/llama-3.1-nemoguard-8b-content-safety-merged}
export NEMOGUARD_URL=${NEMOGUARD_URL:-http://127.0.0.1:8005/v1}
NEMOGUARD_SPEC=""
if [ -f "$NEMOGUARD_MODEL/config.json" ]; then
  NEMOGUARD_SPEC="; $NEMOGUARD_MODEL|1|8005|0.18|--served-model-name llama-3.1-nemoguard-8b-content-safety --max-model-len 16384"
else
  echo "WARNING: $NEMOGUARD_MODEL not found (run slurm/merge_nemoguard.sbatch); stacks with nemo_out will stop at warmup"
fi
# GPU1 budget: Llama Guard 4 (25 GB weights) 0.35 + gpt-oss-20b (13 GB) 0.30 + NemoGuard (15 GB) 0.18 = 0.83.
# Measured Nov 8 (job 502208) at 0.30/0.35: KV for LG4 was 13 GiB (71k tokens, 2.2 full-length
# requests) while gpt-oss-20b had 33 GiB (953k tokens); 0.05 moved to LG4. ~24 GiB stays free for
# the in-process classifiers.
export GUARD_MODELS=${GUARD_MODELS:-"$LLAMAGUARD_MODEL|1|8001|0.35; $GUARD_MODEL|1|8002|0.30$NEMOGUARD_SPEC"}
export GUARD_URL=${GUARD_URL:-http://127.0.0.1:8002/v1}
export LLAMAGUARD_URL=${LLAMAGUARD_URL:-http://127.0.0.1:8001/v1}

: "${RUN_ID:?set RUN_ID (same RUN_ID on resubmission resumes the run)}"
OUTDIR=${OUTDIR:-$PROJ/runs}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
MODE=${MODE:-throughput}
WORKERS=${WORKERS:-16}
SAMPLE=${SAMPLE:-1.0}

source slurm/serve_models.sh   # registers the EXIT trap that stops vLLM
PY_PID=
on_usr1() {  # walltime near: stop Python first (records are flushed per case), then vLLM
  echo "USR1: stopping; resubmit with the same RUN_ID to resume"
  [ -n "$PY_PID" ] && kill -INT "$PY_PID" 2>/dev/null && wait "$PY_PID" 2>/dev/null
  stop_models; exit 0
}
trap on_usr1 USR1

# in-process classifiers (PromptGuard, DeBERTa, GLiNER) share the guard GPU
export CUDA_VISIBLE_DEVICES=1
if [ -n "$STACKS" ]; then LIST=($STACKS); elif [ -n "$ISOLATED" ]; then LIST=("iso:$ISOLATED"); else LIST=("$STACK"); fi
for item in "${LIST[@]}"; do
  if [[ "$item" == iso:* ]]; then SEL=(--isolated "${item#iso:}"); else SEL=(--stack "$item"); fi
  echo "== $(date +%T) ${SEL[*]}"
  python scripts/run_stack.py --stack-config "${STACK_CONFIG:-configs/stacks.yaml}" "${SEL[@]}" --model "$AGENT_MODEL" --agent-url "$GUARDSTACK_AGENT_URL" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" --workers "$WORKERS" --mode "$MODE" \
    --sample "$SAMPLE" --run-id "$RUN_ID" --out-dir "$OUTDIR" $EXTRA_ARGS &
  PY_PID=$!
  wait $PY_PID || wait $PY_PID || echo "WARNING: ${SEL[*]} exited non-zero; continuing (resume will retry)"
done
