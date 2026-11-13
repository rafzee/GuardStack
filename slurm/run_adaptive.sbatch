#!/bin/bash
# Adaptive / STACK-style attacks (run 7, run 6 re-attack) on one h200 node.
# GPU0: agent. GPU1: Llama Guard 4 + judge (gpt-oss-20b) + attacker (FP8), started one at a time.
#   RUN_ID=gptoss_v1 STACKS="A_bare B C D E" ATTACK=rewrite sbatch --array=0-3%2 slurm/run_adaptive.sbatch
#   RUN_ID=gptoss_v1 STACKS="E" ATTACK=staged sbatch --array=0-3%2 slurm/run_adaptive.sbatch
#SBATCH --job-name=gs_adaptive
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=340G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=B:USR1@600
set -eo pipefail
: "${RUN_ID:?set RUN_ID}"; : "${STACKS:?set STACKS (space-separated stack ids, or iso:<layer>)}"
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
module load miniconda; eval "$(conda shell.bash hook)"; conda activate "${ENV:-$HOME/work/envs/guardstack}"
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PROJ:$PYTHONPATH
export AGENT_MODEL=${AGENT_MODEL:-openai/gpt-oss-120b}
export AGENT_VLLM_ARGS=${AGENT_VLLM_ARGS:-"--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768"}
export AGENT_GPUS=0
export LLAMAGUARD_MODEL=${LLAMAGUARD_MODEL:-meta-llama/Llama-Guard-4-12B}
export GUARD_MODEL=${GUARD_MODEL:-openai/gpt-oss-20b}
export ATTACKER_MODEL=${ATTACKER_MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}
# GPU1: Llama Guard 4 (25 GB weights) 0.30 + gpt-oss-20b (13 GB) 0.22 + Mistral Small 3.2 FP8 (25 GB) 0.33
# = 0.85. From the static jobs: Llama Guard 4 has the tightest KV cache (13 GiB at 0.30) and gpt-oss-20b
# needs far less than it had (33 GiB at 0.35); ~20 GiB stays for the in-process classifiers.
ATTACKER_ARGS=${ATTACKER_ARGS:-"--tokenizer-mode mistral --config-format mistral --load-format mistral --quantization fp8"}
export GUARD_MODELS=${GUARD_MODELS:-"$LLAMAGUARD_MODEL|1|8001|0.30; $GUARD_MODEL|1|8002|0.22; $ATTACKER_MODEL|1|8003|0.33|$ATTACKER_ARGS"}
export GUARD_URL=http://127.0.0.1:8002/v1 LLAMAGUARD_URL=http://127.0.0.1:8001/v1 ATTACKER_URL=http://127.0.0.1:8003/v1
ATTACK=${ATTACK:-rewrite}; BUDGET=${BUDGET:-10}; FEEDBACK=${FEEDBACK:-trace}; SAMPLE=${SAMPLE:-0.3}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
source slurm/serve_models.sh
PY_PID=
on_usr1() { [ -n "$PY_PID" ] && kill -INT "$PY_PID" 2>/dev/null && wait "$PY_PID" 2>/dev/null; stop_models; exit 0; }
trap on_usr1 USR1
export CUDA_VISIBLE_DEVICES=1
for item in $STACKS; do
  if [[ "$item" == iso:* ]]; then SEL=(--isolated "${item#iso:}"); else SEL=(--stack "$item"); fi
  echo "== $(date +%T) ${SEL[*]} attack=$ATTACK"
  python scripts/run_adaptive.py --stack-config "${STACK_CONFIG:-configs/stacks.yaml}" "${SEL[@]}" --attack "$ATTACK" --budget "$BUDGET" --feedback "$FEEDBACK" \
    --sample "$SAMPLE" --model "$AGENT_MODEL" --agent-url "$GUARDSTACK_AGENT_URL" \
    --attacker-model "$ATTACKER_MODEL" --attacker-url "$ATTACKER_URL" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" --workers "${WORKERS:-8}" --run-id "$RUN_ID" $EXTRA_ARGS &
  PY_PID=$!
  wait $PY_PID || wait $PY_PID || echo "WARNING: ${SEL[*]} exited non-zero; resubmit to resume"
done
