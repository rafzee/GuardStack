#!/bin/bash
# Multi-turn escalation (run 8) on one h200 node.
# GPU0: agent. GPU1: Llama Guard 4, judge (gpt-oss-20b), attacker (FP8), HarmBench classifier.
#   RUN_ID=gptoss_v1 STACKS="A_bare C E" sbatch slurm/run_multiturn.sbatch
#SBATCH --job-name=gs_multiturn
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
: "${RUN_ID:?set RUN_ID}"; : "${STACKS:?set STACKS}"
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
module load miniconda; eval "$(conda shell.bash hook)"; conda activate "${ENV:-$HOME/work/envs/guardstack}"
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$PROJ
export AGENT_MODEL=${AGENT_MODEL:-openai/gpt-oss-120b}
export AGENT_VLLM_ARGS=${AGENT_VLLM_ARGS:-"--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768"}
export AGENT_GPUS=0
export LLAMAGUARD_MODEL=${LLAMAGUARD_MODEL:-meta-llama/Llama-Guard-4-12B} GUARD_MODEL=${GUARD_MODEL:-openai/gpt-oss-20b}
export ATTACKER_MODEL=${ATTACKER_MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}
export JUDGE_MODEL=${JUDGE_MODEL:-cais/HarmBench-Llama-2-13b-cls}
ATTACKER_ARGS=${ATTACKER_ARGS:-"--tokenizer-mode mistral --config-format mistral --load-format mistral --quantization fp8"}
# first-guess memory split on GPU1; the smoke tests tell us whether it fits
export GUARD_MODELS=${GUARD_MODELS:-"$LLAMAGUARD_MODEL|1|8001|0.20; $GUARD_MODEL|1|8002|0.18; $ATTACKER_MODEL|1|8003|0.25|$ATTACKER_ARGS; $JUDGE_MODEL|1|8004|0.25|--max-model-len 4096"}
export LLAMAGUARD_URL=http://127.0.0.1:8001/v1 GUARD_URL=http://127.0.0.1:8002/v1
export ATTACKER_URL=http://127.0.0.1:8003/v1 JUDGE_URL=http://127.0.0.1:8004/v1
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
source slurm/serve_models.sh
PY_PID=
on_usr1() { [ -n "$PY_PID" ] && kill -INT "$PY_PID" 2>/dev/null && wait "$PY_PID" 2>/dev/null; stop_models; exit 0; }
trap on_usr1 USR1
export CUDA_VISIBLE_DEVICES=1
for item in $STACKS; do
  if [[ "$item" == iso:* ]]; then SEL=(--isolated "${item#iso:}"); else SEL=(--stack "$item"); fi
  python scripts/run_multiturn.py --stack-config "${STACK_CONFIG:-configs/stacks.yaml}" "${SEL[@]}" \
    --model "$AGENT_MODEL" --agent-url "$GUARDSTACK_AGENT_URL" --attacker-model "$ATTACKER_MODEL" --attacker-url "$ATTACKER_URL" \
    --judge-model "$JUDGE_MODEL" --judge-url "$JUDGE_URL" --harmbench-dir "${DATA_ROOT:-$HOME/work/guardstack_data}/harmbench" \
    --repeats "${REPEATS:-5}" --max-turns "${MAX_TURNS:-8}" --max-backtracks "${MAX_BACKTRACKS:-6}" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS" --workers "${WORKERS:-6}" --run-id "$RUN_ID" $EXTRA_ARGS &
  PY_PID=$!
  wait $PY_PID || wait $PY_PID || echo "WARNING: ${SEL[*]} exited non-zero; resubmit to resume"
done
