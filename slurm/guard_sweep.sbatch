#!/bin/bash
# Guard-only sweep (no agent): every layer of a stack scored on the vetted datasets. One H200.
#   RUN_ID=sweep_v1 sbatch slurm/guard_sweep.sbatch
#SBATCH --job-name=gs_sweep
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
: "${RUN_ID:?set RUN_ID}"
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
module load miniconda; eval "$(conda shell.bash hook)"; conda activate "${ENV:-$HOME/work/envs/guardstack}"
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$PROJ
export LLAMAGUARD_MODEL=${LLAMAGUARD_MODEL:-meta-llama/Llama-Guard-4-12B} GUARD_MODEL=${GUARD_MODEL:-openai/gpt-oss-20b}
# the "agent" slot serves Llama Guard here; the judge goes next to it on the same GPU
export AGENT_MODEL=$LLAMAGUARD_MODEL AGENT_VLLM_ARGS="--max-model-len 32768" AGENT_GPUS=0 AGENT_MEM=0.35 AGENT_PORT=8001
export NEMOGUARD_MODEL=${NEMOGUARD_MODEL:-$HOME/work/models/llama-3.1-nemoguard-8b-content-safety-merged}
NEMOGUARD_SPEC=""
[ -f "$NEMOGUARD_MODEL/config.json" ] && NEMOGUARD_SPEC="; $NEMOGUARD_MODEL|0|8005|0.18|--served-model-name llama-3.1-nemoguard-8b-content-safety --max-model-len 16384" \
  || echo "WARNING: $NEMOGUARD_MODEL not found (run slurm/merge_nemoguard.sbatch); nemo_out will stop at warmup"
export GUARD_MODELS=${GUARD_MODELS:-"$GUARD_MODEL|0|8002|0.30$NEMOGUARD_SPEC"}
export LLAMAGUARD_URL=http://127.0.0.1:8001/v1 GUARD_URL=http://127.0.0.1:8002/v1 NEMOGUARD_URL=http://127.0.0.1:8005/v1
source slurm/serve_models.sh
export CUDA_VISIBLE_DEVICES=0
python scripts/guard_sweep.py --data-root "${DATA_ROOT:-$HOME/work/guardstack_data}" --stack "${STACK:-shadow_all}" \
  --max-per-dataset "${MAX_PER_DATASET:-0}" --out results/"$RUN_ID"/guard_sweep
