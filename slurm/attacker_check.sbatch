#!/bin/bash
# Refusal-rate check for an attacker candidate (one GPU, ~20 min).
#   sbatch slurm/attacker_check.sbatch                                   # Mistral Small 3.2 (FP8)
#   ATTACKER_MODEL=meta-llama/Llama-3.3-70B-Instruct ATTACKER_ARGS="--quantization fp8" sbatch slurm/attacker_check.sbatch
#   ATTACKER_MODEL=openai/gpt-oss-120b ATTACKER_ARGS="" sbatch slurm/attacker_check.sbatch
#SBATCH --job-name=gs_attacker_check
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/attacker_check_%j.out
#SBATCH --error=logs/attacker_check_%j.out
set -eo pipefail
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
module load miniconda; eval "$(conda shell.bash hook)"; conda activate "${ENV:-$HOME/work/envs/guardstack}"
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$PROJ
export AGENT_MODEL=${ATTACKER_MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}
export AGENT_VLLM_ARGS=${ATTACKER_ARGS-"--tokenizer-mode mistral --config-format mistral --load-format mistral --quantization fp8"}
export AGENT_VLLM_ARGS="$AGENT_VLLM_ARGS --max-model-len 16384" AGENT_GPUS=0 AGENT_MEM=0.85 GUARD_MODELS=""
source slurm/serve_models.sh
python scripts/attacker_check.py --attacker-model "$AGENT_MODEL" --attacker-url "$GUARDSTACK_AGENT_URL" --n ${N:-40}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
