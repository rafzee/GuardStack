#!/bin/bash
# Merge the NemoGuard 8B content-safety LoRA into Llama-3.1-8B-Instruct (CPU job, ~20 min).
# Needs both in the HF cache (tiers 1 and 2 of slurm/download_models.sh). Output (~16 GB):
#   ~/work/models/llama-3.1-nemoguard-8b-content-safety-merged   (served by run_stack / guard_sweep)
#   sbatch slurm/merge_nemoguard.sbatch
#SBATCH --job-name=gs_merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.out
set -eo pipefail
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
module load miniconda; eval "$(conda shell.bash hook)"; conda activate "${ENV:-$HOME/work/envs/guardstack}"
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface} HF_HUB_OFFLINE=1 PYTHONPATH=$PROJ
OUT=${OUT:-$HOME/work/models/llama-3.1-nemoguard-8b-content-safety-merged}
python scripts/hf_models.py check meta-llama/Llama-3.1-8B-Instruct nvidia/llama-3.1-nemoguard-8b-content-safety
python scripts/merge_lora.py --base meta-llama/Llama-3.1-8B-Instruct \
  --adapter nvidia/llama-3.1-nemoguard-8b-content-safety --out "$OUT"
python -c "import sys; from guardstack.llm import assert_allowed; assert_allowed(sys.argv[1]); print('allowed:', sys.argv[1])" "$OUT"
ls -la "$OUT"; du -sh "$OUT"
