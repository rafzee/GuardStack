#!/bin/bash
# Pass/fail gate between jobs: fails (so `afterok` dependents never start) unless every result
# folder of RUN_ID has records and at most MAX_ERR of its cases errored.
#   G=$(RUN_ID=smoke_adaptive_gptoss sbatch --parsable --dependency=afterany:<smoke job> slurm/gate.sbatch)
#   DEPEND=afterok:$G bash slurm/submit_path_a.sh gptoss adaptive
#SBATCH --job-name=gs_gate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gate_%j.out
#SBATCH --error=logs/gate_%j.out
set -eo pipefail
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
: "${RUN_ID:?set RUN_ID}"
module load miniconda; eval "$(conda shell.bash hook)"; conda activate "${ENV:-$HOME/work/envs/guardstack}"
python scripts/summarize_run.py "runs/$RUN_ID" --check --max-error-rate "${MAX_ERR:-0.05}"
