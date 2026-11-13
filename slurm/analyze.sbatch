#!/bin/bash
# CPU-only analysis of a finished shadow_all run (subset enumeration is 2^L, seconds for L<=16).
#   RUN_ID=12345 sbatch slurm/analyze.sbatch
#SBATCH --job-name=gs_analyze
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
PROJ=${PROJ:-$HOME/work/guardstack}
cd "$PROJ"
module load miniconda
eval "$(conda shell.bash hook)"
conda activate "${ENV:-$HOME/work/envs/guardstack}"
: "${RUN_ID:?set RUN_ID}"
# analyze.py reads the shard files itself and skips *.events.jsonl; latency-mode records (if any)
# supply the reported per-layer costs.
LAT=()
if [ -n "$LATENCY_RUN_ID" ] && [ -d runs/"$LATENCY_RUN_ID" ]; then LAT=(--latency-records runs/"$LATENCY_RUN_ID")
elif [ -n "$LATENCY_RUN_ID" ]; then echo "note: runs/$LATENCY_RUN_ID not there yet; costs come from the throughput records"; fi
python scripts/analyze.py --records runs/"$RUN_ID"/shadow_all "${LAT[@]}" --out results/"$RUN_ID"/analysis
