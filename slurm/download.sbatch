#!/bin/bash
# Model + dataset downloads as a CPU batch job (the probe showed compute nodes have internet).
# Needs the HF token saved once beforehand (after Step 3):
#   module load miniconda; eval "$(conda shell.bash hook)"; conda activate ~/work/envs/guardstack
#   HF_HOME=~/work/.cache/huggingface hf auth login        # paste the read token
# Then:
#   sbatch slurm/download.sbatch                 # tiers 1+2+3 and datasets
#   TIERS="1" sbatch slurm/download.sbatch       # only the smoke-test models
#   TIERS="" sbatch slurm/download.sbatch        # datasets only
#SBATCH --job-name=gs_download
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.out
set -eo pipefail
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
module load miniconda; eval "$(conda shell.bash hook)"; conda activate "${ENV:-$HOME/work/envs/guardstack}"
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface} PYTHONPATH=$PROJ
export TMPDIR=$HOME/scratch/tmp_dl_$SLURM_JOB_ID; mkdir -p "$TMPDIR"   # /tmp on nodes is a RAM disk
trap 'rm -rf "$TMPDIR"' EXIT
[ -f "$HF_HOME/token" ] || [ -n "$HF_TOKEN" ] || echo "WARNING: no HF token found; gated repos will fail"
for t in ${TIERS-1 2 3}; do   # TIERS="" -> datasets only
  echo "===== tier $t  $(date +%T)"
  bash slurm/download_models.sh "$t" || echo "tier $t had failures (see above)"
done
if [ -z "$SKIP_DATA" ]; then
  echo "===== datasets  $(date +%T)"
  python data/download_datasets.py --out "$HOME/work/guardstack_data" --skip-gated || echo "ungated datasets had failures"
  python data/download_datasets.py --out "$HOME/work/guardstack_data" || echo "gated datasets had failures (access pending?)"
fi
echo "===== check  $(date +%T)"
bash slurm/download_models.sh check
du -sh "$HF_HOME" "$HOME/work/guardstack_data" 2>/dev/null || true
