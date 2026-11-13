#!/bin/bash
# Pack the vLLM env into one tar (~20 GB) so jobs can unpack it onto the node's RAM disk
# (/tmp) instead of importing thousands of files from ~/work: see serve_models.sh. Re-run after
# any change to ~/work/envs/vllm. CPU job:
#   sbatch slurm/pack_envs.sbatch
#SBATCH --job-name=gs_pack
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pack_%j.out
#SBATCH --error=logs/pack_%j.out
set -eo pipefail
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
module load miniconda; eval "$(conda shell.bash hook)"
SRC=${SRC:-$HOME/work/envs/vllm}
OUT=${OUT:-$HOME/work/envs/vllm.tar}
du -sh "$SRC"
t0=$SECONDS
rm -f "$OUT.part" "$OUT"     # a failed pack then leaves no tar, and jobs fall back to ~/work
# A plain copy of the whole env. (conda-pack was tried first: it packs conda-managed packages from
# conda's own file lists, so the pip-upgraded setuptools came out half old, half new and vLLM failed
# with "No module named 'distutils.compilers.errors'". It also left out half of the 20 GB.)
tar -cf "$OUT.part" -C "$SRC" .
mv "$OUT.part" "$OUT"
echo "packed in $((SECONDS - t0))s: $(du -sh "$OUT" | cut -f1)"

# check it: unpack to the RAM disk and time the import from both places. (`vllm serve --help` cannot
# run here: it needs a GPU to infer the device. Each GPU job runs it on its own copy before use.)
D=/tmp/gs_packtest_$SLURM_JOB_ID; trap 'rm -rf "$D"' EXIT
t0=$SECONDS; mkdir -p "$D"; tar -xf "$OUT" -C "$D"
echo "unpack: $((SECONDS - t0))s"
for py in "$D/bin/python" "$SRC/bin/python"; do
  t0=$SECONDS
  "$py" -c "import vllm, torch, transformers; from vllm.entrypoints.cli.main import main; print('vllm', vllm.__version__)"
  echo "  import from $(dirname "$(dirname "$py")"): $((SECONDS - t0))s"
done
