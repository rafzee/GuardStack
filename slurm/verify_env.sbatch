#!/bin/bash
# Verify both envs on a GPU compute node (the login node's memory cap can't load torch).
#   sbatch slurm/verify_env.sbatch   ->  send me logs/verify_env_<jobid>.out
#SBATCH --job-name=gs_verify
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/verify_env_%j.out
#SBATCH --error=logs/verify_env_%j.out
PROJ=${PROJ:-$HOME/work/guardstack}; cd "$PROJ"
module load miniconda; eval "$(conda shell.bash hook)"
export XDG_CACHE_HOME=$HOME/work/.cache PIP_CACHE_DIR=$HOME/work/.cache/pip
echo "== limits on this node"; ulimit -v; free -g | head -2

echo "== 1. harness env"
conda activate "$HOME/work/envs/guardstack"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import transformers, llamafirewall, nemoguardrails, presidio_analyzer, gliner; print('adapter libs import OK; transformers', transformers.__version__)"
echo "== 2. spaCy model for Presidio"
python -m spacy download en_core_web_lg 2>&1 | tail -2
echo "== 3. test suite"
PYTHONPATH=$PROJ python -m pytest -q tests/ 2>&1 | tail -4
conda deactivate

echo "== 4. vLLM env"
"$HOME/work/envs/vllm/bin/python" -c "import torch, vllm; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
echo "== done"
