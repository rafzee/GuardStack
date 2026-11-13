#!/bin/bash
# One-time setup. Run on a LOGIN node (light work only: env creation + downloads).
# Compute-node internet access is unverified, so every model and dataset is fetched here and
# jobs run with HF_HUB_OFFLINE=1.
#
#   bash slurm/setup_env.sh            # env + models + datasets
#   SKIP_MODELS=1 bash slurm/setup_env.sh
set -eo pipefail

PROJ=${PROJ:-$HOME/work/guardstack}
ENV=${ENV:-$HOME/work/envs/guardstack}          # harness + guard adapters
VLLM_ENV=${VLLM_ENV:-$HOME/work/envs/vllm}      # model server only (incompatible torch/transformers pins)
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface}
export PIP_CACHE_DIR=$HOME/work/.cache/pip CONDA_PKGS_DIRS=$HOME/work/.cache/conda XDG_CACHE_HOME=$HOME/work/.cache
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

module load miniconda
eval "$(conda shell.bash hook)"
# env 2: vLLM server
if [ ! -d "$VLLM_ENV" ]; then conda create -y -p "$VLLM_ENV" python=3.12; fi
conda activate "$VLLM_ENV"
pip install -r "$PROJ/requirements/serve.txt"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
conda deactivate

# env 1: harness + adapters
if [ ! -d "$ENV" ]; then conda create -y -p "$ENV" python=3.12; fi
conda activate "$ENV"
pip install -r "$PROJ/requirements/base.txt"
pip install -r "$PROJ/requirements/adapters.txt"
# spaCy model + tests need torch, which the login node's memory cap can't load: they run in
# slurm/verify_env.sbatch on a compute node instead.

# Models: refuse anything from a banned entity before downloading it.
MODELS=${MODELS:-"meta-llama/Llama-3.3-70B-Instruct meta-llama/Llama-3.1-8B-Instruct openai/gpt-oss-120b \
mistralai/Mistral-Small-3.2-24B-Instruct-2506 openai/gpt-oss-20b meta-llama/Llama-Guard-4-12B meta-llama/Llama-Prompt-Guard-2-86M \
protectai/deberta-v3-base-prompt-injection-v2"}
if [ -z "$SKIP_MODELS" ]; then
  [ -z "$HF_TOKEN" ] && echo "WARNING: HF_TOKEN not set; gated Llama models will fail. export HF_TOKEN=..." >&2
  for m in $MODELS; do
    PYTHONPATH=$PROJ python -c "import sys; from guardstack.llm import assert_allowed; assert_allowed(sys.argv[1])" "$m"
    if command -v hf > /dev/null; then DL=(hf download); else DL=(huggingface-cli download); fi
    "${DL[@]}" "$m" --exclude "original/*" || echo "FAILED: $m" >&2
  done
fi

# Datasets (script written by the dataset work; pinned revisions + manifest)
if [ -z "$SKIP_DATA" ] && [ -f "$PROJ/data/download_datasets.py" ]; then
  python "$PROJ/data/download_datasets.py" --out "$HOME/work/guardstack_data" --skip-gated \
    || echo "WARNING: some datasets failed (see above); Path A core (AgentDojo) does not need them"
fi

echo "install finished. Next: sbatch slurm/verify_env.sbatch  (tests + GPU check on a compute node)"
