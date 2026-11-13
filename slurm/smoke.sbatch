#!/bin/bash
# GPU smoke test on ONE H200: small agent (Llama 3.1 8B), in-process guards only.
# Checks: vLLM starts offline, tool calling parses, AgentDojo scores, guards load, records are sane.
#   sbatch slurm/smoke.sbatch            -> send me logs/smoke_<jobid>.out
#   AGENT_MODEL=openai/gpt-oss-120b AGENT_VLLM_ARGS="--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768" sbatch slurm/smoke.sbatch
#SBATCH --job-name=gs_smoke
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.out
set -eo pipefail
PROJ=${PROJ:-$HOME/work/guardstack}
cd "$PROJ"
module load miniconda
eval "$(conda shell.bash hook)"
conda activate "${ENV:-$HOME/work/envs/guardstack}"
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PROJ:$PYTHONPATH
export AGENT_MODEL=${AGENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
export AGENT_VLLM_ARGS=${AGENT_VLLM_ARGS:-"--enable-auto-tool-choice --tool-call-parser llama3_json --max-model-len 32768"}
export AGENT_GPUS=0 AGENT_MEM=${AGENT_MEM:-0.60} GUARD_MODELS=""
RUN_ID=${RUN_ID:-smoke_${SLURM_JOB_ID}}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('harness env: torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available())"
"$HOME/work/envs/vllm/bin/python" -c "import torch, vllm; print('vllm env: vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available())"
source slurm/serve_models.sh
export CUDA_VISIBLE_DEVICES=0
for st in A_bare smoke_shadow smoke_guards; do
  python scripts/run_stack.py --stack "$st" --model "$AGENT_MODEL" --agent-url "$GUARDSTACK_AGENT_URL" \
    --suites banking slack --sample ${SAMPLE:-0.2} --workers 8 --run-id "$RUN_ID" 2>&1 | grep -v "HTTP Request" | tail -5
done
python scripts/summarize_run.py runs/"$RUN_ID"
echo "== vLLM log tail"; tail -5 "$LOGDIR"/vllm_agent_*.log
