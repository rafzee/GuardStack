#!/bin/bash
# Download model weights in priority tiers so smoke tests can start before the big models land.
# Normally run through slurm/download.sbatch (compute nodes have internet). By hand, in a job:
#   bash slurm/download_models.sh 1        # tier 1: ~20 GB, enough for the smoke test
#   bash slurm/download_models.sh check    # are the WEIGHT FILES there? (offline, seconds)
#   bash slurm/download_models.sh diag     # why downloads fail: token, CDN reachability, test file
# The real work is in scripts/hf_models.py; errors are printed, never sent to /dev/null.
set -eo pipefail
PROJ=${PROJ:-$HOME/work/guardstack}
export HF_HOME=${HF_HOME:-$HOME/work/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" "$PROJ/logs"

TIER1=(
  meta-llama/Llama-3.1-8B-Instruct                 # smoke-test agent; also the NemoGuard base (16 GB)
  meta-llama/Llama-Prompt-Guard-2-86M              # PromptGuard 2
  protectai/deberta-v3-base-prompt-injection-v2    # DeBERTa PI classifier
)
TIER2=(
  openai/gpt-oss-120b                              # agent 1 (~65 GB)
  meta-llama/Llama-3.3-70B-Instruct                # agent 2 (~141 GB bf16; served as FP8)
  meta-llama/Llama-Guard-4-12B                     # safety classifier (~24 GB)
  openai/gpt-oss-20b                               # AlignmentCheck / NeMo self-check judge (~14 GB)
  urchade/gliner_multi_pii-v1                      # GLiNER PII (no config.json: gliner_config.json)
  nvidia/llama-3.1-nemoguard-8b-content-safety     # nemo_out: LoRA adapter in a subfolder (~0.2 GB)
)
TIER3=(
  mistralai/Mistral-Small-3.2-24B-Instruct-2506    # reserve agent / attacker candidate
  cais/HarmBench-Llama-2-13b-cls                   # judge for direct-vector datasets
  allenai/wildguard                                # gated: request access on its HF page
  google/shieldgemma-2b                            # gated: accept the Gemma terms on its HF page
  ibm-granite/granite-guardian-3.3-8b
  openai/gpt-oss-safeguard-20b
)

module load miniconda 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${ENV:-$HOME/work/envs/guardstack}" 2>/dev/null || true
export PYTHONPATH=$PROJ:$PYTHONPATH
cd "$PROJ"

fetch() {
  local ok=()
  for m in "$@"; do
    if python -c "import sys; from guardstack.llm import assert_allowed; assert_allowed(sys.argv[1], check_local_config=False)" "$m"; then
      ok+=("$m"); else echo "BANNED by name, skipped: $m"; fi
  done
  local rc=0
  python scripts/hf_models.py fetch "${ok[@]}" || rc=$?
  for m in "${ok[@]}"; do   # second check now that config.json / README.md are on disk
    python -c "import sys; from guardstack.llm import assert_allowed; assert_allowed(sys.argv[1])" "$m" \
      || echo "BANNED after config check: $m -- delete it from \$HF_HOME"
  done
  return $rc
}

case "$1" in
  1) fetch "${TIER1[@]}" ;;
  2) fetch "${TIER2[@]}" ;;
  3) fetch "${TIER3[@]}" ;;
  check) python scripts/hf_models.py check "${TIER1[@]}" "${TIER2[@]}" "${TIER3[@]}" ;;
  diag)  python scripts/hf_models.py diag ;;
  *) echo "usage: $0 1|2|3|check|diag"; exit 1 ;;
esac
