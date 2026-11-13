#!/bin/bash
# Submit one phase of the Path A study for one agent model.
#   bash slurm/submit_path_a.sh <model> <phase>
#     model: gptoss | llama70
#     phase: static | latency | adaptive | staged | reattack | multiturn | sweep | analyze
# Static phase sized from the Nov 8 smoke tests: 4 shards x 19 stacks, 2 shards running per model.
# Cluster caps: 4 running / 100 submitted jobs per user. Every job is resumable with the same RUN_ID.
set -eo pipefail
MODEL_KEY=${1:?model: gptoss|llama70}; PHASE=${2:?phase}
cd "${PROJ:-$HOME/work/guardstack}"; mkdir -p logs

case "$MODEL_KEY" in
  gptoss)  export AGENT_MODEL=openai/gpt-oss-120b
           export AGENT_VLLM_ARGS="--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768";;
  llama70) export AGENT_MODEL=meta-llama/Llama-3.3-70B-Instruct
           export AGENT_VLLM_ARGS="--enable-auto-tool-choice --tool-call-parser llama3_json --max-model-len 32768 --quantization fp8";;
  *) echo "unknown model $MODEL_KEY"; exit 1;;
esac
export RUN_ID=${RUN_ID:-pathA_${MODEL_KEY}}
SHARDS=${SHARDS:-4}
TASKS=${TASKS:-0-$((SHARDS-1))%2}   # array tasks to submit; e.g. TASKS=1 resubmits shard 1 only
DEP=${DEPEND:+--dependency=$DEPEND}  # e.g. DEPEND=afterok:12345 (start only if that job succeeded)

ISO="iso:pg2_in iso:pg2_tool iso:align iso:codeshield iso:regex_tool iso:nemo_in iso:nemo_ret iso:nemo_out iso:presidio iso:lg_in iso:lg_out iso:pi_deberta iso:gate"
case "$PHASE" in
  static)    # runs 1-4: bare, shadow-all, each layer alone, nested B-E
    # 24 workers: the agent servers had headroom at 16 (gpt-oss-120b KV 1.19M tokens, job 502208)
    # STACKS may be preset for a repair pass (only stacks with errored cases; see summarize_run.py)
    sbatch $DEP --array=$TASKS --export=ALL,STACKS="${STACKS:-A_bare shadow_all $ISO B C D E}",NUM_SHARDS=$SHARDS,WORKERS=${WORKERS:-24} slurm/run_stack.sbatch ;;
  latency)   # run 9: 10% stratified sample, one session at a time
    sbatch $DEP --array=0 --time=1-00:00:00 --export=ALL,RUN_ID=${RUN_ID}_lat,STACKS="A_bare shadow_all B C D E",MODE=latency,SAMPLE=0.1,WORKERS=1,NUM_SHARDS=1 slurm/run_stack.sbatch ;;
  adaptive)  # run 7: LLM rewriter, 30% sample -> runs/<RUN_ID>_adaptive/<stack>/adaptive_rewrite_<FEEDBACK>_b<BUDGET>
    # default = trace feedback, budget 10; the Nov 9 weak-attacker ablation was FEEDBACK=guard BUDGET=5
    sbatch $DEP --array=$TASKS --export=ALL,RUN_ID=${RUN_ID}_adaptive,STACKS="${STACKS:-A_bare iso:pg2_tool iso:gate iso:align B C D E}",ATTACK=${ATTACK:-edit_staged},BUDGET=${BUDGET:-10},FEEDBACK=${FEEDBACK:-trace},NUM_SHARDS=$SHARDS slurm/run_adaptive.sbatch ;;
  staged)    # STACK-style staged attack on the full stacks -> runs/<RUN_ID>_staged
    sbatch $DEP --array=$TASKS --export=ALL,RUN_ID=${RUN_ID}_staged,STACKS="${STACKS:-C D E}",ATTACK=staged,BUDGET=${BUDGET:-10},FEEDBACK=${FEEDBACK:-trace},NUM_SHARDS=$SHARDS slurm/run_adaptive.sbatch ;;
  reattack)  # run 6: optimizer stacks (export first: scripts/export_stacks.py -> configs/stacks_optimized_${MODEL_KEY}.yaml)
    CFG=configs/stacks_optimized_${MODEL_KEY}.yaml; [ -f "$CFG" ] || { echo "missing $CFG (run export_stacks.py)"; exit 1; }
    IDS=$(python -c "import yaml,sys; print(' '.join(s['id'] for s in yaml.safe_load(open('$CFG'))['stacks']))")
    sbatch $DEP --array=$TASKS --export=ALL,RUN_ID=${RUN_ID}_reattack,STACK_CONFIG=$CFG,STACKS="$IDS",NUM_SHARDS=$SHARDS,WORKERS=${WORKERS:-24} slurm/run_stack.sbatch
    sbatch $DEP --array=$TASKS --export=ALL,RUN_ID=${RUN_ID}_reattack_adaptive,STACK_CONFIG=$CFG,STACKS="$IDS",ATTACK=${ATTACK:-edit_staged},BUDGET=${BUDGET:-10},FEEDBACK=${FEEDBACK:-trace},NUM_SHARDS=$SHARDS slurm/run_adaptive.sbatch ;;
  multiturn) # run 8
    sbatch $DEP --array=0-1 --export=ALL,RUN_ID=${RUN_ID}_multiturn,STACKS="${STACKS:-A_bare iso:lg_in iso:nemo_in C D E}",NUM_SHARDS=2 slurm/run_multiturn.sbatch ;;
  sweep)     # guard-only sweep (model-independent; run once)
    sbatch $DEP --export=ALL,RUN_ID=sweep_v1 slurm/guard_sweep.sbatch ;;
  analyze)
    sbatch $DEP --export=ALL,LATENCY_RUN_ID=${RUN_ID}_lat slurm/analyze.sbatch ;;
  *) echo "unknown phase $PHASE"; exit 1;;
esac
squeue --me
