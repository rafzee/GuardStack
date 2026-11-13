#!/bin/bash
# Source from inside a SLURM job:  source slurm/serve_models.sh
# Starts vLLM OpenAI-compatible servers in the background, waits until each answers, and
# registers a trap that stops them when the job ends (the cluster allows no long-lived services).
#
# Configure with env vars before sourcing:
#   AGENT_MODEL, AGENT_GPUS (e.g. "0" or "0,1"), AGENT_VLLM_ARGS, AGENT_PORT (8000)
#   GUARD_MODELS: ";"-separated "model_id|gpus|port|mem_fraction|extra vllm args" entries
#     (extra args optional), e.g.
#     "meta-llama/Llama-Guard-4-12B|1|8001|0.25; openai/gpt-oss-20b|1|8002|0.25"

LOGDIR=${LOGDIR:-$PWD/logs/${SLURM_JOB_ID:-local}}
mkdir -p "$LOGDIR"
VLLM_PIDS=()
# keep vLLM / triton / torch-compile caches off the 50 GB home quota
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/work/.cache} VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HOME/work/.cache/vllm}
GUARD_MAX_LEN=${GUARD_MAX_LEN:-32768}   # >= the agent's context, or trace judges fail on long sessions
# Importing vLLM from ~/work (MooseFS) takes ~4 min per process, and every server starts two
# (API server + engine core): ~8 of the 11-16 min each server needed on Nov 8. If the env was
# packed (slurm/pack_envs.sbatch, a plain tar of the env), unpack it onto the node's RAM disk and
# serve from there through its own python (shebangs and a few .pth files still name ~/work, which
# stays valid).
VLLM_PACK=${VLLM_PACK:-$HOME/work/envs/vllm.tar}
_FAST_ENV=""
if [ -z "${VLLM_BIN:-}" ] && [ -f "$VLLM_PACK" ]; then
  _FAST_ENV=/tmp/gs_vllm_${SLURM_JOB_ID:-$$}
  _t0=$SECONDS
  if mkdir -p "$_FAST_ENV" && tar -xf "$VLLM_PACK" -C "$_FAST_ENV" \
      && "$_FAST_ENV/bin/python" -m vllm.entrypoints.cli.main serve --help > /dev/null 2>&1; then
    # call vLLM through the unpacked python, so no script shebang can point back to ~/work
    printf '#!/bin/bash\nexec "%s/bin/python" -m vllm.entrypoints.cli.main "$@"\n' "$_FAST_ENV" > "$_FAST_ENV/gs_vllm"
    chmod +x "$_FAST_ENV/gs_vllm"
    VLLM_BIN=$_FAST_ENV/gs_vllm
    echo "vLLM env unpacked to $_FAST_ENV in $((SECONDS - _t0))s"
  else
    echo "WARNING: could not unpack $VLLM_PACK; serving from the env on ~/work" >&2
    rm -rf "$_FAST_ENV"; _FAST_ENV=""
  fi
fi
VLLM_BIN=${VLLM_BIN:-$HOME/work/envs/vllm/bin/vllm}   # separate env (see requirements/serve.txt)
# vLLM 0.33 samples top-k/top-p with FlashInfer, which JIT-compiles its kernel at startup and needs
# nvcc; the cluster's compute nodes have none (all five smoke/attacker jobs of Nov 7 died on it). Use vLLM's
# PyTorch sampler instead. Attention (FlashAttention 3) and the MoE / FP8 kernels are unaffected.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
declare -A VLLM_PID_OF VLLM_LOG_OF
[ -x "$VLLM_BIN" ] || { echo "vLLM not found at $VLLM_BIN (run slurm/setup_env.sh)" >&2; return 1 2>/dev/null || exit 1; }

_snapshot() {  # repo id -> its cached snapshot folder (a path is returned as is)
  # Serving the folder keeps vLLM away from snapshot_download: huggingface_hub >= 1.3 refuses an
  # offline snapshot when files were skipped at download time (we skip original/* and metal/*).
  local m=$1 d="${HF_HOME:-$HOME/work/.cache/huggingface}/hub/models--${1//\//--}"
  if [ -d "$m" ]; then echo "$m"
  elif [ -f "$d/refs/main" ] && [ -d "$d/snapshots/$(cat "$d/refs/main")" ]; then echo "$d/snapshots/$(cat "$d/refs/main")"
  else echo "$m"; fi
}

_start() {  # model gpus port mem extra_args name
  local model=$1 gpus=$2 port=$3 mem=$4 extra=$5 name=$6
  local tp path names=()
  tp=$(awk -F, '{print NF}' <<< "$gpus")
  python -c "import sys; from guardstack.llm import assert_allowed; assert_allowed(sys.argv[1])" "$model"
  path=$(_snapshot "$model")
  # clients keep using the repo id as the model name
  [[ "$path" != "$model" && "$extra" != *--served-model-name* ]] && names=(--served-model-name "$model")
  CUDA_VISIBLE_DEVICES=$gpus nohup "$VLLM_BIN" serve "$path" "${names[@]}" --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$tp" --gpu-memory-utilization "$mem" --seed 0 --generation-config vllm $extra \
    > "$LOGDIR/vllm_${name}_${port}.log" 2>&1 &
  VLLM_PIDS+=($!)
  VLLM_PID_OF[$port]=$!
  VLLM_LOG_OF[$port]="$LOGDIR/vllm_${name}_${port}.log"
  echo "started $model on GPU(s) $gpus port $port (pid $!) from $path"
}

_wait() {  # port timeout_s -- returns as soon as the server answers, or dies
  local port=$1 t=${2:-1800} i=0 pid=${VLLM_PID_OF[$1]} log=${VLLM_LOG_OF[$1]}
  until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM on $port exited during startup after ${i}s; errors in $log:" >&2
      grep -oE "[A-Za-z_.]*(Error|Exception): .*" "$log" | grep -v "Engine core initialization failed" \
        | awk '!seen[$0]++' | tail -3 >&2
      return 1
    fi
    sleep 10; i=$((i + 10))
    if [ $i -ge "$t" ]; then echo "vLLM on $port not ready after ${t}s" >&2; return 1; fi
  done
  echo "port $port ready after ${i}s"
}

stop_models() {
  for p in "${VLLM_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  [ -n "$_FAST_ENV" ] && rm -rf "$_FAST_ENV"    # free the RAM disk
  return 0
}

AGENT_PORT=${AGENT_PORT:-8000}
trap 'stop_models' EXIT
_start "$AGENT_MODEL" "${AGENT_GPUS:-0}" "$AGENT_PORT" "${AGENT_MEM:-0.90}" "$AGENT_VLLM_ARGS" agent
# guard servers share one GPU: start them one at a time (concurrent memory profiling misbehaves)
IFS=';' read -ra _SPECS <<< "$GUARD_MODELS"
for spec in "${_SPECS[@]}"; do
  spec=$(echo "$spec" | sed 's/^ *//; s/ *$//'); [ -z "$spec" ] && continue
  IFS='|' read -r m g p f x <<< "$spec"
  _start "$m" "$g" "$p" "$f" "--max-model-len $GUARD_MAX_LEN $x" guard
  _wait "$p"
done
_wait "$AGENT_PORT"
export GUARDSTACK_AGENT_URL=http://127.0.0.1:$AGENT_PORT/v1
