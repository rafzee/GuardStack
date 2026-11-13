#!/bin/bash
# 5-minute probe: answers the open questions that decide how the env is built.
#   sbatch slurm/probe.sbatch      -> send me logs/probe_<jobid>.out
#SBATCH --job-name=gs_probe
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.out
echo "== node: $(hostname)  date: $(date)"
echo "== 1. GPU + driver (decides the torch/vLLM CUDA build)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep -i "cuda version"
echo "== 2. Internet from a compute node (decides whether downloads can run in jobs)"
for u in https://huggingface.co https://pypi.org https://github.com https://openaipublic.blob.core.windows.net; do
  code=$(curl -s -o /dev/null -m 10 -w "%{http_code}" "$u"); echo "$u -> ${code:-fail}"
done
echo "== 3. Modules"
module avail cuda miniconda python 2>&1 | tr -s ' ' | head -20
echo "== 4. Storage"
mfsgetquota -H ~ 2>/dev/null | tail -3; mfsgetquota -H ~/work 2>/dev/null | tail -3
df -h ~/scratch 2>/dev/null | tail -1
echo "== 5. Local disk / TMPDIR"
echo "TMPDIR=$TMPDIR"; df -h /tmp | tail -1
echo "== 6. Groups (for /groups/<pi>)"
id -Gn; ls -d /groups/* 2>/dev/null | head
echo "== done"
