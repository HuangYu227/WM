#!/usr/bin/env bash
# Long-history diagnostic and native rollout; preserves the existing scene split.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: PYTHON_BIN=/path/to/python bash $0 COMPLETED_SMOKE_RUN [NEW_RUN_DIR]" >&2
  exit 2
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base=$1
run=${2:-$repo/runs/grail-v2-gpu2-long-$(date -u +%Y%m%dT%H%M%SZ)}
[[ $base = /* && $run = /* ]] || { echo 'Use absolute run paths' >&2; exit 2; }
[[ -f "$base/settings.json" && -f "$base/train/best.pt" ]] || {
  echo "Missing smoke settings/checkpoint in $base" >&2; exit 2;
}
[[ ! -e "$run" ]] || { echo "Output already exists: $run" >&2; exit 2; }
python_bin=${PYTHON_BIN:-python}
command -v "$python_bin" >/dev/null
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
[[ $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]] || { echo 'Select exactly one GPU' >&2; exit 2; }
export SANA_WM_STAGE1_NVFP4=0 PYTHONUNBUFFERED=1
cd "$repo"
mkdir -p "$run"
exec > >(tee "$run/console.log") 2>&1
trap 'status=$?; echo "FAILED exit=$status; inspect $run/console.log"; exit "$status"' ERR
echo "RUN=$run"
echo "GPU=$CUDA_VISIBLE_DEVICES PYTHON=$python_bin"
git rev-parse HEAD
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.total,memory.used --format=csv

"$python_bin" -m worldttt.grail_long prepare --settings "$base/settings.json" --output "$run" \
  --max-steps "${MAX_STEPS:-300}" --clips-per-scene "${CLIPS_PER_SCENE:-8}"

evaluate_stage() {
  local phase=$1 checkpoint=$2 horizon
  for horizon in 31 61 121; do
    if [[ ! -f "$run/cases-$horizon.json" ]]; then
      echo "No qualifying camera returns at $horizon latents; this horizon is skipped."
      continue
    fi
    echo "Evaluating $phase at $horizon latent frames"
    "$python_bin" -m worldttt grail-experiment --settings "$run/settings.json" \
      --adapter "$checkpoint" --output "$run/$phase-$horizon" --split val --frames "$horizon" \
      --cases "$run/cases-$horizon.json" --seeds 3407 3408 \
      --histories real generated --variants ridge no_read prototype shuffle_value
  done
}

evaluate_stage before "$base/train/best.pt"
echo "Long-history training begins; live progress: $run/train/train.jsonl"
"$python_bin" -m worldttt grail-train --settings "$run/settings.json" --output "$run/train"
evaluate_stage after "$run/train/last.pt"

for horizon in 31 61 121; do
  if [[ -f "$run/cases-$horizon.json" ]]; then rollout_cases="$run/cases-$horizon.json"; fi
done
echo "Native sampler off/online comparison with 20 denoising steps per chunk, CFG=1"
"$python_bin" -m worldttt.grail_long sample --settings "$run/settings.json" \
  --adapter "$run/train/last.pt" --cases "$rollout_cases" --output "$run/rollout" --steps 20
# Decode in a fresh process so the 1.6B backbone is no longer on the GPU.
"$python_bin" -m worldttt.grail_long decode --settings "$run/settings.json" --output "$run/rollout"
echo "Completed long-history experiment: $run"
echo "Results: audit.json, before-*/summary.json, after-*/summary.json, train/train.jsonl, rollout/"
