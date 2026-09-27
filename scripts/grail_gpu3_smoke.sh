#!/usr/bin/env bash
# Full-checkpoint execution gate, not a research-quality training run.

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=GPU PYTHON_BIN=/path/to/python bash $0 BASE_CHECKPOINT RAW_ZIP_DIR LATENT_ZIP_DIR SCENES_JSONL NEW_RUN_DIR" >&2
  exit 2
fi

base=$1
raw=$2
latents=$3
manifest=$4
run=$5
for path in "$base" "$raw" "$latents" "$manifest" "$run"; do
  if [[ $path != /* ]]; then
    echo "All input and output paths must be absolute: $path" >&2
    exit 2
  fi
done
[[ -f $base ]] || { echo "Missing checkpoint: $base" >&2; exit 2; }
[[ -d $raw ]] || { echo "Missing raw ZIP directory: $raw" >&2; exit 2; }
[[ -d $latents ]] || { echo "Missing latent ZIP directory: $latents" >&2; exit 2; }
[[ -f $manifest ]] || { echo "Missing scenes.jsonl: $manifest" >&2; exit 2; }
[[ ! -e $run ]] || { echo "Run directory already exists: $run" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
[[ $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]] || {
  echo "Select exactly one physical GPU index in CUDA_VISIBLE_DEVICES" >&2
  exit 2
}
export SANA_WM_STAGE1_NVFP4=0
python_bin=${PYTHON_BIN:-python}
command -v "$python_bin" >/dev/null || { echo "Missing Python: $python_bin" >&2; exit 2; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
printf 'Python: %s\nModel: %s\nRaw: %s\nLatents: %s\nManifest: %s\nRun: %s\nGPU: %s\n' \
  "$python_bin" "$base" "$raw" "$latents" "$manifest" "$run" "$CUDA_VISIBLE_DEVICES"
mkdir -p "$run"

"$python_bin" - "$base" "$raw" "$latents" "$manifest" "$run/settings.json" <<'PY'
import json
from pathlib import Path
import sys

base, raw, latents, manifest, destination = sys.argv[1:]
settings = json.loads(Path('configs/worldttt/grail-v2.example.json').read_text(encoding='utf-8'))
rows = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
names = {row['key'].split('/', 1)[0] for row in rows if '/' in row['key']}
if len(names) != 1 or len(rows) != sum('/' in row['key'] for row in rows):
    raise ValueError('Expected one dataset-name prefix in all manifest keys for one raw ZIP directory')
settings['base_checkpoint'] = base
settings['manifest'] = manifest
settings['data']['data_dir'] = {names.pop(): raw}
settings['data']['vae_cache_dir'] = latents
settings['data']['min_latent_file_size'] = 0  # validate exact frames instead of size-filtering ZIP entries
settings.update(frames=10, max_steps=2, real_prefix_steps=1,
                curriculum=[[0, 3]], save_every=2, val_max_samples=1)
Path(destination).write_text(json.dumps(settings, indent=2), encoding='utf-8')
PY

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,memory.total,memory.free --format=csv
"$python_bin" -m worldttt doctor 2>&1 | tee "$run/doctor.log"
"$python_bin" -c 'import fla; print("fla import:", fla.__file__)' 2>&1 | tee "$run/fla.log"
"$python_bin" -m worldttt grail-train --settings "$run/settings.json" --output "$run/train" 2>&1 | tee "$run/train-console.log"
"$python_bin" -m worldttt grail-experiment --settings "$run/settings.json" \
  --adapter "$run/train/best.pt" --output "$run/eval" --split val --samples 2 \
  --histories real generated --variants ridge no_read 2>&1 | tee "$run/eval-console.log"
echo "Smoke complete: $run/eval/summary.json"
