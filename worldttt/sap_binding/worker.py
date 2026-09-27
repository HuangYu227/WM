"""Resumable sharded fixture/feature preparation for four independent GPUs."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .features import encode_scene, extract_features


def run(settings, manifest, output, shard, shards):
    if not 0 <= shard < shards:
        raise ValueError('Shard index must be in [0, shards)')
    cases = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = [(index, case) for index, case in enumerate(cases) if index % shards == shard]
    if not selected:
        raise ValueError('Selected shard is empty')
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    layers = tuple(settings['sap_binding']['layers'])
    for _, case in tqdm(selected, desc=f'SAP-Bind feature shard {shard}', unit='episode'):
        folder = root / case['id']; folder.mkdir(parents=True, exist_ok=True)
        fixture = folder / 'fixture.pt'
        features = {layer: (folder / 'features.pt' if len(layers) == 1 else
                   root / f'layer_{layer}' / case['id'] / 'features.pt') for layer in layers}
        if all(feature.is_file() for feature in features.values()):
            try:
                saved = {layer: torch.load(feature, map_location='cpu', weights_only=True)
                         for layer, feature in features.items()}
                if all(record.get('source') == 'sap_binding_teacher_forced_ground_truth' and
                       record.get('scene_id') == case['id'] and record.get('layer') == layer and
                       record.get('query_obscured') for layer, record in saved.items()):
                    continue
            except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
                pass
        if not fixture.is_file():
            encode_scene(settings, case, fixture)
        target = folder / 'features.pt' if len(layers) == 1 else root
        extract_features(settings, fixture, Path(case['image']).with_name('supervision.npz'), target)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args(); settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    print(f'Prepared {run(settings, args.manifest, args.output, args.shard, args.shards)} assigned episodes')


if __name__ == '__main__':
    main()
