"""Prepare disjoint feature/flow configs from existing server-local v2 fixtures."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from .config import sap_settings


def prepare(base, features_dir, output, *, train_count=8, val_count=4, batch_size=4,
            layer=7):
    files = sorted(Path(features_dir).resolve().glob('scene_*/features.pt'))
    if min(train_count, val_count, batch_size) < 1 or len(files) <= train_count + val_count:
        raise ValueError('Need nonempty train/val/test scene splits and a positive batch size')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh configuration directory to preserve split provenance')
    scenes = []
    for path in files:
        record = torch.load(path, map_location='cpu', weights_only=True)
        if record['base_checkpoint'] != base['base_checkpoint'] or record['layer'] != layer:
            raise ValueError(f'Feature checkpoint/layer mismatch: {path}')
        scenes.append(record['scene_id'])
    if len(set(scenes)) != len(scenes):
        raise ValueError('Duplicate scene identities across feature files')
    splits = {'train': files[:train_count], 'val': files[train_count:train_count + val_count],
              'test': files[train_count + val_count:]}
    fixtures = [str(p.with_name('fixture.pt')) for p in splits['train']]
    if not all(Path(p).is_file() for p in fixtures):
        raise FileNotFoundError('Missing encoded fixture for procedural flow training')
    flow = sap_settings(base, procedural_fixtures=fixtures, multimodal=True, layers=(layer,))
    joint = {name + '_features': [str(p) for p in paths] for name, paths in splits.items()}
    joint.update(sap=copy.deepcopy(flow['sap']), steps=300, batch_size=batch_size, val_every=25,
                 lr=1e-4, seed=3407, val_seed=12345, support_tokens=256, query_tokens=64,
                 address_weight=.1, write_weight=.01, query_source='native')
    variants = {'joint-native.json': joint, 'flow.json': flow}
    for name, updates in (
        ('joint-no-history.json', {'query_source': 'no_history'}),
        ('joint-linear-memory.json', {'sap': dict(joint['sap'], memory_arch='linear')}),
        ('joint-linear-address.json', {'sap': dict(joint['sap'], address_arch='linear')})):
        variant = copy.deepcopy(joint); variant.update(updates); variants[name] = variant
    variants['split.json'] = dict(order='lexicographic_scene_path_before_training',
        train=[p.parent.name for p in splits['train']], val=[p.parent.name for p in splits['val']],
        test=[p.parent.name for p in splits['test']],
        note='Small mechanism split; not evidence of generated-video efficacy')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in variants.items():
        (output / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-count', type=int, default=8)
    parser.add_argument('--val-count', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--layer', type=int, default=7)
    args = parser.parse_args()
    print(prepare(json.loads(Path(args.base).read_text(encoding='utf-8')), args.features_dir,
                  args.output, train_count=args.train_count, val_count=args.val_count,
                  batch_size=args.batch_size, layer=args.layer))


if __name__ == '__main__':
    main()
