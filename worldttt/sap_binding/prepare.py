"""Create reproducible SAP-Bind value, structure, causal and Flow configs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from .config import binding_settings


VALUE_RUNS = {'old': 'old', 'centered_h7': 'centered-h7',
              'highpass_h7': 'highpass-h7', 'highpass_h7_latent': 'highpass-h7-latent'}


def fixture_for_feature(path):
    path = Path(path)
    local = path.with_name('fixture.pt')
    if local.is_file():
        return local
    shared = path.parents[2] / path.parent.name / 'fixture.pt'
    if shared.is_file():
        return shared
    raise FileNotFoundError(f'Missing encoded SAP-Bind fixture for {path}')


def prepare(base, features_dir, output, value_root, *, layer=7):
    paths = sorted(Path(features_dir).resolve().glob('binding_*/features.pt'))
    if len(paths) != 64:
        raise ValueError(f'Expected 64 paired SAP-Bind feature records, found {len(paths)}')
    records = [(path, torch.load(path, map_location='cpu', weights_only=True)) for path in paths]
    grouped = {}
    for path, record in records:
        if record.get('source') != 'sap_binding_teacher_forced_ground_truth':
            raise ValueError(f'Unexpected feature protocol: {path}')
        if record.get('layer') != layer:
            raise ValueError(f'SAP-Bind feature layer mismatch: {path}')
        if base.get('base_checkpoint') and record.get('base_checkpoint') != base['base_checkpoint']:
            raise ValueError(f'SAP-Bind feature backbone mismatch: {path}')
        grouped.setdefault(record['layout_id'], []).append((path, record))
    if len(grouped) != 32 or any(len(group) != 2 or {r['variant'] for _, r in group} != {0, 1}
                                 for group in grouped.values()):
        raise ValueError('Every one of 32 layouts must contain variants 0 and 1')
    layouts = sorted(grouped)
    split_layouts = {'train': layouts[:24], 'val': layouts[24:28], 'test': layouts[28:]}
    splits = {split: [str(path) for layout in ids for path, _ in grouped[layout]]
              for split, ids in split_layouts.items()}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh SAP-Bind config directory')
    output.mkdir(parents=True)
    protocol = binding_settings({}, layers=(layer,))['sap_binding']
    for mode, slug in VALUE_RUNS.items():
        config = dict(value_mode=mode, value_dim=512, seed=3407, layer=layer,
                      train_features=splits['train'], val_features=splits['val'])
        (output / f'value-{slug}.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    common = dict(sap_binding=protocol, **{f'{s}_features': p for s, p in splits.items()},
        value_stats=str((Path(value_root).resolve() / 'highpass-h7-latent/value_stats.pt')),
        steps=600, batch_size=16, val_every=50, lr=1e-4, seed=3407, val_seed=12345,
        support_tokens=512, causal_support_tokens=256, protected_anchors=128, query_tokens=64,
        loss_weights=dict(addr=.2, value=1., fast=.5, shuffle=.5, mean=.25, write=.01))
    variants = {
        'joint-hybrid.json': {},
        'joint-bank-only.json': {'architecture': 'bank_only'},
        'joint-fast-only.json': {'architecture': 'fast_only'},
        'joint-no-constraints.json': {'loss_weights': dict(common['loss_weights'], shuffle=0., mean=0.)},
    }
    for name, updates in variants.items():
        config = copy.deepcopy(common)
        if 'architecture' in updates: config['sap_binding']['architecture'] = updates['architecture']
        if 'loss_weights' in updates: config['loss_weights'] = updates['loss_weights']
        (output / name).write_text(json.dumps(config, indent=2), encoding='utf-8')
    flow = binding_settings(base, layers=(layer,))
    flow['sap_binding'] = protocol
    flow['binding_val_features'] = splits['val']
    flow['sap_binding_train'] = dict(max_steps=300, gradient_accumulation=2, val_every=50,
        val_max_samples=4, val_seed=12345, outer_lr=1e-4, procedural_steps=100,
        generated_history_probability=.5, feature_checkpoint='FILL_AFTER_CAUSAL_GATE',
        procedural_fixtures=[str(fixture_for_feature(path)) for path in splits['train']])
    flow['binding_causal_gate'] = 'FILL_AFTER_CAUSAL_GATE'
    flow['binding_structure_gate'] = 'FILL_AFTER_STRUCTURE_COMPARISON'
    flow['steps'] = 50; flow['save_rollout_state'] = False
    flow['relative_revisit'] = True
    (output / 'flow-pilot.json').write_text(json.dumps(flow, indent=2), encoding='utf-8')
    (output / 'split.json').write_text(json.dumps(dict(version=1,
        protocol='layout_disjoint_paired_stickers', layouts=split_layouts,
        episodes={key: len(value) for key, value in splits.items()}), indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--features-dir', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--value-root', required=True)
    parser.add_argument('--layer', type=int, default=7)
    args = parser.parse_args()
    print(prepare(json.loads(Path(args.base).read_text(encoding='utf-8')), args.features_dir,
                  args.output, args.value_root, layer=args.layer))


if __name__ == '__main__':
    main()
