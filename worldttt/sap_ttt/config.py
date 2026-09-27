"""Derive SAP settings from an already verified server-local WorldTTT config."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def sap_settings(base, *, procedural_fixtures=(), probe_checkpoint=None, multimodal=False,
                 layers=(7,)):
    settings = copy.deepcopy(base)
    layers = tuple(layers)
    if not layers or len(set(layers)) != len(layers) or any(layer < 1 or layer > 20 for layer in layers):
        raise ValueError('SAP layers must be unique one-based SANA block indices in 1..20')
    settings['sap'] = {'mode': 'online', 'layers': list(layers), 'address_mode': 'selective_geometry',
                       'dim': 256, 'ray_dim': 48, 'support_tokens': 256,
                       'inner_lr': .5, 'seed': 3407}
    settings['sap_train'] = {'max_steps': 300, 'gradient_accumulation': 2,
                             'val_every': 50, 'val_max_samples': 4, 'val_seed': 12345,
                             'outer_lr': 1e-4, 'lambda_delayed': .1,
                             'probe_pretrain_steps': 100 if procedural_fixtures else 0,
                             'real_prefix_steps': 100,
                             'procedural_fixtures': list(procedural_fixtures)}
    if probe_checkpoint:
        settings['sap_train']['probe_checkpoint'] = str(probe_checkpoint)
    if multimodal:
        settings['sap'].update(address_arch='multimodal', memory_arch='swiglu', dim=512,
                               heads=8, address_depth=2, memory_hidden_dim=128, inner_lr=.1,
                               normalized_value=True)
        settings['sap_train']['lambda_exact'] = .1
    settings['steps'] = 50
    settings['save_rollout_state'] = False
    settings['address_trace'] = False
    settings['address_analysis'] = False
    settings['relative_revisit'] = True
    return settings


def main():
    parser = argparse.ArgumentParser(description='Prepare SAP settings without changing base server paths')
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--procedural-dir')
    parser.add_argument('--probe-checkpoint')
    parser.add_argument('--multimodal', action='store_true')
    parser.add_argument('--layers', nargs='+', type=int, default=[7])
    args = parser.parse_args()
    fixtures = []
    if args.procedural_dir:
        fixtures = sorted(str(p.resolve()) for p in Path(args.procedural_dir).glob('scene_*/fixture.pt'))
        if not fixtures:
            raise FileNotFoundError('No encoded scene_*/fixture.pt under procedural-dir')
    base = json.loads(Path(args.base).read_text(encoding='utf-8'))
    settings = sap_settings(base, procedural_fixtures=fixtures,
                            probe_checkpoint=args.probe_checkpoint, multimodal=args.multimodal,
                            layers=args.layers)
    Path(args.output).write_text(json.dumps(settings, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
