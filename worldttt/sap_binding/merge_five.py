"""Assemble five independently validated SAP-Bind feature adapters for Flow training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .config import FIVE_BINDING_LAYERS, BindingConfig, binding_settings
from .prepare import fixture_for_feature


FIVE_LAYERS = FIVE_BINDING_LAYERS


def _paths_by_layer(paths, name):
    if set(paths) != set(FIVE_LAYERS):
        raise ValueError(f'{name} requires layers {FIVE_LAYERS}')
    return {layer: Path(paths[layer]).resolve() for layer in FIVE_LAYERS}


def _scenes(settings, split):
    return tuple(sorted(Path(path).parent.name for path in settings[split + '_features']))


def merge_five(checkpoints, causal_reports, structure_reports, base, output):
    """Merge only five compatible feature adapters that each passed both gates."""
    checkpoints = _paths_by_layer(checkpoints, 'Checkpoints')
    causal_reports = _paths_by_layer(causal_reports, 'Causal reports')
    structure_reports = _paths_by_layer(structure_reports, 'Structure reports')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh five-layer SAP-Bind output directory')

    reference_config = reference_backbone = reference_splits = None
    modules, sources, val_features = {}, {}, {}
    state_bytes = 0
    for layer in FIVE_LAYERS:
        path = checkpoints[layer]
        payload = torch.load(path, map_location='cpu', weights_only=True)
        extra = payload.get('extra', {})
        config = copy.deepcopy(payload.get('config', {}))
        if (payload.get('version') != 1 or payload.get('kind') != 'sap_binding_adapter' or
                set(payload.get('modules', {})) != {layer} or
                tuple(config.pop('layers', ())) != (layer,) or
                extra.get('stage') != 'binding_feature_joint' or
                extra.get('flow_trained') is not False or
                config.get('architecture') != 'hybrid'):
            raise ValueError(f'Layer {layer} needs a hybrid feature-stage SAP-Bind checkpoint: {path}')
        settings = extra.get('settings', {})
        try:
            splits = {name: _scenes(settings, name) for name in ('train', 'val', 'test')}
        except KeyError as error:
            raise ValueError(f'Layer {layer} lacks recorded data split') from error
        if not all(splits.values()) or set(splits['train']) & set(splits['val']) or (
                set(splits['train']) | set(splits['val'])) & set(splits['test']):
            raise ValueError(f'Layer {layer} has an invalid feature split')
        backbone = payload.get('base_checkpoint')
        if base.get('base_checkpoint') != backbone:
            raise ValueError(f'Layer {layer} backbone differs from Flow settings')
        if reference_config is None:
            reference_config, reference_backbone, reference_splits = config, backbone, splits
        elif (config != reference_config or backbone != reference_backbone or
              splits != reference_splits):
            raise ValueError(f'Layer {layer} has an incompatible five-layer protocol or split')
        tensors = payload['modules'][layer]
        if not tensors or any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f'Layer {layer} has nonfinite or missing parameters')
        if not bool(tensors.get('value.whitening_fitted', torch.tensor(False))):
            raise ValueError(f'Layer {layer} has no train-fitted Value whitening')
        causal = json.loads(causal_reports[layer].read_text(encoding='utf-8'))
        structure = json.loads(structure_reports[layer].read_text(encoding='utf-8'))
        if (causal.get('kind') != 'sap_binding_causal_gate' or causal.get('split') != 'val' or
                causal.get('architecture') != 'hybrid' or
                not causal.get('gates', {}).get('passed') or
                Path(causal.get('adapter', '')).resolve() != path):
            raise ValueError(f'Layer {layer} causal binding gate failed or mismatched')
        if (not structure.get('flow_allowed') or
                Path(structure.get('adapters', {}).get('hybrid', '')).resolve() != path):
            raise ValueError(f'Layer {layer} hybrid structure gate failed or mismatched')
        state_bytes += int(causal.get('state_bytes_per_branch', 0))
        modules[layer] = tensors
        val_features[str(layer)] = settings['val_features']
        sources[str(layer)] = dict(path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), step=extra.get('step'),
            settings=settings, causal_report=str(causal_reports[layer]),
            structure_report=str(structure_reports[layer]))

    merged_path = (output / 'merged.pt').resolve()
    causal_path = (output / 'causal-gate.json').resolve()
    structure_path = (output / 'structure-gate.json').resolve()
    merged_config = dict(reference_config, layers=FIVE_LAYERS)
    BindingConfig(**merged_config)
    merged = dict(version=1, kind='sap_binding_adapter', config=merged_config,
        base_checkpoint=reference_backbone, modules=modules,
        extra=dict(stage='binding_feature_joint_multilayer', flow_trained=False,
                   layer_sources=sources))
    flow = binding_settings(base, layers=FIVE_LAYERS)
    flow['sap_binding'] = dict(merged_config, layers=list(FIVE_LAYERS))
    flow['binding_val_features'] = val_features
    flow['binding_causal_gate'] = str(causal_path)
    flow['binding_structure_gate'] = str(structure_path)
    fixtures = [str(fixture_for_feature(path)) for path in sources['3']['settings']['train_features']]
    flow['sap_binding_train'] = dict(max_steps=300, gradient_accumulation=2,
        val_every=50, val_max_samples=4, val_seed=12345, outer_lr=1e-4,
        procedural_steps=100, generated_history_probability=.5,
        feature_checkpoint=str(merged_path), procedural_fixtures=fixtures)
    flow['relative_revisit'] = True

    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'merged.tmp.pt'
    try:
        torch.save(merged, temporary)
        temporary.replace(merged_path)
    finally:
        temporary.unlink(missing_ok=True)
    causal_path.write_text(json.dumps(dict(version=1, kind='sap_binding_causal_gate',
        split='val', architecture='hybrid', adapter=str(merged_path),
        layers=list(FIVE_LAYERS), gates={'passed': True},
        state_bytes_per_branch=state_bytes, layer_sources=sources),
        indent=2), encoding='utf-8')
    structure_path.write_text(json.dumps(dict(flow_allowed=True,
        adapters={'hybrid': str(merged_path)}, layers=list(FIVE_LAYERS),
        layer_sources=sources), indent=2), encoding='utf-8')
    (output / 'flow-pilot.json').write_text(json.dumps(flow, indent=2), encoding='utf-8')
    (output / 'split.json').write_text(json.dumps(reference_splits, indent=2), encoding='utf-8')
    return merged_path


def _named_paths(items, parser):
    paths = {}
    for item in items:
        layer_text, separator, path = item.partition('=')
        if not separator or not path:
            parser.error(f'Expected LAYER=PATH, got {item}')
        try:
            layer = int(layer_text)
        except ValueError:
            parser.error(f'Invalid layer: {layer_text}')
        if layer in paths:
            parser.error(f'Duplicate layer: {layer}')
        paths[layer] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='LAYER=PATH')
    parser.add_argument('--causal', action='append', required=True, metavar='LAYER=PATH')
    parser.add_argument('--structure', action='append', required=True, metavar='LAYER=PATH')
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text(encoding='utf-8'))
    print(merge_five(_named_paths(args.checkpoint, parser),
                     _named_paths(args.causal, parser),
                     _named_paths(args.structure, parser), base, args.output))


if __name__ == '__main__':
    main()
