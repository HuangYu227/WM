"""Assemble independently pretrained SAP layers into one five-layer adapter.

This produces a feature-stage checkpoint. The normal SAP Flow trainer must
still train the five residual gates against the frozen SANA backbone.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch


FIVE_GDN_LAYERS = (3, 7, 11, 15, 19)


def merge_layers(checkpoints: dict[int, str | Path], output: str | Path):
    if set(checkpoints) != set(FIVE_GDN_LAYERS):
        raise ValueError(f'Expected five independently pretrained layers: {FIVE_GDN_LAYERS}')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)

    reference_config = reference_backbone = reference_source = None
    modules, provenance = {}, {}
    for layer in FIVE_GDN_LAYERS:
        path = Path(checkpoints[layer]).resolve()
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('version') != 1 or set(payload.get('modules', {})) != {layer}:
            raise ValueError(f'Expected a single-layer SAP checkpoint for layer {layer}: {path}')
        extra = payload.get('extra', {})
        if extra.get('stage') != 'feature_joint' or extra.get('flow_trained') is not False:
            raise ValueError(f'Layer {layer} must use a feature-stage checkpoint: {path}')
        config = copy.deepcopy(payload['config'])
        if tuple(config.pop('layers')) != (layer,):
            raise ValueError(f'Layer {layer} checkpoint configuration mismatch: {path}')
        backbone = payload['base_checkpoint']
        source = extra.get('query_source')
        if reference_config is None:
            reference_config, reference_backbone, reference_source = config, backbone, source
        elif (config != reference_config or backbone != reference_backbone or
              source != reference_source):
            raise ValueError(f'Incompatible layer {layer} pretraining protocol: {path}')
        tensors = payload['modules'][layer]
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f'Nonfinite layer {layer} parameters: {path}')
        modules[layer] = tensors
        provenance[str(layer)] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'step': extra.get('step')}

    merged_config = dict(reference_config, layers=FIVE_GDN_LAYERS)
    merged = {'version': 1, 'config': merged_config, 'base_checkpoint': reference_backbone,
              'modules': modules,
              'extra': {'stage': 'feature_joint_multilayer', 'flow_trained': False,
                        'checkpoint_kind': 'sap_adapter_only', 'query_source': reference_source,
                        'layer_sources': provenance}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    try:
        torch.save(merged, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='LAYER=PATH',
                        help='Repeat once for each of layers 3, 7, 11, 15, 19')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.checkpoint:
        layer_text, separator, path = item.partition('=')
        if not separator or not path:
            parser.error(f'Invalid --checkpoint value: {item}')
        try:
            layer = int(layer_text)
        except ValueError:
            parser.error(f'Invalid SAP layer: {layer_text}')
        if layer in paths:
            parser.error(f'Duplicate SAP layer: {layer}')
        paths[layer] = path
    print(merge_layers(paths, args.output))


if __name__ == '__main__':
    main()
