"""Diagnose SAP-Bind Value failures on the sealed validation split only.

This is descriptive: it cannot turn a failed Value audit into a passing one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .value import BindingValueEncoder
from .value_audit import _instance_centroids, _values


def _mse(a, b):
    return float((a.float() - b.float()).square().mean())


def _paired_variant_signal(records):
    by_layout = {}
    for record in records:
        by_layout.setdefault(record['layout_id'], {})[record['variant']] = record
    rows = []
    for layout, variants in sorted(by_layout.items()):
        if set(variants) != {0, 1}:
            raise ValueError(f'Missing paired validation variants: {layout}')
        left, right = variants[0], variants[1]
        labels = left['supervision']['support_instance'].flatten()
        if not torch.equal(labels, right['supervision']['support_instance'].flatten()):
            raise ValueError(f'Paired variants have different geometry: {layout}')
        for chunk, (a, b) in enumerate(zip(left['supports'], right['supports'])):
            frames, height, width = a['latent'].shape[2:]
            count = frames * height * width
            start = sum(part['post'].shape[1] for part in left['supports'][:chunk])
            inside = labels[start:start + count] > 0
            if not inside.any() or inside.all():
                continue
            for name in ('latent', 'post'):
                if name == 'latent':
                    x = a[name].float().permute(0, 2, 3, 4, 1).reshape(count, -1)
                    y = b[name].float().permute(0, 2, 3, 4, 1).reshape(count, -1)
                else:
                    x, y = a[name][0].float(), b[name][0].float()
                if x.shape != y.shape or x.shape[0] != count:
                    raise ValueError(f'Paired {name} token shape mismatch: {layout}/{chunk}')
                delta = (x - y).square().mean(-1)
                object_mse = float(delta[inside].mean())
                background_mse = float(delta[~inside].mean())
                rows.append(dict(layout_id=layout, chunk=chunk, source=name,
                                 object_mse=object_mse, background_mse=background_mse,
                                 object_over_background=object_mse / max(background_mse, 1e-12),
                                 object_tokens=int(inside.sum()), background_tokens=int((~inside).sum())))
    return rows


def _centroid_scores(encoder, records):
    rows = []
    for record in records:
        centroids = _instance_centroids(
            _values(encoder, record), record['supervision']['support_instance'])
        for first in range(len(centroids)):
            for second in range(first + 1, len(centroids)):
                shared = sorted(set(centroids[first]) & set(centroids[second]))
                if len(shared) < 2:
                    continue
                for identity in shared:
                    query = centroids[first][identity]
                    distances = {other: _mse(query, centroids[second][other])
                                 for other in shared}
                    within = [_mse(query, centroids[first][other])
                              for other in shared if other != identity]
                    rows.append(dict(scene_id=record['scene_id'], first=first, second=second,
                                     instance=identity, cross_same=distances[identity],
                                     cross_hard_negative=min(v for k, v in distances.items()
                                                             if k != identity),
                                     within_hard_negative=min(within),
                                     top1=min(distances, key=distances.get) == identity,
                                     candidates=len(shared)))
    return rows


def diagnose(settings, stats_path):
    records = [torch.load(path, map_location='cpu', weights_only=True)
               for path in settings['val_features']]
    if not records or any(r.get('layer') != settings['layer'] for r in records):
        raise ValueError('Missing or mismatched validation feature layer')
    if any(r.get('source') != 'sap_binding_teacher_forced_ground_truth' for r in records):
        raise ValueError('Unexpected SAP-Bind feature source')
    stats = torch.load(stats_path, map_location='cpu', weights_only=True)
    if stats.get('mode') != settings['value_mode'] or stats.get('layer') != settings['layer']:
        raise ValueError('Value statistics do not match the requested mode/layer')
    sample = records[0]['supports'][0]
    encoder = BindingValueEncoder(sample['post'].shape[-1], sample['latent'].shape[1],
                                  settings['value_mode'], settings['value_dim'], settings['seed'])
    encoder.load_state_dict(stats['encoder'])
    encoder.eval()
    with torch.no_grad():
        signal = _paired_variant_signal(records)
        centroids = _centroid_scores(encoder, records)
    summary = {}
    for name in ('latent', 'post'):
        selected = [r for r in signal if r['source'] == name]
        summary[f'{name}_object_over_background'] = (
            sum(r['object_over_background'] for r in selected) / len(selected) if selected else None)
    summary['centroid_queries'] = len(centroids)
    summary['centroid_top1'] = (sum(r['top1'] for r in centroids) / len(centroids)
                                if centroids else None)
    summary['centroid_random_top1'] = (sum(1 / r['candidates'] for r in centroids)
                                       / len(centroids) if centroids else None)
    for field in ('cross_same', 'cross_hard_negative', 'within_hard_negative'):
        summary[field] = (sum(r[field] for r in centroids) / len(centroids)
                          if centroids else None)
    return dict(layer=settings['layer'], value_mode=settings['value_mode'],
                validation_only=True, descriptive_not_gate=True,
                summary=summary, paired_variant_rows=signal, centroid_rows=centroids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--value-stats', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    report = diagnose(settings, args.value_stats)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
