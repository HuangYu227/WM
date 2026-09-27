"""Check whether fixed SAP-Bind Values preserve exact cross-chunk object evidence.

Only completed support chunks from validation scenes are compared. Labels and
world coordinates select diagnostic pairs; they never enter the model reader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from worldttt.sap_ttt.pairs import match_historical_tokens

from .value import BindingValueEncoder
from .value_audit import _values


BOUNDS = (0, 4, 7, 10)


def score_pair(old, query, old_instance, query_instance, positive, max_queries=128):
    """Compare one Value at its true history position with identical-budget baselines."""
    if max_queries < 1:
        raise ValueError('max_queries must be positive')
    old = old.float().reshape(-1, old.shape[-1])
    query = query.float().reshape(-1, query.shape[-1])
    old_instance = torch.as_tensor(old_instance, device=old.device).flatten().long()
    query_instance = torch.as_tensor(query_instance, device=query.device).flatten().long()
    positive = torch.as_tensor(positive, device=query.device).flatten().long()
    if old.shape[-1] != query.shape[-1] or len(old_instance) != len(old):
        raise ValueError('Historical Value/instance shape mismatch')
    if len(query_instance) != len(query) or len(positive) != len(query):
        raise ValueError('Query Value/instance/correspondence shape mismatch')
    eligible = torch.where((positive >= 0) & (query_instance > 0))[0]
    total_objects = int((query_instance > 0).sum())
    if not len(eligible):
        return dict(valid=False, object_coverage=0., matched=0)
    if len(eligible) > max_queries:
        positions = torch.linspace(0, len(eligible) - 1, max_queries,
                                   device=eligible.device).long()
        eligible = eligible[positions]
    selected = query[eligible]
    identity = query_instance[eligible]
    correct = positive[eligible]
    if bool((correct >= len(old)).any()):
        raise ValueError('Correspondence index exceeds historical Value length')
    dim = old.shape[-1]
    distance = ((selected.square().sum(-1, keepdim=True) + old.square().sum(-1)[None]
                 - 2 * selected @ old.T) / dim).clamp_min(0)
    positive_mse = (selected - old[correct]).square().mean(-1)
    mean_mse = (selected - old.mean(0, keepdim=True)).square().mean(-1)
    wrong = distance.masked_fill((old_instance[None] == identity[:, None]) |
                                 (old_instance[None] == 0), float('inf'))
    wrong_best = wrong.min(-1).values
    has_wrong = torch.isfinite(wrong_best)
    nearest = distance.argmin(-1)
    return dict(valid=True, object_coverage=float((positive.ge(0) & (query_instance > 0)).sum()
                                                  / max(total_objects, 1)),
                matched=len(eligible), positive_mse=float(positive_mse.mean()),
                mean_value_mse=float(mean_mse.mean()),
                wrong_instance_nearest_mse=(float(wrong_best[has_wrong].mean())
                                            if bool(has_wrong.any()) else None),
                positive_better_than_mean=float((positive_mse < mean_mse).float().mean()),
                positive_better_than_wrong=(float((positive_mse[has_wrong] < wrong_best[has_wrong])
                                                  .float().mean()) if bool(has_wrong.any()) else None),
                instance_top1=float((old_instance[nearest] == identity).float().mean()),
                instance_random_top1=float((old_instance[None] == identity[:, None])
                                           .float().mean(-1).mean()),
                exact_top1=float((nearest == correct).float().mean()))


def _source_values(record, source, encoder, device):
    if source == 'value':
        return [part[0] for part in _values(encoder, record)]
    if source == 'post':
        return [part['post'][0].to(device).float() for part in record['supports']]
    if source == 'latent':
        return [part['latent'][0].to(device).float().permute(1, 2, 3, 0)
                .reshape(-1, part['latent'].shape[1]) for part in record['supports']]
    raise ValueError(f'Unknown Value probe source: {source}')


def probe(settings, stats_path, scenes_root, max_queries=128, source='value'):
    records = [torch.load(path, map_location='cpu', weights_only=True)
               for path in settings['val_features']]
    if not records or any(r.get('layer') != settings['layer'] for r in records):
        raise ValueError('Missing or mismatched validation feature layer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = None
    if source == 'value':
        if not stats_path:
            raise ValueError('Value statistics are required for encoded Values')
        stats = torch.load(stats_path, map_location='cpu', weights_only=True)
        if stats.get('mode') != settings['value_mode'] or stats.get('layer') != settings['layer']:
            raise ValueError('Value statistics do not match the requested mode/layer')
        part = records[0]['supports'][0]
        encoder = BindingValueEncoder(part['post'].shape[-1], part['latent'].shape[1],
                                      settings['value_mode'], settings['value_dim'], settings['seed'])
        encoder.load_state_dict(stats['encoder'])
        encoder.to(device).eval()
    elif source not in ('post', 'latent'):
        raise ValueError(f'Unknown Value probe source: {source}')
    rows = []
    with torch.no_grad():
        for record in records:
            if record.get('source') != 'sap_binding_teacher_forced_ground_truth':
                raise ValueError('Unexpected SAP-Bind validation feature source')
            path = Path(scenes_root) / record['scene_id'] / 'supervision.npz'
            with np.load(path, allow_pickle=False) as labels:
                instances, world = labels['instance'].copy(), labels['world'].copy()
            values = _source_values(record, source, encoder, device)
            for chunk in (1, 2):
                start, end = BOUNDS[chunk:chunk + 2]
                positive, _ = match_historical_tokens(
                    instances[:4], world[:4], instances[start:end], world[start:end])
                row = score_pair(values[0], values[chunk],
                                 instances[:4], instances[start:end], positive, max_queries)
                rows.append(dict(scene_id=record['scene_id'], chunk=chunk, **row))
    valid = [row for row in rows if row['valid']]
    fields = ('object_coverage', 'positive_mse', 'mean_value_mse',
              'wrong_instance_nearest_mse', 'positive_better_than_mean',
              'positive_better_than_wrong', 'instance_top1',
              'instance_random_top1', 'exact_top1')
    summary = {field: (sum(row[field] for row in valid if row[field] is not None)
                       / sum(row[field] is not None for row in valid)
                       if any(row[field] is not None for row in valid) else None)
               for field in fields}
    summary['valid_pairs'] = len(valid)
    summary['selected_queries'] = sum(row['matched'] for row in valid)
    return dict(layer=settings['layer'], value_mode=settings['value_mode'], source=source,
                validation_only=True, descriptive_not_gate=True,
                summary=summary, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--value-stats')
    parser.add_argument('--source', choices=('value', 'post', 'latent'), default='value')
    parser.add_argument('--scenes-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-queries', type=int, default=128)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    report = probe(settings, args.value_stats, args.scenes_root, args.max_queries,
                   source=args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
