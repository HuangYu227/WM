"""Audit candidate Values before training any SAP-Bind reader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .value import BindingValueEncoder


def _load(paths):
    records = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    if any(r.get('source') != 'sap_binding_teacher_forced_ground_truth' or not r.get('query_obscured')
           for r in records):
        raise ValueError('Value audit requires labeled, obscured SAP-Bind features')
    return records


def _values(encoder, record, raw=False):
    result = []
    device = encoder.hidden_projection.device
    for part in record['supports']:
        latent = part.get('latent')
        if latent is None:
            raise ValueError('SAP-Bind feature record is missing historical latent Values')
        frames, height, width = latent.shape[2:]
        fn = encoder.raw if raw else encoder
        result.append(fn(part['post'].to(device).float(), latent.to(device).float(), frames, height, width))
    return result


def _instance_centroids(values, labels):
    centroids = []
    offset = 0
    for value in values:
        count = value.shape[1]
        instance = labels.flatten()[offset:offset + count].to(value.device)
        offset += count
        centroids.append({identity: value[0, instance == identity].mean(0)
                          for identity in range(1, 7) if (instance == identity).any()})
    return centroids


def _record_metrics(encoder, record):
    values = _values(encoder, record)
    centroids = _instance_centroids(values, record['supervision']['support_instance'])
    same, different = [], []
    for first in range(len(centroids)):
        for second in range(first + 1, len(centroids)):
            shared = sorted(set(centroids[first]) & set(centroids[second]))
            for identity in shared:
                same.append((centroids[first][identity] - centroids[second][identity]).square().mean())
                negatives = [other for other in shared if other != identity]
                if negatives:
                    different.append((centroids[first][identity] - centroids[second][negatives[0]]).square().mean())
    joined = torch.cat([x.reshape(-1, x.shape[-1]) for x in values])
    mean_error = (joined - joined.mean(0, keepdim=True)).square().mean()
    generator = torch.Generator().manual_seed(3407 + int(record['variant']))
    permutation = torch.randperm(joined.shape[0], generator=generator)
    shuffle_error = (joined - joined[permutation]).square().mean()
    return dict(scene_id=record['scene_id'], layout_id=record['layout_id'], variant=record['variant'],
        valid=bool(same and different), same_instance=float(torch.stack(same).mean()) if same else None,
        different_instance=float(torch.stack(different).mean()) if different else None,
        mean_error=float(mean_error), shuffle_error=float(shuffle_error), tokens=int(joined.shape[0]))


def run(settings, output):
    mode = settings['value_mode']
    train, val = _load(settings['train_features']), _load(settings['val_features'])
    layer = settings.get('layer')
    if layer is not None and any(record.get('layer') != layer for record in (*train, *val)):
        raise ValueError('SAP-Bind Value audit feature layer mismatch')
    if not train or not val:
        raise ValueError('Value selection requires nonempty train and validation splits')
    layouts = {s: {r['layout_id'] for r in records} for s, records in [('train', train), ('val', val)]}
    if layouts['train'] & layouts['val']:
        raise ValueError('Paired layouts leaked across train and validation')
    sample = train[0]['supports'][0]
    encoder = BindingValueEncoder(sample['post'].shape[-1], sample['latent'].shape[1], mode,
                                  int(settings.get('value_dim', 512)), int(settings.get('seed', 3407)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    # Statistics are computed strictly from train records and then frozen.
    raw = []
    for record in train:
        raw.extend(_values(encoder, record, raw=True))
    encoder.fit_whitening(raw)
    rows = {split: [_record_metrics(encoder, record) for record in records]
            for split, records in [('train', train), ('val', val)]}
    summary = {}
    for split, group in rows.items():
        valid = [x for x in group if x['valid']]
        if not valid:
            raise ValueError(f'No valid same/different-instance comparisons in {split}')
        summary[split] = {key: sum(x[key] for x in valid) / len(valid)
                          for key in ('same_instance', 'different_instance', 'mean_error', 'shuffle_error')}
        summary[split]['separation_ratio'] = (summary[split]['different_instance'] /
                                              max(summary[split]['same_instance'], 1e-12))
        summary[split]['valid_episodes'] = len(valid)
    val = summary['val']
    separation = val['different_instance'] / max(val['same_instance'], 1e-12)
    versus_mean = val['different_instance'] / max(val['mean_error'], 1e-12)
    audit_passed = separation >= 1.15 and versus_mean >= 1.15
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(version=1, kind='sap_binding_value_stats', mode=mode, layer=layer,
                    train_layouts=sorted(layouts['train']),
                    audit_passed=audit_passed,
                    encoder={k: v.cpu() for k, v in encoder.state_dict().items()}),
               output / 'value_stats.pt')
    report = dict(version=1, mode=mode, summary=summary, rows=rows,
                  test_sealed=True, audit_passed=audit_passed,
                  separation_vs_mean=versus_mean,
                  selection_rule='train_and_validation_only')
    (output / 'value_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return output / 'value_audit.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(run(json.loads(Path(args.settings).read_text(encoding='utf-8')), args.output))


if __name__ == '__main__':
    main()
