"""Measure fixed bank/fast mixtures on paired held-out binding queries."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .causal_eval import _sample, evaluate_batch
from .config import BindingConfig
from .joint import _load_records, attach_values, collate
from .model import BindingBlock


ALPHAS = (0., .25, .5, .75, .9, .99, 1.)


def run(settings, adapter, output, *, split='val', coverage_split=False):
    if split not in {'val', 'test'}:
        raise ValueError('Probe split must be val or test')
    config = BindingConfig(**settings['sap_binding'])
    if config.architecture != 'hybrid' or len(config.layers) != 1:
        raise ValueError('Fusion probe requires a single-layer hybrid adapter')
    payload = torch.load(adapter, map_location='cpu', weights_only=True)
    layer = config.layers[0]
    if (payload.get('kind') != 'sap_binding_adapter' or
            asdict(BindingConfig(**payload.get('config', {}))) != asdict(config) or
            set(payload.get('modules', {})) != {layer}):
        raise ValueError('Fusion probe adapter/config mismatch')
    records = _load_records(settings[split + '_features'])
    if not records or any(record.get('layer') != layer or
                          record.get('base_checkpoint') != payload['base_checkpoint']
                          for record in records):
        raise ValueError('Fusion probe features mismatch')
    sample = records[0]['supports'][0]
    module = BindingBlock(sample['visual'].shape[-1], sample['latent'].shape[1], config)
    module.load_state_dict(payload['modules'][layer], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module.to(device).eval()
    attach_values(records, module.value)
    learned_fusion = module.fusion
    fixed_fusion = nn.Linear(4, config.heads).to(device)
    with torch.no_grad():
        fixed_fusion.weight.zero_()
    rows = []
    coverage_counts = []
    budget = int(settings.get('causal_support_tokens', settings.get('support_tokens', 256)))
    with torch.no_grad():
        try:
            for record in records:
                for noise in range(len(record['queries'])):
                    reference = _sample(record, noise, settings, budget, 2)
                    episodes = [('all', reference)]
                    if coverage_split:
                        positive = record['supervision']['positive'].flatten()
                        valid = record['supervision']['valid'].flatten().bool() & (positive >= 0)
                        first = set(reference['first_indices'])
                        groups = {
                            'covered': [i for i in torch.where(valid)[0].tolist()
                                        if int(positive[i]) in first],
                            'uncovered': [i for i in torch.where(valid)[0].tolist()
                                          if int(positive[i]) not in first],
                        }
                        coverage_counts.append(dict(scene_id=record['scene_id'],
                            noise_sigma=reference['noise_sigma'],
                            eligible=len(groups['covered']) + len(groups['uncovered']),
                            covered=len(groups['covered'])))
                        episodes = []
                        for group, indices in groups.items():
                            rng = random.Random(int(settings.get('eval_seed', 12345)) +
                                                noise + (10000 if group == 'uncovered' else 20000))
                            selected = rng.sample(indices, min(len(indices),
                                int(settings.get('query_tokens', 64))))
                            episodes.append((group, _sample(record, noise, settings, budget, 2,
                                selected, include_uncovered=True)))
                    for group, episode in episodes:
                        if not episode['positive'].numel():
                            continue
                        batch = collate([episode], device)
                        module.fusion = learned_fusion
                        scores = {'learned': evaluate_batch(module, batch, config, 'online')['query_mse'],
                                  'frozen_fast': evaluate_batch(module, batch, config,
                                                                'bank_frozen_fast')['query_mse'],
                                  'bank': evaluate_batch(module, batch, config, 'bank_read_only')['query_mse']}
                        module.fusion = fixed_fusion
                        for alpha in ALPHAS[:-1]:
                            logit = -30. if alpha == 0 else math.log(alpha / (1 - alpha))
                            fixed_fusion.bias.fill_(logit)
                            scores[f'alpha_{alpha:g}'] = evaluate_batch(module, batch, config,
                                                                         'online')['query_mse']
                        scores['alpha_1'] = scores['bank']
                        rows.append(dict(scene_id=record['scene_id'], group=group,
                                         query_count=len(episode['query_indices']),
                                         noise_sigma=episode['noise_sigma'], scores=scores))
        finally:
            module.fusion = learned_fusion
    if not rows:
        raise ValueError('No matched queries for fusion probe')
    groups = ('covered', 'uncovered') if coverage_split else ('all',)
    means = {group: {name: sum(row['scores'][name] for row in rows if row['group'] == group) /
                     sum(row['group'] == group for row in rows)
                     for name in rows[0]['scores']} for group in groups
             if any(row['group'] == group for row in rows)}
    counts = {group: sum(row['query_count'] for row in rows if row['group'] == group)
              for group in groups}
    report = dict(kind='sap_binding_fusion_probe', split=split, adapter=str(Path(adapter).resolve()),
                  support_tokens=budget, distractor_writes=2, coverage_split=coverage_split,
                  mean_query_mse=means if coverage_split else means['all'],
                  query_count=counts, valid_records=len(rows),
                  coverage=dict(eligible=sum(r['eligible'] for r in coverage_counts),
                                covered=sum(r['covered'] for r in coverage_counts),
                                fraction=sum(r['covered'] for r in coverage_counts) /
                                max(1, sum(r['eligible'] for r in coverage_counts)))
                  if coverage_split else None, rows=rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--coverage-split', action='store_true')
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    print(run(settings, args.adapter, args.output, split=args.split,
              coverage_split=args.coverage_split))


if __name__ == '__main__':
    main()
