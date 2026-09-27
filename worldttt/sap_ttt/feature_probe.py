"""Frozen-feature address ablations with matched parameter counts and budgets."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from .address import SapAddress


def _choose(record, support, indices):
    part = record['supports'][support]
    return {name: value[:, indices] if name in {'visual', 'post', 'rays', 'sigma'} and value is not None else value
            for name, value in part.items()}


def select_query_source(record, source):
    if source == 'native':
        return record
    if source != 'no_history':
        raise ValueError(f'Unknown query_source: {source}')
    if 'queries_no_history' not in record:
        raise ValueError('Feature record lacks queries_no_history; re-extract features')
    return dict(record, queries=record['queries_no_history'])


def select_batch(record, *, noise_index=0, max_queries=64, random_negatives=128, seed=0):
    if record.get('source') != 'teacher_forced_ground_truth':
        raise ValueError('Address labels are valid only for teacher-forced procedural records')
    labels = record['supervision']
    positives = labels['positive'].flatten()
    valid = labels['valid'].flatten().bool() & (positives >= 0)
    query_id = labels['query_instance'].flatten()
    valid &= query_id > 0
    rng = random.Random(seed)
    choices = torch.where(valid)[0].tolist()
    if not choices:
        raise ValueError('No evaluable historical instance queries')
    selected = sorted(rng.sample(choices, min(max_queries, len(choices))))
    old_count = record['supports'][0]['visual'].shape[1]
    segment_lengths = [p['visual'].shape[1] for p in record['supports']]
    support_id = labels['support_instance'].flatten()
    if segment_lengths != [4 * labels['support_instance'].shape[1] * labels['support_instance'].shape[2],
                           3 * labels['support_instance'].shape[1] * labels['support_instance'].shape[2],
                           3 * labels['support_instance'].shape[1] * labels['support_instance'].shape[2]]:
        raise ValueError('Support labels do not align with SANA frame/token order')
    mandatory = sorted(set(int(positives[i]) for i in selected))
    if max(mandatory) >= old_count:
        raise ValueError('Historical correspondence exceeds A token count')
    other = [j for j in range(sum(segment_lengths)) if j not in mandatory and support_id[j] > 0]
    negatives = rng.sample(other, min(random_negatives, len(other)))
    positions = sorted(mandatory + negatives)
    offsets = [0, segment_lengths[0], sum(segment_lengths[:2]), sum(segment_lengths)]
    groups = []
    for i in range(3):
        local = [j - offsets[i] for j in positions if offsets[i] <= j < offsets[i + 1]]
        if local:
            groups.append(_choose(record, i, local))
    query = record['queries'][noise_index]
    query = {name: value[:, selected] if name in {'visual', 'post', 'rays', 'sigma'} and value is not None else value
             for name, value in query.items()}
    candidate_index = {position: index for index, position in enumerate(positions)}
    return dict(query=query, candidates=groups, query_instance=query_id[selected],
                candidate_instance=support_id[positions],
                positive_candidate=torch.tensor([candidate_index[int(positives[i])] for i in selected]),
                query_indices=torch.tensor(selected), candidate_indices=torch.tensor(positions),
                spatial_shape=tuple(labels['support_instance'].shape[1:]),
                coverage=len(choices) / len(valid), scene_id=record['scene_id'])


def position_baseline(batch):
    """Audit whether pixel position alone solves the selected retrieval task."""
    height, width = batch['spatial_shape']
    span = height * width
    query_pixels = batch['query_indices'] % span
    candidate_pixels = batch['candidate_indices'] % span
    query_xy = torch.stack((query_pixels // width, query_pixels % width), dim=-1)
    candidate_xy = torch.stack((candidate_pixels // width, candidate_pixels % width), dim=-1)
    distance = (query_xy[:, None] - candidate_xy[None]).square().sum(-1)
    nearest = distance.argmin(dim=1)
    same = batch['query_instance'][:, None] == batch['candidate_instance'][None]
    return {'instance_top1': float(same.gather(1, nearest[:, None]).float().mean()),
            'exact_top1': float((nearest == batch['positive_candidate']).float().mean()),
            'random_instance_top1': float(same.float().mean(1).mean()),
            'random_exact_top1': 1 / len(candidate_pixels),
            'queries': len(query_pixels), 'candidates': len(candidate_pixels),
            'coverage': batch['coverage'] if 'coverage' in batch else 1.}


def _move(part, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in part.items()}


def shuffled_query_text(batch, donor):
    """Perturb the query semantics while leaving committed history untouched."""
    return dict(batch, query=dict(batch['query'], text=donor['text'],
                                  text_mask=donor['text_mask']))


def address_objective(address: SapAddress, batch, *, mode='selective_geometry', temperature=.07):
    device = next(address.parameters()).device
    query = _move(batch['query'], device)
    candidates = [_move(part, device) for part in batch['candidates']]
    q = address.query(query['visual'], query['text'], query['text_mask'],
                      query['rays'], query['sigma'], mode=mode)[0]
    keys = torch.cat([address.write(p['visual'], p['text'], p['text_mask'],
                                    p['rays'], mode=mode) for p in candidates], 1)[0]
    query_id = batch['query_instance'].to(device)
    key_id = batch['candidate_instance'].to(device)
    scores = q.float() @ keys.float().T
    positives = (query_id[:, None] == key_id[None]) & (query_id[:, None] > 0)
    if not positives.any(dim=1).all():
        raise ValueError('Address batch has a query with no historical positive')
    logits = scores / temperature
    instance_loss = (torch.logsumexp(logits, 1) -
                     torch.logsumexp(logits.masked_fill(~positives, float('-inf')), 1)).mean()
    exact_target = batch['positive_candidate'].to(device)
    loss = instance_loss + F.cross_entropy(logits, exact_target)
    with torch.no_grad():
        ranking = scores.argsort(dim=1, descending=True)
        top1 = (key_id[ranking[:, 0]] == query_id).float().mean()
        top5 = (key_id[ranking[:, :min(5, len(key_id))]] == query_id[:, None]).any(1).float().mean()
        exact = (ranking[:, 0] == exact_target).float().mean()
        best_true = scores.masked_fill(~positives, float('-inf')).max(1).values
        best_false = scores.masked_fill(positives, float('-inf')).max(1).values
    return loss, dict(instance_top1=float(top1), instance_top5=float(top5),
                      exact_top1=float(exact), margin=float((best_true - best_false).mean()),
                      coverage=batch['coverage'], queries=len(query_id))


def run(settings, output):
    train_paths = [Path(p) for p in settings['train_features']]
    test_paths = [Path(p) for p in settings['test_features']]
    source = settings.get('query_source', 'native')
    training = [select_query_source(torch.load(p, map_location='cpu', weights_only=True), source)
                for p in train_paths]
    testing = [select_query_source(torch.load(p, map_location='cpu', weights_only=True), source)
               for p in test_paths]
    if not training or not testing or {r['scene_id'] for r in training} & {r['scene_id'] for r in testing}:
        raise ValueError('Need nonempty scene-disjoint train/test feature records')
    sample = training[0]['supports'][0]
    dim_in, text_dim, ray_dim = (sample['visual'].shape[-1], sample['text'].shape[-1], sample['rays'].shape[-1])
    dim = int(settings.get('dim', 256))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(int(settings.get('seed', 3407)))
    template = SapAddress(dim_in, text_dim, ray_dim, dim)
    initial = copy.deepcopy(template.state_dict())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    baseline = []
    for record in testing:
        for noise in range(len(record['queries'])):
            batch = select_batch(record, noise_index=noise,
                                 seed=int(settings.get('seed', 3407)) + noise)
            baseline.append({'scene_id': record['scene_id'],
                             'noise_sigma': record['noise_sigmas'][noise],
                             **position_baseline(batch)})
    (output / 'baseline_results.json').write_text(
        json.dumps(baseline, indent=2, allow_nan=False), encoding='utf-8')
    summary = []
    for mode in ('vision', 'global', 'selective', 'selective_geometry'):
        address = SapAddress(dim_in, text_dim, ray_dim, dim).to(device)
        address.load_state_dict(initial)
        optimizer = torch.optim.AdamW((p for p in address.parameters() if p.requires_grad),
                                       lr=float(settings.get('lr', 1e-4)))
        for step in range(int(settings.get('steps', 300))):
            record = training[step % len(training)]
            batch = select_batch(record, noise_index=step % len(record['queries']),
                                 seed=int(settings.get('seed', 3407)) + step)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = address_objective(address, batch, mode=mode)
            loss.backward()
            optimizer.step()
        rows = []
        for scene_index, record in enumerate(testing):
            for noise in range(len(record['queries'])):
                batch = select_batch(record, noise_index=noise,
                                     seed=int(settings.get('seed', 3407)) + noise)
                with torch.no_grad():
                    _, metric = address_objective(address, batch, mode=mode)
                rows.append(dict(scene_id=record['scene_id'], noise_sigma=record['noise_sigmas'][noise],
                                 control='normal', **metric))
                donors = [r['queries'][noise] for i, r in enumerate(testing) if i != scene_index
                          and not torch.equal(r['queries'][noise]['text'], record['queries'][noise]['text'])]
                if donors:
                    shuffled = shuffled_query_text(batch, donors[0])
                    with torch.no_grad():
                        _, randomized = address_objective(address, shuffled, mode=mode)
                    rows.append(dict(scene_id=record['scene_id'], noise_sigma=record['noise_sigmas'][noise],
                                     control='shuffled_text', **randomized))
                else:
                    rows.append(dict(scene_id=record['scene_id'], noise_sigma=record['noise_sigmas'][noise],
                                     control='shuffled_text_unavailable', reason='no_distinct_text_donor'))
        torch.save({'mode': mode, 'address': address.state_dict(), 'dim': dim,
                    'base_checkpoint': testing[0]['base_checkpoint'], 'steps': settings.get('steps', 300)},
                   output / f'{mode}.pt')
        summary.append({'mode': mode, 'rows': rows})
    (output / 'address_results.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description='Compare semantic/geometric SAP addressing on frozen SANA features')
    parser.add_argument('--settings', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.settings).read_text(encoding='utf-8')), args.output)


if __name__ == '__main__':
    main()
