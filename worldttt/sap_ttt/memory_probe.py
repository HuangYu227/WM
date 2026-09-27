"""Train delayed preservation with a fixed address encoder; audit replay costs."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from .address import SapAddress
from .feature_probe import _choose, _move
from .memory import SapMemory
from .probe import delayed_episode_loss, forgetting_curve


def sample_delayed_batch(record, noise_index, *, seed, query_budget=32, support_budget=256):
    labels = record['supervision']
    positive = labels['positive'].flatten()
    valid = labels['valid'].flatten().bool() & (positive >= 0)
    positions = torch.where(valid)[0].tolist()
    if not positions:
        raise ValueError('No delayed A-to-A-prime correspondences')
    rng = random.Random(seed)
    selected = sorted(rng.sample(positions, min(query_budget, len(positions))))
    mandatory = sorted(set(int(positive[i]) for i in selected))
    if len(mandatory) > support_budget:
        selected = selected[:support_budget]
        mandatory = sorted(set(int(positive[i]) for i in selected))
    old_count = record['supports'][0]['visual'].shape[1]
    extra = rng.sample([i for i in range(old_count) if i not in mandatory],
                       min(support_budget - len(mandatory), old_count - len(mandatory)))
    old = sorted(mandatory + extra)
    supports = [_choose(record, 0, old)]
    for i in (1, 2):
        count = record['supports'][i]['visual'].shape[1]
        ids = sorted(rng.sample(range(count), min(count, support_budget)))
        supports.append(_choose(record, i, ids))
    query_raw = record['queries'][noise_index]
    query = {name: value[:, selected] if name in {'visual', 'post', 'rays', 'sigma'} and value is not None else value
             for name, value in query_raw.items()}
    old_lookup = {index: local for local, index in enumerate(old)}
    return supports, query, torch.tensor([old_lookup[int(positive[i])] for i in selected])


def _evaluate(record, address, memory, mode, *, seed, replay='none', capacity=64):
    scores = []
    for noise in range(len(record['queries'])):
        supports, query, positive = sample_delayed_batch(record, noise, seed=seed + noise)
        supports = [_move(p, next(address.parameters()).device) for p in supports]
        query = _move(query, next(address.parameters()).device)
        with torch.no_grad():
            pairs = [(address.write(p['visual'], p['text'], p['text_mask'], p['rays'], mode=mode),
                      address.value_for(p['post'])) for p in supports]
            q = address.query(query['visual'], query['text'], query['text_mask'],
                              query['rays'], query['sigma'], mode=mode)
        curve, state = forgetting_curve(memory, pairs, replay=replay, replay_capacity=capacity,
                                        seed=seed, return_state=True)
        with torch.no_grad():
            estimated = memory.read(state, q)
            target = pairs[0][1][:, positive.to(q.device)]
            query_mse = float(F.mse_loss(estimated, target))
        scores.append({'noise_sigma': record['noise_sigmas'][noise], 'query_mse': query_mse,
                       'old_a_mse': curve[-1]['old_key_mse'][0],
                       'new_write_mse': curve[-1]['new_write_mse'],
                       'curve': curve, 'state_bytes': curve[-1]['state_bytes'],
                       'replay_bytes': curve[-1]['replay_bytes']})
    return scores


def run(settings, output):
    address_file = torch.load(settings['address_checkpoint'], map_location='cpu', weights_only=True)
    mode = address_file['mode']
    training = [torch.load(p, map_location='cpu', weights_only=True) for p in settings['train_features']]
    testing = [torch.load(p, map_location='cpu', weights_only=True) for p in settings['test_features']]
    if not training or not testing or {r['scene_id'] for r in training} & {r['scene_id'] for r in testing}:
        raise ValueError('Persistence probe requires disjoint train/test scenes')
    first = training[0]['supports'][0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    address = SapAddress(first['visual'].shape[-1], first['text'].shape[-1],
                         first['rays'].shape[-1], address_file['dim']).to(device)
    address.load_state_dict(address_file['address'])
    address.requires_grad_(False)
    memory = SapMemory(address_file['dim'], lr=float(settings.get('inner_lr', .5))).to(device)
    ordinary = copy.deepcopy(memory.state_dict())
    optimizer = torch.optim.AdamW(memory.parameters(), lr=float(settings.get('outer_lr', 1e-4)))
    for step in range(int(settings.get('steps', 300))):
        record = training[step % len(training)]
        supports, query, positive = sample_delayed_batch(
            record, step % len(record['queries']), seed=int(settings.get('seed', 3407)) + step)
        supports = [_move(p, device) for p in supports]
        query = _move(query, device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = delayed_episode_loss(address, memory, supports, query, positive.to(device),
                                       mode=mode, address_weight=0.)
        loss.backward()
        optimizer.step()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    torch.save({'version': 1, 'mode': mode, 'address': address.state_dict(),
                'memory': memory.state_dict(), 'dim': address_file['dim'],
                'base_checkpoint': address_file['base_checkpoint']}, output / 'delayed.pt')
    rows = []
    trained = copy.deepcopy(memory.state_dict())
    for name, weights in (('ordinary', ordinary), ('delayed', trained)):
        memory.load_state_dict(weights)
        for policy in ('none', 'anchor', 'random'):
            for record in testing:
                rows.append({'scene_id': record['scene_id'], 'memory': name, 'replay': policy,
                             'noise': _evaluate(record, address, memory, mode,
                                                seed=int(settings.get('seed', 3407)),
                                                replay=policy,
                                                capacity=int(settings.get('replay_capacity', 64)))})
    (output / 'persistence.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Delayed SAP-TTT persistence and bounded replay probe')
    parser.add_argument('--settings', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.settings).read_text(encoding='utf-8')), args.output)


if __name__ == '__main__':
    main()
