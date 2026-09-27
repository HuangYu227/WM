"""Batched address + delayed-memory pretraining on frozen SANA features.

This is a mechanism stage, not video/flow training. Its adapter can initialize
sap_ttt.train, which trains the residual fusion using the actual backbone loss.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from .feature_probe import _choose, select_query_source
from .memory import SapMemoryState
from .runtime import SapBlockMemory, SapConfig


def sample_episode(record, noise_index, *, seed, support_budget=256, query_budget=64):
    if record.get('source') != 'teacher_forced_ground_truth':
        raise ValueError('Correspondence pretraining requires labeled real-history features')
    if min(support_budget, query_budget) < 1:
        raise ValueError('Token budgets must be positive')
    rng = random.Random(seed)
    # Choose writes before inspecting future labels. Never force future positives
    # into the historical bank as the old oracle mechanism probe did.
    ids = [sorted(rng.sample(range(p['visual'].shape[1]), min(support_budget, p['visual'].shape[1])))
           for p in record['supports']]
    supports = [_choose(record, i, indices) for i, indices in enumerate(ids)]
    labels = record['supervision']
    positive = labels['positive'].flatten()
    valid = labels['valid'].flatten().bool() & (positive >= 0)
    lookup = {position: local for local, position in enumerate(ids[0])}
    usable = [j for j in torch.where(valid)[0].tolist() if int(positive[j]) in lookup]
    chosen = sorted(rng.sample(usable, min(query_budget, len(usable))))
    raw = record['queries'][noise_index]
    query = {name: value[:, chosen] if name in {'visual', 'post', 'rays', 'sigma'} and value is not None else value
             for name, value in raw.items()}
    return dict(supports=supports, query=query,
                positive=torch.tensor([lookup[int(positive[j])] for j in chosen], dtype=torch.long),
                coverage=len(usable) / max(1, len(positive)),
                scene_id=record['scene_id'], noise_sigma=record['noise_sigmas'][noise_index])


def _stack_parts(parts, device, *, pad_queries=False):
    lengths = [p['visual'].shape[1] for p in parts]
    if not pad_queries and len(set(lengths)) != 1:
        raise ValueError('Batched supports must have equal sampled token counts')
    result = {}
    for key in ('visual', 'post', 'rays', 'sigma', 'text', 'text_mask'):
        if parts[0].get(key) is None:
            result[key] = None
        else:
            result[key] = pad_sequence([p[key][0] for p in parts], batch_first=True).to(device)
    result['valid'] = torch.arange(max(lengths), device=device)[None] < torch.tensor(lengths, device=device)[:, None]
    return result


def collate_episodes(samples, device):
    if not samples or any(len(s['positive']) == 0 for s in samples):
        raise ValueError('No matched queries after independent write sampling; increase support budget')
    support_count = len(samples[0]['supports'])
    if not support_count or any(len(s['supports']) != support_count for s in samples):
        raise ValueError('Batched episodes must have the same nonzero support count')
    return dict(supports=[_stack_parts([s['supports'][i] for s in samples], device)
                          for i in range(support_count)],
                query=_stack_parts([s['query'] for s in samples], device, pad_queries=True),
                positive=pad_sequence([s['positive'] for s in samples], batch_first=True).to(device),
                samples=samples)


def joint_objective(module, batch, config, *, training, online=True,
                    address_weight=.1, write_weight=.01):
    address, memory = module.address, module.memory
    batch_size = batch['positive'].shape[0]
    state = SapMemoryState.new(memory, 'joint', batch_size, training=training)
    keys, values, writes = [], [], []
    for part in batch['supports']:
        key = address.write(part['visual'], part['text'], part['text_mask'], part['rays'], mode=config.address_mode)
        value = address.value_for(part['post']).detach()
        keys.append(key); values.append(value)
        if online:
            state.weight, loss = memory.update(state.weight, key, value, create_graph=training)
            if not training:
                state.weight = state.weight.detach()
            writes.append(loss / batch_size)
    part = batch['query']
    q = address.query(part['visual'], part['text'], part['text_mask'], part['rays'], part['sigma'], mode=config.address_mode)
    readout = memory.read(state, q)
    target = values[0].gather(1, batch['positive'][..., None].expand(-1, -1, config.dim))
    valid = part['valid']
    def per_episode_average(x):
        return (x * valid).sum(1) / valid.sum(1)
    delayed = per_episode_average((readout - target).square().mean(-1))
    bank = torch.cat(keys, dim=1)
    logits = torch.bmm(q, bank.transpose(1, 2)) / .07
    ce = F.cross_entropy(logits.transpose(1, 2), batch['positive'], reduction='none')
    address_loss = per_episode_average(ce).mean()
    write = torch.stack(writes).mean() if writes else delayed.new_zeros(())
    loss = delayed.mean() + address_weight * address_loss + write_weight * write
    with torch.no_grad():
        exact = per_episode_average((logits.argmax(-1) == batch['positive']).float())
        old = (memory.read(state, keys[0]) - values[0]).square().mean((1, 2))
        new = (memory.read(state, keys[-1]) - values[-1]).square().mean((1, 2))
    return loss, dict(query_mse=float(delayed.mean().detach()), address_ce=float(address_loss.detach()),
        exact_top1=float(exact.mean()), old_a_mse=float(old.mean()), new_write_mse=float(new.mean()),
        matched_queries=int(valid.sum()), candidates=bank.shape[1],
        coverage=sum(s['coverage'] for s in batch['samples']) / batch_size,
        write_objective=float(write.detach()))


def _evaluate(module, records, config, options, device):
    rows = []
    module.eval()
    for record in records:
        for noise in range(len(record['queries'])):
            sample = sample_episode(record, noise, seed=int(options.get('val_seed', 12345)) + noise,
                support_budget=int(options.get('support_tokens', 256)), query_budget=int(options.get('query_tokens', 64)))
            if not len(sample['positive']):
                rows.append(dict(scene_id=record['scene_id'], noise_sigma=sample['noise_sigma'],
                    valid=False, reason='no_correspondence_survived_independent_writes', coverage=sample['coverage']))
                continue
            batch = collate_episodes([sample], device)
            for online in (False, True):
                with torch.no_grad():
                    _, metric = joint_objective(module, batch, config, training=False, online=online)
                rows.append(dict(scene_id=record['scene_id'], noise_sigma=sample['noise_sigma'], valid=True,
                                 mode='online' if online else 'frozen', **metric))
    return rows


def run(settings, output):
    config = SapConfig(**settings['sap'])
    if len(config.layers) != 1:
        raise ValueError('Feature pretraining currently requires one selected layer')
    source = settings.get('query_source', 'native')
    records = {split: [select_query_source(torch.load(p, map_location='cpu', weights_only=True), source)
                      for p in settings[split + '_features']] for split in ('train', 'val', 'test')}
    sets = {s: {r['scene_id'] for r in group} for s, group in records.items()}
    if any(not group or len(sets[s]) != len(group) for s, group in records.items()) or any(
            sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Require nonempty scene-disjoint train/val/test features without duplicates')
    sample = records['train'][0]['supports'][0]
    checkpoint = records['train'][0]['base_checkpoint']
    for group in records.values():
        for record in group:
            if record['base_checkpoint'] != checkpoint or record['layer'] != config.layers[0]:
                raise ValueError('Feature backbone/layer mismatch')
            for part in (*record['supports'], *record['queries']):
                if (part['visual'].shape[-1] != sample['visual'].shape[-1] or
                        part['text'].shape[-1] != sample['visual'].shape[-1] or
                        part['rays'].shape[-1] != config.ray_dim):
                    raise ValueError('Feature dimensions differ from runtime protocol')
    steps, batch_size, val_every = (int(settings.get(k, default)) for k, default in
                                   (('steps', 300), ('batch_size', 4), ('val_every', 25)))
    if min(steps, batch_size, val_every) < 1:
        raise ValueError('Training intervals and batch_size must be positive')
    output = Path(output)
    if any((output / name).exists() for name in ('last.pt', 'best.pt')):
        raise FileExistsError('Use a fresh joint pretraining output directory')
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = int(settings.get('seed', 3407)); torch.manual_seed(seed)
    module = SapBlockMemory(sample['visual'].shape[-1], config).to(device)
    # Feature losses cannot train the video residual. Keep its initialized weights
    # in the adapter for the later flow stage, explicitly label this checkpoint.
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                 lr=float(settings.get('lr', 1e-4)))
    (output / 'resolved.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    def save(name, step, validation):
        payload = dict(version=1, config=asdict(config), base_checkpoint=checkpoint,
            modules={config.layers[0]: module.state_dict()}, extra=dict(stage='feature_joint', step=step,
            flow_trained=False, validation=validation, query_source=source, settings=settings))
        temporary = output / (name + '.tmp.pt'); torch.save(payload, temporary)
        temporary.replace(output / (name + '.pt'))
    started = time.perf_counter(); best = math.inf
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    progress = tqdm(range(steps), desc='SAP joint features', unit='step')
    for step in progress:
        module.train()
        selected = []
        for j in range(batch_size):
            index = step * batch_size + j
            record = records['train'][index % len(records['train'])]
            # Noise cycles independently of scene ordering.
            noise = random.Random(seed + index).randrange(len(record['queries']))
            selected.append(sample_episode(record, noise, seed=seed + index,
                support_budget=int(settings.get('support_tokens', 256)), query_budget=int(settings.get('query_tokens', 64))))
        batch = collate_episodes(selected, device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = joint_objective(module, batch, config, training=True,
            address_weight=float(settings.get('address_weight', .1)), write_weight=float(settings.get('write_weight', .01)))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite joint outer loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        validation = None
        if (step + 1) % val_every == 0 or step + 1 == steps:
            validation = _evaluate(module, records['val'], config, settings, device)
            scores = [r['query_mse'] for r in validation if r.get('mode') == 'online']
            if not scores:
                raise ValueError('No valid validation queries; cannot select best checkpoint')
            score = sum(scores) / len(scores)
            if score < best:
                best = score; save('best', step + 1, validation)
        save('last', step + 1, validation)
        row = dict(step=step + 1, loss=float(loss.detach()), gradient_norm=float(norm), **metrics,
                   best_val_query_mse=best if math.isfinite(best) else None,
                   elapsed_seconds=time.perf_counter() - started, batch_size=batch_size,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if validation is not None:
            with (output / 'val.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step + 1, rows=validation), allow_nan=False) + '\n')
        progress.set_postfix(loss=f'{row["loss"]:.4g}', query=f'{metrics["query_mse"]:.4g}')
    payload = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    module.load_state_dict(payload['modules'][config.layers[0]])
    rows = _evaluate(module, records['test'], config, settings, device)
    (output / 'test_results.json').write_text(json.dumps(dict(stage='feature_mechanism_only',
        selected_step=payload['extra']['step'], query_source=source, rows=rows), indent=2, allow_nan=False), encoding='utf-8')
    return output / 'best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(run(json.loads(Path(args.settings).read_text(encoding='utf-8')), args.output))


if __name__ == '__main__':
    main()
