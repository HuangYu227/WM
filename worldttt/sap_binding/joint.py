"""Batched SAP-Bind mechanism training with explicit anti-shortcut losses."""
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

from worldttt.sap_ttt.memory import SapMemoryState

from .config import BindingConfig
from .memory import BindingBankState, BindingState, binding_scores, explicit_read
from .model import BindingBlock


def _load_records(paths):
    records = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    if any(r.get('source') != 'sap_binding_teacher_forced_ground_truth' or not r.get('query_obscured')
           for r in records):
        raise ValueError('SAP-Bind training requires obscured, labeled feature records')
    return records


def attach_values(records, encoder):
    device = encoder.hidden_projection.device
    with torch.no_grad():
        for record in records:
            for part in record['supports']:
                latent = part['latent'].to(device).float(); frames, height, width = latent.shape[2:]
                part['binding_value'] = encoder(part['post'].to(device).float(), latent,
                                                frames, height, width).cpu()


def _select(part, ids):
    token_fields = {'visual', 'post', 'rays', 'sigma', 'binding_value'}
    return {key: (value[:, ids] if key in token_fields and value is not None else value)
            for key, value in part.items() if key != 'latent'}


def sample_episode(record, noise_index, *, seed, support_tokens=512, query_tokens=64,
                   distractor_writes=2, protected=128, forced_query_indices=None,
                   include_uncovered=False):
    if not 0 <= distractor_writes <= 8 or min(support_tokens, query_tokens) < 1:
        raise ValueError('Invalid binding episode budgets')
    rng = random.Random(seed)
    first = record['supports'][0]
    first_count = support_tokens if distractor_writes == 0 else min(protected, max(1, support_tokens // 2))
    first_ids = sorted(rng.sample(range(first['visual'].shape[1]),
                                  min(first_count, first['visual'].shape[1])))
    lookup = {source: local for local, source in enumerate(first_ids)}
    labels = record['supervision']; positive = labels['positive'].flatten()
    valid = labels['valid'].flatten().bool() & (positive >= 0)
    usable = [i for i in torch.where(valid)[0].tolist() if int(positive[i]) in lookup]
    available = torch.where(valid)[0].tolist() if include_uncovered else usable
    if forced_query_indices is None:
        chosen = sorted(rng.sample(available, min(query_tokens, len(available))))
    else:
        chosen = sorted(set(int(i) for i in forced_query_indices) & set(available))
    query_raw = record['queries'][noise_index]
    query = {key: (value[:, chosen] if key in {'visual', 'post', 'rays', 'sigma'} and value is not None else value)
             for key, value in query_raw.items()}
    writes = [_select(first, first_ids)]
    remaining = support_tokens - len(first_ids)
    if distractor_writes:
        base, extra = divmod(remaining, distractor_writes)
        for index in range(distractor_writes):
            source = record['supports'][1 + index % 2]
            count = min(source['visual'].shape[1], base + (index < extra))
            if not count:
                continue
            ids = sorted(rng.sample(range(source['visual'].shape[1]), count))
            writes.append(_select(source, ids))
    candidates = {key: torch.cat([w[key] for w in writes], 1)
                  for key in ('visual', 'rays', 'binding_value')}
    ages = torch.cat([torch.full((w['visual'].shape[1],), i, dtype=torch.float32)
                      for i, w in enumerate(writes)])
    target = first['binding_value'][:, [int(positive[i]) for i in chosen]]
    return dict(writes=writes, candidates=candidates, query=query, target=target,
        positive=torch.tensor([lookup.get(int(positive[i]), -1) for i in chosen], dtype=torch.long),
        ages=ages, coverage=len(usable) / max(1, len(positive)), scene_id=record['scene_id'],
        layout_id=record['layout_id'], noise_sigma=record['noise_sigmas'][noise_index],
        distractor_writes=distractor_writes, query_indices=chosen, first_indices=first_ids)


def _stack(parts, device, *, pad=False):
    lengths = [p['visual'].shape[1] for p in parts]
    if not pad and len(set(lengths)) != 1:
        raise ValueError('Unpadded binding batch fields must have equal token counts')
    result = {}
    for key in ('visual', 'post', 'rays', 'sigma', 'text', 'text_mask', 'binding_value'):
        if key not in parts[0] or parts[0].get(key) is None:
            continue
        rows = [p[key][0] for p in parts]
        result[key] = (pad_sequence(rows, batch_first=True) if pad else torch.stack(rows)).to(device)
    result['valid'] = torch.arange(max(lengths), device=device)[None] < torch.tensor(lengths, device=device)[:, None]
    return result


def collate(samples, device):
    if not samples or any(not len(s['positive']) for s in samples):
        raise ValueError('No exact correspondence survived independent historical sampling')
    write_count = len(samples[0]['writes'])
    if any(len(s['writes']) != write_count for s in samples):
        raise ValueError('Batch must share one distractor count')
    return dict(writes=[_stack([s['writes'][i] for s in samples], device) for i in range(write_count)],
        candidates=_stack([s['candidates'] for s in samples], device),
        query=_stack([s['query'] for s in samples], device, pad=True),
        target=pad_sequence([s['target'][0] for s in samples], batch_first=True).to(device),
        positive=pad_sequence([s['positive'] for s in samples], batch_first=True).to(device),
        ages=torch.stack([s['ages'] for s in samples]).to(device), samples=samples)


def _per_episode(values, valid):
    return (values * valid).sum(1) / valid.sum(1).clamp_min(1)


def binding_objective(module, batch, config, weights, *, training, online=True):
    b = batch['positive'].shape[0]
    fast = SapMemoryState.new(module.fast, 'joint', b, training=training)
    fast_k = SapMemoryState.new(module.fast, 'shuffle-k', b, training=training)
    fast_v = SapMemoryState.new(module.fast, 'shuffle-v', b, training=training)
    write_losses, keys, values, rays = [], [], [], []
    for part in batch['writes']:
        key = module.address.write(part['visual'], part['text'], part['text_mask'], part['rays'],
                                   mode='selective_geometry')
        value = part['binding_value'].float().detach()
        keys.append(key); values.append(value); rays.append(part['rays'].float())
        if online and config.architecture != 'bank_only':
            fast.weight, loss = module.fast.update(fast.weight, key, value, create_graph=training)
            fast_k.weight, _ = module.fast.update(fast_k.weight, key.roll(1, 1), value,
                                                  create_graph=training)
            fast_v.weight, _ = module.fast.update(fast_v.weight, key, value.roll(1, 1),
                                                  create_graph=training)
            if not training:
                fast.weight = fast.weight.detach()
                fast_k.weight = fast_k.weight.detach()
                fast_v.weight = fast_v.weight.detach()
            write_losses.append((loss if training else loss.detach()) / b)
    candidate_key = torch.cat(keys, 1)
    candidate_value = torch.cat(values, 1)
    candidate_ray = torch.cat(rays, 1)
    valid_candidates = batch['candidates']['valid']
    bank = BindingBankState(candidate_key, candidate_value, candidate_ray,
        torch.ones_like(valid_candidates, dtype=torch.float32), batch['ages'], valid_candidates,
        torch.full((b,), candidate_key.shape[1], device=candidate_key.device, dtype=torch.long),
        torch.zeros_like(valid_candidates))
    state = BindingState('joint', bank, fast, len(keys) - 1, len(keys) if online else 0)
    qpart = batch['query']
    query = module.address.query(qpart['visual'], qpart['text'], qpart['text_mask'],
                                 qpart['rays'], qpart['sigma'], mode='selective_geometry')
    query_valid = qpart['valid']
    target = batch['target']
    score = binding_scores(bank, query, qpart['rays'], module.ray_query, module.ray_key,
                           heads=config.heads).mean(1)
    address_ce = _per_episode(F.cross_entropy(score.transpose(1, 2), batch['positive'],
                                               reduction='none'), query_valid).mean()
    explicit, _, _ = module.read(state, query, qpart['rays'], qpart['sigma'])
    direct, _ = explicit_read(bank, query, qpart['rays'], module.ray_query, module.ray_key,
                              heads=config.heads, topk=config.topk)
    fast_value = module.fast.read(fast, query) if config.architecture != 'bank_only' else None
    bank_k = BindingBankState(torch.cat([key.roll(1, 1) for key in keys], 1), candidate_value,
        torch.cat([ray.roll(1, 1) for ray in rays], 1),
        bank.confidence, bank.age, bank.valid, bank.seen, bank.protected)
    bank_v = BindingBankState(candidate_key,
        torch.cat([value.roll(1, 1) for value in values], 1), candidate_ray,
        bank.confidence, bank.age, bank.valid, bank.seen, bank.protected)
    shuffled_k, _, _ = module.read(BindingState('shuffle-k', bank_k, fast_k, len(keys) - 1),
                                   query, qpart['rays'], qpart['sigma'])
    shuffled_v, _, _ = module.read(BindingState('shuffle-v', bank_v, fast_v, len(keys) - 1),
                                   query, qpart['rays'], qpart['sigma'])
    mean = (candidate_value * valid_candidates[..., None]).sum(1, keepdim=True)
    mean = mean / valid_candidates.sum(1, keepdim=True)[..., None].clamp_min(1)
    mean = mean.expand(-1, query.shape[1], -1)
    def mse(value):
        return _per_episode((value - target).square().mean(-1), query_valid)
    def value_loss(value):
        smooth = F.smooth_l1_loss(value, target, reduction='none').mean(-1)
        cosine = 1 - F.cosine_similarity(value, target, dim=-1)
        return _per_episode(smooth + cosine, query_valid).mean()
    correct_error, sk_error, sv_error, mean_error = mse(explicit), mse(shuffled_k), mse(shuffled_v), mse(mean)
    fast_loss = value_loss(fast_value) if fast_value is not None else query.new_zeros(())
    value = value_loss(direct) if config.architecture != 'fast_only' else query.new_zeros(())
    shuffle_margin = (F.relu(correct_error - .8 * sk_error) +
                      F.relu(correct_error - .8 * sv_error)).mean()
    mean_margin = F.relu(correct_error - .85 * mean_error).mean()
    write = torch.stack(write_losses).mean() if write_losses else query.new_zeros(())
    loss = (weights['addr'] * address_ce + weights['value'] * value +
            weights['fast'] * fast_loss + weights['shuffle'] * shuffle_margin +
            weights['mean'] * mean_margin + weights['write'] * write +
            weights.get('fused', 0.) * correct_error.mean())
    with torch.no_grad():
        exact = _per_episode((score.argmax(-1) == batch['positive']).float(), query_valid)
    metrics = dict(query_mse=float(correct_error.mean().detach()),
        shuffle_key_mse=float(sk_error.mean().detach()), shuffle_value_mse=float(sv_error.mean().detach()),
        mean_value_mse=float(mean_error.mean().detach()), address_ce=float(address_ce.detach()),
        value_loss=float(value.detach()), fast_loss=float(fast_loss.detach()),
        fused_loss=float(correct_error.mean().detach()),
        shuffle_margin=float(shuffle_margin.detach()), mean_margin=float(mean_margin.detach()),
        write_objective=float(write.detach()), exact_top1=float(exact.mean()),
        matched_queries=int(query_valid.sum()), candidates=int(valid_candidates.sum(1).float().mean()),
        coverage=sum(s['coverage'] for s in batch['samples']) / b)
    return loss, metrics


def _evaluate(module, records, config, settings, device):
    module.eval(); rows = []
    for record in records:
        for noise in range(len(record['queries'])):
            sample = sample_episode(record, noise, seed=int(settings.get('val_seed', 12345)) + noise,
                support_tokens=int(settings.get('support_tokens', 512)),
                query_tokens=int(settings.get('query_tokens', 64)), distractor_writes=2,
                protected=int(settings.get('protected_anchors', 128)))
            if not len(sample['positive']):
                rows.append(dict(scene_id=record['scene_id'], valid=False,
                                 reason='no_exact_correspondence_after_independent_sampling'))
                continue
            batch = collate([sample], device)
            with torch.no_grad():
                _, metric = binding_objective(module, batch, config, settings['loss_weights'],
                                               training=False, online=True)
            rows.append(dict(scene_id=record['scene_id'], layout_id=record['layout_id'],
                             noise_sigma=sample['noise_sigma'], valid=True, **metric))
    return rows


def _selection(rows):
    valid = [r for r in rows if r.get('valid')]
    if not valid:
        return math.inf
    mean = lambda key: sum(r[key] for r in valid) / len(valid)
    correct = mean('query_mse')
    return (correct / max(mean('shuffle_key_mse'), 1e-8) +
            correct / max(mean('shuffle_value_mse'), 1e-8) +
            correct / max(mean('mean_value_mse'), 1e-8) +
            (1 - mean('exact_top1')) + correct)


def run(settings, output):
    config = BindingConfig(**settings['sap_binding'])
    if len(config.layers) != 1:
        raise ValueError('SAP-Bind feature pretraining requires one selected layer')
    records = {split: _load_records(settings[split + '_features']) for split in ('train', 'val')}
    if any(record.get('layer') != config.layers[0] for group in records.values() for record in group):
        raise ValueError('SAP-Bind training feature layer mismatch')
    layouts = {split: {r['layout_id'] for r in group} for split, group in records.items()}
    if any(not group for group in records.values()) or layouts['train'] & layouts['val']:
        raise ValueError('SAP-Bind requires nonempty layout-disjoint splits')
    value_stats = torch.load(settings['value_stats'], map_location='cpu', weights_only=True)
    if value_stats.get('kind') != 'sap_binding_value_stats' or value_stats.get('mode') != config.value_mode:
        raise ValueError('Value statistics and SAP-Bind configuration disagree')
    if value_stats.get('layer') is not None and value_stats['layer'] != config.layers[0]:
        raise ValueError('SAP-Bind Value statistics layer mismatch')
    # The centroid audit is diagnostic: cross-view Values need not be close when
    # learned Q/K addresses retrieve the completed historical Value.
    if set(value_stats.get('train_layouts', ())) != layouts['train']:
        raise ValueError('Value whitening statistics must come from this training split')
    seed = int(settings.get('seed', 3407))
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    sample = records['train'][0]['supports'][0]
    module = BindingBlock(sample['visual'].shape[-1], sample['latent'].shape[1], config)
    module.value.load_state_dict(value_stats['encoder'], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); module.to(device)
    attach_values([r for group in records.values() for r in group], module.value)
    steps, batch_size, val_every = (int(settings.get(key, default)) for key, default in
        (('steps', 600), ('batch_size', 16), ('val_every', 50)))
    if min(steps, batch_size, val_every) < 1:
        raise ValueError('Invalid feature training schedule')
    output = Path(output)
    if any((output / name).exists() for name in ('best.pt', 'last.pt')):
        raise FileExistsError('Use a fresh SAP-Bind output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                  lr=float(settings.get('lr', 1e-4)))
    best = math.inf; started = time.perf_counter()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    def save(name, step, validation, score):
        payload = dict(version=1, kind='sap_binding_adapter', config=asdict(config),
            base_checkpoint=records['train'][0]['base_checkpoint'], modules={config.layers[0]: module.state_dict()},
            extra=dict(stage='binding_feature_joint', flow_trained=False, step=step,
                       validation=validation, selection_score=score, settings=settings))
        temporary = output / f'{name}.tmp.pt'; torch.save(payload, temporary); temporary.replace(output / f'{name}.pt')
    progress = tqdm(range(steps), desc='SAP-Bind features', unit='step')
    for step in progress:
        distractors = random.Random(seed + step).randrange(9)
        selected = []
        attempt = 0
        while len(selected) < batch_size and attempt < batch_size * 8:
            index = step * batch_size + attempt
            record = records['train'][index % len(records['train'])]
            noise = random.Random(seed + 17 * index).randrange(len(record['queries']))
            episode = sample_episode(record, noise, seed=seed + index,
                support_tokens=int(settings.get('support_tokens', 512)),
                query_tokens=int(settings.get('query_tokens', 64)), distractor_writes=distractors,
                protected=int(settings.get('protected_anchors', 128)))
            if len(episode['positive']): selected.append(episode)
            attempt += 1
        if len(selected) != batch_size:
            raise ValueError('Insufficient independently sampled exact matches for requested batch')
        batch = collate(selected, device); module.train(); optimizer.zero_grad(set_to_none=True)
        loss, metrics = binding_objective(module, batch, config, settings['loss_weights'],
                                          training=True, online=True)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite SAP-Bind objective')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); validation = None; score = None
        if (step + 1) % val_every == 0 or step + 1 == steps:
            validation = _evaluate(module, records['val'], config, settings, device)
            score = _selection(validation)
            if score < best:
                best = score; save('best', step + 1, validation, score)
        save('last', step + 1, validation, score)
        row = dict(step=step + 1, loss=float(loss.detach()), gradient_norm=float(norm),
                   distractor_writes=distractors, best_selection_score=best if math.isfinite(best) else None,
                   elapsed_seconds=time.perf_counter() - started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                   **metrics)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if validation is not None:
            with (output / 'val.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step + 1, score=score, rows=validation), allow_nan=False) + '\n')
        progress.set_postfix(loss=f'{row["loss"]:.4g}', exact=f'{metrics["exact_top1"]:.3f}')
    payload = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    module.load_state_dict(payload['modules'][config.layers[0]])
    records['test'] = _load_records(settings['test_features'])
    if not records['test'] or {r['layout_id'] for r in records['test']} & (layouts['train'] | layouts['val']):
        raise ValueError('SAP-Bind test layouts overlap train/validation or are empty')
    attach_values(records['test'], module.value)
    test = _evaluate(module, records['test'], config, settings, device)
    (output / 'test_results.json').write_text(json.dumps(dict(stage='binding_feature_mechanism',
        selected_step=payload['extra']['step'], rows=test), indent=2, allow_nan=False), encoding='utf-8')
    return output / 'best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(run(json.loads(Path(args.settings).read_text(encoding='utf-8')), args.output))


if __name__ == '__main__':
    main()
