"""Run the binding-specific causal gates before any Flow training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from worldttt.sap_ttt.memory import SapMemoryState

from .config import BindingConfig
from .joint import _load_records, attach_values, collate, sample_episode
from .memory import BindingBankState, BindingState, binding_scores, explicit_read
from .model import BindingBlock


CONTROLS = ('frozen', 'bank_frozen_fast', 'bank_read_only', 'online', 'shuffle_key', 'shuffle_value',
            'mean_value', 'last_only')


def evaluate_batch(module, batch, config, control, *, modality='normal'):
    if control not in CONTROLS or modality not in {'normal', 'text_swap', 'ray_shuffle', 'vision_only'}:
        raise ValueError('Unknown SAP-Bind causal control')
    b = batch['positive'].shape[0]
    fast = SapMemoryState.new(module.fast, 'causal', b, training=False)
    keys, values, rays, ages = [], [], [], []
    online = control != 'frozen'
    address_mode = 'vision' if modality == 'vision_only' else 'selective_geometry'
    for age, part in enumerate(batch['writes']):
        write_rays = part['rays'].roll(1, 1) if modality == 'ray_shuffle' else part['rays']
        key = module.address.write(part['visual'], part['text'], part['text_mask'], write_rays,
                                   mode=address_mode)
        value = part['binding_value'].float()
        if control == 'shuffle_key':
            key = key.roll(1, 1)
            write_rays = write_rays.roll(1, 1)
        if control == 'shuffle_value': value = value.roll(1, 1)
        if (online and control != 'bank_frozen_fast' and config.architecture != 'bank_only' and
                control != 'mean_value' and
                (control != 'last_only' or age == len(batch['writes']) - 1)):
            proposed, _ = module.fast.update(fast.weight, key, value)
            fast.weight = proposed.detach().requires_grad_(True)
        keys.append(key); values.append(value); rays.append(write_rays.float())
        ages.append(torch.full(key.shape[:2], age, device=key.device, dtype=torch.float32))
    candidate_key, candidate_value = torch.cat(keys, 1), torch.cat(values, 1)
    candidate_ray, age = torch.cat(rays, 1), torch.cat(ages, 1)
    valid = batch['candidates']['valid']
    if not online:
        valid = torch.zeros_like(valid)
    bank = BindingBankState(candidate_key, candidate_value, candidate_ray,
        torch.ones_like(valid, dtype=torch.float32), age, valid,
        valid.sum(1), torch.zeros_like(valid))
    state = BindingState('causal', bank, fast, len(keys) - 1, len(keys) if online else 0)
    query_part = batch['query']; query_rays = query_part['rays']
    if modality == 'ray_shuffle': query_rays = query_rays.roll(1, 1)
    text = query_part['text'].roll(1, 0) if modality == 'text_swap' and b > 1 else query_part['text']
    query = module.address.query(query_part['visual'], text, query_part['text_mask'], query_rays,
                                 query_part['sigma'], mode=address_mode)
    if control == 'bank_read_only':
        value, _ = explicit_read(bank, query, query_rays, module.ray_query, module.ray_key,
                                 heads=config.heads, topk=config.topk)
    else:
        value, _, _ = module.read(state, query, query_rays, query_part['sigma'],
                                  mean_value=control == 'mean_value', last_only=control == 'last_only')
    valid_query = query_part['valid']; target = batch['target']
    error = ((value - target).square().mean(-1) * valid_query).sum() / valid_query.sum()
    if bank.valid.any(1).all():
        score = binding_scores(bank, query, query_rays, module.ray_query, module.ray_key,
                               heads=config.heads).mean(1)
        exact = (((score.argmax(-1) == batch['positive']) * valid_query).sum() /
                 valid_query.sum())
    else:
        exact = error.new_zeros(())
    return dict(query_mse=float(error), exact_top1=float(exact),
                matched_queries=int(valid_query.sum()),
                candidates=int(valid.sum(1).float().mean()),
                coverage=sum(s['coverage'] for s in batch['samples']) / b)


def _sample(record, noise, settings, budget, distractors, forced=None, *, include_uncovered=False):
    return sample_episode(record, noise, seed=int(settings.get('eval_seed', 12345)) + noise,
        support_tokens=budget, query_tokens=int(settings.get('query_tokens', 64)),
        distractor_writes=distractors, protected=min(int(settings.get('protected_anchors', 128)), budget),
        forced_query_indices=forced, include_uncovered=include_uncovered)


def summarize_gates(rows):
    def average(group, key):
        return sum(row[key] for row in group) / len(group) if group else 0.

    interference = [row for row in rows if row['kind'] == 'interference']
    ratios = {}
    complete = True
    for distractors in (2, 8):
        group = {control: [row for row in interference if row['distractor_writes'] == distractors
                           and row['support_tokens'] == 256 and row['control'] == control]
                 for control in ('online', 'shuffle_key', 'shuffle_value', 'mean_value')}
        complete &= all(group.values()) and len({len(value) for value in group.values()}) == 1
        ratios[distractors] = {control: average(group['online'], 'query_mse') /
                               max(average(group[control], 'query_mse'), 1e-12)
                               for control in ('shuffle_key', 'shuffle_value', 'mean_value')}
    exact_rows = [row for row in interference if row['distractor_writes'] == 2 and
                  row['support_tokens'] == 256 and row['control'] == 'online']
    hits = sum(row['exact_top1'] * row['matched_queries'] for row in exact_rows)
    trials = sum(row['matched_queries'] for row in exact_rows)
    proportion = hits / trials if trials else 0.
    # One-sided Wilson bound keeps a tiny, lucky validation set from passing.
    z = 1.645
    lower = ((proportion + z * z / (2 * trials) - z *
              ((proportion * (1 - proportion) + z * z / (4 * trials)) / trials) ** .5) /
             (1 + z * z / trials)) if trials else 0.
    chance = (sum(1 / row['candidates'] for row in exact_rows) / len(exact_rows)
              if exact_rows and all(row.get('candidates', 0) > 0 for row in exact_rows) else 1.)
    budget_rows = [row for row in rows if row['kind'] == 'budget' and row['control'] == 'online']
    budgets = (64, 128, 256, 512)
    budget_groups = {budget: [row for row in budget_rows if row['support_tokens'] == budget
                             and row.get('common_queries', 0) > 0] for budget in budgets}
    complete &= (all(budget_groups.values()) and len({len(value) for value in budget_groups.values()}) == 1
                 and trials > 0 and chance < 1.)
    budget_mse = {budget: average(group, 'query_mse') for budget, group in budget_groups.items()}
    budget_recall = {budget: average(group, 'coverage') for budget, group in budget_groups.items()}
    passed = (complete and ratios[2]['shuffle_key'] <= .8 and ratios[2]['shuffle_value'] <= .8 and
              ratios[2]['mean_value'] <= .85 and ratios[8]['shuffle_key'] <= .85 and
              ratios[8]['shuffle_value'] <= .85 and ratios[8]['mean_value'] <= .85 and
              average(exact_rows, 'exact_top1') >= .15 and lower > chance and
              budget_recall[512] > budget_recall[64] and budget_mse[512] <= 1.05 * budget_mse[64])
    return dict(normal_ratios=ratios[2], eight_write_ratios=ratios[8],
                exact_top1_256=average(exact_rows, 'exact_top1'),
                exact_top1_lower_95=lower, random_top1=chance,
                budget_common_query_mse=budget_mse, budget_recall=budget_recall,
                evidence_complete=bool(complete), passed=bool(passed))


def validate_mechanism(module, records, config, settings, device):
    """Recheck the current Flow adapter on sealed-out procedural validation layouts."""
    rows = []
    budget = int(settings.get('causal_support_tokens', 256))
    if budget != 256:
        raise ValueError('SAP-Bind causal protocol requires 256 standard support tokens')
    with torch.no_grad():
        for record in records:
            for noise in range(len(record['queries'])):
                for distractors in (2, 8):
                    episode = _sample(record, noise, settings, 256, distractors)
                    if not episode['positive'].numel():
                        continue
                    batch = collate([episode], device)
                    for control in ('online', 'shuffle_key', 'shuffle_value', 'mean_value'):
                        rows.append(dict(kind='interference', distractor_writes=distractors,
                            support_tokens=256, control=control,
                            **evaluate_batch(module, batch, config, control)))
                sampled = {size: _sample(record, noise, settings, size, 2)
                           for size in (64, 128, 256, 512)}
                common = set.intersection(*(set(part['query_indices']) for part in sampled.values()))
                for size in sampled:
                    episode = _sample(record, noise, settings, size, 2, common)
                    if not episode['positive'].numel():
                        continue
                    rows.append(dict(kind='budget', support_tokens=size, control='online',
                        common_queries=len(common), **evaluate_batch(module, collate([episode], device),
                                                                      config, 'online')))
    return summarize_gates(rows)


def run(settings, adapter, output, *, split='val'):
    config = BindingConfig(**settings['sap_binding'])
    if len(config.layers) != 1:
        raise ValueError('Per-layer causal evaluation requires one selected layer')
    if split not in {'val', 'test'}:
        raise ValueError('SAP-Bind causal split must be val or test')
    records = _load_records(settings[split + '_features'])
    if any(record.get('layer') != config.layers[0] for record in records):
        raise ValueError('SAP-Bind causal evaluation feature layer mismatch')
    payload = torch.load(adapter, map_location='cpu', weights_only=True)
    if (payload.get('kind') != 'sap_binding_adapter' or
            tuple(payload.get('config', {}).get('layers', ())) != config.layers or
            payload.get('config', {}).get('architecture') != config.architecture):
        raise ValueError('Causal evaluation adapter/config mismatch')
    sample = records[0]['supports'][0]
    module = BindingBlock(sample['visual'].shape[-1], sample['latent'].shape[1], config)
    module.load_state_dict(payload['modules'][config.layers[0]], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); module.to(device).eval()
    attach_values(records, module.value)
    rows = []
    default_budget = int(settings.get('causal_support_tokens', 256))
    with torch.no_grad():
        for record in records:
            for noise in range(len(record['queries'])):
                for distractors in (0, 1, 2, 4, 8):
                    episode = _sample(record, noise, settings, default_budget, distractors)
                    if not len(episode['positive']): continue
                    batch = collate([episode], device)
                    for control in CONTROLS if distractors == 2 else ('online', 'bank_read_only', 'shuffle_key', 'shuffle_value', 'mean_value'):
                        rows.append(dict(kind='interference', scene_id=record['scene_id'],
                            noise_sigma=episode['noise_sigma'], distractor_writes=distractors,
                            support_tokens=default_budget, control=control, modality='normal',
                            **evaluate_batch(module, batch, config, control)))
                budget_samples = {budget: _sample(record, noise, settings, budget, 2)
                                  for budget in (64, 128, 256, 512)}
                common = set.intersection(*(set(x['query_indices']) for x in budget_samples.values()))
                for budget in (64, 128, 256, 512):
                    episode = _sample(record, noise, settings, budget, 2, common)
                    if not len(episode['positive']): continue
                    rows.append(dict(kind='budget', scene_id=record['scene_id'],
                        noise_sigma=episode['noise_sigma'], distractor_writes=2,
                        support_tokens=budget, control='online', modality='normal', common_queries=len(common),
                        **evaluate_batch(module, collate([episode], device), config, 'online')))
        # Paired layouts make text exchange a real same-prompt intervention.
        by_layout = {}
        for record in records: by_layout.setdefault(record['layout_id'], []).append(record)
        for layout, pair in by_layout.items():
            if len(pair) != 2: continue
            for noise in range(len(pair[0]['queries'])):
                episodes = [_sample(record, noise, settings, default_budget, 2) for record in pair]
                if any(not len(x['positive']) for x in episodes): continue
                batch = collate(episodes, device)
                for modality in ('normal', 'text_swap', 'ray_shuffle', 'vision_only'):
                    rows.append(dict(kind='modality', layout_id=layout,
                        noise_sigma=episodes[0]['noise_sigma'], distractor_writes=2,
                        support_tokens=default_budget, control='online', modality=modality,
                        **evaluate_batch(module, batch, config, 'online', modality=modality)))
        for row in rows:
            row.setdefault('query_source', 'native')
        for record in records:
            if 'queries_no_history' not in record:
                continue
            alternative = dict(record, queries=record['queries_no_history'])
            for noise in range(len(alternative['queries'])):
                episode = _sample(alternative, noise, settings, default_budget, 2)
                if not len(episode['positive']): continue
                batch = collate([episode], device)
                for control in CONTROLS:
                    rows.append(dict(kind='cross_source', scene_id=record['scene_id'],
                        noise_sigma=episode['noise_sigma'], distractor_writes=2,
                        support_tokens=default_budget, control=control, modality='normal',
                        query_source='no_history', **evaluate_batch(module, batch, config, control)))
    gates = summarize_gates(rows)
    state_bytes = (config.capacity * (config.address_dim + config.value_dim + config.ray_dim + 2) * 4 +
                   config.capacity * 2 + module.fast.initial_weight.numel() * 4)
    report = dict(version=1, kind='sap_binding_causal_gate', architecture=config.architecture,
                  adapter=str(Path(adapter).resolve()), split=split, gates=gates,
                  state_bytes_per_branch=state_bytes, rows=rows)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True); parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    args = parser.parse_args(); print(run(json.loads(Path(args.settings).read_text(encoding='utf-8')),
                                         args.adapter, args.output, split=args.split))


if __name__ == '__main__':
    main()
