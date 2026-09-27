"""Evaluate trained SAP adapters without changing their slow weights."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from .feature_probe import _choose, select_query_source
from .joint import collate_episodes, sample_episode
from .memory import SapMemoryState
from .runtime import SapBlockMemory, SapConfig, validate_protocol


CONTROLS = ('frozen', 'online', 'shuffle_value', 'shuffle_key', 'last_only', 'mean_value')


def _load_adapter(path, vision_dim, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('version') != 1 or len(payload.get('modules', {})) != 1:
        raise ValueError('Expected one-layer SAP v1 adapter')
    config = SapConfig(**payload['config'])
    module = SapBlockMemory(vision_dim, config).to(device)
    validate_protocol(payload['config'], config)
    state = next(iter(payload['modules'].values()))
    module.load_state_dict(state, strict=True)
    module.eval()
    return module, config, payload


def _read_metrics(module, batch, config, control='online', seed=0):
    if control not in CONTROLS:
        raise ValueError(f'Unknown control: {control}')
    address, memory = module.address, module.memory
    state = SapMemoryState.new(memory, 'mechanism-eval', 1, training=False)
    keys, values = [], []
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for index, part in enumerate(batch['supports']):
        key = address.write(part['visual'], part['text'], part['text_mask'], part['rays'], mode=config.address_mode)
        value = address.value_for(part['post']).detach()
        keys.append(key); values.append(value)
        if control in {'frozen', 'mean_value'}:
            continue
        if control == 'last_only' and index:
            state = SapMemoryState.new(memory, 'mechanism-eval', 1, training=False)
        write_key, write_value = key, value
        if control == 'shuffle_value':
            write_value = value[:, torch.randperm(value.shape[1], generator=generator, device='cpu').to(value.device)]
        elif control == 'shuffle_key':
            write_key = key[:, torch.randperm(key.shape[1], generator=generator, device='cpu').to(key.device)]
        state.weight, _ = memory.update(state.weight, write_key, write_value, create_graph=False)
        state.weight = state.weight.detach().requires_grad_(True)
    part = batch['query']
    query = address.query(part['visual'], part['text'], part['text_mask'], part['rays'], part['sigma'],
                          mode=config.address_mode)
    if control == 'mean_value':
        mean = torch.cat(values, dim=1).mean(1, keepdim=True)
        readout = mean.expand(-1, query.shape[1], -1)
    else:
        readout = memory.read(state, query)
    valid = part['valid']
    target = values[0].gather(1, batch['positive'][..., None].expand(-1, -1, config.dim))
    query_mse = ((readout - target).square().mean(-1) * valid).sum() / valid.sum()
    bank = torch.cat(keys, dim=1)
    logits = torch.bmm(query, bank.transpose(1, 2)) / .07
    exact = (((logits.argmax(-1) == batch['positive']) & valid).sum() / valid.sum()).float()
    if control == 'mean_value':
        old_read = mean.expand_as(values[0]); new_read = mean.expand_as(values[-1])
    else:
        old_read = memory.read(state, keys[0]); new_read = memory.read(state, keys[-1])
    return dict(query_mse=float(query_mse), old_a_mse=float(F.mse_loss(old_read, values[0])),
                new_write_mse=float(F.mse_loss(new_read, values[-1])), exact_top1=float(exact),
                matched_queries=int(valid.sum()), candidates=int(bank.shape[1]),
                coverage=sum(s['coverage'] for s in batch['samples']) / len(batch['samples']))


def _records(settings):
    records = [torch.load(path, map_location='cpu', weights_only=True) for path in settings['test_features']]
    if not records or len({r['scene_id'] for r in records}) != len(records):
        raise ValueError('Need nonempty unique test feature records')
    return records


def _sample(record, source, noise, settings, seed_offset=0):
    return sample_episode(select_query_source(record, source), noise,
        seed=int(settings.get('val_seed', 12345)) + seed_offset + noise,
        support_budget=int(settings.get('support_tokens', 256)),
        query_budget=int(settings.get('query_tokens', 64)))


def _row(module, config, sample, device, **labels):
    controls = labels.pop('controls')
    if not len(sample['positive']):
        return [dict(**labels, control=control, valid=False,
                     reason='no_correspondence_survived_independent_writes',
                     coverage=float(sample['coverage'])) for control in controls]
    batch = collate_episodes([sample], device)
    rows = []
    for control in controls:
        with torch.no_grad():
            metric = _read_metrics(module, batch, config, control,
                                   seed=int(labels.get('seed', 0)))
        rows.append(dict(**labels, control=control, valid=True, **metric))
    return rows


def cross_source(settings, adapters, records, device):
    if set(adapters) != {'native', 'no_history'}:
        raise ValueError('cross_source requires native=PATH and no_history=PATH adapters')
    rows = []
    for adapter_name, path in adapters.items():
        module, config, payload = _load_adapter(path, records[0]['supports'][0]['visual'].shape[-1], device)
        for source in ('native', 'no_history'):
            for record in records:
                for noise in range(len(record['queries'])):
                    sample = _sample(record, source, noise, settings)
                    rows += _row(module, config, sample, device, controls=('frozen', 'online'),
                        adapter=adapter_name, query_source=source, scene_id=record['scene_id'],
                        noise_sigma=float(sample['noise_sigma']), seed=noise)
        if payload['base_checkpoint'] != records[0]['base_checkpoint']:
            raise ValueError('Adapter/feature backbone mismatch')
    return rows


def causal_controls(settings, adapter, records, device):
    module, config, payload = _load_adapter(adapter, records[0]['supports'][0]['visual'].shape[-1], device)
    if payload['base_checkpoint'] != records[0]['base_checkpoint']:
        raise ValueError('Adapter/feature backbone mismatch')
    rows = []
    for record in records:
        for noise in range(len(record['queries'])):
            sample = _sample(record, 'native', noise, settings)
            rows += _row(module, config, sample, device, controls=CONTROLS,
                scene_id=record['scene_id'], noise_sigma=float(sample['noise_sigma']), seed=noise)
    return rows


def interference_curve(settings, adapter, records, device):
    module, config, payload = _load_adapter(adapter, records[0]['supports'][0]['visual'].shape[-1], device)
    if payload['base_checkpoint'] != records[0]['base_checkpoint']:
        raise ValueError('Adapter/feature backbone mismatch')
    lengths = tuple(int(x) for x in settings.get('interference_lengths', (0, 1, 2, 4, 8)))
    if not lengths or min(lengths) < 0 or len(set(lengths)) != len(lengths):
        raise ValueError('interference_lengths must be unique nonnegative integers')
    rows = []
    for record_index, record in enumerate(records):
        for noise in range(len(record['queries'])):
            target = _sample(record, 'native', noise, settings)
            distractors = list(target['supports'][1:])
            offset = 1
            while len(distractors) < max(lengths):
                donor = records[(record_index + offset) % len(records)]
                donor_sample = _sample(donor, 'native', noise % len(donor['queries']), settings,
                                       seed_offset=1009 * offset)
                distractors.extend(donor_sample['supports'])
                offset += 1
            for count in lengths:
                sample = dict(target, supports=[target['supports'][0], *distractors[:count]])
                rows += _row(module, config, sample, device, controls=('frozen', 'online'),
                    scene_id=record['scene_id'], noise_sigma=float(target['noise_sigma']),
                    distractor_writes=count, seed=noise)
    return rows


def _nested_sample(record, source, noise, budgets, query_budget, seed):
    record = select_query_source(record, source)
    rng = random.Random(seed)
    orders = []
    for part in record['supports']:
        order = list(range(part['visual'].shape[1])); rng.shuffle(order); orders.append(order)
    labels = record['supervision']; positive = labels['positive'].flatten()
    valid = labels['valid'].flatten().bool() & (positive >= 0)
    smallest = set(orders[0][:budgets[0]])
    common = [i for i in torch.where(valid)[0].tolist() if int(positive[i]) in smallest]
    chosen = sorted(rng.sample(common, min(query_budget, len(common))))
    if not chosen:
        raise ValueError('No common query survived the smallest support budget')
    raw = record['queries'][noise]
    query = {name: value[:, chosen] if name in {'visual', 'post', 'rays', 'sigma'} and value is not None else value
             for name, value in raw.items()}
    samples = []
    total_valid = max(1, int(valid.sum()))
    for budget in budgets:
        ids = [sorted(order[:min(budget, len(order))]) for order in orders]
        lookup = {position: local for local, position in enumerate(ids[0])}
        available = sum(int(positive[i]) in lookup for i in torch.where(valid)[0].tolist())
        samples.append(dict(supports=[_choose(record, i, idx) for i, idx in enumerate(ids)], query=query,
            positive=torch.tensor([lookup[int(positive[i])] for i in chosen]), coverage=available / total_valid,
            scene_id=record['scene_id'], noise_sigma=record['noise_sigmas'][noise]))
    return samples


def budget_curve(settings, adapter, records, device):
    module, config, payload = _load_adapter(adapter, records[0]['supports'][0]['visual'].shape[-1], device)
    if payload['base_checkpoint'] != records[0]['base_checkpoint']:
        raise ValueError('Adapter/feature backbone mismatch')
    budgets = tuple(sorted(int(x) for x in settings.get('support_budgets', (64, 128, 256, 512))))
    if not budgets or min(budgets) < 1 or len(set(budgets)) != len(budgets):
        raise ValueError('support_budgets must be unique positive integers')
    rows = []
    for record in records:
        for noise in range(len(record['queries'])):
            try:
                samples = _nested_sample(record, 'native', noise, budgets,
                    int(settings.get('query_tokens', 64)), int(settings.get('val_seed', 12345)) + noise)
            except ValueError as exc:
                for budget in budgets:
                    for control in ('frozen', 'online'):
                        rows.append(dict(scene_id=record['scene_id'],
                            noise_sigma=float(record['noise_sigmas'][noise]), support_tokens=budget,
                            control=control, valid=False, reason=str(exc)))
                continue
            for budget, sample in zip(budgets, samples):
                rows += _row(module, config, sample, device, controls=('frozen', 'online'),
                    scene_id=record['scene_id'], noise_sigma=float(sample['noise_sigma']),
                    support_tokens=budget, seed=noise)
    return rows


def _parse_adapters(values):
    adapters = {}
    for value in values:
        if '=' not in value:
            raise ValueError('Adapters must use NAME=PATH')
        name, path = value.split('=', 1)
        if not name or name in adapters:
            raise ValueError('Adapter names must be nonempty and unique')
        adapters[name] = Path(path)
    return adapters


def run(settings, experiment, adapters, output):
    records = _records(settings)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if experiment == 'cross_source':
        rows = cross_source(settings, adapters, records, device)
    else:
        if set(adapters) != {'full'}:
            raise ValueError(f'{experiment} requires full=PATH adapter')
        functions = {'causal_controls': causal_controls, 'interference': interference_curve,
                     'budget': budget_curve}
        rows = functions[experiment](settings, adapters['full'], records, device)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(version=1, stage='feature_mechanism_only', experiment=experiment,
        adapters={k: str(v) for k, v in adapters.items()}, settings=settings, rows=rows),
        indent=2, allow_nan=False), encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--experiment', required=True,
                        choices=('cross_source', 'causal_controls', 'interference', 'budget'))
    parser.add_argument('--adapter', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    print(run(settings, args.experiment, _parse_adapters(args.adapter), args.output))


if __name__ == '__main__':
    main()
