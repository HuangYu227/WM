"""Train only the fusion reader of an existing single-layer hybrid adapter."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import BindingConfig
from .joint import _evaluate, _load_records, attach_values, binding_objective, collate, sample_episode
from .model import BindingBlock


def _mean_query_mse(rows):
    valid = [row['query_mse'] for row in rows if row.get('valid')]
    if not valid or any(not math.isfinite(value) for value in valid):
        raise ValueError('Fusion calibration has no finite validation Query MSE')
    return sum(valid) / len(valid)


def run(settings, adapter, output, *, steps=100, batch_size=16, val_every=25,
        support_tokens=256, lr=1e-4):
    if min(steps, batch_size, val_every, support_tokens) < 1 or lr <= 0:
        raise ValueError('Invalid fusion calibration schedule')
    settings = dict(settings, support_tokens=support_tokens)
    config = BindingConfig(**settings['sap_binding'])
    if len(config.layers) != 1 or config.architecture != 'hybrid':
        raise ValueError('Fusion calibration requires a single-layer hybrid config')
    layer = config.layers[0]
    payload = torch.load(adapter, map_location='cpu', weights_only=True)
    if (payload.get('kind') != 'sap_binding_adapter' or
            asdict(BindingConfig(**payload.get('config', {}))) != asdict(config) or
            payload.get('extra', {}).get('flow_trained') or
            set(payload.get('modules', {})) != {layer}):
        raise ValueError('Fusion calibration adapter/config mismatch')
    records = {split: _load_records(settings[split + '_features'])
               for split in ('train', 'val')}
    layouts = {split: {record['layout_id'] for record in group}
               for split, group in records.items()}
    if (any(not group for group in records.values()) or
            any(record.get('layer') != layer or
                record.get('base_checkpoint') != payload['base_checkpoint']
                for group in records.values() for record in group) or
            layouts['train'] & layouts['val']):
        raise ValueError('Fusion calibration requires matching, layout-disjoint features')

    seed = int(settings.get('seed', 3407))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    sample = records['train'][0]['supports'][0]
    module = BindingBlock(sample['visual'].shape[-1], sample['latent'].shape[1], config)
    module.load_state_dict(payload['modules'][layer], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module.to(device)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for parameter in module.fusion.parameters():
        parameter.requires_grad_(True)
    attach_values([record for group in records.values() for record in group], module.value)

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh fusion calibration output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved.json').write_text(json.dumps(dict(settings=settings,
        adapter=str(Path(adapter).resolve()), steps=steps, batch_size=batch_size,
        val_every=val_every, lr=lr), indent=2), encoding='utf-8')
    weights = dict.fromkeys(('addr', 'value', 'fast', 'shuffle', 'mean', 'write'), 0.)
    weights['fused'] = 1.
    optimizer = torch.optim.AdamW(module.fusion.parameters(), lr=lr)
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def save(name, step, validation, score):
        result = dict(version=1, kind='sap_binding_adapter', config=asdict(config),
            base_checkpoint=payload['base_checkpoint'], modules={layer: module.state_dict()},
            extra=dict(stage='binding_fusion_calibration', flow_trained=False,
                       step=step, validation=validation, selection_score=score,
                       source_adapter=str(Path(adapter).resolve()), settings=settings))
        temporary = output / f'{name}.tmp.pt'
        torch.save(result, temporary)
        temporary.replace(output / f'{name}.pt')

    baseline = _evaluate(module, records['val'], config, settings, device)
    best = _mean_query_mse(baseline)
    save('best', 0, baseline, best)
    (output / 'baseline.json').write_text(json.dumps(dict(query_mse=best, rows=baseline),
                                                indent=2, allow_nan=False), encoding='utf-8')
    progress = tqdm(range(steps), desc='SAP-Bind fusion', unit='step')
    for step in progress:
        distractors = random.Random(seed + step).randrange(9)
        selected = []
        for attempt in range(batch_size * 8):
            if len(selected) == batch_size:
                break
            index = step * batch_size + attempt
            record = records['train'][index % len(records['train'])]
            noise = random.Random(seed + 17 * index).randrange(len(record['queries']))
            episode = sample_episode(record, noise, seed=seed + index,
                support_tokens=support_tokens, query_tokens=int(settings.get('query_tokens', 64)),
                distractor_writes=distractors,
                protected=int(settings.get('protected_anchors', 128)))
            if len(episode['positive']):
                selected.append(episode)
        if len(selected) != batch_size:
            raise ValueError('Insufficient matched training queries for fusion calibration')
        batch = collate(selected, device)
        module.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = binding_objective(module, batch, config, weights,
                                          training=False, online=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fusion calibration loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.fusion.parameters(), 1.,
                                               error_if_nonfinite=True)
        optimizer.step()
        row = dict(step=step + 1, loss=float(loss.detach()),
                   gradient_norm=float(norm), distractor_writes=distractors,
                   elapsed_seconds=time.perf_counter() - started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated()
                   if device.type == 'cuda' else None, **metrics)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if (step + 1) % val_every == 0 or step + 1 == steps:
            validation = _evaluate(module, records['val'], config, settings, device)
            score = _mean_query_mse(validation)
            if score < best:
                best = score
                save('best', step + 1, validation, score)
            save('last', step + 1, validation, score)
            with (output / 'val.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step + 1, query_mse=score,
                    best_query_mse=best), allow_nan=False) + '\n')
        progress.set_postfix(loss=f'{row["loss"]:.4g}', best=f'{best:.4g}')

    records['test'] = _load_records(settings['test_features'])
    if (not records['test'] or
            any(record.get('layer') != layer or
                record.get('base_checkpoint') != payload['base_checkpoint']
                for record in records['test']) or
            {record['layout_id'] for record in records['test']} &
            (layouts['train'] | layouts['val'])):
        raise ValueError('Fusion calibration test features overlap training/validation or mismatch')
    attach_values(records['test'], module.value)
    best_payload = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    module.load_state_dict(best_payload['modules'][layer])
    test = _evaluate(module, records['test'], config, settings, device)
    (output / 'test_results.json').write_text(json.dumps(dict(
        stage='binding_fusion_calibration', selected_step=best_payload['extra']['step'],
        rows=test), indent=2, allow_nan=False), encoding='utf-8')
    return output / 'best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-every', type=int, default=25)
    parser.add_argument('--support-tokens', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    print(run(settings, args.adapter, args.output, steps=args.steps,
              batch_size=args.batch_size, val_every=args.val_every,
              support_tokens=args.support_tokens, lr=args.lr))


if __name__ == '__main__':
    main()
