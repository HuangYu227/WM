"""Single-A800 SAP outer training; select checkpoints on independent queries."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from worldttt.check_cache import require_gate
from worldttt.data import EpisodeDataset
from worldttt.sana import fixture_to_device, load_config, make_fixture, make_pipeline

from .episode import SapEpisodeModel
from .pairs import match_historical_tokens
from .runtime import SapConfig, SapController


def load_supervision(path):
    with np.load(path, allow_pickle=False) as labels:
        instance, world = labels['instance'], labels['world']
        if instance.shape[0] != 13 or world.shape != (*instance.shape, 3):
            raise ValueError('SAP supervision needs 13 aligned latent frames')
        positive, valid = match_historical_tokens(instance[:4], world[:4],
                                                  instance[10:13], world[10:13])
        return {'positive': torch.from_numpy(positive), 'valid': torch.from_numpy(valid),
                'support_instance': torch.from_numpy(instance[:10].copy()),
                'query_instance': torch.from_numpy(instance[10:13].copy())}


def _procedural_batch(path, checkpoint, device, dtype):
    fixture = torch.load(path, map_location='cpu', weights_only=True)
    if fixture.get('version') != 1 or fixture.get('source') != 'procedural_ground_truth':
        raise ValueError('Procedural training requires a labeled encoded fixture')
    if fixture['base_checkpoint'] != checkpoint:
        raise ValueError('Procedural fixture uses a different SANA checkpoint')
    labels = load_supervision(Path(path).with_name('supervision.npz'))
    batch = fixture_to_device(fixture, device, dtype)
    return ({k: batch[k] for k in ('latent', 'camera', 'plucker', 'text', 'mask')},
            labels, fixture['scene_id'])


def _validate(model, data, pipe, seed, limit):
    saved_state, saved_metrics = model.controller.state, model.controller.metrics
    totals = {'real': 0., 'generated': 0.}
    indices = random.Random(seed).sample(range(len(data)), min(limit, len(data)))
    try:
        with torch.no_grad():
            for index in indices:
                batch = make_fixture(data[index], pipe, 'cuda')
                for name in totals:
                    loss, _ = model(**batch, generated=name == 'generated',
                                    seed=seed + index, meta_grad=False)
                    totals[name] += float(loss)
    finally:
        model.controller.state, model.controller.metrics = saved_state, saved_metrics
    return {name + '_query_flow_mse': value / len(indices) for name, value in totals.items()}


def _save(controller, output, name, step, settings, validation):
    temporary = output / f'{name}.tmp.pt'
    controller.save_checkpoint(temporary, extra={'step': step, 'settings': settings,
                                                  'validation': validation, 'checkpoint_kind': 'sap_adapter_only',
                                                  'stage': 'flow', 'flow_trained': True})
    temporary.replace(output / f'{name}.pt')


def train(settings, output, adapter=None):
    if not torch.cuda.is_available():
        raise RuntimeError('SAP outer training requires Linux CUDA')
    require_gate(settings)
    options = settings.get('sap_train', {})
    max_steps = int(options.get('max_steps', 300))
    accumulation = int(options.get('gradient_accumulation', 2))
    val_every = int(options.get('val_every', 50))
    if min(max_steps, accumulation, val_every) < 1:
        raise ValueError('Invalid SAP training intervals')
    seed = int(settings.get('seed', 3407))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    config = load_config(settings['sana_config'])
    pipe = make_pipeline(config, settings['base_checkpoint'], 'cuda', training=True)
    pipe.model.to('cuda')
    controller = SapController(pipe.model, SapConfig(**dict(settings['sap'], mode='online')))
    controller.base_checkpoint = settings['base_checkpoint']
    if adapter and options.get('probe_checkpoint'):
        raise ValueError('Choose either a full SAP adapter warm start or a mechanism probe checkpoint')
    if options.get('probe_checkpoint'):
        if controller.config.address_arch != 'linear' or controller.config.memory_arch != 'linear':
            raise ValueError('Legacy probes do not initialize multimodal SAP; use joint best.pt via --adapter')
        probe = torch.load(options['probe_checkpoint'], map_location='cpu', weights_only=True)
        if (probe['base_checkpoint'] != settings['base_checkpoint'] or
                probe['mode'] != controller.config.address_mode or
                probe['dim'] != controller.config.dim):
            raise ValueError('SAP probe/backbone/address configuration mismatch')
        for module in controller.modules.values():
            module.address.load_state_dict(probe['address'])
            if 'memory' in probe:
                module.memory.load_state_dict(probe['memory'])
    if adapter:
        controller.load_checkpoint(adapter)
    episode = SapEpisodeModel(pipe.model, controller, steps=settings.get('steps', 50),
                              shift=config.scheduler.inference_flow_shift,
                              lambda_delayed=float(options.get('lambda_delayed', .1)),
                              lambda_exact=float(options.get('lambda_exact', 0.)))
    parameters = [p for p in pipe.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(options.get('outer_lr', 1e-4)), weight_decay=.01)
    train_data = EpisodeDataset(settings['data'], settings['manifest'], 'train', frames=13)
    val_data = EpisodeDataset(settings['data'], settings['manifest'], 'val', frames=13)
    loader = DataLoader(train_data, batch_size=None, shuffle=True, num_workers=0)
    iterator = iter(loader)
    procedural = [Path(p) for p in options.get('procedural_fixtures', [])]
    pretrain = int(options.get('probe_pretrain_steps', 0))
    if pretrain and not procedural:
        raise ValueError('probe_pretrain_steps requires procedural_fixtures')
    output = Path(output)
    if any((output / name).exists() for name in ('last.pt', 'best.pt')):
        raise FileExistsError('SAP training needs a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    best = math.inf
    started = time.perf_counter()
    progress = tqdm(range(max_steps), desc='SAP-TTT outer train', unit='step')
    for step in progress:
        total = 0.
        generated_count = 0
        for micro in range(accumulation):
            if step < pretrain:
                selected = procedural[(step * accumulation + micro) % len(procedural)]
                batch, supervision, episode_id = _procedural_batch(
                    selected, settings['base_checkpoint'], 'cuda', pipe.weight_dtype)
                generated = False
                source = 'procedural_gt'
            else:
                try:
                    sample = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    sample = next(iterator)
                batch = make_fixture(sample, pipe, 'cuda')
                episode_id = batch.pop('episode_id')
                supervision = None
                generated = (step >= int(options.get('real_prefix_steps', 100)) and random.random() < .5)
                source = 'generated' if generated else 'real'
            loss, detail = episode(**batch, episode_id=episode_id, generated=generated,
                                   seed=seed + step * accumulation + micro, supervision=supervision)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite SAP outer loss at step {step}')
            (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
            generated_count += int(generated)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        done = step + 1
        validation = None
        if done % val_every == 0 or done == max_steps:
            validation = _validate(episode, val_data, pipe,
                                   int(options.get('val_seed', 12345)),
                                   int(options.get('val_max_samples', 4)))
        _save(controller, output, 'last', done, settings, validation)
        if validation and validation['generated_query_flow_mse'] < best:
            best = validation['generated_query_flow_mse']
            _save(controller, output, 'best', done, settings, validation)
        row = {'step': done, 'loss': total, 'last_micro_source': source,
               'generated_microbatches': generated_count, 'last_micro_detail': detail,
               'gradient_norm': float(grad_norm), 'validation': validation,
               'best_generated_query_flow_mse': None if math.isinf(best) else best,
               'elapsed_seconds': time.perf_counter() - started,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        progress.set_postfix(loss=f'{total:.4g}', best='-' if math.isinf(best) else f'{best:.4g}',
                             mem=f'{row["peak_allocated_bytes"] / 2**30:.1f}GiB')
    return output / 'best.pt'


def main():
    parser = argparse.ArgumentParser(description='Train the SAP adapter on four-chunk episodes')
    parser.add_argument('--settings', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapter')
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    print(train(settings, args.output, args.adapter))


if __name__ == '__main__':
    main()
