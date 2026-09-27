"""Independent SAP-Bind Flow pilot guarded by mechanism evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from worldttt.check_cache import require_gate
from worldttt.data import EpisodeDataset
from worldttt.sana import fixture_to_device, load_config, make_fixture, make_pipeline

from .config import BindingConfig
from .causal_eval import validate_mechanism
from .episode import BindingEpisodeModel
from .joint import _load_records, attach_values
from .runtime import BindingController


def _procedural_batch(path, checkpoint, device, dtype):
    fixture = torch.load(path, map_location='cpu', weights_only=True)
    if fixture.get('version') != 1 or fixture.get('source') != 'sap_binding_procedural_ground_truth':
        raise ValueError('SAP-Bind Flow pretraining requires its paired procedural fixtures')
    if fixture['base_checkpoint'] != checkpoint:
        raise ValueError('Procedural fixture uses another SANA checkpoint')
    batch = fixture_to_device(fixture, device, dtype)
    return {k: batch[k] for k in ('latent', 'camera', 'plucker', 'text', 'mask')}, fixture['scene_id']


def _validate(model, data, pipe, seed, limit):
    saved_state, saved_metrics = model.controller.state, model.controller.metrics
    totals = {'real': 0., 'generated': 0.}
    indices = random.Random(seed).sample(range(len(data)), min(limit, len(data)))
    try:
        with torch.no_grad():
            for index in indices:
                batch = make_fixture(data[index], pipe, 'cuda')
                for name in totals:
                    loss, _ = model(**batch, generated=name == 'generated', seed=seed + index,
                                    meta_grad=False)
                    totals[name] += float(loss)
    finally:
        model.controller.state, model.controller.metrics = saved_state, saved_metrics
    return {name + '_query_flow_mse': value / len(indices) for name, value in totals.items()}


def _require_mechanism_gate(settings, feature_adapter):
    path = settings.get('binding_causal_gate')
    if not path or not Path(path).is_file():
        raise ValueError('SAP-Bind Flow requires a completed causal gate report')
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if (report.get('kind') != 'sap_binding_causal_gate' or report.get('split') != 'val' or
            not report.get('gates', {}).get('passed')):
        raise ValueError('SAP-Bind causal binding gate did not pass; Flow training is blocked')
    try:
        same_adapter = Path(report['adapter']).resolve() == Path(feature_adapter).resolve()
    except (KeyError, TypeError):
        same_adapter = False
    if not same_adapter:
        raise ValueError('Causal gate report belongs to another feature adapter')
    structure_path = settings.get('binding_structure_gate')
    if not structure_path or not Path(structure_path).is_file():
        raise ValueError('SAP-Bind Flow requires the hybrid/single-path comparison report')
    structure = json.loads(Path(structure_path).read_text(encoding='utf-8'))
    if not structure.get('flow_allowed'):
        raise ValueError('Hybrid SAP-Bind did not beat the causal and single-path gates')
    if Path(structure.get('adapters', {}).get('hybrid', '')).resolve() != Path(feature_adapter).resolve():
        raise ValueError('Structure gate report belongs to another feature adapter')
    layers = tuple(settings['sap_binding']['layers'])
    if len(layers) > 1:
        expected = {str(layer) for layer in layers}
        payload = torch.load(feature_adapter, map_location='cpu', weights_only=True)
        sources = payload.get('extra', {}).get('layer_sources', {})
        if (set(sources) != expected or set(report.get('layer_sources', {})) != expected or
                set(structure.get('layer_sources', {})) != expected or
                tuple(report.get('layers', ())) != layers or
                tuple(structure.get('layers', ())) != layers or
                payload.get('extra', {}).get('stage') != 'binding_feature_joint_multilayer'):
            raise ValueError('Incomplete five-layer SAP-Bind mechanism evidence')
        for layer in expected:
            if sources[layer] != report['layer_sources'][layer] or sources[layer] != structure['layer_sources'][layer]:
                raise ValueError(f'Layer {layer} evidence differs from the merged adapter')
            source = sources[layer]
            original = Path(source['path']).resolve()
            if hashlib.sha256(original.read_bytes()).hexdigest() != source['sha256']:
                raise ValueError(f'Layer {layer} source checkpoint changed after merge')
            causal = json.loads(Path(source['causal_report']).read_text(encoding='utf-8'))
            comparison = json.loads(Path(source['structure_report']).read_text(encoding='utf-8'))
            if (Path(causal.get('adapter', '')).resolve() != original or
                    not causal.get('gates', {}).get('passed') or
                    Path(comparison.get('adapters', {}).get('hybrid', '')).resolve() != original or
                    not comparison.get('flow_allowed')):
                raise ValueError(f'Layer {layer} mechanism gate no longer passes')


def _binding_validation_records(settings, module, feature_settings, *, layer=None):
    paths = settings.get('binding_val_features')
    if isinstance(paths, dict):
        paths = paths.get(str(layer))
    if not paths:
        raise ValueError('SAP-Bind Flow requires procedural validation features')
    if {str(Path(path).resolve()) for path in paths} != {
            str(Path(path).resolve()) for path in feature_settings.get('val_features', [])}:
        raise ValueError('Flow procedural validation must use the feature adapter validation split')
    records = _load_records(paths)
    if layer is not None and any(record.get('layer') != layer for record in records):
        raise ValueError(f'Layer {layer} Flow procedural validation feature mismatch')
    train_layouts = {record['layout_id'] for record in
                     _load_records(feature_settings.get('train_features', []))}
    if train_layouts & {record['layout_id'] for record in records}:
        raise ValueError('Procedural validation layouts overlap training')
    attach_values(records, module.value)
    return records


def _validate_all_mechanisms(modules, records, config, settings, device):
    per_layer = {str(layer): validate_mechanism(module, records[layer], config,
                                                settings, device)
                 for layer, module in modules.items()}
    return dict(passed=all(result['passed'] for result in per_layer.values()),
                per_layer=per_layer)


def train(settings, output, adapter):
    if not torch.cuda.is_available():
        raise RuntimeError('SAP-Bind Flow training requires Linux CUDA')
    if not adapter:
        raise ValueError('Initialize SAP-Bind Flow from a feature checkpoint')
    require_gate(settings); _require_mechanism_gate(settings, adapter)
    options = settings['sap_binding_train']
    max_steps, accumulation, val_every = (int(options[key]) for key in
                                           ('max_steps', 'gradient_accumulation', 'val_every'))
    if min(max_steps, accumulation, val_every) < 1:
        raise ValueError('Invalid SAP-Bind Flow schedule')
    seed = int(settings.get('seed', 3407)); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    sana_config = load_config(settings['sana_config'])
    pipe = make_pipeline(sana_config, settings['base_checkpoint'], 'cuda', training=True)
    pipe.model.to('cuda')
    controller = BindingController(pipe.model, BindingConfig(**dict(settings['sap_binding'], mode='online')))
    controller.base_checkpoint = settings['base_checkpoint']
    extra = controller.load_checkpoint(adapter)
    if extra.get('flow_trained') is not False:
        raise ValueError('SAP-Bind Flow initialization must be a feature-only checkpoint')
    feature_sha256 = hashlib.sha256(Path(adapter).read_bytes()).hexdigest()
    episode = BindingEpisodeModel(pipe.model, controller, steps=settings.get('steps', 50),
                                  shift=sana_config.scheduler.inference_flow_shift)
    sources = extra.get('layer_sources')
    mechanism_records = {}
    for layer, module in controller.modules.items():
        feature_settings = (sources[str(layer)]['settings'] if sources is not None
                            else extra.get('settings', {}))
        mechanism_records[layer] = _binding_validation_records(
            settings, module, feature_settings, layer=layer)
    parameters = [p for p in pipe.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(options.get('outer_lr', 1e-4)), weight_decay=.01)
    train_data = EpisodeDataset(settings['data'], settings['manifest'], 'train', frames=13)
    val_data = EpisodeDataset(settings['data'], settings['manifest'], 'val', frames=13)
    loader = DataLoader(train_data, batch_size=None, shuffle=True, num_workers=0); iterator = iter(loader)
    procedural = [Path(p) for p in options.get('procedural_fixtures', [])]
    procedural_steps = int(options.get('procedural_steps', 100))
    if procedural_steps and not procedural:
        raise ValueError('Procedural phase requires encoded SAP-Bind fixtures')
    output = Path(output)
    if any((output / name).exists() for name in ('best.pt', 'last.pt')):
        raise FileExistsError('Use a fresh SAP-Bind Flow output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    def save(name, step, validation):
        temporary = output / f'{name}.tmp.pt'
        controller.save_checkpoint(temporary, extra=dict(step=step, settings=settings,
            validation=validation, checkpoint_kind='sap_binding_adapter', stage='flow',
            flow_trained=True, feature_adapter_sha256=feature_sha256,
            layer_sources=sources))
        temporary.replace(output / f'{name}.pt')
    optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    best = math.inf; started = time.perf_counter()
    progress = tqdm(range(max_steps), desc='SAP-Bind Flow', unit='step')
    for step in progress:
        total = 0.; generated_count = 0
        for micro in range(accumulation):
            if step < procedural_steps:
                path = procedural[(step * accumulation + micro) % len(procedural)]
                batch, episode_id = _procedural_batch(path, settings['base_checkpoint'], 'cuda', pipe.weight_dtype)
                generated = False; source = 'procedural_gt'
            else:
                try: sample = next(iterator)
                except StopIteration: iterator = iter(loader); sample = next(iterator)
                batch = make_fixture(sample, pipe, 'cuda'); episode_id = batch.pop('episode_id')
                generated = random.random() < float(options.get('generated_history_probability', .5))
                source = 'generated' if generated else 'real'
            loss, detail = episode(**batch, episode_id=episode_id, generated=generated,
                                   seed=seed + step * accumulation + micro)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite SAP-Bind Flow loss at {step}')
            (loss / accumulation).backward(); total += float(loss.detach()) / accumulation
            generated_count += int(generated)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); done = step + 1
        validation = None
        if done % val_every == 0 or done == max_steps:
            validation = _validate(episode, val_data, pipe, int(options.get('val_seed', 12345)),
                                   int(options.get('val_max_samples', 4)))
            validation['procedural_binding_gate'] = _validate_all_mechanisms(
                controller.modules, mechanism_records, controller.config,
                settings, torch.device('cuda'))
        save('last', done, validation)
        if (validation and validation['procedural_binding_gate']['passed'] and
                validation['generated_query_flow_mse'] < best):
            best = validation['generated_query_flow_mse']; save('best', done, validation)
        row = dict(step=done, loss=total, last_micro_source=source,
            generated_microbatches=generated_count, last_micro_detail=detail, gradient_norm=float(norm),
            validation=validation, best_generated_query_flow_mse=None if math.isinf(best) else best,
            elapsed_seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        progress.set_postfix(loss=f'{total:.4g}', best='-' if math.isinf(best) else f'{best:.4g}')
    if math.isinf(best):
        raise RuntimeError('No SAP-Bind Flow checkpoint passed the procedural binding gate; inspect last.pt')
    return output / 'best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--adapter', required=True)
    args = parser.parse_args(); print(train(json.loads(Path(args.settings).read_text(encoding='utf-8')),
                                           args.output, args.adapter))


if __name__ == '__main__':
    main()
