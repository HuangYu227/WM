"""Label-free, same-cache GRAIL mechanism experiments."""

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

import torch

from .research_metrics import paired_bootstrap


def episode_record(result, *, scene_id, key, history, seed):
    future = {name: float(value) for name, value in result['variant_future'].items()}
    if 'ridge' not in future or not all(math.isfinite(value) for value in future.values()):
        raise ValueError('future flow values must be finite and include ridge')
    visits = defaultdict(set)
    observations = defaultdict(int)
    for chunk in result['memory_trace']:
        if not chunk['committed'] or chunk['accepted'] != sum(slot['observations'] for slot in chunk['slots']):
            raise ValueError('memory trace contains an incomplete clean commit')
        for slot in chunk['slots']:
            identity = (slot['batch'], slot['slot'], slot['generation'])
            visits[identity].add(chunk['chunk_id'])
            observations[identity] += slot['observations']
    slot_summary = dict(slot_generations=len(visits),
                        reused_across_chunks=sum(len(chunks) > 1 for chunks in visits.values()),
                        accepted_observations=sum(observations.values()),
                        max_observations_per_slot=max(observations.values(), default=0))
    return dict(scene_id=str(scene_id), key=str(key), history=history, seed=int(seed),
                ground_truth_instances=False, intervention_scope='heldout_query_only',
                fixed_history_cache=True, support_chunks=result['support_chunks'],
                query_input_fingerprint=result.get('query_input_fingerprint'),
                future_flow_mse=future,
                delta_vs_ridge={name: value - future['ridge'] for name, value in future.items() if name != 'ridge'},
                gate_mean=result['variant_gate_mean'], slot_coverage=result['variant_slot_coverage'],
                hook_counts=result['variant_hook_counts'], slot_summary=slot_summary,
                memory_trace=result['memory_trace'])


def paired_scene_summary(records, *, samples=10000, seed=3407):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in records:
        for variant, loss in row['future_flow_mse'].items():
            grouped[row['history']][variant][row['scene_id']].append(float(loss))
    summary = {}
    for history, methods in grouped.items():
        if 'ridge' not in methods:
            raise ValueError('paired scene summary requires ridge in every history')
        reference = {scene: sum(values) / len(values) for scene, values in methods['ridge'].items()}
        summary[history] = {}
        for variant, scenes in methods.items():
            if variant == 'ridge':
                continue
            compare = {scene: sum(values) / len(values) for scene, values in scenes.items()}
            summary[history][variant] = paired_bootstrap(reference, compare, seed=seed, samples=samples)
    return summary


def run_experiment(settings, adapter, output, *, split='val', samples=4, frames=None,
                   histories=('real',), variants=('ridge', 'no_read', 'prototype', 'shuffle_value')):
    frames = int(settings.get('frames', 121) if frames is None else frames)
    if frames < 10:
        raise ValueError('GRAIL experiment requires at least 10 latent frames')
    if samples < 1 or split not in {'val', 'test'}:
        raise ValueError('positive sample count and val/test split required')
    histories, variants = tuple(histories), tuple(variants)
    if not histories or set(histories) - {'real', 'generated'} or len(set(histories)) != len(histories):
        raise ValueError('histories must be unique real/generated selections')
    if 'ridge' not in variants or set(variants) - {'ridge', 'no_read', 'prototype', 'shuffle_value'}:
        raise ValueError('variants must include ridge and use supported read interventions')
    if not adapter or not Path(adapter).is_file():
        raise FileNotFoundError('a flow-trained GRAIL adapter checkpoint is required')
    if not torch.cuda.is_available():
        raise RuntimeError('full-checkpoint GRAIL experiment requires CUDA')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('GRAIL experiment is single-GPU only')
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a fresh experiment output directory')

    from .data import EpisodeDataset
    from .grail_native import load_grail_adapter
    from .grail_train import GrailEpisodeModel
    from .provenance import file_sha256
    from .sana import load_config, make_fixture, make_pipeline

    data = EpisodeDataset(settings['data'], settings['manifest'], split, frames=frames)
    selected, seen = [], set()
    for index, row in enumerate(data.rows):
        if row['scene_id'] not in seen:
            selected.append((index, row['scene_id']))
            seen.add(row['scene_id'])
        if len(selected) >= samples:
            break
    if not selected:
        raise ValueError(f'no scenes in {split} split')
    device = 'cuda'
    pipe = make_pipeline(load_config(settings['sana_config']), settings['base_checkpoint'], device, training=True)
    base_hash = file_sha256(settings['base_checkpoint'])
    ctl, extra = load_grail_adapter(pipe.model, adapter, mode='online', base_checkpoint_hash=base_hash,
                                    require_flow_trained=True)
    module = GrailEpisodeModel(pipe.model, steps=settings.get('steps', 4),
                               flow_shift=pipe.config.scheduler.inference_flow_shift,
                               association_weight=settings.get('association_weight', 1.),
                               detach_every=settings.get('detach_every', 0)).eval()
    output.mkdir(parents=True)
    metadata = dict(split=split, requested_scenes=samples, selected_scenes=len(selected), frames=frames,
                    histories=histories, variants=variants, adapter=str(adapter), adapter_sha256=file_sha256(adapter),
                    base_checkpoint_sha256=base_hash, manifest_sha256=file_sha256(settings['manifest']),
                    adapter_step=extra.get('step'), gpu=torch.cuda.get_device_name())
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    records = []
    with (output / 'episodes.jsonl').open('w', encoding='utf-8') as stream:
        for index, scene_id in selected:
            sample = data[index]
            if sample['latent'].shape[1] != frames:
                raise ValueError(f"{sample['key']} has {sample['latent'].shape[1]} latents; requested {frames}")
            fixture = make_fixture(sample, pipe, device)
            query_fingerprint = None
            for history in histories:
                seed = int(settings.get('seed', 3407)) + index
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad():
                    result = module(**fixture, generated=history == 'generated', seed=seed,
                                    query_variants=variants, record_trace=True)
                torch.cuda.synchronize()
                row = episode_record(result, scene_id=scene_id, key=sample['key'], history=history, seed=seed)
                if query_fingerprint is not None and row['query_input_fingerprint'] != query_fingerprint:
                    raise RuntimeError('real/generated histories used different held-out query inputs')
                query_fingerprint = row['query_input_fingerprint']
                row['seconds'] = time.perf_counter() - started
                row['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
                row['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                records.append(row)
                del result
            del fixture, sample
    summary = paired_scene_summary(records, seed=int(settings.get('seed', 3407)))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
                                         encoding='utf-8')
    return summary
