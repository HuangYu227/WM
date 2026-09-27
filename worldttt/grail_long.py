"""Camera-return selection and native GRAIL rollouts; no instance ground truth."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


CRITERIA = dict(return_radius_fraction=.05, return_angle_deg=15.,
                departure_radius_fraction=.25, departure_angle_deg=45.,
                min_departure_frames=3, min_gap_latents=12, window_stride=3,
                query_pose='center_of_final_three_latents',
                translation_scale='window_camera_trajectory_diameter')


def find_return_window(camera, frames):
    """One deterministic window, using camera poses only, never model losses.

    The first visit must be in the first third of the window, followed by a
    sustained departure and a return at the final query's center frame.
    Translation thresholds are fractions of trajectory diameter, not meters.
    """
    poses = np.asarray(camera, dtype=np.float64)
    if poses.ndim == 2 and poses.shape[1] == 20:
        poses = poses[:, :16].reshape(-1, 4, 4)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError('finite camera poses [T,4,4] required')
    if frames < 16 or frames % 3 != 1:
        raise ValueError('return windows require frames=3*n+1 and at least 16')
    for start in range(0, len(poses) - frames + 1, CRITERIA['window_stride']):
        window = poses[start:start + frames]
        position, rotation = window[:, :3, 3], window[:, :3, :3]
        diameter = float(np.linalg.norm(position[:, None] - position[None], axis=-1).max())
        query = frames - 2
        for first in range(min(frames // 3, query - CRITERIA['min_gap_latents'] + 1)):
            distance = np.linalg.norm(position - position[first], axis=-1)
            cosine = (np.einsum('tij,ij->t', rotation, rotation[first]) - 1.) / 2.
            angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
            if (distance[query] > max(1e-6, diameter * CRITERIA['return_radius_fraction'])
                    or angle[query] > CRITERIA['return_angle_deg']):
                continue
            away = ((distance > max(1e-5, diameter * CRITERIA['departure_radius_fraction']))
                    | (angle > CRITERIA['departure_angle_deg']))[first + 1:frames - 3]
            consecutive = CRITERIA['min_departure_frames']
            if len(away) < consecutive or np.convolve(away.astype(int), np.ones(consecutive, int), 'valid').max() < consecutive:
                continue
            return dict(start=start, frames=frames, first_visit=start + first,
                        query_center=start + query, gap_latents=query - first,
                        return_distance=float(distance[query]), return_angle_deg=float(angle[query]),
                        trajectory_diameter=diameter, camera_return_proxy=True,
                        ground_truth_instances=False)
    return None


def slice_episode(sample, case):
    start, frames = int(case['start']), int(case['frames'])
    if start < 0 or frames < 10 or start + frames > sample['latent'].shape[1]:
        raise ValueError('case window exceeds cached latent horizon')
    stop = start + frames
    # Keep the original episode camera anchor and the aligned Plucker channels.
    return dict(sample, latent=sample['latent'][:, start:stop], camera=sample['camera'][start:stop],
                plucker=sample['plucker'][:, start:stop], key=f"{sample['key']}@{start}:{stop}")


def prepare(settings_path, output, max_steps=300, clips_per_scene=8):
    from .data import EpisodeDataset, validate_manifest
    from .provenance import file_sha256

    if max_steps < 3 or clips_per_scene < 1:
        raise ValueError('max_steps >= 3 and clips_per_scene >= 1 required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'audit.json').exists():
        raise FileExistsError('use a fresh experiment directory')
    settings = json.loads(Path(settings_path).read_text(encoding='utf-8'))
    rows = validate_manifest([json.loads(s) for s in Path(settings['manifest']).read_text(encoding='utf-8').splitlines() if s.strip()])
    counts = {split: dict(clips=sum(r['split'] == split for r in rows),
                         scenes=len({r['scene_id'] for r in rows if r['split'] == split}))
              for split in ('train', 'val', 'test')}
    print('Unchanged manifest split counts:', json.dumps(counts), flush=True)
    # This is a new long-history pilot, not a resume of the two-step smoke.
    settings.update(frames=121, steps=20, max_steps=max_steps, real_prefix_steps=max_steps // 6,
                    curriculum=[[0, 10], [max_steps // 3, 20], [2 * max_steps // 3, 40]],
                    save_every=50, val_max_samples=4, shuffle_train=True, diagnostics_every=25)
    (output / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    data = EpisodeDataset(settings['data'], settings['manifest'], 'val', frames=121)
    candidates = {h: [] for h in (31, 61, 121)}
    selected = {h: [] for h in candidates}
    scene_counts = {h: Counter() for h in candidates}
    for index, row in enumerate(data.rows):
        item = data.native.dataset[data.indices[index]]
        # Reuse native pose scaling, episode anchoring and temporal sampling.
        # A 1x1 spatial grid avoids loading VAE latents during pose selection.
        camera, _ = data.native._read_camera_data(item, {}, 121, 1, 1)
        for horizon in candidates:
            case = find_return_window(camera.numpy(), horizon)
            if case is None:
                continue
            case.update(key=row['key'], scene_id=row['scene_id'])
            candidates[horizon].append(case)
            if scene_counts[horizon][row['scene_id']] < clips_per_scene:
                selected[horizon].append(case)
                scene_counts[horizon][row['scene_id']] += 1
        if (index + 1) % 25 == 0:
            print(f'Camera audit {index + 1}/{len(data)} clips', flush=True)
    manifest_hash = file_sha256(settings['manifest'])
    for horizon, cases in selected.items():
        if cases:
            payload = dict(frames=horizon, source_frames=121, split='val',
                           manifest_sha256=manifest_hash, criteria=CRITERIA, cases=cases)
            (output / f'cases-{horizon}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    audit = dict(split_counts=counts, manifest_sha256=manifest_hash, criteria=CRITERIA,
                 inference_scope='single_scene_exploratory' if counts['val']['scenes'] < 2 else 'multi_scene',
                 eligible_clips={h: len(v) for h, v in candidates.items()},
                 selected_clips={h: len(v) for h, v in selected.items()},
                 status='ready' if any(selected.values()) else 'no_camera_return_cases',
                 note='Camera return is a geometric proxy, not instance identity or visual overlap ground truth.')
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)
    if not any(selected.values()):
        raise ValueError('No qualifying camera-return windows. Inspect audit.json; thresholds were not relaxed.')
    return audit


def load_cases(path, settings, split, frames):
    from .provenance import file_sha256
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if (payload['manifest_sha256'] != file_sha256(settings['manifest'])
            or payload['split'] != split or payload['frames'] != frames):
        raise ValueError('case file does not match the manifest, split and requested horizon')
    if not payload['cases']:
        raise ValueError('empty return-case selection')
    return payload


def sample_native(settings_path, adapter, cases_path, output, steps=20):
    """Full native sampler off/online comparison, distinct from same-cache queries."""
    import time
    import torch
    from .data import EpisodeDataset
    from .grail_native import load_grail_adapter, TARGET_LAYERS
    from .grail_resume import input_fingerprint
    from .provenance import file_sha256
    from .sana import load_config, make_fixture, make_pipeline
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl

    if steps < 1:
        raise ValueError('positive denoising step count required')
    settings = json.loads(Path(settings_path).read_text(encoding='utf-8'))
    horizon = json.loads(Path(cases_path).read_text(encoding='utf-8'))['frames']
    payload = load_cases(cases_path, settings, 'val', horizon)
    data = EpisodeDataset(settings['data'], settings['manifest'], 'val', frames=payload['source_frames'])
    case = payload['cases'][0]  # fixed by cameras before model evaluation
    index = next(i for i, row in enumerate(data.rows) if row['key'] == case['key'] and row['scene_id'] == case['scene_id'])
    sample = slice_episode(data[index], case)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pipe = make_pipeline(load_config(settings['sana_config']), settings['base_checkpoint'], 'cuda', training=True)
    ctl, extra = load_grail_adapter(pipe.model, adapter, base_checkpoint_hash=file_sha256(settings['base_checkpoint']),
                                    require_flow_trained=True)
    fixture = make_fixture(sample, pipe, 'cuda')
    seed = int(settings.get('seed', 3407))
    noise = torch.randn(fixture['latent'].shape, device='cuda', dtype=fixture['latent'].dtype,
                        generator=torch.Generator(device='cuda').manual_seed(seed))
    noise[:, :, 0] = fixture['latent'][:, :, 0]
    reference = fixture['latent'].detach().cpu()
    torch.save(reference, output / 'reference.pt')
    records = {}
    for mode in ('off', 'online'):
        ctl.mode = mode
        kwargs = dict(camera_conditions=fixture['camera'], chunk_plucker=fixture['plucker'], mask=fixture['mask'],
                      data_info={'condition_frame_info': {0: 0.}})
        sampler = SelfForcingFlowEulerCamCtrl(pipe.model, fixture['text'], fixture['text'] * 0, 1.,
            flow_shift=pipe.config.scheduler.inference_flow_shift, base_chunk_frames=3,
            num_cached_blocks=2, model_kwargs=kwargs)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            latent = sampler.sample(noise.clone(), steps=steps)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if not torch.isfinite(latent).all():
            raise FloatingPointError(f'nonfinite native rollout: {mode}')
        if not torch.equal(latent[:, :, 0], fixture['latent'][:, :, 0]):
            raise RuntimeError('native sampler modified the conditioned first frame')
        if mode == 'online':
            chunks = (horizon - 1) // 3
            if (len(ctl.metrics) != chunks or not all(row['committed'] for row in ctl.metrics)
                    or ctl.hook_counts != {layer: chunks * (steps + 1) for layer in TARGET_LAYERS}):
                raise RuntimeError('native rollout did not execute all denoising reads and clean writes')
        torch.save(latent.detach().cpu(), output / f'{mode}.pt')
        records[mode] = dict(seconds=seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            latent_query_mse=float((latent[:, :, -3:].float().cpu() - reference[:, :, -3:].float()).square().mean()),
            hook_counts=dict(ctl.hook_counts) if mode == 'online' else {},
            committed_chunks=len(ctl.metrics) if mode == 'online' else 0)
        del latent, sampler
    metadata = dict(case=case, seed=seed, steps=steps, cfg_scale=1., adapter_step=extra.get('step'),
                    history_sampling_steps=settings.get('steps', 4),
                    initial_latent_fingerprint=input_fingerprint(noise),
                    adapter_sha256=file_sha256(adapter), manifest_sha256=payload['manifest_sha256'],
                    comparison='full native autoregressive off/online; histories and caches may diverge',
                    reference='decoded cached VAE latents, not original raw video', records=records)
    (output / 'rollout.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


def decode(settings_path, output):
    import torch
    from diffusion.model.builder import get_vae
    from diffusion.model.utils import get_weight_dtype
    from diffusion.utils.logger import get_root_logger
    from inference_video_scripts.wm.inference_sana_wm import write_video
    from sana.tools import resolve_hf_path
    from .sana import load_config
    from .vae_stream import decode_ltx2_video

    settings = json.loads(Path(settings_path).read_text(encoding='utf-8'))
    config = load_config(settings['sana_config'])
    vae = get_vae(config.vae.vae_type, resolve_hf_path(config.vae.vae_pretrained), device='cuda',
                  dtype=get_weight_dtype(config.vae.weight_dtype), config=config.vae)
    vae.enable_tiling(tile_sample_min_height=384, tile_sample_min_width=384,
                      tile_sample_stride_height=320, tile_sample_stride_width=320)
    output = Path(output)
    for name in ('reference', 'off', 'online'):
        latent = torch.load(output / f'{name}.pt', map_location='cpu', weights_only=True)
        video = decode_ltx2_video(vae, latent, 'cuda')
        expected = 1 + 8 * (latent.shape[2] - 1)
        if len(video) != expected:
            raise ValueError(f'decoded frame count {len(video)} != {expected}')
        write_video(output, name, video, 16, get_root_logger())
        print(f'Decoded {name}: {len(video)} frames', flush=True)
        del latent, video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'sample', 'decode'])
    parser.add_argument('--settings', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--cases')
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--clips-per-scene', type=int, default=8)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.settings, args.output, args.max_steps, args.clips_per_scene)
    elif args.command == 'sample':
        if not args.adapter or not args.cases:
            parser.error('sample requires --adapter and --cases')
        sample_native(args.settings, args.adapter, args.cases, args.output, args.steps)
    else:
        decode(args.settings, args.output)


if __name__ == '__main__':
    main()
