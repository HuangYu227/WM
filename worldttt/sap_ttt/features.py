"""Encode procedural video and capture frozen chunk-causal SANA features.

The encoded fixture is teacher-forced GT; it is not evidence of generated
history improvement. Instance labels are loaded only after model forwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .pairs import match_historical_tokens
from .runtime import SapConfig, SapController
from .scene import ProceduralScene


BOUNDS = (0, 4, 7, 10, 13)


def save_feature_records(output, *, layers, scene_id, base_checkpoint, supports,
                         queries, queries_no_history, supervision, noise_sigmas):
    """Keep the existing single-layer format while saving five depths separately."""
    layers = tuple(layers)
    paths = {}
    for layer in layers:
        path = (Path(output) if len(layers) == 1 else
                Path(output) / f'layer_{layer}' / scene_id / 'features.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {'version': 1, 'source': 'teacher_forced_ground_truth',
                    'scene_id': scene_id, 'base_checkpoint': base_checkpoint,
                    'layer': layer,
                    'supports': [part[layer] for part in supports],
                    'queries': [part[layer] for part in queries],
                    'queries_no_history': [part[layer] for part in queries_no_history],
                    'support_sources': ['reference_plus_ground_truth', 'ground_truth', 'ground_truth'],
                    'supervision': supervision, 'noise_sigmas': noise_sigmas}
        temporary = path.with_name(path.name + '.tmp')
        try:
            torch.save(record, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        paths[layer] = path
    return paths


def validate_scene_case(case: dict, height: int, width: int):
    """Reject mixed case files after a procedural scene revision."""
    from PIL import Image

    scene = ProceduralScene(case['seed'], height, width)
    with Image.open(case['image']) as image:
        first = np.asarray(image.convert('RGB'))
    camera = np.load(case['camera'], allow_pickle=False)
    intrinsics = np.load(case['intrinsics'], allow_pickle=False)
    expected_camera = np.stack([scene.camera(i) for i in range(97)])
    k = scene.intrinsics
    expected_intrinsics = np.repeat(
        np.array([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]]], dtype=np.float32), 97, axis=0)
    if (first.shape != (height, width, 3) or
            not np.array_equal(first, scene.render(0)['rgb']) or
            not np.array_equal(camera, expected_camera) or
            not np.array_equal(intrinsics, expected_intrinsics)):
        raise ValueError('Procedural case files are stale or mixed; re-export into a new directory')
    return scene, camera, intrinsics


def encode_scene(settings: dict, case: dict, output: str | Path):
    from diffusion.model.builder import vae_encode
    from inference_video_scripts.wm.inference_sana_wm import prepare_camera
    from worldttt.sana import load_config, make_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError('SANA VAE feature fixtures require CUDA')
    height, width = 704, 1280
    scene, c2w, intrinsics = validate_scene_case(case, height, width)
    config = load_config(settings['sana_config'])
    pipe = make_pipeline(config, settings['base_checkpoint'], 'cuda', training=False)
    frames = np.stack([scene.render(i)['rgb'] for i in range(97)])
    video = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].to('cuda', dtype=pipe.vae_dtype)
    video = video.div(127.5).sub(1.)
    del frames
    pipe.vae.to('cuda')
    pipe.vae.enable_tiling(tile_sample_min_height=384, tile_sample_min_width=384,
                           tile_sample_stride_height=320, tile_sample_stride_width=320)
    with torch.no_grad():
        latent = vae_encode(config.vae.vae_type, pipe.vae, video, sample_posterior=False,
                            device='cuda').to(pipe.weight_dtype).cpu()
    del video
    pipe.vae.to('cpu')
    torch.cuda.empty_cache()
    if latent.shape[2:] != (13, 22, 40):
        raise ValueError(f'Expected 13x22x40 latent from 97x704x1280 frames, got {tuple(latent.shape)}')
    camera = prepare_camera(c2w, intrinsics, target_size=(height, width),
                            vae_stride=config.vae.vae_stride)
    with torch.no_grad():
        text, mask, _, _ = pipe._encode_prompts(case['prompt'], '')
    fixture = {'version': 1, 'source': 'procedural_ground_truth', 'scene_id': case['id'],
               'seed': case['seed'], 'latent': latent, 'camera': camera['raymap'][None].cpu(),
               'plucker': camera['chunk_plucker'][None].cpu(), 'text': text.cpu(),
               'mask': mask.cpu(), 'base_checkpoint': settings['base_checkpoint']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, output)
    return output


def extract_features(settings: dict, fixture_path: str | Path, supervision_path: str | Path,
                     output: str | Path):
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
    from worldttt.check_cache import require_gate
    from worldttt.runtime import clone_cache
    from worldttt.sana import fixture_to_device, load_config, make_pipeline
    from worldttt.episode import flow_schedule

    if not torch.cuda.is_available():
        raise RuntimeError('SANA feature extraction requires CUDA')
    require_gate(settings)
    fixture = torch.load(fixture_path, map_location='cpu', weights_only=True)
    if fixture['version'] != 1 or fixture['latent'].shape[2] != 13:
        raise ValueError('SAP extraction requires one 4+3+3+3 fixture')
    if fixture['base_checkpoint'] != settings['base_checkpoint']:
        raise ValueError('Fixture/backbone checkpoint mismatch')
    config = load_config(settings['sana_config'])
    # The training pipeline intentionally skips VAE construction. Encoding
    # procedural RGB therefore needs the inference pipeline's ordinary VAE.
    pipe = make_pipeline(config, settings['base_checkpoint'], 'cuda', training=False)
    pipe.vae.to('cpu')
    torch.cuda.empty_cache()
    sap_cfg = SapConfig(**dict(settings['sap'], mode='frozen'))
    controller = SapController(pipe.model, sap_cfg)
    controller.reset_episode(fixture['scene_id'], batch=1)
    inputs = fixture_to_device(fixture, 'cuda', pipe.weight_dtype)
    model = pipe.model
    holder = SimpleNamespace(num_model_blocks=len(model.blocks), num_cached_blocks=2,
                             sink_token=False, _chunk_indices=list(BOUNDS))
    caches = SelfForcingFlowEulerCamCtrl._initialize_kv_cache(holder, 4)
    def old_cache(i):
        return clone_cache(SelfForcingFlowEulerCamCtrl._accumulate_softmax_kv_cache(holder, caches, i)[0])
    supports = []
    with torch.no_grad():
        for chunk in range(3):
            start, end = BOUNDS[chunk:chunk + 2]
            ctx = controller.context(torch.zeros(1, device='cuda'), record_raw=True)
            _, updated = model(inputs['latent'][:, :, start:end], torch.zeros(1, device='cuda'),
                y=inputs['text'], mask=inputs['mask'], camera_conditions=inputs['camera'][:, start:end],
                chunk_plucker=inputs['plucker'][:, :, start:end], start_f=start, end_f=end,
                frame_index=torch.arange(start, end, device='cuda'), kv_cache=old_cache(chunk),
                save_kv_cache=True, sap_context=ctx, data_info={})
            caches[chunk] = clone_cache(updated)
            supports.append(ctx.raw_features)
        start, end = BOUNDS[3:5]
        scheduler = flow_schedule(50, config.scheduler.inference_flow_shift, 'cuda')
        taus = scheduler.timesteps[[0, len(scheduler.timesteps) // 2, -1]]
        queries = []
        queries_no_history = []
        for i, tau in enumerate(taus):
            rng = torch.Generator(device='cuda').manual_seed(fixture['seed'] + 1009 * i)
            clean = inputs['latent'][:, :, start:end]
            sigma = tau.float() / 1000.
            noise = torch.randn(clean.shape, generator=rng, device='cuda', dtype=clean.dtype)
            noisy = (1 - sigma) * clean + sigma * noise
            times = tau.expand(1, 1, end - start)
            ctx = controller.context(times, record_raw=True)
            model(noisy, times, y=inputs['text'], mask=inputs['mask'],
                  camera_conditions=inputs['camera'][:, start:end],
                  chunk_plucker=inputs['plucker'][:, :, start:end], start_f=start, end_f=end,
                  frame_index=torch.arange(start, end, device='cuda'), kv_cache=old_cache(3),
                  save_kv_cache=False, sap_context=ctx, data_info={})
            queries.append(ctx.raw_features)
            empty_ctx = controller.context(times, record_raw=True)
            empty_cache = SelfForcingFlowEulerCamCtrl._initialize_kv_cache(holder, 1)[0]
            model(noisy, times, y=inputs['text'], mask=inputs['mask'],
                  camera_conditions=inputs['camera'][:, start:end],
                  chunk_plucker=inputs['plucker'][:, :, start:end], start_f=start, end_f=end,
                  frame_index=torch.arange(start, end, device='cuda'), kv_cache=empty_cache,
                  save_kv_cache=False, sap_context=empty_ctx, data_info={})
            queries_no_history.append(empty_ctx.raw_features)
    with np.load(supervision_path, allow_pickle=False) as labels:
        positive, valid = match_historical_tokens(labels['instance'][:4], labels['world'][:4],
                                                  labels['instance'][10:13], labels['world'][10:13])
        supervised = {'positive': torch.from_numpy(positive), 'valid': torch.from_numpy(valid),
                      'support_instance': torch.from_numpy(labels['instance'][:10].copy()),
                      'query_instance': torch.from_numpy(labels['instance'][10:13].copy())}
    paths = save_feature_records(output, layers=sap_cfg.layers, scene_id=fixture['scene_id'],
        base_checkpoint=settings['base_checkpoint'], supports=supports, queries=queries,
        queries_no_history=queries_no_history, supervision=supervised,
        noise_sigmas=[float(t / 1000) for t in taus])
    return next(iter(paths.values())) if len(paths) == 1 else paths


def main():
    parser = argparse.ArgumentParser(description='Prepare or extract SAP-TTT feature fixtures')
    commands = parser.add_subparsers(dest='command', required=True)
    encode = commands.add_parser('encode')
    extract = commands.add_parser('extract')
    for cmd in (encode, extract):
        cmd.add_argument('--settings', required=True)
        cmd.add_argument('--output', required=True)
    encode.add_argument('--case', required=True)
    extract.add_argument('--fixture', required=True)
    extract.add_argument('--supervision', required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    if args.command == 'encode':
        case = json.loads(Path(args.case).read_text(encoding='utf-8'))
        encode_scene(settings, case, args.output)
    else:
        extract_features(settings, args.fixture, args.supervision, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
