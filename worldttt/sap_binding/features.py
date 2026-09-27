"""Encode paired scenes and capture selected frozen SANA layers for SAP-Bind."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .data import validate_case
from .config import FIVE_BINDING_LAYERS


BOUNDS = (0, 4, 7, 10, 13)


def encode_scene(settings, case, output):
    from diffusion.model.builder import vae_encode
    from inference_video_scripts.wm.inference_sana_wm import prepare_camera
    from worldttt.sana import load_config, make_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError('SAP-Bind VAE encoding requires CUDA')
    height, width = 704, 1280
    scene, c2w, intrinsics = validate_case(case, height, width)
    config = load_config(settings['sana_config'])
    pipe = make_pipeline(config, settings['base_checkpoint'], 'cuda', training=False)
    frames = np.stack([scene.render(i)['rgb'] for i in range(97)])
    video = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].to('cuda', dtype=pipe.vae_dtype)
    video = video.div(127.5).sub(1.)
    pipe.vae.to('cuda')
    pipe.vae.enable_tiling(tile_sample_min_height=384, tile_sample_min_width=384,
                           tile_sample_stride_height=320, tile_sample_stride_width=320)
    with torch.no_grad():
        latent = vae_encode(config.vae.vae_type, pipe.vae, video, sample_posterior=False,
                            device='cuda').to(pipe.weight_dtype).cpu()
    pipe.vae.to('cpu'); torch.cuda.empty_cache()
    if latent.shape[2:] != (13, 22, 40):
        raise ValueError(f'Expected 13x22x40 latent, received {tuple(latent.shape)}')
    camera = prepare_camera(c2w, intrinsics, target_size=(height, width),
                            vae_stride=config.vae.vae_stride)
    with torch.no_grad():
        text, mask, _, _ = pipe._encode_prompts(case['prompt'], '')
    fixture = dict(version=1, source='sap_binding_procedural_ground_truth', scene_id=case['id'],
        layout_id=case['layout_id'], variant=case['variant'], seed=case['seed'],
        latent=latent, camera=camera['raymap'][None].cpu(),
        plucker=camera['chunk_plucker'][None].cpu(), text=text.cpu(), mask=mask.cpu(),
        base_checkpoint=settings['base_checkpoint'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.pt')
    torch.save(fixture, temporary)
    temporary.replace(output)
    return output


def finalize_feature_records(source_paths, fixture_data, output, *, layers):
    """Attach historical latent Values and keep every layer's record separate."""
    layers = tuple(layers)
    if not isinstance(source_paths, dict):
        if len(layers) != 1:
            raise ValueError('Multilayer feature capture requires one source path per layer')
        source_paths = {layers[0]: source_paths}
    if set(source_paths) != set(layers):
        raise ValueError('SAP-Bind captured layer set mismatch')
    paths = {}
    for layer in layers:
        record = torch.load(source_paths[layer], map_location='cpu', weights_only=True)
        if record.get('layer') != layer or record.get('source') != 'teacher_forced_ground_truth':
            raise ValueError(f'SAP-Bind layer {layer} received the wrong SANA record')
        record['source'] = 'sap_binding_teacher_forced_ground_truth'
        record['layout_id'] = fixture_data['layout_id']
        record['variant'] = fixture_data['variant']
        for chunk, part in enumerate(record['supports']):
            start, end = BOUNDS[chunk:chunk + 2]
            part['latent'] = fixture_data['latent'][:, :, start:end].clone()
        record['query_obscured'] = True
        path = (Path(output) if len(layers) == 1 else
                Path(output) / f'layer_{layer}' / record['scene_id'] / 'features.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        torch.save(record, temporary)
        temporary.replace(path)
        paths[layer] = path
    return paths


def extract_features(settings, fixture, supervision, output):
    from worldttt.sap_ttt.features import extract_features as extract_legacy

    fixture_data = torch.load(fixture, map_location='cpu', weights_only=True)
    local = copy.deepcopy(settings)
    binding = settings['sap_binding']
    local['sap'] = dict(mode='frozen', layers=binding.get('layers', FIVE_BINDING_LAYERS),
        address_mode='selective_geometry', dim=256, ray_dim=binding.get('ray_dim', 48),
        support_tokens=binding.get('support_tokens', 256), inner_lr=.5, seed=binding.get('seed', 3407))
    output = Path(output)
    layers = tuple(binding.get('layers', FIVE_BINDING_LAYERS))
    if len(layers) == 1:
        intermediate = output.with_suffix('.legacy.pt')
        source = extract_legacy(local, fixture, supervision, intermediate)
        try:
            return finalize_feature_records(source, fixture_data, output,
                                            layers=layers)[layers[0]]
        finally:
            intermediate.unlink(missing_ok=True)
    source = extract_legacy(local, fixture, supervision, output)
    return finalize_feature_records(source, fixture_data, output, layers=layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    encode = commands.add_parser('encode'); extract = commands.add_parser('extract')
    for command in (encode, extract):
        command.add_argument('--settings', required=True); command.add_argument('--output', required=True)
    encode.add_argument('--case', required=True)
    extract.add_argument('--fixture', required=True); extract.add_argument('--supervision', required=True)
    args = parser.parse_args(); settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    if args.command == 'encode':
        result = encode_scene(settings, json.loads(Path(args.case).read_text(encoding='utf-8')), args.output)
    else:
        result = extract_features(settings, args.fixture, args.supervision, args.output)
    print(result)


if __name__ == '__main__':
    main()
