"""Decode a saved Stage-1 latent without rerunning the 50-step world model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .sana import load_config
from .vae_stream import decode_ltx2_video


def decode_saved(settings_path, run):
    from diffusion.model.builder import get_vae
    from diffusion.model.utils import get_weight_dtype
    from diffusion.utils.logger import get_root_logger
    from inference_video_scripts.wm.inference_sana_wm import write_video
    from sana.tools import resolve_hf_path

    if not torch.cuda.is_available():
        raise RuntimeError('Saved latent decoding requires CUDA')
    run = Path(run)
    case = json.loads((run / 'case.json').read_text(encoding='utf-8'))
    settings = json.loads(Path(settings_path).read_text(encoding='utf-8'))
    config = load_config(settings['sana_config'])
    latent = torch.load(run / 'latent.pt', map_location='cpu', weights_only=True)
    expected = int(case['num_frames'])
    if latent.ndim != 5 or (latent.shape[2] - 1) * config.vae.vae_stride[0] + 1 != expected:
        raise ValueError('Saved latent length does not match the benchmark case')

    dtype = get_weight_dtype(config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, resolve_hf_path(config.vae.vae_pretrained),
                  device='cuda', dtype=dtype, config=config.vae)
    vae.enable_tiling(tile_sample_min_height=384, tile_sample_min_width=384,
                      tile_sample_stride_height=320, tile_sample_stride_width=320)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    video = decode_ltx2_video(vae, latent, 'cuda')
    seconds = time.perf_counter() - started
    del vae
    torch.cuda.empty_cache()
    write_video(run, 'video', video, 16, get_root_logger())
    np.save(run / 'c2w.npy', np.load(case['camera'], allow_pickle=False))
    summary = dict(decoded_frames=len(video), expected_frames=expected,
                   decode_seconds=seconds,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (run / 'decode_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--run', required=True, help='Failed infer directory with case.json and latent.pt')
    args = parser.parse_args()
    print(json.dumps(decode_saved(args.settings, args.run), indent=2))


if __name__ == '__main__':
    main()
