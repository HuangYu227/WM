"""Bound GPU memory while decoding long SANA-WM videos with the same LTX-2 VAE."""

import numpy as np
import torch


def decode_temporal_tiles(latent, decode_tile, *, temporal_ratio, tile_frames=16, stride_frames=8):
    """Match Diffusers temporal tiling, retaining decoded frames on CPU as uint8."""
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError('Expected one B,C,T,H,W latent video')
    if (tile_frames <= stride_frames or stride_frames < temporal_ratio
            or tile_frames % temporal_ratio or stride_frames % temporal_ratio):
        raise ValueError('Temporal tile and stride must be positive multiples of the VAE ratio')
    expected_frames = (latent.shape[2] - 1) * temporal_ratio + 1
    latent_tile = tile_frames // temporal_ratio
    latent_stride = stride_frames // temporal_ratio
    blend_frames = tile_frames - stride_frames
    video = None
    previous = None
    written = 0

    for start in range(0, latent.shape[2], latent_stride):
        decoded = decode_tile(latent[:, :, start:start + latent_tile + 1]).detach()
        if start:
            decoded = decoded[:, :, :-1]
        decoded = decoded.cpu()
        if previous is not None:
            extent = min(previous.shape[2], decoded.shape[2], blend_frames)
            for index in range(extent):
                decoded[:, :, index] = (previous[:, :, -extent + index] * (1 - index / extent)
                                         + decoded[:, :, index] * (index / extent))
        keep = stride_frames + (1 if start == 0 else 0)
        keep = min(keep, decoded.shape[2], expected_frames - written)
        if keep:
            piece = torch.clamp(127.5 * decoded[:, :, :keep] + 127.5, 0, 255)
            piece = piece.permute(0, 2, 3, 4, 1).to(torch.uint8).numpy()[0]
            if video is None:
                if piece.shape[-1] != 3:
                    raise ValueError('VAE decoder must return RGB video')
                video = np.empty((expected_frames, *piece.shape[1:]), dtype=np.uint8)
            video[written:written + keep] = piece
            written += keep
        previous = decoded

    if written != expected_frames:
        raise RuntimeError(f'VAE temporal decode returned {written} frames; expected {expected_frames}')
    return video


def decode_ltx2_video(vae, latent, device, *, tile_frames=16, stride_frames=8):
    """Decode the normal SANA LTX-2 VAE on small GPU tiles and assemble on CPU."""
    from diffusion.model.builder import vae_decode

    if not getattr(vae, 'use_tiling', False):
        raise ValueError('Enable spatial VAE tiling before streamed decode')
    ratio = int(vae.temporal_compression_ratio)
    framewise = vae.use_framewise_decoding
    vae.use_framewise_decoding = False  # This function owns the temporal loop.
    try:
        with torch.no_grad():
            return decode_temporal_tiles(
                latent,
                lambda tile: vae_decode('LTX2VAE_diffusers', vae, tile.to(device=device, dtype=vae.dtype)),
                temporal_ratio=ratio, tile_frames=tile_frames, stride_frames=stride_frames,
            )
    finally:
        vae.use_framewise_decoding = framewise
