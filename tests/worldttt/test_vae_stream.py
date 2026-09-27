import numpy as np
import pytest
import torch

from worldttt.vae_stream import decode_temporal_tiles


def _fake_decode(tile):
    # One latent frame followed by eight output frames per extra latent.
    return torch.cat((tile[:, :, :1], tile[:, :, 1:].repeat_interleave(8, dim=2)), dim=2)


def _reference(latent, tile_frames=16, stride_frames=8):
    # The temporal tile and blend order in Diffusers AutoencoderKLLTX2Video.
    row = []
    for i in range(0, latent.shape[2], stride_frames // 8):
        tile = latent[:, :, i:i + tile_frames // 8 + 1]
        decoded = _fake_decode(tile)
        if i:
            decoded = decoded[:, :, :-1]
        row.append(decoded)
    result = []
    for i, tile in enumerate(row):
        if i:
            extent = min(row[i - 1].shape[2], tile.shape[2], tile_frames - stride_frames)
            for j in range(extent):
                tile[:, :, j] = row[i - 1][:, :, -extent + j] * (1 - j / extent) + tile[:, :, j] * (j / extent)
            result.append(tile[:, :, :stride_frames])
        else:
            result.append(tile[:, :, :stride_frames + 1])
    pixels = torch.cat(result, dim=2)[:, :, :(latent.shape[2] - 1) * 8 + 1]
    return torch.clamp(127.5 * pixels + 127.5, 0, 255).permute(0, 2, 3, 4, 1).to(torch.uint8).numpy()[0]


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_streamed_temporal_decode_matches_diffusers_blending(dtype):
    latent = torch.linspace(-1, 1, 3 * 6 * 2 * 3).reshape(1, 3, 6, 2, 3).to(dtype)
    calls = []

    def decode(tile):
        calls.append(tile.shape[2])
        return _fake_decode(tile)

    result = decode_temporal_tiles(latent, decode, temporal_ratio=8, tile_frames=16, stride_frames=8)
    np.testing.assert_array_equal(result, _reference(latent))
    assert result.shape == (41, 2, 3, 3)
    assert max(calls) <= 3


def test_streamed_temporal_decode_handles_full_benchmark_length():
    latent = torch.zeros(1, 3, 121, 1, 1)
    result = decode_temporal_tiles(latent, _fake_decode, temporal_ratio=8, tile_frames=16, stride_frames=8)
    assert result.shape == (961, 1, 1, 3)
