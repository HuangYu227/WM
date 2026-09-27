import pytest
import torch
from types import SimpleNamespace

from worldttt.data import (validate_manifest, validate_episode, require_requested_frames,
                           require_vae_stride, native_frame_limit)


def test_scene_split_cannot_leak_and_keys_are_unique():
    rows = [{'key': 'a/1', 'scene_id': 's', 'split': 'train'},
            {'key': 'a/2', 'scene_id': 's', 'split': 'val'}]
    with pytest.raises(ValueError, match='scene'):
        validate_manifest(rows)
    with pytest.raises(ValueError, match='Duplicate'):
        validate_manifest([rows[0], rows[0]])


def test_episode_checks_finite_camera_and_length():
    sample = dict(latent=torch.zeros(128, 10, 2, 2), camera=torch.zeros(10, 20),
                  plucker=torch.zeros(48, 10, 2, 2), prompt='room', key='a')
    sample['camera'][:, :16] = torch.eye(4).flatten()
    sample['camera'][:, 16:18] = 1
    validate_episode(sample)
    sample['camera'][1, 16] = float('nan')
    with pytest.raises(ValueError, match='camera'):
        validate_episode(sample)


def test_fixture_uses_model_dtype_for_camera_and_plucker():
    from worldttt.sana import fixture_to_device, make_fixture

    sample = dict(latent=torch.ones(2, 3, 4, 5), camera=torch.ones(3, 20),
                  plucker=torch.ones(48, 3, 4, 5), prompt='room', key='scene')
    pipeline = SimpleNamespace(weight_dtype=torch.bfloat16,
        _encode_prompts=lambda *args: (torch.ones(1, 2, 6), torch.ones(1, 2, dtype=torch.bool), None, None))
    fixture = make_fixture(sample, pipeline, 'cpu')
    for key in ('latent', 'text', 'camera', 'plucker'):
        assert fixture[key].dtype == torch.bfloat16
    assert fixture['mask'].dtype == torch.bool
    restored = fixture_to_device(fixture, 'cpu', torch.float32)
    for key in ('latent', 'text', 'camera', 'plucker'):
        assert restored[key].dtype == torch.float32
    assert restored['episode_id'] == 'scene'


def test_requested_horizon_rejects_short_latent_cache():
    with pytest.raises(ValueError, match='requested 121'):
        require_requested_frames(torch.zeros(128, 10, 2, 2), 121, 'research/short')
    require_requested_frames(torch.zeros(128, 121, 2, 2), 121, 'research/long')


def test_chunk_plucker_requires_eight_raw_frames_per_latent():
    require_vae_stride(SimpleNamespace(vae_time_stride=8))
    with pytest.raises(ValueError, match='stride 8'):
        require_vae_stride(SimpleNamespace(vae_time_stride=4))


def test_native_frame_limit_preserves_requested_latents_and_camera_horizon():
    assert native_frame_limit(10) == 73
    assert native_frame_limit(121) == 961
