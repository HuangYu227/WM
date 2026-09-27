import json

import numpy as np
import pytest
import torch
from torch import nn

from worldttt.grail_long import find_return_window, load_cases, slice_episode
from worldttt.grail_train import train_step
from worldttt.provenance import file_sha256


def camera(frames=31):
    return np.repeat(np.eye(4)[None], frames, axis=0)


def test_return_requires_departure_and_temporal_gap():
    poses = camera()
    assert find_return_window(poses, 31) is None
    poses[:, 0, 3] = np.arange(31)
    assert find_return_window(poses, 31) is None
    poses[:, 0, 3] = 0
    poses[8:24, 0, 3] = 1
    case = find_return_window(poses, 31)
    assert case['gap_latents'] >= 12
    assert case['return_distance'] == 0
    assert case['query_center'] == 29
    assert case['ground_truth_instances'] is False
    poses[:, 0, 3] = 0
    poses[8:10, 0, 3] = 1
    assert find_return_window(poses, 31) is None


def test_rotation_return_works_without_translation_and_is_scale_invariant():
    poses = camera()
    poses[8:24, :3, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert find_return_window(poses, 31) is not None
    poses[8:24, 0, 3] = 1
    a = find_return_window(poses, 31)
    poses[:, :3, 3] *= 100
    b = find_return_window(poses, 31)
    assert a['first_visit'] == b['first_visit'] and a['gap_latents'] == b['gap_latents']


def test_wrong_return_orientation_is_rejected():
    poses = camera()
    poses[8:24, 0, 3] = 1
    poses[-3:, :3, :3] = np.diag([-1., -1., 1.])
    assert find_return_window(poses, 31) is None


def test_window_keeps_original_camera_anchor_and_aligned_channels():
    sample = dict(latent=torch.arange(61).reshape(1, 61, 1, 1),
                  camera=torch.arange(61)[:, None].expand(61, 20),
                  plucker=torch.arange(61).reshape(1, 61, 1, 1), key='data/clip')
    result = slice_episode(sample, dict(start=3, frames=31))
    assert result['latent'].shape[1] == 31
    assert result['camera'][0, 0] == 3
    assert result['plucker'][0, -1, 0, 0] == 33
    assert sample['latent'].shape[1] == 61
    with pytest.raises(ValueError, match='exceeds'):
        slice_episode(sample, dict(start=40, frames=31))


def test_case_file_cannot_be_reused_with_a_different_split_or_manifest(tmp_path):
    manifest = tmp_path / 'scenes.jsonl'
    manifest.write_text('{}\n')
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(dict(manifest_sha256=file_sha256(manifest), split='val', frames=31,
                                    cases=[dict(start=0, frames=31)])))
    settings = dict(manifest=str(manifest))
    assert load_cases(path, settings, 'val', 31)['cases']
    with pytest.raises(ValueError, match='does not match'):
        load_cases(path, settings, 'test', 31)
    manifest.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match='does not match'):
        load_cases(path, settings, 'val', 31)


def test_future_gradient_diagnostics_separate_association_only_parameters():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.controller = nn.Module()
            self.controller.future = nn.Parameter(torch.tensor(2.))
            self.controller.association = nn.Parameter(torch.tensor(3.))

        def forward(self, **kwargs):
            future = self.controller.future.square()
            association = self.controller.association.square()
            return dict(loss=future + association, future=future, association=association)

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    row = train_step(model, optimizer, {}, diagnostics=True)
    assert row['future_gradients']['future']['norm'] == 4.
    assert row['future_gradients']['association']['with_gradient'] == 0
    assert row['total_gradients']['association']['norm'] == 6.
    assert model.controller.future.item() < 2.


def test_repeated_clips_and_seeds_do_not_create_a_single_scene_confidence_interval():
    from worldttt.grail_experiment import paired_scene_summary
    records = [dict(scene_id='one-scene', history='real', future_flow_mse={'ridge': .2, 'no_read': .3})] * 8
    result = paired_scene_summary(records, samples=20)['real']['no_read']
    assert result['scene_count'] == 1
    assert result['mean'] == pytest.approx(.1)
    assert result['ci95'] is None
