"""CPU checks for native GRAIL geometry and projection heads."""

import pytest
import torch

from worldttt.associative_ttt import AssociativeTTTConfig, AssociativeTTTLedger
from worldttt.grail_geometry import canonical_token_geometry
from worldttt.grail_native import GrailNativeController


def _camera(translation_x=0.0):
    pose = torch.eye(4, dtype=torch.float64)
    pose[0, 3] = translation_x
    return torch.cat((pose.reshape(16), torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float64))).view(1, 1, 20)


def test_identity_camera_center_ray_and_feature_order():
    depth = torch.tensor([[[0.7]]], dtype=torch.float64)
    visible = torch.ones_like(depth)
    geo = canonical_token_geometry(_camera(), (1, 1, 1), 1, depth, visible)
    assert geo.shape == (1, 1, 30)
    torch.testing.assert_close(geo[0, 0, :16], torch.eye(4, dtype=torch.float64).reshape(-1))
    torch.testing.assert_close(geo[0, 0, 16:20], torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float64))
    torch.testing.assert_close(geo[0, 0, 20:22], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(geo[0, 0, 22:28], torch.tensor([0., 0., 1., 0., 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(geo[0, 0, 28:], torch.tensor([1., 0.7], dtype=torch.float64))


def test_later_chunk_retains_episode_anchor_coordinates():
    geo = canonical_token_geometry(_camera(5.0), (1, 1, 1), 1,
                                   torch.ones(1, 1, 1, dtype=torch.float64),
                                   torch.ones(1, 1, 1, dtype=torch.float64))
    assert geo[0, 0, 3] == 5.0
    torch.testing.assert_close(geo[0, 0, 22:28], torch.tensor([0., 0., 1., 0., -5., 0.], dtype=torch.float64))


@pytest.mark.parametrize("change", ["nan", "bad_intrinsics", "bad_depth", "bad_shape"])
def test_malformed_geometry_fails_before_write(change):
    cam = _camera()
    depth = torch.ones(1, 1, 1, dtype=torch.float64)
    visible = torch.ones_like(depth)
    if change == "nan":
        cam[0, 0, 0] = float("nan")
    elif change == "bad_intrinsics":
        cam[0, 0, 16] = 0
    elif change == "bad_depth":
        depth[0, 0, 0] = float("inf")
    else:
        visible = visible[..., 0]
    with pytest.raises((ValueError, FloatingPointError)):
        canonical_token_geometry(cam, (1, 1, 1), 1, depth, visible)


def test_native_controller_rejects_mislabeled_v1_geometry():
    with pytest.raises(ValueError, match="geometry"):
        GrailNativeController(4, AssociativeTTTConfig(geometry_dim=30))
