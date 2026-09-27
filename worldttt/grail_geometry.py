"""Episode-anchor camera geometry at SANA's spatial-token resolution."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def geometry_distance(a: torch.Tensor, b: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Symmetric proxy-point/ray distance; leading dimensions broadcast.

    The depth component is learned, not a measured metric depth. `scale` is
    in the episode camera translation units. Unlike cosine of a flattened
    pose, this distinguishes parallel rays observing different positions.
    """
    oa, ob = a[..., [3, 7, 11]], b[..., [3, 7, 11]]
    da, db = F.normalize(a[..., 22:25], dim=-1), F.normalize(b[..., 22:25], dim=-1)
    pa, pb = oa + a[..., -1:] * da, ob + b[..., -1:] * db
    ab, ba = pa - ob, pb - oa
    ra = ab - (ab * db).sum(-1, keepdim=True) * db
    rb = ba - (ba * da).sum(-1, keepdim=True) * da
    return (0.5 * (pa - pb).square().sum(-1)
            + 0.25 * (ra.square().sum(-1) + rb.square().sum(-1))) / scale ** 2


def canonical_token_geometry(
    camera_conditions: torch.Tensor,
    thw: tuple[int, int, int],
    patch_size: int,
    depth_proxy: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Return pose, intrinsics, UV, Plücker ray, visibility and depth [B,N,30].

    ``camera_conditions`` is already relative to the *episode* anchor. A
    sliced chunk must never be re-anchored to its own first camera.
    """
    if len(thw) != 3 or any(not isinstance(n, int) or n < 1 for n in thw):
        raise ValueError("thw must contain three positive integers")
    if not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    t, h, w = thw
    if camera_conditions.ndim != 3 or camera_conditions.shape[1:] != (t, 20):
        raise ValueError("camera_conditions must be [B,T,20]")
    b = camera_conditions.shape[0]
    expected = (b, t * h * w, 1)
    if depth_proxy.shape != expected or visibility.shape != expected:
        raise ValueError("depth_proxy and visibility must be [B,T*H*W,1]")
    if not all(torch.isfinite(x).all().item() for x in (camera_conditions, depth_proxy, visibility)):
        raise FloatingPointError("non-finite camera geometry")
    if (visibility < 0).any() or (visibility > 1).any() or (depth_proxy < 0).any():
        raise ValueError("visibility must lie in [0,1] and depth must be non-negative")

    cam = camera_conditions.to(device=depth_proxy.device, dtype=depth_proxy.dtype)
    pose = cam[..., :16].reshape(b, t, 4, 4)
    intrinsics = cam[..., 16:]
    if (intrinsics[..., :2] <= 0).any():
        raise ValueError("fx and fy must be positive")
    target_last = pose.new_tensor([0, 0, 0, 1])
    if not torch.allclose(pose[..., 3, :], target_last.expand(b, t, 4), atol=1e-3, rtol=0):
        raise ValueError("camera pose must be a homogeneous C2W transform")

    yy, xx = torch.meshgrid(
        torch.arange(h, device=cam.device, dtype=cam.dtype),
        torch.arange(w, device=cam.device, dtype=cam.dtype),
        indexing="ij",
    )
    uv = torch.stack(((xx + 0.5) * (2.0 / w) - 1.0,
                      (yy + 0.5) * (2.0 / h) - 1.0), dim=-1)
    pixel_x = (xx + 0.5) * patch_size
    pixel_y = (yy + 0.5) * patch_size
    fx, fy, cx, cy = (intrinsics[..., i, None, None] for i in range(4))
    ray_camera = torch.stack(((pixel_x - cx) / fx, (pixel_y - cy) / fy,
                              torch.ones_like(pixel_x).expand(b, t, h, w)), dim=-1)
    ray_camera = F.normalize(ray_camera, dim=-1)
    direction = F.normalize(torch.einsum("btij,bthwj->bthwi", pose[..., :3, :3], ray_camera), dim=-1)
    origin = pose[..., :3, 3][:, :, None, None, :].expand(b, t, h, w, 3)
    moment = torch.cross(origin, direction, dim=-1)
    n = t * h * w
    return torch.cat((
        cam[:, :, None, None, :].expand(b, t, h, w, 20).reshape(b, n, 20),
        uv.expand(b, t, h, w, 2).reshape(b, n, 2),
        direction.reshape(b, n, 3), moment.reshape(b, n, 3),
        visibility, depth_proxy,
    ), dim=-1)
