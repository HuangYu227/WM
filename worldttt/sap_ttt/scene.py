"""Deterministic planar scenes with exact instance/visibility correspondence."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProceduralScene:
    seed: int
    height: int = 128
    width: int = 224
    prompt: str = field(init=False)
    sticker_codes: tuple[int, ...] = field(init=False)

    @property
    def condition(self) -> str:
        return ('relation_same_instance', 'ambiguous_same_instance',
                'relation_hard_negative', 'ambiguous_hard_negative')[self.seed % 4]

    def __post_init__(self):
        if min(self.height, self.width) < 8:
            raise ValueError('Scene resolution is too small')
        rng = np.random.default_rng(self.seed)
        self.sticker_codes = tuple(int(i) for i in rng.integers(100000, 999999, size=6))
        centers = np.array([-7.25, -4.75, -1.25, 1.25, 4.75, 7.25])
        rng.shuffle(centers)
        self.centers = tuple(centers + rng.uniform(-.15, .15, size=6))
        self.start_x = float(rng.uniform(-.35, .35))
        self.start_yaw = float(rng.uniform(-.05, .05))
        self.far_right = float(rng.uniform(5., 7.))
        self.far_left = float(rng.uniform(5., 7.))
        self.return_side = int(rng.choice((-1, 1)))
        self.return_dx = self.return_side * float(rng.uniform(.1, .35))
        self.return_yaw = self.return_side * float(rng.uniform(.12, .2))
        self.window_side = -1 if self.seed % 2 == 0 else 1
        self.prompt = ('A corridor with similar red panels and a blue window to the '
                       + ('left' if self.window_side < 0 else 'right')
                       + '. The target is the red panel closest to that window.') if self.seed % 4 in (0, 2) else (
                           'A corridor with several similar red wall panels and a blue window.')

    def camera(self, frame: int) -> np.ndarray:
        if not 0 <= frame <= 96:
            raise ValueError('The four-chunk trajectory uses frames 0..96')
        times = [0, 24, 48, 72, 84, 96]
        x = np.interp(frame, times, [self.start_x, self.start_x + self.far_right,
                                     self.start_x - self.far_left, self.start_x,
                                     self.start_x + 3., self.start_x + self.return_dx])
        yaw = np.interp(frame, times, [self.start_yaw, self.start_yaw + .04,
                                       self.start_yaw - .04, self.start_yaw,
                                       self.start_yaw + .05 * self.return_side,
                                       self.start_yaw + self.return_yaw])
        cosine, sine = np.cos(yaw), np.sin(yaw)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = ((cosine, 0., sine), (0., 1., 0.), (-sine, 0., cosine))
        matrix[0, 3] = x
        return matrix

    @property
    def intrinsics(self) -> np.ndarray:
        focal = .9 * self.width
        return np.array([[focal, 0, (self.width - 1) / 2],
                         [0, focal, (self.height - 1) / 2],
                         [0, 0, 1]], dtype=np.float32)

    def render(self, frame: int) -> dict[str, np.ndarray]:
        c2w = self.camera(frame)
        K = self.intrinsics
        yy, xx = np.mgrid[:self.height, :self.width]
        dx = (xx - K[0, 2]) / K[0, 0]
        dy = -(yy - K[1, 2]) / K[1, 1]
        cam_x = c2w[0, 3]
        ray_x = c2w[0, 0] * dx + c2w[0, 2]
        ray_z = c2w[2, 0] * dx + c2w[2, 2]
        wall_x, wall_y = cam_x + 10 * ray_x / ray_z, 10 * dy / ray_z
        world = np.stack((wall_x, wall_y, np.full_like(dx, 10.)), -1).astype(np.float32)
        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        grid = ((np.floor(wall_x * 2) + np.floor(wall_y * 2)) % 2).astype(np.uint8)
        rgb[:] = np.stack((180 + 9 * grid, 177 + 8 * grid, 168 + 8 * grid), -1)
        window = (np.abs(wall_x - 3.6 * self.window_side) < .8) & (np.abs(wall_y) < 1.2)
        rgb[window] = np.array([43, 122, 185], dtype=np.uint8)
        instance = np.zeros((self.height, self.width), dtype=np.uint8)
        depth = np.full((self.height, self.width), 10., dtype=np.float32)
        obj_x, obj_y = cam_x + 5 * ray_x / ray_z, 5 * dy / ray_z
        for identity, center in enumerate(self.centers, start=1):
            inside = (np.abs(obj_x - center) < 1.05) & (np.abs(obj_y) < .95)
            if not inside.any():
                continue
            u = (obj_x - center + 1.05) / 2.1
            v = (obj_y + .95) / 1.9
            stripe = (np.floor(u * 9) % 2).astype(np.uint8)
            color = np.stack((155 + 12 * stripe, 52 + 4 * stripe, 49 + 4 * stripe), -1)
            sticker = (np.abs(u - .5) < .3) & (np.abs(v - .15) < .13)
            bits = np.array(list(map(int, f'{self.sticker_codes[identity - 1]:020b}')), dtype=np.uint8)
            column = np.clip(((u - .2) / .6 * len(bits)).astype(int), 0, len(bits) - 1)
            mark = np.where(bits[column, None] == 1,
                            np.array([242, 215, 65], dtype=np.uint8),
                            np.array([35, 71, 198], dtype=np.uint8))
            color = np.where(sticker[..., None], mark, color)
            rgb[inside] = color[inside]
            instance[inside] = identity
            depth[inside] = 5.
            world[inside, 0] = obj_x[inside]
            world[inside, 1] = obj_y[inside]
            world[inside, 2] = 5.
        return {'rgb': rgb, 'instance': instance, 'depth': depth, 'world': world,
                'c2w': c2w, 'intrinsics': K}

    def correspondences(self, first: dict, second: dict) -> np.ndarray:
        ids = first['instance']
        row, col = np.where(ids > 0)
        xyz = first['world'][row, col]
        camera_xyz = (xyz - second['c2w'][:3, 3]) @ second['c2w'][:3, :3]
        K = second['intrinsics']
        new_col = np.rint(K[0, 0] * camera_xyz[:, 0] / camera_xyz[:, 2] + K[0, 2]).astype(int)
        new_row = np.rint(-K[1, 1] * camera_xyz[:, 1] / camera_xyz[:, 2] + K[1, 2]).astype(int)
        valid = (new_row >= 0) & (new_row < self.height) & (new_col >= 0) & (new_col < self.width)
        row, col, new_row, new_col = row[valid], col[valid], new_row[valid], new_col[valid]
        visible = ((second['instance'][new_row, new_col] == ids[row, col]) &
                   (np.abs(second['depth'][new_row, new_col] - first['depth'][row, col]) < .1))
        return np.stack((row[visible], col[visible], new_row[visible], new_col[visible]), -1)
