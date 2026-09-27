"""Offline labels for historical correspondences; never passed to model reads."""
from __future__ import annotations

import numpy as np


def match_historical_tokens(old_instance, old_world, query_instance, query_world, *, max_distance=.18):
    if old_instance.ndim != 3 or query_instance.ndim != 3:
        raise ValueError('Instance labels must be (frames,height,width)')
    if old_world.shape != (*old_instance.shape, 3) or query_world.shape != (*query_instance.shape, 3):
        raise ValueError('World coordinate/instance label shape mismatch')
    old_id, new_id = old_instance.reshape(-1), query_instance.reshape(-1)
    old_xyz, new_xyz = old_world.reshape(-1, 3), query_world.reshape(-1, 3)
    positive = np.full(len(new_id), -1, dtype=np.int64)
    for identity in np.intersect1d(np.unique(old_id), np.unique(new_id)):
        if identity == 0:
            continue
        old_positions = np.flatnonzero(old_id == identity)
        query_positions = np.flatnonzero(new_id == identity)
        distances = np.linalg.norm(new_xyz[query_positions, None] - old_xyz[None, old_positions], axis=-1)
        nearest = distances.argmin(axis=1)
        close = distances[np.arange(len(query_positions)), nearest] <= max_distance
        positive[query_positions[close]] = old_positions[nearest[close]]
    return positive, positive >= 0
