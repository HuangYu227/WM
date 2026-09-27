"""Atomic, protocol-checked rollout snapshots for native GRAIL inference."""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any

import torch

from .associative_ttt import AssociativeTTTConfig, AssociativeTTTState
from .runtime import move_tree


_REQUIRED_PROTOCOL = (
    "base_checkpoint_hash", "source_tree_hash", "data_manifest_hash",
    "config_fingerprint", "coordinate_convention", "cfg_policy", "layer_ids",
    "shape", "steps", "cfg_scale", "flow_shift", "boundaries",
)
_STATE_METADATA = _REQUIRED_PROTOCOL[:7]


def input_fingerprint(*values):
    """Hash complete input tensor trees, including dtype/shape and prompt/camera data."""
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update((str(value.dtype) + str(tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif value is None or isinstance(value, (str, bool, int, float)):
            digest.update((type(value).__name__ + ':' + repr(value) + ';').encode())
        else:
            raise TypeError(f'unsupported GRAIL protocol input type: {type(value).__name__}')
    visit(values)
    return digest.hexdigest()


def validate_grail_protocol(protocol: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_PROTOCOL if not protocol.get(key)]
    if missing:
        raise ValueError(f"GRAIL rollout protocol missing values: {missing}")


def _validate_cache(cache: Any, protocol: dict[str, Any]) -> None:
    if not isinstance(cache, list) or len(cache) != len(protocol["boundaries"]) - 1:
        raise ValueError("GRAIL cache chunk count differs from rollout protocol")


def _validate_adapter(adapter, protocol):
    if 'adapter_hash' in protocol:
        from .grail_native import adapter_fingerprint
        if adapter is None or adapter_fingerprint(adapter) != protocol['adapter_hash']:
            raise ValueError('GRAIL embedded adapter does not match rollout protocol')


def save_grail_rollout(
    path: Path,
    *,
    state: AssociativeTTTState,
    kv_cache: Any,
    latents: torch.Tensor,
    init_latents: torch.Tensor,
    chunk_cursor: int,
    rng_state: torch.Tensor | None,
    protocol: dict[str, Any],
    adapter: dict | None = None,
) -> None:
    """Save only a completed clean-commit boundary, replacing the file atomically."""
    validate_grail_protocol(protocol)
    _validate_adapter(adapter, protocol)
    if chunk_cursor < 1 or not bool(torch.all(state.last_committed_chunk == chunk_cursor - 1)):
        raise ValueError("GRAIL state and rollout chunk cursor disagree")
    for key in _STATE_METADATA:
        if state.metadata.get(key) != protocol[key]:
            raise ValueError(f"GRAIL state {key} differs from rollout protocol")
    if list(latents.shape) != protocol["shape"] or init_latents.shape != latents.shape:
        raise ValueError("GRAIL latent shape differs from rollout protocol")
    _validate_cache(kv_cache, protocol)
    snapshot = {
        "version": 1,
        "protocol": dict(protocol),
        "chunk_cursor": chunk_cursor,
        "state": state.to_payload(),
        "kv_cache": move_tree(kv_cache, "cpu"),
        "latents": latents.detach().cpu().clone(),
        "init_latents": init_latents.detach().cpu().clone(),
        "rng_state": None if rng_state is None else rng_state.detach().cpu().clone(),
        "adapter": adapter,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(snapshot, temporary)
    temporary.replace(path)


def load_grail_rollout(
    path: Path,
    *,
    config: AssociativeTTTConfig,
    expected_protocol: dict[str, Any],
    expected_next_chunk: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Validate everything before returning a restorable state/cache bundle."""
    validate_grail_protocol(expected_protocol)
    snapshot = torch.load(Path(path), map_location="cpu", weights_only=True)
    if snapshot.get("version") != 1 or snapshot.get("protocol") != expected_protocol:
        raise ValueError("GRAIL rollout protocol mismatch")
    _validate_adapter(snapshot.get('adapter'), expected_protocol)
    cursor = snapshot.get("chunk_cursor")
    if not isinstance(cursor, int) or cursor < 1 or (expected_next_chunk is not None and cursor != expected_next_chunk):
        raise ValueError("GRAIL rollout chunk cursor mismatch")
    state = AssociativeTTTState.from_payload(
        snapshot["state"], config, device=device,
        expected_metadata={key: expected_protocol[key] for key in _STATE_METADATA},
    )
    if not bool(torch.all(state.last_committed_chunk == cursor - 1)):
        raise ValueError("GRAIL rollout state cursor mismatch")
    latents = snapshot["latents"]
    init_latents = snapshot["init_latents"]
    if list(latents.shape) != expected_protocol["shape"] or init_latents.shape != latents.shape:
        raise ValueError("GRAIL rollout latent shape mismatch")
    _validate_cache(snapshot["kv_cache"], expected_protocol)
    return {
        "state": state,
        "kv_cache": move_tree(snapshot["kv_cache"], device),
        "latents": latents.to(device),
        "init_latents": init_latents.to(device),
        "rng_state": snapshot["rng_state"],
        "chunk_cursor": cursor,
    }
