"""Closed-form, differentiable associative TTT ledger for WorldTTT-GRAIL.

This module is intentionally independent of the SANA implementation.  It is
the reference slow-clock state machine that a native SANA hook can call later.
The implementation has three properties that are important for the research
claim:

* ``precision`` is a *full* ``[key_dim, key_dim]`` ridge statistic.  It is not
  silently diagonalised.
* A differentiable functional commit is available.  Future losses can
  backpropagate through the sufficient-statistic update, while inference can
  use exactly the same update with ``differentiable=False``.
* Writes are transactional.  A failed finite/holdout check returns the old
  state and does not advance ``last_committed_chunk``.

The class does not own a video backbone.  Callers provide clean observation
keys, values and canonical geometry.  This makes the algebra testable on CPU
before it is connected to a SANA GDN/Softmax hook.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = 2


def _jsonable(value: Any) -> Any:
    """Convert config/metadata values into deterministic JSON values."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    raise TypeError(f"unsupported metadata/config value: {type(value)!r}")


@dataclass(frozen=True)
class AssociativeTTTConfig:
    """Experiment-facing ledger configuration.

    The defaults are the v2 main configuration.  ``precision_mode`` is kept
    as an explicit field so a diagonal approximation can be evaluated as a
    named ablation; this reference implementation intentionally rejects it.
    """

    key_dim: int = 32
    value_dim: int = 64
    geometry_dim: int = 13
    capacity: int = 256
    topk: int = 4
    decay: float = 0.995
    initial_precision: float = 1e-2
    solve_jitter: float = 1e-6
    temperature: float = 0.07
    geometry_temperature: float = 0.25
    geometry_weight: float = 0.25
    age_penalty: float = 0.0
    merge_threshold: float = 0.78
    generated_write_scale: float = 0.10
    min_write_confidence: float = 0.05
    min_read_confidence: float = 0.01
    min_retrieval_margin: float = -float("inf")
    max_uncertainty: float = 1.0
    protect_real_writes: bool = True
    precision_mode: str = "full"
    coordinate_convention: str = "episode_anchor_world_ray_v1"
    geometry_metric: str = "cosine"
    geometry_scale: float = 1.0
    geometry_cutoff: float = 9.0

    def __post_init__(self) -> None:
        if self.geometry_metric not in {"cosine", "ray_point"}:
            raise ValueError("unknown geometry metric")
        if self.geometry_metric == "ray_point" and self.geometry_dim != 30:
            raise ValueError("ray_point geometry needs 30 dimensions")
        if not math.isfinite(self.geometry_scale) or self.geometry_scale <= 0 or self.geometry_cutoff <= 0:
            raise ValueError("positive geometry scale and cutoff required")
        positive_ints = ("key_dim", "value_dim", "geometry_dim", "capacity", "topk")
        if any(int(getattr(self, name)) < 1 for name in positive_ints):
            raise ValueError("dimensions, capacity and topk must be positive")
        if self.topk > self.capacity:
            raise ValueError("topk cannot exceed capacity")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must lie in (0, 1]")
        for name in ("initial_precision", "solve_jitter", "temperature", "geometry_temperature"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.geometry_weight) or self.geometry_weight < 0:
            raise ValueError("geometry_weight must be finite and non-negative")
        if not math.isfinite(self.age_penalty) or self.age_penalty < 0:
            raise ValueError("age_penalty must be finite and non-negative")
        if not math.isfinite(self.merge_threshold):
            raise ValueError("merge_threshold must be finite")
        for name in ("generated_write_scale", "min_write_confidence", "min_read_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not math.isfinite(self.max_uncertainty) or not 0.0 <= self.max_uncertainty <= 1.0:
            raise ValueError("max_uncertainty must lie in [0, 1]")
        if self.precision_mode != "full":
            raise ValueError(
                "AssociativeTTTLedger implements full covariance only; "
                "use a separate named diagonal ablation"
            )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass
class ObservationBatch:
    """Clean chunk observations in one canonical coordinate system.

    All feature tensors use ``[B, N, C]``.  ``confidence`` and ``source_real``
    use ``[B, N]`` (a trailing singleton is accepted and normalised).  The
    caller must ensure that geometry is synchronized with the same chunk and
    camera convention as the query geometry.
    """

    keys: torch.Tensor
    values: torch.Tensor
    geometry: torch.Tensor
    confidence: torch.Tensor
    source_real: torch.Tensor

    def validate(self, config: AssociativeTTTConfig) -> "ObservationBatch":
        if self.keys.ndim != 3 or self.values.ndim != 3 or self.geometry.ndim != 3:
            raise ValueError("keys, values and geometry must be [B, N, C]")
        b, n, k = self.keys.shape
        if self.values.shape[:2] != (b, n) or self.values.shape[-1] != config.value_dim:
            raise ValueError("values shape does not match config")
        if self.geometry.shape[:2] != (b, n) or self.geometry.shape[-1] != config.geometry_dim:
            raise ValueError("geometry shape does not match config")
        if k != config.key_dim:
            raise ValueError("keys shape does not match config")
        if self.confidence.shape not in {(b, n), (b, n, 1)}:
            raise ValueError("confidence must be [B,N] or [B,N,1]")
        if self.source_real.shape not in {(b, n), (b, n, 1)}:
            raise ValueError("source_real must be [B,N] or [B,N,1]")
        tensors = (self.keys, self.values, self.geometry, self.confidence, self.source_real)
        if not all(torch.isfinite(x).all().item() for x in tensors):
            raise FloatingPointError("non-finite observation")
        if (self.confidence < 0).any() or (self.confidence > 1).any():
            raise ValueError("confidence must lie in [0,1]")
        return self

    def normalized(self) -> "ObservationBatch":
        confidence = self.confidence[..., 0] if self.confidence.ndim == 3 else self.confidence
        source_real = self.source_real[..., 0] if self.source_real.ndim == 3 else self.source_real
        return ObservationBatch(
            keys=F.normalize(self.keys, dim=-1, eps=1e-8),
            values=self.values,
            geometry=self.geometry,
            confidence=confidence,
            source_real=source_real.to(dtype=torch.bool),
        )


@dataclass
class HoldoutBatch:
    """Independent observations used only for transactional commit gating."""

    keys: torch.Tensor
    values: torch.Tensor
    geometry: torch.Tensor
    confidence: torch.Tensor

    def as_observation(self) -> ObservationBatch:
        source = torch.ones_like(self.confidence, dtype=torch.bool)
        return ObservationBatch(self.keys, self.values, self.geometry, self.confidence, source)


@dataclass
class AssociativeTTTState:
    """Persistent per-episode state for the slow clock.

    ``precision`` is full covariance ``[B,C,K,K]`` and ``cross`` is ``C`` in
    the ridge normal equations ``W P = C``.  ``last_committed_chunk`` is a
    tensor rather than a Python scalar so CFG/episode rows cannot silently
    drift apart.
    """

    episode_id: str
    keys: torch.Tensor                    # [B,C,K]
    geometries: torch.Tensor              # [B,C,G]
    values: torch.Tensor                  # [B,C,V] prototype/debug value
    cross: torch.Tensor                   # [B,C,V,K]
    precision: torch.Tensor               # [B,C,K,K] full covariance
    confidence: torch.Tensor              # [B,C]
    age: torch.Tensor                     # [B,C], last write chunk
    source_real: torch.Tensor             # [B,C]
    protected: torch.Tensor               # [B,C]
    generation: torch.Tensor              # [B,C]
    valid: torch.Tensor                   # [B,C]
    last_committed_chunk: torch.Tensor    # [B]
    update_count: torch.Tensor            # [B]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        config: AssociativeTTTConfig,
        episode_id: str,
        batch_size: int = 1,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "AssociativeTTTState":
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("state dtype must be floating point")
        dev = torch.device(device)
        b, c, k, v, g = batch_size, config.capacity, config.key_dim, config.value_dim, config.geometry_dim
        f = lambda *shape: torch.zeros(*shape, device=dev, dtype=dtype)
        precision = torch.eye(k, device=dev, dtype=dtype).view(1, 1, k, k).expand(b, c, -1, -1).clone()
        precision.mul_(config.initial_precision)
        md = {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": config.fingerprint(),
            "coordinate_convention": config.coordinate_convention,
            "cfg_policy": "shared_episode_ledger_conditional_observation",
            "layer_ids": [],
            "base_checkpoint_hash": "",
            "source_tree_hash": "",
            "data_manifest_hash": "",
        }
        if metadata:
            md.update(_jsonable(dict(metadata)))
        return cls(
            episode_id=str(episode_id),
            keys=f(b, c, k),
            geometries=f(b, c, g),
            values=f(b, c, v),
            cross=f(b, c, v, k),
            precision=precision,
            confidence=f(b, c),
            age=torch.full((b, c), -1, device=dev, dtype=torch.long),
            source_real=torch.zeros(b, c, device=dev, dtype=torch.bool),
            protected=torch.zeros(b, c, device=dev, dtype=torch.bool),
            generation=torch.zeros(b, c, device=dev, dtype=torch.long),
            valid=torch.zeros(b, c, device=dev, dtype=torch.bool),
            last_committed_chunk=torch.full((b,), -1, device=dev, dtype=torch.long),
            update_count=torch.zeros(b, device=dev, dtype=torch.long),
            metadata=md,
        )

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[1])

    @property
    def device(self) -> torch.device:
        return self.keys.device

    def clone(self, *, detach: bool = False) -> "AssociativeTTTState":
        def cp(x: Any) -> Any:
            if not isinstance(x, torch.Tensor):
                return x
            y = x.detach() if detach else x
            return y.clone()

        return AssociativeTTTState(
            episode_id=self.episode_id,
            keys=cp(self.keys),
            geometries=cp(self.geometries),
            values=cp(self.values),
            cross=cp(self.cross),
            precision=cp(self.precision),
            confidence=cp(self.confidence),
            age=cp(self.age),
            source_real=cp(self.source_real),
            protected=cp(self.protected),
            generation=cp(self.generation),
            valid=cp(self.valid),
            last_committed_chunk=cp(self.last_committed_chunk),
            update_count=cp(self.update_count),
            metadata=dict(self.metadata),
        )

    def detach(self) -> "AssociativeTTTState":
        return self.clone(detach=True)

    def validate(self, config: AssociativeTTTConfig) -> "AssociativeTTTState":
        b, c, k = self.keys.shape
        if (c, k) != (config.capacity, config.key_dim):
            raise ValueError("state key shape does not match config")
        if self.geometries.shape != (b, c, config.geometry_dim):
            raise ValueError("state geometry shape does not match config")
        if self.values.shape != (b, c, config.value_dim):
            raise ValueError("state value shape does not match config")
        if self.cross.shape != (b, c, config.value_dim, config.key_dim):
            raise ValueError("state cross shape does not match config")
        if self.precision.shape != (b, c, config.key_dim, config.key_dim):
            raise ValueError("state precision shape does not match config")
        for name in ("confidence", "age", "source_real", "protected", "generation", "valid"):
            if getattr(self, name).shape != (b, c):
                raise ValueError(f"state {name} shape does not match config")
        if self.last_committed_chunk.shape != (b,) or self.update_count.shape != (b,):
            raise ValueError("state chunk counters must be [B]")
        if self.metadata.get("config_fingerprint") != config.fingerprint():
            raise ValueError("state/config fingerprint mismatch")
        if self.metadata.get("coordinate_convention") != config.coordinate_convention:
            raise ValueError("state/config coordinate convention mismatch")
        if not all(torch.isfinite(x).all().item() for x in (self.keys, self.geometries, self.values, self.cross, self.precision, self.confidence)):
            raise FloatingPointError("non-finite ledger state")
        if not torch.equal(self.precision, self.precision.transpose(-1, -2)):
            # Small numerical asymmetry is harmless, but a large mismatch
            # indicates a caller mutated the state outside the transaction.
            asym = (self.precision - self.precision.transpose(-1, -2)).abs().max()
            if float(asym) > 1e-4:
                raise ValueError("precision must be symmetric")
        return self

    def to_payload(self) -> dict[str, Any]:
        tensors = {
            name: value.detach().cpu()
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor)
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "metadata": _jsonable(self.metadata),
            **tensors,
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.to_payload(), Path(path))

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        config: AssociativeTTTConfig,
        *,
        device: torch.device | str = "cpu",
        expected_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "AssociativeTTTState":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported AssociativeTTTState schema version")
        required = (
            "episode_id", "keys", "geometries", "values", "cross", "precision", "confidence", "age",
            "source_real", "protected", "generation", "valid", "last_committed_chunk", "update_count",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"state payload missing fields: {missing}")
        dev = torch.device(device)
        state = cls(
            episode_id=str(payload["episode_id"]),
            keys=payload["keys"].to(dev),
            geometries=payload["geometries"].to(dev),
            values=payload["values"].to(dev),
            cross=payload["cross"].to(dev),
            precision=payload["precision"].to(dev),
            confidence=payload["confidence"].to(dev),
            age=payload["age"].to(dev),
            source_real=payload["source_real"].to(dev),
            protected=payload["protected"].to(dev),
            generation=payload["generation"].to(dev),
            valid=payload["valid"].to(dev),
            last_committed_chunk=payload["last_committed_chunk"].to(dev),
            update_count=payload["update_count"].to(dev),
            metadata=dict(payload.get("metadata", {})),
        )
        state.validate(config)
        for name, expected in (expected_metadata or {}).items():
            if state.metadata.get(name) != _jsonable(expected):
                raise ValueError(f"state {name} mismatch")
        return state

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: AssociativeTTTConfig,
        *,
        device: torch.device | str = "cpu",
        expected_fingerprint: Optional[str] = None,
        expected_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "AssociativeTTTState":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = cls.from_payload(payload, config, device=device, expected_metadata=expected_metadata)
        if expected_fingerprint is not None and state.fingerprint() != expected_fingerprint:
            raise ValueError("loaded state fingerprint mismatch")
        return state

    def fingerprint(self) -> str:
        """Hash schema, metadata, shapes, dtypes and tensor bytes."""

        digest = hashlib.sha256()
        digest.update(str(SCHEMA_VERSION).encode())
        digest.update(self.episode_id.encode())
        digest.update(json.dumps(_jsonable(self.metadata), sort_keys=True, separators=(",", ":")).encode())
        for name in (
            "keys", "geometries", "values", "cross", "precision", "confidence", "age", "source_real",
            "protected", "generation", "valid", "last_committed_chunk", "update_count",
        ):
            tensor = getattr(self, name).detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()


@dataclass
class ReadResult:
    value: torch.Tensor                 # [B,Q,V], zero on fallback
    slot_ids: torch.Tensor              # [B,Q,K], -1 for invalid
    weights: torch.Tensor               # [B,Q,K]
    scores: torch.Tensor                # [B,Q,K]
    margin: torch.Tensor                # [B,Q]
    uncertainty: torch.Tensor           # [B,Q], lower is better
    has_memory: torch.Tensor            # [B,Q]
    raw_value: torch.Tensor              # before fallback, useful for diagnostics


@dataclass
class SlotRead:
    ids: torch.Tensor
    keys: torch.Tensor
    geometry: torch.Tensor
    values: torch.Tensor
    scores: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    margin: torch.Tensor
    uncertainty: torch.Tensor


class AssociativeTTTLedger(nn.Module):
    """Full-covariance associative memory with transactional closed-form TTT."""

    def __init__(self, config: Optional[AssociativeTTTConfig] = None):
        super().__init__()
        self.config = config or AssociativeTTTConfig()
        # The core ledger accepts already encoded keys/values.  These gates are
        # deliberately separate from camera control and are useful to a native
        # hook, but do not alter the algebraic state update.
        self.read_gate = nn.Parameter(torch.tensor(1.0))

    def new_state(self, episode_id: str, batch_size: int = 1, **kwargs: Any) -> AssociativeTTTState:
        return AssociativeTTTState.new(self.config, episode_id, batch_size, **kwargs)

    def _cast_observation(self, state: AssociativeTTTState, observation: ObservationBatch) -> ObservationBatch:
        observation.validate(self.config)
        dtype, device = state.precision.dtype, state.device
        return ObservationBatch(
            keys=observation.keys.to(device=device, dtype=dtype),
            values=observation.values.to(device=device, dtype=dtype),
            geometry=observation.geometry.to(device=device, dtype=dtype),
            confidence=observation.confidence.to(device=device, dtype=dtype),
            source_real=observation.source_real.to(device=device, dtype=torch.bool),
        ).normalized()

    def _route(
        self,
        state: AssociativeTTTState,
        observation: ObservationBatch,
        *,
        allow_replacement: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
        """Choose discrete slots; only this routing decision is detached.

        The returned assignment is intentionally non-differentiable.  The
        sufficient-statistic update after it remains differentiable with
        respect to keys/values/confidence, which is the desired functional TTT
        path.  Generated observations are never allowed to merge into a
        protected real slot.
        """

        b, n, _ = observation.keys.shape
        c = self.config.capacity
        assignments = torch.full((b, n), -1, dtype=torch.long, device=state.device)
        replace = torch.zeros((b, c), dtype=torch.bool, device=state.device)
        accepted = torch.zeros((b, n), dtype=torch.bool, device=state.device)
        gamma = torch.zeros((b, n), dtype=state.precision.dtype, device=state.device)
        stats = {"accepted": 0, "rejected_confidence": 0, "rejected_protected": 0, "replaced": 0}
        old_keys = F.normalize(state.keys.detach(), dim=-1, eps=1e-8)
        old_geom = F.normalize(state.geometries.detach(), dim=-1, eps=1e-8)
        obs_keys = observation.keys.detach()
        obs_geom = F.normalize(observation.geometry.detach(), dim=-1, eps=1e-8)
        for bi in range(b):
            # Routing within one clean chunk is sequential only for the
            # discrete assignment decision.  Without this local occupancy
            # view, two new observations would both select the same free slot
            # because ``state.valid`` is deliberately not mutated in-place.
            local_valid = state.valid[bi].detach().clone()
            local_protected = state.protected[bi].detach().clone()
            local_source_real = state.source_real[bi].detach().clone()
            local_confidence = state.confidence[bi].detach().clone()
            local_age = state.age[bi].detach().clone()
            local_keys = old_keys[bi].clone()
            local_geom = old_geom[bi].clone()
            local_raw_geometry = state.geometries[bi].detach().clone()
            for ni in range(n):
                conf = float(observation.confidence[bi, ni].detach().cpu())
                if conf < self.config.min_write_confidence:
                    stats["rejected_confidence"] += 1
                    continue
                is_real = bool(observation.source_real[bi, ni].detach().cpu())
                protected_rejection = False
                valid = local_valid
                if valid.any():
                    appearance_score = torch.mv(local_keys, obs_keys[bi, ni])
                    geometry_score = torch.mv(local_geom, obs_geom[bi, ni])
                    if self.config.geometry_metric == "ray_point":
                        from .grail_geometry import geometry_distance
                        distance = geometry_distance(local_raw_geometry, observation.geometry[bi, ni].detach(), self.config.geometry_scale)
                        geometry_score = -distance
                        valid = valid & (distance <= self.config.geometry_cutoff)
                    score = appearance_score + self.config.geometry_weight * geometry_score
                    score = score.masked_fill(~valid, -torch.inf)
                    best_score, best = score.max(dim=0)
                else:
                    best_score, best = torch.tensor(-torch.inf, device=state.device), torch.tensor(-1, device=state.device)
                slot = -1
                if float(best_score) >= self.config.merge_threshold:
                    candidate = int(best)
                    if not (not is_real and bool(local_protected[candidate].detach().cpu())):
                        slot = candidate
                    else:
                        protected_rejection = True
                if slot < 0:
                    free = (~local_valid).nonzero(as_tuple=False).flatten()
                    if len(free):
                        slot = int(free[0])
                    elif allow_replacement:
                        # Real observations may reclaim the weakest slot;
                        # generated observations may only replace generated,
                        # unprotected slots.
                        eligible = local_valid if is_real else local_valid & ~local_protected & ~local_source_real
                        if eligible.any():
                            utility = local_confidence + 1e-3 * local_age.float()
                            utility = utility.masked_fill(~eligible, torch.inf)
                            slot = int(utility.argmin())
                            # Replacing a locally allocated slot invalidates every
                            # earlier observation assigned to its old instance.
                            superseded = assignments[bi] == slot
                            stats['accepted'] -= int(superseded.sum())
                            accepted[bi] = accepted[bi] & ~superseded
                            assignments[bi] = assignments[bi].masked_fill(superseded, -1)
                            gamma[bi] = gamma[bi].masked_fill(superseded, 0.)
                            local_source_real[slot] = False
                            local_protected[slot] = False
                            replace[bi, slot] = True
                            stats["replaced"] += 1
                        else:
                            protected_rejection = True
                if slot >= 0:
                    assignments[bi, ni] = slot
                    accepted[bi, ni] = True
                    scale = 1.0 if is_real else self.config.generated_write_scale
                    gamma[bi, ni] = observation.confidence[bi, ni] * scale
                    stats["accepted"] += 1
                    local_valid[slot] = True
                    local_keys[slot] = obs_keys[bi, ni]
                    local_geom[slot] = obs_geom[bi, ni]
                    local_raw_geometry[slot] = observation.geometry[bi, ni].detach()
                    local_confidence[slot] = conf * scale
                    local_age[slot] = state.last_committed_chunk[bi] + 1
                    local_source_real[slot] |= is_real
                    local_protected[slot] |= bool(is_real and self.config.protect_real_writes)
                elif protected_rejection:
                    stats["rejected_protected"] += 1
        return assignments, replace, accepted, gamma, stats

    def _proposal(
        self,
        state: AssociativeTTTState,
        observation: ObservationBatch,
        assignments: torch.Tensor,
        replace: torch.Tensor,
        accepted: torch.Tensor,
        gamma: torch.Tensor,
        chunk_id: int,
    ) -> AssociativeTTTState:
        b, n, _ = observation.keys.shape
        c = self.config.capacity
        # One-hot routing is discrete, but multiplying it with differentiable
        # observations preserves the outer gradient through the write.
        safe_assign = assignments.clamp_min(0)
        one_hot = F.one_hot(safe_assign, num_classes=c).to(gamma.dtype)
        one_hot = one_hot * accepted.unsqueeze(-1).to(gamma.dtype)  # [B,N,C]
        mass = torch.einsum("bnc,bn->bc", one_hot, gamma)
        key_sum = torch.einsum("bnc,bnk,bn->bck", one_hot, observation.keys, gamma)
        value_sum = torch.einsum("bnc,bnv,bn->bcv", one_hot, observation.values, gamma)
        geom_sum = torch.einsum("bnc,bng,bn->bcg", one_hot, observation.geometry, gamma)
        cross_sum = torch.einsum("bnc,bnv,bnk,bn->bcvk", one_hot, observation.values, observation.keys, gamma)
        precision_sum = torch.einsum("bnc,bnk,bnl,bn->bckl", one_hot, observation.keys, observation.keys, gamma)
        touched = mass > 0
        keep_old = state.valid & ~replace
        old_weight = state.confidence * keep_old.to(state.confidence.dtype)
        old_mass = old_weight * self.config.decay
        denominator = (old_mass + mass).clamp_min(1e-8)

        keys_num = old_mass.unsqueeze(-1) * state.keys + key_sum
        values_num = old_mass.unsqueeze(-1) * state.values + value_sum
        geom_num = old_mass.unsqueeze(-1) * state.geometries + geom_sum
        new_keys = torch.where(touched.unsqueeze(-1), F.normalize(keys_num / denominator.unsqueeze(-1), dim=-1, eps=1e-8), state.keys)
        new_values = torch.where(touched.unsqueeze(-1), values_num / denominator.unsqueeze(-1), state.values)
        new_geometry = torch.where(touched.unsqueeze(-1), geom_num / denominator.unsqueeze(-1), state.geometries)

        old_cross = torch.where(keep_old.unsqueeze(-1).unsqueeze(-1), state.cross * self.config.decay, torch.zeros_like(state.cross))
        eye = torch.eye(self.config.key_dim, device=state.device, dtype=state.precision.dtype).view(1, 1, self.config.key_dim, self.config.key_dim)
        old_precision = torch.where(
            keep_old.unsqueeze(-1).unsqueeze(-1),
            state.precision * self.config.decay,
            self.config.initial_precision * eye,
        )
        new_cross = old_cross + cross_sum
        new_precision = old_precision + precision_sum
        new_precision = 0.5 * (new_precision + new_precision.transpose(-1, -2))

        new_valid = state.valid | touched
        new_confidence = torch.where(touched, (old_mass + mass).clamp(0, 1), state.confidence * self.config.decay)
        new_age = torch.where(touched, torch.full_like(state.age, chunk_id), state.age)
        incoming_real = torch.einsum("bnc,bn->bc", one_hot, observation.source_real.to(gamma.dtype) * gamma) > 0
        new_source_real = torch.where(touched, incoming_real | (state.source_real & keep_old), state.source_real)
        new_protected = torch.where(
            touched,
            (state.protected & keep_old) | (incoming_real & bool(self.config.protect_real_writes)),
            state.protected,
        )
        generation_inc = replace.to(state.generation.dtype)
        new_generation = state.generation + generation_inc
        return AssociativeTTTState(
            episode_id=state.episode_id,
            keys=new_keys,
            geometries=new_geometry,
            values=new_values,
            cross=new_cross,
            precision=new_precision,
            confidence=new_confidence,
            age=new_age,
            source_real=new_source_real,
            protected=new_protected,
            generation=new_generation,
            valid=new_valid,
            last_committed_chunk=torch.full_like(state.last_committed_chunk, chunk_id),
            update_count=state.update_count + 1,
            metadata=dict(state.metadata),
        )

    def _ridge_weights(self, precision: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        p = 0.5 * (precision + precision.transpose(-1, -2))
        eye = torch.eye(self.config.key_dim, device=p.device, dtype=p.dtype)
        chol, info = torch.linalg.cholesky_ex(p + self.config.solve_jitter * eye)
        if bool((info != 0).any()):
            raise FloatingPointError("ledger precision is not positive definite")
        # Solve P X = C^T, then transpose to W = C P^{-1}.
        return torch.cholesky_solve(cross.transpose(-1, -2), chol).transpose(-1, -2)

    def _read_loss(self, state: AssociativeTTTState, holdout: ObservationBatch) -> torch.Tensor:
        result = self.read(state, holdout.keys, holdout.geometry, topk=self.config.topk, chunk_id=None)
        # A missing-memory read predicts zero, giving the first commit a
        # meaningful holdout baseline instead of an artificial zero loss.
        valid = holdout.confidence > 0
        if not bool(valid.any()):
            return torch.zeros((), device=state.device, dtype=state.precision.dtype)
        error = (result.value - holdout.values).square().mean(dim=-1)
        return error.masked_select(valid).mean()

    def commit(
        self,
        state: AssociativeTTTState,
        observation: ObservationBatch,
        chunk_id: int,
        *,
        holdout: Optional[HoldoutBatch | ObservationBatch] = None,
        holdout_tolerance: float = 0.0,
        differentiable: bool = True,
    ) -> tuple[AssociativeTTTState, dict[str, Any]]:
        """Propose one clean-chunk update and atomically commit it.

        ``chunk_id`` must be exactly ``last_committed_chunk + 1`` for every
        batch row.  A holdout rejection returns a detached copy of the old
        state, keeps the cursor unchanged, and reports ``rolled_back=True``.
        """

        state.validate(self.config)
        if not isinstance(chunk_id, int) or chunk_id < 0:
            raise ValueError("chunk_id must be a non-negative integer")
        expected = state.last_committed_chunk + 1
        if not bool(torch.all(expected == chunk_id)):
            raise ValueError(f"expected chunk {expected.tolist()}, received {chunk_id}")
        obs = self._cast_observation(state, observation)
        assignments, replace, accepted, gamma, route_stats = self._route(state, obs)
        proposal = self._proposal(state, obs, assignments, replace, accepted, gamma, chunk_id)
        finite = all(
            torch.isfinite(x).all().item()
            for x in (proposal.keys, proposal.geometries, proposal.values, proposal.cross, proposal.precision)
        )
        before_loss = after_loss = None
        accepted_commit = finite
        if holdout is not None and finite:
            holdout_obs = holdout.as_observation() if isinstance(holdout, HoldoutBatch) else holdout
            holdout_obs = self._cast_observation(state, holdout_obs)
            before_loss = float(self._read_loss(state, holdout_obs).detach().cpu())
            after_loss = float(self._read_loss(proposal, holdout_obs).detach().cpu())
            accepted_commit = after_loss <= before_loss + float(holdout_tolerance)
        report: dict[str, Any] = {
            **route_stats,
            "chunk_id": chunk_id,
            "finite": bool(finite),
            "committed": bool(accepted_commit),
            "rolled_back": not bool(accepted_commit),
            "holdout_before": before_loss,
            "holdout_after": after_loss,
            "last_committed_chunk": chunk_id if accepted_commit else state.last_committed_chunk.detach().cpu().tolist(),
        }
        if not accepted_commit:
            return state.clone(detach=not differentiable), report
        if not differentiable:
            proposal = proposal.detach()
        return proposal, report

    @torch.no_grad()
    def commit_inference(
        self,
        state: AssociativeTTTState,
        observation: ObservationBatch,
        chunk_id: int,
        *,
        holdout: Optional[HoldoutBatch | ObservationBatch] = None,
        holdout_tolerance: float = 0.0,
    ) -> tuple[AssociativeTTTState, dict[str, Any]]:
        return self.commit(
            state,
            observation,
            chunk_id,
            holdout=holdout,
            holdout_tolerance=holdout_tolerance,
            differentiable=False,
        )

    def read(
        self,
        state: AssociativeTTTState,
        query_keys: torch.Tensor,
        query_geometry: Optional[torch.Tensor] = None,
        *,
        topk: Optional[int] = None,
        chunk_id: Optional[int] = None,
    ) -> ReadResult:
        """Content-addressed top-k read with explicit uncertainty fallback."""

        state.validate(self.config)
        dtype, device = state.precision.dtype, state.device
        q = query_keys.to(device=device, dtype=dtype)
        if q.ndim != 3 or q.shape[0] != state.batch_size or q.shape[-1] != self.config.key_dim:
            raise ValueError("query_keys must be [B,Q,key_dim]")
        q = F.normalize(q, dim=-1, eps=1e-8)
        b, nq, _ = q.shape
        if query_geometry is None:
            query_geometry = torch.zeros(b, nq, self.config.geometry_dim, device=device, dtype=dtype)
        else:
            query_geometry = query_geometry.to(device=device, dtype=dtype)
            if query_geometry.shape != (b, nq, self.config.geometry_dim):
                raise ValueError("query_geometry shape does not match query_keys")
        qg = F.normalize(query_geometry, dim=-1, eps=1e-8)
        keys = F.normalize(state.keys, dim=-1, eps=1e-8)
        geometries = F.normalize(state.geometries, dim=-1, eps=1e-8)
        appearance = torch.einsum("bqk,bck->bqc", q, keys) / self.config.temperature
        geom = torch.einsum("bqg,bcg->bqc", qg, geometries) / self.config.geometry_temperature
        distance = None
        if self.config.geometry_metric == 'ray_point':
            from .grail_geometry import geometry_distance
            distance = geometry_distance(query_geometry[:, :, None], state.geometries[:, None], self.config.geometry_scale)
            geom = -distance / self.config.geometry_temperature
        score = appearance + self.config.geometry_weight * geom
        score = score + torch.log(state.confidence.clamp_min(1e-8)).unsqueeze(1)
        if chunk_id is None:
            current_chunk = state.last_committed_chunk.max().detach()
        else:
            current_chunk = torch.tensor(chunk_id, device=device, dtype=state.age.dtype)
        age_delta = (current_chunk - state.age).clamp_min(0).to(dtype)
        score = score - self.config.age_penalty * torch.log1p(age_delta).unsqueeze(1)
        score = score.masked_fill(~state.valid.unsqueeze(1), -torch.inf)
        if distance is not None:
            score = score.masked_fill(distance > self.config.geometry_cutoff, -torch.inf)
        if topk is not None and (int(topk) < 1 or int(topk) > self.config.capacity):
            raise ValueError("topk must lie in [1, capacity]")
        k = min(int(topk or self.config.topk), self.config.capacity)
        top_scores, slot_ids = torch.topk(score, k=k, dim=-1)
        top_valid = torch.isfinite(top_scores)
        safe_scores = torch.where(top_valid, top_scores, torch.zeros_like(top_scores))
        weights = torch.softmax(safe_scores, dim=-1) * top_valid.to(dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        selected_p = state.precision.unsqueeze(1).expand(-1, nq, -1, -1, -1).gather(
            2, slot_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, nq, -1, self.config.key_dim, self.config.key_dim)
        )
        selected_c = state.cross.unsqueeze(1).expand(-1, nq, -1, -1, -1).gather(
            2, slot_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, nq, -1, self.config.value_dim, self.config.key_dim)
        )
        eye = torch.eye(self.config.key_dim, device=device, dtype=dtype)
        selected_p = torch.where(top_valid[..., None, None], selected_p, eye)
        selected_c = torch.where(top_valid[..., None, None], selected_c, torch.zeros_like(selected_c))
        selected_w = self._ridge_weights(selected_p, selected_c)
        # ``selected_w`` is [B,Q,K,V,Kd].  First apply every selected linear
        # ridge map to the query, then mix the resulting values by routing
        # weights.
        predictions = torch.einsum("bqd,bqkvd->bqkv", q, selected_w)
        # This gate is owned by the slow-clock router, not by the camera
        # branch.  It starts open (1.0), so the associative path receives
        # gradients from the first outer step instead of suffering a zero-gate
        # blind phase.
        raw_value = self.read_gate * torch.einsum("bqk,bqkv->bqv", weights, predictions)
        if k > 1:
            finite_second = top_valid[..., 1]
            second = torch.where(finite_second, top_scores[..., 1], top_scores[..., 0] - 1.0)
            margin = torch.where(top_valid[..., 0], top_scores[..., 0] - second, torch.zeros_like(top_scores[..., 0]))
        else:
            margin = torch.where(top_valid[..., 0], torch.full_like(top_scores[..., 0], float("inf")), torch.zeros_like(top_scores[..., 0]))
        selected_conf = state.confidence.unsqueeze(1).expand(-1, nq, -1).gather(2, slot_ids)
        precision_diag = state.precision.diagonal(dim1=-2, dim2=-1)
        selected_strength = precision_diag.unsqueeze(1).expand(-1, nq, -1, -1).gather(
            2, slot_ids.unsqueeze(-1).expand(-1, nq, -1, self.config.key_dim)
        ).mean(-1)
        uncertainty = (1.0 / (1.0 + selected_strength.clamp_min(0))).mul(weights).sum(-1)
        # Both terms lie in [0,1]. Keep their combined scale in [0,1] so
        # low-authority generated writes are not automatically unreadable.
        uncertainty = (uncertainty + 0.5 * (1.0 - (weights * selected_conf).sum(-1))) / 1.5
        has_memory = (
            top_valid[..., 0]
            & ((weights * selected_conf).sum(-1) >= self.config.min_read_confidence)
            & (margin >= self.config.min_retrieval_margin)
            & (uncertainty <= self.config.max_uncertainty)
        )
        value = torch.where(has_memory.unsqueeze(-1), raw_value, torch.zeros_like(raw_value))
        return ReadResult(value, slot_ids.masked_fill(~top_valid, -1), weights, top_scores, margin, uncertainty, has_memory, raw_value)

    def read_slots(self, state, query_keys, query_geometry, *, chunk_id=None, block_size=256):
        """Return individual top-k Ridge predictions for sparse cross-attention.

        Factor each slot once per forward, not once per query. Gathering the
        selected maps is bounded by block_size; no [all_tokens,slots,K,V] tensor.
        """
        from .grail_geometry import geometry_distance
        state.validate(self.config)
        q = F.normalize(query_keys.to(state.precision), dim=-1)
        geometry = query_geometry.to(state.precision)
        eye = torch.eye(self.config.key_dim, device=state.device, dtype=q.dtype)
        p = torch.where(state.valid[..., None, None], state.precision, eye)
        c = torch.where(state.valid[..., None, None], state.cross, torch.zeros_like(state.cross))
        maps = self._ridge_weights(p, c)
        batches = torch.arange(q.shape[0], device=q.device)[:, None, None]
        pieces = {name: [] for name in SlotRead.__dataclass_fields__}
        for start in range(0, q.shape[1], block_size):
            query, geo = q[:, start:start + block_size], geometry[:, start:start + block_size]
            distance = geometry_distance(geo[:, :, None], state.geometries[:, None], self.config.geometry_scale)
            score = torch.einsum('bnk,bsk->bns', query, F.normalize(state.keys, dim=-1)) / self.config.temperature
            score = score - self.config.geometry_weight * distance / self.config.geometry_temperature
            score = score + state.confidence.clamp_min(1e-8).log()[:, None]
            age = ((state.last_committed_chunk[:, None] + 1 if chunk_id is None else chunk_id) - state.age).clamp_min(0)
            score = score - self.config.age_penalty * age.float().log1p()[:, None]
            allowed = state.valid[:, None] & (distance <= self.config.geometry_cutoff)
            score = score.masked_fill(~allowed, -torch.inf)
            scores, ids = score.topk(self.config.topk, dim=-1)
            valid = torch.isfinite(scores)
            selected_maps = maps[batches, ids]
            predictions = torch.einsum('bnjvk,bnk->bnjv', selected_maps, query)
            conf = state.confidence[batches, ids] * valid
            strength = state.precision.diagonal(dim1=-2, dim2=-1).mean(-1)[batches, ids]
            uncertainty = (1 / (1 + strength.clamp_min(0))).mean(-1)
            margin = (scores[..., 0] - scores[..., 1]) if self.config.topk > 1 else torch.ones_like(scores[..., 0])
            margin = torch.nan_to_num(margin, nan=0., posinf=20., neginf=0.)
            valid = valid & (conf >= self.config.min_read_confidence)
            valid = valid & (margin[..., None] >= self.config.min_retrieval_margin)
            valid = valid & (uncertainty[..., None] <= self.config.max_uncertainty)
            result = SlotRead(ids.masked_fill(~valid, -1), state.keys[batches, ids],
                              state.geometries[batches, ids], predictions * valid[..., None],
                              scores, valid, conf, margin, uncertainty)
            for name in pieces:
                pieces[name].append(getattr(result, name))
        return SlotRead(**{name: torch.cat(parts, 1) for name, parts in pieces.items()})

    def association_loss(self, observation):
        """Geometry-matched leave-one-out Ridge, no future observations.

        Targets and loss weights are detached. Learned confidence only weights
        support writes, so reducing it cannot directly turn off the loss.
        Hard geometry membership has no gradient; continuous geometry weights do.
        """
        from .grail_geometry import geometry_distance
        obs = observation.normalized()
        k, v, g = obs.keys.float(), obs.values.float(), obs.geometry.float()
        n = k.shape[1]
        distance = geometry_distance(g[:, :, None], g[:, None], self.config.geometry_scale)
        eligible = (distance <= self.config.geometry_cutoff) & ~torch.eye(n, device=k.device, dtype=torch.bool)[None]
        # A real target is learned from real supports only, preserving authority.
        eligible = eligible & (~obs.source_real[:, :, None] | obs.source_real[:, None])
        source = torch.where(obs.source_real, 1., self.config.generated_write_scale)
        weights = eligible * (-distance).exp() * (obs.confidence * source)[:, None]
        p = torch.einsum('bjn,bnk,bnl->bjkl', weights, k, k)
        p = p + self.config.initial_precision * torch.eye(k.shape[-1], device=k.device)
        c = torch.einsum('bjn,bnv,bnk->bjvk', weights, v, k)
        prediction = torch.einsum('bjvk,bjk->bjv', self._ridge_weights(p, c), k)
        mask = (weights.detach().sum(-1) > 0) * source.detach()
        error = (prediction - v.detach()).square().mean(-1)
        return (error * mask).sum() / mask.sum().clamp_min(1)

    def leave_one_out_association_loss(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        *,
        source_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Differentiable support objective with self-retrieval removed.

        This is the direct inner TTT objective.  It uses the same full
        covariance ridge solve as the ledger, but deliberately operates on a
        single support set so it can be used before a slot assignment exists.
        """

        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError("keys and values must be [B,N,C]")
        b, n, k = keys.shape
        if k != self.config.key_dim or values.shape != (b, n, self.config.value_dim):
            raise ValueError("association tensor shapes do not match config")
        dtype, device = keys.dtype, keys.device
        kx = F.normalize(keys, dim=-1, eps=1e-8)
        if confidence is None:
            conf = torch.ones(b, n, device=device, dtype=dtype)
        else:
            conf = confidence[..., 0] if confidence.ndim == 3 else confidence
            conf = conf.to(device=device, dtype=dtype)
        if source_real is None:
            source_scale = torch.ones_like(conf)
        else:
            source_scale = torch.where(source_real.to(device=device, dtype=torch.bool), torch.ones_like(conf), torch.full_like(conf, self.config.generated_write_scale))
        weights = conf * source_scale
        eye = torch.eye(k, device=device, dtype=dtype)
        losses = []
        for j in range(n):
            mask = torch.ones(n, device=device, dtype=dtype)
            mask[j] = 0
            w = weights * mask.unsqueeze(0)
            p = self.config.initial_precision * eye.view(1, k, k) + torch.einsum("bn,bnk,bnl->bkl", w, kx, kx)
            c = torch.einsum("bn,bnv,bnk->bvk", w, values, kx)
            chol = torch.linalg.cholesky(p + self.config.solve_jitter * eye.view(1, k, k))
            pred_w = torch.cholesky_solve(c.transpose(-1, -2), chol).transpose(-1, -2)
            pred = torch.einsum("bvk,bk->bv", pred_w, kx[:, j])
            losses.append((pred - values[:, j]).square().mean(-1))
        loss = torch.stack(losses, dim=1)
        denom = weights.sum().clamp_min(1e-8)
        return (loss * weights).sum() / denom

    # Semantic aliases make integration code explicit without duplicating the
    # implementation under several subtly different names.
    functional_commit = commit
    update = commit


MemoryRouterV2 = AssociativeTTTLedger
ClosedFormTTTLedger = AssociativeTTTLedger


__all__ = [
    "SCHEMA_VERSION",
    "AssociativeTTTConfig",
    "ObservationBatch",
    "HoldoutBatch",
    "AssociativeTTTState",
    "ReadResult",
    "AssociativeTTTLedger",
    "MemoryRouterV2",
    "ClosedFormTTTLedger",
]
