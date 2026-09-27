"""GRAIL: a geometry-aware, transactional scene ledger for WorldTTT.

This module is deliberately independent of SANA and CUDA.  It is the reference
implementation of the *slow clock* proposed in ``docs/worldttt-grail.md``.  The
native cached GDN remains the fast clock; a caller may invoke :meth:`commit`
only once at a completed chunk boundary and use :meth:`read` during subsequent
denoising steps.  Keeping the ledger here makes mechanism tests possible
without loading a 2.6B parameter backbone.

The ledger stores a per-slot cross statistic C and diagonal ridge precision P
instead of repeatedly overwriting a single K/V prototype.  A slot is an
instance hypothesis, not a token cache: appearance and metric geometry jointly
address it, confidence controls whether it may be written, and an append-only
bounded event index records the source and chunk that created an entry.  All
arithmetic is FP32 and commits are copy-on-write, so malformed observations
leave the previous state untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GrailConfig:
    """Fixed-budget ledger and routing hyperparameters.

    ``capacity`` and ``edge_capacity`` are part of the experiment protocol: a
    larger ledger is a different baseline and must not be silently substituted.
    """

    appearance_dim: int = 128
    geometry_dim: int = 13
    value_dim: int = 128
    address_dim: int = 128
    capacity: int = 512
    edge_capacity: int = 1024
    event_capacity: int = 2048
    topk: int = 4
    temperature: float = 0.07
    geometry_temperature: float = 0.10
    decay: float = 1.0
    ridge_eps: float = 1e-3
    novelty_threshold: float = 0.30
    split_threshold: float = 1.25
    merge_threshold: float = 0.92
    generated_write_scale: float = 0.10
    min_write_confidence: float = 0.05
    protected_slots: int = 128
    seed: int = 3407

    def __post_init__(self):
        integer_fields = ("appearance_dim", "geometry_dim", "value_dim", "address_dim",
                          "capacity", "edge_capacity", "event_capacity", "topk")
        if any(getattr(self, name) < 1 for name in integer_fields):
            raise ValueError("GRAIL dimensions and capacities must be positive")
        if self.topk > self.capacity:
            raise ValueError("topk cannot exceed ledger capacity")
        if not (0 < self.temperature and 0 < self.geometry_temperature and 0 < self.decay <= 1):
            raise ValueError("temperature and decay must be in valid ranges")
        if self.ridge_eps <= 0 or self.split_threshold <= 0 or self.merge_threshold <= 0:
            raise ValueError("ridge_eps and routing thresholds must be positive")
        if not 0 <= self.novelty_threshold <= 1 or not 0 <= self.generated_write_scale <= 1:
            raise ValueError("novelty and generated write gates must lie in [0, 1]")
        if self.merge_threshold < self.novelty_threshold:
            raise ValueError("merge_threshold must be at least novelty_threshold")
        if self.min_write_confidence < 0 or not 0 <= self.protected_slots <= self.capacity:
            raise ValueError("invalid confidence or protected slot budget")


@dataclass
class GrailObservation:
    """Completed-chunk observations in a common world coordinate system.

    Every tensor is batched as ``[B, N, ...]``.  ``geometry`` is a canonical
    metric descriptor (for example world ray origin/direction and camera pose),
    not a pixel index.  ``source_real`` is boolean and is intentionally carried
    to the writer: generated prefixes can be used for adaptation but are never
    allowed the same write authority as measured observations.

    ``relations`` is optional ``[B, R, 8]`` with columns ``src, dst, dx, dy,
    dz, rx, ry, rz``.  The last six values are a compact relative SE(3)
    descriptor.  Pairs are resolved to ledger slot ids during commit.
    """

    appearance: torch.Tensor
    geometry: torch.Tensor
    value: torch.Tensor
    confidence: torch.Tensor
    source_real: torch.Tensor
    relations: Optional[torch.Tensor] = None

    def validate(self, config: GrailConfig):
        if self.appearance.ndim != 3 or self.geometry.ndim != 3 or self.value.ndim != 3:
            raise ValueError("appearance, geometry and value must be [B,N,C]")
        b, n = self.appearance.shape[:2]
        if (self.geometry.shape[:2] != (b, n) or self.value.shape[:2] != (b, n)
                or self.appearance.shape[-1] != config.appearance_dim
                or self.geometry.shape[-1] != config.geometry_dim
                or self.value.shape[-1] != config.value_dim):
            raise ValueError("GRAIL observation dimensions do not match config")
        if self.confidence.shape not in {(b, n), (b, n, 1)}:
            raise ValueError("confidence must be [B,N] or [B,N,1]")
        if self.source_real.shape not in {(b, n), (b, n, 1)}:
            raise ValueError("source_real must be [B,N] or [B,N,1]")
        if self.relations is not None and (self.relations.ndim != 3 or self.relations.shape[0] != b
                                            or self.relations.shape[-1] != 8):
            raise ValueError("relations must be [B,R,8]")
        tensors = (self.appearance, self.geometry, self.value, self.confidence, self.source_real)
        if self.relations is not None:
            tensors += (self.relations,)
        if not all(torch.isfinite(x).all() for x in tensors):
            raise FloatingPointError("nonfinite GRAIL observation")
        return self


@dataclass
class GrailState:
    """Per-episode persistent state.  Batch rows are isolated CFG branches."""

    episode: str
    keys: torch.Tensor                 # [B,C,D], normalized joint address
    values: torch.Tensor               # [B,C,V], prototype/debug value
    cross: torch.Tensor                # [B,C,V,D], ridge cross statistic C
    precision: torch.Tensor            # [B,C,D], diagonal sufficient statistic
    geometry: torch.Tensor             # [B,C,G], EMA prototype
    confidence: torch.Tensor           # [B,C]
    age: torch.Tensor                  # [B,C], last chunk id
    valid: torch.Tensor                # [B,C]
    protected: torch.Tensor            # [B,C]
    slot_generation: torch.Tensor      # [B,C], increments on replacement
    seen: torch.Tensor                 # [B]
    edge_src: torch.Tensor             # [B,E]
    edge_dst: torch.Tensor             # [B,E]
    edge_value: torch.Tensor           # [B,E,6]
    edge_precision: torch.Tensor       # [B,E]
    edge_age: torch.Tensor             # [B,E]
    edge_valid: torch.Tensor           # [B,E]
    event_slot: torch.Tensor           # [B,L]
    event_slot_generation: torch.Tensor # [B,L]
    event_chunk: torch.Tensor          # [B,L]
    event_source_real: torch.Tensor    # [B,L]
    event_confidence: torch.Tensor     # [B,L]
    event_seen: torch.Tensor           # [B]
    last_chunk: int = -1
    updates: int = 0

    @classmethod
    def new(cls, config: GrailConfig, episode: str, batch: int, device="cpu"):
        if batch < 1:
            raise ValueError("batch must be positive")
        f = lambda *shape: torch.zeros(*shape, device=device, dtype=torch.float32)
        return cls(str(episode), f(batch, config.capacity, config.address_dim),
                   f(batch, config.capacity, config.value_dim),
                   f(batch, config.capacity, config.value_dim, config.address_dim),
                   f(batch, config.capacity, config.address_dim),
                   f(batch, config.capacity, config.geometry_dim),
                   f(batch, config.capacity), torch.full((batch, config.capacity), -1,
                       device=device, dtype=torch.long),
                   torch.zeros(batch, config.capacity, device=device, dtype=torch.bool),
                   torch.zeros(batch, config.capacity, device=device, dtype=torch.bool),
                   torch.zeros(batch, config.capacity, device=device, dtype=torch.long),
                   torch.zeros(batch, device=device, dtype=torch.long),
                   torch.zeros(batch, config.edge_capacity, device=device, dtype=torch.long),
                   torch.zeros(batch, config.edge_capacity, device=device, dtype=torch.long),
                   f(batch, config.edge_capacity, 6), f(batch, config.edge_capacity),
                   torch.full((batch, config.edge_capacity), -1, device=device, dtype=torch.long),
                   torch.zeros(batch, config.edge_capacity, device=device, dtype=torch.bool),
                   torch.full((batch, config.event_capacity), -1, device=device, dtype=torch.long),
                   torch.zeros(batch, config.event_capacity, device=device, dtype=torch.long),
                   torch.full((batch, config.event_capacity), -1, device=device, dtype=torch.long),
                   torch.zeros(batch, config.event_capacity, device=device, dtype=torch.bool),
                   f(batch, config.event_capacity),
                   torch.zeros(batch, device=device, dtype=torch.long))

    def clone(self):
        fields = {name: value.clone() if isinstance(value, torch.Tensor) else value
                  for name, value in vars(self).items()}
        return type(self)(**fields)

    def state_dict(self):
        return {"version": 1, **{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                  for k, v in vars(self).items()}}

    def save(self, path: str | Path):
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.pop("version", None) != 1:
            raise ValueError("Unsupported GRAIL state version")
        return cls(**{k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in payload.items()})


@dataclass
class GrailRead:
    value: torch.Tensor
    slot_ids: torch.Tensor
    weights: torch.Tensor
    margin: torch.Tensor
    uncertainty: torch.Tensor
    has_memory: torch.Tensor


def canonical_ray_geometry(c2w: torch.Tensor, intrinsics: torch.Tensor,
                           uv: torch.Tensor, depth: torch.Tensor | None = None,
                           anchor_c2w: torch.Tensor | None = None) -> torch.Tensor:
    """Build the 13-D canonical geometry descriptor used by GRAIL.

    ``c2w`` is ``[B,4,4]`` and ``intrinsics`` is ``[B,3,3]`` for one token
    batch; ``uv`` is ``[B,N,2]`` in the *post-resize/crop pixel coordinates*.
    The world ray is ``o=p_c2w`` and ``d=R_c2w K^{-1}[u,v,1]``.  If an
    ``anchor_c2w`` is supplied, origins and directions are expressed in that
    episode gauge.  The final six entries are the anchor-relative translation
    and a small-angle rotation vector.  Depth may be omitted (its log entry is
    zero), which keeps addressing valid for monocular observations while making
    the uncertainty gate responsible for low-confidence geometry.
    """
    if c2w.shape != (c2w.shape[0], 4, 4) or intrinsics.shape != (intrinsics.shape[0], 3, 3):
        raise ValueError("c2w/intrinsics must be [B,4,4]/[B,3,3]")
    if uv.ndim != 3 or uv.shape[0] != c2w.shape[0] or uv.shape[-1] != 2:
        raise ValueError("uv must be [B,N,2]")
    if depth is not None and depth.shape != uv.shape[:2]:
        raise ValueError("depth must be [B,N]")
    if not all(torch.isfinite(x).all() for x in (c2w, intrinsics, uv)
               + (() if depth is None else (depth,))):
        raise FloatingPointError("nonfinite camera geometry")
    b, n = uv.shape[:2]
    pixel = torch.cat((uv.float(), torch.ones(b, n, 1, device=uv.device)), -1)
    camera_dir = torch.linalg.solve(intrinsics.float(), pixel.transpose(1, 2)).transpose(1, 2)
    rotation = c2w[:, :3, :3].float()
    direction = F.normalize(torch.einsum("bij,bnj->bni", rotation, camera_dir), dim=-1)
    origin = c2w[:, None, :3, 3].float().expand(-1, n, -1)
    relative_pose = uv.new_zeros(b, n, 6, dtype=torch.float32)
    if anchor_c2w is not None:
        if anchor_c2w.shape != c2w.shape:
            raise ValueError("anchor_c2w shape mismatch")
        anchor_rotation = anchor_c2w[:, :3, :3].float()
        anchor_origin = anchor_c2w[:, :3, 3].float()
        origin = torch.einsum("bij,bnj->bni", anchor_rotation.transpose(1, 2),
                              origin - anchor_origin[:, None])
        direction = torch.einsum("bij,bnj->bni", anchor_rotation.transpose(1, 2),
                                 direction)
        relative_pose[..., :3] = origin
        relative_rotation = torch.einsum("bij,bjk->bik", anchor_rotation.transpose(1, 2), rotation)
        cos_angle = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * .5).clamp(-1, 1)
        angle = torch.acos(cos_angle)
        axis = torch.stack((relative_rotation[:, 2, 1] - relative_rotation[:, 1, 2],
                            relative_rotation[:, 0, 2] - relative_rotation[:, 2, 0],
                            relative_rotation[:, 1, 0] - relative_rotation[:, 0, 1]), -1)
        axis = F.normalize(axis, dim=-1, eps=1e-6)
        relative_pose[..., 3:] = (axis * angle[..., None])[:, None]
    log_depth = torch.zeros(b, n, 1, device=uv.device, dtype=torch.float32)
    if depth is not None:
        log_depth[..., 0] = depth.float().clamp_min(1e-4).log()
    # [origin, direction, log depth, pose translation/rotation].  Origin is
    # repeated in relative_pose for the no-anchor gauge; the representation is
    # intentionally redundant so a learned projection can trade off cues.
    return torch.cat((origin, direction, log_depth, relative_pose), -1)


class GrailMemory(nn.Module):
    """Trainable address/value projections plus a non-trainable episode ledger.

    The projections are outer-loop parameters.  The ledger tensors are state,
    not module parameters, and are updated only by :meth:`commit`.  This makes
    it possible to meta-train the address while preserving an auditable write
    protocol at inference time.
    """

    def __init__(self, config: GrailConfig):
        super().__init__()
        self.config = config
        d = config.address_dim
        self.appearance_write = nn.Linear(config.appearance_dim, d // 2, bias=False)
        self.geometry_write = nn.Linear(config.geometry_dim, d - d // 2, bias=False)
        # Read and write addresses share the same coordinate system.  A separate
        # projection here would make dot products meaningless until a special
        # cross-space objective had been trained.
        self.appearance_read = self.appearance_write
        self.geometry_read = self.geometry_write
        self.value_in = nn.Linear(config.value_dim, config.value_dim, bias=False)
        self.value_out = nn.Linear(config.value_dim, config.value_dim, bias=False)
        self.read_gate = nn.Parameter(torch.zeros(()))
        # Identity plus a zero scalar gate keeps the backbone exactly unchanged
        # before the first outer-loop update, while making retrieved values
        # inspectable in mechanism probes.
        nn.init.eye_(self.value_out.weight)

    @staticmethod
    def _norm(x):
        return F.normalize(x.float(), dim=-1, eps=1e-6)

    def _write_key(self, appearance, geometry):
        return self._norm(torch.cat((self.appearance_write(appearance.float()),
                                     self.geometry_write(geometry.float())), -1))

    def _read_key(self, appearance, geometry):
        return self._norm(torch.cat((self.appearance_read(appearance.float()),
                                     self.geometry_read(geometry.float())), -1))

    def encode(self, observation: GrailObservation):
        observation.validate(self.config)
        return (self._write_key(observation.appearance, observation.geometry),
                self.value_in(observation.value.float()))

    @staticmethod
    def _slot_scores(state, query, geometry, config):
        key = state.keys
        score = torch.einsum("bnd,bcd->bnc", query.float(), key.float()) / config.temperature
        g = F.normalize(geometry.float(), dim=-1, eps=1e-6)
        sg = F.normalize(state.geometry.float(), dim=-1, eps=1e-6)
        score = score + torch.einsum("bng,bcg->bnc", g, sg) / config.geometry_temperature
        score = score + state.confidence.clamp_min(1e-4).log()[:, None] - .01 * torch.log1p(
            state.age.clamp_min(0).float())[:, None]
        return score.masked_fill(~state.valid[:, None], -torch.inf)

    @staticmethod
    def _slot_affinity(state, query, geometry):
        """Unit-scale appearance/geometry affinity used by the novelty gate."""
        d = query.shape[-1] // 2
        app = F.normalize(query[..., :d].float(), dim=-1, eps=1e-6)
        kapp = F.normalize(state.keys[..., :d].float(), dim=-1, eps=1e-6)
        geo = F.normalize(query[..., d:].float(), dim=-1, eps=1e-6)
        kgeo = F.normalize(state.keys[..., d:].float(), dim=-1, eps=1e-6)
        return .5 * (torch.einsum("bnd,bcd->bnc", app, kapp)
                     + torch.einsum("bnd,bcd->bnc", geo, kgeo))

    def read(self, state: GrailState, appearance, geometry, *, topk=None,
             differentiable=False):
        """Read with geometry-gated top-k routing; no state mutation occurs."""
        context = torch.enable_grad() if differentiable else torch.no_grad()
        with context:
            if appearance.ndim != 3 or geometry.shape[:2] != appearance.shape[:2]:
                raise ValueError("GRAIL read shape mismatch")
            if appearance.shape[0] != state.keys.shape[0]:
                raise ValueError("episode/CFG batch mismatch")
            query = self._read_key(appearance, geometry)
            k = min(topk or self.config.topk, state.keys.shape[1])
            valid_episode = state.valid.any(-1)
            if not bool(valid_episode.any()):
                shape = (appearance.shape[0], appearance.shape[1], k)
                ids = torch.zeros(shape, dtype=torch.long, device=appearance.device)
                weights = torch.full(shape, 1.0 / k, device=appearance.device)
                zeros = appearance.new_zeros(appearance.shape[0], appearance.shape[1])
                uncertainty = appearance.new_ones(appearance.shape[0], appearance.shape[1])
                return GrailRead(appearance.new_zeros(appearance.shape[0], appearance.shape[1],
                                                         self.config.value_dim), ids, weights,
                                 zeros, uncertainty, appearance.new_zeros(appearance.shape[0],
                                                                           appearance.shape[1], 1))
            score = self._slot_scores(state, query, geometry, self.config)
            top_score, ids = score.topk(k, -1)
            # A CFG branch may legitimately have no committed slots while a
            # sibling branch does.  softmax([-inf,...]) is NaN, so provide a
            # finite neutral distribution before applying has_memory masking.
            weights = torch.nan_to_num(top_score.softmax(-1), nan=1.0 / k)
            cross = torch.gather(state.cross[:, None].expand(-1, query.shape[1], -1, -1, -1),
                                 2, ids[..., None, None].expand(-1, -1, -1,
                                                                  state.cross.shape[-2],
                                                                  state.cross.shape[-1]))
            precision = torch.gather(state.precision[:, None].expand(-1, query.shape[1], -1, -1),
                                     2, ids[..., None].expand(-1, -1, -1, state.precision.shape[-1]))
            operator = cross / precision[..., None, :].clamp_min(self.config.ridge_eps)
            retrieved = torch.einsum("bnkvd,bnd->bnkv", operator, query.float())
            read_value = (weights[..., None] * retrieved).sum(2)
            read_value = read_value * valid_episode[:, None, None]
            best = top_score[..., 0]
            second = top_score[..., 1] if k > 1 else best
            margin = torch.nan_to_num(best - second, nan=0.0, neginf=0.0, posinf=0.0)
            uncertainty = (1.0 / state.precision.clamp_min(self.config.ridge_eps)).mean(-1)
            uncertainty = torch.gather(uncertainty[:, None].expand(-1, query.shape[1], -1),
                                       2, ids[..., :1]).squeeze(-1)
            projected = torch.tanh(self.read_gate.float()) * self.value_out(read_value)
            return GrailRead(projected, ids, weights, margin, uncertainty,
                             valid_episode[:, None, None].expand(-1, query.shape[1], 1).to(read_value.dtype))

    def _choose_slot(self, state, branch, score, affinity, confidence, chunk):
        valid = state.valid[branch]
        if not valid.any():
            return 0
        best = int(score.argmax())
        if float(affinity[best]) >= self.config.novelty_threshold:
            return best
        free = torch.where(~valid)[0]
        if len(free):
            return int(free[0])
        candidates = torch.where(~state.protected[branch] & valid)[0]
        if not len(candidates):
            return best  # protected full ledger: merge into best, never erase history
        # Low-confidence old slots are evicted first.  This is deterministic and
        # therefore reproducible across distributed ranks.
        utility = state.confidence[branch, candidates] - .01 * (chunk - state.age[branch, candidates]).clamp_min(0)
        return int(candidates[utility.argmin()])

    @torch.no_grad()
    def commit(self, state: GrailState, observation: GrailObservation, chunk: int):
        """Atomically write a completed chunk and return ``(new_state, report)``.

        A real observation has unit write authority; generated observations are
        capped by ``generated_write_scale``.  A new slot is allocated when the
        joint appearance/geometry address is novel.  Existing slots use a
        diagonal ridge/Kalman update, which bounds the influence of a single
        noisy observation and preserves an uncertainty estimate.
        """
        if chunk != state.last_chunk + 1:
            raise ValueError(f"expected chunk {state.last_chunk + 1}, received {chunk}")
        observation.validate(self.config)
        keys, values = self.encode(observation)
        proposed = state.clone()
        b, n = keys.shape[:2]
        confidence = observation.confidence.float().reshape(b, n).clamp(0, 1)
        source_real = observation.source_real.bool().reshape(b, n)
        geometry = observation.geometry.float()
        assignments = torch.full((b, n), -1, dtype=torch.long, device=keys.device)
        created = merged = rejected = 0
        for branch in range(b):
            for token in range(n):
                gate = float(confidence[branch, token])
                if not bool(source_real[branch, token]):
                    gate *= self.config.generated_write_scale
                if gate < self.config.min_write_confidence:
                    rejected += 1
                    continue
                query_key = keys[branch:branch + 1, token:token + 1]
                query_geometry = geometry[branch:branch + 1, token:token + 1]
                score = self._slot_scores(proposed, query_key, query_geometry, self.config)[0, 0]
                affinity = self._slot_affinity(proposed, query_key, query_geometry)[0, 0]
                slot = self._choose_slot(proposed, branch, score, affinity, gate, chunk)
                is_new = not bool(proposed.valid[branch, slot])
                if not is_new:
                    observed_geometry = F.normalize(query_geometry[0, 0].float(), dim=-1, eps=1e-6)
                    stored_geometry = F.normalize(proposed.geometry[branch, slot].float(), dim=-1, eps=1e-6)
                    geometry_distance = float((observed_geometry - stored_geometry).norm())
                    # Between novelty and merge thresholds is an ambiguous
                    # address: allocate a fresh slot when possible.  A
                    # protected slot is never overwritten, so ambiguity safely
                    # degrades to a conservative merge.
                    is_new = (float(affinity[slot]) < self.config.merge_threshold
                              or geometry_distance > self.config.split_threshold)
                    is_new = is_new and not bool(proposed.protected[branch, slot])
                if is_new:
                    was_valid = bool(proposed.valid[branch, slot])
                    if was_valid or int(proposed.slot_generation[branch, slot]) == 0:
                        proposed.slot_generation[branch, slot] += 1
                    # A replacement invalidates relations that point to the
                    # old incarnation; event records retain the old generation.
                    if was_valid:
                        proposed.edge_valid[branch] &= (
                            (proposed.edge_src[branch] != slot)
                            & (proposed.edge_dst[branch] != slot))
                    proposed.keys[branch, slot] = keys[branch, token]
                    proposed.values[branch, slot] = values[branch, token]
                    proposed.cross[branch, slot] = torch.outer(values[branch, token], keys[branch, token])
                    proposed.precision[branch, slot] = keys[branch, token].square() + self.config.ridge_eps
                    proposed.geometry[branch, slot] = geometry[branch, token]
                    proposed.confidence[branch, slot] = gate
                    proposed.age[branch, slot] = chunk
                    proposed.valid[branch, slot] = True
                    if chunk == 0 and int(proposed.protected[branch].sum()) < self.config.protected_slots:
                        proposed.protected[branch, slot] = True
                    created += 1
                else:
                    old_precision = proposed.precision[branch, slot].clone()
                    new_precision = self.config.decay * old_precision + gate * keys[branch, token].square()
                    old_numerator = old_precision * proposed.values[branch, slot]
                    numerator = self.config.decay * old_numerator + gate * values[branch, token] * keys[branch, token]
                    proposed.cross[branch, slot] = (self.config.decay * proposed.cross[branch, slot]
                                                    + gate * torch.outer(values[branch, token], keys[branch, token]))
                    proposed.precision[branch, slot] = new_precision.clamp_min(self.config.ridge_eps)
                    proposed.values[branch, slot] = numerator / proposed.precision[branch, slot]
                    total = self.config.decay * proposed.confidence[branch, slot] + gate
                    alpha = gate / total.clamp_min(self.config.ridge_eps)
                    proposed.geometry[branch, slot] = ((1 - alpha) * proposed.geometry[branch, slot]
                                                        + alpha * geometry[branch, token])
                    proposed.confidence[branch, slot] = total.clamp_max(1.)
                    proposed.age[branch, slot] = chunk
                    merged += 1
                assignments[branch, token] = slot
                # Event index is a bounded append-only provenance ring.  The
                # slot bank may evict; the event still tells us what happened.
                event = int(proposed.event_seen[branch] % proposed.event_slot.shape[1])
                proposed.event_slot[branch, event] = slot
                proposed.event_slot_generation[branch, event] = proposed.slot_generation[branch, slot]
                proposed.event_chunk[branch, event] = chunk
                proposed.event_source_real[branch, event] = bool(source_real[branch, token])
                proposed.event_confidence[branch, event] = gate
                proposed.event_seen[branch] += 1
        self._commit_relations(proposed, observation.relations, assignments, chunk)
        proposed.seen += n
        proposed.last_chunk, proposed.updates = chunk, state.updates + 1
        tensors = (proposed.keys, proposed.values, proposed.cross, proposed.precision, proposed.geometry,
                   proposed.confidence, proposed.edge_value)
        if not all(torch.isfinite(x).all() for x in tensors):
            # Copy-on-write means the caller can safely retain ``state``.
            rejected_state = state.clone()
            rejected_state.last_chunk = chunk
            return rejected_state, dict(chunk=chunk, committed=False, reason="nonfinite_proposal",
                               created=0, merged=0, rejected=n)
        report = dict(chunk=chunk, committed=True, created=created, merged=merged,
                      rejected=rejected, active=int(proposed.valid.sum()),
                      events=int(proposed.event_seen.sum()),
                      state_bytes=self.state_bytes(proposed))
        return proposed, report

    def _commit_relations(self, state, relations, assignments, chunk):
        if relations is None:
            return
        for branch in range(relations.shape[0]):
            for row in relations[branch]:
                src, dst = int(row[0]), int(row[1])
                if src < 0 or dst < 0 or src >= assignments.shape[1] or dst >= assignments.shape[1]:
                    continue
                a, b = int(assignments[branch, src]), int(assignments[branch, dst])
                if a < 0 or b < 0 or not bool(state.valid[branch, a]) or not bool(state.valid[branch, b]):
                    continue
                existing = torch.where(state.edge_valid[branch] & (state.edge_src[branch] == a)
                                       & (state.edge_dst[branch] == b))[0]
                if len(existing):
                    edge = int(existing[0])
                    p = state.edge_precision[branch, edge]
                    state.edge_value[branch, edge] = (p * state.edge_value[branch, edge] + row[2:].float()) / (p + 1)
                    state.edge_precision[branch, edge] = p + 1
                    state.edge_age[branch, edge] = chunk
                else:
                    free = torch.where(~state.edge_valid[branch])[0]
                    if not len(free):
                        free = torch.tensor([int(state.edge_age[branch].argmin())], device=state.edge_age.device)
                    edge = int(free[0])
                    state.edge_src[branch, edge], state.edge_dst[branch, edge] = a, b
                    state.edge_value[branch, edge] = row[2:].float()
                    state.edge_precision[branch, edge] = 1
                    state.edge_age[branch, edge] = chunk
                    state.edge_valid[branch, edge] = True

    @staticmethod
    def state_bytes(state: GrailState):
        return int(sum(x.numel() * x.element_size() for x in vars(state).values()
                       if isinstance(x, torch.Tensor)))


class GrailController:
    """Small bridge used by a SANA sampler without importing the backbone.

    A future cached-GDN hook should hold one controller per episode/CFG batch,
    call ``read`` from every denoising step, and call ``commit`` exactly once
    after the clean chunk pass.  The explicit methods make it difficult to
    accidentally write during a denoising timestep.
    """

    def __init__(self, memory: GrailMemory):
        self.memory = memory
        self.state: GrailState | None = None
        self.metrics: list[dict] = []

    def reset_episode(self, episode: str, batch: int, device="cpu"):
        self.state = GrailState.new(self.memory.config, episode, batch, device)
        self.metrics = []
        return self.state

    def _require_state(self):
        if self.state is None:
            raise RuntimeError("reset_episode must be called before GRAIL read/commit")
        return self.state

    def read(self, appearance, geometry, *, topk=None, differentiable=False):
        return self.memory.read(self._require_state(), appearance, geometry,
                                topk=topk, differentiable=differentiable)

    def commit(self, observation: GrailObservation, chunk: int):
        state = self._require_state()
        state, report = self.memory.commit(state, observation, chunk)
        # Even a rejected proposal consumes the chunk cursor; the old ledger
        # remains intact and the next completed chunk can be attempted.
        self.state = state
        self.metrics.append(report)
        return report

    def save_state(self, path: str | Path):
        self._require_state().save(path)

    def load_state(self, path: str | Path, device="cpu"):
        self.state = GrailState.load(path, device)
        self.metrics = []
        return self.state
