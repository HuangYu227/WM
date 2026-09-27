"""Bounded explicit binding bank and hybrid episode state."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from worldttt.sap_ttt.memory import SapMemoryState


@dataclass
class BindingBankState:
    keys: torch.Tensor
    values: torch.Tensor
    rays: torch.Tensor
    confidence: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor
    seen: torch.Tensor
    protected: torch.Tensor

    @classmethod
    def new(cls, batch, capacity, key_dim, value_dim, ray_dim, device):
        if min(batch, capacity, key_dim, value_dim, ray_dim) < 1:
            raise ValueError('Invalid bank dimensions')
        z = lambda *shape: torch.zeros(*shape, device=device, dtype=torch.float32)
        return cls(z(batch, capacity, key_dim), z(batch, capacity, value_dim),
                   z(batch, capacity, ray_dim), z(batch, capacity), z(batch, capacity),
                   torch.zeros(batch, capacity, device=device, dtype=torch.bool),
                   torch.zeros(batch, device=device, dtype=torch.long),
                   torch.zeros(batch, capacity, device=device, dtype=torch.bool))

    def clone(self):
        return BindingBankState(*(x.clone() for x in (self.keys, self.values, self.rays,
            self.confidence, self.age, self.valid, self.seen, self.protected)))

    def state_dict(self):
        return {name: getattr(self, name).detach().cpu() for name in self.__dataclass_fields__}

    @classmethod
    def from_state_dict(cls, payload, device, *, capacity=None, key_dim=None,
                        value_dim=None, ray_dim=None):
        required = set(cls.__dataclass_fields__)
        if set(payload) != required:
            raise ValueError('Binding bank state fields mismatch')
        state = cls(*(payload[name].to(device) for name in cls.__dataclass_fields__))
        if state.keys.ndim != 3:
            raise ValueError('Binding bank keys must be batched vectors')
        batch, slots, keys = state.keys.shape
        if (state.values.shape[:2] != (batch, slots) or state.rays.shape[:2] != (batch, slots) or
            any(x.shape != (batch, slots) for x in
                (state.confidence, state.age, state.valid, state.protected)) or
            state.seen.shape != (batch,) or state.valid.dtype != torch.bool or
            state.protected.dtype != torch.bool or state.seen.dtype != torch.long or
            (state.protected & ~state.valid).any()):
            raise ValueError('Malformed binding bank state')
        if ((capacity is not None and slots != capacity) or
            (key_dim is not None and keys != key_dim) or
            (value_dim is not None and state.values.shape[-1] != value_dim) or
            (ray_dim is not None and state.rays.shape[-1] != ray_dim)):
            raise ValueError('Binding bank state dimensions mismatch')
        if not all(torch.isfinite(x).all() for x in (state.keys, state.values, state.rays,
                                                      state.confidence, state.age)):
            raise ValueError('Nonfinite binding bank state')
        return state

    @torch.no_grad()
    def commit(self, keys, values, rays, confidence, *, chunk, protected_budget, seed):
        if keys.ndim != 3 or values.shape[:2] != keys.shape[:2] or rays.shape[:2] != keys.shape[:2]:
            raise ValueError('Binding bank write shape mismatch')
        if (keys.shape[0] != self.keys.shape[0] or keys.shape[-1] != self.keys.shape[-1] or
                values.shape[-1] != self.values.shape[-1] or rays.shape[-1] != self.rays.shape[-1]):
            raise ValueError('Binding bank write dimension mismatch')
        if confidence.shape != keys.shape[:2] or not all(torch.isfinite(x).all() for x in (keys, values, rays, confidence)):
            raise FloatingPointError('Nonfinite or malformed binding write')
        capacity = self.keys.shape[1]
        for branch in range(keys.shape[0]):
            count = keys.shape[1]
            free = torch.where(~self.valid[branch])[0]
            fill = min(count, len(free))
            if fill:
                slots = free[:fill]
                self.keys[branch, slots] = keys[branch, :fill]
                self.values[branch, slots] = values[branch, :fill]
                self.rays[branch, slots] = rays[branch, :fill]
                self.confidence[branch, slots] = confidence[branch, :fill]
                self.age[branch, slots] = chunk
                self.valid[branch, slots] = True
                if chunk == 0:
                    remaining = max(0, min(protected_budget, capacity) - int(self.protected[branch].sum()))
                    self.protected[branch, slots[:remaining]] = True
            overflow = count - fill
            if overflow:
                candidates = torch.where(~self.protected[branch] & self.valid[branch])[0]
                if len(candidates):
                    device = keys.device
                    generator = torch.Generator(device=device).manual_seed(
                        seed + 104729 * branch + int(self.seen[branch]))
                    sequence = self.seen[branch].float() + torch.arange(
                        fill + 1, count + 1, device=device, dtype=torch.float32)
                    incoming = confidence[branch, fill:].clamp_min(.05)
                    probability = (len(candidates) / sequence).clamp(max=1) * (
                        incoming / incoming.mean().clamp_min(.05)).clamp(max=2)
                    accepted = torch.where(torch.rand(overflow, device=device, generator=generator) < probability)[0]
                    if len(accepted):
                        weights = 1 / self.confidence[branch, candidates].clamp_min(.05)
                        chosen = candidates[torch.multinomial(weights, len(accepted), replacement=True,
                                                              generator=generator)]
                        source = accepted + fill
                        self.keys[branch, chosen] = keys[branch, source]
                        self.values[branch, chosen] = values[branch, source]
                        self.rays[branch, chosen] = rays[branch, source]
                        self.confidence[branch, chosen] = confidence[branch, source]
                        self.age[branch, chosen] = chunk
            self.seen[branch] += count


@dataclass
class BindingState:
    episode: str
    bank: BindingBankState
    fast: SapMemoryState
    last_chunk: int = -1
    updates: int = 0

    def state_dict(self):
        return dict(version=1, episode=self.episode, bank=self.bank.state_dict(),
                    fast=self.fast.state_dict(), last_chunk=self.last_chunk, updates=self.updates)


def binding_scores(state: BindingBankState, query, query_rays, ray_query, ray_key,
                   *, heads, shuffle_key=False):
    if query.ndim != 3 or query_rays.shape[:2] != query.shape[:2]:
        raise ValueError('Binding read shape mismatch')
    b, n, dim = query.shape
    if b != state.keys.shape[0] or dim % heads or state.values.shape[-1] % heads:
        raise ValueError('Binding read branch/head mismatch')
    keys = state.keys
    if shuffle_key:
        keys = keys.roll(1, dims=1)
    hd = dim // heads
    q = F.rms_norm(query.float().reshape(b, n, heads, hd), (hd,)).permute(0, 2, 1, 3)
    k = F.rms_norm(keys.float().reshape(b, -1, heads, hd), (hd,)).permute(0, 2, 1, 3)
    score = torch.einsum('bhnd,bhcd->bhnc', q, k) / hd ** .5
    rq = ray_query(query_rays.float()).reshape(b, n, heads, -1).permute(0, 2, 1, 3)
    rk = ray_key(state.rays.float()).reshape(b, -1, heads, rq.shape[-1]).permute(0, 2, 1, 3)
    score = score + torch.einsum('bhnd,bhcd->bhnc', rq, rk) / rq.shape[-1] ** .5
    score = score + .1 * state.confidence.clamp_min(1e-4).log()[:, None, None]
    return score.masked_fill(~state.valid[:, None, None], -torch.inf)


def explicit_read(state: BindingBankState, query, query_rays, ray_query, ray_key,
                  *, heads, topk, shuffle_key=False, shuffle_value=False):
    active = state.valid.any(1)
    if not active.any():
        b, n = query.shape[:2]
        zeros = query.new_zeros(b, n, state.values.shape[-1], dtype=torch.float32)
        stats = dict(margin=query.new_zeros(b, n, heads), entropy=query.new_ones(b, n, heads),
                     age=query.new_zeros(b, n, heads), has_memory=query.new_zeros(b, n, 1))
        return zeros, stats
    score = binding_scores(state, query, query_rays, ray_query, ray_key,
                           heads=heads, shuffle_key=shuffle_key)
    if not active.all():
        score = score.clone()
        score[~active, :, :, 0] = 0
    b, _, n, _ = score.shape
    values = state.values.roll(1, dims=1) if shuffle_value else state.values
    vd = state.values.shape[-1] // heads
    count = min(topk, score.shape[-1])
    top_score, top_id = score.topk(count, -1)
    weights = top_score.softmax(-1)
    vh = values.float().reshape(b, -1, heads, vd).permute(0, 2, 1, 3)
    selected = torch.gather(vh[:, :, None].expand(-1, -1, n, -1, -1), 3,
                            top_id[..., None].expand(-1, -1, -1, -1, vd))
    out = (weights[..., None] * selected).sum(3).permute(0, 2, 1, 3).reshape(b, n, -1)
    probability = score.softmax(-1)
    entropy = -(probability * probability.clamp_min(1e-9).log()).sum(-1)
    valid_count = state.valid.sum(-1).clamp_min(2).log()[:, None, None]
    entropy = entropy / valid_count
    second = top_score[..., 1] if count > 1 else torch.zeros_like(top_score[..., 0])
    margin = top_score[..., 0] - torch.where(torch.isfinite(second), second,
                                             top_score[..., 0])
    top_age = torch.gather(state.age[:, None, None].expand(-1, heads, n, -1), 3, top_id[..., :1]).squeeze(-1)
    out = out * active[:, None, None]
    return out, dict(margin=margin.permute(0, 2, 1), entropy=entropy.permute(0, 2, 1),
                     age=top_age.permute(0, 2, 1),
                     has_memory=active[:, None, None].expand(b, n, 1).to(query.dtype))
