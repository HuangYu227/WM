"""Functional per-head SwiGLU memory; state is one packed FP32 tensor per branch.

Reuses the SwiGLU fast-weight design from worldttt.memory with a K/V interface.
Packing preserves the existing transactional state/checkpoint machinery.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUMemory(nn.Module):
    def __init__(self, dim=512, heads=8, hidden_dim=128, lr=.1, max_grad_norm=256.):
        super().__init__()
        if min(dim, heads, hidden_dim, lr, max_grad_norm) <= 0 or dim % heads:
            raise ValueError('Invalid SwiGLU memory dimensions or step settings')
        self.dim, self.heads, self.hidden_dim = dim, heads, hidden_dim
        self.max_grad_norm = max_grad_norm
        d = dim // heads
        wg = torch.randn(heads, d, hidden_dim) / d ** .5
        wu = torch.randn(heads, d, hidden_dim) / d ** .5
        wo = torch.zeros(heads, hidden_dim, d)
        self.initial_weight = nn.Parameter(torch.cat([p.flatten() for p in (wg, wu, wo)]))
        self.log_lr = nn.Parameter(torch.full((heads,), float(torch.log(torch.expm1(torch.tensor(lr))))))

    def _read(self, weight, query):
        if query.ndim != 3 or query.shape[0] != weight.shape[0] or query.shape[-1] != self.dim:
            raise ValueError('Query batch/address dimension mismatch')
        if weight.ndim != 2 or weight.shape[1] != self.initial_weight.numel():
            raise ValueError('SwiGLU state shape mismatch')
        b, n, _ = query.shape
        h, d, r = self.heads, self.dim // self.heads, self.hidden_dim
        count = h * d * r
        wg, wu, wo = weight.float().split(count, dim=-1)
        wg, wu, wo = wg.reshape(b, h, d, r), wu.reshape(b, h, d, r), wo.reshape(b, h, r, d)
        q = F.normalize(query.float().reshape(b, n, h, d), dim=-1) * d ** .5
        hidden = F.silu(torch.einsum('bnhd,bhdr->bnhr', q, wg))
        hidden = hidden * torch.einsum('bnhd,bhdr->bnhr', q, wu)
        return torch.einsum('bnhr,bhrd->bnhd', hidden, wo).reshape(b, n, self.dim)

    def read(self, state, query):
        unbatched = query.ndim == 2
        with torch.autocast(device_type=query.device.type, enabled=False):
            value = self._read(state.weight, query[None] if unbatched else query)
        return value[0] if unbatched else value

    def update(self, weight, keys, values, *, create_graph=False, replay=None, replay_weight=0.):
        if keys.ndim == 2:
            keys, values = keys[None], values[None]
        if keys.shape != values.shape or keys.ndim != 3 or not keys.shape[1]:
            raise ValueError('K/V shape mismatch or empty support')
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite support K/V')
        with torch.enable_grad(), torch.autocast(device_type=keys.device.type, enabled=False):
            if not weight.requires_grad:
                weight = weight.detach().float().requires_grad_(True)
            def objective(k, v):
                # Per-head mean error; sum episode branches for batch-invariant SGD.
                return .5 * (self._read(weight, k) - v.float()).square().mean((1, 2)).sum()
            loss = objective(keys, values)
            if replay is not None and replay_weight:
                loss = loss + replay_weight * objective(*replay)
            grad, = torch.autograd.grad(loss, weight, create_graph=create_graph)
            if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                raise FloatingPointError('Nonfinite fast-memory loss or gradient')
            norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scale = (self.max_grad_norm / norm).clamp(max=1)
            per_head = (self.dim // self.heads) * self.hidden_dim
            rates = F.softplus(self.log_lr).repeat_interleave(per_head).repeat(3)
            proposed = weight.float() - rates * grad * scale
            if not torch.isfinite(proposed).all():
                raise FloatingPointError('Nonfinite proposed fast memory')
        return proposed, loss
