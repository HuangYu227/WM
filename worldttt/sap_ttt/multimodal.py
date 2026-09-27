"""Token-local semantic addresses with QK normalization and noise/role AdaLN.

Follows the normalized cross-attention and shift/scale/gate patterns already in
SANA and YUME's ModulateDiT. Uses native PyTorch only. Math SDPA is intentional:
the differentiable inner update needs second derivatives through these addresses.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .address import MODES


class FusionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # Small modulation keeps noise/geometry active from the first step. Only
        # the outer SANA residual gate is zero; zeroing both would block learning.
        nn.init.normal_(self.modulation[-1].weight, std=.01)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, text, mask, condition, mode):
        shift, scale, gate, fs, fc, fg = self.modulation(condition).chunk(6, -1)
        z = self.norm(x) * (1 + scale) + shift
        if mode != 'vision':
            if mode == 'global':
                text = (text * mask[..., None]).sum(1, keepdim=True) / mask.sum(1)[:, None, None]
                mask = torch.ones(text.shape[:2], device=x.device, dtype=torch.bool)
            b, n, d = z.shape
            h, hd = self.heads, d // self.heads
            q = self.q(z).reshape(b, n, h, hd).transpose(1, 2)
            k, v = self.kv(text).chunk(2, -1)
            k = k.reshape(b, -1, h, hd).transpose(1, 2)
            v = v.reshape(b, -1, h, hd).transpose(1, 2)
            q = F.rms_norm(q, (hd,))
            k = F.rms_norm(k, (hd,))
            with sdpa_kernel(SDPBackend.MATH):
                selected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None])
            selected = selected.transpose(1, 2).reshape(b, n, d)
            x = x + gate.sigmoid() * self.out(selected)
        return x + fg.sigmoid() * self.ffn(self.norm(x) * (1 + fc) + fs)


class MultimodalAddress(nn.Module):
    def __init__(self, vision_dim, text_dim, ray_dim=48, dim=512, heads=8, depth=2):
        super().__init__()
        if min(vision_dim, text_dim, ray_dim, dim, heads, depth) < 1 or dim % heads:
            raise ValueError('Invalid multimodal address dimensions')
        self.dim, self.ray_dim = dim, ray_dim
        self.visual_norm = nn.LayerNorm(vision_dim, elementwise_affine=False)
        self.text_norm = nn.LayerNorm(text_dim, elementwise_affine=False)
        self.visual = nn.Linear(vision_dim, dim, bias=False)
        self.text = nn.Linear(text_dim, dim, bias=False)
        self.ray = nn.Sequential(nn.Linear(ray_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(33, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.role = nn.Embedding(2, dim)
        nn.init.normal_(self.role.weight, std=.02)
        self.blocks = nn.ModuleList(FusionBlock(dim, heads) for _ in range(depth))
        self.writer = nn.Linear(dim, dim, bias=False)
        self.reader = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(vision_dim, dim, bias=False)
        nn.init.orthogonal_(self.value.weight)
        self.value.requires_grad_(False)
        self.register_buffer('frequencies', torch.exp(torch.linspace(0, math.log(1000), 16)))

    def _encode(self, visual, text, text_mask, rays, sigma, role, mode):
        if mode not in MODES or visual.ndim != 3:
            raise ValueError('Invalid address mode or visual shape')
        b, n, _ = visual.shape
        if sigma.shape != (b, n, 1) or not torch.isfinite(sigma).all():
            raise ValueError('Invalid per-token sigma')
        if text.ndim != 3 or text.shape[0] != b or text_mask.shape != text.shape[:2]:
            raise ValueError('Invalid text shape/mask')
        mask = text_mask.bool()
        if not mask.any(1).all():
            raise ValueError('Every branch needs valid text')
        with torch.autocast(device_type=visual.device.type, enabled=False):
            x = self.visual(self.visual_norm(visual.float()))
            # Sanitize padding before projections as NaN padding cannot be masked
            # safely after matrix multiplication.
            t = self.text(self.text_norm(text.float().masked_fill(~mask[..., None], 0)))
            angle = sigma.float() * self.frequencies
            c = self.time(torch.cat((sigma.float(), angle.sin(), angle.cos()), -1))
            c = c + self.role.weight[role]
            if mode == 'selective_geometry':
                if rays is None or rays.shape != (b, n, self.ray_dim):
                    raise ValueError('Per-token geometry shape mismatch')
                # asinh bounds large Plucker moments without discarding magnitude.
                c = c + self.ray(torch.asinh(rays.float()))
            for block in self.blocks:
                x = block(x, t, mask, c, mode)
            projection = self.writer if role == 0 else self.reader
            return F.normalize(projection(F.layer_norm(x, (self.dim,))), dim=-1)

    def write(self, visual, text, text_mask, rays=None, *, mode='selective_geometry'):
        return self._encode(visual, text, text_mask, rays,
                            visual.new_zeros(*visual.shape[:2], 1), 0, mode)

    def query(self, visual, text, text_mask, rays, sigma, *, mode='selective_geometry'):
        return self._encode(visual, text, text_mask, rays, sigma, 1, mode)

    def value_for(self, visual):
        with torch.autocast(device_type=visual.device.type, enabled=False):
            return F.layer_norm(self.value(self.visual_norm(visual.float())), (self.dim,))
