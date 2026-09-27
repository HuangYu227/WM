"""Separate clean writer and noisy reader addresses with optional local text/geometry."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODES = {'vision', 'global', 'selective', 'selective_geometry'}


class SapAddress(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, ray_dim: int = 48, dim: int = 256):
        super().__init__()
        self.dim, self.ray_dim = dim, ray_dim
        self.visual = nn.Linear(vision_dim, dim, bias=False)
        self.text_key = nn.Linear(text_dim, dim, bias=False)
        self.text_value = nn.Linear(text_dim, dim, bias=False)
        self.ray = nn.Linear(ray_dim, dim, bias=False)
        self.writer = nn.Linear(dim, dim, bias=False)
        self.reader = nn.Linear(dim + 1, dim, bias=False)
        self.value = nn.Linear(vision_dim, dim, bias=False)
        # Keep the target feature space fixed in the first mechanism study;
        # a learned target can collapse to zero and fake good retrieval MSE.
        self.value.requires_grad_(False)
        self.normalized_value = False

    def _content(self, visual, text, text_mask, rays, mode):
        if mode not in MODES:
            raise ValueError(f'Unknown SAP address mode: {mode}')
        if visual.ndim != 3:
            raise ValueError('Visual tokens must have shape (B,N,C)')
        b, n, _ = visual.shape
        z = self.visual(visual.float())
        if mode != 'vision':
            if text.ndim != 3 or text.shape[0] != b or text_mask.shape != text.shape[:2]:
                raise ValueError('Text tokens/mask batch mismatch')
            keep = text_mask.bool()
            if not keep.any(dim=1).all():
                raise ValueError('Each branch needs a valid text token')
            text = text.float()
            if mode == 'global':
                selected = (self.text_value(text) * keep[..., None]).sum(1, keepdim=True) / keep.sum(1)[:, None, None]
            else:
                scores = torch.bmm(z, self.text_key(text).transpose(1, 2)) * self.dim ** -.5
                scores = scores.masked_fill(~keep[:, None], torch.finfo(scores.dtype).min)
                selected = torch.bmm(scores.softmax(-1), self.text_value(text))
            z = z + selected
        if mode == 'selective_geometry':
            if rays is None or rays.shape != (b, n, self.ray_dim):
                raise ValueError('Per-token ray shape mismatch')
            z = z + self.ray(rays.float())
        return z

    def write(self, visual, text, text_mask, rays=None, *, mode='selective_geometry'):
        return F.normalize(self.writer(self._content(visual, text, text_mask, rays, mode)), dim=-1)

    def query(self, visual, text, text_mask, rays, sigma, *, mode='selective_geometry'):
        z = self._content(visual, text, text_mask, rays, mode)
        if sigma.shape != (*visual.shape[:2], 1):
            raise ValueError('Per-token sigma shape mismatch')
        return F.normalize(self.reader(torch.cat((z, sigma.float()), -1)), dim=-1)

    def value_for(self, visual):
        if self.normalized_value:
            with torch.autocast(device_type=visual.device.type, enabled=False):
                return F.layer_norm(self.value(F.layer_norm(visual.float(), (visual.shape[-1],))), (self.dim,))
        return self.value(visual.float())


def initialize_fixed_value(address, seed):
    """Architecture-independent teacher, held identical across fusion ablations."""
    generator = torch.Generator(device=address.value.weight.device).manual_seed(seed)
    nn.init.orthogonal_(address.value.weight, generator=generator)
    address.value.requires_grad_(False)
