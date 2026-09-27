from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from worldttt.sap_ttt.multimodal import MultimodalAddress
from worldttt.sap_ttt.nonlinear import SwiGLUMemory

from .memory import explicit_read
from .value import BindingValueEncoder


class BindingFastMemory(SwiGLUMemory):
    """SAP-Bind's proximal trust region; the legacy SAP fast memory is untouched."""

    def __init__(self, dim, heads, hidden_dim, lr, trust_weight):
        super().__init__(dim, heads, hidden_dim, lr)
        self.trust_weight = trust_weight

    def update(self, weight, keys, values, *, create_graph=False, replay=None, replay_weight=0.):
        proposed, loss = super().update(weight, keys, values, create_graph=create_graph,
                                        replay=replay, replay_weight=replay_weight)
        if self.trust_weight:
            initial = self.initial_weight.float().unsqueeze(0)
            size = self.initial_weight.numel()
            per_head = (self.dim // self.heads) * self.hidden_dim
            rates = F.softplus(self.log_lr).repeat_interleave(per_head).repeat(3)
            strength = rates * (self.trust_weight / size)
            proposed = (proposed + strength * initial) / (1 + strength)
            loss = loss + .5 * self.trust_weight * (weight.float() - initial).square().mean()
        return proposed, loss


class BindingBlock(nn.Module):
    def __init__(self, vision_dim, latent_dim, config):
        super().__init__()
        self.config = config
        self.address = MultimodalAddress(vision_dim, vision_dim, config.ray_dim,
                                         config.address_dim, config.heads, config.address_depth)
        self.value = BindingValueEncoder(vision_dim, latent_dim, config.value_mode,
                                         config.value_dim, config.seed)
        self.fast = BindingFastMemory(config.address_dim, config.heads,
                                      config.fast_hidden_dim, config.fast_lr,
                                      config.fast_trust_weight)
        ray_head_dim = config.address_dim // config.heads
        self.ray_query = nn.Linear(config.ray_dim, config.heads * ray_head_dim, bias=False)
        self.ray_key = nn.Linear(config.ray_dim, config.heads * ray_head_dim, bias=False)
        self.fusion = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, config.heads))
        self.output_norm = nn.LayerNorm(config.value_dim, elementwise_affine=False)
        self.output = nn.Linear(config.value_dim, vision_dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(1, 2 * vision_dim))
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.fusion[-1].bias)
        nn.init.zeros_(self.modulation[-1].bias)

    def read(self, state, query, rays, sigma, *, shuffle_key=False, shuffle_value=False,
             mean_value=False, last_only=False):
        bank = state.bank
        if mean_value:
            valid = bank.valid[..., None]
            explicit = (bank.values * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
            explicit = explicit.expand(-1, query.shape[1], -1)
            stats = dict(margin=query.new_zeros(*query.shape[:2], self.config.heads),
                         entropy=query.new_ones(*query.shape[:2], self.config.heads),
                         age=query.new_zeros(*query.shape[:2], self.config.heads),
                         has_memory=bank.valid.any(1)[:, None, None].expand(-1, query.shape[1], 1))
            return explicit, stats, explicit.new_ones(*explicit.shape[:2], self.config.heads)
        elif last_only:
            restricted = bank.clone()
            newest = restricted.age.masked_fill(~restricted.valid, -1).max(1).values
            restricted.valid &= restricted.age == newest[:, None]
            explicit, stats = explicit_read(restricted, query, rays, self.ray_query, self.ray_key,
                heads=self.config.heads, topk=self.config.topk)
        else:
            explicit, stats = explicit_read(bank, query, rays, self.ray_query, self.ray_key,
                heads=self.config.heads, topk=self.config.topk,
                shuffle_key=shuffle_key, shuffle_value=shuffle_value)
        fast = self.fast.read(state.fast, query)
        if self.config.architecture == 'bank_only':
            return explicit, stats, explicit.new_ones(*explicit.shape[:2], self.config.heads)
        if self.config.architecture == 'fast_only':
            return fast, stats, explicit.new_zeros(*explicit.shape[:2], self.config.heads)
        age = stats['age'] / max(1, state.last_chunk + 1)
        inputs = torch.stack((stats['margin'], stats['entropy'],
                              sigma.expand(-1, -1, self.config.heads), age), -1)
        logits = self.fusion(inputs)
        mix = logits.diagonal(dim1=-2, dim2=-1).sigmoid()
        vd = self.config.value_dim // self.config.heads
        value = mix[..., None] * explicit.reshape(*explicit.shape[:2], self.config.heads, vd)
        value = value + (1 - mix[..., None]) * fast.reshape(*fast.shape[:2], self.config.heads, vd)
        return value.flatten(-2), stats, mix

    def residual(self, value, sigma):
        shift, scale = self.modulation(sigma.float()).chunk(2, -1)
        modulated = self.output_norm(value.float())
        return self.gate.float() * (self.output(modulated) * (1 + scale) + shift)
