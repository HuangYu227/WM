from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import copy


FIVE_BINDING_LAYERS = (3, 7, 11, 15, 19)


@dataclass
class BindingConfig:
    mode: str = 'online'
    layers: tuple[int, ...] = FIVE_BINDING_LAYERS
    address_dim: int = 512
    value_dim: int = 512
    ray_dim: int = 48
    latent_dim: int = 128
    heads: int = 8
    topk: int = 8
    capacity: int = 512
    protected_anchors: int = 128
    support_tokens: int = 256
    address_depth: int = 2
    fast_hidden_dim: int = 128
    fast_lr: float = .1
    fast_trust_weight: float = 1e-4
    seed: int = 3407
    value_mode: str = 'highpass_h7_latent'
    architecture: str = 'hybrid'
    read_enabled: bool = True
    commit_enabled: bool = True

    def __post_init__(self):
        self.layers = tuple(self.layers)
        if self.mode not in {'frozen', 'online'}:
            raise ValueError('Binding mode must be frozen or online')
        if self.value_mode not in {'old', 'centered_h7', 'highpass_h7', 'highpass_h7_latent'}:
            raise ValueError('Unsupported binding value mode')
        if self.architecture not in {'hybrid', 'bank_only', 'fast_only'}:
            raise ValueError('Unsupported SAP-Bind architecture')
        numbers = (self.address_dim, self.value_dim, self.ray_dim, self.latent_dim,
                   self.heads, self.topk,
                   self.capacity, self.support_tokens, self.address_depth, self.fast_hidden_dim)
        if min(numbers) < 1 or self.fast_lr <= 0 or self.fast_trust_weight < 0:
            raise ValueError('Invalid SAP-Bind dimensions')
        if self.address_dim != self.value_dim:
            raise ValueError('The first SAP-Bind version requires equal address/value dimensions')
        if self.address_dim % self.heads or self.value_dim % self.heads:
            raise ValueError('Address and value dimensions must be divisible by heads')
        if self.topk > self.capacity or not 0 <= self.protected_anchors <= self.capacity:
            raise ValueError('Invalid binding bank budget')
        if not self.layers or min(self.layers) < 1 or len(set(self.layers)) != len(self.layers):
            raise ValueError('Binding layers must be unique one-based indices')


def binding_settings(base, *, layers=FIVE_BINDING_LAYERS):
    settings = copy.deepcopy(base)
    settings['sap_binding'] = asdict(BindingConfig(layers=layers))
    settings['sap_binding']['layers'] = list(settings['sap_binding']['layers'])
    settings['steps'] = 50
    settings['save_rollout_state'] = False
    return settings
