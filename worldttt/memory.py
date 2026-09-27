"""Pure PyTorch fast memory. No SANA/CUDA imports are needed for unit tests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TTTConfig:
    mode: str = 'noise_ttt'
    frozen_source: str | None = None
    layers: tuple[int, ...] = (3, 7, 11, 15)
    input_dim: int = 64
    hidden_dim: int = 128
    support_tokens: int = 256
    anchor_capacity: int = 512
    inner_steps: int = 1
    inner_lr: float = 0.01
    keep_weight: float = 0.1
    trust_weight: float = 0.001
    grad_clip: float = 1.0
    seed: int = 3407
    reset_each_chunk: bool = False
    shuffle_targets: bool = False

    def __post_init__(self):
        self.layers = tuple(self.layers)
        if self.mode not in {'off', 'frozen', 'kv_ttt', 'noise_ttt'}:
            raise ValueError(f'Unknown TTT mode: {self.mode}')
        if self.frozen_source is not None and (self.mode != 'frozen' or self.frozen_source not in {'kv_ttt', 'noise_ttt'}):
            raise ValueError('frozen_source requires frozen mode and a valid source')
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 1:
            raise ValueError('layers must be unique one-based indices')
        for name in ('input_dim', 'hidden_dim', 'support_tokens', 'anchor_capacity', 'inner_steps'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.inner_lr <= 0 or self.grad_clip <= 0 or min(self.keep_weight, self.trust_weight) < 0:
            raise ValueError('Invalid optimizer parameters')


@dataclass
class Features:
    x: torch.Tensor
    m: torch.Tensor
    pose: torch.Tensor
    sigma: torch.Tensor
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    def select(self, indices):
        return Features(**{k: None if v is None else v[:, indices] for k, v in vars(self).items()})

    def detach(self):
        return Features(**{k: None if v is None else v.detach().float() for k, v in vars(self).items()})

    def to(self, device):
        return Features(**{k: None if v is None else v.to(device) for k, v in vars(self).items()})

    def kv(self):
        if self.k is None or self.v is None:
            raise ValueError('kv_ttt requires native keys and values')
        return Features(self.k, torch.zeros_like(self.m), self.pose, self.sigma, self.k, self.v)


class FastMemory(nn.Module):
    def __init__(self, heads: int, head_dim: int, config: TTTConfig):
        super().__init__()
        self.config, self.heads, self.head_dim = config, heads, head_dim
        d, h, p, r = head_dim, heads, config.input_dim, config.hidden_dim
        self.input_proj = nn.Parameter(torch.randn(h, 2 * d, p) / (2 * d) ** 0.5)
        self.condition_proj = nn.Parameter(torch.randn(h, 21, p) / 21 ** 0.5 * 0.01)
        self.w_in = nn.Parameter(torch.randn(h, p, r) / p ** 0.5)
        self.w_gate = nn.Parameter(torch.randn(h, p, r) / p ** 0.5)
        self.w_out = nn.Parameter(torch.zeros(h, r, d))

    def initial_weights(self, batch: int, training=False):
        weights = tuple(w.unsqueeze(0).expand(batch, *w.shape).clone() for w in
                        (self.w_in, self.w_gate, self.w_out))
        return weights if training else tuple(w.detach().requires_grad_(True) for w in weights)

    def forward(self, features: Features, weights):
        # Fast weights and their reductions remain fp32 even in a bf16 outer pass.
        with torch.autocast(device_type=features.x.device.type, enabled=False):
            b, n, _ = features.x.shape
            x = features.x.float().reshape(b, n, self.heads, self.head_dim)
            m = features.m.float().reshape_as(x)
            z = torch.cat((F.layer_norm(x, (self.head_dim,)), F.layer_norm(m, (self.head_dim,))), -1)
            z = torch.einsum('bnhd,hdp->bnhp', z, self.input_proj.float())
            cond = torch.cat((features.pose.float(), features.sigma.float()), -1)
            z = z + torch.einsum('bnc,hcp->bnhp', cond, self.condition_proj.float())
            wi, wg, wo = weights
            hidden = F.silu(torch.einsum('bnhp,bhpr->bnhr', z, wg))
            hidden = hidden * torch.einsum('bnhp,bhpr->bnhr', z, wi)
            return torch.einsum('bnhr,bhrd->bnhd', hidden, wo).reshape(b, n, -1)


@dataclass
class AnchorBank:
    features: Features
    target: torch.Tensor
    seen: int = 0


@dataclass
class WorldTTTState:
    episode: str
    weights: dict[int, tuple[torch.Tensor, ...]]
    seed: int = 3407
    last_chunk: int = -1
    updates: int = 0
    anchors: dict[int, AnchorBank] = field(default_factory=dict)
    rng_state: torch.Tensor | None = None

    def generator(self):
        rng = torch.Generator().manual_seed(self.seed)
        if self.rng_state is not None:
            rng.set_state(self.rng_state.cpu())
        return rng

    def state_dict(self):
        banks = {i: {'features': {k: None if v is None else v.detach().cpu()
                                 for k, v in vars(a.features).items()},
                     'target': a.target.detach().cpu(), 'seen': a.seen} for i, a in self.anchors.items()}
        return {'version': 1, 'episode': self.episode, 'seed': self.seed,
                    'last_chunk': self.last_chunk, 'updates': self.updates, 'rng_state': self.rng_state,
                    'weights': {i: tuple(w.detach().cpu() for w in ws) for i, ws in self.weights.items()},
                    'anchors': banks}

    def save_state(self, path):
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load_state(cls, path, device='cpu'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        return cls.from_state_dict(payload, device)

    @classmethod
    def from_state_dict(cls, payload, device='cpu'):
        payload = dict(payload)
        if payload.pop('version') != 1:
            raise ValueError('Unsupported WorldTTT state version')
        payload['weights'] = {i: tuple(w.to(device).float().requires_grad_(True) for w in ws)
                              for i, ws in payload['weights'].items()}
        payload['anchors'] = {i: AnchorBank(Features(**a['features']).to(device), a['target'].to(device), a['seen'])
                              for i, a in payload['anchors'].items()}
        return cls(**payload)


def normalized_loss(pred, target, heads):
    # Sum independent sample objectives: doubling a CFG batch must not halve its learning rate.
    b, n, c = pred.shape
    target = target.reshape(b, n, heads, c // heads).float()
    scale = target.square().mean(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    error = (pred.reshape_as(target) - target).square() / scale
    return error.mean(dim=(1, 2, 3)).sum()


def _add_anchors(bank, features, target, capacity, rng):
    features, target = features.detach(), target.detach()
    if bank is None:
        bank = AnchorBank(features.select([]), target[:, :0].clone())
    early = capacity // 2
    for j in range(features.x.shape[1]):
        n = bank.features.x.shape[1]
        if n < capacity:
            for key, val in vars(features).items():
                if val is not None:
                    setattr(bank.features, key, torch.cat((getattr(bank.features, key), val[:, j:j + 1]), 1))
            bank.target = torch.cat((bank.target, target[:, j:j + 1]), 1)
        else:
            pool_seen = bank.seen - early
            slot = int(torch.randint(pool_seen + 1, (1,), generator=rng))
            if slot < capacity - early:
                slot += early
                for key, val in vars(features).items():
                    if val is not None:
                        getattr(bank.features, key)[:, slot] = val[:, j]
                bank.target[:, slot] = target[:, j]
        bank.seen += 1
    return bank


def adapt_after_chunk(memories, state, features, targets, chunk, training=False):
    """Transactional functional SGD; shared by training and deployment."""
    if chunk != state.last_chunk + 1:
        raise ValueError(f'Expected chunk {state.last_chunk + 1}, received chunk {chunk}')
    config = next(iter(memories.values())).config
    result = {'chunk': chunk, 'committed': False, 'updates': state.updates}
    if config.mode in {'off', 'frozen'}:
        state.last_chunk = chunk
        return result
    rng = state.generator()
    proposals, anchor_data = {}, {}
    before, after = [], []
    with torch.enable_grad():
        for layer, memory in memories.items():
            f = features[layer].detach()
            if config.mode == 'kv_ttt':
                f, target = f.kv(), f.v.detach().float()
            else:
                target = targets[layer].detach().float()
            if not all(torch.isfinite(v).all() for v in (f.x, f.m, f.pose, f.sigma, target)):
                result['reason'] = f'nonfinite_features_layer_{layer}'
                break
            n = f.x.shape[1]
            if n < 2:
                raise ValueError('At least two valid tokens are needed for support/validation separation')
            perm = torch.randperm(n, generator=rng).to(f.x.device)
            count = min(config.support_tokens, n - 1)
            support, holdout = perm[:count], perm[count:count + config.support_tokens]
            sf, st = f.select(support), target[:, support]
            if config.shuffle_targets:
                st = st[:, torch.randperm(count, generator=rng).to(st.device)]
            weights = state.weights[layer]
            original = tuple(w.detach() for w in weights)
            if not training:
                weights = tuple(w.detach().requires_grad_(True) for w in weights)
            def read(feat, ws):
                return feat.m + memory(feat, ws)
            with torch.no_grad():
                before.append(float(normalized_loss(read(f.select(holdout), weights), target[:, holdout], memory.heads)))
            for _ in range(config.inner_steps):
                loss = normalized_loss(read(sf, weights), st, memory.heads)
                bank = state.anchors.get(layer)
                if bank is not None and config.keep_weight:
                    loss = loss + config.keep_weight * normalized_loss(read(bank.features, weights), bank.target, memory.heads)
                loss = loss + config.trust_weight * sum((w - old).square().flatten(1).mean(1).sum()
                                                       for w, old in zip(weights, original))
                if not torch.isfinite(loss):
                    result['reason'] = f'nonfinite_loss_layer_{layer}'
                    break
                grads = torch.autograd.grad(loss, weights, create_graph=training)
                if not all(torch.isfinite(g).all() for g in grads):
                    result['reason'] = f'nonfinite_gradient_layer_{layer}'
                    break
                norms = sum(g.square().flatten(1).sum(1) for g in grads).clamp_min(1e-24).sqrt()
                factor = (config.grad_clip / norms).clamp(max=1)
                weights = tuple(w - config.inner_lr * g * factor.reshape(-1, 1, 1, 1)
                                for w, g in zip(weights, grads))
            if 'reason' in result or not all(torch.isfinite(w).all() for w in weights):
                result.setdefault('reason', f'nonfinite_weights_layer_{layer}')
                break
            proposals[layer] = weights if training else tuple(w.detach().requires_grad_(True) for w in weights)
            with torch.no_grad():
                after.append(float(normalized_loss(read(f.select(holdout), weights), target[:, holdout], memory.heads)))
                anchor_data[layer] = (sf, read(sf, weights).detach())
        if len(proposals) == len(memories):
            state.weights.update(proposals)
            for layer, (f, t) in anchor_data.items():
                state.anchors[layer] = _add_anchors(state.anchors.get(layer), f, t, config.anchor_capacity, rng)
            state.updates += 1
            result.update(committed=True, updates=state.updates,
                          heldout_before=sum(before) / len(before), heldout_after=sum(after) / len(after))
    state.rng_state = rng.get_state()
    state.last_chunk = chunk
    return result
