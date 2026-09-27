"""Per-episode functional K-to-V fast weights with transactional chunk writes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SapMemory(nn.Module):
    def __init__(self, dim: int = 256, lr: float = .5, max_grad_norm: float = 256.):
        super().__init__()
        if dim < 1 or lr <= 0 or max_grad_norm <= 0:
            raise ValueError('Invalid fast-memory dimensions or update settings')
        self.dim, self.max_grad_norm = dim, max_grad_norm
        self.initial_weight = nn.Parameter(torch.zeros(dim, dim, dtype=torch.float32))
        self.log_lr = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(lr)))))

    def read(self, state: SapMemoryState, query: torch.Tensor) -> torch.Tensor:
        if query.shape[-1] != self.dim:
            raise ValueError('Query address dimension mismatch')
        weight = state.weight
        if query.ndim == 2:
            if weight.shape[0] != 1:
                raise ValueError('Unbatched query requires one episode branch')
            return query.float() @ weight[0]
        if query.ndim != 3 or query.shape[0] != weight.shape[0]:
            raise ValueError('Query batch/CFG branch mismatch')
        return torch.bmm(query.float(), weight)

    def update(self, weight: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
               *, create_graph: bool = False, replay: tuple[torch.Tensor, torch.Tensor] | None = None,
               replay_weight: float = 0.) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.ndim == 2:
            keys, values = keys[None], values[None]
        if keys.shape != values.shape or keys.ndim != 3 or keys.shape[0] != weight.shape[0] or keys.shape[-1] != self.dim:
            raise ValueError('K/V shape or branch mismatch')
        if keys.shape[1] == 0:
            raise ValueError('Cannot write empty chunk')
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite support K/V')
        with torch.enable_grad():
            if not weight.requires_grad:
                weight = weight.detach().requires_grad_(True)
            # Average tokens within each episode branch, then sum branches so
            # CFG does not silently halve the adaptation step for each branch.
            loss = .5 * (torch.bmm(keys.float(), weight) - values.float()).square().sum(-1).mean(1).sum()
            if replay is not None and replay_weight:
                old_k, old_v = replay
                loss = loss + replay_weight * .5 * (torch.bmm(old_k.float(), weight) - old_v.float()).square().sum(-1).mean(1).sum()
            grad, = torch.autograd.grad(loss, weight, create_graph=create_graph)
            if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                raise FloatingPointError('Nonfinite fast-memory loss or gradient')
            norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
            scale = (self.max_grad_norm / norm).clamp(max=1).view(-1, 1, 1)
            proposed = weight - F.softplus(self.log_lr) * grad * scale
            if not torch.isfinite(proposed).all():
                raise FloatingPointError('Nonfinite proposed fast memory')
            return proposed, loss


@dataclass
class SapMemoryState:
    episode: str
    weight: torch.Tensor
    last_chunk: int = -1
    updates: int = 0

    @classmethod
    def new(cls, memory: SapMemory, episode: str, batch: int = 1, training: bool = False):
        if batch < 1:
            raise ValueError('Batch must be positive')
        initial = memory.initial_weight.float().unsqueeze(0).expand(batch, *memory.initial_weight.shape).clone()
        return cls(str(episode), initial if training else initial.detach().requires_grad_(True))

    def commit(self, memory: SapMemory, chunk: int, keys: torch.Tensor, values: torch.Tensor,
               *, training: bool = False, replay=None, replay_weight: float = 0.) -> dict:
        if chunk != self.last_chunk + 1:
            raise ValueError(f'Expected chunk {self.last_chunk + 1}, received {chunk}')
        try:
            proposed, loss = memory.update(self.weight, keys, values, create_graph=training,
                                           replay=replay, replay_weight=replay_weight)
        except FloatingPointError as exc:
            return {'chunk': chunk, 'committed': False, 'reason': str(exc), 'updates': self.updates}
        self.weight = proposed if training else proposed.detach().requires_grad_(True)
        self.last_chunk = chunk
        self.updates += 1
        return {'chunk': chunk, 'committed': True, 'updates': self.updates,
                'write_objective': float(loss.detach())}

    def state_dict(self):
        return {'version': 1, 'episode': self.episode, 'weight': self.weight.detach().cpu(),
                'last_chunk': self.last_chunk, 'updates': self.updates}

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('version') != 1:
            raise ValueError('Unsupported SAP state version')
        return cls(state['episode'], state['weight'].to(device).float().requires_grad_(True),
                   state['last_chunk'], state['updates'])
