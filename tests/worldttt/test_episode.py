from types import SimpleNamespace

import torch
import pytest
from torch import nn
from torch.utils.checkpoint import checkpoint

from worldttt.episode import EpisodeModel
from worldttt.memory import TTTConfig
from worldttt.runtime import WorldTTTController


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        attn = nn.Linear(4, 4)
        attn.heads, attn.dim = 2, 2
        attn.qkv = nn.Linear(4, 12)
        block = nn.Module()
        block.attn = attn
        self.blocks = nn.ModuleList([block])
        self.commits = []

    def forward(self, z, t, kv_cache, save_kv_cache, start_f, worldttt_context=None, **kw):
        b, c, frames, h, w = z.shape
        x = z.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
        m = self.blocks[0].attn(x) + kv_cache[0]
        if worldttt_context is not None:
            m = worldttt_context.apply(self.blocks[0].attn, x, m, kw['camera_conditions'], (frames, h, w))
        if save_kv_cache:
            self.commits.append(start_f)
            kv_cache[0] = m.mean().detach()
        return m.reshape(b, frames, h, w, c).permute(0, 4, 1, 2, 3), kv_cache


@pytest.mark.parametrize('recompute', [False, True])
def test_two_support_one_query_meta_gradients_and_no_future_commit(recompute):
    torch.manual_seed(2)
    model = ToyModel()
    if recompute:
        original = model.forward
        def forward(*args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(original, *args, **kwargs, use_reentrant=False)
            return original(*args, **kwargs)
        model.forward = forward
    ctl = WorldTTTController(model, TTTConfig(layers=(1,), input_dim=3, hidden_dim=5,
                            support_tokens=3, anchor_capacity=6), validate_blocks=False)
    def caches(_):
        state = [[torch.tensor(0.)] for _ in range(3)]
        return state, lambda i: state[max(i - 1, 0)]
    episode = EpisodeModel(model, ctl, cache_factory=caches,
                            schedule_factory=lambda *a: SimpleNamespace(timesteps=torch.tensor([800., 200.])))
    z = torch.randn(1, 4, 10, 2, 2)
    loss = episode(z, torch.randn(1, 10, 20), torch.randn(1, 48, 10, 2, 2), None, None, 'toy')
    loss.backward()
    assert model.commits == [0, 4]
    assert ctl.state.updates == 2
    mem = ctl.memories[1]
    for p in (mem.input_proj, mem.condition_proj, mem.w_out):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert model.blocks[0].attn.weight.grad is None
