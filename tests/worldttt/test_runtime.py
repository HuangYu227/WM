import importlib.util

import pytest
import torch
from torch import nn


def test_runtime_exists():
    assert importlib.util.find_spec('worldttt.runtime') is not None


def test_context_is_readonly_and_collects_pre_residual_targets():
    from worldttt.memory import TTTConfig
    from worldttt.runtime import WorldTTTController
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads, self.dim = 2, 3
            self.qkv = nn.Linear(6, 18)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            block = nn.Module()
            block.attn = Attn()
            self.blocks = nn.ModuleList([block])
    model = Model()
    cfg = TTTConfig(layers=(1,), input_dim=4, hidden_dim=6, support_tokens=2)
    ctl = WorldTTTController(model, cfg, validate_blocks=False)
    ctl.reset_episode('scene', 2)
    ctx = ctl.context(torch.ones(2) * 500, collect=True, chunk=0)
    x, m = torch.randn(2, 8, 6), torch.randn(2, 8, 6)
    pose = torch.randn(2, 2, 20)
    with torch.no_grad():
        out = ctx.apply(model.blocks[0].attn, x, m, pose, (2, 2, 2))
    assert torch.equal(out, m)
    assert ctl.state.last_chunk == -1
    assert ctx.features[1].sigma.eq(0.5).all()
    assert ctx.features[1].x.shape == (2, 4, 6)
    assert ctx.features[1].k.shape == (2, 4, 6)
    ctl.reset_episode('new', 2)
    assert ctl.state.anchors == {} and ctl.state.last_chunk == -1


def test_cache_snapshot_does_not_alias():
    from worldttt.runtime import clone_cache
    cache = [[torch.ones(2), None]]
    snapshot = clone_cache(cache)
    snapshot[0][0].zero_()
    assert cache[0][0].eq(1).all()


def test_frozen_kv_checkpoint_preserves_query_read_and_rejects_wrong_active_mode(tmp_path):
    from worldttt.memory import TTTConfig
    from worldttt.runtime import WorldTTTController

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads, self.dim = 2, 3
            self.qkv = nn.Linear(6, 18)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            block = nn.Module()
            block.attn = Attn()
            self.blocks = nn.ModuleList([block])

    source = WorldTTTController(Model(), TTTConfig(mode='kv_ttt', layers=(1,), input_dim=4,
                                                    hidden_dim=6), validate_blocks=False)
    path = tmp_path / 'kv.pt'
    source.save_checkpoint(path)
    frozen = WorldTTTController(Model(), TTTConfig(mode='frozen', layers=(1,), input_dim=4,
                                                    hidden_dim=6), validate_blocks=False)
    frozen.load_checkpoint(path)
    assert frozen.config.frozen_source == 'kv_ttt'
    frozen.reset_episode('scene', 1)
    layer = frozen.model.blocks[0].attn
    captured = []
    def capture(features, weights):
        captured.append(features)
        return torch.zeros_like(features.x)
    layer.worldttt_memory.forward = capture
    x, m = torch.randn(1, 4, 6), torch.randn(1, 4, 6)
    with torch.no_grad():
        frozen.context(torch.zeros(1)).apply(layer, x, m, torch.randn(1, 1, 20), (1, 2, 2))
    q = layer.qkv(x).reshape(1, 4, 3, 6)[:, :, 0]
    assert torch.allclose(captured[0].x, q)
    assert torch.count_nonzero(captured[0].m) == 0

    wrong = WorldTTTController(Model(), TTTConfig(mode='noise_ttt', layers=(1,), input_dim=4,
                                                   hidden_dim=6), validate_blocks=False)
    with pytest.raises(ValueError, match='mode'):
        wrong.load_checkpoint(path)
