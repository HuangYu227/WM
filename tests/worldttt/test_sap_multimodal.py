import pytest
import torch
from torch import nn


def build_address():
    from worldttt.sap_ttt.multimodal import MultimodalAddress
    return MultimodalAddress(12, 10, 6, dim=16, heads=2, depth=2)


def inputs(batch=2):
    torch.manual_seed(23)
    return (torch.randn(batch, 7, 12), torch.randn(batch, 5, 10),
            torch.tensor([[1, 1, 1, 0, 0]]).bool().expand(batch, -1),
            torch.randn(batch, 7, 6), torch.full((batch, 7, 1), .7))


def test_multimodal_mask_sigma_and_token_locality():
    a = build_address()
    x, t, mask, r, s = inputs()
    q = a.query(x, t, mask, r, s)
    assert q.shape == (2, 7, 16)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(2, 7))
    changed = t.clone(); changed[:, 3:] = 1000
    torch.testing.assert_close(q, a.query(x, changed, mask, r, s))
    assert not torch.allclose(q, a.query(x, t, mask, r, s * 0))
    future = x.clone(); future[:, -1] += 100
    torch.testing.assert_close(q[:, :-1], a.query(future, t, mask, r, s)[:, :-1])
    with pytest.raises(ValueError, match='text'):
        a.query(x, t, mask * False, r, s)
    assert not any(p.requires_grad for p in a.value.parameters())


def test_swiglu_update_is_branch_independent_and_meta_differentiable():
    from worldttt.sap_ttt.nonlinear import SwiGLUMemory
    from worldttt.sap_ttt.memory import SapMemoryState
    torch.manual_seed(4)
    a = build_address()
    mem = SwiGLUMemory(16, heads=2, hidden_dim=12, lr=.1)
    x, t, mask, r, s = inputs()
    k = a.write(x, t, mask, r)
    v = a.value_for(x)
    state = SapMemoryState.new(mem, 'meta', batch=2, training=True)
    before = mem.read(state, k).detach()
    assert state.commit(mem, 0, k, v, training=True)['committed']
    after = mem.read(state, k)
    assert (after - v).square().mean() < (before - v).square().mean()
    single = SapMemoryState.new(mem, 'one', batch=1, training=True)
    single.commit(mem, 0, k[:1], v[:1], training=True)
    torch.testing.assert_close(single.weight[0], state.weight[0])
    q = a.query(x * .8, t, mask, r, s)
    loss = (mem.read(state, q) - v.detach()).square().mean()
    loss.backward()
    for name, param in [('initial', mem.initial_weight), ('lr', mem.log_lr),
                        ('writer', a.writer.weight), ('reader', a.reader.weight),
                        ('text', a.text.weight), ('visual', a.visual.weight)]:
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_nonlinear_state_rejects_nan_and_survives_save(tmp_path):
    from worldttt.sap_ttt.nonlinear import SwiGLUMemory
    from worldttt.sap_ttt.memory import SapMemoryState
    mem = SwiGLUMemory(8, heads=2, hidden_dim=8)
    state = SapMemoryState.new(mem, 'a', 2)
    k, v = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    initial = state.weight.clone()
    assert not state.commit(mem, 0, k * float('nan'), v)['committed']
    torch.testing.assert_close(initial, state.weight)
    assert state.commit(mem, 0, k, v)['committed']
    with pytest.raises(ValueError, match='chunk'):
        state.commit(mem, 0, k, v)
    state.save(tmp_path / 'state.pt')
    restored = SapMemoryState.load(tmp_path / 'state.pt')
    torch.testing.assert_close(mem.read(restored, k), mem.read(state, k))


def controller():
    from worldttt.sap_ttt.runtime import SapConfig, SapController
    model = nn.Module()
    block = nn.Module(); block.norm2 = nn.LayerNorm(12)
    model.blocks = nn.ModuleList([block])
    return SapController(model, SapConfig(layers=(1,), dim=16, ray_dim=6,
        address_arch='multimodal', memory_arch='swiglu', heads=2,
        address_depth=2, memory_hidden_dim=12, support_tokens=5, inner_lr=.1))


def test_controller_v2_zero_output_live_flow_and_protocol(tmp_path):
    ctl = controller(); ctl.reset_episode('a', 2, training=True)
    block = ctl.model.blocks[0]
    x, _, mask, r, s = inputs()
    t = torch.randn(2, 5, 12)
    clean = ctl.context(torch.zeros(2), collect=True, training=True)
    with torch.no_grad():
        assert torch.equal(clean.apply(block, x, x, t, mask, r, 1), x)
    ctl.commit(clean, 0, training=True)
    query = ctl.context(torch.ones(2) * 700, training=True)
    out = query.apply(block, x, x, t, mask, r, 1)
    out.square().mean().backward(retain_graph=True)
    assert block.sap_ttt.gate.grad.abs().sum() > 0
    block.sap_ttt.gate.data.fill_(.1)
    ctl.model.zero_grad(set_to_none=True)
    query.apply(block, x, x, t, mask, r, 1).square().mean().backward()
    assert block.sap_ttt.address.writer.weight.grad.abs().sum() > 0
    assert block.sap_ttt.memory.initial_weight.grad.abs().sum() > 0
    ctl.save_checkpoint(tmp_path / 'adapter.pt')
    ctl.save_state(tmp_path / 'state.pt')
    other = controller(); other.load_checkpoint(tmp_path / 'adapter.pt')
    other.load_state(tmp_path / 'state.pt')
    torch.testing.assert_close(ctl.state[1].weight, other.state[1].weight)
    other.config.heads = 4
    with pytest.raises(ValueError, match='protocol'):
        other.load_checkpoint(tmp_path / 'adapter.pt')


def test_episode_load_rejects_wrong_protocol_without_replacing_live_state(tmp_path):
    ctl = controller(); ctl.reset_episode('original', 1)
    ctl.save_state(tmp_path / 'state.pt')
    other = controller(); other.reset_episode('live', 1)
    other.config.heads = 4
    with pytest.raises(ValueError, match='protocol'):
        other.load_state(tmp_path / 'state.pt')
    assert other.state[1].episode == 'live'


def test_episode_restore_requires_the_same_adapter_weights(tmp_path):
    ctl = controller(); ctl.reset_episode('recorded', 1)
    ctl.save_state(tmp_path / 'state.pt')
    other = controller(); other.reset_episode('live', 1)
    with pytest.raises(ValueError, match='adapter'):
        other.load_state(tmp_path / 'state.pt')
    assert other.state[1].episode == 'live'


def test_fast_update_stays_fp32_inside_bfloat16_autocast():
    from worldttt.sap_ttt.nonlinear import SwiGLUMemory
    from worldttt.sap_ttt.memory import SapMemoryState
    mem = SwiGLUMemory(8, heads=2, hidden_dim=8)
    state = SapMemoryState.new(mem, 'amp')
    k, v = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert state.commit(mem, 0, k, v)['committed']
        out = mem.read(state, k)
    assert state.weight.dtype == out.dtype == torch.float32


def test_inner_learning_rate_meta_gradient_matches_finite_difference():
    from worldttt.sap_ttt.nonlinear import SwiGLUMemory
    from worldttt.sap_ttt.memory import SapMemoryState
    torch.manual_seed(25)
    mem = SwiGLUMemory(8, heads=2, hidden_dim=8)
    k, q, v = torch.randn(1, 4, 8), torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    def objective():
        state = SapMemoryState.new(mem, 'grad', training=True)
        state.commit(mem, 0, k, v, training=True)
        return mem.read(state, q).square().mean()
    analytical, = torch.autograd.grad(objective(), mem.log_lr)
    original = mem.log_lr.detach().clone(); eps = .002
    with torch.no_grad():
        mem.log_lr.copy_(original + eps)
    plus = objective().detach()
    with torch.no_grad():
        mem.log_lr.copy_(original - eps)
    minus = objective().detach()
    with torch.no_grad():
        mem.log_lr.copy_(original)
    torch.testing.assert_close(analytical.sum(), (plus - minus) / (2 * eps), rtol=.01, atol=1e-6)


def test_address_ablation_uses_identical_fixed_value_targets():
    from worldttt.sap_ttt.runtime import SapConfig, SapBlockMemory
    config = dict(layers=(1,), dim=16, ray_dim=6, heads=2, normalized_value=True)
    simple = SapBlockMemory(12, SapConfig(**config))
    strong = SapBlockMemory(12, SapConfig(**config, address_arch='multimodal'))
    x = torch.randn(2, 7, 12)
    torch.testing.assert_close(simple.address.value_for(x), strong.address.value_for(x))


def test_four_chunk_flow_episode_jointly_trains_fusion_and_memory():
    from worldttt.sap_ttt.episode import SapEpisodeModel
    from worldttt.sap_ttt.runtime import SapController, SapConfig

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            block = nn.Module(); block.norm2 = nn.LayerNorm(12)
            self.blocks = nn.ModuleList([block])
            self.calls = []

        def forward(self, z, times, *, y, mask, sap_context, chunk_plucker, kv_cache, save_kv_cache, **kwargs):
            x = z.flatten(2).transpose(1, 2)
            out = sap_context.apply(self.blocks[0], x, x, y, mask, chunk_plucker, z.shape[2])
            self.calls.append((save_kv_cache, sap_context.collect))
            return out.transpose(1, 2).reshape_as(z), kv_cache

    torch.manual_seed(91)
    backbone = Backbone()
    ctl = SapController(backbone, SapConfig(layers=(1,), dim=16, ray_dim=6, heads=2,
        address_arch='multimodal', memory_arch='swiglu', memory_hidden_dim=12,
        support_tokens=8, normalized_value=True))
    episode = SapEpisodeModel(backbone, ctl, lambda_exact=.1,
                              cache_factory=lambda _: ([None] * 4, lambda i: None))
    z, cam, rays = torch.randn(1, 12, 13, 1, 2), torch.zeros(1, 13, 20), torch.randn(1, 6, 13, 1, 2)
    text, mask = torch.randn(1, 4, 12), torch.ones(1, 4, dtype=torch.bool)
    labels = dict(positive=torch.arange(6), valid=torch.ones(6, dtype=torch.bool),
                  support_instance=torch.ones(10, 1, 2, dtype=torch.long),
                  query_instance=torch.ones(3, 1, 2, dtype=torch.long))
    loss, metrics = episode(z, cam, rays, text, mask, 'scene', seed=9, supervision=labels)
    loss.backward()
    module = ctl.modules[1]
    assert ctl.state[1].updates == 3
    assert backbone.calls == [(True, True)] * 3 + [(False, False)]
    assert module.gate.grad.abs().sum() > 0
    assert module.address.writer.weight.grad.abs().sum() > 0
    assert module.memory.initial_weight.grad.abs().sum() > 0
    assert module.memory.log_lr.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.blocks[0].norm2.parameters())
    assert metrics['matched_queries'] == 6
    assert metrics['exact_ce'] > 0
