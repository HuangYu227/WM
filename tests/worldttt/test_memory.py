import copy
import importlib.util

import pytest
import torch


def test_package_exists():
    assert importlib.util.find_spec('worldttt') is not None, 'WorldTTT implementation is missing'


def setup_memory(mode='noise_ttt', batch=2):
    from worldttt.memory import TTTConfig, FastMemory, WorldTTTState, Features
    torch.manual_seed(7)
    cfg = TTTConfig(mode=mode, layers=(1,), input_dim=4, hidden_dim=6,
                    support_tokens=5, anchor_capacity=8, inner_lr=0.05)
    memory = FastMemory(heads=2, head_dim=3, config=cfg)
    state = WorldTTTState('scene', {1: memory.initial_weights(batch, training=True)}, seed=9)
    x = torch.randn(batch, 9, 6)
    m = torch.randn_like(x)
    features = Features(x, m, torch.randn(batch, 9, 20), torch.ones(batch, 9, 1) * 0.5,
                        k=torch.randn_like(x), v=torch.randn_like(x))
    return memory, state, features


def test_update_persists_meta_gradients_and_sample_isolation():
    from worldttt.memory import adapt_after_chunk
    memory, state, f = setup_memory()
    before = memory(f, state.weights[1])
    target = f.m + 0.4
    target[1] = f.m[1]
    metrics = adapt_after_chunk({1: memory}, state, {1: f}, {1: target}, 0, training=True)
    after = memory(f, state.weights[1])
    assert metrics['committed']
    assert after[0].abs().sum() > before[0].abs().sum()
    assert torch.equal(after[1], before[1])
    after.square().mean().backward()
    assert memory.input_proj.grad is not None and memory.input_proj.grad.abs().sum() > 0
    assert memory.w_out.grad is not None
    assert state.last_chunk == 0
    with pytest.raises(ValueError, match='chunk'):
        adapt_after_chunk({1: memory}, state, {1: f}, {1: target}, 0)


@pytest.mark.parametrize('mode', ['off', 'frozen'])
def test_disabled_updates(mode):
    from worldttt.memory import adapt_after_chunk
    memory, state, f = setup_memory(mode)
    before = [w.detach().clone() for w in state.weights[1]]
    adapt_after_chunk({1: memory}, state, {1: f}, {1: f.m + 1}, 0)
    assert all(torch.equal(a, b) for a, b in zip(before, state.weights[1]))


def test_nonfinite_update_is_transactional():
    from worldttt.memory import adapt_after_chunk
    memory, state, f = setup_memory()
    before = [w.detach().clone() for w in state.weights[1]]
    result = adapt_after_chunk({1: memory}, state, {1: f}, {1: f.m * float('nan')}, 0)
    assert not result['committed']
    assert all(torch.equal(a, b) for a, b in zip(before, state.weights[1]))


def test_bounded_anchors_and_exact_resume(tmp_path):
    from worldttt.memory import adapt_after_chunk, WorldTTTState
    memory, state, f = setup_memory()
    for c in range(5):
        adapt_after_chunk({1: memory}, state, {1: f}, {1: f.m + 0.2}, c)
    assert state.anchors[1].features.x.shape[1] <= 8
    path = tmp_path / 'state.pt'
    state.save_state(path)
    resumed = WorldTTTState.load_state(path)
    adapt_after_chunk({1: memory}, state, {1: f}, {1: f.m + 0.3}, 5)
    adapt_after_chunk({1: memory}, resumed, {1: f}, {1: f.m + 0.3}, 5)
    assert all(torch.equal(a, b) for a, b in zip(state.weights[1], resumed.weights[1]))
    assert torch.equal(state.anchors[1].features.x, resumed.anchors[1].features.x)


def test_kv_objective_uses_keys_and_values():
    from worldttt.memory import adapt_after_chunk
    memory, state, f = setup_memory('kv_ttt')
    _, other, _ = setup_memory('kv_ttt')
    adapt_after_chunk({1: memory}, state, {1: f}, {1: f.m + 10}, 0)
    adapt_after_chunk({1: memory}, other, {1: f}, {1: f.m - 10}, 0)
    assert all(torch.equal(a, b) for a, b in zip(state.weights[1], other.weights[1]))


def test_meta_gradient_matches_finite_difference():
    from worldttt.memory import WorldTTTState, adapt_after_chunk
    memory, _, f = setup_memory(batch=1)
    memory.config.grad_clip = 1000.  # keep this test away from clipping boundaries
    def objective():
        state = WorldTTTState('gradcheck', {1: memory.initial_weights(1, training=True)}, seed=9)
        adapt_after_chunk({1: memory}, state, {1: f}, {1: f.m + .4}, 0, training=True)
        return (memory(f, state.weights[1]) - .2).square().mean()
    for parameter in (memory.w_out, memory.input_proj):
        gradient, = torch.autograd.grad(objective(), parameter)
        index = tuple(int(i) for i in torch.unravel_index(gradient.abs().argmax(), gradient.shape))
        with torch.no_grad():
            original = parameter[index].item()
            parameter[index] = original + .001
        plus = objective().item()
        with torch.no_grad():
            parameter[index] = original - .001
        minus = objective().item()
        with torch.no_grad():
            parameter[index] = original
        assert gradient[index].item() == pytest.approx((plus - minus) / .002, rel=.02, abs=2e-5)
