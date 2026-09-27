import pytest
import torch
from torch import nn


def _bank(capacity=4):
    from worldttt.sap_binding.memory import BindingBankState
    return BindingBankState.new(1, capacity, 8, 8, 4, 'cpu')


def test_bank_is_bounded_and_protects_first_chunk_anchors():
    bank = _bank()
    def write(chunk, count):
        x = torch.randn(1, count, 8)
        bank.commit(x, x + chunk, torch.randn(1, count, 4), torch.ones(1, count),
                    chunk=chunk, protected_budget=2, seed=9)
    write(0, 4)
    protected_keys = bank.keys[0, bank.protected[0]].clone()
    for chunk in range(1, 8):
        write(chunk, 3)
    assert bank.valid.sum() == 4 and bank.protected.sum() == 2
    torch.testing.assert_close(bank.keys[0, bank.protected[0]], protected_keys)
    assert bank.seen.item() == 25


def test_explicit_read_depends_on_correct_key_value_binding():
    from worldttt.sap_binding.memory import explicit_read
    bank = _bank(capacity=3)
    keys = torch.eye(8)[:3][None]
    values = torch.stack((torch.ones(8), torch.ones(8) * 2, torch.ones(8) * 3))[None]
    bank.commit(keys, values, torch.zeros(1, 3, 4), torch.ones(1, 3),
                chunk=0, protected_budget=0, seed=1)
    projection = nn.Linear(4, 8, bias=False); projection.weight.data.zero_()
    correct, _ = explicit_read(bank, keys[:, :1], torch.zeros(1, 1, 4), projection, projection,
                               heads=2, topk=1)
    shuffled, _ = explicit_read(bank, keys[:, :1], torch.zeros(1, 1, 4), projection, projection,
                                heads=2, topk=1, shuffle_value=True)
    assert not torch.equal(correct, shuffled)
    torch.testing.assert_close(correct, torch.ones_like(correct))


def test_nonfinite_bank_write_is_transactionally_rejected():
    bank = _bank()
    before = bank.clone()
    with pytest.raises(FloatingPointError):
        bank.commit(torch.full((1, 1, 8), float('nan')), torch.zeros(1, 1, 8),
                    torch.zeros(1, 1, 4), torch.ones(1, 1), chunk=0,
                    protected_budget=1, seed=1)
    torch.testing.assert_close(bank.keys, before.keys)
    assert torch.equal(bank.valid, before.valid)


def test_topk_handles_one_anchor_and_an_empty_cfg_branch():
    from worldttt.sap_binding.memory import BindingBankState, explicit_read

    bank = BindingBankState.new(2, 8, 8, 8, 4, 'cpu')
    bank.keys[0, 0, 0] = 1
    bank.values[0, 0] = 2
    bank.valid[0, 0] = True
    projection = nn.Linear(4, 8, bias=False)
    out, stats = explicit_read(bank, torch.randn(2, 3, 8), torch.randn(2, 3, 4),
                               projection, projection, heads=2, topk=8)
    assert torch.isfinite(out).all()
    assert torch.isfinite(stats['margin']).all()
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]))
    assert stats['has_memory'][:, 0, 0].tolist() == [1, 0]


def test_mean_value_control_reads_only_the_historical_mean():
    from worldttt.sap_binding.config import BindingConfig
    from worldttt.sap_binding.memory import BindingBankState, BindingState
    from worldttt.sap_binding.model import BindingBlock
    from worldttt.sap_ttt.memory import SapMemoryState

    config = BindingConfig(layers=(1,), address_dim=8, value_dim=8, ray_dim=4,
                           latent_dim=4, heads=2, topk=2, capacity=4,
                           protected_anchors=1, support_tokens=2, address_depth=1,
                           fast_hidden_dim=6)
    module = BindingBlock(8, 4, config)
    bank = BindingBankState.new(1, 4, 8, 8, 4, 'cpu')
    bank.valid[0, :2] = True
    bank.values[0, 0] = 1
    bank.values[0, 1] = 3
    state = BindingState('test', bank, SapMemoryState.new(module.fast, 'test'))
    value, _, _ = module.read(state, torch.randn(1, 3, 8), torch.zeros(1, 3, 4),
                              torch.zeros(1, 3, 1), mean_value=True)
    torch.testing.assert_close(value, torch.full_like(value, 2))


def test_binding_trust_region_shrinks_fast_update_toward_initialization():
    from worldttt.sap_binding.model import BindingFastMemory

    torch.manual_seed(5)
    memory = BindingFastMemory(8, 2, 6, .1, 100.)
    weight = (memory.initial_weight[None].detach() + .5).requires_grad_(True)
    key = torch.randn(1, 3, 8); value = torch.randn(1, 3, 8)
    unconstrained, _ = super(BindingFastMemory, memory).update(weight, key, value)
    constrained, _ = memory.update(weight, key, value)
    center = memory.initial_weight[None]
    assert (constrained - center).norm() < (unconstrained - center).norm()
