import pytest
import torch
from torch import nn
import ast
from pathlib import Path


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = nn.LayerNorm(8)


def controller(mode='online'):
    from worldttt.sap_binding import BindingConfig, BindingController
    model = nn.Module(); model.blocks = nn.ModuleList([ToyBlock()])
    config = BindingConfig(mode=mode, layers=(1,), address_dim=8, value_dim=8,
        ray_dim=4, latent_dim=4, heads=2, topk=2, capacity=6,
        protected_anchors=2, support_tokens=4, address_depth=1, fast_hidden_dim=6)
    return BindingController(model, config)


def inputs():
    torch.manual_seed(7)
    return (torch.randn(1, 18, 8), torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool),
            torch.randn(1, 18, 4), torch.randn(1, 4, 2, 3, 3))


def test_zero_gate_read_is_exact_and_commit_changes_only_future_state():
    ctl = controller(); ctl.reset_episode('one', 1)
    block = ctl.model.blocks[0]; x, text, mask, rays, latent = inputs()
    noisy = ctl.context(torch.tensor([500.]), latent=latent)
    assert torch.equal(noisy.apply(block, x, x, text, mask, rays, 2), x)
    clean = ctl.context(torch.zeros(1), latent=latent, collect=True, chunk=0)
    assert torch.equal(clean.apply(block, x, x, text, mask, rays, 2), x)
    row = ctl.commit(clean, 0)
    assert row['committed'] and ctl.state[1].updates == 1
    assert ctl.state[1].bank.valid.sum() == 4


def test_binding_checkpoint_and_episode_reject_legacy_or_other_adapter(tmp_path):
    ctl = controller(); ctl.base_checkpoint = 'base'; ctl.reset_episode('one', 1)
    x, text, mask, rays, latent = inputs(); block = ctl.model.blocks[0]
    clean = ctl.context(torch.zeros(1), latent=latent, collect=True)
    clean.apply(block, x, x, text, mask, rays, 2); ctl.commit(clean, 0)
    ctl.save_checkpoint(tmp_path / 'adapter.pt'); ctl.save_state(tmp_path / 'state.pt')
    second = controller(); second.base_checkpoint = 'base'
    second.load_checkpoint(tmp_path / 'adapter.pt'); second.load_state(tmp_path / 'state.pt')
    torch.testing.assert_close(second.state[1].bank.values, ctl.state[1].bank.values)
    torch.testing.assert_close(second.state[1].fast.weight, ctl.state[1].fast.weight)
    torch.save({'version': 1, 'modules': {}}, tmp_path / 'legacy.pt')
    with pytest.raises(ValueError, match='not a SAP-Bind'):
        second.load_checkpoint(tmp_path / 'legacy.pt')


def test_episode_load_rejects_malformed_bank_without_changing_state(tmp_path):
    ctl = controller(); ctl.base_checkpoint = 'base'; ctl.reset_episode('one', 1)
    ctl.save_state(tmp_path / 'state.pt')
    payload = torch.load(tmp_path / 'state.pt', weights_only=True)
    payload['states'][1]['bank']['keys'] = torch.zeros(1, 5, 8)
    torch.save(payload, tmp_path / 'bad.pt')
    before = ctl.state[1]
    with pytest.raises(ValueError, match='bank state'):
        ctl.load_state(tmp_path / 'bad.pt')
    assert ctl.state[1] is before


def test_frozen_episode_state_roundtrip(tmp_path):
    ctl = controller('frozen'); ctl.base_checkpoint = 'base'; ctl.reset_episode('one', 1)
    ctl.commit(ctl.context(torch.zeros(1), chunk=0), 0)
    ctl.save_checkpoint(tmp_path / 'adapter.pt'); ctl.save_state(tmp_path / 'state.pt')
    restored = controller('frozen'); restored.base_checkpoint = 'base'
    restored.load_checkpoint(tmp_path / 'adapter.pt'); restored.load_state(tmp_path / 'state.pt')
    assert restored.state[1].last_chunk == 0 and restored.state[1].updates == 0


def test_failed_commit_does_not_modify_fast_or_explicit_state():
    ctl = controller(); ctl.reset_episode('one', 1)
    state = ctl.state[1]
    from types import SimpleNamespace
    bad = SimpleNamespace(features={1: (torch.full((1, 1, 8), float('nan')),
        torch.zeros(1, 1, 8), torch.zeros(1, 1, 4), torch.ones(1, 1))}, source='bad')
    before_weight, before_bank = state.fast.weight.clone(), state.bank.clone()
    row = ctl.commit(bad, 0)
    assert not row['committed']
    torch.testing.assert_close(state.fast.weight, before_weight)
    torch.testing.assert_close(state.bank.keys, before_bank.keys)


def test_binding_hook_is_after_cross_attention_and_exclusive_with_legacy_sap():
    path = Path(__file__).resolve().parents[2] / 'diffusion/model/nets/sana_multi_scale_video_camctrl.py'
    source = path.read_text(encoding='utf-8'); tree = ast.parse(source)
    block = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                 node.name == 'SanaVideoMSCamCtrlBlock')
    for name in ('forward_frame_aware', 'forward'):
        method = next(node for node in block.body if isinstance(node, ast.FunctionDef) and node.name == name)
        lines = ast.get_source_segment(source, method).splitlines()
        cross = next(i for i, line in enumerate(lines) if 'self.cross_attn(x, y' in line)
        binding = next(i for i, line in enumerate(lines) if 'binding_context.apply(' in line)
        ffn = next(i for i, line in enumerate(lines) if 'self.norm2(x)' in line)
        assert cross < binding < ffn
    assert source.count('Legacy SAP and SAP-Bind cannot run in the same forward') == 2


def test_cfg_branches_are_isolated_and_read_is_pure():
    ctl = controller(); ctl.reset_episode('cfg', 2)
    block = ctl.model.blocks[0]
    x, text, mask, rays, latent = inputs()
    x = torch.cat((x, x + 3), 0); text = text.expand(2, -1, -1)
    mask = mask.expand(2, -1); rays = torch.cat((rays, rays + 2), 0)
    latent = torch.cat((latent, latent + 4), 0)
    clean = ctl.context(torch.zeros(2), latent=latent, collect=True)
    clean.apply(block, x, x, text, mask, rays, 2); ctl.commit(clean, 0)
    state = ctl.state[1]
    assert not torch.equal(state.bank.keys[0], state.bank.keys[1])
    before = {name: tensor.clone() for name, tensor in state.bank.state_dict().items()}
    noisy = ctl.context(torch.full((2,), 500.), latent=latent)
    noisy.apply(block, x, x, text, mask, rays, 2)
    for name, tensor in before.items():
        torch.testing.assert_close(state.bank.state_dict()[name], tensor)
