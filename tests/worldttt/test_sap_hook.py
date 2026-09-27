import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = nn.LayerNorm(8)


def test_sap_hook_zero_gate_and_chunk_commit():
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    ctl = SapController(model, SapConfig(layers=(1,), dim=4, ray_dim=6,
                                         support_tokens=3, mode='online'))
    ctl.reset_episode('one', batch=2)
    block = model.blocks[0]
    x = torch.randn(2, 5, 8)
    text = torch.randn(2, 3, 8)
    rays = torch.randn(2, 5, 6)
    ctx = ctl.context(torch.full((2, 1, 1), 500.))
    assert torch.equal(ctx.apply(block, x, x, text, [3, 3], rays, 1), x)
    assert ctl.state[1].updates == 0
    clean = ctl.context(torch.zeros(2), collect=True, chunk=0,
                        source='reference_plus_generated')
    clean.apply(block, x, x, text, [3, 3], rays, 1)
    row = ctl.commit(clean, 0)
    assert row['committed']
    assert row['source'] == 'reference_plus_generated'
    assert ctl.state[1].updates == 1
    assert not torch.equal(ctl.state[1].weight[0], ctl.state[1].weight[1])


def test_sap_hook_is_after_cross_attention_before_ffn():
    path = Path(__file__).resolve().parents[2] / 'diffusion/model/nets/sana_multi_scale_video_camctrl.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SanaVideoMSCamCtrlBlock')
    for name in ('forward_frame_aware', 'forward'):
        method = next(n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == name)
        lines = ast.get_source_segment(path.read_text(encoding='utf-8'), method).splitlines()
        cross = next(i for i, line in enumerate(lines) if 'self.cross_attn(x, y' in line)
        sap = next(i for i, line in enumerate(lines) if 'sap_context.apply(' in line)
        ffn = next(i for i, line in enumerate(lines) if 'self.norm2(x)' in line)
        assert cross < sap < ffn


def test_hook_skips_unselected_sana_blocks():
    path = Path(__file__).resolve().parents[2] / 'diffusion/model/nets/sana_multi_scale_video_camctrl.py'
    source = path.read_text(encoding='utf-8')
    assert source.count("if sap_context is not None and hasattr(self, 'sap_ttt'):") >= 2


def test_raw_feature_capture_is_read_only_and_excludes_supervision():
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    ctl = SapController(model, SapConfig(layers=(1,), dim=4, ray_dim=6,
                                         mode='frozen'))
    ctl.reset_episode('probe', 1)
    block = model.blocks[0]
    x = torch.randn(1, 5, 8)
    text = torch.randn(1, 2, 8)
    ctx = ctl.context(torch.tensor([500.]), record_raw=True)
    weight = ctl.state[1].weight.clone()
    out = ctx.apply(block, x, x, text, [2], torch.randn(1, 5, 6), 1)
    assert torch.equal(out, x)
    assert torch.equal(ctl.state[1].weight, weight)
    assert set(ctx.raw_features[1]) == {'visual', 'post', 'text', 'text_mask',
                                        'rays', 'sigma'}


def test_raw_feature_capture_preserves_large_finite_bfloat16_values():
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    ctl = SapController(model, SapConfig(layers=(1,), dim=4, ray_dim=6,
                                         mode='frozen'))
    ctl.reset_episode('large-feature', 1)
    x = torch.full((1, 5, 8), 100000., dtype=torch.bfloat16)
    text = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    rays = torch.ones(1, 5, 6, dtype=torch.bfloat16)
    ctx = ctl.context(torch.tensor([500.]), record_raw=True)
    ctx.apply(model.blocks[0], x, x, text, [2], rays, 1)
    assert ctx.raw_features[1]['visual'].dtype == torch.bfloat16
    assert torch.isfinite(ctx.raw_features[1]['visual']).all()
    torch.testing.assert_close(ctx.raw_features[1]['visual'], x.cpu())


def test_raw_feature_capture_rejects_nonfinite_backbone_output():
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    ctl = SapController(model, SapConfig(layers=(1,), dim=4, ray_dim=6,
                                         mode='frozen'))
    ctl.reset_episode('nonfinite-feature', 1)
    x = torch.zeros(1, 5, 8, dtype=torch.bfloat16)
    x[0, 0, 0] = float('inf')
    text = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    ctx = ctl.context(torch.tensor([500.]), record_raw=True)
    with pytest.raises(FloatingPointError, match='layer 1 raw visual.*before capture'):
        ctx.apply(model.blocks[0], x, x, text, [2], None, 1)


def test_meta_commit_keeps_gradient_through_completed_support():
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    ctl = SapController(model, SapConfig(layers=(1,), dim=4, ray_dim=6,
                                         support_tokens=3, mode='online'))
    ctl.reset_episode('meta', 1, training=True)
    block = model.blocks[0]
    x = torch.randn(1, 5, 8)
    text = torch.randn(1, 2, 8)
    rays = torch.randn(1, 5, 6)
    with torch.no_grad():
        support = ctl.context(torch.zeros(1), collect=True, chunk=0, training=True)
        support.apply(block, x, x, text, [2], rays, 1)
    ctl.commit(support, 0, training=True)
    query = ctl.context(torch.tensor([500.]), training=True)
    out = query.apply(block, x, x, text, [2], rays, 1)
    out.square().mean().backward()
    assert block.sap_ttt.gate.grad is not None
    assert block.sap_ttt.memory.initial_weight.grad is not None


def test_query_capture_preserves_gradients_and_support_indices():
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    ctl = SapController(model, SapConfig(layers=(1,), dim=4, ray_dim=6,
                                         support_tokens=3, mode='online'))
    ctl.reset_episode('meta', 1, training=True)
    block = model.blocks[0]
    x = torch.randn(1, 5, 8)
    text = torch.randn(1, 2, 8)
    rays = torch.randn(1, 5, 6)
    clean = ctl.context(torch.zeros(1), collect=True, chunk=0, training=True)
    clean.apply(block, x, x, text, [2], rays, 1)
    assert clean.support_indices[1].shape == (3,)
    noisy = ctl.context(torch.tensor([500.]), record_query=True, training=True)
    noisy.apply(block, x, x, text, [2], rays, 1)
    assert noisy.query_addresses[1].shape == (1, 5, 4)
    assert noisy.query_addresses[1].requires_grad


def test_controller_state_save_restore_keeps_cfg_branches(tmp_path):
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    config = SapConfig(layers=(1,), dim=4, ray_dim=6, support_tokens=3)
    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    ctl = SapController(model, config)
    ctl.reset_episode('scene', 2)
    with torch.no_grad():
        ctl.state[1].weight[0, 0, 0] = 1.
    path = tmp_path / 'sap_state.pt'
    ctl.save_state(path)
    second = nn.Module()
    second.blocks = nn.ModuleList([ToyBlock()])
    restored = SapController(second, config)
    ctl.save_checkpoint(tmp_path / 'adapter.pt')
    restored.load_checkpoint(tmp_path / 'adapter.pt')
    restored.load_state(path)
    assert restored.state[1].episode == 'scene'
    assert restored.state[1].weight.shape[0] == 2
    torch.testing.assert_close(restored.state[1].weight, ctl.state[1].weight)


def test_unpack_text_accepts_sana_non_xformers_four_dimensional_layout():
    from worldttt.sap_ttt.runtime import unpack_text

    text = torch.randn(2, 1, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.int16)
    unpacked, valid = unpack_text(text, mask, 2)
    assert unpacked.shape == (2, 5, 8)
    assert valid.tolist() == [[True, True, True, False, False],
                              [True, True, False, False, False]]


def test_sap_read_and_query_text_ablations_do_not_change_clean_write():
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    config = SapConfig(layers=(1,), dim=4, ray_dim=6, support_tokens=3)
    ctl = SapController(model, config)
    ctl.reset_episode('ablation', 1)
    block = model.blocks[0]
    x = torch.randn(1, 5, 8)
    text = torch.randn(1, 4, 8)
    rays = torch.randn(1, 5, 6)
    normal = ctl.context(torch.zeros(1), collect=True, record_query=True)
    normal_output = normal.apply(block, x, x, text, [4], rays, 1)
    config.shuffle_query_text = True
    shuffled = ctl.context(torch.zeros(1), collect=True, record_query=True)
    shuffled.apply(block, x, x, text, [4], rays, 1)
    torch.testing.assert_close(normal.features[1][0], shuffled.features[1][0])
    assert not torch.equal(normal.query_addresses[1], shuffled.query_addresses[1])
    config.read_enabled = False
    disabled = ctl.context(torch.zeros(1), collect=True)
    assert torch.equal(disabled.apply(block, x, x, text, [4], rays, 1), x)
    torch.testing.assert_close(normal.features[1][0], disabled.features[1][0])
    assert torch.equal(normal_output, x)


def test_adapter_load_rejects_address_protocol_mismatch(tmp_path):
    import pytest
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    model = nn.Module()
    model.blocks = nn.ModuleList([ToyBlock()])
    ctl = SapController(model, SapConfig(layers=(1,), dim=4, ray_dim=6,
                                         address_mode='selective_geometry'))
    ctl.base_checkpoint = 'base'
    saved = tmp_path / 'adapter.pt'
    ctl.save_checkpoint(saved)
    second = nn.Module()
    second.blocks = nn.ModuleList([ToyBlock()])
    other = SapController(second, SapConfig(layers=(1,), dim=4, ray_dim=6,
                                            address_mode='vision'))
    other.base_checkpoint = 'base'
    with pytest.raises(ValueError, match='protocol'):
        other.load_checkpoint(saved)
