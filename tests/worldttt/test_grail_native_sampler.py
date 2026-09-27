"""Controller transactions and sampler wiring without SANA CUDA imports."""

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from worldttt.associative_ttt import AssociativeTTTConfig, HoldoutBatch
from worldttt.grail_native import (
    GrailNativeController, NATIVE_COORDINATE_CONVENTION, TARGET_LAYERS, attach_grail_native,
)


class _Block(nn.Module):
    hidden_size = 4


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(17))
        self.patch_size = (1, 1, 1)
        self.softmax_every_n = 4
        self.camctrl_layers_num = 16


def _camera():
    return torch.cat((torch.eye(4).reshape(16), torch.tensor([1., 1., 0.5, 0.5]))).view(1, 1, 20)


def _setup():
    torch.manual_seed(11)
    cfg = AssociativeTTTConfig(key_dim=3, value_dim=2, geometry_dim=30, capacity=8, topk=1,
                               geometry_metric='ray_point', coordinate_convention=NATIVE_COORDINATE_CONVENTION)
    controller = GrailNativeController(4, cfg, mode="online")
    model = _Model()
    attach_grail_native(model, controller)
    controller.reset_episode("cfg", 1, metadata={"base_checkpoint_hash": "test"})
    return controller, model.blocks[2]


def _complete_clean(controller, clean, x, thw=(1, 1, 1)):
    for layer in TARGET_LAYERS:
        if layer in clean.call_counts:
            continue
        block = _Block()
        block.grail_layer_index = layer
        block.grail_patch_size = 1
        clean.apply(block, x, _camera(), thw, None)


def test_cfg_reads_both_branches_but_only_conditional_clean_features_write():
    controller, block = _setup()
    x = torch.tensor([[[1., 0., 0., 0.]], [[0., 1., 0., 0.]]])
    clean = controller.context("online", collect=True, chunk_id=0, cfg_conditional_start=1,
                               real_frame_indices=(0,))
    clean.apply(block, x, _camera(), (1, 1, 1), None)
    assert clean.call_counts == {2: 1}
    assert len(clean.observations) == 1
    _, expected_key, expected_value, _ = controller.writer.encode(
        x[1:], _camera(), (1, 1, 1), 1, torch.ones(1, 1, 1), torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(clean.observations[0].keys, expected_key)
    torch.testing.assert_close(clean.observations[0].values, expected_value)
    assert clean.observations[0].source_real.tolist() == [[True]]
    _complete_clean(controller, clean, x)
    assert controller.commit_clean(clean, 0)["committed"]
    assert controller.state.batch_size == 1
    assert int(controller.state.last_committed_chunk[0]) == 0

    query = controller.context("online", cfg_conditional_start=1)
    output = query.apply(block, x, _camera(), (1, 1, 1), None)
    assert not torch.equal(output[0], x[0])
    assert not torch.equal(output[1], x[1])
    assert query.observations == []
    with pytest.raises(ValueError):
        controller.commit_clean(clean, 0)


def test_off_frozen_online_are_distinct_state_paths():
    controller, block = _setup()
    x = torch.randn(1, 1, 4)
    off = controller.context("off")
    frozen = controller.context("frozen")
    torch.testing.assert_close(off.apply(block, x, _camera(), (1, 1, 1), None), x)
    torch.testing.assert_close(frozen.apply(block, x, _camera(), (1, 1, 1), None), x)
    assert controller.state.last_committed_chunk.tolist() == [-1]
    with pytest.raises(ValueError):
        controller.context("frozen", collect=True, chunk_id=0)
    clean = controller.context("online", collect=True, chunk_id=0)
    clean.apply(block, x, _camera(), (1, 1, 1), None)
    _complete_clean(controller, clean, x)
    assert controller.commit_clean(clean, 0)["committed"]
    assert controller.state.last_committed_chunk.tolist() == [0]
    assert not torch.equal(controller.context("frozen").apply(block, x, _camera(), (1, 1, 1), None), x)


def test_cfg_bad_batch_and_repeated_clean_are_rejected():
    controller, block = _setup()
    x = torch.randn(2, 1, 4)
    with pytest.raises(ValueError):
        controller.context("online").apply(block, x, _camera(), (1, 1, 1), None)
    with pytest.raises(ValueError, match="CFG"):
        controller.context("online", cfg_conditional_start=1).apply(
            block, x[:1], _camera(), (1, 1, 1), None)
    clean = controller.context("online", collect=True, chunk_id=0, cfg_conditional_start=1)
    clean.apply(block, x, _camera(), (1, 1, 1), None)
    with pytest.raises(ValueError, match="repeated"):
        clean.apply(block, x, _camera(), (1, 1, 1), None)
    with pytest.raises(ValueError, match="repeated|once|duplicate"):
        controller.commit_clean(clean, 0)


def test_clean_support_is_bounded_by_layer_budget():
    controller, block = _setup()
    x = torch.randn(1, 6, 4)
    clean = controller.context("online", collect=True, chunk_id=0)
    clean.apply(block, x, _camera(), (1, 1, 6), None)
    assert clean.observations[0].keys.shape[1] == 6  # one writer, not capacity divided among eight layers


def test_partial_clean_pass_cannot_commit():
    controller, block = _setup()
    clean = controller.context("online", collect=True, chunk_id=0)
    clean.apply(block, torch.randn(1, 1, 4), _camera(), (1, 1, 1), None)
    with pytest.raises(ValueError, match="eight|layers|missing"):
        controller.commit_clean(clean, 0)
    assert controller.state.last_committed_chunk.tolist() == [-1]


def test_controller_holdout_rollback_keeps_chunk_cursor():
    controller, block = _setup()
    x = torch.randn(1, 1, 4)
    clean = controller.context("online", collect=True, chunk_id=0)
    clean.apply(block, x, _camera(), (1, 1, 1), None)
    _complete_clean(controller, clean, x)
    first = clean.observations[0]
    holdout = HoldoutBatch(first.keys, -10 * first.values, first.geometry, first.confidence)
    report = controller.commit_clean(clean, 0, holdout=holdout)
    assert report["rolled_back"]
    assert controller.state.last_committed_chunk.tolist() == [-1]


def test_attach_moves_heads_to_base_model_dtype():
    model = _Model().double()
    model.register_parameter("base_weight", nn.Parameter(torch.ones(1, dtype=torch.float64)))
    controller = GrailNativeController(4)
    attach_grail_native(model, controller)
    assert controller.writer.visual.weight.dtype == torch.float64


def test_sampler_has_reset_read_clean_commit_wiring():
    source = Path(__file__).resolve().parents[2] / "diffusion/scheduler/self_forcing_flow_euler_sampler.py"
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    sample = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "SelfForcingFlowEulerCamCtrl"
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "sample_chunks")
    body = ast.get_source_segment(text, sample)
    assert "grail.reset_episode(" in body
    assert body.count('"grail_context"') >= 2
    assert "grail.commit_clean(" in body
    assert 'if not grail_report["committed"]:' in body
    assert "load_grail_rollout(" in body
    assert "save_grail_rollout(" in body
    assert body.index("grail.reset_episode(") < body.index("grail.commit_clean(")
