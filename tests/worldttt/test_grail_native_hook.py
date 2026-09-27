"""Opt-in hook checks without importing SANA's unavailable CUDA packages."""

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from worldttt.associative_ttt import AssociativeTTTConfig
from worldttt.grail_native import (
    GrailNativeController, NATIVE_COORDINATE_CONVENTION, TARGET_GDN_LAYERS, TARGET_LAYERS, TARGET_SOFTMAX_LAYERS,
    attach_grail_native,
)


def _camera():
    pose = torch.eye(4).reshape(16)
    return torch.cat((pose, torch.tensor([1., 1., 0.5, 0.5]))).view(1, 1, 20)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(17))
        self.patch_size = (1, 1, 1)
        self.softmax_every_n = 4
        self.camctrl_layers_num = 16


def _controller():
    return GrailNativeController(4, AssociativeTTTConfig(
        key_dim=3, value_dim=2, geometry_dim=30, capacity=16, topk=1,
        geometry_metric='ray_point', coordinate_convention=NATIVE_COORDINATE_CONVENTION,
        min_read_confidence=0,
    ))


def test_attach_marks_only_target_blocks_and_registers_one_controller():
    model = _Model()
    controller = _controller()
    assert not any("grail" in name for name, _ in model.named_parameters())
    attach_grail_native(model, controller)
    assert model.worldttt_grail_controller is controller
    assert {i for i, block in enumerate(model.blocks) if hasattr(block, "grail_layer_index")} == {2, 6, 10, 14, 3, 7, 11, 15}
    assert sum(name.startswith("worldttt_grail_controller.readers.") for name, _ in model.named_parameters()) > 0
    assert not any("grail" in name for name, _ in _Model().named_parameters())


def test_context_off_empty_and_online_read_paths():
    torch.manual_seed(4)
    controller = _controller()
    model = _Model()
    attach_grail_native(model, controller)
    controller.reset_episode("test", 1, metadata={})
    block = model.blocks[2]
    x = torch.randn(1, 1, 4)
    off = controller.context("off")
    assert torch.equal(off.apply(block, x, _camera(), (1, 1, 1), None), x)
    frozen = controller.context("frozen")
    assert torch.equal(frozen.apply(block, x, _camera(), (1, 1, 1), None), x)
    assert frozen.call_counts[2] == 1

    clean = controller.context("online", collect=True, chunk_id=0, real_frame_indices=(0,))
    clean.apply(block, x, _camera(), (1, 1, 1), None)
    assert clean.call_counts[2] == 1
    for layer in TARGET_LAYERS:
        if layer != 2:
            clean.apply(model.blocks[layer], x, _camera(), (1, 1, 1), None)
    assert controller.commit_clean(clean, 0)["committed"]
    after = controller.context("frozen").apply(block, x, _camera(), (1, 1, 1), None)
    assert not torch.equal(after, x)
    assert torch.isfinite(after).all()


def test_attach_fails_if_block_layout_cannot_support_target_layers():
    model = _Model()
    model.blocks = nn.ModuleList(_Block() for _ in range(15))
    with pytest.raises(ValueError):
        attach_grail_native(model, _controller())


def test_attach_rejects_hybrid_schedule_that_changes_target_attention_type():
    model = _Model()
    model.softmax_every_n = 5
    with pytest.raises(ValueError, match="softmax"):
        attach_grail_native(model, _controller())


def test_both_native_forward_sites_place_grail_between_cross_attention_and_ffn():
    source = Path(__file__).resolve().parents[2] / "diffusion/model/nets/sana_multi_scale_video_camctrl.py"
    cls = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "SanaVideoMSCamCtrlBlock")
    for method_name in ("forward_frame_aware", "forward"):
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
        body = ast.get_source_segment(source.read_text(encoding="utf-8"), method)
        assert body.index("self.cross_attn(") < body.index("grail_context.apply(") < body.index("mlp_kwargs =")


def test_zero_based_target_indices_match_native_every_fourth_softmax_schedule():
    source = Path(__file__).resolve().parents[2] / "diffusion/model/nets/sana_multi_scale_video_camctrl.py"
    function = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_build_camctrl_cls_list")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    class GDN:
        pass
    class Softmax:
        pass
    schedule = namespace["_build_camctrl_cls_list"](GDN, Softmax, 16, 16, 4)
    assert TARGET_GDN_LAYERS == (2, 6, 10, 14)
    assert TARGET_SOFTMAX_LAYERS == (3, 7, 11, 15)
    assert all(schedule[i] is GDN for i in TARGET_GDN_LAYERS)
    assert all(schedule[i] is Softmax for i in TARGET_SOFTMAX_LAYERS)
