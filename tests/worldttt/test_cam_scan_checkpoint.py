"""Streaming camera-scan rematerialization keeps gradients while saving graph state."""

import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks
from torch.utils import checkpoint as checkpoint_module

from diffusion.model.ops import fused_streaming


@pytest.mark.parametrize('save_kv_cache', [False, True])
def test_cached_camera_scan_checkpoint_preserves_gradients_and_saves_graph(monkeypatch, save_kv_cache):
    torch.manual_seed(7)
    frames, spatial = 2, 3
    base = [torch.randn(1, 2, 4, frames * spatial) for _ in range(3)]
    base += [torch.rand(1, 2, frames, spatial), torch.rand(1, 2, frames) * 0.2 + 0.7]
    initial = torch.randn(1, 2, 4, 4) * 0.1

    def run():
        saved_bytes = 0

        def pack(tensor):
            nonlocal saved_bytes
            saved_bytes += tensor.numel() * tensor.element_size()
            return tensor

        inputs = [tensor.clone().requires_grad_() for tensor in base]
        with saved_tensors_hooks(pack, lambda tensor: tensor):
            output, final_state = fused_streaming._cam_main_triton(
                *inputs, initial, save_kv_cache, frames, spatial)
        gradients = torch.autograd.grad(output.square().mean(), inputs)
        return output, final_state, gradients, saved_bytes

    original = checkpoint_module.checkpoint
    monkeypatch.setattr(checkpoint_module, 'checkpoint',
                        lambda function, *args, **kwargs: function(*args))
    direct = run()
    monkeypatch.setattr(checkpoint_module, 'checkpoint', original)
    checked = run()

    torch.testing.assert_close(checked[0], direct[0], rtol=0, atol=0)
    torch.testing.assert_close(checked[1], direct[1], rtol=0, atol=0)
    for actual, expected in zip(checked[2], direct[2]):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert checked[3] < direct[3]
