import torch
from torch import nn


def test_layer_trace_aligns_full_prefix_and_cached_chunk():
    from worldttt.check_cache import LayerTrace, error_stats

    class Stage(nn.Module):
        def forward(self, x):
            return x + 1, None

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn, self.mlp = Stage(), Stage()

        def forward(self, x):
            x, _ = self.attn(x)
            x, _ = self.mlp(x)
            return x

    model = nn.Module()
    model.blocks = nn.ModuleList([Block()])
    full = torch.arange(8.).reshape(1, 8, 1)
    with LayerTrace(model, total_frames=4, current_frames=2) as trace:
        model.blocks[0](full)
    assert torch.equal(trace.values['1.attn'], full[:, -4:] + 1)
    assert torch.equal(trace.values['1.mlp'], full[:, -4:] + 2)
    assert torch.equal(trace.values['1.block'], full[:, -4:] + 2)
    assert torch.equal(trace.values['1.input'], full[:, -4:])
    assert not model.blocks[0]._forward_pre_hooks
    assert not model.blocks[0]._forward_hooks
    assert not model.blocks[0].attn._forward_hooks
    assert not model.blocks[0].mlp._forward_hooks

    with LayerTrace(model, total_frames=2, current_frames=2) as cached:
        model.blocks[0](full[:, -4:])
    assert error_stats(trace.values['1.block'], cached.values['1.block'])['relative_rms'] == 0
