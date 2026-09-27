import torch
from torch import nn


def test_binding_four_chunk_episode_meta_gradient_and_flow_gate():
    from worldttt.sap_binding import BindingConfig, BindingController
    from worldttt.sap_binding.episode import BindingEpisodeModel

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__(); block = nn.Module(); block.norm2 = nn.LayerNorm(8)
            self.blocks = nn.ModuleList([block]); self.calls = []
        def forward(self, z, times, *, y, mask, binding_context, chunk_plucker,
                    kv_cache, save_kv_cache, **kwargs):
            raw = z.flatten(2).transpose(1, 2)
            x = torch.cat((raw, raw), -1)
            out = binding_context.apply(self.blocks[0], x, x, y, mask, chunk_plucker, z.shape[2])
            self.calls.append((save_kv_cache, binding_context.collect))
            return out[..., :4].transpose(1, 2).reshape_as(z), kv_cache

    backbone = Backbone()
    config = BindingConfig(layers=(1,), address_dim=8, value_dim=8, ray_dim=4,
        latent_dim=4, heads=2, topk=2, capacity=8, protected_anchors=2,
        support_tokens=4, address_depth=1, fast_hidden_dim=6)
    ctl = BindingController(backbone, config)
    episode = BindingEpisodeModel(backbone, ctl,
        cache_factory=lambda _: ([None] * 4, lambda i: None))
    z = torch.randn(1, 4, 13, 1, 2)
    loss, metrics = episode(z, torch.zeros(1, 13, 2), torch.randn(1, 4, 13, 1, 2),
                            torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool),
                            'scene', seed=4)
    loss.backward()
    module = ctl.modules[1]
    assert metrics['flow_mse'] > 0 and ctl.state[1].updates == 3
    assert backbone.calls == [(True, True)] * 3 + [(False, False)]
    assert module.gate.grad is not None and module.gate.grad.abs().sum() > 0
    assert module.fast.initial_weight.grad is not None
    assert module.address.writer.weight.grad is not None
    assert all(p.grad is None for p in backbone.blocks[0].norm2.parameters())
