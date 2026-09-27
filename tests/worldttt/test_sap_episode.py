from types import SimpleNamespace

import torch
from torch import nn


def test_four_chunk_episode_commits_three_supports_then_scores_independent_query():
    from worldttt.sap_ttt.episode import SapEpisodeModel

    class Controller:
        config = SimpleNamespace(layers=(1,))
        def __init__(self):
            self.commits = []
        def reset_episode(self, episode, batch, training):
            assert batch == 1
        def context(self, times, collect=False, chunk=0, training=False, record_query=False,
                    source='unspecified'):
            return SimpleNamespace(collect=collect, chunk=chunk, source=source)
        def commit(self, context, chunk, training=False):
            assert context.collect
            self.commits.append((chunk, context.source))
            return {'committed': True}
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(.1))
            self.calls = []
        def forward(self, z, times, y=None, *, sap_context, save_kv_cache, kv_cache, **kw):
            self.calls.append((sap_context.collect, save_kv_cache, z.shape[2]))
            return z * 0 + self.bias, kv_cache
    ctl, backbone = Controller(), Backbone()
    model = SapEpisodeModel(backbone, ctl, cache_factory=lambda _: (
        [None] * 4, lambda i: None))
    loss, metrics = model(torch.randn(1, 4, 13, 2, 2), torch.zeros(1, 13, 20),
                          torch.zeros(1, 48, 13, 2, 2), torch.zeros(1, 2, 8),
                          torch.ones(1, 2), 'scene', seed=7)
    assert ctl.commits == [(0, 'reference_plus_ground_truth'),
                           (1, 'ground_truth'), (2, 'ground_truth')]
    assert backbone.calls == [(True, True, 4), (True, True, 3),
                              (True, True, 3), (False, False, 3)]
    assert metrics['flow_mse'] > 0
    loss.backward()
    assert backbone.bias.grad is not None
