import torch

from test_sap_feature_probe import _record


def test_delayed_batch_uses_only_matched_historical_tokens():
    from worldttt.sap_ttt.memory_probe import sample_delayed_batch

    supports, query, positive = sample_delayed_batch(_record(), 0, seed=9,
                                                       query_budget=3, support_budget=3)
    assert len(supports) == 3
    assert query['visual'].shape[1] == len(positive) == 3
    assert (positive >= 0).all()
    assert (positive < supports[0]['visual'].shape[1]).all()
    assert 'query_instance' not in query
