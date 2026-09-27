import torch


def test_address_rank_exceeds_shuffled_label_baseline():
    from worldttt.address_analysis import address_scores
    keys = torch.eye(8).reshape(8, 1, 8)
    score = address_scores(keys, keys, torch.arange(8), seed=9)
    assert score['top1'] == 1.
    assert score['mrr'] == 1.
    assert score['shuffled_top1'] < 1.
    partial = address_scores(keys[:3], keys, torch.arange(3), seed=9)
    assert partial['tokens'] == 3 and partial['top1'] == 1.
