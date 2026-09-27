import torch
import pytest


def test_retrieval_reports_valid_coverage_and_hard_negative_rank():
    from worldttt.sap_ttt.probe import retrieval_metrics

    keys = torch.eye(3)
    queries = torch.stack((keys[1], keys[0], keys[2]))
    result = retrieval_metrics(queries, keys, torch.tensor([1, 0, 2]),
                               valid=torch.tensor([True, True, False]))
    assert result['coverage'] == pytest.approx(2 / 3)
    assert result['top1'] == 1.
    assert result['top5'] == 1.
    assert result['margin'] > 0
    shuffled = retrieval_metrics(queries, keys.flip(0), torch.tensor([1, 0, 2]),
                                  valid=torch.tensor([True, True, False]))
    assert shuffled['top1'] < result['top1']


def test_delayed_loss_backpropagates_through_three_fast_updates():
    from worldttt.sap_ttt.address import SapAddress
    from worldttt.sap_ttt.memory import SapMemory
    from worldttt.sap_ttt.probe import delayed_episode_loss

    torch.manual_seed(7)
    address = SapAddress(vision_dim=6, text_dim=6, ray_dim=3, dim=4)
    memory = SapMemory(dim=4, lr=.4)
    text = torch.randn(1, 3, 6)
    text_mask = torch.ones(1, 3, dtype=torch.bool)
    supports = [dict(visual=torch.randn(1, 2, 6), text=text, text_mask=text_mask,
                     rays=torch.randn(1, 2, 3)) for _ in range(3)]
    query = dict(visual=torch.randn(1, 2, 6), text=text, text_mask=text_mask,
                 rays=torch.randn(1, 2, 3), sigma=torch.full((1, 2, 1), .5))
    loss, detail = delayed_episode_loss(address, memory, supports, query,
                                        torch.tensor([1, 0]), mode='selective_geometry')
    loss.backward()
    assert detail['old_key_mse'] >= 0
    assert address.writer.weight.grad is not None
    assert address.reader.weight.grad is not None
    assert memory.initial_weight.grad is not None
    assert memory.log_lr.grad is not None
    assert torch.isfinite(loss)


def test_delayed_supervision_selects_targets_only_after_query_encoding():
    from worldttt.sap_ttt.probe import delayed_supervision_loss

    query = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], requires_grad=True)
    key = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    value = torch.tensor([[[2., 0.], [0., 2.]]])
    read = query.clone()
    loss, detail = delayed_supervision_loss(
        query, key, value, read, torch.tensor([0, 2]),
        torch.tensor([0, -1, 2]), torch.tensor([True, False, True]),
        torch.tensor([1, 0, 2]), torch.tensor([1, 0, 2]))
    loss.backward()
    assert detail['matched_queries'] == 2
    assert query.grad is not None
    assert key.grad is not None


def test_forgetting_curves_report_equal_bounded_replay_budget():
    from worldttt.sap_ttt.memory import SapMemory
    from worldttt.sap_ttt.probe import forgetting_curve

    memory = SapMemory(dim=2, lr=.3)
    supports = [(torch.tensor([[[1., 0.], [0., 1.]]]), torch.randn(1, 2, 2))
                for _ in range(4)]
    plain = forgetting_curve(memory, supports)
    anchor = forgetting_curve(memory, supports, replay='anchor', replay_capacity=2)
    random = forgetting_curve(memory, supports, replay='random', replay_capacity=2)
    assert len(plain) == len(anchor) == len(random) == 4
    assert all(row['replay_bytes'] <= 2 * 2 * 4 * 2 for row in anchor + random)
    assert all(len(row['old_key_mse']) == i + 1 for i, row in enumerate(plain))
