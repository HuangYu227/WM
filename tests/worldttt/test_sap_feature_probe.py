import torch
import copy
import pytest


def _record():
    torch.manual_seed(3)
    def part(n):
        return dict(visual=torch.randn(1, n, 6), post=torch.randn(1, n, 6),
                    text=torch.randn(1, 3, 6), text_mask=torch.ones(1, 3, dtype=torch.bool),
                    rays=torch.randn(1, n, 3), sigma=torch.zeros(1, n, 1))
    query = part(3)
    query['sigma'].fill_(.5)
    return dict(version=1, source='teacher_forced_ground_truth', scene_id='synthetic_1',
                supports=[part(4), part(3), part(3)], queries=[query],
                supervision=dict(positive=torch.tensor([0, 1, 3]),
                                 valid=torch.tensor([True, True, True]),
                                 support_instance=torch.tensor([1, 2, 0, 1, 3, 3, 3, 4, 4, 4]).reshape(10, 1, 1),
                                 query_instance=torch.tensor([1, 2, 1]).reshape(3, 1, 1)))


def test_address_probe_scores_same_instance_and_backpropagates():
    from worldttt.sap_ttt.address import SapAddress
    from worldttt.sap_ttt.feature_probe import address_objective, select_batch

    record = _record()
    batch = select_batch(record, noise_index=0, max_queries=3, random_negatives=2, seed=4)
    assert batch['query_instance'].tolist() == [1, 2, 1]
    assert batch['candidate_instance'].numel() >= 3
    address = SapAddress(6, 6, 3, 4)
    loss, metric = address_objective(address, batch, mode='selective_geometry')
    loss.backward()
    assert torch.isfinite(loss)
    assert 0 <= metric['instance_top1'] <= 1
    assert address.writer.weight.grad is not None


def test_shuffled_text_changes_query_only_and_preserves_historical_evidence():
    from worldttt.sap_ttt.feature_probe import shuffled_query_text

    batch = {'query': {'text': torch.ones(1, 3, 6), 'text_mask': torch.ones(1, 3, dtype=torch.bool)},
             'candidates': [{'text': torch.ones(1, 3, 6)}]}
    donor = {'text': torch.zeros(1, 3, 6), 'text_mask': torch.ones(1, 3, dtype=torch.bool)}
    randomized = shuffled_query_text(batch, donor)
    assert torch.equal(randomized['query']['text'], donor['text'])
    assert torch.equal(randomized['candidates'][0]['text'], batch['candidates'][0]['text'])


def test_address_loss_distinguishes_exact_correspondences_within_one_instance():
    from worldttt.sap_ttt.address import SapAddress
    from worldttt.sap_ttt.feature_probe import address_objective, select_batch

    torch.manual_seed(11)
    batch = select_batch(_record(), max_queries=3, random_negatives=2, seed=4)
    changed = copy.deepcopy(batch)
    same_instance = torch.where(batch['candidate_instance'] == batch['query_instance'][0])[0]
    other = same_instance[same_instance != batch['positive_candidate'][0]][0]
    changed['positive_candidate'][0] = other
    address = SapAddress(6, 6, 3, 4)
    original, _ = address_objective(address, batch, mode='vision')
    alternative, _ = address_objective(address, changed, mode='vision')
    assert not torch.allclose(original, alternative)


def test_position_baseline_audits_spatial_shortcut_without_visual_features():
    from worldttt.sap_ttt.feature_probe import position_baseline

    batch = {'query_indices': torch.tensor([0, 1]),
             'candidate_indices': torch.tensor([1, 0]),
             'spatial_shape': (1, 2),
             'query_instance': torch.tensor([1, 2]),
             'candidate_instance': torch.tensor([2, 1]),
             'positive_candidate': torch.tensor([1, 0])}
    metric = position_baseline(batch)
    assert metric['instance_top1'] == metric['exact_top1'] == 1.
    assert metric['random_instance_top1'] == metric['random_exact_top1'] == .5


def test_query_source_switch_keeps_native_history_and_checks_no_history_record():
    from worldttt.sap_ttt.feature_probe import select_query_source

    record = _record()
    without_history = copy.deepcopy(record['queries'])
    without_history[0]['visual'].zero_()
    record['queries_no_history'] = without_history
    assert torch.equal(select_query_source(record, 'native')['queries'][0]['visual'],
                       record['queries'][0]['visual'])
    assert torch.equal(select_query_source(record, 'no_history')['queries'][0]['visual'],
                       without_history[0]['visual'])
    with pytest.raises(ValueError, match='queries_no_history'):
        select_query_source(_record(), 'no_history')
