import torch

from worldttt.sap_binding.exact_value_probe import _source_values, score_pair


def test_exact_value_probe_prefers_corresponding_instance():
    values = torch.tensor([[1., 0.], [0., 1.]])
    labels = torch.tensor([[[1, 2]]])
    result = score_pair(values, values.clone(), labels, labels,
                        torch.tensor([0, 1]))
    assert result['valid']
    assert result['object_coverage'] == 1
    assert result['positive_mse'] == 0
    assert result['instance_top1'] == 1
    assert result['instance_random_top1'] == .5
    assert result['positive_better_than_mean'] == 1
    assert result['positive_better_than_wrong'] == 1


def test_exact_value_probe_reports_unmatched_objects():
    values = torch.tensor([[1., 0.], [0., 1.]])
    labels = torch.tensor([[[1, 2]]])
    result = score_pair(values, values, labels, labels,
                        torch.tensor([-1, -1]))
    assert not result['valid']
    assert result['object_coverage'] == 0


def test_raw_sources_preserve_frame_spatial_token_order():
    latent = torch.arange(8.).reshape(1, 2, 1, 2, 2)
    post = torch.arange(12.).reshape(1, 4, 3)
    record = {'supports': [{'latent': latent, 'post': post}]}
    device = torch.device('cpu')
    assert torch.equal(_source_values(record, 'post', None, device)[0], post[0])
    rows = _source_values(record, 'latent', None, device)[0]
    assert rows.shape == (4, 2)
    assert torch.equal(rows[0], torch.tensor([0., 4.]))
    assert torch.equal(rows[-1], torch.tensor([3., 7.]))
