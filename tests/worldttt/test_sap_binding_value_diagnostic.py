import torch

from worldttt.sap_binding.value import BindingValueEncoder
from worldttt.sap_binding.value_diagnostic import _centroid_scores, _paired_variant_signal


def test_paired_variant_signal_uses_validation_object_region():
    labels = torch.tensor([[[1, 0], [2, 0]]])
    first = dict(layout_id='layout_1', variant=0,
                 supervision={'support_instance': labels},
                 supports=[{'latent': torch.zeros(1, 1, 1, 2, 2),
                            'post': torch.zeros(1, 4, 3)}])
    second = dict(layout_id='layout_1', variant=1,
                  supervision={'support_instance': labels.clone()},
                  supports=[{'latent': torch.ones(1, 1, 1, 2, 2),
                             'post': torch.ones(1, 4, 3)}])
    second['supports'][0]['latent'][:, :, :, 0, 0] = 3
    second['supports'][0]['post'][:, 0] = 3
    rows = _paired_variant_signal([first, second])
    assert {row['source'] for row in rows} == {'latent', 'post'}
    assert all(row['object_over_background'] > 1 for row in rows)
    assert all(row['object_tokens'] == 2 and row['background_tokens'] == 2 for row in rows)


def test_centroid_scores_separate_stable_instances():
    tokens = torch.tensor([[[3., 0., 0.], [3., 0., 0.],
                            [0., 3., 0.], [0., 3., 0.]]])
    part = {'post': tokens, 'latent': torch.zeros(1, 1, 1, 2, 2)}
    record = dict(scene_id='example', supports=[part, part],
                  supervision={'support_instance': torch.tensor(
                      [[[1, 1], [2, 2]], [[1, 1], [2, 2]]])})
    encoder = BindingValueEncoder(3, 1, 'old', value_dim=4)
    rows = _centroid_scores(encoder, [record])
    assert len(rows) == 2
    assert all(row['top1'] for row in rows)
    assert all(row['cross_same'] < row['cross_hard_negative'] for row in rows)
