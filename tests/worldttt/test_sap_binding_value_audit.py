import json

import torch


def record(scene, layout, variant, shift=0.):
    supports = []
    labels = []
    for frames in (2, 1, 1):
        instance = torch.tensor(([1, 1, 2, 2] * frames), dtype=torch.long)
        hidden = torch.zeros(1, len(instance), 8)
        hidden[0, instance == 1, 0] = 1 + shift
        hidden[0, instance == 2, 1] = 1 + shift
        supports.append(dict(post=hidden, latent=torch.randn(1, 4, frames, 2, 2)))
        labels.append(instance.reshape(frames, 2, 2))
    return dict(source='sap_binding_teacher_forced_ground_truth', query_obscured=True,
                layer=7,
                scene_id=scene, layout_id=layout, variant=variant, supports=supports,
                supervision={'support_instance': torch.cat(labels)})


def test_value_audit_seals_test_and_fits_only_train(tmp_path):
    from worldttt.sap_binding.value_audit import run

    paths = []
    for i, item in enumerate((record('a', 'l1', 0), record('b', 'l1', 1, .1),
                              record('c', 'l2', 0, .2))):
        path = tmp_path / f'{i}.pt'; torch.save(item, path); paths.append(str(path))
    settings = dict(value_mode='highpass_h7_latent', value_dim=8, seed=3, layer=7,
                    train_features=paths[:2], val_features=paths[2:])
    result = run(settings, tmp_path / 'out')
    report = json.loads(result.read_text())
    assert report['test_sealed'] and set(report['summary']) == {'train', 'val'}
    stats = torch.load(tmp_path / 'out/value_stats.pt', weights_only=True)
    assert stats['kind'] == 'sap_binding_value_stats' and stats['train_layouts'] == ['l1']
    assert stats['layer'] == 7
