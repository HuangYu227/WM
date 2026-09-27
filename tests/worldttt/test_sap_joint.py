import copy

import pytest
import torch


def record(scene='s0'):
    torch.manual_seed(13)
    def part(n):
        return dict(visual=torch.randn(1, n, 12), post=torch.randn(1, n, 12),
                    text=torch.randn(1, 4, 12), text_mask=torch.ones(1, 4, dtype=torch.bool),
                    rays=torch.randn(1, n, 6), sigma=torch.full((1, n, 1), .8))
    return dict(version=1, source='teacher_forced_ground_truth', scene_id=scene,
        layer=1, base_checkpoint='test-base', noise_sigmas=[.8],
        supports=[part(8), part(6), part(6)], queries=[part(6)],
        supervision=dict(positive=torch.tensor([0, 1, 2, 3, 4, 5]), valid=torch.ones(6, dtype=torch.bool),
            support_instance=torch.ones(10, 1, 2, dtype=torch.long),
            query_instance=torch.ones(3, 1, 2, dtype=torch.long)))


def test_sampling_writes_do_not_use_future_correspondences():
    from worldttt.sap_ttt.joint import sample_episode
    original = record()
    changed = copy.deepcopy(original)
    changed['supervision']['positive'] = original['supervision']['positive'].roll(1)
    first = sample_episode(original, 0, seed=6, support_budget=4, query_budget=3)
    second = sample_episode(changed, 0, seed=6, support_budget=4, query_budget=3)
    for a, b in zip(first['supports'], second['supports']):
        torch.testing.assert_close(a['visual'], b['visual'])


def test_batched_joint_loss_matches_independent_episodes_and_has_meta_gradients():
    from worldttt.sap_ttt.joint import sample_episode, collate_episodes, joint_objective
    from worldttt.sap_ttt.runtime import SapConfig, SapBlockMemory
    config = SapConfig(layers=(1,), dim=16, ray_dim=6, address_arch='multimodal',
                       memory_arch='swiglu', heads=2, memory_hidden_dim=12, inner_lr=.1)
    module = SapBlockMemory(12, config)
    samples = [sample_episode(record('s' + str(i)), 0, seed=i, support_budget=8, query_budget=3)
               for i in range(2)]
    batch = collate_episodes(samples, 'cpu')
    loss, rows = joint_objective(module, batch, config, training=True)
    single = [joint_objective(module, collate_episodes([s], 'cpu'), config, training=True)[0]
              for s in samples]
    torch.testing.assert_close(loss, torch.stack(single).mean(), rtol=2e-5, atol=2e-5)
    loss.backward()
    assert module.address.writer.weight.grad.abs().sum() > 0
    assert module.memory.initial_weight.grad.abs().sum() > 0
    assert rows['matched_queries'] == 6
    assert module.output.weight.grad is None  # feature pretraining does not train video fusion


def test_joint_runner_saves_best_last_and_rejects_overlap(tmp_path):
    from worldttt.sap_ttt.joint import run
    paths = {}
    for split in ('train', 'val', 'test'):
        path = tmp_path / (split + '.pt'); torch.save(record(split), path)
        paths[split + '_features'] = [str(path)]
    settings = dict(paths, sap=dict(layers=[1], dim=16, ray_dim=6, address_arch='multimodal',
        memory_arch='swiglu', heads=2, memory_hidden_dim=12, inner_lr=.1),
        steps=2, batch_size=2, support_tokens=8, query_tokens=3, val_every=1)
    run(settings, tmp_path / 'run')
    assert (tmp_path / 'run/best.pt').exists()
    assert (tmp_path / 'run/last.pt').exists()
    import json
    result = json.loads((tmp_path / 'run/test_results.json').read_text())
    assert len(result['rows']) == 2  # frozen and online, same selected adapter
    settings['val_features'] = paths['train_features']
    with pytest.raises(ValueError, match='disjoint'):
        run(settings, tmp_path / 'invalid')
