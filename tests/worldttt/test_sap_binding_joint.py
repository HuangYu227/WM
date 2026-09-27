import json
from dataclasses import replace

import pytest
import torch


def _part(tokens, frames):
    visual = torch.randn(1, tokens, 8)
    return dict(visual=visual, post=visual + .1, rays=torch.randn(1, tokens, 4),
                sigma=torch.zeros(1, tokens, 1), text=torch.randn(1, 3, 8),
                text_mask=torch.ones(1, 3, dtype=torch.bool),
                latent=torch.randn(1, 4, frames, 2, 2))


def _record(scene, layout, variant):
    supports = [_part(8, 2), _part(4, 1), _part(4, 1)]
    query = _part(4, 1); query.pop('latent')
    return dict(source='sap_binding_teacher_forced_ground_truth', query_obscured=True,
        scene_id=scene, layout_id=layout, variant=variant, base_checkpoint='base', layer=1,
        supports=supports, queries=[query], noise_sigmas=[.8],
        supervision=dict(positive=torch.arange(4), valid=torch.ones(4, dtype=torch.bool),
                         support_instance=torch.ones(4, 2, 2, dtype=torch.long),
                         query_instance=torch.ones(1, 2, 2, dtype=torch.long)))


def test_joint_training_writes_binding_checkpoint_and_causal_metrics(tmp_path, monkeypatch):
    from worldttt.sap_binding.joint import run
    from worldttt.sap_binding.causal_eval import run as causal_run, evaluate_batch
    from worldttt.sap_binding.joint import attach_values, collate, sample_episode
    from worldttt.sap_binding.config import BindingConfig
    from worldttt.sap_binding.model import BindingBlock
    from worldttt.sap_binding.value import BindingValueEncoder

    paths = {}
    for split, items in {'train': [('a0', 'a', 0), ('a1', 'a', 1)],
                         'val': [('b0', 'b', 0)], 'test': [('c0', 'c', 0)]}.items():
        paths[split] = []
        for args in items:
            path = tmp_path / f'{args[0]}.pt'; torch.save(_record(*args), path); paths[split].append(str(path))
    encoder = BindingValueEncoder(8, 4, value_dim=8, seed=3)
    stats = tmp_path / 'stats.pt'
    torch.save(dict(version=1, kind='sap_binding_value_stats', mode='highpass_h7_latent',
                    encoder=encoder.state_dict(), train_layouts=['a'], audit_passed=False), stats)
    settings = dict(sap_binding=dict(mode='online', layers=[1], address_dim=8, value_dim=8,
        ray_dim=4, latent_dim=4, heads=2, topk=2, capacity=8, protected_anchors=2,
        support_tokens=8, address_depth=1, fast_hidden_dim=6, fast_lr=.1, seed=3,
        value_mode='highpass_h7_latent', architecture='hybrid'),
        train_features=paths['train'], val_features=paths['val'], test_features=paths['test'],
        value_stats=str(stats), steps=2, batch_size=2, val_every=1, support_tokens=8,
        protected_anchors=2, query_tokens=2, seed=3, val_seed=4, eval_seed=4, lr=1e-3,
        loss_weights=dict(addr=.2, value=1., fast=.5, shuffle=.5, mean=.25, write=.01))
    result = run(settings, tmp_path / 'out')
    payload = torch.load(result, weights_only=True)
    assert payload['kind'] == 'sap_binding_adapter' and payload['extra']['flow_trained'] is False
    rows = json.loads((tmp_path / 'out/test_results.json').read_text())['rows']
    assert rows and all(row['valid'] for row in rows)
    assert {'query_mse', 'shuffle_key_mse', 'shuffle_value_mse', 'mean_value_mse', 'exact_top1'} <= rows[0].keys()
    report_path = causal_run(settings, result, tmp_path / 'causal.json')
    report = json.loads(report_path.read_text())
    assert report['kind'] == 'sap_binding_causal_gate'
    assert report['split'] == 'val'
    from worldttt.sap_binding.fusion_calibrate import run as calibrate_fusion
    calibrated = calibrate_fusion(settings, result, tmp_path / 'fusion-calibration',
                                  steps=2, batch_size=2, val_every=1)
    calibrated_last = torch.load(tmp_path / 'fusion-calibration/last.pt', weights_only=True)
    calibrated_best = torch.load(calibrated, weights_only=True)
    assert calibrated_best['extra']['stage'] == 'binding_fusion_calibration'
    assert (tmp_path / 'fusion-calibration/test_results.json').is_file()
    from worldttt.sap_binding.fusion_probe import run as probe_fusion
    probe = json.loads(probe_fusion(settings, calibrated, tmp_path / 'fusion-probe.json',
                                   split='val').read_text())
    assert probe['rows']
    assert probe['mean_query_mse']['bank'] == pytest.approx(
        probe['mean_query_mse']['alpha_1'], abs=1e-7)
    assert probe['mean_query_mse']['learned'] >= 0
    split_probe = json.loads(probe_fusion(settings, calibrated,
        tmp_path / 'fusion-coverage.json', split='val', coverage_split=True).read_text())
    assert {'covered', 'uncovered'} == set(split_probe['mean_query_mse'])
    assert all(split_probe['query_count'][group] > 0 for group in ('covered', 'uncovered'))
    for name, initial in payload['modules'][1].items():
        if not name.startswith('fusion.'):
            assert torch.equal(initial, calibrated_last['modules'][1][name]), name
    assert any(not torch.equal(initial, calibrated_last['modules'][1][name])
               for name, initial in payload['modules'][1].items() if name.startswith('fusion.'))
    config = BindingConfig(**settings['sap_binding'])
    module = BindingBlock(8, 4, config)
    module.load_state_dict(payload['modules'][1])
    record = torch.load(paths['val'][0], weights_only=True)
    attach_values([record], module.value)
    episode = sample_episode(record, 0, seed=4, support_tokens=8,
                             query_tokens=2, distractor_writes=2, protected=2)
    uncovered_index = next(i for i in range(4) if i not in episode['first_indices'])
    uncovered = sample_episode(record, 0, seed=4, support_tokens=8,
        query_tokens=2, distractor_writes=2, protected=2,
        forced_query_indices=[uncovered_index], include_uncovered=True)
    assert uncovered['positive'].tolist() == [-1]
    assert torch.equal(uncovered['writes'][0]['visual'], episode['writes'][0]['visual'])
    assert torch.isfinite(uncovered['target']).all()
    batch = collate([episode], torch.device('cpu'))
    calls = []
    update = module.fast.update
    def counted_update(*args, **kwargs):
        calls.append(1)
        return update(*args, **kwargs)
    monkeypatch.setattr(module.fast, 'update', counted_update)
    frozen = evaluate_batch(module, batch, config, 'frozen')
    bank_only_update = evaluate_batch(module, batch, config, 'bank_frozen_fast')
    assert frozen['candidates'] == 0 and bank_only_update['candidates'] > 0
    assert not calls
    evaluate_batch(module, batch, config, 'online')
    assert calls
    bank_config = replace(config, architecture='bank_only')
    bank_module = BindingBlock(8, 4, bank_config)
    bank_module.load_state_dict(module.state_dict())
    expected = evaluate_batch(bank_module, batch, bank_config, 'online')
    same_checkpoint_bank = evaluate_batch(module, batch, config, 'bank_read_only')
    assert same_checkpoint_bank['query_mse'] == pytest.approx(expected['query_mse'], abs=1e-7)
    assert same_checkpoint_bank['exact_top1'] == expected['exact_top1']
    assert {'normal_ratios', 'eight_write_ratios', 'exact_top1_256',
            'budget_common_query_mse', 'budget_recall', 'passed'} <= report['gates'].keys()

    corrupt = torch.load(stats, weights_only=True)
    corrupt['layer'] = 2
    torch.save(corrupt, stats)
    with pytest.raises(ValueError, match='layer'):
        run(settings, tmp_path / 'wrong-layer')
    corrupt['layer'] = 1
    corrupt['train_layouts'] = ['b']
    torch.save(corrupt, stats)
    with pytest.raises(ValueError, match='training split'):
        run(settings, tmp_path / 'wrong-stats')
