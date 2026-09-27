import json

import torch


def _record(scene='s0'):
    torch.manual_seed(sum(map(ord, scene)))
    def part(n):
        return dict(visual=torch.randn(1, n, 12), post=torch.randn(1, n, 12),
                    text=torch.randn(1, 4, 12), text_mask=torch.ones(1, 4, dtype=torch.bool),
                    rays=torch.randn(1, n, 6), sigma=torch.full((1, n, 1), .8))
    query = part(6)
    no_history = dict(query, visual=query['visual'] + .25)
    return dict(version=1, source='teacher_forced_ground_truth', scene_id=scene,
        layer=1, base_checkpoint='test-base', noise_sigmas=[.8],
        supports=[part(8), part(6), part(6)], queries=[query], queries_no_history=[no_history],
        supervision=dict(positive=torch.arange(6), valid=torch.ones(6, dtype=torch.bool),
            support_instance=torch.ones(10, 1, 2, dtype=torch.long),
            query_instance=torch.ones(3, 1, 2, dtype=torch.long)))


def _adapter(path):
    from dataclasses import asdict
    from worldttt.sap_ttt.runtime import SapBlockMemory, SapConfig
    config = SapConfig(layers=(1,), dim=16, ray_dim=6, support_tokens=8,
        address_arch='multimodal', memory_arch='swiglu', heads=2,
        memory_hidden_dim=12, inner_lr=.1, normalized_value=True)
    module = SapBlockMemory(12, config)
    torch.save(dict(version=1, config=asdict(config), base_checkpoint='test-base',
        modules={1: module.state_dict()}, extra={'stage': 'test'}), path)


def test_four_read_only_mechanism_experiments(tmp_path):
    from worldttt.sap_ttt.mechanism_eval import run
    features = []
    for scene in ('s0', 's1'):
        path = tmp_path / f'{scene}.pt'; torch.save(_record(scene), path); features.append(str(path))
    settings = dict(test_features=features, support_tokens=8, query_tokens=4,
                    support_budgets=[4, 6, 8], interference_lengths=[0, 1, 3])
    native = tmp_path / 'native.pt'; no_history = tmp_path / 'no-history.pt'
    _adapter(native); _adapter(no_history)
    runs = {
        'cross_source': {'native': native, 'no_history': no_history},
        'causal_controls': {'full': native},
        'interference': {'full': native},
        'budget': {'full': native},
    }
    expected = {'cross_source': 16, 'causal_controls': 12, 'interference': 12, 'budget': 12}
    for experiment, adapters in runs.items():
        output = tmp_path / f'{experiment}.json'
        run(settings, experiment, adapters, output)
        result = json.loads(output.read_text())
        assert result['experiment'] == experiment
        assert len(result['rows']) == expected[experiment]
        assert all(torch.isfinite(torch.tensor(row['query_mse']))
                   for row in result['rows'] if row['valid'])
    controls = {row['control'] for row in json.loads((tmp_path / 'causal_controls.json').read_text())['rows']}
    assert controls == {'frozen', 'online', 'shuffle_value', 'shuffle_key', 'last_only', 'mean_value'}


def test_budget_curve_uses_same_queries_and_nested_supports():
    from worldttt.sap_ttt.mechanism_eval import _nested_sample
    samples = _nested_sample(_record(), 'native', 0, (4, 6, 8), 2, 7)
    expected_query = samples[0]['query']['visual']
    expected_targets = samples[0]['supports'][0]['visual'][:, samples[0]['positive']]
    assert all(torch.equal(expected_query, sample['query']['visual']) for sample in samples)
    assert all(torch.equal(expected_targets,
                           sample['supports'][0]['visual'][:, sample['positive']])
               for sample in samples)
    assert [len(sample['supports'][0]['visual'][0]) for sample in samples] == [4, 6, 8]
    assert samples[0]['coverage'] <= samples[1]['coverage'] <= samples[2]['coverage']
