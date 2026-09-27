import json

import pytest

from worldttt.sap_binding.compare import compare


def test_structure_comparison_is_paired_and_uses_validation_only(tmp_path):
    files = {}
    for name, architecture, mse in [('hybrid', 'hybrid', .7),
                                     ('bank', 'bank_only', 1.),
                                     ('fast', 'fast_only', 1.2),
                                     ('unconstrained', 'hybrid', .9)]:
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(dict(kind='sap_binding_causal_gate', split='val',
            architecture=architecture, gates=dict(passed=True), rows=[dict(
                kind='interference', scene_id='one', noise_sigma=.5,
                distractor_writes=8, control='online', query_mse=mse)])))
        files[name] = path
    result = compare(files['hybrid'], files['bank'], files['fast'], files['unconstrained'],
                     tmp_path / 'report.json')
    assert result['flow_allowed'] and result['hybrid_gain_over_best_single'] > .05
    payload = json.loads(files['fast'].read_text()); payload['split'] = 'test'
    files['fast'].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='validation'):
        compare(files['hybrid'], files['bank'], files['fast'], files['unconstrained'],
                tmp_path / 'report.json')
