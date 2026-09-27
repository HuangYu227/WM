"""Label-free GRAIL experiment summaries, independent of SANA weights."""

import pytest
import torch

from worldttt.grail_experiment import episode_record, paired_scene_summary, run_experiment


def test_episode_record_counts_reuse_by_slot_generation_not_slot_number():
    trace = [
        dict(chunk_id=0, committed=True, accepted=2, slots=[dict(batch=0, slot=0, generation=0, observations=2,
                                                real_observations=2, replaced=False)]),
        dict(chunk_id=1, committed=True, accepted=1, slots=[dict(batch=0, slot=0, generation=0, observations=1,
                                                real_observations=1, replaced=False)]),
        dict(chunk_id=2, committed=True, accepted=1, slots=[dict(batch=0, slot=0, generation=1, observations=1,
                                                real_observations=0, replaced=True)]),
    ]
    result = dict(variant_future={'ridge': torch.tensor(.2), 'no_read': torch.tensor(.5)},
                  variant_gate_mean={'ridge': .1, 'no_read': 0.},
                  variant_slot_coverage={'ridge': .75, 'no_read': .75},
                  variant_hook_counts={'ridge': {2: 1}, 'no_read': {2: 1}},
                  memory_trace=trace, support_chunks=3)
    row = episode_record(result, scene_id='scene-a', key='research/a', history='real', seed=7)
    assert row['slot_summary']['reused_across_chunks'] == 1
    assert row['slot_summary']['slot_generations'] == 2
    assert row['slot_summary']['accepted_observations'] == 4
    assert row['future_flow_mse']['ridge'] == pytest.approx(.2)
    assert row['delta_vs_ridge']['no_read'] == pytest.approx(.3)
    assert row['ground_truth_instances'] is False
    assert row['intervention_scope'] == 'heldout_query_only'


def test_paired_scene_summary_averages_episodes_before_bootstrap():
    records = [
        dict(scene_id='a', history='real', future_flow_mse={'ridge': 1., 'no_read': 3.}),
        dict(scene_id='a', history='real', future_flow_mse={'ridge': 3., 'no_read': 5.}),
        dict(scene_id='b', history='real', future_flow_mse={'ridge': 2., 'no_read': 3.}),
    ]
    summary = paired_scene_summary(records, samples=100, seed=1)
    assert summary['real']['no_read']['scenes'] == ['a', 'b']
    assert summary['real']['no_read']['mean'] == 1.5


def test_experiment_rejects_invalid_frames_before_loading_sana(tmp_path):
    with pytest.raises(ValueError, match='at least 10'):
        run_experiment({}, 'adapter.pt', tmp_path / 'out', frames=9)


def test_experiment_cli_exposes_explicit_gpu_evaluation_options(monkeypatch, capsys):
    from worldttt.__main__ import main
    monkeypatch.setattr('sys.argv', ['worldttt', 'grail-experiment', '--help'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert '--samples' in help_text and '--frames' in help_text and '--histories' in help_text
