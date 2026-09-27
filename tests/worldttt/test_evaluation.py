import json

import pytest

from worldttt.evaluate import verify_video_metrics


def test_zero_exit_of_external_evaluator_does_not_imply_metrics(tmp_path):
    with pytest.raises(ValueError, match='Missing'):
        verify_video_metrics(tmp_path, 'test', {'scene'})
    (tmp_path / 'test').mkdir()
    (tmp_path / 'test/eval_poses.json').write_text('{}')
    with pytest.raises(ValueError, match='camera'):
        verify_video_metrics(tmp_path, 'test', {'scene'})


def test_comparison_uses_matching_frozen_adapter_for_each_method(tmp_path, monkeypatch):
    from pathlib import Path
    from worldttt.evaluate import evaluate

    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'adapters': {'noise_ttt': 'noise.pt', 'kv_ttt': 'kv.pt'}}))
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'scene'}) + '\n')
    seen = []

    def fake_run(command, check):
        mode = command[command.index('--mode') + 1]
        adapter = command[command.index('--adapter') + 1] if '--adapter' in command else None
        run = Path(command[command.index('--output') + 1])
        (run / 'video_generated.mp4').write_bytes(b'video')
        (run / 'metrics.json').write_text('{}')
        (run / 'updates.jsonl').write_text('')
        seen.append((run.parent.parent.name, mode, adapter))

    monkeypatch.setattr('worldttt.evaluate.subprocess.run', fake_run)
    evaluate(settings, cases, tmp_path / 'out', adapter=None)
    assert ('frozen_noise', 'frozen', 'noise.pt') in seen
    assert ('frozen_kv', 'frozen', 'kv.pt') in seen
    assert ('noise_ttt', 'noise_ttt', 'noise.pt') in seen
    assert ('kv_ttt', 'kv_ttt', 'kv.pt') in seen


def test_a800_modes_need_only_kv_adapter(tmp_path, monkeypatch):
    from pathlib import Path
    from worldttt.evaluate import evaluate
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'adapters': {'kv_ttt': 'kv.pt'}}))
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'scene'}) + '\n')
    seen = []
    def fake_run(command, check):
        run = Path(command[command.index('--output') + 1])
        (run / 'video_generated.mp4').write_bytes(b'video')
        (run / 'metrics.json').write_text('{}')
        (run / 'updates.jsonl').write_text('')
        seen.append((run.parent.parent.name, command[command.index('--mode') + 1],
                     command[command.index('--ablation') + 1] if '--ablation' in command else None))
    monkeypatch.setattr('worldttt.evaluate.subprocess.run', fake_run)
    evaluate(settings, cases, tmp_path / 'out', adapter=None,
             modes=('off', 'frozen_kv', 'kv_ttt', 'kv_ttt_reset', 'kv_ttt_shuffle'))
    assert seen == [('off', 'off', None), ('frozen_kv', 'frozen', None),
                    ('kv_ttt', 'kv_ttt', None), ('kv_ttt_reset', 'kv_ttt', 'reset'),
                    ('kv_ttt_shuffle', 'kv_ttt', 'shuffle')]
