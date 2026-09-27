import json
from pathlib import Path

import pytest

from worldttt.evaluate import evaluate
from worldttt.evaluation_resume import request_record, reusable_run


def completed(run, case):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'case.json').write_text(json.dumps(case))
    (run / 'latent.pt').write_bytes(b'latent')
    (run / 'video_generated.mp4').write_bytes(b'video')
    (run / 'updates.jsonl').write_text('')
    metrics = dict(case=case, mode='off', ablation=None, adapter=None, refiner=None,
                   base_checkpoint='base', ttt={'mode': 'off'},
                   generation=dict(step=50, cfg_scale=5., seed=42, num_frames=97,
                                   negative_prompt='', num_cached_blocks=2))
    (run / 'metrics.json').write_text(json.dumps(metrics))


def test_resume_skips_complete_and_preserves_interrupted_case(tmp_path, monkeypatch):
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'base_checkpoint': 'base', 'ttt': {'mode': 'off'}}))
    cases = [{'id': name, 'num_frames': 97, 'seed': 42} for name in ('a', 'b')]
    manifest = tmp_path / 'cases.jsonl'
    manifest.write_text('\n'.join(json.dumps(case) for case in cases))
    root = tmp_path / 'out'
    completed(root / 'off/runs/a', cases[0])
    interrupted = root / 'off/runs/b'
    interrupted.mkdir(parents=True)
    (interrupted / 'case.json').write_text(json.dumps(cases[1]))
    (interrupted / 'latent.pt').write_bytes(b'failed-case-latent')
    seen = []
    def infer(command, check):
        run = Path(command[command.index('--output') + 1])
        case = json.loads((run / 'case.json').read_text())
        seen.append(case['id'])
        completed(run, case)
    monkeypatch.setattr('worldttt.evaluate.subprocess.run', infer)
    monkeypatch.setattr('worldttt.evaluation_resume.complete_video', lambda *args: True)
    evaluate(settings, manifest, root, None, modes=['off'], resume=True)
    assert seen == ['b']
    assert len(json.loads((root / 'comparison.json').read_text())['runs']) == 2
    archived = list((root / '_interrupted/off').glob('b-*/latent.pt'))
    assert len(archived) == 1 and archived[0].read_bytes() == b'failed-case-latent'
    evaluate(settings, manifest, root, None, modes=['off'], resume=True)
    assert seen == ['b']


def test_resume_rejects_changed_adapter_content(tmp_path, monkeypatch):
    adapter = tmp_path / 'adapter.pt'
    adapter.write_bytes(b'old weights')
    case = {'id': 'a', 'num_frames': 97, 'seed': 42}
    request = request_record({'base_checkpoint': 'base'}, case, 'sap_online', adapter, None)
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'evaluation_request.json').write_text(json.dumps(request))
    adapter.write_bytes(b'new weights')
    changed = request_record({'base_checkpoint': 'base'}, case, 'sap_online', adapter, None)
    with pytest.raises(ValueError, match='changed settings, adapter or inputs'):
        reusable_run(run, changed)


def test_truncated_video_is_not_reused(tmp_path, monkeypatch):
    case = {'id': 'a', 'num_frames': 97, 'seed': 42}
    completed(tmp_path, case)
    request = request_record({'base_checkpoint': 'base'}, case, 'off', None, None)
    monkeypatch.setattr('worldttt.evaluation_resume.complete_video', lambda *args: False)
    assert not reusable_run(tmp_path, request)


def test_legacy_generation_mismatch_rejected(tmp_path):
    case = {'id': 'a', 'num_frames': 97, 'seed': 42}
    completed(tmp_path, case)
    request = request_record({'base_checkpoint': 'base', 'steps': 4}, case, 'off', None, None)
    with pytest.raises(ValueError, match='generation settings differ'):
        reusable_run(tmp_path, request)
