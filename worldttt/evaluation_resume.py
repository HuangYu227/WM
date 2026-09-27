"""Resume completed evaluation cases; interrupted SAP rollouts restart per case."""
import hashlib
import json
from pathlib import Path


def request_record(settings, case, mode, adapter, ablation):
    paths = [case.get(key) for key in ('image', 'camera', 'intrinsics')]
    paths += [settings.get('sana_config'), adapter]
    files = {}
    for name in paths:
        if name is None:
            continue
        path = Path(name).resolve()
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            files[str(path)] = digest.hexdigest()
        else:
            files[str(path)] = None
    return dict(settings=settings, case=case, mode=mode,
                adapter=str(Path(adapter).resolve()) if adapter else None,
                ablation=ablation, files=files)


def complete_video(path, expected_frames):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        count = 0
        while capture.read()[0]:
            count += 1
        return count == expected_frames
    finally:
        capture.release()


def _validate_legacy(metrics, request):
    """Old runs have no input hashes: validate their recorded provenance only."""
    settings, case, mode = request['settings'], request['case'], request['mode']
    expected = dict(case=case, mode=mode, ablation=request['ablation'],
                    base_checkpoint=settings.get('base_checkpoint'), refiner=settings.get('refiner'))
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(f'Cannot resume: existing {key} differs from this request')
    adapter = metrics.get('adapter')
    if (str(Path(adapter).resolve()) if adapter else None) != request['adapter']:
        raise ValueError('Cannot resume: existing adapter differs from this request')
    generation = dict(step=settings.get('steps', 50), cfg_scale=settings.get('cfg_scale', 5.),
                      seed=case.get('seed', 42), num_frames=case.get('num_frames', 73),
                      negative_prompt=case.get('negative_prompt', ''),
                      num_cached_blocks=settings.get('num_cached_blocks', 2))
    if any(metrics.get('generation', {}).get(key) != value for key, value in generation.items()):
        raise ValueError('Cannot resume: existing generation settings differ')
    if mode.startswith('sap_'):
        config = dict(settings.get('sap', {}),
                      mode='frozen' if mode == 'sap_frozen' else 'online',
                      read_enabled=mode != 'sap_no_read',
                      commit_enabled=mode != 'sap_no_commit',
                      shuffle_query_text=mode == 'sap_shuffle_text')
    elif mode in ('binding_frozen', 'binding_online'):
        config = dict(settings.get('sap_binding', {}),
                      mode='frozen' if mode == 'binding_frozen' else 'online')
    else:
        config = dict(settings.get('ttt', {}), mode='off' if mode == 'binding_off' else mode)
        if request['ablation'] == 'reset':
            config['reset_each_chunk'] = True
        elif request['ablation'] == 'no_protection':
            config.update(keep_weight=0., trust_weight=0.)
        elif request['ablation'] == 'shuffle':
            config['shuffle_targets'] = True
    if any(metrics.get('ttt', {}).get(key) != value for key, value in config.items()):
        raise ValueError('Cannot resume: existing adaptation settings differ')


def reusable_run(run, request):
    run = Path(run)
    snapshot = run / 'evaluation_request.json'
    if snapshot.is_file() and json.loads(snapshot.read_text(encoding='utf-8')) != request:
        raise ValueError(f'Cannot resume changed settings, adapter or inputs: {run}')
    case_path = run / 'case.json'
    if case_path.is_file() and json.loads(case_path.read_text(encoding='utf-8')) != request['case']:
        raise ValueError(f'Cannot resume a different case: {run}')
    required = ('metrics.json', 'updates.jsonl', 'latent.pt', 'video_generated.mp4')
    if not all((run / name).is_file() for name in required):
        return False
    if not (run / 'latent.pt').stat().st_size:
        return False
    try:
        metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
        for line in (run / 'updates.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                json.loads(line)
    except (ValueError, OSError):
        return False
    _validate_legacy(metrics, request)
    if not snapshot.is_file():
        print(f'[resume] Legacy metadata validated; original input hashes unavailable: {run}', flush=True)
    frames = request['case'].get('num_frames', 73)
    if not complete_video(run / 'video_generated.mp4', frames):
        return False
    if request['settings'].get('refiner'):
        if not complete_video(run / 'stage1_generated.mp4', frames):
            return False
    return True
