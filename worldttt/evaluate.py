"""Matched experiment runner and adapters to the upstream video evaluators."""
import copy
import gc
import json
import math
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import torch


SAP_MODES = ('sap_frozen', 'sap_online', 'sap_no_read', 'sap_no_commit',
             'sap_shuffle_text')
BINDING_MODES = ('binding_off', 'binding_frozen', 'binding_online')


def evaluate(settings_path, cases_path, output, adapter, ablations=False, video_metrics=False, modes=None, resume=False):
    from .evaluation_resume import request_record, reusable_run
    if resume and video_metrics:
        raise ValueError('Resume supports per-case evaluation; run upstream video metrics in a fresh directory')
    settings = json.loads(Path(settings_path).read_text(encoding='utf-8'))
    cases = [json.loads(s) for s in Path(cases_path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not cases:
        raise ValueError('No evaluation cases')
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids) or any(not s or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in s) for s in ids):
        raise ValueError('Case IDs must be unique file-safe identifiers')
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError('Output exists; use --resume with unchanged settings to reuse complete cases')
    root.mkdir(parents=True, exist_ok=True)
    available = [('off', 'off', None, None),
             ('frozen_noise', 'frozen', 'noise_ttt', None),
             ('frozen_kv', 'frozen', 'kv_ttt', None),
             ('kv_ttt', 'kv_ttt', 'kv_ttt', None),
             ('noise_ttt', 'noise_ttt', 'noise_ttt', None)]
    sap_modes = [(mode, mode, 'sap_online', None) for mode in SAP_MODES]
    binding_modes = [(mode, mode, None if mode == 'binding_off' else 'binding_online', None)
                     for mode in BINDING_MODES]
    ablation_modes = [(f'{mode}_{ablation}', mode, mode, ablation)
                      for mode in ('kv_ttt', 'noise_ttt')
                      for ablation in ('reset', 'no_protection', 'shuffle')]
    if modes is None:
        if ablations:
            available += ablation_modes
    else:
        available += ablation_modes + sap_modes + binding_modes
        requested = set(modes)
        known = {name for name, *_ in available}
        if unknown := requested - known:
            raise ValueError(f'Unknown evaluation modes: {sorted(unknown)}')
        if len(modes) != len(requested):
            raise ValueError('Duplicate evaluation mode')
        available = [entry for name in modes for entry in available if entry[0] == name]
    results = []
    split = settings.get('benchmark_split', 'worldttt')
    for name, mode, adapter_source, ablation in available:
        method = root / name
        videos = method / split
        videos.mkdir(parents=True, exist_ok=True)
        selected_adapter = settings.get('adapters', {}).get(adapter_source, adapter) if adapter_source else None
        if adapter_source and not selected_adapter:
            raise ValueError(f'Missing trained adapter for {adapter_source}')
        for case in cases:
            run = method / 'runs' / case['id']
            request = request_record(settings, case, mode, selected_adapter, ablation)
            reuse = resume and run.exists() and reusable_run(run, request)
            if resume and not reuse and run.exists() and any(run.iterdir()):
                archive = root / '_interrupted' / name / (case['id'] + '-' + uuid.uuid4().hex[:8])
                if not run.resolve().is_relative_to(root) or not archive.resolve().is_relative_to(root):
                    raise ValueError('Interrupted run archive must stay inside the evaluation directory')
                archive.parent.mkdir(parents=True, exist_ok=True)
                run.rename(archive)
                print(f'[resume] Restarting interrupted case; preserved files at {archive}', flush=True)
            run.mkdir(parents=True, exist_ok=True)
            case_path = run / 'case.json'
            case_path.write_text(json.dumps(case, indent=2), encoding='utf-8')
            (run / 'evaluation_request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
            command = [sys.executable, '-m', 'worldttt', 'infer', '--settings', str(Path(settings_path).resolve()),
                       '--case', str(case_path), '--output', str(run), '--mode', mode]
            if selected_adapter:
                command += ['--adapter', str(selected_adapter)]
            if ablation:
                command += ['--ablation', ablation]
            if reuse:
                print(f'[resume] Reusing complete video: {name}/{case["id"]}', flush=True)
            else:
                subprocess.run(command, check=True)
            if settings.get('address_analysis', False):
                from .address_analysis import analyze as analyze_address
                analyze_address(run)
            if settings.get('relative_revisit', False):
                from .relative_revisit import analyze as analyze_relative
                analyze_relative(run)
            shutil.copy2(run / 'video_generated.mp4', videos / f'{case["id"]}_generated.mp4')
            if settings.get('refiner'):
                stage1_dir = root / (name + '_stage1') / split
                stage1_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run / 'stage1_generated.mp4', stage1_dir / f'{case["id"]}_generated.mp4')
            metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
            updates = [json.loads(s) for s in (run / 'updates.jsonl').read_text().splitlines()]
            results.append(dict(experiment=name, id=case['id'], metrics=metrics, updates=updates))
        info = dict(model='WorldTTT', variant=name, fps=16, refiner=settings.get('refiner'))
        (method / 'method_info.json').write_text(json.dumps(info), encoding='utf-8')
        if video_metrics:
            run_video_metrics(settings, method, split)
            if settings.get('refiner'):
                stage1_method = root / (name + '_stage1')
                (stage1_method / 'method_info.json').write_text(json.dumps(dict(info, refiner=None)), encoding='utf-8')
                run_video_metrics(settings, stage1_method, split)
    report = dict(runs=results, video_metrics='upstream per-method eval directories' if video_metrics else 'not measured',
                  conclusion='No effectiveness claim is inferred from support loss.')
    (root / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


def run_video_metrics(settings, method, split):
    # Native camera evaluator needs GT benchmark NPZs and its run manifest;
    # fail explicitly instead of accepting eval_unified's empty camera result.
    for key in ('benchmark_meta', 'benchmark_manifest'):
        if not settings.get(key) or not Path(settings[key]).is_file():
            raise ValueError(f'--video-metrics requires {key}')
    subprocess.run([sys.executable, 'tools/metrics/sana_wm/eval_benchmark_poses.py',
        '--result_folder', str(method / split), '--manifest', settings['benchmark_manifest'],
        '--pi3_ckpt', settings.get('pi3_checkpoint', 'yyfz233/Pi3')], check=True)
    subprocess.run([sys.executable, 'tools/metrics/sana_wm/eval_unified.py', '--method_dir', str(method),
        '--split', split, '--benchmark_meta', settings['benchmark_meta'],
        '--metrics', 'vbench', 'revisit', 'camera', 'temporal', '--revisit_lpips'], check=True)
    expected = {p.name.removesuffix('_generated.mp4') for p in (method / split).glob('*_generated.mp4')}
    verify_video_metrics(method, split, expected)


def verify_video_metrics(method, split, expected):
    def read(path):
        if not path.is_file():
            raise ValueError(f'Missing evaluator output (exit status alone is insufficient): {path}')
        return json.loads(path.read_text(encoding='utf-8'))
    camera = read(method / split / 'eval_poses.json')
    if not expected or not expected.issubset(camera):
        raise ValueError(f'Incomplete camera metrics: {expected - camera.keys()}')
    for scene in expected:
        if any(not math.isfinite(camera[scene].get(k, float('nan'))) for k in ('RotErr', 'TransErr_rel', 'CamMC_rel')):
            raise ValueError(f'Nonfinite camera metrics: {scene}')
    summary = read(method / 'eval' / split / 'summary.json')
    if summary.get('vbench', {}).get('n_dimensions', 0) < 9:
        raise ValueError('Incomplete VBench dimensions')
    revisit = read(method / 'eval' / split / 'revisit_consistency.json')
    evaluated = revisit.get('per_scene', {})
    if not expected.issubset(evaluated) or revisit.get('summary', {}).get('n_lpips_pairs', 0) <= 0:
        raise ValueError('Missing revisit pairs or LPIPS results; use a benchmark with valid revisit trajectories')


def query_evaluate(settings, fixture_path, adapter, output, histories=('real', 'generated'),
                   modes=('off', 'frozen_kv', 'kv_ttt')):
    from .check_cache import require_gate
    from .episode import EpisodeModel
    from .memory import TTTConfig
    from .runtime import WorldTTTController
    from .sana import build_backbone, fixture_to_device, load_config
    require_gate(settings)
    config = load_config(settings['sana_config'])
    fixture_paths = [fixture_path] if isinstance(fixture_path, (str, Path)) else list(fixture_path)
    if not fixture_paths:
        raise ValueError('At least one held-out test fixture is required')
    rows = []
    available = {'off': ('off', None), 'frozen_kv': ('frozen', 'kv_ttt'),
                 'kv_ttt': ('kv_ttt', 'kv_ttt'), 'frozen_noise': ('frozen', 'noise_ttt'),
                 'noise_ttt': ('noise_ttt', 'noise_ttt')}
    if unknown := set(modes) - available.keys():
        raise ValueError(f'Unknown query modes: {sorted(unknown)}')
    if set(histories) - {'real', 'generated'}:
        raise ValueError('Histories must be real or generated')
    for name in modes:
        mode, source = available[name]
        model = build_backbone(config, settings['base_checkpoint'], 'cuda', torch.bfloat16)
        ctl = WorldTTTController(model, TTTConfig(**dict(settings['ttt'], mode=mode)))
        ctl.base_checkpoint = settings['base_checkpoint']
        if mode != 'off':
            selected_adapter = settings.get('adapters', {}).get(source, adapter)
            if not selected_adapter:
                raise ValueError(f'Missing trained adapter for {source}')
            ctl.load_checkpoint(selected_adapter)
        episode = EpisodeModel(model, ctl, settings.get('steps', 50), config.scheduler.inference_flow_shift)
        # This is a held-out FUTURE query; it is never passed to adapt().
        for fixture_index, path in enumerate(fixture_paths):
            saved = torch.load(path, map_location='cpu', weights_only=True)
            batch = fixture_to_device(saved, 'cuda', torch.bfloat16)
            for history in histories:
                loss = episode(**batch, generated=history == 'generated',
                               seed=settings.get('seed', 3407) + fixture_index, meta_grad=False)
                rows.append(dict(experiment=name, mode=mode, history=history, fixture=str(path),
                                 future_query_flow_mse=float(loss.detach()), updates=list(ctl.metrics)))
        del loss, episode, ctl, model
        gc.collect()
        torch.cuda.empty_cache()
    Path(output).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
