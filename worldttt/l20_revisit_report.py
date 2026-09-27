"""Paired long-return report for the trained L20 Noise-TTT adapter."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .a800_report import paired_delta


MODES = ('off', 'frozen_noise', 'noise_ttt')
METRICS = ('lpips', 'psnr', 'ssim')


def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def _score(row, field, metric):
    values = row.get(field)
    if values is None:
        return None
    value = float(values[metric])
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {field}.{metric} for revisit pair {row["revisit"]}')
    return value


def _scene_means(case_scores):
    by_scene = defaultdict(list)
    for key, value in case_scores.items():
        by_scene[key.split('|', 1)[0]].append(value)
    return {scene: float(np.mean(values)) for scene, values in by_scene.items()}


def build_report(evaluations, split='simple_60s'):
    """Only pair the same preselected frames, scene and seed across modes."""
    cases, pairs, updates, costs = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    videos = defaultdict(dict)
    case_specs, generation_specs, adapter_specs = {}, {}, {}
    camera_ok = quality_ok = True
    camera, quality_motion = defaultdict(dict), {}
    for evaluation in map(Path, evaluations):
        comparison = _read(evaluation / 'comparison.json')
        if comparison is None:
            raise FileNotFoundError(f'Missing comparison.json in {evaluation}')
        evaluation_keys = defaultdict(dict)
        for entry in comparison['runs']:
            mode = entry['experiment']
            if mode not in MODES:
                continue
            scene = entry['id']
            seed = entry['metrics']['case']['seed']
            key = f'{scene}|{seed}'
            evaluation_keys[mode][scene] = key
            if mode in cases[key]:
                raise ValueError(f'Duplicate {mode} result for {key}')
            if key in case_specs and entry['metrics']['case'] != case_specs[key]:
                raise ValueError(f'Generation inputs differ across modes for {key}')
            case_specs[key] = entry['metrics']['case']
            generation = {field: entry['metrics'].get(field) for field in
                          ('generation', 'base_checkpoint', 'gpu', 'torch', 'refiner')}
            if key in generation_specs and generation != generation_specs[key]:
                raise ValueError(f'Generation settings differ across modes for {key}')
            generation_specs[key] = generation
            if mode in ('frozen_noise', 'noise_ttt'):
                adapter = entry['metrics'].get('adapter')
                if key in adapter_specs and adapter != adapter_specs[key]:
                    raise ValueError(f'Frozen and online adapters differ for {key}')
                adapter_specs[key] = adapter
            result = _read(evaluation / mode / 'runs' / scene / 'relative_revisit.json')
            if result is None:
                raise FileNotFoundError(f'Missing relative_revisit.json for {mode}/{key}')
            rows = result['rows']
            videos[key][mode] = result.get('video_validation')
            if not rows:
                raise ValueError(f'No long revisit pairs for {mode}/{key}')
            identifiers = [tuple(r['revisit']) for r in rows]
            if len(set(identifiers)) != len(identifiers) or any(a <= 0 or b - a < 240 for a, b in identifiers):
                raise ValueError(f'Invalid or duplicate long revisit pairs for {mode}/{key}')
            cases[key][mode] = rows
            pairs[key][mode] = rows
            events = entry.get('updates', [])
            frames = entry['metrics']['case'].get('num_frames')
            block = (entry['metrics'].get('generation') or {}).get('num_frame_per_block')
            expected_chunks = (((int(frames) - 1) // 8 + 1) // int(block)) if frames and block else None
            complete_updates = (mode == 'noise_ttt' and expected_chunks is not None
                                and len(events) == expected_chunks
                                and sorted(e.get('chunk') for e in events) == list(range(expected_chunks))
                                and all(e.get('committed') is True for e in events))
            updates[key][mode] = dict(total=len(events), committed=sum(bool(e.get('committed')) for e in events),
                                      expected_chunks=expected_chunks if mode == 'noise_ttt' else None,
                                      all_chunks_committed=complete_updates,
                                      failures=[e for e in events if not e.get('committed')])
            metrics = entry['metrics']
            costs[key][mode] = {name: metrics.get(name) for name in
                                ('stage1_seconds', 'end_to_end_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}
        for mode in MODES:
            pose = _read(evaluation / mode / split / 'eval_poses.json')
            summary = _read(evaluation / mode / 'eval' / split / 'summary.json')
            expected = evaluation_keys[mode]
            if pose is None or not expected.keys() <= pose.keys():
                camera_ok = False
            else:
                camera[mode].update({key: pose[scene] for scene, key in expected.items()})
            if summary is None or summary.get('vbench', {}).get('n_dimensions', 0) < 9 or 'temporal_degradation' not in summary:
                quality_ok = False
            else:
                quality_motion[f'{evaluation}|{mode}'] = dict(
                    vbench=summary['vbench'], temporal_degradation=summary['temporal_degradation'])

    coverage, scores = {}, {mode: defaultdict(dict) for mode in MODES}
    incomplete = []
    for key, modes in sorted(cases.items()):
        if set(modes) != set(MODES):
            incomplete.append(dict(case=key, reason='missing_mode', modes=sorted(modes)))
            continue
        frame_lists = [[tuple(r['revisit']) for r in modes[mode]] for mode in MODES]
        if any(ids != frame_lists[0] for ids in frame_lists[1:]):
            raise ValueError(f'Preselected revisit pairs differ across modes for {key}')
        controls = [[r.get('control') for r in modes[mode]] for mode in MODES]
        if any(items != controls[0] for items in controls[1:]):
            raise ValueError(f'Preselected nonreturn controls differ across modes for {key}')
        valid = [i for i in range(len(frame_lists[0]))
                 if all(modes[mode][i].get('revisit_valid', modes[mode][i].get('valid', False)) for mode in MODES)]
        control = [i for i in valid if all(modes[mode][i].get('valid') for mode in MODES)]
        coverage[key] = dict(total_pairs=len(frame_lists[0]), revisit_pairs=len(valid), control_pairs=len(control),
                             invalid=[dict(pair=list(frame_lists[0][i]),
                                           reasons={mode: modes[mode][i].get('reason') for mode in MODES})
                                      for i in range(len(frame_lists[0])) if i not in valid],
                             missing_controls=[dict(pair=list(frame_lists[0][i]),
                                                    reasons={mode: modes[mode][i].get('reason') for mode in MODES})
                                               for i in valid if i not in control])
        if not valid:
            incomplete.append(dict(case=key, reason='no_common_valid_revisit_pair'))
            continue
        for mode in MODES:
            for metric in METRICS:
                revisit = float(np.mean([_score(modes[mode][i], 'revisit_metrics', metric) for i in valid]))
                scores[mode][metric][key] = revisit
                if control:
                    scores[mode][f'control_{metric}'][key] = float(np.mean([
                        _score(modes[mode][i], 'control_metrics', metric) for i in control]))
                    gap = [_score(modes[mode][i], 'revisit_metrics', metric)
                           - _score(modes[mode][i], 'control_metrics', metric) for i in control]
                    scores[mode][f'{metric}_return_minus_control'][key] = float(np.mean(gap))
            if control and all(modes[mode][i].get('nonreturn_segment') for i in control):
                for name, field in (('segment_frame_change', 'frame_change_mean'),
                                    ('segment_sharpness', 'laplacian_variance_mean')):
                    scores[mode][name][key] = float(np.mean([
                        _score(modes[mode][i], 'nonreturn_segment', field) for i in control]))

    scene_scores = {mode: {metric: _scene_means(values) for metric, values in metrics.items()}
                    for mode, metrics in scores.items()}
    paired = {}
    for reference in ('frozen_noise', 'off'):
        comparisons = {}
        for metric in (*METRICS, *(f'control_{m}' for m in METRICS),
                       *(f'{m}_return_minus_control' for m in METRICS),
                       'segment_frame_change', 'segment_sharpness'):
            common = scores[reference][metric].keys() & scores['noise_ttt'][metric].keys()
            comparisons[metric] = paired_delta(
                _scene_means({key: scores[reference][metric][key] for key in common}),
                _scene_means({key: scores['noise_ttt'][metric][key] for key in common}))
        paired[f'noise_ttt_vs_{reference}'] = comparisons
    camera_ok = camera_ok and all(coverage.keys() <= camera[mode].keys() for mode in MODES)
    if camera_ok:
        for mode in MODES:
            for scene, values in camera[mode].items():
                if any(not math.isfinite(float(values.get(name, float('nan'))))
                       for name in ('RotErr', 'TransErr_rel', 'CamMC_rel')):
                    camera_ok = False
                    break
    online_committed = all(updates[key].get('noise_ttt', {}).get('all_chunks_committed', False) for key in coverage)
    video_complete = all(videos[key].get(mode) is not None
                         and videos[key][mode].get('complete') is True
                         and videos[key][mode].get('decoded_frames') == case_specs[key].get('num_frames')
                         for key in coverage for mode in MODES)
    control_available = all(row['control_pairs'] > 0 for row in coverage.values())
    segment_ok = all(key in scores[mode]['segment_frame_change']
                     and key in scores[mode]['segment_sharpness']
                     for key in coverage for mode in MODES)
    trajectory_frames = {key: case_specs[key].get('num_frames') for key in case_specs}
    official_length = bool(trajectory_frames) and all(length == 961 for length in trajectory_frames.values())
    exploratory_valid = (bool(coverage) and not incomplete and online_committed and video_complete
                         and control_available and segment_ok)
    return dict(label='Long-gap within-video revisit; exploratory until camera/quality checks pass',
                modes=list(MODES), coverage=coverage, incomplete=incomplete, pairs=dict(pairs),
                scores={mode: dict(values) for mode, values in scores.items()},
                scene_scores=scene_scores, paired=paired,
                updates=dict(updates), costs=dict(costs), video_validation=dict(videos), camera=dict(camera),
                quality_motion=quality_motion,
                trajectory_frames=trajectory_frames, full_60s_length=official_length,
                camera_status='verified' if camera_ok else 'unverified',
                nonreturn_segment_status='measured' if segment_ok and control_available else 'unverified',
                quality_motion_status='verified' if quality_ok and segment_ok and control_available else 'unverified',
                online_updates_committed=online_committed,
                exploratory_valid=exploratory_valid,
                pilot_valid=exploratory_valid and official_length,
                conclusion='No effectiveness claim is inferred automatically; require camera and quality checks.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='simple_60s')
    args = parser.parse_args()
    report = build_report(args.evaluation, args.split)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(coverage=report['coverage'], paired=report['paired'],
                          camera_status=report['camera_status'], pilot_valid=report['pilot_valid']), indent=2))


if __name__ == '__main__':
    main()
