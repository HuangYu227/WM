"""Paired SAP video report; no automatic memory-effectiveness claim."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from worldttt.a800_report import paired_delta


MODES = ('off', 'sap_frozen', 'sap_online')
METRICS = ('lpips', 'psnr', 'ssim')


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def _mean(rows, field, metric):
    values = [float(row[field][metric]) for row in rows]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Empty or nonfinite {field}.{metric}')
    return float(np.mean(values))


def build_report(evaluation, *, split='simple_60s'):
    root = Path(evaluation)
    comparison = _read(root / 'comparison.json')
    by_case = defaultdict(dict)
    raw = {}
    for entry in comparison['runs']:
        mode = entry['experiment']
        if mode not in MODES:
            continue
        case = entry['metrics']['case']
        key = f'{entry["id"]}|{case["seed"]}'
        if mode in by_case[key]:
            raise ValueError(f'Duplicate {mode}/{key}')
        result = _read(root / mode / 'runs' / entry['id'] / 'relative_revisit.json')
        by_case[key][mode] = (entry, result)
        raw.setdefault(key, {})[mode] = result['rows']

    coverage, scores, costs, updates, invalid = {}, defaultdict(lambda: defaultdict(dict)), {}, {}, []
    camera_status = 'verified'
    camera = {}
    for mode in MODES:
        path = root / mode / split / 'eval_poses.json'
        if path.is_file():
            camera[mode] = _read(path)
        else:
            camera_status = 'unverified'
    for key, modes in sorted(by_case.items()):
        if set(modes) != set(MODES):
            invalid.append({'case': key, 'reason': 'missing_mode', 'modes': sorted(modes)})
            continue
        entries = {mode: modes[mode][0] for mode in MODES}
        results = {mode: modes[mode][1] for mode in MODES}
        reference = entries['off']['metrics']
        for mode in MODES:
            candidate = entries[mode]['metrics']
            for name in ('case', 'base_checkpoint', 'generation', 'refiner'):
                if candidate.get(name) != reference.get(name):
                    raise ValueError(f'Unpaired {name} for {key}')
        if entries['sap_frozen']['metrics'].get('adapter') != entries['sap_online']['metrics'].get('adapter'):
            raise ValueError(f'Frozen and online adapters differ for {key}')
        frame_lists = [[(row['revisit'], row.get('control')) for row in results[m]['rows']] for m in MODES]
        if any(items != frame_lists[0] for items in frame_lists[1:]):
            raise ValueError(f'Preselected frame pairs differ for {key}')
        if len({tuple(pair[0]) for pair in frame_lists[0]}) != len(frame_lists[0]):
            raise ValueError(f'Duplicate revisit pair for {key}')
        total = len(frame_lists[0])
        valid = [i for i in range(total) if all(results[m]['rows'][i].get('revisit_valid') for m in MODES)]
        control = [i for i in valid if all(results[m]['rows'][i].get('valid') for m in MODES)]
        coverage[key] = {'total_pairs': total, 'revisit_pairs': len(valid),
                         'control_pairs': len(control),
                         'invalid': [{'pair': frame_lists[0][i][0],
                                      'reasons': {m: results[m]['rows'][i].get('reason') for m in MODES}}
                                     for i in range(total) if i not in valid]}
        if not valid:
            invalid.append({'case': key, 'reason': 'no_common_valid_revisit'})
            continue
        for mode in MODES:
            rows = results[mode]['rows']
            video = results[mode].get('video_validation', {})
            if not video.get('complete') or video.get('decoded_frames') != reference['case']['num_frames']:
                invalid.append({'case': key, 'mode': mode, 'reason': 'incomplete_video'})
            for metric in METRICS:
                scores[mode][metric][key] = _mean([rows[i] for i in valid], 'revisit_metrics', metric)
                if control:
                    scores[mode][f'control_{metric}'][key] = _mean(
                        [rows[i] for i in control], 'control_metrics', metric)
            if control and all(rows[i].get('nonreturn_segment') for i in control):
                for name in ('frame_change_mean', 'laplacian_variance_mean'):
                    scores[mode][name][key] = _mean([rows[i] for i in control], 'nonreturn_segment', name)
            metrics = entries[mode]['metrics']
            costs.setdefault(key, {})[mode] = {name: metrics.get(name) for name in
                ('stage1_seconds', 'end_to_end_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}
            events = entries[mode].get('updates', [])
            expected = ((reference['case']['num_frames'] - 1) // 8 + 1) // 3
            committed = (len(events) == expected and
                         [row.get('chunk') for row in events] == list(range(expected)) and
                         all(row.get('committed') for row in events))
            updates.setdefault(key, {})[mode] = {'total': len(events), 'expected': expected,
                                                  'committed_all': bool(committed)}
            if camera_status == 'verified':
                scene = entries[mode]['id']
                pose = camera[mode].get(scene)
                if pose is None or any(not math.isfinite(float(pose.get(name, float('nan'))))
                                       for name in ('RotErr', 'TransErr_rel', 'CamMC_rel')):
                    camera_status = 'unverified'

    scene_scores = {}
    for mode, fields in scores.items():
        scene_scores[mode] = {}
        for metric, values in fields.items():
            grouped = defaultdict(list)
            for key, score in values.items():
                grouped[key.split('|', 1)[0]].append(score)
            scene_scores[mode][metric] = {name: float(np.mean(scores)) for name, scores in grouped.items()}
    paired = {}
    for reference in ('sap_frozen', 'off'):
        paired[f'sap_online_vs_{reference}'] = {
            metric: paired_delta(scene_scores.get(reference, {}).get(metric, {}),
                                 scene_scores.get('sap_online', {}).get(metric, {}))
            for metric in (*METRICS, *(f'control_{m}' for m in METRICS),
                           'frame_change_mean', 'laplacian_variance_mean')}
    online_ok = bool(coverage) and all(updates.get(key, {}).get('sap_online', {}).get('committed_all')
                                           for key in coverage)
    return {'label': 'SAP within-video relative revisit; not an official benchmark result',
            'coverage': coverage, 'invalid': invalid, 'raw_pairs': raw,
            'case_scores': {mode: dict(fields) for mode, fields in scores.items()},
            'scene_scores': scene_scores, 'paired': paired, 'updates': updates,
            'online_updates_committed': online_ok, 'costs': costs,
            'camera_status': camera_status, 'camera': camera,
            'conclusion': 'No memory claim without independent query, camera and nonreturn checks.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='simple_60s')
    args = parser.parse_args()
    report = build_report(args.evaluation, split=args.split)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'coverage': report['coverage'], 'paired': report['paired'],
                      'camera_status': report['camera_status']}, indent=2))


if __name__ == '__main__':
    main()
