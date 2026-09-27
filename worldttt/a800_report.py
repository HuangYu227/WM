"""Paired A800 mechanism summary with scene bootstrap intervals."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_delta(reference, online, seed=3407):
    common = sorted(reference.keys() & online.keys())
    if not common:
        return dict(scenes=[], missing_online=sorted(reference), mean_online_minus_reference=None, ci95=None)
    differences = np.asarray([online[s] - reference[s] for s in common], dtype=float)
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(common), (10000, len(common)))].mean(axis=1)
    return dict(scenes=common, missing_online=sorted(reference.keys() - online.keys()),
                mean_online_minus_reference=float(differences.mean()),
                ci95=[float(x) for x in np.quantile(boot, [.025, .975])],
                per_scene={s: float(d) for s, d in zip(common, differences)})


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def metric_maps(root, modes, split, stage1=False):
    values = {}
    for mode in modes:
        method = root / (mode + '_stage1' if stage1 else mode)
        native = read_json(method / 'eval' / split / 'revisit_consistency.json')
        if native:
            values[mode] = {metric: {scene: row[key] for scene, row in native['per_scene'].items() if key in row}
                            for metric, key in [('psnr', 'mean_psnr'), ('ssim', 'mean_ssim'),
                                                ('lpips', 'mean_lpips')]}
    return values


def run_report(root, query=None, split='simple_60s'):
    root = Path(root)
    selected = read_json(root / 'comparison.json')
    if selected is None:
        raise FileNotFoundError('comparison.json is required')
    modes = sorted({row['experiment'] for row in selected['runs']})
    report = dict(label='A800 exploratory paired analysis; no automatic effectiveness claim',
                  completed_modes=modes, stage1_from_refiner={}, output_video={}, query={},
                  relative={}, relative_latent={}, camera={}, address={},
                  sources=dict(native_metrics='SANA-WM eval_unified',
                               relative='gap-matched nonreturn within generated videos, not official R2M'))
    for stage1, section in ((True, 'stage1_from_refiner'), (False, 'output_video')):
        values = metric_maps(root, modes, split, stage1=stage1)
        for online in ('kv_ttt', 'kv_ttt_reset', 'kv_ttt_shuffle'):
            if 'frozen_kv' in values and online in values:
                report[section][f'{online}_vs_frozen_kv'] = {
                    metric: paired_delta(values['frozen_kv'][metric], values[online][metric])
                    for metric in ('psnr', 'ssim', 'lpips')}
    for mode in modes:
        camera = read_json(root / mode / split / 'eval_poses.json')
        if camera:
            report['camera'][mode] = {key: {scene: float(v[key]) for scene, v in camera.items() if key in v}
                                      for key in ('RotErr', 'TransErr_rel', 'CamMC_rel')}
        relative = {}
        relative_latent = {}
        addressed = []
        for row in selected['runs']:
            if row['experiment'] != mode:
                continue
            result = read_json(root / mode / 'runs' / row['id'] / 'relative_revisit.json')
            if result:
                valid = [p for p in result['rows'] if p['valid']]
                if valid:
                    relative[row['id']] = {
                        prefix + '_' + metric: float(np.mean([p[prefix + '_metrics'][metric] for p in valid]))
                        for prefix in ('revisit', 'control') for metric in ('psnr', 'ssim', 'lpips')}
                    relative[row['id']].update({
                        'gap_' + metric: relative[row['id']]['revisit_' + metric]
                                        - relative[row['id']]['control_' + metric]
                        for metric in ('psnr', 'ssim', 'lpips')})
                    relative_latent[row['id']] = {
                        prefix + '_' + metric: float(np.mean([p[prefix + '_latent'][metric] for p in valid]))
                        for prefix in ('revisit', 'control') for metric in ('mse', 'cosine')}
                    relative_latent[row['id']].update({
                        'gap_' + metric: relative_latent[row['id']]['revisit_' + metric]
                                        - relative_latent[row['id']]['control_' + metric]
                        for metric in ('mse', 'cosine')})
            address = read_json(root / mode / 'runs' / row['id'] / 'address_analysis.json')
            if address:
                addressed.append(address)
        report['relative'][mode] = relative
        report['relative_latent'][mode] = relative_latent
        if addressed:
            valid = [r for scene in addressed for r in scene['rows'] if r['valid']]
            report['address'][mode] = dict(pairs=sum(scene['pairs'] for scene in addressed),
                valid_pairs=sum(scene['valid_pairs'] for scene in addressed),
                mean_mrr=float(np.mean([r['mrr'] for r in valid])) if valid else None,
                mean_shuffled_mrr=float(np.mean([r['shuffled_mrr'] for r in valid])) if valid else None,
                invalid_reasons={reason: sum(r.get('reason') == reason for scene in addressed for r in scene['rows'])
                                 for reason in sorted({r['reason'] for scene in addressed for r in scene['rows']
                                                       if not r['valid']})})
    if 'frozen_kv' in report['camera'] and 'kv_ttt' in report['camera']:
        report['camera']['kv_ttt_vs_frozen_kv'] = {
            metric: paired_delta(report['camera']['frozen_kv'][metric], report['camera']['kv_ttt'][metric])
            for metric in ('RotErr', 'TransErr_rel', 'CamMC_rel')}
    if 'frozen_kv' in report['relative'] and 'kv_ttt' in report['relative']:
        report['relative']['kv_ttt_vs_frozen_kv'] = {
            metric: paired_delta({s: v[metric] for s, v in report['relative']['frozen_kv'].items()},
                                 {s: v[metric] for s, v in report['relative']['kv_ttt'].items()})
            for metric in (prefix + '_' + name for prefix in ('revisit', 'control', 'gap')
                           for name in ('psnr', 'ssim', 'lpips'))}
        report['relative_latent']['kv_ttt_vs_frozen_kv'] = {
            metric: paired_delta({s: v[metric] for s, v in report['relative_latent']['frozen_kv'].items()},
                                 {s: v[metric] for s, v in report['relative_latent']['kv_ttt'].items()})
            for metric in (prefix + '_' + name for prefix in ('revisit', 'control', 'gap')
                           for name in ('mse', 'cosine'))}
    if query:
        rows = read_json(Path(query))
        for history in ('real', 'generated'):
            by_mode = {mode: {row['fixture']: row['future_query_flow_mse'] for row in rows
                              if row['experiment'] == mode and row['history'] == history}
                       for mode in modes}
            report['query'][history] = {f'{mode}_vs_frozen_kv': paired_delta(by_mode['frozen_kv'], by_mode[mode])
                                        for mode in ('off', 'kv_ttt') if mode in by_mode and 'frozen_kv' in by_mode}
    target = root / 'a800_paired_report.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--query')
    parser.add_argument('--split', default='simple_60s')
    args = parser.parse_args()
    run_report(args.evaluation, args.query, args.split)


if __name__ == '__main__':
    main()
