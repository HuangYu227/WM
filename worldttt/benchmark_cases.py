"""Deterministic SANA-WM simple-60s case selection before model evaluation."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_cases(bench_root, output, min_gap=240, per_category=2, seeds=(42, 3407), l20=False):
    bench_root, output = Path(bench_root), Path(output)
    split = bench_root / 'benchmark_v2_smooth_60s'
    meta = json.loads((split / 'scene_trajectories_v2.json').read_text(encoding='utf-8'))
    manifest = {row['id']: row for line in (split / 'sanawm_export_v2/run_manifest.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip() for row in (json.loads(line),)}
    eligible = defaultdict(list)
    selected_pairs = {}
    excluded = {}
    for scene in meta['scenes']:
        scene_id = scene['scene_id']
        pairs = [p for p in scene.get('evaluation_pairs', [])
                 if (0 < int(p['frame_a']) if l20 else 0 <= int(p['frame_a']))
                 and int(p['frame_a']) < int(p['frame_b']) < 961
                 and int(p['frame_b']) - int(p['frame_a']) >= min_gap]
        if l20 and scene_id in manifest:
            pairs = sorted(pairs, key=lambda p: (p.get('quality_score', 999), p['frame_a'], p['frame_b']))
            if pairs:
                from .relative_revisit import choose_control
                with np.load(bench_root / manifest[scene_id]['camera_path'], allow_pickle=False) as trajectory:
                    camera = trajectory['c2w']
                pairs = [dict(p, control=choose_control(camera, int(p['frame_a']), int(p['frame_b'])))
                         for p in pairs]
            pairs = [p for p in pairs if p['control'] is not None][:5]
            if not pairs:
                excluded[scene_id] = 'no_valid_long_revisit_control'
                continue
        if pairs and scene_id in manifest:
            category = scene_id.rsplit('_', 1)[0]
            eligible[category].append(scene_id)
            selected_pairs[scene_id] = pairs
    limit = 1 if l20 else per_category
    chosen = [scene_id for category in sorted(eligible)
              for scene_id in sorted(eligible[category])[:limit]][:4 if l20 else 8]
    if not chosen:
        raise ValueError('No benchmark scenes have a qualifying long-gap revisit pair')
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for scene_id in chosen:
        row = manifest[scene_id]
        image = (bench_root / row['image_path']).resolve()
        npz = (bench_root / row['camera_path']).resolve()
        if not image.is_file() or not npz.is_file():
            raise FileNotFoundError(f'Missing benchmark input for {scene_id}: {image} or {npz}')
        with np.load(npz, allow_pickle=False) as traj:
            c2w, intrinsics = traj['c2w'], traj['intrinsics']
        if c2w.shape != (961, 4, 4) or intrinsics.shape != (961, 3, 3):
            raise ValueError(f'Expected 961-frame camera/intrinsics for {scene_id}')
        if not np.isfinite(c2w).all() or not np.isfinite(intrinsics).all():
            raise ValueError(f'Nonfinite camera/intrinsics for {scene_id}')
        camera_path = output / f'{scene_id}_c2w.npy'
        intrinsics_path = output / f'{scene_id}_intrinsics.npy'
        np.save(camera_path, c2w)
        np.save(intrinsics_path, intrinsics)
        cases.append(dict(id=scene_id, image=str(image), prompt=row['prompt'],
                          camera=str(camera_path.resolve()), intrinsics=str(intrinsics_path.resolve()),
                          num_frames=961, seed=seeds[0], revisit_pairs=selected_pairs[scene_id]))
    for seed in seeds:
        case_path = output / f'cases-seed-{seed}.jsonl'
        case_path.write_text(''.join(json.dumps(dict(case, seed=seed)) + '\n' for case in cases), encoding='utf-8')
    (output / 'selection.json').write_text(json.dumps(dict(split='simple_60s', profile='l20' if l20 else 'default',
        excluded=excluded, min_gap=min_gap,
        per_category=limit, scene_order=chosen,
        qualifying_pairs={key: selected_pairs[key] for key in chosen}), indent=2), encoding='utf-8')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--l20', action='store_true', help='Select four metadata-only long-return scenes for L20')
    args = parser.parse_args()
    cases = build_cases(args.bench, args.output, l20=args.l20)
    print(f'Preselected {len(cases)} scenes: {", ".join(case["id"] for case in cases)}')


if __name__ == '__main__':
    main()
