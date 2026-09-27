"""Export deterministic A/B/C/A' camera cases; keep labels out of model inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .scene import ProceduralScene


def export_scene(seed: int, root: str | Path, *, height=704, width=1280):
    if height % 32 or width % 32:
        raise ValueError('SANA VAE spatial dimensions must be divisible by 32')
    root = Path(root)
    folder = root / f'scene_{seed:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    scene = ProceduralScene(seed, height, width)
    first = scene.render(0)
    Image.fromarray(first['rgb']).save(folder / 'first.png')
    cameras = np.stack([scene.camera(i) for i in range(97)])
    np.save(folder / 'camera.npy', cameras)
    K = scene.intrinsics
    intrinsics = np.repeat(np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]],
                                    dtype=np.float32), 97, axis=0)
    np.save(folder / 'intrinsics.npy', intrinsics)
    latent_scene = ProceduralScene(seed, height // 32, width // 32)
    labels = [latent_scene.render(i * 8) for i in range(13)]
    np.savez_compressed(folder / 'supervision.npz',
                        instance=np.stack([r['instance'] for r in labels]),
                        depth=np.stack([r['depth'] for r in labels]),
                        world=np.stack([r['world'] for r in labels]),
                        visible=np.stack([r['instance'] > 0 for r in labels]))
    case = dict(id=f'scene_{seed:06d}', image=str((folder / 'first.png').resolve()),
                camera=str((folder / 'camera.npy').resolve()),
                intrinsics=str((folder / 'intrinsics.npy').resolve()),
                prompt=scene.prompt, num_frames=97, seed=seed, condition=scene.condition,
                revisit_pairs=[{'frame_a': 8, 'frame_b': 80, 'control': [24, 96]}])
    # The evaluator finds its separate supervision.npz in this folder.
    (folder / 'case.json').write_text(json.dumps(case, indent=2), encoding='utf-8')
    return case


def export_split(root: str | Path, *, seed_base: int, counts: tuple[int, int, int],
                 height=704, width=1280):
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError('Need positive train/val/test scene counts')
    root = Path(root)
    rows = []
    for offset, split in zip((0, counts[0], counts[0] + counts[1]), ('train', 'val', 'test')):
        count = counts[('train', 'val', 'test').index(split)]
        for seed in range(seed_base + offset, seed_base + offset + count):
            rows.append(dict(export_scene(seed, root, height=height, width=width), split=split))
    for split in ('train', 'val', 'test'):
        with (root / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                if row['split'] == split:
                    stream.write(json.dumps(row) + '\n')
    (root / 'scene_split.json').write_text(json.dumps({
        'seed_base': seed_base, 'counts': dict(zip(('train', 'val', 'test'), counts)),
        'scenes': [{'id': row['id'], 'seed': row['seed'], 'split': row['split'],
                    'condition': row['condition']} for row in rows]}, indent=2), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Build fixed procedural SAP-TTT SANA cases')
    parser.add_argument('--output', required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--seeds', type=int, nargs='+')
    selection.add_argument('--split-counts', type=int, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--seed-base', type=int, default=10000)
    parser.add_argument('--height', type=int, default=704)
    parser.add_argument('--width', type=int, default=1280)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.split_counts:
        cases = export_split(root, seed_base=args.seed_base, counts=tuple(args.split_counts),
                             height=args.height, width=args.width)
    else:
        cases = [export_scene(seed, root, height=args.height, width=args.width) for seed in args.seeds]
    with (root / 'cases.jsonl').open('w', encoding='utf-8') as stream:
        for case in cases:
            stream.write(json.dumps(case) + '\n')
    print(f'Exported {len(cases)} scenes to {root}')


if __name__ == '__main__':
    main()
