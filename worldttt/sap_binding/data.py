"""Paired anti-shortcut procedural episodes for SAP-Bind."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from worldttt.sap_ttt.scene import ProceduralScene


class BindingScene:
    """One layout with an independently permuted set of instance stickers."""
    def __init__(self, layout_seed, appearance_seed, height=128, width=224):
        self.layout_seed, self.appearance_seed = int(layout_seed), int(appearance_seed)
        self.base = ProceduralScene(self.layout_seed, height, width)
        rng = np.random.default_rng(self.appearance_seed)
        codes = np.asarray(self.base.sticker_codes)
        permutation = rng.permutation(len(codes))
        if np.array_equal(permutation, np.arange(len(codes))):
            permutation = np.roll(permutation, 1)
        self.permutation = tuple(int(x) for x in permutation)
        self.base.sticker_codes = tuple(int(x) for x in codes[permutation])

    def __getattr__(self, name):
        return getattr(self.base, name)

    def render(self, frame):
        result = self.base.render(frame)
        # A' keeps geometry and instance silhouette but cannot expose the
        # historical sticker. Labels are unchanged and never enter the Query.
        if frame >= 73:
            instance = result['instance']
            yy, xx = np.mgrid[:self.height, :self.width]
            rng = np.random.default_rng(self.appearance_seed + 1000003 * frame)
            phase = rng.integers(0, 11, size=7)
            texture = np.empty_like(result['rgb'])
            for identity in range(1, 7):
                stripe = ((xx + 2 * yy + phase[identity]) % 11 < 5).astype(np.uint8)
                color = np.stack((142 + 14 * stripe, 55 + 3 * stripe, 52 + 3 * stripe), -1)
                texture[instance == identity] = color[instance == identity]
            result['rgb'][instance > 0] = texture[instance > 0]
        return result


def export_episode(layout_seed, variant, root, *, height=704, width=1280):
    if variant not in (0, 1) or height % 32 or width % 32:
        raise ValueError('Binding episode needs variant 0/1 and VAE-aligned dimensions')
    appearance_seed = 100000 + 2 * int(layout_seed) + variant
    scene = BindingScene(layout_seed, appearance_seed, height, width)
    episode_id = f'binding_{layout_seed:06d}_{variant}'
    folder = Path(root) / episode_id
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scene.render(0)['rgb']).save(folder / 'first.png')
    cameras = np.stack([scene.camera(i) for i in range(97)])
    np.save(folder / 'camera.npy', cameras)
    k = scene.intrinsics
    intrinsics = np.repeat(np.array([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]]],
                                    dtype=np.float32), 97, 0)
    np.save(folder / 'intrinsics.npy', intrinsics)
    latent_scene = BindingScene(layout_seed, appearance_seed, height // 32, width // 32)
    labels = [latent_scene.render(i * 8) for i in range(13)]
    revisit_candidates = []
    for first in (8, 16, 24):
        for returned in (80, 88, 96):
            matched = len(latent_scene.correspondences(latent_scene.render(first),
                                                        latent_scene.render(returned)))
            revisit_candidates.append((-matched, first, returned))
    overlap, first, returned = min(revisit_candidates)
    np.savez_compressed(folder / 'supervision.npz',
        instance=np.stack([x['instance'] for x in labels]),
        depth=np.stack([x['depth'] for x in labels]), world=np.stack([x['world'] for x in labels]),
        visible=np.stack([x['instance'] > 0 for x in labels]),
        layout_seed=np.asarray(layout_seed), appearance_seed=np.asarray(appearance_seed),
        sticker_codes=np.asarray(scene.sticker_codes), permutation=np.asarray(scene.permutation))
    case = dict(id=episode_id, layout_id=f'layout_{layout_seed:06d}', variant=variant,
        layout_seed=layout_seed, appearance_seed=appearance_seed,
        image=str((folder / 'first.png').resolve()), camera=str((folder / 'camera.npy').resolve()),
        intrinsics=str((folder / 'intrinsics.npy').resolve()), prompt=scene.prompt,
        num_frames=97, seed=appearance_seed, condition=scene.condition,
        revisit_pairs=[dict(frame_a=first, frame_b=returned,
                            quality_score=overlap, preselected_latent_correspondences=-overlap)])
    (folder / 'case.json').write_text(json.dumps(case, indent=2), encoding='utf-8')
    return case


def export_dataset(root, *, layout_seed_base=20000, layouts=32, height=704, width=1280):
    if layouts != 32:
        raise ValueError('The registered SAP-Bind protocol uses exactly 32 base layouts')
    rows = []
    for index in range(layouts):
        split = 'train' if index < 24 else ('val' if index < 28 else 'test')
        for variant in (0, 1):
            rows.append(dict(export_episode(layout_seed_base + index, variant, root,
                                            height=height, width=width), split=split))
    root = Path(root)
    for split in ('train', 'val', 'test'):
        with (root / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                if row['split'] == split:
                    stream.write(json.dumps(row) + '\n')
    with (root / 'cases.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    manifest = dict(version=1, protocol='sap_binding_paired_v1', layout_seed_base=layout_seed_base,
                    split_layouts={'train': 24, 'val': 4, 'test': 4}, episodes=len(rows),
                    rows=[{k: r[k] for k in ('id', 'layout_id', 'variant', 'split')} for r in rows])
    (root / 'binding_split.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return rows


def validate_case(case, height, width):
    scene = BindingScene(case['layout_seed'], case['appearance_seed'], height, width)
    image = np.asarray(Image.open(case['image']).convert('RGB'))
    camera = np.load(case['camera'], allow_pickle=False)
    intrinsics = np.load(case['intrinsics'], allow_pickle=False)
    k = scene.intrinsics
    expected_k = np.repeat(np.array([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]]], dtype=np.float32), 97, 0)
    if (not np.array_equal(image, scene.render(0)['rgb']) or
            not np.array_equal(camera, np.stack([scene.camera(i) for i in range(97)])) or
            not np.array_equal(intrinsics, expected_k)):
        raise ValueError('SAP-Bind case files are stale or mixed')
    return scene, camera, intrinsics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--layout-seed-base', type=int, default=20000)
    parser.add_argument('--height', type=int, default=704)
    parser.add_argument('--width', type=int, default=1280)
    args = parser.parse_args()
    print(f'Exported {len(export_dataset(args.output, layout_seed_base=args.layout_seed_base, height=args.height, width=args.width))} episodes')


if __name__ == '__main__':
    main()
