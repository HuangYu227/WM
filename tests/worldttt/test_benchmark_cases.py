import json

import numpy as np


def test_preselection_and_case_conversion_are_deterministic(tmp_path):
    from worldttt.benchmark_cases import build_cases
    bench = tmp_path / 'bench'
    split = bench / 'benchmark_v2_smooth_60s'
    export = split / 'sanawm_export_v2'
    export.mkdir(parents=True)
    (bench / 'images').mkdir()
    scenes, rows = [], []
    for group in ('a', 'b'):
        for index in range(1, 4):
            name = f'{group}_{index:03d}'
            (bench / 'images' / f'{name}.png').write_bytes(b'image')
            np.savez(export / f'{name}.npz', c2w=np.broadcast_to(np.eye(4), (961, 4, 4)),
                     intrinsics=np.broadcast_to(np.eye(3), (961, 3, 3)))
            rows.append({'id': name, 'image_path': f'images/{name}.png',
                         'camera_path': f'benchmark_v2_smooth_60s/sanawm_export_v2/{name}.npz',
                         'prompt': name})
            gap = 100 if index == 1 else 300
            scenes.append({'scene_id': name, 'evaluation_pairs': [{'frame_a': 10, 'frame_b': 10 + gap}]})
    (split / 'scene_trajectories_v2.json').write_text(json.dumps({'scenes': scenes}))
    (export / 'run_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = build_cases(bench, tmp_path / 'out')
    assert [r['id'] for r in result] == ['a_002', 'a_003', 'b_002', 'b_003']
    assert all(r['num_frames'] == 961 and r['seed'] == 42 for r in result)
    assert np.load(result[0]['camera']).shape == (961, 4, 4)
    assert np.load(result[0]['intrinsics']).shape == (961, 3, 3)
    assert len((tmp_path / 'out' / 'cases-seed-3407.jsonl').read_text().splitlines()) == 4
    assert result[0]['revisit_pairs'] == [{'frame_a': 10, 'frame_b': 310}]


def test_l20_selection_excludes_input_frame_and_freezes_controls(tmp_path):
    from worldttt.benchmark_cases import build_cases

    bench = tmp_path / 'bench'
    split = bench / 'benchmark_v2_smooth_60s'
    export = split / 'sanawm_export_v2'
    export.mkdir(parents=True)
    (bench / 'images').mkdir()
    rows, scenes = [], []
    for name in ('a_001', 'a_002', 'b_001'):
        (bench / 'images' / f'{name}.png').write_bytes(b'image')
        camera = np.broadcast_to(np.eye(4), (961, 4, 4)).copy()
        camera[:, 0, 3] = np.arange(961) / 100
        camera[320, 0, 3] = camera[16, 0, 3]
        np.savez(export / f'{name}.npz', c2w=camera,
                 intrinsics=np.broadcast_to(np.eye(3), (961, 3, 3)))
        rows.append(dict(id=name, image_path=f'images/{name}.png',
                         camera_path=f'benchmark_v2_smooth_60s/sanawm_export_v2/{name}.npz',
                         prompt=name))
        pairs = [{'frame_a': 0, 'frame_b': 320, 'quality_score': 0}]
        if name != 'a_001':
            pairs.append({'frame_a': 16, 'frame_b': 320, 'quality_score': 1})
        scenes.append(dict(scene_id=name, evaluation_pairs=pairs))
    (split / 'scene_trajectories_v2.json').write_text(json.dumps({'scenes': scenes}))
    (export / 'run_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    result = build_cases(bench, tmp_path / 'out', l20=True)
    assert [r['id'] for r in result] == ['a_002', 'b_001']
    assert all(len(r['revisit_pairs']) == 1 for r in result)
    assert all(r['revisit_pairs'][0]['control'] is not None
               and r['revisit_pairs'][0]['control'][0] > 0 for r in result)
    selection = json.loads((tmp_path / 'out' / 'selection.json').read_text())
    assert selection['scene_order'] == ['a_002', 'b_001']
    assert selection['excluded']['a_001'] == 'no_valid_long_revisit_control'
