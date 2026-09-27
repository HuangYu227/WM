import json

import numpy as np
import pytest


def test_scene_export_has_sana_case_and_separate_supervision(tmp_path):
    from worldttt.sap_ttt.export import export_scene

    case = export_scene(11, tmp_path, height=256, width=384)
    assert set(case) >= {'id', 'image', 'camera', 'intrinsics', 'prompt', 'num_frames'}
    assert case['num_frames'] == 97
    assert np.load(case['camera']).shape == (97, 4, 4)
    assert np.load(case['intrinsics']).shape == (97, 4)
    labels = np.load(tmp_path / 'scene_000011' / 'supervision.npz')
    assert labels['instance'].shape == (13, 8, 12)
    assert labels['world'].shape == (13, 8, 12, 3)
    assert labels['visible'].shape == (13, 8, 12)
    assert 'instance' not in json.dumps(case)


def test_scene_splits_are_seeded_and_disjoint(tmp_path):
    from worldttt.sap_ttt.export import export_split

    rows = export_split(tmp_path, seed_base=100, counts=(2, 1, 1), height=256, width=384)
    assert len({row['id'] for row in rows}) == 4
    assert [row['split'] for row in rows] == ['train', 'train', 'val', 'test']
    assert set(row['condition'] for row in rows) == {
        'relation_same_instance', 'ambiguous_same_instance',
        'relation_hard_negative', 'ambiguous_hard_negative'}


def test_feature_encoder_rejects_stale_procedural_camera(tmp_path):
    from worldttt.sap_ttt.export import export_scene
    from worldttt.sap_ttt.features import validate_scene_case

    case = export_scene(11, tmp_path, height=256, width=384)
    validate_scene_case(case, 256, 384)
    camera = np.load(case['camera'])
    camera[0, 0, 3] += 1
    np.save(case['camera'], camera)
    with pytest.raises(ValueError, match='re-export'):
        validate_scene_case(case, 256, 384)
