import numpy as np
import torch


def test_procedural_revisit_is_exact_and_prompt_hides_instance_detail():
    from worldttt.sap_ttt.scene import ProceduralScene

    scene = ProceduralScene(seed=77, height=48, width=80)
    first = scene.render(0)
    revisit = scene.render(72)
    assert np.array_equal(first['rgb'], revisit['rgb'])
    assert not np.array_equal(scene.render(8)['rgb'], scene.render(80)['rgb'])
    assert np.array_equal(first['instance'], revisit['instance'])
    assert first['rgb'].shape == (48, 80, 3)
    assert first['depth'].shape == (48, 80)
    visible = sorted(set(np.unique(first['instance'])) - {0})
    assert len(visible) >= 2
    assert str(scene.sticker_codes[0]) not in scene.prompt
    assert str(scene.sticker_codes[1]) not in scene.prompt
    assert not np.array_equal(first['rgb'][first['instance'] == visible[0]].mean(0),
                              first['rgb'][first['instance'] == visible[1]].mean(0))


def test_projected_correspondence_obeys_visibility():
    from worldttt.sap_ttt.scene import ProceduralScene

    scene = ProceduralScene(seed=81, height=48, width=80)
    a, b = scene.render(0), scene.render(8)
    pairs = scene.correspondences(a, b)
    assert pairs.shape[1] == 4
    assert len(pairs) > 20
    assert np.all(a['instance'][pairs[:, 0], pairs[:, 1]] ==
                  b['instance'][pairs[:, 2], pairs[:, 3]])
    assert np.all(a['instance'][pairs[:, 0], pairs[:, 1]] > 0)
    first_world = a['world'][pairs[:, 0], pairs[:, 1]]
    second_world = b['world'][pairs[:, 2], pairs[:, 3]]
    assert np.linalg.norm(first_world - second_world, axis=-1).max() < .25


def test_scene_seeds_change_layout_and_return_view_without_losing_revisit():
    from worldttt.sap_ttt.scene import ProceduralScene

    first = ProceduralScene(seed=10012, height=22, width=40)
    second = ProceduralScene(seed=10013, height=22, width=40)
    assert not np.array_equal(first.render(0)['instance'], second.render(0)['instance'])
    assert not np.array_equal(first.camera(96), second.camera(96))
    assert not np.array_equal(first.render(0)['rgb'], first.render(96)['rgb'])
    for scene in (first, second):
        history = [scene.render(i * 8) for i in range(4)]
        query = [scene.render(i * 8) for i in range(10, 13)]
        old_ids = np.stack([frame['instance'] for frame in history])
        new_ids = np.stack([frame['instance'] for frame in query])
        assert len(set(np.unique(old_ids)) & set(np.unique(new_ids)) - {0}) >= 2


def test_test_scenes_keep_correspondence_but_defeat_pixel_only_retrieval():
    from worldttt.sap_ttt.feature_probe import position_baseline, select_batch
    from worldttt.sap_ttt.pairs import match_historical_tokens
    from worldttt.sap_ttt.scene import ProceduralScene

    scores = []
    for seed in range(10012, 10016):
        frames = [ProceduralScene(seed, 22, 40).render(i * 8) for i in range(13)]
        ids = np.stack([frame['instance'] for frame in frames])
        world = np.stack([frame['world'] for frame in frames])
        positive, valid = match_historical_tokens(ids[:4], world[:4],
                                                  ids[10:13], world[10:13])
        assert int(valid.sum()) > 500
        record = {'source': 'teacher_forced_ground_truth', 'scene_id': str(seed),
                  'supports': [{'visual': torch.empty(1, n * 880, 1)} for n in (4, 3, 3)],
                  'queries': [{'visual': torch.empty(1, 3 * 880, 1)}],
                  'supervision': {'positive': torch.from_numpy(positive),
                                  'valid': torch.from_numpy(valid),
                                  'support_instance': torch.from_numpy(ids[:10]),
                                  'query_instance': torch.from_numpy(ids[10:13])}}
        scores.append(position_baseline(select_batch(record, seed=3407)))
    assert np.mean([score['instance_top1'] for score in scores]) < .75
    assert np.mean([score['exact_top1'] for score in scores]) < .35
