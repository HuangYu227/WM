import json

import numpy as np


def test_paired_scenes_share_layout_but_swap_hidden_stickers():
    from worldttt.sap_binding.data import BindingScene

    first, second = BindingScene(3, 100), BindingScene(3, 101)
    assert first.centers == second.centers and first.prompt == second.prompt
    assert first.sticker_codes != second.sticker_codes
    assert np.array_equal(first.camera(50), second.camera(50))
    a, b = first.render(0), second.render(0)
    assert np.array_equal(a['instance'], b['instance']) and not np.array_equal(a['rgb'], b['rgb'])
    # The revisit preserves silhouettes but removes the episode's sticker code.
    revisit = first.render(90)
    assert np.array_equal(revisit['instance'], first.base.render(90)['instance'])
    assert not np.array_equal(revisit['rgb'], first.base.render(90)['rgb'])


def test_export_split_keeps_paired_layouts_together(tmp_path):
    from worldttt.sap_binding import data

    # Keep the test light while exercising the split invariant directly.
    original = data.export_episode
    data.export_episode = lambda seed, variant, root, **kw: dict(
        id=f'{seed}_{variant}', layout_id=f'layout_{seed}', variant=variant,
        layout_seed=seed, appearance_seed=seed * 2 + variant,
        image='i', camera='c', intrinsics='k', prompt='p', num_frames=97,
        seed=seed, condition='x')
    try:
        rows = data.export_dataset(tmp_path)
    finally:
        data.export_episode = original
    assert len(rows) == 64
    by_layout = {}
    for row in rows:
        by_layout.setdefault(row['layout_id'], set()).add(row['split'])
    assert all(len(splits) == 1 for splits in by_layout.values())
    assert {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')} == {
        'train': 48, 'val': 8, 'test': 8}
    assert json.loads((tmp_path / 'binding_split.json').read_text())['protocol'] == 'sap_binding_paired_v1'


def test_short_revisit_pair_is_chosen_from_geometry_before_generation(tmp_path):
    from worldttt.sap_binding.data import export_episode

    case = export_episode(20000, 0, tmp_path, height=256, width=256)
    pair = case['revisit_pairs'][0]
    assert pair['frame_a'] in (8, 16, 24)
    assert pair['frame_b'] in (80, 88, 96)
    assert pair['preselected_latent_correspondences'] > 0
