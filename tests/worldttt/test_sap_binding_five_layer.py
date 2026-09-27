from pathlib import Path

import pytest
import torch


FIVE = (3, 7, 11, 15, 19)


def test_five_layer_settings_attach_independent_modules_and_states():
    from torch import nn
    from worldttt.sap_binding.config import binding_settings, BindingConfig
    from worldttt.sap_binding.runtime import BindingController

    settings = binding_settings({}, layers=FIVE)
    assert settings['sap_binding']['layers'] == list(FIVE)
    assert binding_settings({})['sap_binding']['layers'] == list(FIVE)
    assert binding_settings({}, layers=(7,))['sap_binding']['layers'] == [7]
    settings['sap_binding'].update(address_dim=8, value_dim=8, ray_dim=4,
        latent_dim=4, heads=2, topk=2, capacity=8, protected_anchors=2,
        support_tokens=4, address_depth=1, fast_hidden_dim=6)
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Module() for _ in range(20)])
    for block in model.blocks:
        block.norm2 = nn.LayerNorm(8)
    controller = BindingController(model, BindingConfig(**settings['sap_binding']))
    controller.reset_episode('scene', 1)
    assert set(controller.modules) == set(FIVE)
    assert set(controller.state) == set(FIVE)
    assert len({id(state.bank) for state in controller.state.values()}) == 5


def test_binding_feature_finalize_routes_all_five_layers(tmp_path):
    from worldttt.sap_binding.features import finalize_feature_records

    fixture = {'layout_id': 'layout_01', 'variant': 0,
               'latent': torch.randn(1, 4, 13, 2, 2)}
    source = {}
    for layer in FIVE:
        record = dict(source='teacher_forced_ground_truth', scene_id='binding_01_0',
            layer=layer, supports=[{} for _ in range(3)])
        path = tmp_path / f'legacy-{layer}.pt'
        torch.save(record, path)
        source[layer] = path
    paths = finalize_feature_records(source, fixture, tmp_path / 'features',
                                     layers=FIVE)
    assert set(paths) == set(FIVE)
    for layer, path in paths.items():
        assert path == tmp_path / 'features' / f'layer_{layer}' / 'binding_01_0' / 'features.pt'
        saved = torch.load(path, weights_only=True)
        assert saved['layer'] == layer
        assert saved['source'] == 'sap_binding_teacher_forced_ground_truth'
        assert saved['supports'][0]['latent'].shape[2] == 4
        assert saved['query_obscured'] is True


def test_prepare_rejects_mixed_layer_records(tmp_path):
    from worldttt.sap_binding.prepare import prepare

    root = tmp_path / 'layer_3'
    for layout in range(32):
        for variant in (0, 1):
            folder = root / f'binding_{layout:06d}_{variant}'
            folder.mkdir(parents=True)
            torch.save(dict(source='sap_binding_teacher_forced_ground_truth',
                query_obscured=True, layer=7 if layout == 31 else 3,
                layout_id=f'layout_{layout:06d}', variant=variant), folder / 'features.pt')
    with pytest.raises(ValueError, match='layer'):
        prepare({}, root, tmp_path / 'configs', tmp_path / 'values', layer=3)


def test_prepare_five_layer_layout_and_shared_fixture_paths(tmp_path):
    import json
    from worldttt.sap_binding.prepare import prepare

    root = tmp_path / 'features'
    for layer in FIVE:
        for layout in range(32):
            for variant in (0, 1):
                scene = f'binding_{layout:06d}_{variant}'
                folder = root / f'layer_{layer}' / scene
                folder.mkdir(parents=True)
                torch.save(dict(source='sap_binding_teacher_forced_ground_truth',
                    query_obscured=True, layer=layer,
                    layout_id=f'layout_{layout:06d}', variant=variant,
                    base_checkpoint='base'), folder / 'features.pt')
                if layer == FIVE[0]:
                    fixture = root / scene / 'fixture.pt'
                    fixture.parent.mkdir()
                    torch.save({'scene_id': scene}, fixture)
        out = tmp_path / f'config-{layer}'
        prepare({'base_checkpoint': 'base'}, root / f'layer_{layer}',
                out, tmp_path / f'value-{layer}', layer=layer)
        flow = json.loads((out / 'flow-pilot.json').read_text())
        assert flow['sap_binding']['layers'] == [layer]
        assert len(flow['sap_binding_train']['procedural_fixtures']) == 48
        assert all('/layer_' not in path.replace('\\', '/') for path in
                   flow['sap_binding_train']['procedural_fixtures'])


def test_bfloat16_binding_raw_capture_preserves_large_finite_values():
    from torch import nn
    from worldttt.sap_binding.config import BindingConfig
    from worldttt.sap_binding.runtime import BindingController

    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].norm2 = nn.LayerNorm(8)
    cfg = BindingConfig(mode='frozen', layers=(1,), address_dim=8, value_dim=8,
        ray_dim=4, latent_dim=4, heads=2, topk=2, capacity=8,
        protected_anchors=2, support_tokens=4, address_depth=1, fast_hidden_dim=6)
    ctl = BindingController(model, cfg)
    ctl.reset_episode('scene', 1)
    x = torch.full((1, 8, 8), 81920., dtype=torch.bfloat16)
    ctx = ctl.context(torch.tensor([500.]), latent=torch.ones(1, 4, 2, 2, 2), record_raw=True)
    ctx.apply(model.blocks[0], x, x, torch.ones(1, 2, 8),
              torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 8, 4), 2)
    assert ctx.raw_features[1]['visual'].dtype == torch.bfloat16
    assert torch.isfinite(ctx.raw_features[1]['visual']).all()


def test_five_layer_commit_is_transactional_and_state_roundtrips(tmp_path):
    from torch import nn
    from worldttt.sap_binding.config import BindingConfig
    from worldttt.sap_binding.runtime import BindingController

    def make_controller():
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Module() for _ in range(20)])
        for block in model.blocks:
            block.norm2 = nn.LayerNorm(8)
        cfg = BindingConfig(layers=FIVE, address_dim=8, value_dim=8,
            ray_dim=4, latent_dim=4, heads=2, topk=2, capacity=8,
            protected_anchors=2, support_tokens=4, address_depth=1,
            fast_hidden_dim=6)
        ctl = BindingController(model, cfg)
        ctl.base_checkpoint = 'base'
        return ctl

    ctl = make_controller()
    ctl.reset_episode('scene', 1)
    x = torch.randn(1, 8, 8)
    text = torch.randn(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    rays = torch.randn(1, 8, 4)
    latent = torch.randn(1, 4, 2, 2, 2)
    clean = ctl.context(torch.zeros(1), latent=latent, collect=True, chunk=0)
    for layer in FIVE:
        clean.apply(ctl.model.blocks[layer - 1], x, x, text, mask, rays, 2)
    clean.features[19] = tuple(torch.full_like(item, float('nan')) if i == 0 else item
                               for i, item in enumerate(clean.features[19]))
    assert not ctl.commit(clean, 0)['committed']
    assert all(state.updates == 0 and not state.bank.valid.any() for state in ctl.state.values())
    clean = ctl.context(torch.zeros(1), latent=latent, collect=True, chunk=0)
    for layer in FIVE:
        clean.apply(ctl.model.blocks[layer - 1], x, x, text, mask, rays, 2)
    assert ctl.commit(clean, 0)['committed']
    assert all(state.updates == 1 for state in ctl.state.values())
    ctl.save_checkpoint(tmp_path / 'adapter.pt')
    ctl.save_state(tmp_path / 'state.pt')
    other = make_controller()
    other.load_checkpoint(tmp_path / 'adapter.pt')
    other.load_state(tmp_path / 'state.pt')
    for layer in FIVE:
        torch.testing.assert_close(other.state[layer].bank.keys, ctl.state[layer].bank.keys)
