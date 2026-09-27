"""Five-layer SAP configuration and causal outer-loop checks."""

import pytest
import torch
from torch import nn


def test_five_layer_settings_keep_single_layer_default():
    from worldttt.sap_ttt.config import sap_settings

    assert sap_settings({})['sap']['layers'] == [7]
    settings = sap_settings({}, multimodal=True, layers=(3, 7, 11, 15, 19))
    assert settings['sap']['layers'] == [3, 7, 11, 15, 19]


def test_five_layer_episode_supervises_every_layer():
    from worldttt.sap_ttt.episode import SapEpisodeModel
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Module() for _ in range(5)])
            for block in self.blocks:
                block.norm2 = nn.LayerNorm(12)

        def forward(self, z, times, *, sap_context, y, mask, chunk_plucker,
                    kv_cache, save_kv_cache, **kwargs):
            x = z.flatten(2).transpose(1, 2)
            for block in self.blocks:
                x = sap_context.apply(block, x, x, y, mask, chunk_plucker, z.shape[2])
            return x.transpose(1, 2).reshape_as(z), kv_cache

    torch.manual_seed(91)
    backbone = Backbone()
    ctl = SapController(backbone, SapConfig(layers=(1, 2, 3, 4, 5), dim=16,
        ray_dim=6, heads=2, address_arch='multimodal', memory_arch='swiglu',
        memory_hidden_dim=12, support_tokens=8, normalized_value=True))
    episode = SapEpisodeModel(backbone, ctl, lambda_exact=.1,
                              cache_factory=lambda _: ([None] * 4, lambda i: None))
    z = torch.randn(1, 12, 13, 1, 2)
    cam = torch.zeros(1, 13, 20)
    rays = torch.randn(1, 6, 13, 1, 2)
    text = torch.randn(1, 4, 12)
    mask = torch.ones(1, 4, dtype=torch.bool)
    labels = dict(positive=torch.arange(6), valid=torch.ones(6, dtype=torch.bool),
                  support_instance=torch.ones(10, 1, 2, dtype=torch.long),
                  query_instance=torch.ones(3, 1, 2, dtype=torch.long))
    loss, metrics = episode(z, cam, rays, text, mask, 'scene', seed=9,
                            supervision=labels)
    loss.backward()
    assert set(metrics['per_layer']) == {'1', '2', '3', '4', '5'}
    assert all(ctl.state[i].updates == 3 for i in ctl.modules)
    assert all(ctl.modules[i].gate.grad is not None for i in ctl.modules)
    assert all(ctl.modules[i].memory.initial_weight.grad is not None for i in ctl.modules)


def test_multilayer_writes_use_the_same_token_positions():
    from worldttt.sap_ttt.runtime import SapConfig, SapController

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Module() for _ in range(5)])
            for block in self.blocks:
                block.norm2 = nn.LayerNorm(12)

    model = Backbone()
    controller = SapController(model, SapConfig(layers=(1, 2, 3, 4, 5), dim=16,
        ray_dim=6, support_tokens=4))
    controller.reset_episode('scene', 1)
    context = controller.context(torch.zeros(1), collect=True, chunk=0)
    x = torch.randn(1, 8, 12)
    y = torch.randn(1, 3, 12)
    mask = torch.ones(1, 3, dtype=torch.bool)
    rays = torch.randn(1, 8, 6)
    for block in model.blocks:
        context.apply(block, x, x, y, mask, rays, frames=4)
    first = context.support_indices[1]
    assert first.numel() == 4
    assert all(torch.equal(first, context.support_indices[i]) for i in range(2, 6))


def test_merge_rejects_missing_layer(tmp_path):
    from worldttt.sap_ttt.merge_layers import merge_layers

    with pytest.raises(ValueError, match='five'):
        merge_layers({}, tmp_path / 'merged.pt')


def test_merge_preserves_independent_layer_weights(tmp_path):
    from dataclasses import asdict
    from worldttt.sap_ttt.merge_layers import merge_layers
    from worldttt.sap_ttt.runtime import SapBlockMemory, SapConfig

    paths = {}
    for layer in (3, 7, 11, 15, 19):
        cfg = SapConfig(layers=(layer,), dim=16, ray_dim=6, heads=2,
                        address_arch='multimodal', memory_arch='swiglu',
                        memory_hidden_dim=12, normalized_value=True)
        module = SapBlockMemory(12, cfg)
        with torch.no_grad():
            module.gate.fill_(layer / 100)
        path = tmp_path / f'{layer}.pt'
        torch.save({'version': 1, 'config': asdict(cfg), 'base_checkpoint': 'base',
                    'modules': {layer: module.state_dict()},
                    'extra': {'stage': 'feature_joint', 'flow_trained': False,
                              'step': 100}}, path)
        paths[layer] = path
    output = tmp_path / 'five.pt'
    merge_layers(paths, output)
    merged = torch.load(output, map_location='cpu', weights_only=True)
    assert tuple(merged['config']['layers']) == (3, 7, 11, 15, 19)
    assert set(merged['modules']) == set(paths)
    assert [float(merged['modules'][i]['gate']) for i in sorted(paths)] == pytest.approx(
        [i / 100 for i in sorted(paths)])
    assert merged['extra']['flow_trained'] is False
    from worldttt.sap_ttt.runtime import SapController
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Module() for _ in range(19)])
    for block in model.blocks:
        block.norm2 = nn.LayerNorm(12)
    controller = SapController(model, SapConfig(**merged['config']))
    controller.base_checkpoint = 'base'
    controller.load_checkpoint(output)
    assert [float(controller.modules[i].gate.detach()) for i in sorted(paths)] == pytest.approx(
        [i / 100 for i in sorted(paths)])
    with pytest.raises(FileExistsError):
        merge_layers(paths, output)


def test_prepare_accepts_a_specific_gdn_layer(tmp_path):
    import json
    from worldttt.sap_ttt.prepare_multimodal import prepare

    for i in range(4):
        scene = tmp_path / 'data' / f'scene_{i:06d}'
        scene.mkdir(parents=True)
        torch.save({'scene_id': scene.name, 'base_checkpoint': 'base', 'layer': 19},
                   scene / 'features.pt')
        torch.save({}, scene / 'fixture.pt')
    prepare({'base_checkpoint': 'base'}, tmp_path / 'data', tmp_path / 'out',
            train_count=2, val_count=1, layer=19)
    settings = json.loads((tmp_path / 'out' / 'joint-native.json').read_text())
    assert settings['sap']['layers'] == [19]


def test_one_extraction_writes_separate_layer_records(tmp_path):
    from worldttt.sap_ttt.features import save_feature_records

    layers = (3, 7, 11, 15, 19)
    part = lambda: {layer: {'visual': torch.full((1, 2, 4), layer)} for layer in layers}
    paths = save_feature_records(tmp_path / 'features', layers=layers,
        scene_id='scene_010000', base_checkpoint='base', supports=[part()] * 3,
        queries=[part()] * 3, queries_no_history=[part()] * 3,
        supervision={'positive': torch.tensor([0, 1])}, noise_sigmas=[1., .5, .1])
    assert set(paths) == set(layers)
    for layer, path in paths.items():
        assert path == tmp_path / 'features' / f'layer_{layer}' / 'scene_010000' / 'features.pt'
        record = torch.load(path, map_location='cpu', weights_only=True)
        assert record['layer'] == layer
        assert record['supports'][0]['visual'][0, 0, 0] == layer
