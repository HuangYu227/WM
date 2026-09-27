import json

import pytest
import torch


FIVE = (3, 7, 11, 15, 19)


def _inputs(root, *, failed=None):
    from worldttt.sap_binding.config import BindingConfig
    from worldttt.sap_binding.model import BindingBlock
    from dataclasses import asdict

    checkpoints, causal, structure = {}, {}, {}
    for layer in FIVE:
        layer_root = root / f'layer_{layer}'
        layer_root.mkdir()
        settings = {split + '_features': [str(root / f'layer_{layer}' /
                    f'binding_{scene}' / 'features.pt') for scene in scenes]
                    for split, scenes in {'train': ['a0', 'a1'],
                                          'val': ['b0'], 'test': ['c0']}.items()}
        cfg = BindingConfig(layers=(layer,), address_dim=8, value_dim=8,
            ray_dim=4, latent_dim=4, heads=2, topk=2, capacity=8,
            protected_anchors=2, support_tokens=4, address_depth=1,
            fast_hidden_dim=6)
        config = asdict(cfg)
        block = BindingBlock(8, 4, cfg)
        block.gate.data.fill_(float(layer))
        block.value.whitening_fitted.fill_(True)
        checkpoint = layer_root / 'best.pt'
        torch.save(dict(version=1, kind='sap_binding_adapter',
            base_checkpoint='base', config=config,
            modules={layer: block.state_dict()},
            extra=dict(stage='binding_feature_joint', flow_trained=False,
                       settings=settings, step=5)), checkpoint)
        causal_report = layer_root / 'causal.json'
        causal_report.write_text(json.dumps(dict(kind='sap_binding_causal_gate',
            architecture='hybrid', split='val', adapter=str(checkpoint),
            gates={'passed': layer != failed})))
        structure_report = layer_root / 'structure.json'
        structure_report.write_text(json.dumps(dict(flow_allowed=layer != failed,
            adapters={'hybrid': str(checkpoint)})))
        checkpoints[layer] = checkpoint
        causal[layer] = causal_report
        structure[layer] = structure_report
    for scene in ('a0', 'a1', 'b0', 'c0'):
        fixture = root / f'binding_{scene}' / 'fixture.pt'
        fixture.parent.mkdir()
        torch.save({'scene_id': scene}, fixture)
    return checkpoints, causal, structure


def test_merge_five_binding_layers_and_flow_gate(tmp_path):
    from worldttt.sap_binding.merge_five import merge_five
    from worldttt.sap_binding.train import _require_mechanism_gate

    inputs = _inputs(tmp_path)
    base = dict(base_checkpoint='base', seed=3, data='data', manifest='manifest',
                sana_config='sana.yaml')
    merged = merge_five(*inputs, base, tmp_path / 'assembled')
    payload = torch.load(merged, map_location='cpu', weights_only=True)
    assert tuple(payload['config']['layers']) == FIVE
    assert set(payload['modules']) == set(FIVE)
    assert [float(payload['modules'][layer]['gate']) for layer in FIVE] == list(FIVE)
    assert set(payload['extra']['layer_sources']) == {str(layer) for layer in FIVE}
    flow = json.loads((tmp_path / 'assembled/flow-pilot.json').read_text())
    assert flow['sap_binding']['layers'] == list(FIVE)
    assert set(flow['binding_val_features']) == {str(layer) for layer in FIVE}
    _require_mechanism_gate(flow, merged)
    from torch import nn
    from worldttt.sap_binding.config import BindingConfig
    from worldttt.sap_binding.runtime import BindingController
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Module() for _ in range(20)])
    for block in model.blocks:
        block.norm2 = nn.LayerNorm(8)
    controller = BindingController(model, BindingConfig(**payload['config']))
    controller.base_checkpoint = 'base'
    controller.load_checkpoint(merged)
    assert [float(controller.modules[layer].gate.detach()) for layer in FIVE] == list(FIVE)


def test_merge_refuses_a_failed_layer(tmp_path):
    from worldttt.sap_binding.merge_five import merge_five

    inputs = _inputs(tmp_path, failed=19)
    with pytest.raises(ValueError, match='19'):
        merge_five(*inputs, {'base_checkpoint': 'base'}, tmp_path / 'assembled')
    assert not (tmp_path / 'assembled/merged.pt').exists()


def test_flow_gate_rejects_incomplete_five_layer_report(tmp_path):
    from worldttt.sap_binding.merge_five import merge_five
    from worldttt.sap_binding.train import _require_mechanism_gate

    inputs = _inputs(tmp_path)
    merged = merge_five(*inputs, {'base_checkpoint': 'base'}, tmp_path / 'assembled')
    flow = json.loads((tmp_path / 'assembled/flow-pilot.json').read_text())
    causal_path = tmp_path / 'assembled/causal-gate.json'
    causal = json.loads(causal_path.read_text())
    causal['layer_sources'].pop('19')
    causal_path.write_text(json.dumps(causal))
    with pytest.raises(ValueError, match='five-layer|layer'):
        _require_mechanism_gate(flow, merged)


def test_flow_validation_checks_every_layer(monkeypatch):
    from worldttt.sap_binding import train

    monkeypatch.setattr(train, 'validate_mechanism',
        lambda module, records, config, settings, device:
            {'passed': module != 'bad'})
    result = train._validate_all_mechanisms(
        {3: 'good', 7: 'good', 11: 'bad'},
        {3: [], 7: [], 11: []}, None, {}, 'cpu')
    assert result['passed'] is False
    assert result['per_layer']['11']['passed'] is False
