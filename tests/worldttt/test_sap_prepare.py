import json

import pytest
import torch


def test_prepare_splits_and_config_variants(tmp_path):
    from worldttt.sap_ttt.prepare_multimodal import prepare
    for i in range(4):
        scene = tmp_path / 'data' / f'scene_{i:06d}'
        scene.mkdir(parents=True)
        torch.save({'scene_id': scene.name, 'base_checkpoint': 'base', 'layer': 7}, scene / 'features.pt')
        torch.save({}, scene / 'fixture.pt')
    base = {'base_checkpoint': 'base', 'sana_config': '/server/sana.yaml', 'seed': 3407}
    prepare(base, tmp_path / 'data', tmp_path / 'configs', train_count=2, val_count=1, batch_size=4)
    native = json.loads((tmp_path / 'configs/joint-native.json').read_text())
    empty = json.loads((tmp_path / 'configs/joint-no-history.json').read_text())
    assert native['train_features'] == empty['train_features']
    assert len(native['test_features']) == 1
    assert native['batch_size'] == 4
    flow = json.loads((tmp_path / 'configs/flow.json').read_text())
    assert len(flow['sap_train']['procedural_fixtures']) == 2
    assert flow['sap'] == native['sap']
    assert flow['sana_config'] == base['sana_config']
    assert (tmp_path / 'configs/joint-linear-memory.json').exists()
    assert (tmp_path / 'configs/joint-linear-address.json').exists()
    with pytest.raises(FileExistsError):
        prepare(base, tmp_path / 'data', tmp_path / 'configs', train_count=2, val_count=1)
