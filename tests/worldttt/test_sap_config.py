def test_sap_settings_preserve_server_paths_and_add_single_layer():
    from worldttt.sap_ttt.config import sap_settings

    base = {'sana_config': '/server/config.yaml', 'base_checkpoint': 'ckpt',
            'cache_gate': '/server/gate.json', 'ttt': {'mode': 'kv_ttt'}}
    settings = sap_settings(base, procedural_fixtures=['/server/scene/fixture.pt'])
    assert settings['sana_config'] == base['sana_config']
    assert settings['ttt'] == base['ttt']
    assert settings['sap']['layers'] == [7]
    assert settings['sap_train']['probe_pretrain_steps'] > 0
    assert settings['sap_train']['procedural_fixtures'] == ['/server/scene/fixture.pt']


def test_multimodal_settings_preserve_paths_and_select_complete_architecture():
    from worldttt.sap_ttt.config import sap_settings
    settings = sap_settings({'sana_config': '/server/config.yaml'}, multimodal=True)
    assert settings['sana_config'] == '/server/config.yaml'
    assert settings['sap']['address_arch'] == 'multimodal'
    assert settings['sap']['memory_arch'] == 'swiglu'
    assert settings['sap']['dim'] == 512
    assert settings['sap_train']['lambda_exact'] > 0
