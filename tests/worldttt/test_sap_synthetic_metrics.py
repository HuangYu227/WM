def test_ground_truth_renderer_recovers_hidden_sticker_bits():
    from worldttt.sap_ttt.scene import ProceduralScene
    from worldttt.sap_ttt.synthetic_metrics import sticker_bit_accuracy

    scene = ProceduralScene(31, 128, 224)
    first = scene.render(8)['rgb']
    returned = scene.render(80)['rgb']
    assert sticker_bit_accuracy(scene, first, 8)['mean_accuracy'] == 1.
    assert sticker_bit_accuracy(scene, returned, 80)['mean_accuracy'] == 1.
