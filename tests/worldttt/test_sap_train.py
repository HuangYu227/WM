import numpy as np
import pytest


def test_supervision_loader_keeps_labels_separate_from_model_fixture(tmp_path):
    from worldttt.sap_ttt.scene import ProceduralScene
    from worldttt.sap_ttt.train import load_supervision

    scene = ProceduralScene(4, 22, 40)
    rendered = [scene.render(i * 8) for i in range(13)]
    label_path = tmp_path / 'supervision.npz'
    np.savez(label_path, instance=np.stack([x['instance'] for x in rendered]),
             world=np.stack([x['world'] for x in rendered]))
    labels = load_supervision(label_path)
    assert set(labels) == {'positive', 'valid', 'support_instance', 'query_instance'}
    assert labels['valid'].any()
    with pytest.raises(ValueError, match='13'):
        np.savez(label_path, instance=np.zeros((12, 22, 40), dtype=np.uint8),
                 world=np.zeros((12, 22, 40, 3), dtype=np.float32))
        load_supervision(label_path)
