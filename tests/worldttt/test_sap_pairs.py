import numpy as np


def test_historical_pairs_use_instance_and_world_position_without_query_leakage():
    from worldttt.sap_ttt.scene import ProceduralScene
    from worldttt.sap_ttt.pairs import match_historical_tokens

    scene = ProceduralScene(9, 22, 40)
    history = [scene.render(i * 8) for i in range(4)]
    query = [scene.render(i * 8) for i in range(10, 13)]
    positive, valid = match_historical_tokens(
        np.stack([x['instance'] for x in history]),
        np.stack([x['world'] for x in history]),
        np.stack([x['instance'] for x in query]),
        np.stack([x['world'] for x in query]))
    assert positive.shape == valid.shape == (3 * 22 * 40,)
    assert valid.sum() > 10
    old_id = np.stack([x['instance'] for x in history]).reshape(-1)
    new_id = np.stack([x['instance'] for x in query]).reshape(-1)
    assert np.array_equal(old_id[positive[valid]], new_id[valid])
