import torch
import pytest


def test_sap_memory_commit_and_read_are_separate():
    from worldttt.sap_ttt.memory import SapMemory, SapMemoryState

    memory = SapMemory(dim=4, lr=0.5)
    state = SapMemoryState.new(memory, episode='one')
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    before = memory.read(state, query)
    assert state.updates == 0
    state.commit(memory, 0, query, value)
    after = memory.read(state, query)
    assert state.updates == 1
    assert (after - value).square().mean() < (before - value).square().mean()
    assert state.updates == 1


def test_sap_memory_rejects_duplicate_and_nonfinite_without_changing_state(tmp_path):
    from worldttt.sap_ttt.memory import SapMemory, SapMemoryState

    memory = SapMemory(dim=4, lr=0.5)
    state = SapMemoryState.new(memory, episode='scene', batch=2)
    key = torch.eye(4)[:2].unsqueeze(1)
    value = key.roll(1, dims=-1)
    assert state.commit(memory, 0, key, value)['committed']
    saved = state.weight.clone()
    with pytest.raises(ValueError, match='Expected chunk 1'):
        state.commit(memory, 0, key, value)
    bad = value.clone()
    bad[0, 0, 0] = float('nan')
    assert not state.commit(memory, 1, key, bad)['committed']
    assert torch.equal(state.weight, saved)
    assert state.last_chunk == 0
    path = tmp_path / 'state.pt'
    state.save(path)
    restored = SapMemoryState.load(path)
    assert restored.episode == 'scene'
    assert torch.equal(restored.weight, saved)


def test_sap_address_clean_and_noisy_paths_share_normalized_space():
    from worldttt.sap_ttt.address import SapAddress

    address = SapAddress(vision_dim=8, text_dim=8, ray_dim=6, dim=4)
    visual = torch.randn(2, 3, 8)
    text = torch.randn(2, 5, 8)
    rays = torch.randn(2, 3, 6)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    key = address.write(visual, text, mask, rays, mode='selective_geometry')
    query = address.query(visual, text, mask, rays, torch.full((2, 3, 1), .7),
                          mode='selective_geometry')
    assert key.shape == query.shape == (2, 3, 4)
    torch.testing.assert_close(key.norm(dim=-1), torch.ones(2, 3))
    torch.testing.assert_close(query.norm(dim=-1), torch.ones(2, 3))
    with pytest.raises(ValueError, match='ray'):
        address.write(visual, text, mask, rays[:, :2], mode='selective_geometry')


def test_one_chunk_step_actually_writes_four_independent_key_value_pairs():
    from worldttt.sap_ttt.memory import SapMemory, SapMemoryState

    memory = SapMemory(dim=4, lr=.5)
    state = SapMemoryState.new(memory, 'orthogonal')
    keys = torch.eye(4)[None]
    values = keys.roll(1, dims=-1)
    before = (memory.read(state, keys) - values).square().mean()
    state.commit(memory, 0, keys, values)
    after = (memory.read(state, keys) - values).square().mean()
    assert after < before


def test_write_log_does_not_mislabel_summed_objective_as_mse():
    from worldttt.sap_ttt.memory import SapMemory, SapMemoryState

    memory = SapMemory(dim=4)
    state = SapMemoryState.new(memory, 'log')
    row = state.commit(memory, 0, torch.eye(4)[None], torch.ones(1, 4, 4))
    assert 'write_objective' in row
    assert 'write_mse' not in row


def test_cfg_branch_updates_match_separate_single_branch_updates():
    from worldttt.sap_ttt.memory import SapMemory, SapMemoryState

    memory = SapMemory(dim=4, lr=.5)
    keys = torch.eye(4)[:2, None]
    values = keys.roll(1, dims=-1)
    joint = SapMemoryState.new(memory, 'joint', batch=2)
    joint.commit(memory, 0, keys, values)
    for branch in range(2):
        separate = SapMemoryState.new(memory, 'separate', batch=1)
        separate.commit(memory, 0, keys[branch:branch + 1], values[branch:branch + 1])
        torch.testing.assert_close(joint.weight[branch], separate.weight[0])
