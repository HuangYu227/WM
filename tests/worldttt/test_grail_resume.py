"""Strict, atomic GRAIL rollout snapshots on CPU."""

import pytest
import torch

from worldttt.associative_ttt import AssociativeTTTConfig, AssociativeTTTLedger, ObservationBatch
from worldttt.grail_resume import load_grail_rollout, save_grail_rollout, validate_grail_protocol


def _fixture():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=30, capacity=2, topk=1)
    protocol = {
        "base_checkpoint_hash": "checkpoint-a",
        "source_tree_hash": "source-a",
        "data_manifest_hash": "data-a",
        "config_fingerprint": cfg.fingerprint(),
        "coordinate_convention": cfg.coordinate_convention,
        "cfg_policy": "shared_episode_ledger_conditional_observation",
        "layer_ids": [2, 6, 10, 14, 3, 7, 11, 15],
        "shape": [1, 1, 1, 1, 1],
        "steps": 2,
        "cfg_scale": 1.0,
        "flow_shift": 1.0,
        "boundaries": [0, 1],
    }
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("resume", metadata=protocol)
    obs = ObservationBatch(
        torch.tensor([[[1., 0.]]]), torch.tensor([[[2.]]]),
        torch.zeros(1, 1, 30), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool),
    )
    state, _ = ledger.commit(state, obs, 0)
    return cfg, protocol, state


def test_rollout_round_trip_and_continuation_state(tmp_path):
    cfg, protocol, state = _fixture()
    path = tmp_path / "grail.pt"
    latents = torch.tensor([[[[[3.]]]]])
    cache = [[torch.tensor([4.])]]
    rng = torch.Generator().manual_seed(9).get_state()
    save_grail_rollout(path, state=state, kv_cache=cache, latents=latents,
                       init_latents=latents.clone(), chunk_cursor=1,
                       rng_state=rng, protocol=protocol)
    restored = load_grail_rollout(path, config=cfg, expected_protocol=protocol, expected_next_chunk=1)
    assert restored["state"].fingerprint() == state.fingerprint()
    torch.testing.assert_close(restored["latents"], latents)
    torch.testing.assert_close(restored["kv_cache"][0][0], cache[0][0])
    torch.testing.assert_close(restored["rng_state"], rng)
    assert restored["chunk_cursor"] == 1
    assert not path.with_suffix(".pt.tmp").exists()


@pytest.mark.parametrize("field", ["base_checkpoint_hash", "source_tree_hash", "data_manifest_hash",
                                   "cfg_policy", "coordinate_convention", "config_fingerprint"])
def test_mismatched_protocol_rejected_without_changing_snapshot(tmp_path, field):
    cfg, protocol, state = _fixture()
    path = tmp_path / "grail.pt"
    latents = torch.ones(1, 1, 1, 1, 1)
    save_grail_rollout(path, state=state, kv_cache=[[]], latents=latents,
                       init_latents=latents, chunk_cursor=1, rng_state=None, protocol=protocol)
    before = path.read_bytes()
    wrong = dict(protocol)
    wrong[field] = "wrong"
    with pytest.raises(ValueError):
        load_grail_rollout(path, config=cfg, expected_protocol=wrong, expected_next_chunk=1)
    assert path.read_bytes() == before


def test_wrong_cursor_and_incomplete_metadata_rejected(tmp_path):
    cfg, protocol, state = _fixture()
    path = tmp_path / "grail.pt"
    latents = torch.ones(1, 1, 1, 1, 1)
    save_grail_rollout(path, state=state, kv_cache=[[]], latents=latents,
                       init_latents=latents, chunk_cursor=1, rng_state=None, protocol=protocol)
    with pytest.raises(ValueError):
        load_grail_rollout(path, config=cfg, expected_protocol=protocol, expected_next_chunk=2)
    with pytest.raises(ValueError):
        save_grail_rollout(tmp_path / "bad.pt", state=state, kv_cache=[[]], latents=latents,
                           init_latents=latents, chunk_cursor=1, rng_state=None,
                           protocol={**protocol, "base_checkpoint_hash": ""})


def test_protocol_can_be_validated_before_rollout_starts():
    _, protocol, _ = _fixture()
    validate_grail_protocol(protocol)
    with pytest.raises(ValueError, match="base_checkpoint_hash"):
        validate_grail_protocol({**protocol, "base_checkpoint_hash": ""})


def test_corrupt_cache_chunk_count_rejected_before_restore(tmp_path):
    cfg, protocol, state = _fixture()
    path = tmp_path / "grail.pt"
    latents = torch.ones(1, 1, 1, 1, 1)
    save_grail_rollout(path, state=state, kv_cache=[[]], latents=latents,
                       init_latents=latents, chunk_cursor=1, rng_state=None, protocol=protocol)
    payload = torch.load(path, weights_only=True)
    payload["kv_cache"] = []
    torch.save(payload, path)
    with pytest.raises(ValueError, match="cache"):
        load_grail_rollout(path, config=cfg, expected_protocol=protocol, expected_next_chunk=1)


def test_native_snapshot_requires_matching_embedded_adapter(tmp_path):
    cfg, protocol, state = _fixture()
    protocol['adapter_hash'] = 'required-canonical-adapter'
    with pytest.raises(ValueError, match='adapter'):
        save_grail_rollout(tmp_path / 'missing.pt', state=state, kv_cache=[[]],
            latents=torch.ones(1, 1, 1, 1, 1), init_latents=torch.ones(1, 1, 1, 1, 1),
            chunk_cursor=1, rng_state=None, protocol=protocol)
