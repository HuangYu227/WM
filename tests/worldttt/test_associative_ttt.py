"""CPU checks for the full-covariance GRAIL v2 state transition."""

import numpy as np
import pytest
import torch

from worldttt.associative_ttt import (
    AssociativeTTTConfig,
    AssociativeTTTLedger,
    AssociativeTTTState,
    HoldoutBatch,
    ObservationBatch,
)


def _observation(key, value):
    return ObservationBatch(
        keys=torch.tensor([[key]], dtype=torch.float64),
        values=torch.tensor([[value]], dtype=torch.float64),
        geometry=torch.zeros(1, 1, 1, dtype=torch.float64),
        confidence=torch.ones(1, 1, dtype=torch.float64),
        source_real=torch.ones(1, 1, dtype=torch.bool),
    )


def test_replacement_within_one_chunk_discards_superseded_instance_statistics():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1,
                               merge_threshold=1.5, initial_precision=.7)
    ledger = AssociativeTTTLedger(cfg)
    obs = ObservationBatch(torch.eye(2)[None], torch.tensor([[[9.], [2.]]]), torch.zeros(1, 2, 1),
                            torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool))
    state, report = ledger.commit(ledger.new_state('replacement'), obs, 0)
    torch.testing.assert_close(state.cross[0, 0], torch.tensor([[0., 2.]]))
    torch.testing.assert_close(state.precision[0, 0], torch.tensor([[.7, 0.], [0., 1.7]]))
    assert report['accepted'] == 1


def test_new_slot_matches_full_ridge_numpy_reference():
    cfg = AssociativeTTTConfig(
        key_dim=2, value_dim=2, geometry_dim=1, capacity=2, topk=1,
        decay=1.0, initial_precision=0.7,
    )
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("cpu-ridge", dtype=torch.float64)
    key = [0.6, 0.8]
    value = [2.0, -3.0]
    state, report = ledger.commit(state, _observation(key, value), 0)

    key_np = np.asarray(key)
    value_np = np.asarray(value)
    expected_p = 0.7 * np.eye(2) + np.outer(key_np, key_np)
    expected_c = np.outer(value_np, key_np)
    expected_read = expected_c @ np.linalg.solve(expected_p, key_np)
    assert report["committed"]
    np.testing.assert_allclose(state.precision[0, 0].detach().numpy(), expected_p, atol=1e-10)
    np.testing.assert_allclose(state.cross[0, 0].detach().numpy(), expected_c, atol=1e-10)
    result = ledger.read(state, torch.tensor([[key]], dtype=torch.float64))
    assert bool(result.has_memory[0, 0])
    np.testing.assert_allclose(result.value[0, 0].detach().numpy(), expected_read, atol=1e-4)


def test_empty_ledger_has_a_holdout_baseline_for_first_commit():
    cfg = AssociativeTTTConfig(
        key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1,
        initial_precision=0.01,
    )
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("holdout", dtype=torch.float64)
    obs = _observation([1.0, 0.0], [1.0])
    holdout = HoldoutBatch(
        torch.tensor([[[0.8, 0.6]]], dtype=torch.float64),
        torch.tensor([[[0.8]]], dtype=torch.float64),
        obs.geometry.clone(), obs.confidence.clone(),
    )
    updated, report = ledger.commit(state, obs, 0, holdout=holdout)
    assert report["committed"]
    assert int(updated.last_committed_chunk[0]) == 0
    assert report["holdout_after"] < report["holdout_before"]


def test_first_generated_write_can_be_read_under_default_uncertainty_gate():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("generated")
    obs = _observation([1.0, 0.0], [2.0])
    obs.source_real.zero_()
    state, report = ledger.commit(state, obs, 0)
    assert report["committed"]
    read = ledger.read(state, obs.keys, obs.geometry)
    assert bool(read.has_memory[0, 0])
    assert 0 <= float(read.uncertainty[0, 0]) <= 1


def test_two_chunk_full_covariance_matches_numpy_and_stateful_read():
    cfg = AssociativeTTTConfig(
        key_dim=2, value_dim=2, geometry_dim=1, capacity=1, topk=1,
        decay=0.8, initial_precision=0.7, merge_threshold=0.5,
    )
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("two-chunk", dtype=torch.float64)
    k0, k1 = np.array([1.0, 0.0]), np.array([0.6, 0.8])
    v0, v1 = np.array([2.0, -1.0]), np.array([1.0, 3.0])
    state, _ = ledger.commit(state, _observation(k0.tolist(), v0.tolist()), 0)
    state, _ = ledger.commit(state, _observation(k1.tolist(), v1.tolist()), 1)

    expected_p = 0.8 * (0.7 * np.eye(2) + np.outer(k0, k0)) + np.outer(k1, k1)
    expected_c = 0.8 * np.outer(v0, k0) + np.outer(v1, k1)
    np.testing.assert_allclose(state.precision[0, 0].detach().numpy(), expected_p, atol=1e-10)
    np.testing.assert_allclose(state.cross[0, 0].detach().numpy(), expected_c, atol=1e-10)
    q = torch.tensor([[[0.8, 0.6]]], dtype=torch.float64)
    expected_read = expected_c @ np.linalg.solve(expected_p, np.array([0.8, 0.6]))
    np.testing.assert_allclose(ledger.read(state, q).value[0, 0].detach().numpy(), expected_read, atol=1e-4)


def test_functional_write_has_finite_nonzero_key_and_value_gradients():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1)
    ledger = AssociativeTTTLedger(cfg).double()
    query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)

    def future_loss(key, value):
        state = ledger.new_state("gradient", dtype=torch.float64)
        obs = ObservationBatch(
            key.view(1, 1, 2), value.view(1, 1, 1),
            torch.zeros(1, 1, 1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.bool),
        )
        updated, _ = ledger.commit(state, obs, 0, differentiable=True)
        return ledger.read(updated, query).value.square().sum()

    key = torch.tensor([0.8, 0.6], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(future_loss, (key, value), eps=1e-6, atol=1e-4)
    key_grad, value_grad = torch.autograd.grad(future_loss(key, value), (key, value))
    assert torch.isfinite(key_grad).all() and key_grad.abs().sum() > 0
    assert torch.isfinite(value_grad).all() and value_grad.abs().sum() > 0


def test_failed_holdout_commit_preserves_state_and_chunk_cursor():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("rollback", dtype=torch.float64)
    state, _ = ledger.commit(state, _observation([1.0, 0.0], [1.0]), 0)
    before = state.fingerprint()
    holdout = HoldoutBatch(
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[[1.0]]], dtype=torch.float64),
        torch.zeros(1, 1, 1, dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.float64),
    )
    rejected, report = ledger.commit(state, _observation([1.0, 0.0], [-1.0]), 1, holdout=holdout)
    assert report["rolled_back"] and not report["committed"]
    assert report["holdout_after"] > report["holdout_before"]
    assert rejected.fingerprint() == before
    assert int(rejected.last_committed_chunk[0]) == 0


def test_state_load_rejects_missing_config_identity():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1)
    state = AssociativeTTTLedger(cfg).new_state("identity")
    payload = state.to_payload()
    payload["metadata"].pop("config_fingerprint")
    with pytest.raises(ValueError, match="fingerprint"):
        AssociativeTTTState.from_payload(payload, cfg)


def test_state_load_checks_coordinate_and_requested_source_identity():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1)
    state = AssociativeTTTLedger(cfg).new_state("identity")
    payload = state.to_payload()
    payload["metadata"]["coordinate_convention"] = "wrong_convention"
    with pytest.raises(ValueError, match="coordinate"):
        AssociativeTTTState.from_payload(payload, cfg)

    payload["metadata"]["coordinate_convention"] = cfg.coordinate_convention
    with pytest.raises(ValueError, match="base_checkpoint_hash"):
        AssociativeTTTState.from_payload(
            payload, cfg, expected_metadata={"base_checkpoint_hash": "checkpoint-A"}
        )


def test_save_load_retains_read_and_chunk_state(tmp_path):
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=2, topk=1)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("resume", dtype=torch.float64,
                             metadata={"base_checkpoint_hash": "checkpoint-A", "layer_ids": [3, 4]})
    state, _ = ledger.commit(state, _observation([1.0, 0.0], [2.0]), 0)
    before = state.fingerprint()
    query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    prediction = ledger.read(state, query).value
    path = tmp_path / "ledger.pt"
    state.save(path)

    loaded = AssociativeTTTState.load(
        path, cfg, expected_fingerprint=before,
        expected_metadata={"base_checkpoint_hash": "checkpoint-A", "layer_ids": [3, 4]},
    )
    assert loaded.fingerprint() == before
    torch.testing.assert_close(ledger.read(loaded, query).value, prediction)
    assert int(loaded.last_committed_chunk[0]) == 0


def test_leave_one_out_loss_never_predicts_a_token_from_itself():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1,
                               initial_precision=1.0, solve_jitter=1e-9)
    ledger = AssociativeTTTLedger(cfg)
    keys = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
    values = torch.tensor([[[1.0], [3.0]]], dtype=torch.float64)
    # Each target is predicted from the other support: 3/2 and 1/2.
    loss = ledger.leave_one_out_association_loss(keys, values)
    torch.testing.assert_close(loss, torch.tensor(3.25, dtype=torch.float64), atol=1e-7, rtol=0)


def test_multiple_tokens_in_one_chunk_share_a_full_covariance_update():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1,
                               initial_precision=0.7, merge_threshold=0.5)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("multi-token", dtype=torch.float64)
    keys = torch.tensor([[[1.0, 0.0], [0.6, 0.8]]], dtype=torch.float64)
    values = torch.tensor([[[2.0], [3.0]]], dtype=torch.float64)
    obs = ObservationBatch(keys, values, torch.zeros(1, 2, 1, dtype=torch.float64),
                           torch.ones(1, 2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.bool))
    state, report = ledger.commit(state, obs, 0)
    expected_p = 0.7 * np.eye(2) + np.outer([1.0, 0.0], [1.0, 0.0]) + np.outer([0.6, 0.8], [0.6, 0.8])
    expected_c = np.outer([2.0], [1.0, 0.0]) + np.outer([3.0], [0.6, 0.8])
    assert report["accepted"] == 2
    np.testing.assert_allclose(state.precision[0, 0].detach().numpy(), expected_p, atol=1e-10)
    np.testing.assert_allclose(state.cross[0, 0].detach().numpy(), expected_c, atol=1e-10)


def test_real_observation_can_reclaim_capacity_but_generated_cannot_replace_protected_real():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("capacity", dtype=torch.float64)
    state, _ = ledger.commit(state, _observation([1.0, 0.0], [1.0]), 0)
    generated = _observation([0.0, 1.0], [9.0])
    generated.source_real.fill_(False)
    unchanged, generated_report = ledger.commit(state, generated, 1)
    assert generated_report["accepted"] == 0
    assert unchanged.source_real[0, 0]

    real = _observation([0.0, 1.0], [2.0])
    updated, real_report = ledger.commit(unchanged, real, 2)
    assert real_report["accepted"] == 1
    assert real_report["replaced"] == 1
    assert int(updated.generation[0, 0]) == 1
    torch.testing.assert_close(updated.keys[0, 0], real.keys[0, 0])


def test_capacity_replacement_prefers_older_equal_confidence_slot():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1,
                               capacity=2, topk=1, decay=1.0)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("age", dtype=torch.float64)
    state, _ = ledger.commit(state, _observation([1.0, 0.0], [1.0]), 0)
    state, _ = ledger.commit(state, _observation([-1.0, 0.0], [2.0]), 1)
    updated, report = ledger.commit(state, _observation([0.0, 1.0], [3.0]), 2)
    assert report["replaced"] == 1
    torch.testing.assert_close(updated.keys[0, 0], torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(updated.keys[0, 1], torch.tensor([-1.0, 0.0], dtype=torch.float64))


def test_read_solves_only_selected_valid_slots():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=2, topk=1)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("selected-only", dtype=torch.float64)
    state, _ = ledger.commit(state, _observation([1.0, 0.0], [2.0]), 0)
    # Unused capacity has no learned precision to solve. Its contents must not
    # affect a query routed to the valid slot.
    state.precision[0, 1] = -torch.eye(2, dtype=torch.float64)
    result = ledger.read(state, torch.tensor([[[1.0, 0.0]]], dtype=torch.float64))
    assert bool(result.has_memory[0, 0])
    assert torch.isfinite(result.value).all()


def test_streaming_and_batched_support_have_the_same_ridge_state():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1,
                               capacity=1, topk=1, decay=1.0, merge_threshold=0.5)
    ledger = AssociativeTTTLedger(cfg)
    first = _observation([1.0, 0.0], [2.0])
    second = _observation([0.6, 0.8], [3.0])
    batch = ObservationBatch(
        torch.cat([first.keys, second.keys], dim=1),
        torch.cat([first.values, second.values], dim=1),
        torch.cat([first.geometry, second.geometry], dim=1),
        torch.cat([first.confidence, second.confidence], dim=1),
        torch.cat([first.source_real, second.source_real], dim=1),
    )
    batched, _ = ledger.commit(ledger.new_state("batched", dtype=torch.float64), batch, 0)
    streamed, _ = ledger.commit(ledger.new_state("streamed", dtype=torch.float64), first, 0)
    streamed, _ = ledger.commit(streamed, second, 1)
    torch.testing.assert_close(batched.precision, streamed.precision)
    torch.testing.assert_close(batched.cross, streamed.cross)


def test_inference_and_differentiable_updates_are_numerically_identical():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1, capacity=1, topk=1)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("same-update", dtype=torch.float64)
    obs = _observation([0.6, 0.8], [2.0])
    differentiable, _ = ledger.commit(state, obs, 0, differentiable=True)
    inference, _ = ledger.commit_inference(state, obs, 0)
    torch.testing.assert_close(differentiable.precision, inference.precision)
    torch.testing.assert_close(differentiable.cross, inference.cross)


def test_nearly_collinear_support_is_finite_with_cholesky_jitter():
    cfg = AssociativeTTTConfig(key_dim=2, value_dim=1, geometry_dim=1,
                               capacity=1, topk=1, initial_precision=1e-8,
                               solve_jitter=1e-6, merge_threshold=0.5)
    ledger = AssociativeTTTLedger(cfg)
    state = ledger.new_state("collinear", dtype=torch.float64)
    state, _ = ledger.commit(state, _observation([1.0, 0.0], [1.0]), 0)
    state, _ = ledger.commit(state, _observation([1.0, 1e-6], [1.0]), 1)
    result = ledger.read(state, torch.tensor([[[1.0, 0.0]]], dtype=torch.float64))
    assert torch.isfinite(result.value).all()
