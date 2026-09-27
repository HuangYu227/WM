"""Behavioral checks for canonical writes and sparse memory interaction."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldttt.associative_ttt import AssociativeTTTConfig
from worldttt.grail_native import (
    GrailNativeController, NATIVE_COORDINATE_CONVENTION, TARGET_LAYERS,
    WRITER_LAYER, attach_grail_native, load_grail_adapter,
)
from worldttt.grail_network import GrailNetworkConfig

torch.set_num_threads(2)


def camera(t=2, batch=1):
    row = torch.cat((torch.eye(4).flatten(), torch.tensor([2., 2., 1., 1.])))
    return row.view(1, 1, 20).repeat(batch, t, 1)


def setup(mode='online'):
    torch.manual_seed(5)
    cfg = AssociativeTTTConfig(key_dim=4, value_dim=8, geometry_dim=30, capacity=16, topk=4,
                               merge_threshold=1.5, geometry_metric='ray_point',
                               coordinate_convention=NATIVE_COORDINATE_CONVENTION)
    network = GrailNetworkConfig(width=16, heads=4, neighbors=4, candidates=8, support_tokens=8, query_block=4)
    model = nn.Module()
    model.blocks = nn.ModuleList(nn.Linear(16, 16) for _ in range(16))
    for block in model.blocks:
        block.hidden_size = 16
    model.softmax_every_n, model.camctrl_layers_num, model.patch_size = 4, 16, (1, 1, 1)
    ctl = GrailNativeController(16, cfg, network=network, mode=mode)
    attach_grail_native(model, ctl)
    ctl.reset_episode('test', 1)
    return model, ctl


def run(model, context, x, cam=None):
    for i in TARGET_LAYERS:
        x = context.apply(model.blocks[i], x, camera() if cam is None else cam, (2, 2, 2), None)
    return x


def test_one_writer_eight_readers_empty_fallback_and_future_gradients():
    model, ctl = setup()
    x = torch.randn(1, 8, 16)
    clean = ctl.context(collect=True, chunk_id=0, real_frame_indices=(0, 1))
    torch.testing.assert_close(run(model, clean, x), x, rtol=0, atol=0)
    assert len(clean.observations) == 1
    assert clean.observations[0].keys.shape == (1, 8, 4)
    assert ctl.commit_clean(clean, 0)['writer_layer'] == WRITER_LAYER
    before = ctl.state.fingerprint()
    query = ctl.context('frozen', sigma=torch.tensor(0.5))
    out = run(model, query, x + .2)
    assert not torch.equal(out, x + .2)
    assert ctl.state.fingerprint() == before and not query.observations
    out.square().mean().backward()
    for module in (ctl.writer.address, ctl.writer.value, ctl.writer.depth, ctl.writer.write_gate[-1],
                   ctl.writer.qkv, ctl.readers['15'].query, ctl.readers['15'].key, ctl.readers['15'].gate[-1]):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
    assert all(p.grad is None for block in model.blocks for p in block.parameters())


def test_cfg_only_conditional_writer_and_single_commit():
    model, ctl = setup()
    x = torch.randn(2, 8, 16)
    clean = ctl.context(collect=True, chunk_id=0, cfg_conditional_start=1, real_frame_indices=(0,))
    run(model, clean, x)
    assert len(clean.observations) == 1 and clean.observations[0].keys.shape[0] == 1
    assert clean.observations[0].source_real.tolist() == [[True] * 4 + [False] * 4]
    ctl.commit_clean(clean, 0)
    assert ctl.hook_counts == {i: 1 for i in TARGET_LAYERS}
    with pytest.raises(ValueError, match='repeated'):
        ctl.commit_clean(clean, 0)
    query = ctl.context(cfg_conditional_start=1)
    out = run(model, query, x)
    assert not torch.equal(out[0], x[0]) and not torch.equal(out[1], x[1])


def test_association_excludes_self_and_reaches_writer():
    model, ctl = setup()
    clean = ctl.context(collect=True, chunk_id=0, real_frame_indices=(0, 1))
    run(model, clean, torch.randn(1, 8, 16))
    loss = ctl.ledger.association_loss(clean.observations[0])
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert ctl.writer.address.weight.grad.abs().sum() > 0
    assert ctl.writer.write_gate[-1].weight.grad.abs().sum() > 0


def test_reader_uses_selected_history_values():
    model, ctl = setup()
    clean = ctl.context(collect=True, chunk_id=0, real_frame_indices=(0, 1))
    x = torch.randn(1, 8, 16)
    run(model, clean, x)
    ctl.commit_clean(clean, 0)
    query = ctl.context('frozen')
    original = run(model, query, x)
    with torch.no_grad():
        ctl.state.cross = -ctl.state.cross
    altered = run(model, ctl.context('frozen'), x)
    assert not torch.allclose(original, altered)


def test_incompatible_layout_partial_clean_and_replayed_writer_fail():
    model, ctl = setup()
    clean = ctl.context(collect=True, chunk_id=0)
    clean.apply(model.blocks[2], torch.randn(1, 8, 16), camera(), (2, 2, 2), None)
    with pytest.raises(ValueError, match='eight'):
        ctl.commit_clean(clean, 0)
    with pytest.raises(ValueError, match='repeated'):
        clean.apply(model.blocks[2], torch.randn(1, 8, 16), camera(), (2, 2, 2), None)
    with pytest.raises(ValueError, match='writer'):
        ctl.context().apply(model.blocks[3], torch.randn(1, 8, 16), camera(), (2, 2, 2), None)


def test_adapter_roundtrip_freeze_and_architecture(tmp_path):
    model, ctl = setup()
    path = tmp_path / 'adapter.pt'
    ctl.base_checkpoint_hash = 'base'
    ctl.save_checkpoint(path, step=1)
    fresh, _ = setup()
    del fresh.worldttt_grail_controller
    restored, extra = load_grail_adapter(fresh, path, base_checkpoint_hash='base')
    assert restored.adapter_fingerprint() == ctl.adapter_fingerprint()
    assert extra['step'] == 1
    with pytest.raises(ValueError, match='hash'):
        another, _ = setup()
        del another.worldttt_grail_controller
        load_grail_adapter(another, path, base_checkpoint_hash='wrong')


def test_geometry_separates_identical_appearance_at_distinct_positions():
    from worldttt.associative_ttt import AssociativeTTTLedger, ObservationBatch
    from worldttt.grail_geometry import canonical_token_geometry
    _, ctl = setup()
    ledger = AssociativeTTTLedger(replace(ctl.config, merge_threshold=.8))
    cam = camera(t=2)
    cam[:, 1, 3] = 20.
    geometry = canonical_token_geometry(cam, (2, 1, 1), 1, torch.ones(1, 2, 1), torch.ones(1, 2, 1))
    keys = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
    observation = ObservationBatch(keys, torch.randn(1, 2, 8), geometry, torch.ones(1, 2),
                                   torch.ones(1, 2, dtype=torch.bool))
    state, _ = ledger.commit(ledger.new_state('spatial'), observation, 0)
    assert state.valid.sum() == 2
    read = ledger.read_slots(state, keys, geometry)
    assert read.ids[0, 0, 0] == 0 and read.ids[0, 1, 0] == 1
    assert read.valid.sum(-1).tolist() == [[1, 1]]
    # The rollback proxy must obey the same physical geometry cutoff.
    remote = geometry[:, :1].clone()
    remote[..., 3] += 100
    assert not ledger.read(state, keys[:, :1], remote).has_memory.any()


def test_writer_attention_aggregates_other_observations_but_preserves_source_mask():
    _, ctl = setup()
    x = torch.randn(1, 8, 16)
    visible = torch.ones(1, 8, 1)
    real = torch.tensor([[True] * 4 + [False] * 4])
    first = ctl.writer.encode(x, camera(), (2, 2, 2), 1, visible, real)[1]
    x[:, 4:] += torch.randn_like(x[:, 4:]) * 10
    changed = ctl.writer.encode(x, camera(), (2, 2, 2), 1, visible, real)[1]
    torch.testing.assert_close(first[:, :4], changed[:, :4], rtol=0, atol=0)
    assert not torch.equal(first[:, 4:], changed[:, 4:])
    # A changed real neighbour affects another token, proving this isn't pointwise MLP only.
    before = changed.clone()
    x[:, 1] += torch.randn_like(x[:, 1]) * 10
    after = ctl.writer.encode(x, camera(), (2, 2, 2), 1, visible, real)[1]
    assert not torch.equal(before[:, 0], after[:, 0])


def test_protected_slot_survives_200_far_chunks_then_revisit():
    from worldttt.associative_ttt import AssociativeTTTLedger, ObservationBatch
    from worldttt.grail_geometry import canonical_token_geometry
    _, ctl = setup()
    ledger = AssociativeTTTLedger(replace(ctl.config, capacity=2, topk=1, merge_threshold=.8))
    geom = canonical_token_geometry(camera(1), (1, 1, 1), 1, torch.ones(1, 1, 1), torch.ones(1, 1, 1))
    obs = ObservationBatch(torch.tensor([[[1., 0., 0., 0.]]]), torch.ones(1, 1, 8), geom,
                           torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool))
    state, _ = ledger.commit_inference(ledger.new_state('gap'), obs, 0)
    baseline = ledger.read_slots(state, obs.keys, obs.geometry).values.clone()
    far_geometry = geom.clone()
    far_geometry[..., 3] += 20
    far = ObservationBatch(obs.keys, -obs.values, far_geometry, obs.confidence,
                            torch.zeros_like(obs.source_real))
    for chunk in range(1, 201):
        state, _ = ledger.commit_inference(state, far, chunk)
    read = ledger.read_slots(state, obs.keys, obs.geometry)
    assert read.ids.item() == 0 and state.protected[0, 0]
    torch.testing.assert_close(read.values, baseline, rtol=2e-4, atol=2e-4)
