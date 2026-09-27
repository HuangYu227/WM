import pytest

torch = pytest.importorskip("torch")


def _observation(cfg, *, source_real=True, n=2, offset=0.0):
    from worldttt.grail import GrailObservation
    g = torch.Generator().manual_seed(7)
    appearance = torch.randn(1, n, cfg.appearance_dim, generator=g)
    geometry = torch.randn(1, n, cfg.geometry_dim, generator=g) + offset
    value = torch.randn(1, n, cfg.value_dim, generator=g)
    confidence = torch.ones(1, n)
    source = torch.full((1, n), source_real, dtype=torch.bool)
    relations = torch.tensor([[[0, 1, 1., 0., 0., 0., 0., 0.]]]) if n >= 2 else None
    return GrailObservation(appearance, geometry, value, confidence, source, relations)


def test_empty_read_is_finite_and_reports_no_memory():
    from worldttt.grail import GrailConfig, GrailMemory, GrailState
    cfg = GrailConfig(appearance_dim=4, geometry_dim=3, value_dim=5, address_dim=6,
                      capacity=4, topk=2, protected_slots=4)
    memory, state = GrailMemory(cfg), GrailState.new(cfg, "episode", 1)
    result = memory.read(state, torch.zeros(1, 3, 4), torch.zeros(1, 3, 3))
    assert torch.isfinite(result.value).all()
    assert float(result.has_memory.sum()) == 0
    assert torch.isfinite(result.weights).all()


def test_commit_is_copy_on_write_and_records_generation_and_edges():
    from worldttt.grail import GrailConfig, GrailMemory, GrailState
    cfg = GrailConfig(appearance_dim=4, geometry_dim=3, value_dim=5, address_dim=6,
                      capacity=4, edge_capacity=4, event_capacity=8, topk=2,
                      protected_slots=1)
    memory, state = GrailMemory(cfg), GrailState.new(cfg, "episode", 1)
    old = state.state_dict()
    state, report = memory.commit(state, _observation(cfg), 0)
    assert report["committed"] and report["created"] == 2
    assert state.updates == 1 and state.last_chunk == 0
    assert int(state.edge_valid.sum()) == 1
    assert int(state.event_seen.item()) == 2
    assert (old["valid"] == 0).all()  # original copy remains empty
    generation = state.slot_generation.clone()
    generated = _observation(cfg, source_real=False, n=1, offset=20.)
    next_state, report = memory.commit(state, generated, 1)
    assert report["committed"]
    assert (next_state.slot_generation >= generation).all()
    with pytest.raises(ValueError, match="expected chunk"):
        memory.commit(next_state, generated, 1)


def test_read_supports_outer_loop_gradients_when_requested():
    from worldttt.grail import GrailConfig, GrailMemory, GrailState
    cfg = GrailConfig(appearance_dim=4, geometry_dim=3, value_dim=5, address_dim=6,
                      capacity=4, topk=1, protected_slots=4)
    memory, state = GrailMemory(cfg), GrailState.new(cfg, "episode", 1)
    state, _ = memory.commit(state, _observation(cfg, n=1), 0)
    result = memory.read(state, torch.randn(1, 2, 4), torch.randn(1, 2, 3),
                         differentiable=True)
    result.value.square().mean().backward()
    assert memory.read_gate.grad is not None
    assert memory.appearance_write.weight.grad is not None


def test_canonical_geometry_uses_anchor_gauge():
    from worldttt.grail import canonical_ray_geometry
    c2w = torch.eye(4).unsqueeze(0)
    k = torch.eye(3).unsqueeze(0)
    uv = torch.tensor([[[0., 0.], [1., 0.]]])
    result = canonical_ray_geometry(c2w, k, uv, torch.ones(1, 2), c2w)
    assert result.shape == (1, 2, 13)
    assert torch.allclose(result[..., :3], torch.zeros_like(result[..., :3]))
    assert torch.allclose(result[..., 3:6].norm(dim=-1), torch.ones(1, 2))
