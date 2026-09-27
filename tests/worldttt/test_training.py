import random
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_episode import ToyModel
from worldttt.episode import EpisodeModel
from worldttt.memory import TTTConfig
from worldttt.runtime import WorldTTTController
from worldttt import train as training


def make_episode():
    model = ToyModel()
    ctl = WorldTTTController(model, TTTConfig(layers=(1,), input_dim=3,
        hidden_dim=5, support_tokens=3, anchor_capacity=6), validate_blocks=False)

    def caches(_):
        state = [[torch.tensor(0.)] for _ in range(3)]
        return state, lambda i: state[max(i - 1, 0)]

    class ToySchedule:
        timesteps = torch.tensor([800., 200.])
        def step(self, prediction, timestep, sample, *args, **kwargs):
            return (sample + .02 * prediction,)

    episode = EpisodeModel(model, ctl, cache_factory=caches,
        schedule_factory=lambda *args: ToySchedule())
    batch = dict(latent=torch.randn(1, 4, 10, 2, 2), camera=torch.randn(1, 10, 20),
                 plucker=torch.randn(1, 48, 10, 2, 2), text=None, mask=None, episode_id='val')
    return episode, ctl, batch


def test_validation_episode_adapts_without_outer_graph_or_parameter_changes():
    episode, ctl, batch = make_episode()
    before = [p.detach().clone() for p in episode.parameters()]
    with torch.no_grad():
        loss = episode(**batch, seed=21, meta_grad=False)
    assert torch.isfinite(loss) and not loss.requires_grad
    assert ctl.state.updates == 2
    assert all(w.grad_fn is None for weights in ctl.state.weights.values() for w in weights)
    assert all(p.grad is None and torch.equal(p, old) for p, old in zip(episode.parameters(), before))


def test_validation_subset_has_no_ddp_padding_or_missing_samples():
    for world in (1, 2, 4, 8):
        shards = [training.validation_indices(7, 5, 123, rank, world) for rank in range(world)]
        union = [i for shard in shards for i in shard]
        assert len(union) == len(set(union)) == 5
        assert set(union) == set(training.validation_indices(7, 5, 123, 0, 1))
    assert len(training.validation_indices(3, 16, 123, 0, 1)) == 3


def test_validation_repeatability_rng_and_training_state_isolation(monkeypatch):
    episode, ctl, batch = make_episode()
    sentinel = ctl.reset_episode('training', 1, training=True)
    saved_metrics = ctl.metrics
    calls = []

    def fixture(sample, pipeline, device):
        calls.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
        return sample

    monkeypatch.setattr(training, 'make_fixture', fixture)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    first = training.validate(episode, [batch, batch], None, torch.device('cpu'), [0, 1], 222)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert ctl.state is sentinel and ctl.metrics is saved_metrics
    second = training.validate(episode, [batch, batch], None, torch.device('cpu'), [0, 1], 222)
    assert first == second and first['samples'] == 2
    assert calls[:2] == calls[2:]


def test_validation_separately_scores_generated_history(monkeypatch):
    episode, _, batch = make_episode()
    monkeypatch.setattr(training, 'make_fixture', lambda sample, *args: sample)
    result = training.validate(episode, [batch], None, torch.device('cpu'), [0], 222,
                               histories=('real', 'generated'))
    assert result['samples'] == 1
    assert set(result) == {'loss', 'generated_loss', 'samples'}
    assert all(np.isfinite(result[key]) for key in ('loss', 'generated_loss'))


def test_real_and_generated_history_share_heldout_query_noise():
    episode, _, batch = make_episode()
    seen = []
    forward = episode.backbone.forward
    def capture(z, t, *args, **kwargs):
        if kwargs['start_f'] == 7:
            seen.append((z.detach().clone(), t.detach().clone()))
        return forward(z, t, *args, **kwargs)
    episode.backbone.forward = capture
    episode(**batch, generated=False, seed=41, meta_grad=False)
    episode(**batch, generated=True, seed=41, meta_grad=False)
    assert len(seen) == 2
    assert all(torch.equal(a, b) for a, b in zip(*seen))


def test_validation_reduces_sum_and_count_including_empty_rank(monkeypatch):
    episode, _, _ = make_episode()
    # A rank with no selected samples must still participate in the reduction.
    def reduce(stats):
        assert stats.tolist() == [0., 0.]
        stats.add_(torch.tensor([9., 3.], dtype=stats.dtype))
    monkeypatch.setattr(training.dist, 'all_reduce', reduce)
    result = training.validate(episode, [], None, torch.device('cpu'), [], 22, world=4)
    assert result == {'loss': 3., 'samples': 3}


def test_last_advances_best_only_improves_and_rejects_nan(tmp_path):
    _, ctl, _ = make_episode()
    best = float('inf')
    for step, val in ((1, 3.), (2, 4.), (3, 2.), (4, 2.), (5, None)):
        best = training.save_checkpoints(ctl, tmp_path, step, {}, val, best)
    last = torch.load(tmp_path / 'last.pt', weights_only=True)
    saved_best = torch.load(tmp_path / 'best.pt', weights_only=True)
    assert last['extra']['step'] == 5
    assert saved_best['extra']['step'] == 3
    assert saved_best['extra']['val_loss'] == best == 2.
    assert set(p.name for p in tmp_path.iterdir()) == {'last.pt', 'best.pt'}
    ctl.load_checkpoint(tmp_path / 'best.pt')  # Existing inference/warm-start format.
    with pytest.raises(FloatingPointError):
        training.save_checkpoints(ctl, tmp_path, 6, {}, float('nan'), best)
    assert torch.load(tmp_path / 'last.pt', weights_only=True)['extra']['step'] == 5


def test_failed_checkpoint_write_preserves_previous_last(tmp_path, monkeypatch):
    _, ctl, _ = make_episode()
    training.save_checkpoints(ctl, tmp_path, 1, {}, 3., float('inf'))
    def fail(path, extra):
        path.write_bytes(b'incomplete')
        raise OSError('simulated full disk')
    monkeypatch.setattr(ctl, 'save_checkpoint', fail)
    with pytest.raises(OSError):
        training.save_checkpoints(ctl, tmp_path, 2, {}, 2., 3.)
    assert torch.load(tmp_path / 'last.pt', weights_only=True)['extra']['step'] == 1
    assert torch.load(tmp_path / 'best.pt', weights_only=True)['extra']['step'] == 1


def test_train_loop_progress_validation_and_final_save_on_cpu(tmp_path, monkeypatch, capsys):
    # Exercise the real optimizer/train loop; only CUDA and heavyweight SANA loading are replaced.
    _, _, batch = make_episode()
    class CpuTorch:
        cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda _: None,
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 1234)
        @staticmethod
        def device(*args):
            return torch.device('cpu')
        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(training, 'torch', CpuTorch())
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(training, 'load_config', lambda _: SimpleNamespace(
        scheduler=SimpleNamespace(inference_flow_shift=9.8)))
    monkeypatch.setattr(training, 'make_pipeline', lambda *args, **kwargs: SimpleNamespace(model=ToyModel()))
    monkeypatch.setattr(training, 'WorldTTTController', lambda model, cfg:
        WorldTTTController(model, cfg, validate_blocks=False))
    monkeypatch.setattr(training, 'EpisodeDataset', lambda *args: [batch, batch, batch])
    monkeypatch.setattr(training, 'make_fixture', lambda sample, *args: sample)
    def episode(model, controller, *args):
        def caches(_):
            state = [[torch.tensor(0.)] for _ in range(3)]
            return state, lambda i: state[max(i - 1, 0)]
        return EpisodeModel(model, controller, cache_factory=caches,
            schedule_factory=lambda *args: SimpleNamespace(timesteps=torch.tensor([800., 200.])))
    monkeypatch.setattr(training, 'EpisodeModel', episode)
    cfg = dict(sana_config='toy', base_checkpoint='toy', data={}, manifest='unused',
        ttt=dict(layers=[1], input_dim=3, hidden_dim=5, support_tokens=3, anchor_capacity=6),
        max_steps=3, real_prefix_steps=3, gradient_accumulation=2,
        save_every=2, val_every=2, val_max_samples=2)
    training.train(cfg, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / 'train.jsonl').read_text().splitlines()]
    vals = [json.loads(line) for line in (tmp_path / 'val.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2, 3]
    assert [row['step'] for row in vals] == [2, 3]  # Final step is always evaluated.
    assert all(row['samples'] == 2 for row in vals)
    assert torch.load(tmp_path / 'last.pt', weights_only=True)['extra']['step'] == 3
    assert torch.load(tmp_path / 'best.pt', weights_only=True)['extra']['val_loss'] == min(row['loss'] for row in vals)
    assert not list(tmp_path.glob('*.tmp.pt')) and not (tmp_path / 'adapter.pt').exists()
    assert 'WorldTTT train' in capsys.readouterr().err
    with pytest.raises(FileExistsError, match='fresh output'):
        training.train(cfg, tmp_path)
