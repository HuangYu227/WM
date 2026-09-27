"""Exercise the actual sample_chunks control flow using a CPU model/scheduler."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from test_episode import ToyModel
from worldttt.memory import TTTConfig
from worldttt.runtime import WorldTTTController


class Scheduler:
    def __init__(self, **kw):
        self.timesteps = torch.tensor([800., 200.])

    def step(self, prediction, t, sample, **kw):
        return (sample + prediction * .02,)


def actual_sampler():
    source = Path(__file__).resolve().parents[2] / 'diffusion/scheduler/self_forcing_flow_euler_sampler.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SelfForcingFlowEulerCamCtrl')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_chunks')
    ns = dict(torch=torch, os=os, FlowMatchEulerDiscreteScheduler=Scheduler, _NUM_CACHE_SLOTS=10,
              retrieve_timesteps=lambda scheduler, *a: (scheduler.timesteps, 2),
              tqdm=lambda it, **kw: it, Transformer2DModelOutput=type('UnusedOutput', (), {}))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
    return ns['sample_chunks']


def run(mode, cfg_scale=1., snapshot=None, save_path=None, stop=False, tracing=False):
    torch.manual_seed(4)
    model = ToyModel()
    ctl = WorldTTTController(model, TTTConfig(mode=mode, layers=(1,), input_dim=3,
        hidden_dim=5, support_tokens=3, anchor_capacity=6), validate_blocks=False)
    ctl.resume_snapshot = snapshot
    ctl.rollout_path = save_path
    trace_calls = []
    class Trace:
        def begin_chunk(self, *args):
            trace_calls.append(('begin', args[0]))
        def end_chunk(self):
            trace_calls.append(('end', None))
    # The real camera sampler injects sliced conditions through its forward_long
    # wrapper. This toy has no forward_long; provide the same slice at its entry.
    class Call:
        worldttt_controller = ctl
        worldttt_address_trace = Trace() if tracing else None
        def __call__(self, z, t, y=None, **kw):
            return model(z, t, camera_conditions=torch.zeros(z.shape[0], z.shape[2], 20), **kw)
    sampler = SimpleNamespace(model=Call(), condition=torch.ones(1, 1, 1, 4),
        uncondition=torch.zeros(1, 1, 1, 4), mask=None, cfg_scale=cfg_scale, flow_shift=9.8,
        base_chunk_frames=3, num_cached_blocks=2, num_model_blocks=1, sink_token=False,
        model_kwargs={'data_info': {'condition_frame_info': {0: 0}}},
        create_autoregressive_segments=lambda n: [0, 4, 7, 10],
        _initialize_kv_cache=lambda n: [[torch.tensor(0.)] for _ in range(n)],
        accumulate_kv_cache=lambda cache, i: (cache[max(0, i - 1)], min(i, 2), 0, 0))
    z = torch.randn(1, 4, 10, 2, 2)
    initial = z[:, :, 0].clone()
    iterator = actual_sampler()(sampler, z, steps=2)
    chunks = [next(iterator)] if stop else list(iterator)
    iterator.close()
    assert model.commits == ([0] if stop else [4, 7] if snapshot else [0, 4, 7])
    assert torch.equal(z[:, :, 0], initial)
    assert len(chunks) == (1 if stop else 2 if snapshot else 3)
    if tracing:
        assert trace_calls == [item for chunk in range(3) for item in (('begin', chunk), ('end', None))]
    return ctl, z


def test_real_sampler_commits_once_and_adapts_under_no_grad():
    ctl, adapted = run('noise_ttt')
    assert ctl.state.updates == 3
    assert [r['chunk'] for r in ctl.metrics] == [0, 1, 2]
    frozen, without_updates = run('frozen')
    assert frozen.state.updates == 0
    assert not torch.equal(adapted[:, :, 4:], without_updates[:, :, 4:])


def test_cfg_allocates_separate_fast_weights():
    ctl, _ = run('noise_ttt', cfg_scale=5.)
    assert all(w.shape[0] == 2 for w in ctl.state.weights[1])


def test_off_trace_is_read_only_and_cache_commit_count_unchanged():
    _, original = run('off')
    _, traced = run('off', tracing=True)
    assert torch.equal(original, traced)


def test_complete_sampler_resume_matches_uninterrupted(tmp_path):
    full_ctl, full_video = run('noise_ttt', cfg_scale=5.)
    path = tmp_path / 'rollout.pt'
    run('noise_ttt', cfg_scale=5., save_path=path, stop=True)
    snapshot = torch.load(path, weights_only=True)
    assert snapshot['next_chunk'] == 1
    resumed, video = run('noise_ttt', cfg_scale=5., snapshot=snapshot)
    assert torch.equal(video, full_video)
    assert resumed.metrics == full_ctl.metrics
    for a, b in zip(resumed.state.weights[1], full_ctl.state.weights[1]):
        assert torch.equal(a, b)


def test_sap_sampler_reads_each_denoise_step_and_commits_once_after_clean_pass():
    events = []
    class Sap:
        config = SimpleNamespace(mode='online')
        def reset_episode(self, episode, batch):
            events.append(('reset', batch))
        def context(self, t, collect=False, chunk=0, source='unspecified'):
            return SimpleNamespace(collect=collect, chunk=chunk, source=source)
        def commit(self, context, chunk):
            events.append(('commit', chunk))
            assert context.collect and context.chunk == chunk
    class Model:
        worldttt_controller = None
        worldttt_sap_controller = Sap()
        worldttt_address_trace = None
        def __call__(self, z, t, y=None, **kw):
            ctx = kw['sap_context']
            events.append(('clean' if ctx.collect else 'read', ctx.chunk))
            return torch.zeros_like(z), kw['kv_cache']
    sampler = SimpleNamespace(model=Model(), condition=torch.ones(1, 1, 1, 4),
        uncondition=torch.zeros(1, 1, 1, 4), mask=None, cfg_scale=1., flow_shift=9.8,
        base_chunk_frames=3, num_cached_blocks=2, num_model_blocks=1, sink_token=False,
        model_kwargs={'data_info': {'condition_frame_info': {0: 0}}},
        create_autoregressive_segments=lambda n: [0, 4, 7, 10],
        _initialize_kv_cache=lambda n: [[torch.tensor(0.)] for _ in range(n)],
        accumulate_kv_cache=lambda cache, i: (cache[max(0, i - 1)], min(i, 2), 0, 0))
    list(actual_sampler()(sampler, torch.zeros(1, 4, 10, 2, 2), steps=2))
    assert events == [('reset', 1)] + [item for chunk in range(3) for item in
        [('read', 0), ('read', 0), ('clean', chunk), ('commit', chunk)]]
    assert [e for e in events if e[0] == 'read'] == [('read', 0)] * 6
    assert [e for e in events if e[0] == 'clean'] == [('clean', i) for i in range(3)]
    assert [e for e in events if e[0] == 'commit'] == [('commit', i) for i in range(3)]
