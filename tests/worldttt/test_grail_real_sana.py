"""Actual SANA classes, random small weights. No model/sampler mocks or AST extraction.

Run explicitly in the SANA environment (CPU reference kernels or CUDA kernels).
These tests do not measure pretrained video quality.
"""
import os
import json
import time
import pytest
os.environ.setdefault('GDN_DISABLE_COMPILE', '1')
os.environ.setdefault('SANA_USE_LIGER', '0')
os.environ.setdefault('DISABLE_XFORMERS', '1')

import torch
from torch import nn


def make_model():
    from diffusion.model.nets.sana_multi_scale_video_camctrl import SanaMSVideoCamCtrlStreaming
    torch.set_num_threads(2)
    torch.manual_seed(421)
    model = SanaMSVideoCamCtrlStreaming(input_size=2, patch_size=(1, 1, 1), in_channels=4,
        hidden_size=64, depth=16, num_heads=4, linear_head_dim=16, caption_channels=32,
        model_max_length=4, learn_sigma=False, pred_sigma=False, qk_norm=True,
        attn_type='BidirectionalGDNTriton', ffn_type='CachedGLUMBConvTemp',
        pos_embed_type='casual_wan_rope', camctrl_layers_num=16, cam_attn_compress=1,
        chunk_size=3, chunk_split_strategy='first_chunk_plus_one', conv_kernel_size=4,
        use_autograd_kernel=True).eval()
    # Native initialization zeros the output head; nonzero random head is required
    # for a meaningful adapter-sensitivity / gradient check, not a pretrained claim.
    nn.init.normal_(model.final_layer.linear.weight, std=.02)
    for block in model.blocks:
        block.cross_attn.set_use_xformers(False)
    return model


def fixture():
    torch.manual_seed(123)
    camera = torch.cat((torch.eye(4).reshape(16), torch.tensor([1., 1., .5, .5]))).reshape(1, 1, 20).repeat(1, 10, 1)
    return dict(latent=torch.randn(1, 4, 10, 2, 2), camera=camera,
                text=torch.randn(1, 1, 4, 32), mask=torch.ones(1, 4), episode_id='native-smoke')


def attach(model):
    from worldttt.associative_ttt import AssociativeTTTConfig
    from worldttt.grail_native import GrailNativeController, attach_grail_native, NATIVE_COORDINATE_CONVENTION
    from worldttt.grail_network import GrailNetworkConfig
    ctl = GrailNativeController(64, AssociativeTTTConfig(key_dim=4, value_dim=8, capacity=16,
        topk=4, geometry_dim=30, geometry_metric='ray_point', coordinate_convention=NATIVE_COORDINATE_CONVENTION,
        merge_threshold=1.5, min_read_confidence=0.), network=GrailNetworkConfig(width=16, heads=4,
        neighbors=4, candidates=16, support_tokens=16, query_block=8), mode='online')
    attach_grail_native(model, ctl)
    return ctl


def test_real_sana_forward_and_outer_future_gradient(record_property):
    from worldttt.grail_train import GrailEpisodeModel
    model = make_model()
    ctl = attach(model)
    module = GrailEpisodeModel(model, steps=2)
    history = {'key': [], 'value': []}
    def capture(name):
        def hook(module, inputs, output):
            output.retain_grad()
            history[name].append(output)
        return hook
    handles = [ctl.writer.address.register_forward_hook(capture('key')),
               ctl.writer.value.register_forward_hook(capture('value'))]
    result = module(**fixture())
    assert result['variant_gate_mean'] == {}
    result['future'].backward()
    for handle in handles:
        handle.remove()
    history_gradients = {name: [0. if t.grad is None else float(t.grad.abs().sum()) for t in outputs[:2]]
                         for name, outputs in history.items()}
    assert all(sum(values) > 0 for values in history_gradients.values())
    record_property('historical_projection_gradient_l1', json.dumps(history_gradients))
    assert torch.isfinite(result['loss'])
    assert result['query_context'].call_counts == {i: 1 for i in (2, 3, 6, 7, 10, 11, 14, 15)}
    assert ctl.state.update_count.tolist() == [2]
    for parameter in (ctl.writer.address.weight, ctl.writer.value.weight, ctl.writer.qkv.weight,
                      ctl.writer.write_gate[0].weight, ctl.writer.depth.weight,
                      ctl.readers['15'].query.weight, ctl.readers['15'].gate[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith('worldttt_grail_controller.'))
    record_property('future_loss', float(result['future'].detach()))
    record_property('association_loss', float(result['association'].detach()))
    record_property('future_gradient_l1', json.dumps({name: float(p.grad.abs().sum())
        for name, p in ctl.named_parameters() if p.grad is not None}))


def test_paired_future_query_uses_one_history_and_read_only_variants():
    from worldttt.grail_train import GrailEpisodeModel
    model = make_model()
    ctl = attach(model)
    module = GrailEpisodeModel(model, steps=2)
    with torch.no_grad():
        result = module(**fixture(), seed=42,
                        query_variants=('ridge', 'no_read', 'prototype', 'shuffle_value'), record_trace=True)
    assert set(result['variant_future']) == {'ridge', 'no_read', 'prototype', 'shuffle_value'}
    torch.testing.assert_close(result['future'], result['variant_future']['ridge'])
    assert all(torch.isfinite(loss) for loss in result['variant_future'].values())
    assert ctl.state.update_count.tolist() == [2]
    assert result['query_context'].call_counts == {i: 1 for i in (2, 3, 6, 7, 10, 11, 14, 15)}
    assert result['variant_hook_counts']['no_read'] == result['variant_hook_counts']['ridge']
    assert result['variant_gate_mean']['no_read'] == 0.
    assert result['variant_gate_mean']['prototype'] == pytest.approx(result['variant_gate_mean']['ridge'])
    assert result['variant_gate_mean']['shuffle_value'] == pytest.approx(result['variant_gate_mean']['ridge'])
    assert result['variant_slot_coverage']['ridge'] > 0.
    assert len(result['memory_trace']) == 2
    assert all(row['committed'] and row['hook_counts'] == {i: 1 for i in (2, 3, 6, 7, 10, 11, 14, 15)}
               for row in result['memory_trace'])
    assert all(row['accepted'] == sum(slot['observations'] for slot in row['slots'])
               for row in result['memory_trace'])
    assert result['memory_trace'][1]['precision_delta_fro'] > 0.
    assert result['query_input_fingerprint']
    with torch.no_grad():
        generated = module(**fixture(), generated=True, seed=42, query_variants=('ridge',), record_trace=True)
    assert generated['query_input_fingerprint'] == result['query_input_fingerprint']


def test_real_sana_future_gradient_logging():
    from worldttt.grail_train import GrailEpisodeModel, train_step
    model = make_model()
    ctl = attach(model)
    module = GrailEpisodeModel(model, steps=2)
    optimizer = torch.optim.AdamW([p for p in ctl.parameters() if p.requires_grad], lr=1e-4)
    row = train_step(module, optimizer, fixture(), diagnostics=True)
    for name in ('writer.address', 'writer.value', 'writer.write_gate.0', 'readers.15.query'):
        assert row['future_gradients'][name]['norm'] > 0
    assert all(torch.isfinite(p).all() for p in ctl.parameters())


def sampler(model, batch, cfg=2.):
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
    return SelfForcingFlowEulerCamCtrl(model, batch['text'], batch['text'] * 0, cfg,
        flow_shift=1., base_chunk_frames=3, num_cached_blocks=2,
        model_kwargs=dict(camera_conditions=batch['camera'].repeat(2 if cfg > 1 else 1, 1, 1), mask=batch['mask'].repeat(2 if cfg > 1 else 1, 1),
                          data_info={'condition_frame_info': {0: 0.}}))


def test_native_sampler_modes_counts_and_resume(tmp_path, record_property):
    model, batch = make_model(), fixture()
    ctl = attach(model)
    model.worldttt_grail_protocol = dict(base_checkpoint_hash='random-smoke', source_tree_hash='smoke',
                                       data_manifest_hash='synthetic')
    outputs, durations = {}, {}
    initial_weights = {n: p.detach().clone() for n, p in ctl.named_parameters()}
    for mode in ('off', 'frozen', 'online'):
        ctl.mode = mode
        started = time.perf_counter()
        with torch.no_grad():
            outputs[mode] = sampler(model, batch).sample(batch['latent'].clone(), steps=2,
                generator=torch.Generator().manual_seed(9))
        durations[mode] = time.perf_counter() - started
        if mode != 'off':
            assert ctl.hook_counts == {i: 9 for i in (2, 3, 6, 7, 10, 11, 14, 15)}
    torch.testing.assert_close(outputs['off'], outputs['frozen'], rtol=0, atol=0)
    assert (outputs['online'][:, :, 4:] - outputs['off'][:, :, 4:]).abs().max() > 0
    assert len(ctl.metrics) == 3 and ctl.state.update_count.tolist() == [3]
    assert all(row['writer_layer'] == 2 for row in ctl.metrics)
    for name, p in ctl.named_parameters():
        torch.testing.assert_close(p, initial_weights[name], rtol=0, atol=0)
    complete_state = ctl.state.fingerprint()
    path = tmp_path / 'rollout.pt'
    ctl.rollout_path = path
    with torch.no_grad():
        iterator = sampler(model, batch).sample_chunks(batch['latent'].clone(), steps=2,
            generator=torch.Generator().manual_seed(9))
        next(iterator)
        iterator.close()
    assert ctl.state.update_count.tolist() == [1]
    ctl.resume_path, ctl.rollout_path = path, None
    with torch.no_grad():
        resumed = sampler(model, batch).sample(batch['latent'].clone(), steps=2,
            generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(resumed, outputs['online'], rtol=0, atol=0)
    assert ctl.state.fingerprint() == complete_state
    record_property('off_frozen_max_abs', float((outputs['off'] - outputs['frozen']).abs().max()))
    record_property('off_online_max_abs', float((outputs['off'] - outputs['online']).abs().max()))
    record_property('hook_counts_full', json.dumps({i: 9 for i in (2, 3, 6, 7, 10, 11, 14, 15)}))
    record_property('seconds', json.dumps(durations))


def test_outer_optimizer_real_generated_and_production_mount(tmp_path, record_property):
    from worldttt.grail_train import GrailEpisodeModel, train_step
    from worldttt.grail_entry import mount_grail
    from worldttt.provenance import file_sha256
    from inference_video_scripts.wm.inference_sana_wm import SanaWMPipeline, GenerationParams, _build_parser
    model = make_model()
    base = tmp_path / 'random-base.pt'
    torch.save(model.state_dict(), base)
    ctl = attach(model)
    ctl.base_checkpoint_hash = file_sha256(base)
    module = GrailEpisodeModel(model, steps=2)
    optimizer = torch.optim.AdamW([p for p in ctl.parameters() if p.requires_grad], lr=1e-3)
    original = ctl.writer.address.weight.detach().clone()
    metrics = [train_step(module, optimizer, fixture(), generated=g, seed=17) for g in (False, True)]
    assert not torch.equal(original, ctl.writer.address.weight)
    adapter = tmp_path / 'adapter.pt'
    ctl.save_checkpoint(adapter, step=2, flow_trained=True)
    fresh = make_model()
    restored = mount_grail(fresh, adapter, mode='online', base_checkpoint=base)
    assert restored.adapter_fingerprint() == ctl.adapter_fingerprint()
    assert not any(p.requires_grad for n, p in fresh.named_parameters() if not n.startswith('worldttt_grail_controller.'))
    # Exercise the actual production solver dispatch without downloading a VAE/text model.
    pipeline = object.__new__(SanaWMPipeline)
    pipeline.model = fresh
    batch = fixture()
    with torch.no_grad():
        latent = pipeline._dispatch_solver('self_forcing', batch['latent'].clone(), batch['text'],
            batch['text'] * 0, 1., 1., 2,
            dict(mask=batch['mask'], camera_conditions=batch['camera'], data_info={'condition_frame_info': {0: 0.}}),
            [0, 4, 7, 10], torch.Generator().manual_seed(10), GenerationParams(sampling_algo='self_forcing'))
    assert torch.isfinite(latent).all() and restored.state.update_count.tolist() == [3]
    record_property('optimizer_steps', json.dumps(metrics))
    args = _build_parser().parse_args(['--image', 'image.png', '--prompt', 'prompt.txt', '--output_dir', str(tmp_path),
        '--action', 'w-8', '--grail_adapter', str(adapter), '--grail_mode', 'online'])
    assert args.grail_adapter == adapter and args.grail_mode == 'online'


def test_native_cpu_short_convolution_cache():
    from fla.modules import ShortConvolution
    from diffusion.model.ops.fused_streaming import _cached_temporal_short_conv
    conv = ShortConvolution(1, 3, bias=False, activation=None)
    with torch.no_grad():
        conv.weight.fill_(1.)
    x = torch.arange(1., 5.).reshape(1, 4, 1)
    out, cache = _cached_temporal_short_conv(x, conv, (4, 1, 1), None, True)
    torch.testing.assert_close(out.flatten(), torch.tensor([6., 10., 10., 9.]))
    out, _ = _cached_temporal_short_conv(torch.tensor([[[5.]]]), conv, (1, 1, 1), cache, False)
    torch.testing.assert_close(out.flatten(), torch.tensor([12.]))


def test_reusing_production_sampler_does_not_retain_old_episode_closures():
    import gc
    import weakref
    model, batch = make_model(), fixture()
    sampler(model, batch)
    old = weakref.ref(model.forward_long)
    sampler(model, batch)
    gc.collect()
    assert old() is None
