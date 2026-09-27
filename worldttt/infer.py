"""Actual WM pipeline entry, including all adaptation costs in stage-1 timing."""
import json
import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .check_cache import require_gate
from .memory import TTTConfig
from .runtime import WorldTTTController
from .sap_ttt.runtime import SapConfig, SapController
from .sap_binding import BindingConfig, BindingController
from .sana import load_config, make_pipeline


def infer(settings, case, output, mode, adapter=None, ablation=None, resume=None):
    from inference_video_scripts.wm.inference_sana_wm import (
        GenerationParams, RefinerSettings, load_intrinsics, resize_and_center_crop,
        transform_intrinsics_for_crop, write_video)
    if not torch.cuda.is_available():
        raise RuntimeError('Inference requires Linux CUDA')
    baseline = mode in {'off', 'binding_off'}
    if not baseline:
        require_gate(settings)
    if not baseline and not adapter:
        raise ValueError('Provide a trained adapter for frozen/kv_ttt/noise_ttt evaluation')
    config = load_config(settings['sana_config'])
    refiner = RefinerSettings(**settings['refiner']) if settings.get('refiner') else None
    pipe = make_pipeline(config, settings['base_checkpoint'], 'cuda', refiner=refiner)
    sap_run = mode in {'sap_frozen', 'sap_online', 'sap_no_read', 'sap_no_commit',
                       'sap_shuffle_text'}
    binding_run = mode in {'binding_frozen', 'binding_online'}
    if binding_run:
        if ablation is not None or resume is not None or settings.get('save_rollout_state', False):
            raise ValueError('SAP-Bind has independent paired modes and episode state')
        cfg = BindingConfig(**dict(settings['sap_binding'],
            mode='frozen' if mode == 'binding_frozen' else 'online'))
        ctl = BindingController(pipe.model, cfg)
    elif sap_run:
        if ablation is not None or resume is not None or settings.get('save_rollout_state', False):
            raise ValueError('SAP uses its own paired modes; rollout resume and legacy ablations do not apply')
        cfg = SapConfig(**dict(settings['sap'],
            mode='frozen' if mode == 'sap_frozen' else 'online',
            read_enabled=mode != 'sap_no_read',
            commit_enabled=mode != 'sap_no_commit',
            shuffle_query_text=mode == 'sap_shuffle_text'))
        ctl = SapController(pipe.model, cfg)
    else:
        cfg = TTTConfig(**dict(settings['ttt'], mode='off' if baseline else mode))
        if ablation == 'reset':
            cfg.reset_each_chunk = True
        elif ablation == 'no_protection':
            cfg.keep_weight = cfg.trust_weight = 0.
        elif ablation == 'shuffle':
            cfg.shuffle_targets = True
        ctl = WorldTTTController(pipe.model, cfg)
    ctl.base_checkpoint = settings['base_checkpoint']
    if adapter and not baseline:
        extra = ctl.load_checkpoint(adapter)
        if sap_run and extra.get('flow_trained') is False:
            raise ValueError('SAP feature-only adapter has untrained video fusion; run sap_ttt.train before generation')
        if binding_run and extra.get('flow_trained') is False:
            raise ValueError('SAP-Bind feature adapter has an untrained video gate; run sap_binding.train first')
    params = GenerationParams(num_frames=case.get('num_frames', 73), step=settings.get('steps', 50),
        cfg_scale=settings.get('cfg_scale', 5.), seed=case.get('seed', 42), sampling_algo='self_forcing',
        negative_prompt=case.get('negative_prompt', ''), save_stage1=True,
        num_cached_blocks=settings.get('num_cached_blocks', 2), sink_token=False,
        flow_shift=config.scheduler.inference_flow_shift)
    if params.num_frames < 73 or (params.num_frames - 1) % 8:
        raise ValueError('Use >=73 pixel frames with (num_frames-1) divisible by 8')
    image, src, resized, offset = resize_and_center_crop(Image.open(case['image']).convert('RGB'))
    c2w = np.load(case['camera'], allow_pickle=False)
    intrinsics = load_intrinsics(Path(case['intrinsics']), params.num_frames)
    intrinsics = transform_intrinsics_for_crop(intrinsics, src, resized, offset)
    if c2w.shape != (params.num_frames, 4, 4) or not np.isfinite(c2w).all():
        raise ValueError('Camera trajectory must contain one finite c2w matrix per output frame')
    if not np.isfinite(intrinsics).all() or not (intrinsics[:, :2] > 0).all():
        raise ValueError('Invalid measured intrinsics')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pipe.stage1_latent_path = output / 'latent.pt'

    def save_stage1_progress():
        ctl.write_metrics(output / 'updates.jsonl')
        if settings.get('save_episode_state', True) and ctl.state is not None:
            if sap_run:
                ctl.save_state(output / 'sap_state.pt')
            elif binding_run:
                ctl.save_state(output / 'binding_state.pt')
            else:
                ctl.state.save_state(output / 'ttt_state.pt')
        (output / 'stage1_meta.json').write_text(json.dumps(dict(
            stage1_seconds=pipe.stage1_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()), indent=2), encoding='utf-8')

    pipe.stage1_complete = save_stage1_progress
    trace = None
    if settings.get('address_trace', False):
        from .address_trace import AddressTrace
        pairs = sorted(case.get('revisit_pairs', []), key=lambda p: p.get('quality_score', 999))[:8]
        frames = {min(round(int(pair[key]) / 8), (params.num_frames - 1) // 8)
                  for pair in pairs for key in ('frame_a', 'frame_b')}
        if not frames:
            raise ValueError('address_trace requires preselected revisit_pairs in the case')
        trace = AddressTrace(pipe.model, cfg.layers, frames, c2w,
                             token_budget=settings.get('address_trace_tokens', 64))
        object.__setattr__(pipe.model, 'worldttt_address_trace', trace)
    signature = hashlib.sha256(json.dumps(dict(case=case, settings=settings, adapter=adapter), sort_keys=True).encode())
    for key in ('image', 'camera', 'intrinsics'):
        signature.update(Path(case[key]).read_bytes())
    if adapter:
        signature.update(Path(adapter).read_bytes())
    ctl.rollout_signature = signature.hexdigest()
    if settings.get('save_rollout_state', False):
        ctl.rollout_path = output / 'rollout.pt'
    if resume:
        if baseline:
            raise ValueError('WorldTTT rollout resume requires an active adapter mode')
        ctl.resume_snapshot = torch.load(resume, map_location='cpu', weights_only=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        result = pipe.generate(image, case['prompt'], c2w, intrinsics, params)
    finally:
        if trace is not None:
            trace.close()
            object.__setattr__(pipe.model, 'worldttt_address_trace', None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if trace is not None:
        trace.save(output / 'address_trace.pt')
    metrics = dict(mode=mode, ablation=ablation, case=case, adapter=str(adapter) if adapter else None,
        stage1_seconds=pipe.stage1_seconds, end_to_end_seconds=seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        ttt=asdict(cfg), generation=asdict(params), base_checkpoint=settings['base_checkpoint'],
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, refiner=settings.get('refiner'),
        resumed_from=resume, rollout_checkpoint_io_in_timing=ctl.rollout_path is not None)
    metrics['address_trace_records'] = len(trace.records) if trace is not None else 0
    torch.save(result['latent'], output / 'latent.pt')
    np.save(output / 'c2w.npy', c2w)
    video = result['video']
    if 'stage1_video' in result:
        # Native refiner drops the input frame. Restore it for benchmark frame
        # indices / camera GT; native VBench can then skip it explicitly.
        video = np.concatenate((result['stage1_video'][:1], video), axis=0)
    write_video(output, 'video', video, params.fps, pipe.logger)
    if 'stage1_video' in result:
        write_video(output, 'stage1', result['stage1_video'], params.fps, pipe.logger)
    if settings.get('save_episode_state', True) and ctl.state is not None:
        if sap_run:
            ctl.save_state(output / 'sap_state.pt')
        elif binding_run:
            ctl.save_state(output / 'binding_state.pt')
        else:
            ctl.state.save_state(output / 'ttt_state.pt')
    ctl.write_metrics(output / 'updates.jsonl')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    return metrics
