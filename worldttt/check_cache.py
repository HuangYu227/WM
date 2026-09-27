"""Real checkpoint numerical gate; must run on the target CUDA stack."""
import gc
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from .episode import native_cache_manager
from .memory import TTTConfig
from .runtime import BASE_REVISION, WorldTTTController, clone_cache
from .sana import build_backbone, fixture_to_device, load_config


def config_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_gate(settings):
    path = settings.get('cache_gate')
    if not path or not Path(path).is_file():
        raise ValueError('Run worldttt check-cache first and set cache_gate to its passing JSON report')
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if not report.get('passed') or report['base_revision'] != BASE_REVISION:
        raise ValueError('Cached backbone equivalence gate has not passed')
    if report['base_checkpoint'] != settings['base_checkpoint'] or report['config_sha256'] != config_digest(settings['sana_config']):
        raise ValueError('Cache gate was measured with a different checkpoint/config')
    from diffusion.model.ops.fused_gdn import _resolve_launch_config

    setting, dot_precision, state_fp32, _ = _resolve_launch_config()
    precision = dict(setting=setting, dot_precision=dot_precision, state_fp32=state_fp32)
    if report.get('gdn_precision') != precision or report.get('gpu') != torch.cuda.get_device_name():
        raise ValueError('Cache gate was measured with a different GPU or GDN kernel precision')


def error_stats(a, b):
    diff = a.float() - b.float()
    return dict(max_abs=float(diff.abs().max()), relative_rms=float(diff.square().mean().sqrt() /
                                        a.float().square().mean().sqrt().clamp_min(1e-8)))


class LayerTrace:
    """Sample matching positions at block and attention branch boundaries."""

    def __init__(self, model, total_frames, current_frames, max_tokens_per_frame=64):
        self.values = {}
        self.handles = []
        self.gate_overrides = []
        self.total_frames = total_frames
        self.current_frames = current_frames
        self.max_tokens_per_frame = max_tokens_per_frame
        self.model = model

    def _record(self, key, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f'{key}: expected B,N,C tensor, got {type(value)}')
        batch, tokens, width = value.shape
        if tokens % self.total_frames:
            raise ValueError(f'{key}: {tokens} tokens cannot be split into {self.total_frames} frames')
        spatial = tokens // self.total_frames
        positions = torch.linspace(0, spatial - 1, min(spatial, self.max_tokens_per_frame),
                                   device=value.device).long()
        current = value.reshape(batch, self.total_frames, spatial, width)[:, -self.current_frames:]
        self.values[key] = current[:, :, positions].reshape(batch, -1, width).detach().float().cpu()

    def __enter__(self):
        for index, block in enumerate(self.model.blocks, 1):
            self.handles.append(block.register_forward_pre_hook(
                lambda module, args, i=index: self._record(f'{i}.input', args[0])))
            for name in ('attn', 'mlp'):
                module = getattr(block, name, None)
                if module is not None:
                    self.handles.append(module.register_forward_pre_hook(
                        lambda module, args, i=index, n=name: self._record(f'{i}.{n}.input', args[0])))
                    self.handles.append(module.register_forward_hook(
                        lambda module, args, output, i=index, n=name: self._record(f'{i}.{n}', output)))
            attn = getattr(block, 'attn', None)
            if attn is not None and hasattr(attn, 'out_proj_cam'):
                self.handles.append(attn.out_proj_cam.register_forward_hook(
                    lambda module, args, output, i=index: self._record(f'{i}.cam', output)))
            if attn is not None and hasattr(attn, '_apply_output_gate'):
                old_gate = attn._apply_output_gate
                old_override = attn.__dict__.get('_apply_output_gate')
                self.gate_overrides.append((attn, old_override))

                def record_combined(out, gate_x, i=index, gate=old_gate):
                    self._record(f'{i}.combined', out)
                    return gate(out, gate_x)

                attn._apply_output_gate = record_combined
            self.handles.append(block.register_forward_hook(
                lambda module, args, output, i=index: self._record(f'{i}.block', output)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for attn, old_override in self.gate_overrides:
            if old_override is None:
                del attn._apply_output_gate
            else:
                attn._apply_output_gate = old_override
        self.gate_overrides.clear()
        for index in range(1, len(self.model.blocks) + 1):
            combined, cam = self.values.get(f'{index}.combined'), self.values.get(f'{index}.cam')
            if combined is not None and cam is not None:
                self.values[f'{index}.main'] = combined - cam


def check_cache(settings, fixture_path, output, diagnose=False):
    from diffusion.model.ops.fused_gdn import _resolve_launch_config

    fixture = torch.load(fixture_path, map_location='cpu', weights_only=True)
    kernel_precision, dot_precision, state_fp32, _ = _resolve_launch_config()
    # The default GDN kernel uses BF16 tensor-core dots even for FP32 inputs.
    # Its independent full-prefix and cached reductions differ at that precision.
    fp32_tolerance = 5e-3 if dot_precision == 0 else 2e-4
    rows = []
    for dtype, tolerance in ((torch.float32, fp32_tolerance), (torch.bfloat16, 3e-2)):
        tensors = fixture_to_device(fixture, 'cuda', dtype)
        z, text, mask = (tensors[k] for k in ('latent', 'text', 'mask'))
        if z.shape[0] != 1 or z.shape[2] != 10:
            raise ValueError('Fixture must contain one 10-frame latent episode')
        bounds = [0, 4, 7, 10]
        refs = []
        reference_traces = []
        original = build_backbone(load_config(settings['sana_config'], cached=False),
                                  settings['base_checkpoint'], 'cuda', dtype)
        with torch.no_grad():
            for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
                times = torch.zeros(1, 1, end, device='cuda')
                times[:, :, start:] = 500
                trace = LayerTrace(original, end, end - start) if diagnose and dtype == torch.float32 else nullcontext()
                with trace:
                    output_ref = original(z[:, :, :end], times, text, mask=mask,
                        camera_conditions=tensors['camera'][:, :end], chunk_plucker=tensors['plucker'][:, :, :end],
                        chunk_index=bounds[:i + 1])
                refs.append(output_ref[:, :, start:end].cpu())
                reference_traces.append(trace.values if isinstance(trace, LayerTrace) else None)
        del original
        gc.collect()
        torch.cuda.empty_cache()
        cached = build_backbone(load_config(settings['sana_config']), settings['base_checkpoint'], 'cuda', dtype)
        caches, accumulate = native_cache_manager(cached)
        first_chunk_error = None
        with torch.no_grad():
            for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
                history = clone_cache(accumulate(i))
                kwargs = dict(y=text, mask=mask, camera_conditions=tensors['camera'][:, start:end],
                              chunk_plucker=tensors['plucker'][:, :, start:end], start_f=start, end_f=end)
                times = z.new_full((1, 1, end - start), 500.)
                trace = LayerTrace(cached, end - start, end - start) if reference_traces[i] is not None else nullcontext()
                with trace:
                    actual, _ = cached(z[:, :, start:end], times, kv_cache=clone_cache(history), save_kv_cache=False, **kwargs)
                stats = error_stats(refs[i], actual.cpu())
                # Attaching off must preserve the native cached output AND cache.
                WorldTTTController(cached, TTTConfig(mode='off'))
                off, off_cache = cached(z[:, :, start:end], times, kv_cache=clone_cache(history), save_kv_cache=False, **kwargs)
                assert torch.equal(actual, off), 'off changed native cached output'
                def same(a, b):
                    if isinstance(a, torch.Tensor):
                        return isinstance(b, torch.Tensor) and torch.equal(a, b)
                    if isinstance(a, (tuple, list)):
                        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
                    return a == b
                assert same(history, off_cache), 'Read-only forward modified cached history'
                row = dict(dtype=str(dtype), chunk=i, **stats,
                           passed=stats['relative_rms'] <= tolerance, tolerance=tolerance)
                if dtype == torch.float32:
                    if first_chunk_error is None:
                        first_chunk_error = stats['relative_rms']
                    else:
                        # Chunk 0 has no history; bound the extra error after
                        # cache accumulation relative to that kernel baseline.
                        row['history_limit'] = max(2e-4, 1.25 * first_chunk_error)
                        row['passed'] = row['passed'] and stats['relative_rms'] <= row['history_limit']
                if isinstance(trace, LayerTrace):
                    row['stages'] = [dict(stage=stage, **error_stats(reference, trace.values[stage]))
                                     for stage, reference in reference_traces[i].items()
                                     if stage in trace.values]
                    row['unmatched_stages'] = sorted(set(reference_traces[i]) ^ set(trace.values))
                rows.append(row)
                _, saved = cached(z[:, :, start:end], z.new_zeros(1), kv_cache=history, save_kv_cache=True, **kwargs)
                caches[i] = clone_cache(saved)
        del cached
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(passed=all(r['passed'] for r in rows), base_revision=BASE_REVISION,
                  base_checkpoint=settings['base_checkpoint'], config_sha256=config_digest(settings['sana_config']),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  gdn_precision=dict(setting=kernel_precision, dot_precision=dot_precision, state_fp32=state_fp32),
                  rows=rows)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError(f'Cache conversion failed: inspect {output}; do not start TTT experiments')
