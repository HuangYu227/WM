"""P0 numerical gate: zero SAP gate leaves cached SANA output/cache unchanged."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def tree_max_error(a, b):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape:
            raise ValueError('Cache tensor shape/type mismatch')
        return float((a.float() - b.float()).abs().max()) if a.numel() else 0.
    if isinstance(a, (list, tuple)):
        if not isinstance(b, type(a)) or len(a) != len(b):
            raise ValueError('Cache sequence structure mismatch')
        return max((tree_max_error(x, y) for x, y in zip(a, b)), default=0.)
    if a is None and b is None:
        return 0.
    if type(a) in (int, float) and type(b) is type(a):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Nonfinite cache type flag')
        return abs(float(a) - float(b))
    raise ValueError('Unsupported cache leaf')


def check(settings, fixture, output):
    from worldttt.check_cache import require_gate
    from worldttt.episode import native_cache_manager
    from worldttt.runtime import clone_cache
    from worldttt.sana import build_backbone, fixture_to_device, load_config
    from .runtime import SapConfig, SapController

    if not torch.cuda.is_available():
        raise RuntimeError('SAP zero-gate numerical check requires CUDA')
    require_gate(settings)
    config = load_config(settings['sana_config'])
    model = build_backbone(config, settings['base_checkpoint'], 'cuda', torch.bfloat16)
    sample = torch.load(fixture, map_location='cpu', weights_only=True)
    sample = fixture_to_device(sample, 'cuda', torch.bfloat16)
    if sample['latent'].shape[0] != 1 or sample['latent'].shape[2] < 4:
        raise ValueError('Need an ordinary SANA fixture with at least four latent frames')
    caches, accumulate = native_cache_manager(model)
    cache = clone_cache(accumulate(0))
    times = torch.full((1, 1, 4), 500., device='cuda')
    kwargs = dict(y=sample['text'], mask=sample['mask'],
                  camera_conditions=sample['camera'][:, :4],
                  chunk_plucker=sample['plucker'][:, :, :4],
                  start_f=0, end_f=4, frame_index=torch.arange(4, device='cuda'),
                  data_info={}, save_kv_cache=True)
    with torch.no_grad():
        baseline, baseline_cache = model(sample['latent'][:, :, :4], times,
                                         kv_cache=clone_cache(cache), **kwargs)
    controller = SapController(model, SapConfig(**dict(settings['sap'], mode='frozen')))
    controller.reset_episode('p0', 1)
    assert all(float(module.gate.detach()) == 0 for module in controller.modules.values())
    with torch.no_grad():
        candidate, candidate_cache = model(sample['latent'][:, :, :4], times,
            kv_cache=clone_cache(cache), sap_context=controller.context(times), **kwargs)
    report = {'base_checkpoint': settings['base_checkpoint'], 'gpu': torch.cuda.get_device_name(),
              'dtype': str(sample['latent'].dtype),
              'output_max_abs': tree_max_error(baseline, candidate),
              'cache_max_abs': tree_max_error(baseline_cache, candidate_cache)}
    report['passed'] = report['output_max_abs'] == 0 and report['cache_max_abs'] == 0
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError(f'SAP zero-gate equivalence failed; inspect {path}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
    print(json.dumps(check(settings, args.fixture, args.output), indent=2))


if __name__ == '__main__':
    main()
