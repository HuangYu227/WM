"""P0: zero SAP-Bind gate preserves SANA output and cache exactly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from worldttt.sap_ttt.check_gate import tree_max_error


def check(settings, fixture, output):
    from worldttt.check_cache import require_gate
    from worldttt.episode import native_cache_manager
    from worldttt.runtime import clone_cache
    from worldttt.sana import build_backbone, fixture_to_device, load_config
    from .config import BindingConfig
    from .runtime import BindingController

    if not torch.cuda.is_available():
        raise RuntimeError('SAP-Bind zero-gate check requires CUDA')
    require_gate(settings)
    config = load_config(settings['sana_config'])
    model = build_backbone(config, settings['base_checkpoint'], 'cuda', torch.bfloat16)
    sample = fixture_to_device(torch.load(fixture, map_location='cpu', weights_only=True),
                               'cuda', torch.bfloat16)
    if sample['latent'].shape[0] != 1 or sample['latent'].shape[2] < 4:
        raise ValueError('Need a SANA fixture with at least four latent frames')
    caches, accumulate = native_cache_manager(model); cache = clone_cache(accumulate(0))
    latent = sample['latent'][:, :, :4]; times = torch.full((1, 1, 4), 500., device='cuda')
    kwargs = dict(y=sample['text'], mask=sample['mask'], camera_conditions=sample['camera'][:, :4],
        chunk_plucker=sample['plucker'][:, :, :4], start_f=0, end_f=4,
        frame_index=torch.arange(4, device='cuda'), data_info={}, save_kv_cache=True)
    with torch.no_grad():
        baseline, baseline_cache = model(latent, times, kv_cache=clone_cache(cache), **kwargs)
    controller = BindingController(model, BindingConfig(**dict(settings['sap_binding'], mode='frozen')))
    controller.reset_episode('p0', 1)
    if any(float(module.gate.detach()) != 0 for module in controller.modules.values()):
        raise ValueError('SAP-Bind output gate must initialize to zero')
    with torch.no_grad():
        candidate, candidate_cache = model(latent, times, kv_cache=clone_cache(cache),
            binding_context=controller.context(times, latent=latent), **kwargs)
    report = dict(base_checkpoint=settings['base_checkpoint'], gpu=torch.cuda.get_device_name(),
        dtype=str(latent.dtype), output_max_abs=tree_max_error(baseline, candidate),
        cache_max_abs=tree_max_error(baseline_cache, candidate_cache))
    report['passed'] = report['output_max_abs'] == 0 and report['cache_max_abs'] == 0
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError(f'SAP-Bind zero-gate equivalence failed; inspect {path}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True); parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(check(json.loads(Path(args.settings).read_text()),
                                                      args.fixture, args.output), indent=2))


if __name__ == '__main__':
    main()
