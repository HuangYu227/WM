"""Four-chunk A/B/C/A' outer objective with causal support commits."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from worldttt.episode import flow_schedule
from worldttt.runtime import clone_cache

from .probe import delayed_supervision_loss


BOUNDS = (0, 4, 7, 10, 13)


def native_four_chunk_cache(model):
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
    holder = SimpleNamespace(num_model_blocks=len(model.blocks), num_cached_blocks=2,
                             sink_token=False, _chunk_indices=list(BOUNDS))
    caches = SelfForcingFlowEulerCamCtrl._initialize_kv_cache(holder, 4)
    def accumulate(i):
        return SelfForcingFlowEulerCamCtrl._accumulate_softmax_kv_cache(holder, caches, i)[0]
    return caches, accumulate


class SapEpisodeModel(nn.Module):
    def __init__(self, backbone, controller, *, steps=50, shift=9.8, lambda_delayed=.1,
                 cache_factory=native_four_chunk_cache, schedule_factory=flow_schedule, lambda_exact=0.):
        super().__init__()
        self.backbone, self.controller = backbone, controller
        self.steps, self.shift, self.lambda_delayed = steps, shift, lambda_delayed
        self.cache_factory, self.schedule_factory = cache_factory, schedule_factory
        self.lambda_exact = lambda_exact

    def forward(self, latent, camera, plucker, text, mask, episode_id, *, generated=False,
                seed=0, meta_grad=True, supervision=None):
        if latent.shape[0] != 1 or latent.shape[2] != 13:
            raise ValueError('SAP episode requires one 4+3+3+3 latent sequence')
        ctl = self.controller
        ctl.reset_episode(episode_id, 1, training=meta_grad)
        caches, accumulate = self.cache_factory(self.backbone)
        rng = torch.Generator(device=latent.device).manual_seed(seed)
        first_context = None
        for chunk in range(4):
            start, end = BOUNDS[chunk:chunk + 2]
            clean = latent[:, :, start:end]
            cache = clone_cache(accumulate(chunk))
            kwargs = dict(y=text, mask=mask, camera_conditions=camera[:, start:end],
                          chunk_plucker=plucker[:, :, start:end], start_f=start, end_f=end,
                          frame_index=torch.arange(start, end, device=latent.device), data_info={})
            if chunk == 3:
                query_rng = torch.Generator(device=latent.device).manual_seed(seed + 10_000_019)
                u = torch.rand((), generator=query_rng, device=latent.device)
                sigma = self.shift * u / (1 + (self.shift - 1) * u)
                noise = torch.randn(clean.shape, generator=query_rng, device=clean.device, dtype=clean.dtype)
                noisy = (1 - sigma) * clean + sigma * noise
                times = (sigma * 1000).expand(1, 1, end - start)
                use_labels = supervision is not None and not generated
                query_context = ctl.context(times, training=meta_grad, record_query=use_labels)
                prediction, _ = self.backbone(noisy, times, kv_cache=cache, save_kv_cache=False,
                                              sap_context=query_context, **kwargs)
                flow = (prediction.float() - (noise - clean).float()).square().mean()
                total = flow
                metrics = {'flow_mse': float(flow.detach()), 'generated_history': generated}
                if use_labels:
                    labels = {name: tensor.to(prediction.device) for name, tensor in supervision.items()}
                    auxiliaries, diagnostics = [], {}
                    for idx in ctl.config.layers:
                        old_k, old_v = first_context.features[idx]
                        q = query_context.query_addresses[idx]
                        readout = ctl.modules[idx].memory.read(ctl.state[idx], q)
                        auxiliary, diagnostic = delayed_supervision_loss(
                            q, old_k, old_v, readout, first_context.support_indices[idx],
                            labels['positive'], labels['valid'],
                            labels['support_instance'][:4], labels['query_instance'],
                            exact_weight=self.lambda_exact)
                        auxiliaries.append(auxiliary)
                        diagnostics[str(idx)] = diagnostic
                    total = total + self.lambda_delayed * torch.stack(auxiliaries).mean()
                    if len(diagnostics) == 1:
                        metrics.update(next(iter(diagnostics.values())))
                    else:
                        metrics['per_layer'] = diagnostics
                        metrics['matched_queries'] = min(row['matched_queries'] for row in diagnostics.values())
                return total, metrics

            if generated:
                scheduler = self.schedule_factory(self.steps, self.shift, latent.device)
                with torch.no_grad():
                    current = torch.randn(clean.shape, generator=rng, device=clean.device, dtype=clean.dtype)
                    if chunk == 0:
                        current[:, :, 0] = clean[:, :, 0]
                    for tau in scheduler.timesteps:
                        times = tau.expand(1, 1, end - start).clone()
                        if chunk == 0:
                            times[:, :, 0] = 0
                        prediction, _ = self.backbone(current, times, kv_cache=clone_cache(cache),
                            save_kv_cache=False, sap_context=ctl.context(times), **kwargs)
                        token_times = times[:, 0, :, None].expand(
                            1, end - start, clean.shape[3] * clean.shape[4]).reshape(1, -1)
                        flat = scheduler.step(-prediction.flatten(2).transpose(1, 2), tau,
                            current.flatten(2).transpose(1, 2), per_token_timesteps=token_times,
                            return_dict=False)[0]
                        current = flat.transpose(1, 2).reshape_as(clean).to(clean.dtype)
                        if chunk == 0:
                            current[:, :, 0] = clean[:, :, 0]
                    clean = current.detach()
            with torch.no_grad():
                zero = clean.new_zeros(1)
                provenance = ('reference_plus_generated' if chunk == 0 else 'model_generated') if generated else (
                    'reference_plus_ground_truth' if chunk == 0 else 'ground_truth')
                context = ctl.context(zero, collect=True, chunk=chunk,
                                      training=meta_grad, source=provenance)
                _, updated = self.backbone(clean, zero, kv_cache=cache, save_kv_cache=True,
                                           sap_context=context, **kwargs)
                caches[chunk] = clone_cache(updated)
            result = ctl.commit(context, chunk, training=meta_grad)
            if not result['committed']:
                raise FloatingPointError(f'SAP support update rejected: {result}')
            if chunk == 0:
                first_context = context
        raise AssertionError('No A-prime query')
