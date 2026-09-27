"""Four-chunk Flow episode using causal SAP-Bind commits."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from worldttt.episode import flow_schedule
from worldttt.runtime import clone_cache


BOUNDS = (0, 4, 7, 10, 13)


def native_four_chunk_cache(model):
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
    holder = SimpleNamespace(num_model_blocks=len(model.blocks), num_cached_blocks=2,
                             sink_token=False, _chunk_indices=list(BOUNDS))
    caches = SelfForcingFlowEulerCamCtrl._initialize_kv_cache(holder, 4)
    return caches, lambda i: SelfForcingFlowEulerCamCtrl._accumulate_softmax_kv_cache(holder, caches, i)[0]


class BindingEpisodeModel(nn.Module):
    def __init__(self, backbone, controller, *, steps=50, shift=9.8,
                 cache_factory=native_four_chunk_cache, schedule_factory=flow_schedule):
        super().__init__()
        self.backbone, self.controller = backbone, controller
        self.steps, self.shift = steps, shift
        self.cache_factory, self.schedule_factory = cache_factory, schedule_factory

    def forward(self, latent, camera, plucker, text, mask, episode_id, *, generated=False,
                seed=0, meta_grad=True):
        if latent.shape[0] != 1 or latent.shape[2] != 13:
            raise ValueError('SAP-Bind episode requires one 4+3+3+3 latent sequence')
        ctl = self.controller; ctl.reset_episode(episode_id, 1, training=meta_grad)
        caches, accumulate = self.cache_factory(self.backbone)
        generator = torch.Generator(device=latent.device).manual_seed(seed)
        for chunk in range(4):
            start, end = BOUNDS[chunk:chunk + 2]
            clean = latent[:, :, start:end]
            cache = clone_cache(accumulate(chunk))
            kwargs = dict(y=text, mask=mask, camera_conditions=camera[:, start:end],
                chunk_plucker=plucker[:, :, start:end], start_f=start, end_f=end,
                frame_index=torch.arange(start, end, device=latent.device), data_info={})
            if chunk == 3:
                query_generator = torch.Generator(device=latent.device).manual_seed(seed + 10_000_019)
                u = torch.rand((), generator=query_generator, device=latent.device)
                sigma = self.shift * u / (1 + (self.shift - 1) * u)
                noise = torch.randn(clean.shape, generator=query_generator, device=clean.device, dtype=clean.dtype)
                noisy = (1 - sigma) * clean + sigma * noise
                times = (sigma * 1000).expand(1, 1, end - start)
                context = ctl.context(times, latent=noisy, training=meta_grad)
                prediction, _ = self.backbone(noisy, times, kv_cache=cache, save_kv_cache=False,
                                               binding_context=context, **kwargs)
                flow = (prediction.float() - (noise - clean).float()).square().mean()
                return flow, dict(flow_mse=float(flow.detach()), generated_history=generated)
            if generated:
                scheduler = self.schedule_factory(self.steps, self.shift, latent.device)
                with torch.no_grad():
                    current = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
                    if chunk == 0: current[:, :, 0] = clean[:, :, 0]
                    for tau in scheduler.timesteps:
                        times = tau.expand(1, 1, end - start).clone()
                        if chunk == 0: times[:, :, 0] = 0
                        context = ctl.context(times, latent=current)
                        prediction, _ = self.backbone(current, times, kv_cache=clone_cache(cache),
                            save_kv_cache=False, binding_context=context, **kwargs)
                        token_times = times[:, 0, :, None].expand(
                            1, end - start, clean.shape[3] * clean.shape[4]).reshape(1, -1)
                        flat = scheduler.step(-prediction.flatten(2).transpose(1, 2), tau,
                            current.flatten(2).transpose(1, 2), per_token_timesteps=token_times,
                            return_dict=False)[0]
                        current = flat.transpose(1, 2).reshape_as(clean).to(clean.dtype)
                        if chunk == 0: current[:, :, 0] = clean[:, :, 0]
                    clean = current.detach()
            with torch.no_grad():
                zero = clean.new_zeros(1)
                source = (('reference_plus_generated' if chunk == 0 else 'model_generated')
                          if generated else ('reference_plus_ground_truth' if chunk == 0 else 'ground_truth'))
                context = ctl.context(zero, latent=clean, collect=True, chunk=chunk,
                                      training=meta_grad, source=source)
                _, updated = self.backbone(clean, zero, kv_cache=cache, save_kv_cache=True,
                                           binding_context=context, **kwargs)
                caches[chunk] = clone_cache(updated)
            result = ctl.commit(context, chunk, training=meta_grad)
            if not result['committed']:
                raise FloatingPointError(f'SAP-Bind support update rejected: {result}')
        raise AssertionError('No A-prime query')
