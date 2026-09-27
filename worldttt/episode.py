"""Two support chunks, one query; fast adaptation stays differentiable."""
from types import SimpleNamespace

import torch
from torch import nn

from .runtime import clone_cache


def flow_schedule(steps, shift, device):
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    scheduler.set_timesteps(steps, device=device)
    return scheduler


def native_cache_manager(model):
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
    holder = SimpleNamespace(num_model_blocks=len(model.blocks), num_cached_blocks=2,
                             sink_token=False, _chunk_indices=[0, 4, 7, 10])
    caches = SelfForcingFlowEulerCamCtrl._initialize_kv_cache(holder, 3)
    def accumulate(i):
        return SelfForcingFlowEulerCamCtrl._accumulate_softmax_kv_cache(holder, caches, i)[0]
    return caches, accumulate


class EpisodeModel(nn.Module):
    def __init__(self, model, controller, steps=50, shift=9.8, cache_factory=native_cache_manager,
                 schedule_factory=flow_schedule):
        super().__init__()
        self.backbone = model
        self.controller = controller
        self.steps, self.shift = steps, shift
        self.cache_factory, self.schedule_factory = cache_factory, schedule_factory

    def forward(self, latent, camera, plucker, text, mask, episode_id, generated=False, seed=0,
                meta_grad=True):
        if latent.shape[0] != 1 or latent.shape[2] != 10:
            raise ValueError('v1 requires one 4+3+3 episode per GPU')
        ctl = self.controller
        ctl.reset_episode(episode_id, 1, training=meta_grad)
        caches, accumulate = self.cache_factory(self.backbone)
        rng = torch.Generator(device=latent.device).manual_seed(seed)
        bounds = [0, 4, 7, 10]
        for chunk in range(3):
            start, end = bounds[chunk:chunk + 2]
            clean = latent[:, :, start:end]
            cache = clone_cache(accumulate(chunk))
            ctl.begin_chunk(chunk, 1, training=meta_grad)
            kwargs = dict(y=text, mask=mask, camera_conditions=camera[:, start:end],
                          chunk_plucker=plucker[:, :, start:end], start_f=start, end_f=end,
                          frame_index=torch.arange(start, end, device=latent.device), data_info={})
            if chunk == 2:
                # Same shifted flow distribution as the deployment schedule,
                # sampled continuously for the outer training objective.
                # Keep the held-out query identical across real/generated support modes.
                query_rng = torch.Generator(device=latent.device).manual_seed(seed + 10_000_019)
                u = torch.rand((), generator=query_rng, device=latent.device)
                sigma = self.shift * u / (1 + (self.shift - 1) * u)
                noise = torch.randn(clean.shape, generator=query_rng, device=clean.device, dtype=clean.dtype)
                noisy = (1 - sigma) * clean + sigma * noise
                times = (sigma * 1000).expand(1, 1, end - start)
                # Only query carries backbone gradients; never wrap it in no_grad.
                prediction, _ = self.backbone(noisy, times, kv_cache=cache, save_kv_cache=False,
                                               worldttt_context=ctl.context(times), **kwargs)
                return (prediction.float() - (noise - clean).float()).square().mean()

            scheduler = self.schedule_factory(self.steps, self.shift, latent.device)
            with torch.no_grad():
                if generated:
                    current = torch.randn(clean.shape, generator=rng, device=clean.device, dtype=clean.dtype)
                    if chunk == 0:
                        current[:, :, 0] = clean[:, :, 0]
                    for tau in scheduler.timesteps:
                        times = tau.expand(1, 1, end - start).clone()
                        if chunk == 0:
                            times[:, :, 0] = 0
                        prediction, _ = self.backbone(current, times, kv_cache=clone_cache(cache),
                            save_kv_cache=False, worldttt_context=ctl.context(times), **kwargs)
                        token_times = times[:, 0, :, None].expand(1, end - start, clean.shape[3] * clean.shape[4]).reshape(1, -1)
                        # Match SelfForcingFlowEulerCamCtrl's per-token convention.
                        flat = scheduler.step(-prediction.flatten(2).transpose(1, 2), tau,
                            current.flatten(2).transpose(1, 2), per_token_timesteps=token_times,
                            return_dict=False)[0]
                        current = flat.transpose(1, 2).reshape_as(clean).to(clean.dtype)
                        if chunk == 0:
                            current[:, :, 0] = clean[:, :, 0]
                    clean = current.detach()
                learning = ctl.config.mode in {'kv_ttt', 'noise_ttt'}
                if learning:
                    noisy_ctx = ctl.noisy_support(clean, scheduler.timesteps, self.backbone,
                        dict(**kwargs, kv_cache=clone_cache(cache), save_kv_cache=False), chunk,
                        conditioned=(0,) if chunk == 0 else ())
                times = clean.new_zeros(1)
                clean_ctx = ctl.context(times, collect=learning, chunk=chunk)
                _, updated = self.backbone(clean, times, kv_cache=cache, save_kv_cache=True,
                                           worldttt_context=clean_ctx, **kwargs)
                caches[chunk] = clone_cache(updated)
            if learning:
                ctl.adapt(noisy_ctx, clean_ctx, chunk, training=meta_grad)
        raise AssertionError('Missing query chunk')
