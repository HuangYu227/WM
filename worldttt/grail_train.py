"""Canonical GRAIL outer training: support association + read-only future flow."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .grail_native import GrailNativeController, attach_grail_native, load_grail_adapter
from .grail_network import GrailNetworkConfig
from .associative_ttt import AssociativeTTTConfig
from .runtime import clone_cache


def chunk_boundaries(frames, chunk_size=3):
    boundaries = [0, min(chunk_size + 1, frames)]
    while boundaries[-1] < frames:
        boundaries.append(min(boundaries[-1] + chunk_size, frames))
    if len(boundaries) < 3:
        raise ValueError('need at least one history chunk and one held-out query')
    return boundaries


class GrailEpisodeModel(nn.Module):
    def __init__(self, backbone, *, steps=4, flow_shift=9.8, association_weight=1., detach_every=0):
        super().__init__()
        self.backbone = backbone
        self.steps, self.flow_shift = steps, flow_shift
        self.association_weight, self.detach_every = association_weight, detach_every

    @property
    def controller(self):
        return self.backbone.worldttt_grail_controller

    def forward(self, latent, camera, text, mask, episode_id, plucker=None, generated=False, seed=0):
        from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
        from diffusers import FlowMatchEulerDiscreteScheduler
        ctl, model = self.controller, self.backbone
        if latent.ndim != 5 or camera.shape[:2] != (latent.shape[0], latent.shape[2]):
            raise ValueError('trajectory needs latent [B,C,T,H,W] and camera [B,T,20]')
        ctl.reset_episode(episode_id, latent.shape[0])
        bounds = chunk_boundaries(latent.shape[2])
        chunks = len(bounds) - 1
        holder = SimpleNamespace(num_model_blocks=len(model.blocks), num_cached_blocks=2,
                                 sink_token=False, _chunk_indices=bounds)
        caches = SelfForcingFlowEulerCamCtrl._initialize_kv_cache(holder, chunks)
        rng = torch.Generator(device=latent.device).manual_seed(seed)
        associations = []
        for chunk in range(chunks):
            start, end = bounds[chunk:chunk + 2]
            cache = clone_cache(SelfForcingFlowEulerCamCtrl._accumulate_softmax_kv_cache(holder, caches, chunk)[0])
            clean = latent[:, :, start:end]
            kwargs = dict(y=text, mask=mask, camera_conditions=camera[:, start:end], start_f=start, end_f=end,
                          frame_index=torch.arange(start, end, device=latent.device), data_info={})
            if plucker is not None:
                kwargs['chunk_plucker'] = plucker[:, :, start:end]
            if chunk == chunks - 1:
                # Separate RNG makes the held-out query identical across history modes.
                query_rng = torch.Generator(device=latent.device).manual_seed(seed + 10_000_019)
                u = torch.rand((), device=latent.device, generator=query_rng)
                sigma = self.flow_shift * u / (1 + (self.flow_shift - 1) * u)
                noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=query_rng)
                noisy = (1 - sigma) * clean + sigma * noise
                times = (1000 * sigma).expand(clean.shape[0], 1, end - start)
                context = ctl.context('frozen', sigma=times / 1000., chunk_id=chunk)
                cursor = ctl.state.last_committed_chunk.clone()
                prediction, _ = model(noisy, times, kv_cache=cache, save_kv_cache=False, grail_context=context, **kwargs)
                if context.observations or not torch.equal(cursor, ctl.state.last_committed_chunk):
                    raise RuntimeError('held-out future query must be read-only')
                future = (prediction.float() - (noise - clean).float()).square().mean()
                association = torch.stack(associations).mean()
                return dict(loss=future + self.association_weight * association, future=future,
                            association=association, query_context=context, support_chunks=chunk)
            real_frames = tuple(range(end - start))
            if generated:
                real_frames = (0,) if chunk == 0 else ()
                with torch.no_grad():
                    current = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=rng)
                    if chunk == 0:
                        current[:, :, 0] = clean[:, :, 0]
                    schedule = FlowMatchEulerDiscreteScheduler(shift=self.flow_shift)
                    schedule.set_timesteps(self.steps, device=latent.device)
                    for tau in schedule.timesteps:
                        times = tau.expand(clean.shape[0], 1, end - start).clone()
                        if chunk == 0:
                            times[:, :, 0] = 0
                        context = ctl.context('frozen', sigma=times / 1000.)
                        prediction, _ = model(current, times, kv_cache=clone_cache(cache), save_kv_cache=False,
                                              grail_context=context, **kwargs)
                        token_times = times[:, 0, :, None].expand(-1, -1, clean.shape[3] * clean.shape[4]).flatten(1)
                        current = schedule.step(-prediction.flatten(2).transpose(1, 2), tau,
                                                current.flatten(2).transpose(1, 2), per_token_timesteps=token_times,
                                                return_dict=False)[0].transpose(1, 2).reshape_as(clean).to(clean.dtype)
                        if chunk == 0:
                            current[:, :, 0] = clean[:, :, 0]
                    clean = current.detach()
            # Frozen backbone, grad-enabled writer/readers, functional Ridge commit.
            context = ctl.context('online', collect=True, chunk_id=chunk, real_frame_indices=real_frames)
            _, updated_cache = model(clean, clean.new_zeros(clean.shape[0]), kv_cache=cache, save_kv_cache=True,
                                     grail_context=context, **kwargs)
            associations.append(ctl.ledger.association_loss(context.observations[0]))
            if not ctl.commit_clean(context, chunk)['committed']:
                raise FloatingPointError('GRAIL training write rejected')
            # The outer temporal gradient is through the ledger, not native KV.
            caches[chunk] = clone_cache(updated_cache)
            if self.detach_every and (chunk + 1) % self.detach_every == 0 and chunk + 1 < chunks - 1:
                ctl.state = ctl.state.detach()
        raise AssertionError('missing future query')


def train_step(module, optimizer, batch, *, generated=False, seed=0):
    optimizer.zero_grad(set_to_none=True)
    result = module(**batch, generated=generated, seed=seed)
    if not torch.isfinite(result['loss']):
        raise FloatingPointError('nonfinite GRAIL loss')
    result['loss'].backward()
    parameters = [p for p in module.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    return {name: float(result[name].detach()) for name in ('loss', 'future', 'association')} | {'gradient_norm': float(norm)}


def train(settings, output, adapter=None):
    from .data import EpisodeDataset
    from .provenance import file_sha256
    from .sana import load_config, make_fixture, make_pipeline
    if not torch.cuda.is_available():
        raise RuntimeError('full-checkpoint training needs CUDA; use grail_validate for CPU native smoke')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('this trainer currently supports one GPU; do not launch it with multi-rank torchrun')
    device = torch.device('cuda')
    torch.manual_seed(settings.get('seed', 3407))
    pipe = make_pipeline(load_config(settings['sana_config']), settings['base_checkpoint'], device, training=True)
    if adapter:
        ctl, _ = load_grail_adapter(pipe.model, adapter, base_checkpoint_hash=file_sha256(settings['base_checkpoint']))
    else:
        config = dict(geometry_dim=30, geometry_metric='ray_point',
                      coordinate_convention='episode_anchor_pose_uv_plucker_depth_v2') | settings.get('ledger', {})
        ctl = GrailNativeController(pipe.model.blocks[0].hidden_size, AssociativeTTTConfig(**config),
                                    network=GrailNetworkConfig(**settings.get('network', {})), mode='online')
        ctl.base_checkpoint_hash = file_sha256(settings['base_checkpoint'])
        attach_grail_native(pipe.model, ctl)
    module = GrailEpisodeModel(pipe.model, steps=settings.get('steps', 4),
                               flow_shift=pipe.config.scheduler.inference_flow_shift,
                               association_weight=settings.get('association_weight', 1.),
                               detach_every=settings.get('detach_every', 0))
    optimizer = torch.optim.AdamW([p for p in ctl.parameters() if p.requires_grad], lr=settings.get('outer_lr', 1e-4))
    output = Path(output)
    if (output / 'last.pt').exists():
        raise FileExistsError('use a fresh output directory; --adapter is a warm start')
    output.mkdir(parents=True, exist_ok=True)
    data = EpisodeDataset(settings['data'], settings['manifest'], 'train', frames=settings.get('frames', 121))
    validation = EpisodeDataset(settings['data'], settings['manifest'], 'val', frames=settings.get('frames', 121))
    best = float('inf')
    for step in range(settings.get('max_steps', 1000)):
        fixture = make_fixture(data[step % len(data)], pipe, device)
        curriculum = settings.get('curriculum', [[0, 4], [250, 12], [500, 24], [750, 40]])
        active_chunks = max(chunks for threshold, chunks in curriculum if step >= threshold)
        frames = min(fixture['latent'].shape[2], active_chunks * 3 + 1)
        fixture['latent'], fixture['camera'] = fixture['latent'][:, :, :frames], fixture['camera'][:, :frames]
        fixture['plucker'] = fixture['plucker'][:, :, :frames]
        generated = step >= settings.get('real_prefix_steps', 250) and step % 2 == 1
        row = train_step(module, optimizer, fixture, generated=generated, seed=settings.get('seed', 3407) + step)
        row.update(step=step + 1, generated=generated, frames=frames)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        if (step + 1) % settings.get('save_every', 100) == 0 or step + 1 == settings.get('max_steps', 1000):
            with torch.no_grad():
                validation_losses = []
                for index in range(min(len(validation), settings.get('val_max_samples', 4))):
                    heldout = make_fixture(validation[index], pipe, device)
                    metrics = module(**heldout, generated=False, seed=12345 + index)
                    validation_losses.append(float(metrics['future']))
            validation_loss = sum(validation_losses) / len(validation_losses)
            extra = dict(step=step + 1, settings=settings, validation_future=validation_loss,
                         flow_trained=True, optimizer=optimizer.state_dict())
            ctl.save_checkpoint(output / 'last.pt', **extra)
            if validation_loss < best:
                best = validation_loss
                ctl.save_checkpoint(output / 'best.pt', **extra)
