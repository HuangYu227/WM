"""Explicit read contexts and once-per-chunk adaptation (no forward-hook writes)."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, dataclass, field

import torch

from .memory import FastMemory, Features, TTTConfig, WorldTTTState, adapt_after_chunk

BASE_REVISION = 'f9178744c096dcf2a2ea773da183e341bcbeb044'


def clone_cache(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_cache(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_cache(v) for v in value)
    return value


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, (list, tuple)):
        return type(value)(move_tree(v, device) for v in value)
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    return value


def expand_condition(value, batch, frames, tokens, channels, device):
    if value is None:
        return torch.zeros(batch, tokens, channels, device=device)
    value = value.to(device).float()
    if channels == 1:
        value = value.reshape(batch, -1, 1)
    else:
        value = value.reshape(batch, -1, channels)
    if value.shape[1] == 1:
        value = value.expand(-1, frames, -1)
    if value.shape[1] != frames or tokens % frames:
        raise ValueError('Condition/frame/token mismatch')
    return value.repeat_interleave(tokens // frames, dim=1)


@dataclass
class ReadContext:
    weights: dict
    sigma: torch.Tensor
    config: TTTConfig
    collect: bool = False
    chunk: int = 0
    features: dict = field(default_factory=dict)

    def apply(self, layer, x, m, camera, hw):
        memory = getattr(layer, 'worldttt_memory', None)
        if memory is None or self.config.mode == 'off':
            return m
        idx = layer.worldttt_layer_id
        b, n, _ = x.shape
        if self.weights[idx][0].shape[0] != b:
            raise ValueError('Episode/CFG batch size changed without reset')
        pose = expand_condition(camera, b, hw[0], n, 20, x.device)
        sigma = expand_condition(self.sigma, b, hw[0], n, 1, x.device)
        f = Features(x, m, pose, sigma)
        if self.collect:
            if torch.is_grad_enabled():
                raise RuntimeError('Feature capture must run outside checkpointed gradient forwards')
            # The same deterministic subset is used for clean and noisy passes.
            rng = torch.Generator().manual_seed(self.config.seed + 1009 * self.chunk + idx)
            ids = torch.randperm(n, generator=rng)[:min(n, 2 * self.config.support_tokens)].to(x.device)
            selected = f.select(ids).detach()
            # Conventional TTT baseline uses inherited raw projected K/V, before
            # GDN's shortconv/RoPE; it does not claim equivalence to a GDN write.
            qkv = layer.qkv(x[:, ids]).reshape(b, len(ids), 3, layer.heads * layer.dim)
            selected.k, selected.v = qkv[:, :, 1].detach().float(), qkv[:, :, 2].detach().float()
            self.features[idx] = selected
        read_features = f
        if self.config.mode == 'kv_ttt' or self.config.frozen_source == 'kv_ttt':
            # The same raw projected Q/K address space is used for reads/writes.
            q = layer.qkv(x).reshape(b, n, 3, layer.heads * layer.dim)[:, :, 0]
            read_features = Features(q, torch.zeros_like(m), pose, sigma)
        correction = memory(read_features, self.weights[idx])
        return m + correction.to(m.dtype)


class WorldTTTController:
    def __init__(self, model, config: TTTConfig, validate_blocks=True):
        self.model, self.config = model, config
        self.memories = {}
        self.state = None
        self.metrics = []
        self.base_checkpoint = None
        self.rollout_path = None
        self.resume_snapshot = None
        self.rollout_signature = None
        self.model.requires_grad_(False)
        if config.mode != 'off':
            for idx in config.layers:
                if idx > len(model.blocks):
                    raise ValueError(f'No block {idx}')
                attn = model.blocks[idx - 1].attn
                if validate_blocks and type(attn).__name__ != 'CachedChunkCausalGDNUCPESinglePathLiteLA':
                    raise ValueError(f'Block {idx} must be cached camera GDN, got {type(attn).__name__}')
                memory = FastMemory(attn.heads, attn.dim, config).to(next(attn.parameters()).device)
                attn.add_module('worldttt_memory', memory)
                attn.worldttt_layer_id = idx
                self.memories[idx] = memory
        object.__setattr__(model, 'worldttt_controller', self)

    def reset_episode(self, episode, batch, training=False):
        self.state = WorldTTTState(str(episode), {i: m.initial_weights(batch, training) for i, m in self.memories.items()},
                                   seed=self.config.seed)
        self.metrics = []
        return self.state

    def context(self, timesteps, collect=False, chunk=0):
        if self.config.mode == 'off':
            return None
        if self.state is None:
            raise RuntimeError('Call reset_episode before reading memory')
        return ReadContext(dict(self.state.weights), timesteps.float() / 1000.0, self.config, collect, chunk)

    def adapt(self, noisy, clean, chunk, training=False):
        if self.config.mode == 'off':
            return {'chunk': chunk, 'committed': False}
        if set(noisy.features) != set(self.memories) or set(clean.features) != set(self.memories):
            raise RuntimeError('Missing support features: verify explicit context propagation to GDN')
        with torch.inference_mode(False), torch.enable_grad():
            result = adapt_after_chunk(self.memories, self.state, noisy.features,
                                       {i: f.m for i, f in clean.features.items()}, chunk, training)
        self.metrics.append(result)
        return result

    def begin_chunk(self, chunk, batch, training=False):
        if self.state is None:
            self.reset_episode('inference', batch, training)
        if self.config.reset_each_chunk and chunk > 0:
            episode = self.state.episode
            metrics = self.metrics
            self.reset_episode(episode, batch, training)
            self.metrics = metrics
            self.state.last_chunk = chunk - 1

    def noisy_support(self, clean, timesteps, model_call, call_kwargs, chunk, conditioned=()):
        """One extra backbone pass on actual flow-corrupted completed latents."""
        cfg = self.config
        rng = torch.Generator(device=clean.device).manual_seed(cfg.seed + chunk * 7919)
        schedule = timesteps.detach().flatten().to(clean.device)
        index = int(torch.randint(len(schedule), (1,), generator=rng, device=clean.device))
        tau = schedule[index].float()
        if cfg.mode == 'kv_ttt':
            tau = tau * 0  # K/V reconstruction reads a clean completed chunk.
        sigma = tau / 1000.0
        noisy = (1 - sigma) * clean + sigma * torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=rng)
        times = tau.expand(clean.shape[0], 1, clean.shape[2]).clone()
        for i in conditioned:
            noisy[:, :, i] = clean[:, :, i]
            times[:, :, i] = 0
        ctx = self.context(times, collect=True, chunk=chunk)
        with torch.no_grad():
            model_call(noisy, times, worldttt_context=ctx, **call_kwargs)
        return ctx

    def save_checkpoint(self, path, extra=None):
        torch.save({'version': 1, 'base_revision': BASE_REVISION, 'base_checkpoint': self.base_checkpoint,
                    'config': asdict(self.config), 'memories': {i: m.state_dict() for i, m in self.memories.items()},
                    'extra': extra or {}}, path)

    def load_checkpoint(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['base_revision'] != BASE_REVISION or set(saved['memories']) != set(self.memories):
            raise ValueError('WorldTTT checkpoint backbone/layer mismatch')
        saved_mode = saved['config']['mode']
        if self.config.mode == 'frozen':
            if saved_mode not in {'kv_ttt', 'noise_ttt'}:
                raise ValueError(f'Cannot freeze checkpoint mode {saved_mode}')
            self.config.frozen_source = saved_mode
        elif saved_mode != self.config.mode:
            raise ValueError(f'WorldTTT checkpoint mode mismatch: {saved_mode} vs {self.config.mode}')
        if self.base_checkpoint is not None and saved.get('base_checkpoint') != self.base_checkpoint:
            raise ValueError('WorldTTT checkpoint baseline identifier mismatch')
        for i, memory in self.memories.items():
            memory.load_state_dict(saved['memories'][i], strict=True)
        return saved.get('extra', {})

    def rollout_spec(self, shape, steps, cfg_scale, flow_shift, boundaries):
        return dict(shape=list(shape), steps=steps, cfg_scale=cfg_scale, flow_shift=flow_shift,
                    boundaries=list(boundaries), config=asdict(self.config), signature=self.rollout_signature,
                    base_revision=BASE_REVISION, base_checkpoint=self.base_checkpoint)

    def restore_rollout(self, snapshot, spec, device):
        if snapshot.get('version') != 1 or snapshot.get('spec') != spec:
            raise ValueError('Rollout checkpoint configuration/input mismatch')
        self.state = WorldTTTState.from_state_dict(snapshot['ttt'], device)
        if set(self.state.weights) != set(self.memories):
            raise ValueError('Rollout checkpoint layer mismatch')
        if self.state.last_chunk != snapshot['next_chunk'] - 1:
            raise ValueError('Rollout checkpoint chunk mismatch')
        self.metrics = snapshot['metrics']
        self.resume_snapshot = None
        return move_tree(snapshot['cache'], device), move_tree(snapshot['init_latents'], device)

    def save_rollout(self, latents, init_latents, cache, next_chunk, spec, generator):
        if self.rollout_path is None:
            return
        snapshot = dict(version=1, spec=spec, next_chunk=next_chunk,
                        latents=move_tree(latents, 'cpu'), init_latents=move_tree(init_latents, 'cpu'),
                        cache=move_tree(cache, 'cpu'), ttt=self.state.state_dict(), metrics=self.metrics,
                        generator_state=None if generator is None else generator.get_state().cpu())
        path = Path(self.rollout_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(snapshot, temporary)
        temporary.replace(path)

    def write_metrics(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for row in self.metrics:
                f.write(json.dumps(row, allow_nan=False) + '\n')
