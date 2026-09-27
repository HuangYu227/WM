"""SANA block hook and transactional episode runtime for SAP-Bind."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from worldttt.sap_ttt.memory import SapMemoryState
from worldttt.sap_ttt.runtime import token_rays, unpack_text

from .config import BindingConfig
from .memory import BindingBankState, BindingState
from .model import BindingBlock


PROTOCOL_FIELDS = ('layers', 'address_dim', 'value_dim', 'ray_dim', 'latent_dim', 'heads',
                   'topk', 'capacity', 'protected_anchors', 'support_tokens',
                   'address_depth', 'fast_hidden_dim', 'fast_lr', 'fast_trust_weight',
                   'seed', 'value_mode',
                   'architecture')


def validate_protocol(saved, config):
    if not isinstance(saved, dict):
        raise ValueError('Missing SAP-Bind checkpoint protocol')
    expected = asdict(config)
    if any(saved.get(name) != expected[name] for name in PROTOCOL_FIELDS):
        raise ValueError('SAP-Bind checkpoint protocol mismatch')


def _sigma(timesteps, batch, tokens, frames):
    times = timesteps.float().reshape(batch, -1) / 1000.
    if times.shape[1] == 1:
        return times[:, None].expand(batch, tokens, 1)
    if times.shape[1] == frames and tokens % frames == 0:
        return times.repeat_interleave(tokens // frames, 1)[..., None]
    raise ValueError('SAP-Bind sigma/frame/token mismatch')


@dataclass
class BindingContext:
    controller: 'BindingController'
    timesteps: torch.Tensor
    latent: torch.Tensor | None = None
    collect: bool = False
    chunk: int = 0
    training: bool = False
    record_raw: bool = False
    source: str = 'unspecified'
    shuffle_key: bool = False
    shuffle_value: bool = False
    mean_value: bool = False
    last_only: bool = False
    features: dict = field(default_factory=dict)
    raw_features: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def apply(self, block, x_post, x_visual, y, mask, raw_rays, frames):
        module = block.sap_binding
        layer = block.binding_layer_id
        state = self.controller.state[layer]
        batch, tokens, _ = x_post.shape
        if state.bank.keys.shape[0] != batch:
            raise ValueError('SAP-Bind CFG branch count changed within episode')
        text, text_mask = unpack_text(y, mask, batch)
        rays = token_rays(raw_rays, batch, tokens, self.controller.config.ray_dim)
        if rays is None:
            rays = x_post.new_zeros(batch, tokens, self.controller.config.ray_dim)
        sigma = _sigma(self.timesteps, batch, tokens, frames)
        if self.record_raw:
            def snapshot(name, value):
                if value is None:
                    return None
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f'SAP-Bind layer {layer} raw {name} nonfinite before capture')
                saved = value.detach().to('cpu', dtype=torch.bfloat16)
                if not torch.isfinite(saved).all():
                    raise FloatingPointError(f'SAP-Bind layer {layer} raw {name} nonfinite after capture')
                return saved

            self.raw_features[layer] = dict(
                visual=snapshot('visual', x_visual), post=snapshot('post', x_post),
                text=snapshot('text', text), text_mask=text_mask.detach().cpu(),
                rays=snapshot('rays', rays), sigma=sigma.detach().cpu(),
                latent=snapshot('latent', self.latent))
            return x_post
        query = module.address.query(x_visual, text, text_mask, rays, sigma,
                                     mode='selective_geometry')
        if self.controller.config.read_enabled:
            value, stats, mix = module.read(state, query, rays, sigma,
                shuffle_key=self.shuffle_key, shuffle_value=self.shuffle_value,
                mean_value=self.mean_value, last_only=self.last_only)
            residual = module.residual(value, sigma).to(x_post.dtype)
            self.diagnostics[layer] = dict(stats=stats, mix=mix)
        else:
            residual = 0
        if self.collect:
            if not torch.all(sigma == 0):
                raise ValueError('SAP-Bind writes require a completed zero-sigma chunk')
            if self.latent is None:
                raise ValueError('SAP-Bind clean write is missing its completed latent')
            height, width = self.latent.shape[-2:]
            if tokens != frames * height * width:
                raise ValueError('SAP-Bind latent and layer token grids differ')
            generator = torch.Generator().manual_seed(
                self.controller.config.seed + 1009 * self.chunk + layer)
            ids = torch.randperm(tokens, generator=generator)[:min(tokens,
                self.controller.config.support_tokens)].to(x_post.device)
            with torch.enable_grad() if self.training else torch.no_grad():
                key = module.address.write(x_visual[:, ids].detach(), text.detach(), text_mask,
                                           rays[:, ids].detach(), mode='selective_geometry')
                target = module.value(x_post.detach(), self.latent.detach(), frames, height, width)[:, ids]
            # Residual-energy confidence favors anchors carrying identifiable
            # detail while retaining a nonzero chance for low-energy regions.
            confidence = torch.sigmoid(target.float().square().mean(-1).sqrt())
            self.features[layer] = (key.float(), target.float(), rays[:, ids].float(), confidence)
        return x_post + residual


class BindingController:
    def __init__(self, model: nn.Module, config: BindingConfig):
        if getattr(model, 'worldttt_sap_controller', None) is not None:
            raise ValueError('Legacy SAP and SAP-Bind cannot be attached together')
        self.model, self.config = model, config
        self.modules, self.state, self.metrics = {}, {}, []
        self.base_checkpoint = None
        self.rollout_path = None
        model.requires_grad_(False)
        for layer in config.layers:
            if layer > len(model.blocks):
                raise ValueError(f'No SANA block {layer}')
            block = model.blocks[layer - 1]
            vision_dim = int(block.norm2.normalized_shape[0])
            module = BindingBlock(vision_dim, config.latent_dim, config).to(next(block.parameters()).device)
            block.add_module('sap_binding', module)
            block.binding_layer_id = layer
            self.modules[layer] = module
        object.__setattr__(model, 'worldttt_binding_controller', self)

    def reset_episode(self, episode, batch, training=False):
        self.state = {}
        for layer, module in self.modules.items():
            device = next(module.parameters()).device
            bank = BindingBankState.new(batch, self.config.capacity, self.config.address_dim,
                                         self.config.value_dim, self.config.ray_dim, device)
            fast = SapMemoryState.new(module.fast, str(episode), batch, training)
            self.state[layer] = BindingState(str(episode), bank, fast)
        self.metrics = []

    def context(self, timesteps, latent=None, collect=False, chunk=0, training=False,
                record_raw=False, source='unspecified', **controls):
        if not self.state:
            raise RuntimeError('Reset SAP-Bind episode before reading')
        return BindingContext(self, timesteps, latent, collect, chunk, training,
                              record_raw, source, **controls)

    def commit(self, context: BindingContext, chunk: int, training=False):
        if any(chunk != state.last_chunk + 1 for state in self.state.values()):
            raise ValueError('SAP-Bind duplicate/out-of-order chunk commit')
        if self.config.mode == 'frozen' or not self.config.commit_enabled:
            for state in self.state.values():
                state.last_chunk = chunk
            row = dict(chunk=chunk, committed=False,
                       mode='frozen' if self.config.mode == 'frozen' else 'no_commit',
                       source=context.source)
            self.metrics.append(row)
            return row
        if set(context.features) != set(self.modules):
            raise RuntimeError('SAP-Bind clean support missing for selected layer')
        proposals = {}
        try:
            for layer, module in self.modules.items():
                key, value, rays, confidence = context.features[layer]
                state = self.state[layer]
                fast, loss = module.fast.update(state.fast.weight, key, value,
                                                 create_graph=training)
                bank = state.bank.clone()
                bank.commit(key.detach(), value.detach(), rays.detach(), confidence.detach(),
                    chunk=chunk, protected_budget=self.config.protected_anchors,
                    seed=self.config.seed + layer)
                proposals[layer] = fast, loss, bank
        except (FloatingPointError, ValueError) as exc:
            row = dict(chunk=chunk, committed=False, reason=str(exc), source=context.source)
        else:
            for layer, (fast, _, bank) in proposals.items():
                state = self.state[layer]
                state.fast.weight = fast if training else fast.detach().requires_grad_(True)
                state.fast.last_chunk = state.last_chunk = chunk
                state.fast.updates += 1
                state.updates += 1
                state.bank = bank
            row = dict(chunk=chunk, committed=True, source=context.source,
                write_objective={str(k): float(v[1].detach()) for k, v in proposals.items()},
                anchors={str(k): int(v[2].valid.sum()) for k, v in proposals.items()})
        self.metrics.append(row)
        return row

    def adapter_fingerprint(self):
        digest = hashlib.sha256()
        for layer, module in sorted(self.modules.items()):
            for name, tensor in sorted(module.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(f'{layer}:{name}:{value.dtype}:{tuple(value.shape)}'.encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def save_checkpoint(self, path, extra=None):
        torch.save(dict(version=1, kind='sap_binding_adapter', config=asdict(self.config),
            base_checkpoint=self.base_checkpoint,
            modules={i: m.state_dict() for i, m in self.modules.items()}, extra=extra or {}), path)

    def load_checkpoint(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('version') != 1 or data.get('kind') != 'sap_binding_adapter':
            raise ValueError('Checkpoint is not a SAP-Bind adapter')
        validate_protocol(data.get('config'), self.config)
        if self.base_checkpoint is not None and data.get('base_checkpoint') != self.base_checkpoint:
            raise ValueError('SAP-Bind backbone checkpoint mismatch')
        if set(data.get('modules', {})) != set(self.modules):
            raise ValueError('SAP-Bind checkpoint layer mismatch')
        for layer, module in self.modules.items():
            module.load_state_dict(data['modules'][layer], strict=True)
        return data.get('extra', {})

    def save_state(self, path):
        if not self.state:
            raise RuntimeError('Cannot save uninitialized SAP-Bind state')
        torch.save(dict(version=1, kind='sap_binding_state', config=asdict(self.config),
            base_checkpoint=self.base_checkpoint, adapter_fingerprint=self.adapter_fingerprint(),
            states={i: s.state_dict() for i, s in self.state.items()}), path)

    def load_state(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('version') != 1 or data.get('kind') != 'sap_binding_state':
            raise ValueError('File is not a SAP-Bind episode state')
        validate_protocol(data.get('config'), self.config)
        if data.get('base_checkpoint') != self.base_checkpoint or data.get('adapter_fingerprint') != self.adapter_fingerprint():
            raise ValueError('SAP-Bind episode belongs to another backbone or adapter')
        if set(data.get('states', {})) != set(self.modules):
            raise ValueError('SAP-Bind episode layer mismatch')
        restored = {}
        for layer, payload in data['states'].items():
            device = next(self.modules[layer].parameters()).device
            bank = BindingBankState.from_state_dict(payload['bank'], device,
                capacity=self.config.capacity, key_dim=self.config.address_dim,
                value_dim=self.config.value_dim, ray_dim=self.config.ray_dim)
            fast_payload = payload['fast']
            expected_weight = (bank.keys.shape[0], self.modules[layer].fast.initial_weight.numel())
            if (fast_payload.get('version') != 1 or fast_payload['weight'].shape != expected_weight or
                    not torch.isfinite(fast_payload['weight']).all() or
                    fast_payload['episode'] != payload['episode'] or
                    fast_payload['last_chunk'] > payload['last_chunk'] or
                    fast_payload['updates'] != payload['updates'] or
                    payload['last_chunk'] < -1 or payload['updates'] < 0):
                raise ValueError('Malformed SAP-Bind fast episode state')
            fast = SapMemoryState(fast_payload['episode'], fast_payload['weight'].to(device).float().requires_grad_(True),
                                  fast_payload['last_chunk'], fast_payload['updates'])
            restored[layer] = BindingState(payload['episode'], bank, fast,
                                            payload['last_chunk'], payload['updates'])
        self.state = restored
        return restored

    def write_metrics(self, path):
        with open(path, 'w', encoding='utf-8') as stream:
            for row in self.metrics:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
