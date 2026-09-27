"""Post-text SANA residual hook and episode-scoped SAP fast memory."""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .address import MODES, SapAddress, initialize_fixed_value
from .memory import SapMemory, SapMemoryState


PROTOCOL_FIELDS = ('layers', 'address_mode', 'dim', 'ray_dim', 'support_tokens',
                   'inner_lr', 'seed', 'address_arch', 'memory_arch', 'heads',
                   'address_depth', 'memory_hidden_dim', 'normalized_value')


def validate_protocol(saved, config):
    expected, actual = asdict(config), asdict(SapConfig(**saved))
    if any(actual[name] != expected[name] for name in PROTOCOL_FIELDS):
        raise ValueError('SAP adapter address/write protocol mismatch')


@dataclass
class SapConfig:
    mode: str = 'online'
    layers: tuple[int, ...] = (7,)
    address_mode: str = 'selective_geometry'
    dim: int = 256
    ray_dim: int = 48
    support_tokens: int = 256
    inner_lr: float = .5
    seed: int = 3407
    read_enabled: bool = True
    commit_enabled: bool = True
    shuffle_query_text: bool = False
    address_arch: str = 'linear'
    memory_arch: str = 'linear'
    heads: int = 8
    address_depth: int = 2
    memory_hidden_dim: int = 128
    normalized_value: bool = False

    def __post_init__(self):
        self.layers = tuple(self.layers)
        if self.mode not in {'frozen', 'online'} or self.address_mode not in MODES:
            raise ValueError('Invalid SAP mode or address mode')
        if not self.layers or min(self.layers) < 1 or len(set(self.layers)) != len(self.layers):
            raise ValueError('SAP layers must be unique one-based indices')
        if min(self.dim, self.ray_dim, self.support_tokens) < 1 or self.inner_lr <= 0:
            raise ValueError('Invalid SAP dimensions or update rate')
        if self.address_arch not in {'linear', 'multimodal'} or self.memory_arch not in {'linear', 'swiglu'}:
            raise ValueError('Invalid SAP architecture')
        if min(self.heads, self.address_depth, self.memory_hidden_dim) < 1:
            raise ValueError('Invalid SAP architecture dimensions')
        if (self.address_arch == 'multimodal' or self.memory_arch == 'swiglu') and self.dim % self.heads:
            raise ValueError('SAP dim must be divisible by heads')


def unpack_text(y: torch.Tensor, lengths, batch: int):
    """Undo SANA's xFormers packed text, preserving CFG branch boundaries."""
    if y.ndim == 4 and y.shape[1] == 1:
        y = y.squeeze(1)
    if isinstance(lengths, torch.Tensor):
        if y.shape[0] != batch or lengths.shape != y.shape[:2]:
            raise ValueError('Unpacked SANA text/mask batch mismatch')
        return y, lengths.bool()
    if lengths is None:
        if y.shape[0] != batch:
            raise ValueError('Missing packed text lengths')
        return y, torch.ones(y.shape[:2], device=y.device, dtype=torch.bool)
    sizes = [int(n) for n in lengths]
    if len(sizes) != batch or min(sizes) < 1:
        raise ValueError('Packed text length/CFG branch mismatch')
    if y.shape[0] == batch and y.shape[1] >= max(sizes):
        mask = torch.arange(y.shape[1], device=y.device)[None] < torch.tensor(sizes, device=y.device)[:, None]
        return y, mask
    if y.shape[0] != 1 or y.shape[1] != sum(sizes):
        raise ValueError('Packed text token count mismatch')
    padded = pad_sequence(y[0].split(sizes, dim=0), batch_first=True)
    mask = torch.arange(padded.shape[1], device=y.device)[None] < torch.tensor(sizes, device=y.device)[:, None]
    return padded, mask


def token_rays(raw: torch.Tensor | None, batch: int, tokens: int, channels: int):
    if raw is None:
        return None
    if raw.ndim == 3 and raw.shape == (batch, tokens, channels):
        return raw
    if raw.ndim == 5 and raw.shape[0] == batch and raw.shape[1] == channels:
        rays = raw.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
        if rays.shape[1] == tokens:
            return rays
    raise ValueError('Raw Plucker ray/token shape mismatch')


class SapBlockMemory(nn.Module):
    def __init__(self, vision_dim: int, config: SapConfig):
        super().__init__()
        if config.address_arch == 'multimodal':
            from .multimodal import MultimodalAddress
            self.address = MultimodalAddress(vision_dim, vision_dim, config.ray_dim,
                                             config.dim, config.heads, config.address_depth)
        else:
            self.address = SapAddress(vision_dim, vision_dim, config.ray_dim, config.dim)
        if config.normalized_value or config.address_arch == 'multimodal':
            initialize_fixed_value(self.address, config.seed)
            self.address.normalized_value = True
        if config.memory_arch == 'swiglu':
            from .nonlinear import SwiGLUMemory
            self.memory = SwiGLUMemory(config.dim, config.heads, config.memory_hidden_dim, config.inner_lr)
        else:
            self.memory = SapMemory(config.dim, config.inner_lr)
        self.output = nn.Linear(config.dim, vision_dim, bias=False)
        nn.init.normal_(self.output.weight, std=.01)
        self.gate = nn.Parameter(torch.zeros(()))
        if config.address_arch == 'multimodal':
            self.read_gate = nn.Sequential(nn.Linear(config.dim + 1, config.heads), nn.Sigmoid())

    def residual(self, query, value, sigma):
        with torch.autocast(device_type=query.device.type, enabled=False):
            if hasattr(self, 'read_gate'):
                gates = self.read_gate(torch.cat((query.float(), sigma.float()), -1))
                value = (value.reshape(*value.shape[:2], gates.shape[-1], -1) * gates[..., None]).flatten(-2)
            return self.gate.float() * self.output(value.float())


@dataclass
class SapContext:
    controller: SapController
    timesteps: torch.Tensor
    collect: bool = False
    chunk: int = 0
    record_raw: bool = False
    training: bool = False
    record_query: bool = False
    source: str = 'unspecified'
    features: dict = field(default_factory=dict)
    raw_features: dict = field(default_factory=dict)
    query_addresses: dict = field(default_factory=dict)
    support_indices: dict = field(default_factory=dict)

    def apply(self, block, x_post, x_visual, y, mask, raw_rays, frames):
        module = block.sap_ttt
        idx = block.sap_layer_id
        b, n, _ = x_post.shape
        state = self.controller.state[idx]
        if state.weight.shape[0] != b:
            raise ValueError('SAP CFG batch/branch count changed within episode')
        text, text_mask = unpack_text(y, mask, b)
        rays = token_rays(raw_rays, b, n, self.controller.config.ray_dim)
        times = self.timesteps.float().reshape(b, -1) / 1000.
        if times.shape[1] == 1:
            sigma = times[:, None, :].expand(b, n, 1)
        elif times.shape[1] == frames and n % frames == 0:
            sigma = times.repeat_interleave(n // frames, 1)[..., None]
        else:
            raise ValueError('SAP sigma/frame/token mismatch')
        if self.record_raw:
            def snapshot(name, value):
                if value is None:
                    return None
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f'SAP layer {idx} raw {name} is nonfinite before capture')
                # The frozen BF16 backbone can exceed FP16's 65504 range at
                # deeper layers. BF16 keeps the same storage size and range.
                saved = value.detach().to('cpu', dtype=torch.bfloat16)
                if not torch.isfinite(saved).all():
                    raise FloatingPointError(f'SAP layer {idx} raw {name} is nonfinite after capture')
                return saved

            self.raw_features[idx] = {
                'visual': snapshot('visual', x_visual),
                'post': snapshot('post', x_post),
                'text': snapshot('text', text),
                'text_mask': text_mask.detach().cpu(),
                'rays': snapshot('rays', rays),
                'sigma': sigma.detach().cpu(),
            }
            return x_post
        mode = self.controller.config.address_mode
        query_text = text.roll(1, dims=-1) if self.controller.config.shuffle_query_text else text
        query = module.address.query(x_visual, query_text, text_mask, rays, sigma, mode=mode)
        if self.record_query:
            self.query_addresses[idx] = query
        if self.controller.config.read_enabled:
            value = module.memory.read(state, query)
            residual = module.residual(query, value, sigma).to(x_post.dtype)
        else:
            residual = 0
        if self.collect:
            if not torch.all(sigma == 0):
                raise ValueError('SAP write capture requires completed zero-sigma chunk')
            # Multi-layer supervision needs the same historical token positions
            # at every selected depth. Keep the established single-layer seed.
            sample_layer = self.controller.config.layers[0] if len(self.controller.config.layers) > 1 else idx
            rng = torch.Generator().manual_seed(self.controller.config.seed + 1009 * self.chunk + sample_layer)
            ids = torch.randperm(n, generator=rng)[:min(n, self.controller.config.support_tokens)].to(x_post.device)
            self.support_indices[idx] = ids
            with torch.enable_grad() if self.training else torch.no_grad():
                key = module.address.write(x_visual[:, ids].detach(), text.detach(), text_mask,
                                           None if rays is None else rays[:, ids].detach(), mode=mode)
                target = module.address.value_for(x_post[:, ids].detach())
            self.features[idx] = (key.float(), target.float()) if self.training else (
                key.detach().float(), target.detach().float())
        return x_post + residual


class SapController:
    def __init__(self, model: nn.Module, config: SapConfig):
        self.model, self.config = model, config
        self.modules = {}
        self.state = {}
        self.metrics = []
        self.base_checkpoint = None
        self.rollout_path = None
        model.requires_grad_(False)
        for idx in config.layers:
            if idx > len(model.blocks):
                raise ValueError(f'No SANA block {idx}')
            block = model.blocks[idx - 1]
            dim = block.norm2.normalized_shape[0]
            module = SapBlockMemory(dim, config).to(next(block.parameters()).device)
            block.add_module('sap_ttt', module)
            block.sap_layer_id = idx
            self.modules[idx] = module
        object.__setattr__(model, 'worldttt_sap_controller', self)

    def reset_episode(self, episode: str, batch: int, training: bool = False):
        self.state = {i: SapMemoryState.new(m.memory, episode, batch, training)
                      for i, m in self.modules.items()}
        self.metrics = []

    def context(self, timesteps, collect=False, chunk=0, record_raw=False, training=False,
                record_query=False, source='unspecified'):
        if not self.state:
            raise RuntimeError('Reset SAP episode before reading')
        return SapContext(self, timesteps, collect, chunk, record_raw, training, record_query,
                          source)

    def commit(self, context: SapContext, chunk: int, training: bool = False):
        if self.config.mode == 'frozen' or not self.config.commit_enabled:
            for state in self.state.values():
                if chunk != state.last_chunk + 1:
                    raise ValueError('SAP frozen chunk sequence mismatch')
                state.last_chunk = chunk
            row = {'chunk': chunk, 'committed': False,
                   'mode': 'frozen' if self.config.mode == 'frozen' else 'no_commit',
                   'source': context.source}
        else:
            if set(context.features) != set(self.modules):
                raise RuntimeError('SAP clean support missing for selected layer')
            proposals = {}
            try:
                for idx, module in self.modules.items():
                    state = self.state[idx]
                    if chunk != state.last_chunk + 1:
                        raise ValueError('SAP duplicate/out-of-order chunk commit')
                    keys, values = context.features[idx]
                    proposals[idx] = module.memory.update(state.weight, keys, values, create_graph=training)
            except FloatingPointError as exc:
                row = {'chunk': chunk, 'committed': False, 'reason': str(exc),
                       'source': context.source}
            else:
                for idx, (proposed, _) in proposals.items():
                    state = self.state[idx]
                    state.weight = proposed if training else proposed.detach().requires_grad_(True)
                    state.last_chunk = chunk
                    state.updates += 1
                row = {'chunk': chunk, 'committed': True, 'source': context.source,
                       'write_objective': {str(idx): float(loss.detach()) for idx, (_, loss) in proposals.items()}}
        self.metrics.append(row)
        return row

    def save_checkpoint(self, path, extra=None):
        torch.save({'version': 1, 'config': asdict(self.config), 'base_checkpoint': self.base_checkpoint,
                    'modules': {i: module.state_dict() for i, module in self.modules.items()},
                    'extra': extra or {}}, path)

    def save_state(self, path):
        if not self.state:
            raise RuntimeError('Cannot save an uninitialized SAP episode')
        torch.save({'version': 1, 'layers': self.config.layers, 'config': asdict(self.config),
                    'base_checkpoint': self.base_checkpoint,
                    'adapter_fingerprint': self.adapter_fingerprint(),
                    'states': {i: state.state_dict() for i, state in self.state.items()}}, path)

    def adapter_fingerprint(self):
        digest = hashlib.sha256()
        for idx, module in sorted(self.modules.items()):
            for name, tensor in sorted(module.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(f'{idx}:{name}:{value.dtype}:{tuple(value.shape)}'.encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def load_state(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('version') != 1 or tuple(data['layers']) != self.config.layers:
            raise ValueError('SAP episode state version/layer mismatch')
        if 'config' in data:
            validate_protocol(data['config'], self.config)
        elif self.config.address_arch != 'linear' or self.config.memory_arch != 'linear':
            raise ValueError('Legacy SAP state has no compatible architecture protocol')
        if data.get('base_checkpoint') != self.base_checkpoint:
            raise ValueError('SAP episode backbone checkpoint mismatch')
        if 'adapter_fingerprint' in data:
            if data['adapter_fingerprint'] != self.adapter_fingerprint():
                raise ValueError('SAP episode belongs to a different adapter')
        elif self.config.address_arch != 'linear' or self.config.memory_arch != 'linear':
            raise ValueError('SAP episode is missing adapter identity')
        if set(data['states']) != set(self.modules):
            raise ValueError('SAP episode state layer mismatch')
        proposed = {}
        for i, payload in data['states'].items():
            device = next(self.modules[i].parameters()).device
            weight = payload['weight']
            expected_shape = self.modules[i].memory.initial_weight.shape
            if weight.shape[1:] != expected_shape or weight.shape[0] < 1 or not torch.isfinite(weight).all():
                raise ValueError('SAP episode state shape/nonfinite mismatch')
            proposed[i] = SapMemoryState(payload['episode'],
                payload['weight'].to(device).float().requires_grad_(True),
                payload['last_chunk'], payload['updates'])
        self.state = proposed
        return self.state

    def load_checkpoint(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('version') != 1 or set(data['modules']) != set(self.modules):
            raise ValueError('SAP checkpoint version/layer mismatch')
        validate_protocol(data.get('config', {}), self.config)
        if self.base_checkpoint is not None and data['base_checkpoint'] != self.base_checkpoint:
            raise ValueError('SAP backbone checkpoint mismatch')
        for i, module in self.modules.items():
            expected, actual = module.state_dict(), data['modules'][i]
            if set(expected) != set(actual) or any(actual[k].shape != v.shape or not torch.isfinite(actual[k]).all()
                                                 for k, v in expected.items()):
                raise ValueError('SAP adapter parameter shape/nonfinite mismatch')
        for i, module in self.modules.items():
            module.load_state_dict(data['modules'][i], strict=True)
        return data['extra']

    def write_metrics(self, path):
        with open(path, 'w', encoding='utf-8') as stream:
            for row in self.metrics:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
