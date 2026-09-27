"""Native GRAIL: one canonical writer, one Ridge ledger, eight sparse readers."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .associative_ttt import AssociativeTTTConfig, AssociativeTTTLedger, ObservationBatch
from .grail_network import CanonicalMemoryWriter, GrailNetworkConfig, SparseMemoryReader, sample_indices

TARGET_GDN_LAYERS = (2, 6, 10, 14)
TARGET_SOFTMAX_LAYERS = (3, 7, 11, 15)
TARGET_LAYERS = tuple(sorted(TARGET_GDN_LAYERS + TARGET_SOFTMAX_LAYERS))
WRITER_LAYER = 2
ARCHITECTURE = 'grail_v2_canonical_writer_v1'
NATIVE_COORDINATE_CONVENTION = 'episode_anchor_pose_uv_plucker_depth_v2'


def adapter_fingerprint(payload):
    metadata = {k: payload[k] for k in ('architecture', 'hidden_dim', 'config', 'network', 'base_checkpoint_hash')}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name, tensor in sorted(payload['model'].items()):
        digest.update((name + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class GrailNativeController(nn.Module):
    def __init__(self, hidden_dim, config=None, *, network=None, mode='off', rollout_path=None, resume_path=None):
        super().__init__()
        if mode not in {'off', 'frozen', 'online'}:
            raise ValueError('mode must be off/frozen/online')
        self.hidden_dim, self.mode = hidden_dim, mode
        self.rollout_path = None if rollout_path is None else Path(rollout_path)
        self.resume_path = None if resume_path is None else Path(resume_path)
        self.config = config or AssociativeTTTConfig(geometry_dim=30, geometry_metric='ray_point',
                                                     coordinate_convention=NATIVE_COORDINATE_CONVENTION)
        if self.config.geometry_dim != 30 or self.config.coordinate_convention != NATIVE_COORDINATE_CONVENTION:
            raise ValueError('native canonical geometry convention mismatch')
        if self.config.geometry_metric != 'ray_point':
            raise ValueError('canonical native memory requires ray_point geometry')
        self.network = network or GrailNetworkConfig()
        self.ledger = AssociativeTTTLedger(self.config)
        self.ledger.read_gate.requires_grad_(False)
        self.writer = CanonicalMemoryWriter(hidden_dim, self.config, self.network)
        self.readers = nn.ModuleDict({str(i): SparseMemoryReader(hidden_dim, self.config, self.network) for i in TARGET_LAYERS})
        self.state = None
        self.hook_counts, self.metrics = {}, []
        self.base_checkpoint_hash = ''

    def reset_episode(self, episode_id, batch_size, *, metadata=None):
        metadata = {**(metadata or {}), 'layer_ids': list(TARGET_LAYERS), 'architecture': ARCHITECTURE}
        self.state = self.ledger.new_state(episode_id, batch_size, device=next(self.parameters()).device, metadata=metadata)
        self.hook_counts, self.metrics = {}, []

    def context(self, mode=None, *, collect=False, chunk_id=None, cfg_conditional_start=None,
                real_frame_indices=(), sigma=None):
        return GrailNativeContext(self, mode or self.mode, collect, chunk_id, cfg_conditional_start,
                                  real_frame_indices, sigma)

    def commit_clean(self, context, chunk_id, *, holdout=None):
        if context.controller is not self or context.mode != 'online' or not context.collect:
            raise ValueError('only this controller online clean context can commit')
        if context.invalid or context.committed or context.chunk_id != chunk_id:
            raise ValueError('repeated/invalid clean transaction or chunk id mismatch')
        if set(context.call_counts) != set(TARGET_LAYERS) or len(context.observations) != 1:
            raise ValueError('clean pass requires eight reader layers and exactly one canonical writer')
        self.state, report = self.ledger.commit(self.state, context.observations[0], chunk_id,
                                               holdout=holdout, differentiable=torch.is_grad_enabled())
        context.committed = True
        report.update(writer_layer=WRITER_LAYER, writer_observations=context.observations[0].keys.shape[1],
                      hook_counts=dict(context.call_counts))
        self.metrics.append(report)
        return report

    def adapter_fingerprint(self):
        return adapter_fingerprint(self.checkpoint_payload())

    def checkpoint_payload(self, **extra):
        return dict(architecture=ARCHITECTURE, hidden_dim=self.hidden_dim, config=asdict(self.config),
                    network=asdict(self.network), model={k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                    base_checkpoint_hash=self.base_checkpoint_hash, extra=extra)

    def save_checkpoint(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(self.checkpoint_payload(**extra), temporary)
        temporary.replace(path)


class GrailNativeContext:
    def __init__(self, controller, mode, collect, chunk_id, cfg_conditional_start, real_frame_indices, sigma):
        if mode not in {'off', 'frozen', 'online'} or (collect and (mode != 'online' or chunk_id is None)):
            raise ValueError('invalid GRAIL context mode/clean collection')
        self.controller, self.mode, self.collect, self.chunk_id = controller, mode, collect, chunk_id
        self.cfg_conditional_start, self.real_frame_indices, self.sigma = cfg_conditional_start, real_frame_indices, sigma
        self.call_counts, self.observations, self.gates = {}, [], {}
        self.committed = self.invalid = False
        self.slot_reads = None

    def apply(self, block, x, camera_conditions, thw, frame_valid_mask):
        if self.mode == 'off':
            return x
        ctl, state = self.controller, self.controller.state
        if state is None:
            raise RuntimeError('reset_episode must run before a GRAIL read')
        layer = getattr(block, 'grail_layer_index', None)
        if layer not in TARGET_LAYERS or x.ndim != 3 or x.shape[1] != thw[0] * thw[1] * thw[2]:
            raise ValueError('invalid GRAIL layer/token layout')
        b, n, _ = x.shape
        cfg = self.cfg_conditional_start
        if (cfg is None and b != state.batch_size) or (cfg is not None and (cfg != state.batch_size or b != 2 * cfg)):
            raise ValueError('GRAIL CFG batch must be unconditional then conditional')
        self.call_counts[layer] = self.call_counts.get(layer, 0) + 1
        ctl.hook_counts[layer] = ctl.hook_counts.get(layer, 0) + 1
        if self.call_counts[layer] != 1:
            self.invalid = True
            raise ValueError('repeated GRAIL layer call: use a fresh context per forward, no checkpoint replay')
        if camera_conditions is None:
            raise ValueError('GRAIL requires camera_conditions')
        if camera_conditions.shape[0] == state.batch_size and cfg is not None:
            camera_conditions = camera_conditions.repeat(2, 1, 1)
        if frame_valid_mask is not None and frame_valid_mask.shape[0] == state.batch_size and cfg is not None:
            frame_valid_mask = frame_valid_mask.repeat((2,) + (1,) * (frame_valid_mask.ndim - 1))
        visibility = (x.new_ones(b, n, 1) if frame_valid_mask is None else
                      block._build_frame_token_mask(frame_valid_mask, B=b, T=thw[0], N=n, device=x.device, dtype=x.dtype))
        if layer == WRITER_LAYER:
            real = torch.zeros(b, n, device=x.device, dtype=torch.bool)
            for frame in self.real_frame_indices:
                if not 0 <= frame < thw[0]:
                    raise ValueError('real frame index outside chunk')
                real[:, frame * thw[1] * thw[2]:(frame + 1) * thw[1] * thw[2]] = True
            z, keys, values, geometry = ctl.writer.encode(x, camera_conditions, thw, block.grail_patch_size,
                                                         visibility, real if self.collect else None)
            pairs = [(keys, geometry)] if cfg is None else list(zip(keys.split(cfg), geometry.split(cfg)))
            self.slot_reads = [ctl.ledger.read_slots(state, q, g, chunk_id=self.chunk_id, block_size=ctl.network.query_block)
                               for q, g in pairs]
            if self.collect:
                start = cfg or 0
                confidence = ctl.writer.confidence(z[start:], visibility[start:], real[start:], self.slot_reads[-1], values[start:])
                ids = sample_indices(n, ctl.network.support_tokens, x.device)
                self.observations.append(ObservationBatch(keys[start:, ids], values[start:, ids], geometry[start:, ids],
                                                           confidence[:, ids], real[start:, ids]))
        if self.slot_reads is None:
            raise ValueError('canonical writer layer must execute before readers')
        sigma = torch.as_tensor(0. if self.sigma is None else self.sigma, device=x.device, dtype=x.dtype)
        if sigma.numel() == 1:
            sigma = sigma.expand(b, n, 1)
        else:
            sigma = sigma.reshape(b, -1, 1).repeat_interleave(thw[1] * thw[2], dim=1)
        features = [x] if cfg is None else x.split(cfg)
        times = [sigma] if cfg is None else sigma.split(cfg)
        masks = [visibility] if cfg is None else visibility.split(cfg)
        results = [ctl.readers[str(layer)](h, slots, time, mask)
                   for h, slots, time, mask in zip(features, self.slot_reads, times, masks)]
        output = torch.cat([r[0] for r in results], 0)
        self.gates[layer] = torch.cat([r[1] for r in results], 0)
        if not torch.isfinite(output).all():
            raise FloatingPointError('nonfinite GRAIL output')
        return output


def attach_grail_native(model, controller):
    if hasattr(model, 'worldttt_grail_controller'):
        raise ValueError('GRAIL controller already attached')
    if not hasattr(model, 'blocks') or len(model.blocks) <= max(TARGET_LAYERS):
        raise ValueError('SANA model lacks GRAIL target layers')
    if getattr(model, 'softmax_every_n', None) != 4 or getattr(model, 'camctrl_layers_num', 0) < 16:
        raise ValueError('GRAIL requires softmax_every_n=4 and 16 camera-controlled layers')
    patch = getattr(model, 'patch_size', None)
    if not isinstance(patch, (tuple, list)) or len(patch) != 3 or patch[0] != 1 or patch[1] != patch[2]:
        raise ValueError('GRAIL requires temporal patch 1 and equal spatial patch sizes')
    if any(model.blocks[i].hidden_size != controller.hidden_dim for i in TARGET_LAYERS):
        raise ValueError('GRAIL width differs from SANA')
    if controller.state is not None:
        raise ValueError('attach before initializing episode state')
    base = next(model.parameters(), None)
    model.requires_grad_(False)
    if base is not None:
        controller.to(device=base.device, dtype=base.dtype)
    model.worldttt_grail_controller = controller
    for i in TARGET_LAYERS:
        model.blocks[i].grail_layer_index, model.blocks[i].grail_patch_size = i, patch[1]


def load_grail_adapter(model, path, *, mode='online', base_checkpoint_hash=None, rollout_path=None, resume_path=None,
                       require_flow_trained=False):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('architecture') != ARCHITECTURE:
        raise ValueError('incompatible GRAIL architecture: old independent-layer heads cannot be migrated implicitly')
    if base_checkpoint_hash is not None and payload['base_checkpoint_hash'] != base_checkpoint_hash:
        raise ValueError('GRAIL/base checkpoint hash mismatch')
    if require_flow_trained and not payload.get('extra', {}).get('flow_trained'):
        raise ValueError('GRAIL adapter lacks a completed Future Flow optimizer step')
    ctl = GrailNativeController(payload['hidden_dim'], AssociativeTTTConfig(**payload['config']),
                                network=GrailNetworkConfig(**payload['network']), mode=mode,
                                rollout_path=rollout_path, resume_path=resume_path)
    ctl.load_state_dict(payload['model'], strict=True)
    ctl.base_checkpoint_hash = payload['base_checkpoint_hash']
    attach_grail_native(model, ctl)
    return ctl, payload.get('extra', {})
