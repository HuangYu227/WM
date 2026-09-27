"""Production mounting and execution context; no SANA import at module load."""
from functools import wraps
from inspect import signature
from pathlib import Path

import torch

from .grail_native import load_grail_adapter
from .provenance import file_sha256, source_tree_hash


def configure_grail_sana(config):
    """Select the checkpoint-compatible cached architecture used by outer training."""
    model = config.model
    if model.chunk_size != 3 or model.chunk_split_strategy != 'first_chunk_plus_one' or model.softmax_every_n != 4:
        raise ValueError('GRAIL requires causal chunks 4+3+3 and softmax_every_n=4')
    model.model = 'SanaMSVideoCamCtrlStreaming_1600M_P1_D20'
    model.camctrl_type = None
    model.attn_type = 'BidirectionalGDNTriton'
    model.ffn_type = 'CachedGLUMBConvTemp'
    model.pos_embed_type = 'casual_wan_rope'
    model.use_autograd_kernel = True
    return config


def mount_grail(model, adapter, *, mode='off', base_checkpoint=None, data_manifest=None,
                rollout=None, resume=None):
    if mode == 'off' and adapter is None:
        if rollout or resume:
            raise ValueError('GRAIL rollout/resume requires an adapter and online mode')
        return None
    if adapter is None:
        raise ValueError('active GRAIL mode requires an explicitly trained --grail_adapter')
    if (rollout or resume) and (mode != 'online' or not data_manifest):
        raise ValueError('GRAIL rollout/resume requires online mode and --grail_data_manifest')
    if 'Streaming' not in type(model).__name__:
        raise ValueError('GRAIL production mounting requires a SANA Streaming model config')
    base_hash = file_sha256(base_checkpoint)
    ctl, extra = load_grail_adapter(model, adapter, mode=mode, base_checkpoint_hash=base_hash,
                                     rollout_path=rollout, resume_path=resume, require_flow_trained=True)
    model.worldttt_grail_protocol = dict(base_checkpoint_hash=base_hash,
        source_tree_hash=source_tree_hash(Path(__file__).resolve().parents[1]),
        data_manifest_hash=file_sha256(data_manifest) if data_manifest else 'unsaved-input')
    return ctl


def generation_context(function):
    call_signature = signature(function)
    @wraps(function)
    def generate(self, *args, **kwargs):
        ctl = getattr(self.model, 'worldttt_grail_controller', None)
        if ctl is not None and ctl.mode != 'off':
            bound = call_signature.bind(self, *args, **kwargs)
            params = bound.arguments.get('params')
            if params is not None and params.sampling_algo != 'self_forcing':
                raise ValueError('active GRAIL requires sampling_algo=self_forcing')
            # Ridge state tensors must remain ordinary tensors, not inference tensors.
            with torch.inference_mode(False), torch.no_grad():
                return function(self, *args, **kwargs)
        with torch.inference_mode():
            return function(self, *args, **kwargs)
    return generate
