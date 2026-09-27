"""Read-only raw Q/K/V samples from the clean cache-save forward."""
import math

import torch


class AddressTrace:
    def __init__(self, model, layers, latent_frames, camera, token_budget=64):
        self.layers = tuple(layers)
        self.latent_frames = set(latent_frames)
        self.camera = camera
        self.token_budget = token_budget
        self.records = []
        self.active = None
        self.seen = set()
        self.hooks = []
        for layer in self.layers:
            attn = model.blocks[layer - 1].attn
            self.hooks.append(attn.qkv.register_forward_hook(self._hook(layer, attn.heads, attn.dim)))

    def begin_chunk(self, chunk, start, end, spatial_tokens):
        frames = sorted(self.latent_frames.intersection(range(start, end)))
        self.active = (chunk, start, end, spatial_tokens, frames) if frames else None
        self.seen.clear()

    def _hook(self, layer, heads, dim):
        def capture(_module, _inputs, output):
            if self.active is None or layer in self.seen:
                return
            chunk, start, end, latent_shape, frames = self.active
            if output.ndim != 3 or output.shape[1] % (end - start):
                return
            spatial = output.shape[1] // (end - start)
            if output.shape[-1] != 3 * heads * dim:
                raise ValueError('Unexpected raw GDN qkv projection shape')
            if isinstance(latent_shape, tuple):
                grid_h = round(math.sqrt(spatial * latent_shape[0] / latent_shape[1]))
                if grid_h < 1 or spatial % grid_h:
                    raise ValueError('Cannot infer 2D token grid for address trace')
                grid_w = spatial // grid_h
            else:
                grid_h, grid_w = 1, spatial
            self.seen.add(layer)
            # In CFG, the conditional branch is second. One record per selected frame.
            sample = output[-1].detach().reshape(end - start, spatial, 3, heads, dim)
            if isinstance(latent_shape, tuple):
                cols = min(grid_w, max(1, round(math.sqrt(self.token_budget * grid_w / grid_h))))
                rows = min(grid_h, max(1, self.token_budget // cols))
                ys = torch.linspace(0, grid_h - 1, rows, device=output.device).round().long()
                xs = torch.linspace(0, grid_w - 1, cols, device=output.device).round().long()
                tokens = (ys[:, None] * grid_w + xs[None, :]).flatten().unique()
            else:
                tokens = torch.linspace(0, spatial - 1, min(spatial, self.token_budget),
                                        device=output.device).round().long().unique()
            for frame in frames:
                qkv = sample[frame - start, tokens].float().cpu()
                pixel_frame = min(8 * frame, len(self.camera) - 1)
                self.records.append(dict(layer=layer, chunk=chunk, latent_frame=frame,
                    pixel_frame=pixel_frame, spatial_indices=tokens.cpu(),
                    token_grid=(grid_h, grid_w),
                    camera=torch.as_tensor(self.camera[pixel_frame]).float().cpu(),
                    cfg_branch='conditional', q=qkv[:, 0].clone(), k=qkv[:, 1].clone(),
                    v=qkv[:, 2].clone()))
        return capture

    def end_chunk(self):
        self.active = None

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def save(self, path):
        torch.save({'version': 1, 'layers': self.layers, 'latent_frames': sorted(self.latent_frames),
                    'records': self.records}, path)
