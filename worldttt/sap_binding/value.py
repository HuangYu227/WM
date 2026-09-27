"""Non-collapsible residual Values for SAP-Bind."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _tokens_to_video(tokens: torch.Tensor, frames: int, height: int, width: int):
    if tokens.ndim != 3 or tokens.shape[1] != frames * height * width:
        raise ValueError('Token/video shape mismatch')
    return tokens.reshape(tokens.shape[0], frames, height, width, -1).permute(0, 4, 1, 2, 3)


def spatial_highpass(tokens: torch.Tensor, frames: int, height: int, width: int):
    video = _tokens_to_video(tokens.float(), frames, height, width)
    centered = video - video.mean((2, 3, 4), keepdim=True)
    padded = F.pad(centered, (1, 1, 1, 1, 0, 0), mode='replicate')
    low = F.avg_pool3d(padded, (1, 3, 3), stride=1)
    residual = centered - low
    residual = residual - residual.mean((2, 3, 4), keepdim=True)
    return residual.permute(0, 2, 3, 4, 1).reshape_as(tokens.float())


def chunk_center(tokens: torch.Tensor):
    if tokens.ndim != 3:
        raise ValueError('Expected batched token features')
    return tokens.float() - tokens.float().mean(1, keepdim=True)


def latent_tokens(latent: torch.Tensor, frames: int, height: int, width: int):
    if latent is None or latent.ndim != 5:
        raise ValueError('SAP-Bind Value requires the current latent chunk')
    x = latent.float()
    if x.shape[2:] != (frames, height, width):
        x = F.interpolate(x, size=(frames, height, width), mode='trilinear', align_corners=False)
    return x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], frames * height * width, x.shape[1])


def fixed_orthogonal(rows: int, cols: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    if rows >= cols:
        matrix = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
        return torch.linalg.qr(matrix, mode='reduced').Q.transpose(0, 1).contiguous()
    # Expansion cannot have orthogonal rows; use an isometric embedding with
    # orthonormal columns so it still preserves all input directions.
    matrix = torch.randn(cols, rows, generator=generator, dtype=torch.float32)
    return torch.linalg.qr(matrix, mode='reduced').Q.contiguous()


class BindingValueEncoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, mode='highpass_h7_latent',
                 value_dim=512, seed=3407):
        super().__init__()
        if value_dim % 2 or mode not in {'old', 'centered_h7', 'highpass_h7', 'highpass_h7_latent'}:
            raise ValueError('Invalid binding Value configuration')
        self.hidden_dim, self.latent_dim = hidden_dim, latent_dim
        self.mode, self.value_dim = mode, value_dim
        half = value_dim // 2
        self.register_buffer('hidden_projection', fixed_orthogonal(hidden_dim, half, seed))
        self.register_buffer('latent_projection', fixed_orthogonal(latent_dim, half, seed + 1))
        self.register_buffer('whiten_mean', torch.zeros(value_dim))
        self.register_buffer('whiten_scale', torch.ones(value_dim))
        self.register_buffer('whitening_fitted', torch.tensor(False))

    def raw(self, hidden, latent, frames, height, width):
        if self.mode == 'old':
            h = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        elif self.mode == 'centered_h7':
            h = chunk_center(hidden)
        else:
            h = spatial_highpass(hidden, frames, height, width)
        h = F.linear(h, self.hidden_projection)
        if self.mode == 'highpass_h7_latent':
            z = latent_tokens(latent, frames, height, width)
            z = F.linear(spatial_highpass(z, frames, height, width), self.latent_projection)
        else:
            z = torch.zeros_like(h)
        return torch.cat((h, z), -1)

    @torch.no_grad()
    def fit_whitening(self, batches):
        rows = [x.detach().float().reshape(-1, self.value_dim).cpu() for x in batches]
        if not rows:
            raise ValueError('Cannot fit whitening without training Values')
        joined = torch.cat(rows)
        if joined.shape[0] < 2 or not torch.isfinite(joined).all():
            raise ValueError('Invalid whitening observations')
        self.whiten_mean.copy_(joined.mean(0))
        self.whiten_scale.copy_(joined.std(0, unbiased=False).clamp_min(1e-4))
        self.whitening_fitted.fill_(True)

    def forward(self, hidden, latent, frames, height, width):
        raw = self.raw(hidden, latent, frames, height, width)
        return (raw - self.whiten_mean) / self.whiten_scale
