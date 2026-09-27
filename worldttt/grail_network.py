"""Canonical observation encoder and hidden-conditioned sparse memory readers."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .grail_geometry import canonical_token_geometry, geometry_distance


@dataclass(frozen=True)
class GrailNetworkConfig:
    width: int = 128
    heads: int = 4
    neighbors: int = 16
    candidates: int = 256
    support_tokens: int = 256
    query_block: int = 256

    def __post_init__(self):
        if any(v < 1 for v in asdict(self).values()) or self.width % self.heads:
            raise ValueError("positive network sizes and width divisible by heads required")


def gather_tokens(x, ids):
    return x[torch.arange(x.shape[0], device=x.device)[:, None, None], ids]


def sample_indices(n, limit, device):
    return torch.linspace(0, n - 1, min(n, limit), device=device).long()


def sparse_attention(q, k, v, bias, heads):
    """Q [B,N,D], K/V [B,N,L,D], bias [B,N,L]; no dense N x N attention."""
    b, n, d = q.shape
    length, dh = k.shape[-2], d // heads
    q = q.reshape(b * n, heads, 1, dh)
    k = k.reshape(b * n, length, heads, dh).transpose(1, 2)
    v = v.reshape(b * n, length, heads, dh).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.reshape(b * n, 1, 1, length), dropout_p=0.)
    return out.reshape(b, n, d)


class CanonicalMemoryWriter(nn.Module):
    """Only this module produces persistent K/V, always from SANA block 2."""
    def __init__(self, hidden_dim, ledger_config, network):
        super().__init__()
        self.network, self.ledger_config = network, ledger_config
        w = network.width
        self.norm = nn.LayerNorm(hidden_dim)
        self.visual = nn.Linear(hidden_dim, w)
        self.depth = nn.Linear(w, 1)
        self.geometry = nn.Sequential(nn.Linear(30, 64), nn.SiLU(), nn.Linear(64, w), nn.LayerNorm(w))
        self.qkv = nn.Linear(w, 3 * w)
        self.attn_out = nn.Linear(w, w)
        self.ffn_norm = nn.LayerNorm(w)
        self.ffn = nn.Sequential(nn.Linear(w, 2 * w), nn.SiLU(), nn.Linear(2 * w, w))
        self.address = nn.Linear(w, ledger_config.key_dim)
        self.value = nn.Linear(w, ledger_config.value_dim)
        self.write_gate = nn.Sequential(nn.Linear(w + 4, 64), nn.SiLU(), nn.Linear(64, 1))

    def encode(self, x, camera, thw, patch_size, visibility, source_real=None):
        z = self.visual(self.norm(x))
        depth = 0.1 + F.softplus(self.depth(z))
        geo = canonical_token_geometry(camera, thw, patch_size, depth, visibility)
        scaled_geo = geo.sign() * geo.abs().log1p()
        z = z + self.geometry(scaled_geo.to(z.dtype))
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        ids = sample_indices(x.shape[1], self.network.candidates, x.device)
        candidate_geo, candidate_valid = geo[:, ids], visibility[:, ids, 0] > 0
        outputs = []
        for start in range(0, x.shape[1], self.network.query_block):
            end = start + self.network.query_block
            distance = geometry_distance(geo[:, start:end, None].float(), candidate_geo[:, None].float(),
                                         self.ledger_config.geometry_scale)
            allowed = candidate_valid[:, None].expand_as(distance)
            if source_real is not None:
                # A protected real observation must never aggregate generated content.
                allowed = allowed & (~source_real[:, start:end, None] | source_real[:, None, ids])
            scores = (-distance).masked_fill(~allowed, -torch.inf)
            selected, neighbors = scores.topk(min(self.network.neighbors, len(ids)), dim=-1)
            message = sparse_attention(q[:, start:end], gather_tokens(k[:, ids], neighbors),
                                       gather_tokens(v[:, ids], neighbors), selected, self.network.heads)
            outputs.append(message)
        z = z + self.attn_out(torch.cat(outputs, dim=1))
        z = z + self.ffn(self.ffn_norm(z))
        keys = F.normalize(self.address(z).float(), dim=-1)
        # Fixed normalization prevents arbitrary value scale from controlling Ridge.
        values = F.layer_norm(self.value(z).float(), (self.ledger_config.value_dim,))
        return z, keys, values, geo.float()

    def confidence(self, z, visibility, source_real, retrieved, values):
        valid = retrieved.valid.any(-1)
        scores = retrieved.scores.masked_fill(~retrieved.valid, -1e4)
        weights = scores.softmax(-1) * retrieved.valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        prediction = (weights[..., None] * retrieved.values).sum(-2)
        disagreement = (prediction - values).square().mean(-1).clamp_min(1e-8).sqrt().clamp_max(10)
        diagnostics = torch.stack((visibility[..., 0], source_real.float(),
                                   (~valid).float(), disagreement), dim=-1).to(z.dtype)
        return visibility[..., 0] * self.write_gate(torch.cat((z, diagnostics), -1)).sigmoid()[..., 0]


class SparseMemoryReader(nn.Module):
    """Q=current SANA hidden; K=canonical slot key/geometry; V=slot Ridge(q)."""
    def __init__(self, hidden_dim, ledger_config, network):
        super().__init__()
        w = network.width
        self.network = network
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, w)
        self.key = nn.Linear(ledger_config.key_dim + 30, w)
        self.value = nn.Linear(ledger_config.value_dim, w)
        self.output = nn.Linear(w, hidden_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(2 * w + 4, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.constant_(self.gate[-1].bias, -2.)

    def forward(self, x, slots, sigma, visibility):
        outputs, gates = [], []
        for start in range(0, x.shape[1], self.network.query_block):
            end = start + self.network.query_block
            q = self.query(self.norm(x[:, start:end]))
            geom = slots.geometry[:, start:end]
            geom = geom.sign() * geom.abs().log1p()
            k = self.key(torch.cat((slots.keys[:, start:end], geom), -1).to(q.dtype))
            v = self.value(slots.values[:, start:end].to(q.dtype))
            valid = slots.valid[:, start:end]
            bias = slots.scores[:, start:end].masked_fill(~valid, -torch.inf)
            message = sparse_attention(q, k, v, bias, self.network.heads)
            confidence = slots.confidence[:, start:end].max(-1).values
            uncertainty = slots.uncertainty[:, start:end]
            margin = slots.margin[:, start:end].clamp(0, 20) / 20
            time = sigma[:, start:end, 0].to(confidence.dtype)
            diag = torch.stack((time, confidence, uncertainty, margin), dim=-1).to(q.dtype)
            gate = self.gate(torch.cat((q, message, diag), -1)).sigmoid()
            gate = gate * valid.any(-1, keepdim=True) * visibility[:, start:end]
            outputs.append(self.output(message) * gate)
            gates.append(gate)
        return x + torch.cat(outputs, 1).to(x.dtype), torch.cat(gates, 1)
