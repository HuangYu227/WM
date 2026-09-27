"""Low shared-memory GPU fallback for the GDN phase-A backward matmuls."""

import torch


def _rounded_bf16(x):
    # Match the Triton kernel's bf16 dot operands and fp32 accumulation.
    return x.to(torch.bfloat16).float()


def phase_a_kv_bwd_torch(k, v, beta, da, dp):
    k_dot = _rounded_bf16(k)
    v_dot = _rounded_bf16(v)
    da_dot = _rounded_bf16(da)
    dp_dot = _rounded_bf16(dp)
    k_dp = k_dot @ dp_dot
    k_da = k_dot @ da_dot
    b = beta.float().unsqueeze(-1)
    dk = b * (k_dp + k_dot @ dp_dot.transpose(-1, -2) + v_dot @ da_dot.transpose(-1, -2))
    dv = b * k_da
    dbeta = (k_dp * k.float() + k_da * v.float()).sum(-1)
    return dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype)


def phase_a_z_bwd_torch(k, beta, db, dp):
    k_dot = _rounded_bf16(k)
    dp_dot = _rounded_bf16(dp)
    k_dp = k_dot @ dp_dot
    b = beta.float().unsqueeze(-1)
    dk = b * (k_dp + k_dot @ dp_dot.transpose(-1, -2) + db.float().unsqueeze(-2))
    dbeta = (k_dp * k.float()).sum(-1) + (k.float() * db.float().unsqueeze(-2)).sum(-1)
    return dk.to(k.dtype), dbeta.to(beta.dtype)
