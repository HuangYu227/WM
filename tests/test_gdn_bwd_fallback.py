import importlib.util
from pathlib import Path

import torch

module_path = Path(__file__).resolve().parents[1] / "diffusion/model/ops/gdn_bwd_fallback.py"
spec = importlib.util.spec_from_file_location("gdn_bwd_fallback", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
phase_a_kv_bwd_torch = module.phase_a_kv_bwd_torch
phase_a_z_bwd_torch = module.phase_a_z_bwd_torch


def test_phase_a_kv_matches_autograd():
    torch.manual_seed(7)
    k = torch.randn(2, 3, 5, 112).to(torch.bfloat16).float().requires_grad_()
    v = torch.randn_like(k).to(torch.bfloat16).float().requires_grad_()
    beta = torch.rand(2, 3, 5, requires_grad=True)
    da = torch.randn(2, 3, 112, 112).to(torch.bfloat16).float()
    dp = torch.randn_like(da).to(torch.bfloat16).float()
    a = k.transpose(-1, -2) @ (beta.unsqueeze(-1) * v)
    p = k.transpose(-1, -2) @ (beta.unsqueeze(-1) * k)
    expected = torch.autograd.grad((a * da).sum() + (p * dp).sum(), (k, v, beta))
    actual = phase_a_kv_bwd_torch(k.detach(), v.detach(), beta.detach(), da, dp)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_phase_a_z_matches_autograd():
    torch.manual_seed(8)
    k = torch.randn(2, 3, 5, 112).to(torch.bfloat16).float().requires_grad_()
    beta = torch.rand(2, 3, 5, requires_grad=True)
    db = torch.randn(2, 3, 112)
    dp = torch.randn(2, 3, 112, 112).to(torch.bfloat16).float()
    b = (beta.unsqueeze(-1) * k).sum(-2)
    p = k.transpose(-1, -2) @ (beta.unsqueeze(-1) * k)
    expected = torch.autograd.grad((b * db).sum() + (p * dp).sum(), (k, beta))
    actual = phase_a_z_bwd_torch(k.detach(), beta.detach(), db, dp)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
