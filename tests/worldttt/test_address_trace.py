from types import SimpleNamespace

import torch


def test_trace_collects_raw_projected_qkv_only_in_selected_clean_pass(tmp_path):
    from worldttt.address_trace import AddressTrace
    torch.manual_seed(7)
    attn = SimpleNamespace(qkv=torch.nn.Linear(4, 3 * 2 * 2, bias=False), heads=2, dim=2)
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)])
    camera = torch.eye(4).repeat(17, 1, 1).numpy()
    trace = AddressTrace(model, layers=(1,), latent_frames=(0, 4), camera=camera, token_budget=3)
    x = torch.randn(2, 8, 4)
    baseline = attn.qkv(x).clone()
    trace.begin_chunk(0, 0, 2, 4)
    assert torch.equal(attn.qkv(x), baseline)
    trace.end_chunk()
    assert torch.equal(attn.qkv(x), baseline)
    trace.begin_chunk(1, 2, 5, 4)
    attn.qkv(torch.randn(2, 12, 4))
    trace.end_chunk()
    trace.close()
    rows = trace.records
    assert {(r['layer'], r['latent_frame']) for r in rows} == {(1, 0), (1, 4)}
    assert all(r['q'].shape == (3, 2, 2) and r['k'].shape == (3, 2, 2) for r in rows)
    assert all(r['cfg_branch'] == 'conditional' for r in rows)
