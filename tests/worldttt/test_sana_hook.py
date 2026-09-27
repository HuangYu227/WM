"""Execute the real cached camera forward in isolation from CUDA imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_memory_is_between_branch_merge_and_gate():
    root = Path(__file__).resolve().parents[2]
    source = root / 'diffusion/model/nets/sana_gdn_camctrl_blocks.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CachedChunkCausalGDNUCPESinglePathLiteLA')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    ns = {'torch': torch, 'CachedChunkCausalGDN': SimpleNamespace(forward=lambda self, x, **kw: (2 * x, kw['kv_cache']))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
    layer = SimpleNamespace(_compute_frame_gates=lambda x, hw: None,
                            _cached_cam_branch=lambda *args: torch.tensor(3.),
                            out_proj_cam=lambda x: x,
                            _apply_output_gate=lambda value, x: value * 5,
                            proj=SimpleNamespace(weight=torch.tensor(1.)))
    class Projection:
        weight = torch.tensor(1.)
        def __call__(self, x):
            return x * 7
    layer.proj = Projection()
    ctx = SimpleNamespace(apply=lambda layer, x, m, camera, hw: m + 1)
    out, _ = ns['forward'](layer, torch.ones(1, 1, 1), HW=(1, 1, 1),
                          camera_conditions=torch.ones(1), kv_cache=[None], worldttt_context=ctx)
    assert out.item() == 210
    out, _ = ns['forward'](layer, torch.ones(1, 1, 1), HW=(1, 1, 1),
                          camera_conditions=torch.ones(1), kv_cache=[None])
    assert out.item() == 175
