"""Camera kernel dispatch can be checked without importing CUDA/Triton."""
import ast
from pathlib import Path

import torch


def test_camera_autograd_dispatch_requires_live_gradients():
    source = Path(__file__).resolve().parents[2] / 'diffusion/model/nets/sana_gdn_blocks_triton.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_needs_autograd_kernel')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), namespace)
    choose = namespace['_needs_autograd_kernel']
    x = torch.ones(1, requires_grad=True)

    assert choose(True, x)
    assert not choose(False, x)
    assert not choose(True, x.detach())
    with torch.no_grad():
        assert not choose(True, x)

    for name in ('ChunkCausalGDNUCPESinglePathLiteLABothTriton',
                 'BidirectionalGDNUCPESinglePathLiteLABothTriton'):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        branch = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_forward_cam_branch')
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == '_needs_autograd_kernel' for node in ast.walk(branch))
