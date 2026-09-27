import torch
import pytest


def test_tree_max_error_detects_cache_and_output_changes():
    from worldttt.sap_ttt.check_gate import tree_max_error

    a = (torch.ones(2), [torch.zeros(1)])
    b = (torch.ones(2), [torch.ones(1)])
    assert tree_max_error(a, a) == 0
    assert tree_max_error(a, b) == 1


def test_tree_max_error_compares_sana_scalar_cache_type_flags():
    from worldttt.sap_ttt.check_gate import tree_max_error

    baseline = [[torch.ones(1), None, 1.0], [torch.zeros(1), None, 0.0]]
    same = [[torch.ones(1), None, 1.0], [torch.zeros(1), None, 0.0]]
    wrong_flag = [[torch.ones(1), None, 0.0], [torch.zeros(1), None, 0.0]]
    assert tree_max_error(baseline, same) == 0
    assert tree_max_error(baseline, wrong_flag) == 1
    with pytest.raises(ValueError, match='Unsupported cache leaf'):
        tree_max_error(baseline, [[torch.ones(1), None, object()], [torch.zeros(1), None, 0.0]])
