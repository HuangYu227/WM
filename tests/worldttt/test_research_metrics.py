import pytest

np = pytest.importorskip("numpy")


def test_paired_bootstrap_uses_common_scene_ids_and_reports_missing():
    from worldttt.research_metrics import paired_bootstrap
    result = paired_bootstrap({"a": 1., "b": 2.}, {"a": 1.5, "c": 9.})
    assert result["scenes"] == ["a"]
    assert result["mean"] == pytest.approx(.5)
    assert result["missing_online"] == ["b"]
    assert result["missing_reference"] == ["c"]


def test_scene_aggregation_drops_invalid_rows_without_zero_imputation():
    from worldttt.research_metrics import aggregate_scene_rows, method_delta, r2m
    rows = [
        {"scene_id": "a", "method": "off", "lpips": .8},
        {"scene_id": "a", "method": "off", "lpips": .6},
        {"scene_id": "a", "method": "online", "lpips": .5},
        {"scene_id": "b", "method": "off", "lpips": .7, "valid": False},
    ]
    grouped = aggregate_scene_rows(rows)
    assert grouped["off"]["a"]["lpips"] == pytest.approx(.7)
    delta = method_delta(grouped, reference="off", online="online", metric="lpips")
    assert delta["mean"] == pytest.approx(-.2)
    assert r2m(.5, .7) > 0


def test_protocol_rejects_future_write_and_wrong_coordinate_stride():
    from worldttt.research_metrics import validate_worldttt_protocol
    good = {"reference_fps": 16, "latent_frame_stride": 8,
            "write_path": "completed_clean_sigma0_only", "query_path": "future_read_only",
            "cfg_branches_independent": True}
    assert validate_worldttt_protocol(good)
    with pytest.raises(ValueError):
        validate_worldttt_protocol({**good, "query_path": "future_write"})
