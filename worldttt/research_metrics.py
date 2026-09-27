"""Scene-level statistics for the WorldTTT research protocol.

The native SANA evaluator emits frame/pair rows.  This module intentionally
aggregates by scene first, then bootstraps paired scene deltas; otherwise a
scene with many revisit pairs would be counted as many independent samples.
It has no dependency on the SANA model and can be used to audit result JSON
before a paper table is generated.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def r2m(lpips_return, lpips_control, eps=1e-8):
    """Relative revisit gain; positive means return is better than control."""
    return 1.0 - float(lpips_return) / (float(lpips_control) + eps)


def paired_bootstrap(reference, online, *, seed=3407, samples=10000):
    """Return a paired mean delta and percentile CI over common scene ids."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    common = sorted(set(reference) & set(online))
    if not common:
        return {"scenes": [], "mean": None, "ci95": None, "missing_reference": sorted(set(online) - set(reference)),
                "missing_online": sorted(set(reference) - set(online))}
    delta = np.asarray([float(online[s]) - float(reference[s]) for s in common], dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("paired metrics contain nonfinite values")
    rng = np.random.default_rng(seed)
    draw = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(1)
    return {"scenes": common, "mean": float(delta.mean()),
            "ci95": [float(x) for x in np.quantile(draw, [.025, .975])],
            "missing_reference": sorted(set(online) - set(reference)),
            "missing_online": sorted(set(reference) - set(online))}


def aggregate_scene_rows(rows, *, scene_key="scene_id", method_key="method"):
    """Mean numeric metrics per scene/method, rejecting invalid rows.

    Missing values are kept missing rather than replaced by zeros.  The caller
    should report coverage separately and pass only metrics measured by the
    same evaluator/version.
    """
    groups = defaultdict(list)
    for row in rows:
        if scene_key not in row or method_key not in row:
            raise ValueError("each metric row needs scene_id and method")
        if row.get("valid", True) is False:
            continue
        groups[(str(row[method_key]), str(row[scene_key]))].append(row)
    result = {}
    for (method, scene), items in groups.items():
        numeric = {}
        names = set().union(*(x.keys() for x in items))
        for name in names:
            values = [x[name] for x in items if isinstance(x.get(name), (int, float))]
            if values:
                values = np.asarray(values, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite metric {name} for {method}/{scene}")
                numeric[name] = float(values.mean())
        result.setdefault(method, {})[scene] = numeric
    return result


def method_delta(scene_metrics, *, reference, online, metric):
    ref = {scene: values[metric] for scene, values in scene_metrics.get(reference, {}).items()
           if metric in values}
    new = {scene: values[metric] for scene, values in scene_metrics.get(online, {}).items()
           if metric in values}
    return paired_bootstrap(ref, new)


def retention_curve(rows, *, method, score_key="score", distractor_key="distractor_chunks"):
    """Scene-mean retention by distractor count for a no-reset stream."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("method") != method or row.get("valid", True) is False:
            continue
        if distractor_key not in row or score_key not in row:
            raise ValueError("retention row missing distractor/score")
        groups[int(row[distractor_key])].append((str(row["scene_id"]), float(row[score_key])))
    output = []
    for distractors, values in sorted(groups.items()):
        by_scene = defaultdict(list)
        for scene, score in values:
            by_scene[scene].append(score)
        scene_means = np.asarray([np.mean(v) for v in by_scene.values()], dtype=np.float64)
        if not np.isfinite(scene_means).all():
            raise ValueError("nonfinite retention score")
        output.append({"distractor_chunks": distractors, "scene_count": int(len(scene_means)),
                       "mean": float(scene_means.mean())})
    return output


def validate_worldttt_protocol(meta):
    """Reject result metadata that cannot be compared under the shared axes."""
    required = {"reference_fps": 16, "latent_frame_stride": 8,
                "write_path": "completed_clean_sigma0_only"}
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(f"protocol mismatch for {key}: expected {expected!r}")
    if meta.get("query_path") != "future_read_only":
        raise ValueError("future query must be read-only")
    if meta.get("cfg_branches_independent") is not True:
        raise ValueError("CFG branches must have independent memory state")
    return True
