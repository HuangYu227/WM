import json

import pytest


def test_l20_report_pairs_modes_and_preserves_missing_control(tmp_path):
    from worldttt.l20_revisit_report import build_report

    modes = {'off': .5, 'frozen_noise': .4, 'noise_ttt': .3}
    runs = []
    for mode, lpips in modes.items():
        run = tmp_path / mode / 'runs' / 'scene_001'
        run.mkdir(parents=True)
        rows = [
            dict(revisit=[16, 320], control=[32, 336], revisit_valid=True, valid=True,
                 revisit_metrics=dict(lpips=lpips, psnr=20., ssim=.7),
                 control_metrics=dict(lpips=.8, psnr=10., ssim=.2)),
            dict(revisit=[400, 720], control=None, revisit_valid=True, valid=False,
                 revisit_metrics=dict(lpips=lpips + .1, psnr=18., ssim=.6),
                 control_metrics=None, reason='no_pose_distinct_gap_match_or_short_video'),
        ]
        (run / 'relative_revisit.json').write_text(json.dumps(dict(rows=rows)))
        updates = [dict(chunk=0, committed=True)] if mode == 'noise_ttt' else []
        runs.append(dict(experiment=mode, id='scene_001',
                         metrics=dict(case=dict(id='scene_001', seed=42), stage1_seconds=5.,
                                      peak_allocated_bytes=100), updates=updates))
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(runs=runs)))

    report = build_report([tmp_path])
    assert report['coverage']['scene_001|42']['revisit_pairs'] == 2
    assert report['coverage']['scene_001|42']['control_pairs'] == 1
    assert report['coverage']['scene_001|42']['missing_controls'][0]['pair'] == [400, 720]
    assert report['scene_scores']['noise_ttt']['lpips']['scene_001'] == pytest.approx(.35)
    assert report['scene_scores']['noise_ttt']['control_lpips']['scene_001'] == pytest.approx(.8)
    assert report['paired']['noise_ttt_vs_frozen_noise']['lpips']['mean_online_minus_reference'] == pytest.approx(-.1)
    assert report['paired']['noise_ttt_vs_off']['lpips']['mean_online_minus_reference'] == pytest.approx(-.2)
    assert report['camera_status'] == 'unverified'
    assert report['quality_motion_status'] == 'unverified'
    assert report['updates']['scene_001|42']['noise_ttt']['committed'] == 1
    assert report['pairs']['scene_001|42']['noise_ttt'][1]['control_metrics'] is None


def test_l20_report_rejects_different_preselected_pairs(tmp_path):
    from worldttt.l20_revisit_report import build_report

    runs = []
    for mode, pair in [('off', [16, 320]), ('frozen_noise', [16, 320]), ('noise_ttt', [16, 400])]:
        run = tmp_path / mode / 'runs' / 'scene'
        run.mkdir(parents=True)
        (run / 'relative_revisit.json').write_text(json.dumps(dict(rows=[dict(
            revisit=pair, revisit_valid=True, valid=False,
            revisit_metrics=dict(lpips=.2, psnr=20., ssim=.8))])))
        runs.append(dict(experiment=mode, id='scene', metrics=dict(case=dict(id='scene', seed=42)), updates=[]))
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(runs=runs)))
    with pytest.raises(ValueError, match='Preselected revisit pairs differ'):
        build_report([tmp_path])


def test_l20_report_never_pairs_different_seeds_of_same_scene(tmp_path):
    from worldttt.l20_revisit_report import build_report

    roots = []
    for mode, seed in [('off', 42), ('frozen_noise', 42), ('noise_ttt', 3407)]:
        root = tmp_path / mode
        run = root / mode / 'runs' / 'scene'
        run.mkdir(parents=True)
        (run / 'relative_revisit.json').write_text(json.dumps(dict(rows=[dict(
            revisit=[16, 320], revisit_valid=True, valid=False,
            revisit_metrics=dict(lpips=.2, psnr=20., ssim=.8))])))
        (root / 'comparison.json').write_text(json.dumps(dict(runs=[dict(
            experiment=mode, id='scene', metrics=dict(case=dict(id='scene', seed=seed)), updates=[])])))
        roots.append(root)
    report = build_report(roots)
    assert report['paired']['noise_ttt_vs_frozen_noise']['lpips']['mean_online_minus_reference'] is None
    assert report['pilot_valid'] is False


def test_l20_report_rejects_mismatched_generation_settings(tmp_path):
    from worldttt.l20_revisit_report import build_report

    runs = []
    for mode, steps in [('off', 50), ('frozen_noise', 50), ('noise_ttt', 4)]:
        run = tmp_path / mode / 'runs' / 'scene'
        run.mkdir(parents=True)
        (run / 'relative_revisit.json').write_text(json.dumps(dict(rows=[dict(
            revisit=[16, 320], revisit_valid=True, valid=False,
            revisit_metrics=dict(lpips=.2, psnr=20., ssim=.8))])))
        runs.append(dict(experiment=mode, id='scene', metrics=dict(
            case=dict(id='scene', seed=42), generation=dict(step=steps)), updates=[]))
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(runs=runs)))
    with pytest.raises(ValueError, match='Generation settings differ'):
        build_report([tmp_path])


def test_l20_report_preserves_camera_checks_for_both_seeds(tmp_path):
    from worldttt.l20_revisit_report import build_report

    roots = []
    for seed in (42, 3407):
        root = tmp_path / str(seed)
        runs = []
        for mode in ('off', 'frozen_noise', 'noise_ttt'):
            run = root / mode / 'runs' / 'scene'
            run.mkdir(parents=True)
            (run / 'relative_revisit.json').write_text(json.dumps(dict(rows=[dict(
                revisit=[16, 320], revisit_valid=True, valid=False,
                revisit_metrics=dict(lpips=.2, psnr=20., ssim=.8))])))
            pose = root / mode / 'simple_60s'
            pose.mkdir()
            (pose / 'eval_poses.json').write_text(json.dumps(dict(scene=dict(
                RotErr=float(seed), TransErr_rel=1., CamMC_rel=1.))))
            runs.append(dict(experiment=mode, id='scene', metrics=dict(
                case=dict(id='scene', seed=seed)),
                updates=[dict(committed=True)] if mode == 'noise_ttt' else []))
        (root / 'comparison.json').write_text(json.dumps(dict(runs=runs)))
        roots.append(root)
    report = build_report(roots)
    assert report['camera_status'] == 'verified'
    assert report['camera']['noise_ttt']['scene|42']['RotErr'] == 42.
    assert report['camera']['noise_ttt']['scene|3407']['RotErr'] == 3407.


def test_l20_full_pilot_requires_complete_video_and_every_chunk_update(tmp_path):
    from worldttt.l20_revisit_report import build_report

    runs = []
    for mode in ('off', 'frozen_noise', 'noise_ttt'):
        run = tmp_path / mode / 'runs' / 'scene'
        run.mkdir(parents=True)
        row = dict(revisit=[16, 320], control=[32, 336], revisit_valid=True, valid=True,
                   revisit_metrics=dict(lpips=.2, psnr=20., ssim=.8),
                   control_metrics=dict(lpips=.8, psnr=10., ssim=.2),
                   nonreturn_segment=dict(frame_change_mean=3., laplacian_variance_mean=100.))
        (run / 'relative_revisit.json').write_text(json.dumps(dict(
            rows=[row], video_validation=dict(decoded_frames=961, expected_frames=961,
                                              complete=True, fps=16.))))
        runs.append(dict(experiment=mode, id='scene', metrics=dict(
            case=dict(id='scene', seed=42, num_frames=961),
            generation=dict(num_frame_per_block=3)),
            updates=[dict(chunk=i, committed=True) for i in range(40)] if mode == 'noise_ttt' else []))
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(runs=runs)))
    assert build_report([tmp_path])['pilot_valid'] is True
    runs[-1]['updates'] = [dict(chunk=0, committed=True)]
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(runs=runs)))
    report = build_report([tmp_path])
    assert report['pilot_valid'] is False
    assert report['online_updates_committed'] is False
    runs[-1]['updates'] = [dict(chunk=i, committed=True) for i in range(40)]
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(runs=runs)))
    for mode in ('off', 'frozen_noise', 'noise_ttt'):
        path = tmp_path / mode / 'runs' / 'scene' / 'relative_revisit.json'
        result = json.loads(path.read_text())
        result['video_validation']['decoded_frames'] = 960
        result['video_validation']['complete'] = False
        path.write_text(json.dumps(result))
    assert build_report([tmp_path])['pilot_valid'] is False
    for entry in runs:
        entry['metrics']['case']['num_frames'] = 353
    runs[-1]['updates'] = [dict(chunk=i, committed=True) for i in range(15)]
    (tmp_path / 'comparison.json').write_text(json.dumps(dict(runs=runs)))
    for mode in ('off', 'frozen_noise', 'noise_ttt'):
        path = tmp_path / mode / 'runs' / 'scene' / 'relative_revisit.json'
        result = json.loads(path.read_text())
        result['video_validation'].update(decoded_frames=353, expected_frames=353, complete=True)
        path.write_text(json.dumps(result))
    report = build_report([tmp_path])
    assert report['exploratory_valid'] is True
    assert report['pilot_valid'] is False
