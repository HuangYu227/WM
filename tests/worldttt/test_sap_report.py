import json


def test_sap_report_pairs_identical_cases_and_checks_online_commits(tmp_path):
    from worldttt.sap_ttt.report import build_report

    case = {'id': 'room_1', 'seed': 42, 'num_frames': 961}
    modes = ('off', 'sap_frozen', 'sap_online')
    runs = []
    for mode, lpips in zip(modes, (.8, .9, .7)):
        row = {'experiment': mode, 'id': 'room_1',
               'metrics': {'case': case, 'base_checkpoint': 'base',
                           'adapter': None if mode == 'off' else 'sap.pt',
                           'generation': {'num_frame_per_block': 3},
                           'stage1_seconds': 10.},
               'updates': [{'chunk': n, 'committed': mode == 'sap_online'} for n in range(40)]}
        runs.append(row)
        folder = tmp_path / mode / 'runs' / 'room_1'
        folder.mkdir(parents=True)
        payload = {'video_validation': {'complete': True, 'decoded_frames': 961},
                   'rows': [{'revisit': [276, 796], 'control': [100, 620],
                             'revisit_valid': True, 'valid': True,
                             'revisit_metrics': {'lpips': lpips, 'psnr': 10., 'ssim': .2},
                             'control_metrics': {'lpips': .85, 'psnr': 9., 'ssim': .1},
                             'nonreturn_segment': {'frame_change_mean': 12.,
                                                   'laplacian_variance_mean': 100.}}]}
        (folder / 'relative_revisit.json').write_text(json.dumps(payload))
    (tmp_path / 'comparison.json').write_text(json.dumps({'runs': runs}))
    report = build_report(tmp_path)
    assert report['coverage']['room_1|42']['revisit_pairs'] == 1
    assert report['paired']['sap_online_vs_sap_frozen']['lpips']['mean_online_minus_reference'] < 0
    assert report['online_updates_committed']
    assert report['camera_status'] == 'unverified'
