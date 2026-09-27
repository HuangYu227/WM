def test_paired_summary_uses_only_common_scenes_and_reports_coverage():
    from worldttt.a800_report import paired_delta
    frozen = {'a': 20., 'b': 30., 'c': 100.}
    online = {'a': 22., 'b': 29.}
    report = paired_delta(frozen, online)
    assert report['scenes'] == ['a', 'b']
    assert report['missing_online'] == ['c']
    assert report['mean_online_minus_reference'] == .5
