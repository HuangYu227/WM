from worldttt.sap_binding.causal_eval import summarize_gates


def _rows(online=.07, shuffled=.10, mean=.09, exact=.2, recall64=.2, recall512=.6):
    rows = []
    for distractors in (2, 8):
        for control, mse in [('online', online), ('shuffle_key', shuffled),
                             ('shuffle_value', shuffled), ('mean_value', mean)]:
            rows.append(dict(kind='interference', distractor_writes=distractors,
                             support_tokens=256, control=control, query_mse=mse,
                             exact_top1=exact, coverage=.4, matched_queries=64,
                             candidates=256))
    for budget, recall in [(64, recall64), (128, .3), (256, .4), (512, recall512)]:
        rows.append(dict(kind='budget', support_tokens=budget, control='online',
                         query_mse=online, coverage=recall, common_queries=8))
    return rows


def test_causal_gate_requires_correct_binding_at_normal_and_eight_writes():
    assert summarize_gates(_rows())['passed']
    rows = _rows()
    for row in rows:
        if row['kind'] == 'interference' and row['distractor_writes'] == 8 and row['control'] == 'shuffle_key':
            row['query_mse'] = .07
    assert not summarize_gates(rows)['passed']
    rows = _rows()
    for row in rows:
        if row['kind'] == 'interference' and row['distractor_writes'] == 2 and row['control'] == 'mean_value':
            row['query_mse'] = .07
    assert not summarize_gates(rows)['passed']


def test_causal_gate_requires_coverage_growth_and_complete_evidence():
    assert not summarize_gates(_rows(recall64=.6, recall512=.6))['passed']
    assert not summarize_gates(_rows()[:-1])['passed']
