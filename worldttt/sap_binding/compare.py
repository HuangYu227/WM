"""Compare four structure runs and enforce the hybrid improvement gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(hybrid, bank, fast, unconstrained, output):
    reports = {name: json.loads(Path(path).read_text(encoding='utf-8'))
               for name, path in [('hybrid', hybrid), ('bank_only', bank),
                                  ('fast_only', fast), ('no_constraints', unconstrained)]}
    for name, report in reports.items():
        expected = 'hybrid' if name in {'hybrid', 'no_constraints'} else name
        if (report.get('kind') != 'sap_binding_causal_gate' or report.get('split') != 'val' or
                report.get('architecture') != expected):
            raise ValueError(f'{name} is not a matching validation causal report')
    def scenes(report):
        return {(row['scene_id'], row['noise_sigma']) for row in report['rows']
                if row['kind'] == 'interference' and row['distractor_writes'] == 8 and
                row['control'] == 'online'}
    paired = [scenes(report) for report in reports.values()]
    if not paired[0] or any(group != paired[0] for group in paired[1:]):
        raise ValueError('Structure reports do not evaluate paired validation scenes')
    def score(report):
        rows = [r for r in report['rows'] if r['kind'] == 'interference' and
                r['distractor_writes'] == 8 and r['control'] == 'online']
        if not rows: raise ValueError('Structure report lacks eight-distractor online rows')
        return sum(r['query_mse'] for r in rows) / len(rows)
    scores = {name: score(report) for name, report in reports.items()}
    best_single = min(scores['bank_only'], scores['fast_only'])
    hybrid_gain = (best_single - scores['hybrid']) / max(best_single, 1e-12)
    result = dict(scores=scores, adapters={name: report.get('adapter') for name, report in reports.items()},
        hybrid_gain_over_best_single=hybrid_gain,
        hybrid_path_gate=hybrid_gain >= .05,
        causal_gate_passed=bool(reports['hybrid']['gates']['passed']),
        flow_allowed=hybrid_gain >= .05 and bool(reports['hybrid']['gates']['passed']))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('hybrid', 'bank', 'fast', 'unconstrained', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), required=True)
    args = parser.parse_args(); print(json.dumps(compare(args.hybrid, args.bank, args.fast,
                                                         args.unconstrained, args.output), indent=2))


if __name__ == '__main__':
    main()
