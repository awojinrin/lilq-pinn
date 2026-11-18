"""
Package 5, item 3: the time to reach an error
=============================================

Section 6, "Report": "A table of the time to reach the errors 1e-2, 1e-3,
1e-4, 1e-5, 1e-6 for every method, problem and device (the smallest time over
the sizes of that method; median over seeds; a dash where no size reaches the
level within the budget)." From the history files of the network runs
(``p5_3_networks``) and the LiL runs (``p5_3_lil``) under ``<out>/P5_3_cost/``.

**Per run and level:** the time of the first logged iterate whose error is at
most the level (``time_s``, the run's clock), or "not reached". L-BFGS logs the
error every 10 iterations, so its time is that of the first logged iterate at
or below the level (resolution 10 iterations); Levenberg-Marquardt, ``trf``,
LiL-Q and the square collocation log every iterate.

**Per size:** the median over the seeds, a seed that does not reach the level
counting as infinitely late, so the median exists only when at least two of
three seeds reach it. The LiL methods are deterministic: one run per size.

**Per method, problem, device and level:** the smallest of those over the
sizes, and the size that gives it; a dash where no size reaches the level.

**Beside the clock** (the question of what the clock covers, put to the
advisor): the same time plus the run's setup and pretraining (``setup_s``,
``pretrain_s``), median over the seeds in the same way (``with_setup_s``).
Either reading is then in the table without a rerun.

**Flags.** A run whose history has no time (a timed pass that did not
reproduce the logged pass), or a LiL run made without the protocol
(``--no-timing``: ``timed`` false), is left out of the table and listed in
``flagged.csv``; so is a run that ended in ``failure`` (non-finite loss): its
history up to the failure still counts.

**Outputs** under ``<out>/P5_3_cost/``: ``time_to_error_runs.csv`` (every
run, every level), ``time_to_error.csv`` (the table, long form),
``time_to_error.md`` (one table per problem and device), ``flagged.csv``.

Usage::

    python experiments/p5_3_table.py --out <root>
"""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from experiments.p4_common import read_csv, write_csv

ITEM = 'P5_3_cost'
LEVELS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
BENCHES = ('bratu', 'burgers', 'bl')
BENCH_LABELS = {'bratu': 'Bratu', 'burgers': 'Burgers', 'bl': 'Buckley-Leverett (viscous)'}
METHOD_ORDER = ('nil_n', 'nil_q', 'lm', 'lbfgs', 'trf', 'lilq', 'square')
MIN_SEEDS = 2                    # of three: the median exists only if at least two reach the level


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def time_to(rows, level):
    """``(time_s, iteration)`` of the first logged iterate with error <= level, or None."""
    for r in rows:
        e = _f(r['error'])
        if math.isfinite(e) and e <= level:
            return _f(r['time_s']), int(r['iteration'])
    return None


def runs(out_root):
    """Every run under ``<out>/P5_3_cost/{networks,lil}``: its record and history."""
    base = Path(out_root) / ITEM
    for kind in ('networks', 'lil'):
        for rj in sorted((base / kind).glob('*_P*/*/run.json')):
            rec = json.loads(rj.read_text())
            hist = rj.parent / 'history.csv'
            if hist.exists():
                yield kind, rj.parent, rec, read_csv(hist)


def per_run(out_root, levels=LEVELS):
    """One row per run and level, and the flagged runs."""
    out, flagged = [], []
    for kind, d, rec, hist in runs(out_root):
        untimed = any(not math.isfinite(_f(r['time_s'])) for r in hist)
        if untimed or rec.get('timing_matches') is False or rec.get('timed') is False:
            flagged.append({'run': str(d.relative_to(Path(out_root))), 'reason': 'no time (timed passes did not '
                            'reproduce the logged pass, or an untimed run)'})
            continue
        if rec.get('stopping_reason') == 'failure':
            flagged.append({'run': str(d.relative_to(Path(out_root))), 'reason': 'failure (non-finite loss); '
                            'its history up to the failure is used'})
        extra = (_f(rec.get('setup_s')) if rec.get('setup_s') is not None else 0.0) + \
            (_f(rec.get('pretrain_s')) if rec.get('pretrain_s') is not None else 0.0)
        for level in levels:
            hit = time_to(hist, level)
            out.append({'benchmark': rec['benchmark'], 'method_key': rec['method_key'], 'method': rec['method'],
                        'device': rec['device'], 'P': int(rec['P']), 'seed': rec.get('seed', ''), 'level': level,
                        'reached': hit is not None, 'time_s': hit[0] if hit else '',
                        'iteration': hit[1] if hit else '', 'setup_and_pretraining_s': extra,
                        'with_setup_s': hit[0] + extra if hit else '',
                        'stopping_reason': rec.get('stopping_reason'), 'final_error': rec.get('final_error'),
                        'run': str(d.relative_to(Path(out_root)))})
    return out, flagged


def _median_reached(values, n_runs):
    """The median over the runs, a run that does not reach counting as infinitely late."""
    vals = sorted(values) + [math.inf] * (n_runs - len(values))
    m = statistics.median(vals) if vals else math.inf
    return m if math.isfinite(m) else None


def table(rows, levels=LEVELS):
    """The long table: per benchmark, method, device and level, the smallest
    over the sizes of the median over the seeds."""
    groups = {}
    for r in rows:
        groups.setdefault((r['benchmark'], r['method_key'], r['device'], r['level']), []).append(r)
    out = []
    for (bench, key, device, level), rs in sorted(groups.items(), key=lambda kv: (
            BENCHES.index(kv[0][0]) if kv[0][0] in BENCHES else 99,
            METHOD_ORDER.index(kv[0][1]) if kv[0][1] in METHOD_ORDER else 99, kv[0][2], -kv[0][3])):
        by_size = {}
        for r in rs:
            by_size.setdefault(r['P'], []).append(r)
        best = None
        per_size = {}
        for P, seeds in sorted(by_size.items()):
            n = len(seeds)
            hit = [s for s in seeds if s['reached']]
            t = _median_reached([s['time_s'] for s in hit], n)
            tw = _median_reached([s['with_setup_s'] for s in hit], n)
            per_size[P] = {'runs': n, 'reached': len(hit), 'time_s': t, 'with_setup_s': tw}
            if t is not None and (n == 1 or len(hit) >= MIN_SEEDS) and (best is None or t < best[1]):
                best = (P, t, tw, len(hit), n)
        best_w = min(((P, v['with_setup_s']) for P, v in per_size.items()
                      if v['with_setup_s'] is not None and (v['runs'] == 1 or v['reached'] >= MIN_SEEDS)),
                     key=lambda x: x[1], default=None)
        out.append({'benchmark': bench, 'method_key': key, 'method': rs[0]['method'], 'device': device,
                    'level': level, 'time_s': best[1] if best else '', 'P': best[0] if best else '',
                    'seeds_reached': f'{best[3]}/{best[4]}' if best else '',
                    'with_setup_s': best_w[1] if best_w else '', 'P_with_setup': best_w[0] if best_w else '',
                    'sizes': json.dumps({str(P): v for P, v in per_size.items()})})
    return out


def _fmt(t):
    if t in ('', None):
        return '-'
    return f'{t:.3g}'


def markdown(long_rows, levels=LEVELS):
    lines = ['# Package 5, item 3: time to reach an error', '',
             'Seconds from the starting data to the iterate: the run\'s clock plus its setup and pretraining '
             '(the networks\' pretraining; the LiL methods\' system and initial coefficients, on '
             'Buckley-Leverett the fit of the initial profile), as the advisor\'s reply of 17 November 2026 '
             'asks; the smallest over the sizes of the median over the seeds, and the size (P) that gives it. '
             'Then the optimizer alone (the clock from the initial coefficients), with its size if another one '
             'gives its smallest. A dash where no size reaches the level.', '']
    keys = sorted({(r['benchmark'], r['device']) for r in long_rows},
                  key=lambda k: (BENCHES.index(k[0]) if k[0] in BENCHES else 99, k[1]))
    for bench, device in keys:
        lines += [f'## {BENCH_LABELS.get(bench, bench)}, {device}', '',
                  '| method | ' + ' | '.join(f'{lv:g}' for lv in levels) + ' |',
                  '|---|' + '---|' * len(levels)]
        methods = sorted({r['method_key'] for r in long_rows if (r['benchmark'], r['device']) == (bench, device)},
                         key=lambda k: METHOD_ORDER.index(k) if k in METHOD_ORDER else 99)
        for key in methods:
            cells, label = [], key
            for lv in levels:
                r = next((x for x in long_rows if (x['benchmark'], x['device'], x['method_key'], x['level']) ==
                          (bench, device, key, lv)), None)
                if r is None or r['time_s'] == '':
                    cells.append('-')
                else:
                    label = r['method']
                    other = '' if str(r['P']) == str(r['P_with_setup']) else f" (P = {r['P']})"
                    cells.append(f"{_fmt(r['with_setup_s'])} (P = {r['P_with_setup']}); optimizer only "
                                 f"{_fmt(r['time_s'])}{other}")
            if label == key:
                label = next((x['method'] for x in long_rows if x['method_key'] == key), key)
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines) + '\n'


def summarize(out_root, levels=LEVELS):
    rows, flagged = per_run(out_root, levels)
    base = Path(out_root) / ITEM
    base.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(base / 'time_to_error_runs.csv', rows)
    long_rows = table(rows, levels)
    if long_rows:
        write_csv(base / 'time_to_error.csv', long_rows)
    (base / 'time_to_error.md').write_text(markdown(long_rows, levels), encoding='utf-8')
    write_csv(base / 'flagged.csv', flagged or [{'run': '', 'reason': ''}])
    return rows, long_rows, flagged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rows, long_rows, flagged = summarize(args.out)
    print(f'{len(rows)} run-level rows, {len(long_rows)} table rows, {len(flagged)} flagged')


if __name__ == '__main__':
    main()
