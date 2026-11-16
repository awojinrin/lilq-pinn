"""Package 5, item 3: the time-to-error table (``experiments/p5_3_table.py``), on synthetic runs."""

import json
import math

import pytest

import experiments.p5_3_table as t
from experiments.p5_3_networks import write_history


def make_run(root, kind, bench, P, key, device, errors, times, seed=None, setup=0.5, pretrain=None, **extra):
    name = f'{key}_s{seed}_{device}' if seed is not None else f'{key}_{device}'
    d = root / t.ITEM / kind / f'{bench}_P{P}' / name
    d.mkdir(parents=True)
    write_history(d / 'history.csv', [{'iteration': 10 * i, 'evaluations': i, 'time_s': s, 'loss': e * e,
                                       'error': e} for i, (e, s) in enumerate(zip(errors, times))])
    rec = {'benchmark': bench, 'P': P, 'method_key': key, 'method': key.upper(), 'device': device,
           'setup_s': setup, 'stopping_reason': 'budget', 'final_error': errors[-1], **extra}
    if seed is not None:
        rec['seed'] = seed
    if pretrain is not None:
        rec['pretrain_s'] = pretrain
    (d / 'run.json').write_text(json.dumps(rec))
    return d


def test_time_to():
    rows = [{'iteration': 0, 'time_s': 0.0, 'error': 1.0}, {'iteration': 10, 'time_s': 2.0, 'error': 5e-3},
            {'iteration': 20, 'time_s': 4.0, 'error': 1e-3}]
    assert t.time_to(rows, 1e-2) == (2.0, 10)
    assert t.time_to(rows, 1e-3) == (4.0, 20)                     # at the level counts
    assert t.time_to(rows, 1e-4) is None


def test_the_median_over_seeds_needs_two_of_three(tmp_path):
    # P = 25: seeds reach 1e-2 at 1, 3 and never -> median 3; 1e-3 only one seed -> not at this size
    make_run(tmp_path, 'networks', 'bratu', 25, 'nil_n', 'cpu', [1, 5e-3, 5e-4], [0, 1, 2], seed=0, pretrain=0.25)
    make_run(tmp_path, 'networks', 'bratu', 25, 'nil_n', 'cpu', [1, 1e-1, 5e-3], [0, 1, 3], seed=1, pretrain=0.25)
    make_run(tmp_path, 'networks', 'bratu', 25, 'nil_n', 'cpu', [1, 1e-1, 5e-2], [0, 1, 9], seed=2, pretrain=0.25)
    rows, long_rows, flagged = t.summarize(tmp_path)
    get = {r['level']: r for r in long_rows if r['method_key'] == 'nil_n'}
    assert get[1e-2]['time_s'] == 3 and get[1e-2]['P'] == 25 and get[1e-2]['seeds_reached'] == '2/3'
    assert get[1e-2]['with_setup_s'] == pytest.approx(3.75)       # + setup 0.5 + pretraining 0.25
    assert get[1e-3]['time_s'] == '' and not flagged


def test_the_smallest_over_the_sizes(tmp_path):
    make_run(tmp_path, 'lil', 'bl', 64, 'lilq', 'cpu', [1, 2e-2, 8e-3], [0, 0.1, 0.2])
    make_run(tmp_path, 'lil', 'bl', 256, 'lilq', 'cpu', [1, 5e-3, 5e-4], [0, 0.4, 0.8], setup=0.01)
    _, long_rows, _ = t.summarize(tmp_path)
    get = {r['level']: r for r in long_rows}
    assert (get[1e-2]['time_s'], get[1e-2]['P']) == (0.2, 64)
    assert (get[1e-3]['time_s'], get[1e-3]['P']) == (0.8, 256)
    # with setup, the smaller size's 0.2 + 0.5 is later than the larger's 0.4 + 0.01
    assert (get[1e-2]['with_setup_s'], get[1e-2]['P_with_setup']) == (pytest.approx(0.41), 256)


def test_runs_without_a_time_are_flagged_and_left_out(tmp_path):
    make_run(tmp_path, 'lil', 'burgers', 36, 'trf', 'cpu', [1, 1e-3], [0, math.nan])
    make_run(tmp_path, 'lil', 'burgers', 36, 'lilq', 'cpu', [1, 1e-3], [0, 0.5], timing_matches=False)
    make_run(tmp_path, 'lil', 'burgers', 100, 'lilq', 'cpu', [1, 1e-3], [0, 0.5], timed=False)
    rows, long_rows, flagged = t.summarize(tmp_path)
    assert not rows and not long_rows and len(flagged) == 3


def test_the_markdown_table(tmp_path):
    make_run(tmp_path, 'lil', 'bratu', 25, 'lilq', 'cpu', [1, 5e-3], [0, 0.01])
    t.summarize(tmp_path)
    md = (tmp_path / t.ITEM / 'time_to_error.md').read_text(encoding='utf-8')
    assert '## Bratu, cpu' in md and '| LILQ | 0.01 (0.51), P = 25 | - | - | - | - |' in md
