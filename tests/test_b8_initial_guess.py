"""Task B8 (Addendum v2.1): the Buckley-Leverett initial-guess option and
the sensitivity driver."""

import csv
import dataclasses

import numpy as np
import pytest

import experiments.b8_initial_guess as b8
from problems.buckley_leverett import BLConfig, BLPhysics


@pytest.mark.parametrize("gravity", [False, True])
def test_initial_guess_modes(gravity):
    base = BLConfig.with_gravity() if gravity else BLConfig()
    x, t = np.linspace(0, 1, 7), np.full(7, 0.1)
    ic = BLPhysics(base).initial_condition(x)
    assert np.array_equal(BLPhysics(dataclasses.replace(base, initial_guess='ic')).initial_guess(x, t), ic)
    assert not BLPhysics(dataclasses.replace(base, initial_guess='zero')).initial_guess(x, t).any()
    # Default: the paper's choice -- zero with gravity, the profile without.
    default = BLPhysics(base)
    assert default.initial_guess_mode == ('zero' if gravity else 'ic')
    with pytest.raises(ValueError):
        BLPhysics(dataclasses.replace(base, initial_guess='linear'))


def test_planned_runs_are_the_128_of_the_addendum():
    runs = list(b8.planned_runs())
    assert len(runs) == 2 * 2 * 4 * (1 + 1 + 3 + 3) == 128
    assert len(set(runs)) == 128


def test_driver_writes_resumable_rows_with_lil_q_monitors(tmp_path):
    selection = dict(cases=('gravity',), guesses=('zero', 'ic'), sizes=(8,),
                     methods=('LiL-Q', 'NiL-N'), seeds=(0,))
    path = b8.run_b8(tmp_path, quick=True, verbose=False, **selection)
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and set(rows[0]) == set(b8.COLUMNS)
    assert all(r['stopping_reason'] in ('target', 'iteration_cap', 'line_search_cap') for r in rows)
    lilq = [r for r in rows if r['method'] == 'LiL-Q']
    assert all(len(r['chi_history'].split(';')) == int(r['iterations']) for r in lilq)
    assert (tmp_path / 'lilq_logs' / 'gravity_ic_P64' / 'iterations.csv').exists()
    b8.run_b8(tmp_path, quick=True, verbose=False, **selection)   # resume: nothing new
    with open(path, newline='') as f:
        assert len(list(csv.DictReader(f))) == 4
