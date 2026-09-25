"""Component C (Package 1 v2.0 Section 5): the sweep's row counts, run
count, point distributions, and a resumable smoke run."""

import csv

import numpy as np
import pytest

import experiments.component_c as cc
from lilq.collocation import generate_collocation_points_2d, points_1d
from problems.kovasznay import KovasznayConfig, _generate_collocation


@pytest.mark.parametrize("k", [0.3, 1.0, 2.7, 10.0, 25.0])
@pytest.mark.parametrize("floor", [1, None])
def test_row_count_formula_matches_the_generators(k, floor):
    """floor=None: the paper's minimums; 1: the sweep's."""
    for N in (10, 20):
        kw = {} if floor is None else {'collocation_floor': floor}
        pts = _generate_collocation(KovasznayConfig(N_x=N, N_y=N, k_ratio=k, **kw), 3 * N * N)
        assert cc.n_rows('kovasznay', N, k, floor=floor) == 3 * pts['n_pde'] + 8 * pts['n_bc_edge'] + 1
    for N in (10, 15):
        pts = generate_collocation_points_2d((0, 1), (0, 1), N, N, k_ratio=k, collocation_ratios=(0.85, 0.15),
                                             floor=floor)
        assert cc.n_rows('bratu', N, k, floor=floor) == pts['n_pde'] + 4 * pts['n_bc']


def test_196_runs_and_every_ratio_within_ten_percent():
    assert len(cc.BENCHMARKS) * len(cc.RATIOS) * len(cc.DISTRIBUTIONS) == 196
    for b, N in cc.BENCHMARKS:
        P = cc.n_params(b, N)
        for r in cc.RATIOS:
            assert abs(cc.k_for_ratio(b, N, r)[1] / P / r - 1) < 0.1


def test_cgl_points_cluster_towards_the_ends():
    x = points_1d(0.0, 1.0, 9, 'cgl')
    assert x[0] == 0.0 and x[-1] == 1.0 and np.all(np.diff(x) > 0)
    assert np.diff(x)[0] < np.diff(x)[4]
    assert np.allclose(x, (1 - np.cos(np.pi * np.arange(9) / 8)) / 2)


def test_smoke_sweep_is_resumable(tmp_path):
    kw = dict(benchmarks=(('bratu', 10),), ratios=(1, 3), distributions=(('paper', None), ('random', 0)),
              quick=True, verbose=False)
    path = cc.run_sweep(tmp_path, **kw)
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 4 and all(r['iterations'] for r in rows)
    assert (tmp_path / 'figures' / 'bratu_P100.pdf').exists()
    cc.run_sweep(tmp_path, **kw)
    assert len(list(csv.DictReader(open(path)))) == 4
