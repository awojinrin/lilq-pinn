"""Regression guard for Kovasznay's conditioning-analysis flag -- same
pattern as test_beltrami_conditioning_flag.py. solve_kovasznay used to
compute np.linalg.cond(A_sys) (a full SVD) every outer iteration with
nothing consuming it, costing 1.3-3.6x the whole solve (see DECISIONS.md).
"""

import math

import numpy as np

from problems.kovasznay import KovasznayConfig, solve_kovasznay


def _tiny_config():
    return KovasznayConfig(N_x=4, N_y=4, max_iter=2)


def test_conditioning_not_computed_by_default():
    result = solve_kovasznay(_tiny_config(), verbose=False)
    assert all(math.isnan(c) for c in result['history']['cond_number'])


def test_conditioning_computed_when_requested():
    result = solve_kovasznay(_tiny_config(), verbose=False, analyze_conditioning=True)
    cond_values = result['history']['cond_number']
    assert all(math.isfinite(c) and c > 0 for c in cond_values)


def test_flag_does_not_change_the_solve():
    plain = solve_kovasznay(_tiny_config(), verbose=False)
    with_cond = solve_kovasznay(_tiny_config(), verbose=False, analyze_conditioning=True)
    for key in ('theta_u', 'theta_v', 'theta_p'):
        assert np.array_equal(plain[key], with_cond[key])
