"""baselines/square_chebyshev.py: the port of the advisor's Bratu pilot
(Package 2, item 1). On the pilot's own data, experiments/p2_3_classical.py
check-pilot reproduces rows.json at p = 12, 14, 20, 22 to 1e-11 relative
(check C1; DECISIONS.md); these tests check the parts in isolation."""

import numpy as np
import pytest

import baselines.square_chebyshev as sc


@pytest.mark.parametrize('m', [5, 8, 13])
def test_clenshaw_curtis_weights_are_exact_to_degree_m_minus_1(m):
    x, w = sc.cgl(m), sc.cc_weights(m)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    for d in range(m):
        assert w @ x ** d == pytest.approx(1.0 / (d + 1), abs=1e-13)


def test_hard_space_vanishes_on_the_boundary_and_its_laplacian_is_right():
    space = sc.TrialSpace('hard', 4)
    rng = np.random.default_rng(0)
    beta = rng.standard_normal(16)
    s = np.linspace(0, 1, 7)
    for x, y in ((s, 0 * s), (s, 0 * s + 1), (0 * s, s), (0 * s + 1, s)):
        assert np.abs(space.values(beta, x, y)).max() < 1e-14
    x, y, h = np.array([0.3]), np.array([0.6]), 1e-4
    Phi, Lap = space.rows(x, y)
    f = lambda a, b: space.values(beta, np.array([a]), np.array([b]))[0]  # noqa: E731
    fd = (f(0.3 + h, 0.6) + f(0.3 - h, 0.6) + f(0.3, 0.6 + h) + f(0.3, 0.6 - h) - 4 * f(0.3, 0.6)) / h ** 2
    assert (Lap @ beta)[0] == pytest.approx(fd, rel=1e-5)


def test_layouts_match_the_pilot():
    sq = sc.square_layout(7)
    assert (len(sq.xi), len(sq.xb), sq.square) == (25, 24, True)            # (p-2)^2 and 4p-4
    hard = sc.hard_layout(10, 1.5)
    assert len(hard.xi) == 169 and not hard.square                          # m = 15: 13^2 >= 1.5 * 100
    assert hard.wi ** 2 @ np.ones(len(hard.wi)) < 1.0                       # interior CC weights only
    ms = sc.weak_ms_layout(12, 3, lam_bc=10.0)
    assert ms.N == 21 ** 2 and ms.wb[0] == pytest.approx(np.sqrt(10.0 / len(ms.wb)))
    assert sc.free_coefficients('SQ-CGL', 7) == 25 and sc.free_coefficients('LS-hard-CGL-1.5', 5) == 25


def test_newton_converges_and_the_square_system_is_the_n_equals_p_hard_system():
    ax = np.linspace(0, 1, 41)
    ref = sc.reference(24, ax)[2]
    sq = sc.newton('SQ-CGL', 8, reference=(ax, ref))
    assert sq['stop'] == 'tolerance' and sq['k_stop'] <= 8 and sq['rank'] == sq['P_trial']
    assert sq['err_final'] < 1e-2 and sq['history'][sq['k_plateau']]['err'] <= 1.05 * sq['err_final']
    # LS-hard at p = 6 on the interior points of the 8-point CGL grid: the same square system
    space, layout = sc.TrialSpace('hard', 6), sc.hard_layout(6, None, m=8)
    rows, beta = sc.Rows(space, layout), np.zeros(36)
    for _ in range(10):
        A, f, _ = sc.assemble(rows, beta)
        beta = sc.solve_linear(A, f, square=False)
    assert np.abs(space.grid_values(beta, ax) - sq['space'].grid_values(sq['beta'], ax)).max() < 1e-13


def test_diagnostics_off_gives_the_same_coefficients():
    on, off = sc.newton('LS-hard-CGL-1.5', 6), sc.newton('LS-hard-CGL-1.5', 6, diagnostics=False)
    assert np.array_equal(on['beta'], off['beta']) and 'kappa' not in off
