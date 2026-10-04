"""baselines/square_kovasznay.py: the square P_N - P_{N-2} Kovasznay
collocation of Package 2, Section 4.2 (on the real sizes, p_d = 10 and 15:
full rank with the corner pin, 10 and 6 Newton iterations; DECISIONS.md)."""

import numpy as np
import pytest

import baselines.square_kovasznay as sk


def test_counts_and_rows():
    s = sk.System(8)
    assert s.n == 3 * 8 ** 2 - 4 * 8 + 4 == 2 * 64 + 36
    J, f = s.assemble(np.zeros(s.n))
    assert J.shape == (s.n, s.n) and len(f) == s.n
    assert len(s.xi) == 36 and len(s.xb) == 28                        # (p_d - 2)^2 and 4 p_d - 4
    assert s.pin_point == sk.CORNER
    assert sk.System(8, 'interior').pin_point == (s.xi[s.pin_row], s.yi[s.pin_row])


def test_newton_matrix_is_the_jacobian_of_the_residual():
    """J(beta) beta - f(beta) is the nonlinear residual N(beta); its
    finite-difference Jacobian must equal J(beta)."""
    s = sk.System(6)
    rng = np.random.default_rng(1)
    beta = 0.1 * rng.standard_normal(s.n)
    N = lambda b: (lambda J, f: J @ b - f)(*s.assemble(b))  # noqa: E731
    J, _ = s.assemble(beta)
    h = 1e-6
    for j in rng.choice(s.n, 12, replace=False):
        e = np.zeros(s.n)
        e[j] = h
        fd = (N(beta + e) - N(beta - e)) / (2 * h)
        assert np.abs(fd - J[:, j]).max() <= 1e-6 * max(1.0, np.abs(J[:, j]).max())


def test_the_exact_solution_satisfies_the_boundary_rows_and_newton_converges():
    r8, r12 = sk.newton(8), sk.newton(12)
    for r in (r8, r12):
        assert r['stop'] == 'tolerance' and r['rank'] == r['n_unknowns']
    assert r12['eps_u'] < r8['eps_u'] / 10                              # spectral convergence
    s = r12['system']
    a, b, _ = s.split(r12['beta'])
    assert np.abs(s.Vb @ a - s.ub).max() < 1e-10 and np.abs(s.Vb @ b - s.vb).max() < 1e-10
    # the pin holds at the solution
    assert s.pin_phi @ s.split(r12['beta'])[2] == pytest.approx(s.pin_value, abs=1e-10)


def test_the_pin_changes_only_the_pressure_gauge():
    c, i = sk.newton(10, 'corner'), sk.newton(10, 'interior')
    assert c['eps_u'] == pytest.approx(i['eps_u'], rel=1e-8) and c['eps_p_meanfree'] == pytest.approx(i['eps_p_meanfree'], rel=1e-6)
