"""lilq/references.py (Package 2, Section 6.2): the Cole-Hopf reference for
Burgers, checked against Basdevant et al. (1986) and against an independent
series evaluation; the eps_ref helper and log column."""

import json

import numpy as np
import pytest

from lilq import references as R


def test_cole_hopf_passes_the_basdevant_check():
    b = R.BASDEVANT
    ux = R.cole_hopf_ux(np.array([b['x']]), b['t'], b['nu'])[0]
    assert f"{ux:.5f}" == f"{b['u_x']:.5f}"                     # -152.00516


@pytest.mark.parametrize('t', [0.02, 0.4, 1.0])
def test_quadrature_and_series_agree_at_nu_0_1(t):
    x = np.linspace(-1, 1, 41)
    q, s = R.cole_hopf(x, t, 0.1), R.cole_hopf_series(x, t, 0.1)
    assert np.abs(q - s).max() < 1e-12
    assert abs(q[0]) < 1e-15 and abs(q[-1]) < 1e-15


def test_initial_condition_and_the_pde():
    x = np.linspace(-0.9, 0.9, 7)
    assert np.allclose(R.cole_hopf(x, 0.0, 0.1), -np.sin(np.pi * x), atol=0)
    # u_t + u u_x - nu u_xx = 0 by central differences of the series
    nu, t, h, k = 0.1, 0.3, 1e-3, 1e-4
    u = lambda xx, tt: R.cole_hopf_series(np.atleast_1d(xx), tt, nu)  # noqa: E731
    ut = (u(x, t + k) - u(x, t - k)) / (2 * k)
    ux = (u(x + h, t) - u(x - h, t)) / (2 * h)
    uxx = (u(x + h, t) - 2 * u(x, t) + u(x - h, t)) / h ** 2
    assert np.abs(ut + u(x, t) * ux - nu * uxx).max() < 1e-5


def test_eps_ref_helper_and_reference_files(tmp_path):
    from lilq.basis import create_basis_2d
    basis = create_basis_2d('chebyshev', 3, 3, (0, 1), (0, 1))
    x = y = np.linspace(0, 1, 11)
    beta = np.zeros(9)
    beta[0] = 2.0                                                 # u = 2 (T_0 T_0)
    np.savez_compressed(tmp_path / 'r.npz', x=x, y=y, u=np.ones((11, 11)), meta=np.array(json.dumps({'k': 1})))
    axes, ref, meta = R.load_reference(tmp_path / 'r.npz')
    assert meta == {'k': 1} and ref.shape == (11, 11)
    assert R.make_eps_ref_fn(basis, tmp_path / 'r.npz')(beta)['eps_ref'] == pytest.approx(1.0)
    assert R.reference_path(tmp_path, 'burgers').name == 'burgers_cole_hopf.npz'


def test_save_npz_atomic_replaces_and_leaves_no_temporary(tmp_path):
    import numpy as np
    from lilq.references import save_npz_atomic, write_text_atomic
    path = tmp_path / 'ref.npz'
    save_npz_atomic(path, u=np.zeros(3))
    save_npz_atomic(path, u=np.ones(3))
    with np.load(path) as z:
        assert z['u'].tolist() == [1.0, 1.0, 1.0]
    write_text_atomic(tmp_path / 'a.json', '{}')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.json', 'ref.npz']
