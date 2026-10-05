"""Package 2, item 6b (P2-9): the boundary-conforming bases. The lifted sine
basis and its derivatives, the lifted LiL-Q run on the paper's grid (boundary
data exact, the B2 identity), the Bratu bases, and the rows of Section 9.2."""

import json
from pathlib import Path

import numpy as np
import pytest

import experiments.p2_9_bases as m

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode,weighted', [('both', True), ('cos', True), ('both', False)])
def test_lifted_sine_basis(mode, weighted):
    b = m.LiftedSineBasis(5, (0.0, 0.4), mode, weighted=weighted)
    assert b.n_basis == 25
    t = np.linspace(0, 0.4, 7)
    assert np.abs(b.evaluate(0 * t, t)).max() < 1e-15 and np.abs(b.evaluate(0 * t + 1, t)).max() < 1e-15
    x, tt, h = np.linspace(0.05, 0.95, 9), np.linspace(0.02, 0.38, 9), 1e-5
    v = lambda xx, ttt: b.evaluate(xx, ttt)  # noqa: E731
    np.testing.assert_allclose(b.derivative(x, tt, dx=1), (v(x + h, tt) - v(x - h, tt)) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(b.derivative(x, tt, dy=1), (v(x, tt + h) - v(x, tt - h)) / (2 * h), atol=1e-6)
    d1 = lambda xx: b.derivative(xx, tt, dx=1)  # noqa: E731
    np.testing.assert_allclose(b.derivative(x, tt, dx=2), (d1(x + h) - d1(x - h)) / (2 * h), atol=1e-5)
    assert m.lifting(np.array([0.0, 1.0])).tolist() == [1.0, 0.0]


def _bratu_reference(tmp_path):
    import baselines.square_chebyshev as sc
    ref = tmp_path / 'reference'
    ref.mkdir(exist_ok=True)
    r = sc.newton('SQ-CGL', 16, diagnostics=False)
    g = np.linspace(0, 1, 201)
    np.savez(ref / 'bratu_ref_p48.npz', x=g, y=g, u=r['space'].grid_values(r['beta'], g))
    return ref


def test_bratu_bases(tmp_path):
    ref = _bratu_reference(tmp_path)
    out = tmp_path / 'P2_9_bases'
    s = m.bratu_run('sin_sin', 5, out, ref)
    c = m.bratu_run('chebyshev_weak', 5, out, ref)
    assert s['bc_max_violation'] < 1e-14 and c['bc_max_violation'] > 1e-6       # sine: exact; Chebyshev: weak rows
    for r in (s, c):
        assert r['P'] == 25 and r['rank'] == 25 and r['k_stop'] <= 60 and r['b2_max_rel_err'] < 1e-12
        assert 0 < r['delta_P'] <= r['eps_ref_stop'] * 1.0001                   # nothing beats the best fit by much
    basis = json.loads((out / 'bratu_chebyshev_weak_P25_zero' / 'run.json').read_text())['basis_description']
    assert 'chebyshev' in json.dumps(basis)


def test_lifted_bl_run_on_the_paper_grid(tmp_path):
    import experiments.p2_2_nu_refinement as nr
    ref = tmp_path / 'reference'
    for case in ('viscous', 'gravity'):                               # coarse stand-ins (the solver's own, 200 intervals)
        from problems.buckley_leverett import TEST_GRID, reference_solution
        config, _, _ = nr.setup(case, 8, 0.1)
        ref.mkdir(exist_ok=True)
        np.savez(ref / f'bl_fd_ref_{case}_nu0.1.npz', x=np.linspace(0, 1, TEST_GRID[0]),
                 t=np.linspace(0, config.T_final, TEST_GRID[1]), u=np.array(reference_solution(config, 200)))
    out = tmp_path / 'P2_9_bases'
    specified = {}
    for case in ('viscous', 'gravity'):
        lifted = specified[case] = m.bl_run(case, 'lifted_sine_x(1-x)', 8, 'ic', out, ref)
        paper = m.bl_run(case, 'paper', 8, 'ic', out, ref)
        assert lifted['bc_max_violation'] < 1e-14 and lifted['b2_max_rel_err'] < 1e-12
        assert 0 < lifted['eps_ref_60'] < 1 and lifted['rank'] <= 64
        rl = json.loads((out / f'bl_{case}_lifted_sine_x1mx_P64_ic' / 'run.json').read_text())
        rp = json.loads((out / f'bl_{case}_paper_P64_ic' / 'run.json').read_text())
        assert rl['N_composition'] == rp['N_composition'] and rl['row_weights'] == rp['row_weights']   # the paper's grid
    plain = m.bl_run('viscous', 'lifted_sine_plain', 8, 'ic', out, ref)
    assert plain['bc_max_violation'] < 1e-14 and plain['b2_max_rel_err'] < 1e-12
    assert 0 < plain['delta_P'] < specified['viscous']['delta_P']           # the plain sines fit better
    assert 'plain' in json.loads((out / 'bl_viscous_lifted_sine_plain_P64_ic' / 'run.json').read_text())[
        'basis_description']['family']
    zero = m.bl_run('viscous', 'lifted_sine_x(1-x)', 8, 'zero', out, ref)
    assert zero['eps_over_delta'] == pytest.approx(zero['eps_ref_stop'] / zero['delta_P'])
    assert zero['guess'] == 'zero' and zero['basis'] == 'lifted_sine_x(1-x)' and plain['basis'] == 'lifted_sine_plain'
    for col in ('benchmark', 'case', 'basis', 'P', 'guess', 'k_target', 'k_rule', 'eps_ref_stop', 'eps_ref_60',
                'kappa', 'rank'):
        assert col in zero                                            # Section 9.2's columns


def test_item_6b_job():
    text = (REPO / 'scripts' / 'cluster' / 'package2' / 'p2s2_bases.slurm').read_text()
    assert '# lilq-resources: cpu' in text and 'p2_9_bases.py' in text
