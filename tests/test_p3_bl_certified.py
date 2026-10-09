"""Package 3, item 2: Buckley-Leverett on CC-CGL grids (``experiments/p3_2_bl_certified.py``)."""

import json
from pathlib import Path

import numpy as np
import pytest

import experiments.p3_2_bl_certified as m
from lilq import certified as cf


def _paper_blocks(N, gravity):
    """The paper's random grid at N with its per-block weights, as Rows blocks."""
    from experiments.run_bl import paper_setup
    from lilq.collocation import generate_collocation_points_2d
    config, opt = paper_setup(N, gravity)
    np.random.seed(config.seed)
    pts = generate_collocation_points_2d(config.x_domain, (0, config.T_final), config.N_x, config.N_t,
                                         k_ratio=config.k_ratio, collocation_ratios=(0.9, 0.05, 0.05),
                                         has_initial_condition=True, seed=config.seed, sampling=config.sampling)
    n = {k: len(pts[k]) for k in ('x_pde', 'x_ic', 'x_bc_left', 'x_bc_right')}
    blocks = {'interior': (pts['x_pde'], pts['y_pde'], np.full(n['x_pde'], opt.lambda_pde / n['x_pde'])),
              'initial': (pts['x_ic'], pts['y_ic'], np.full(n['x_ic'], opt.lambda_ic / n['x_ic'])),
              'left': (pts['x_bc_left'], pts['y_bc_left'], np.full(n['x_bc_left'], opt.lambda_bc / n['x_bc_left'])),
              'right': (pts['x_bc_right'], pts['y_bc_right'], np.full(n['x_bc_right'], opt.lambda_bc / n['x_bc_right']))}
    return config, opt, pts, n, blocks


@pytest.mark.parametrize('case', ('viscous', 'gravity'))
@pytest.mark.parametrize('variant', ('cheb', 'sine'))
def test_assembly_matches_the_papers_builder(case, variant):
    """At the paper's points and weights the new assembly is the paper's own
    ``_make_lil_q_system_fn`` (2b: with P2-9's lifting column, coefficient 1)."""
    import problems.buckley_leverett as bl
    config, opt, pts, n, blocks = _paper_blocks(8, case == 'gravity')
    physics = bl.BLPhysics(config)
    space = m.TrialSpace(variant, 8, config.T_final)
    beta = 0.1 * np.random.default_rng(0).standard_normal(space.n_basis)
    A, b = m.Rows(space, physics, blocks).assemble(beta)
    M = bl._prepare_lil_matrices(config, physics, space, pts)
    mats = space.mats(pts['x_pde'], pts['y_pde'])
    core = (mats['val'], mats['dx'], mats['dt'], mats['dxx'], M[4], M[5], M[6])
    if variant == 'sine':
        L = lambda x, d=0: space.lift(x, d)[:, None]  # noqa: E731
        xp = pts['x_pde']
        lift = (L(xp), L(xp, 1), 0 * L(xp), L(xp, 2), L(pts['x_ic']), L(pts['x_bc_left']), L(pts['x_bc_right']))
        core = tuple(np.hstack([c, l]) for c, l in zip(core, lift))
        beta_full = np.concatenate([beta, [1.0]])
    else:
        beta_full = beta
    fn = bl._make_lil_q_system_fn(*core, *M[7:], physics, opt.lambda_pde, opt.lambda_ic, opt.lambda_bc,
                                  n['x_pde'], n['x_ic'], n['x_bc_left'], n['x_bc_right'])
    Ap, bp = fn(beta_full)
    if variant == 'sine':
        Ap, bp = Ap[:, :-1], bp - Ap[:, -1]
    assert np.array_equal(A, Ap)
    assert np.abs(b - bp).max() <= 1e-15 * np.abs(bp).max()


@pytest.mark.parametrize('variant', ('cheb', 'sine'))
def test_newton_identity_on_a_cc_grid(variant):
    _, physics = m.physics_for('gravity')
    space = m.TrialSpace(variant, 8, 0.175)
    rows = m.Rows(space, physics, m.collocation_blocks(cf.bl_grid(64, 10, 0.175)))
    beta = 0.05 * np.random.default_rng(1).standard_normal(64)
    A, b = rows.assemble(beta)
    R = rows.residual(beta)
    assert np.linalg.norm(A @ beta - b - R) < 1e-13 * np.linalg.norm(R)


def test_the_y_norm_carries_each_blocks_total_weight():
    colloc = m.collocation_blocks(cf.bl_grid(256, 5, 0.4))
    quad = m.quadrature_blocks(colloc, 0.4, 96)
    for name in colloc:
        assert np.sum(quad[name][2]) == pytest.approx(np.sum(colloc[name][2]), rel=1e-13)
    assert np.sum(colloc['left'][2]) < 10 * 0.4 / 2.8              # the cut lateral rule: 1 - w_0 (P2-16)


def test_section6_constants():
    rng = np.random.default_rng(2)
    B = rng.standard_normal((300, 20))
    c = m.section6_constants(B, B)                                  # the same set and norm: c1 = c2 = 1
    assert abs(c['c1'] - 1) < 1e-12 and abs(c['c2'] - 1) < 1e-12 and c['dropped'] == 0
    B2 = np.column_stack([B, B[:, 0] * 3.0])                        # a column inside the span is dropped
    assert m.section6_constants(B2, B2)['dropped'] == 1
    half = m.section6_constants(B * 0.5, B)                         # the collocation norm half the Y-norm
    assert abs(half['c1'] - 0.5) < 1e-12 and abs(half['c2'] - 0.5) < 1e-12


def test_2b_lateral_rows_are_zero_rows_and_left_out_of_section6():
    _, physics = m.physics_for('viscous')
    space = m.TrialSpace('sine', 8, 0.4)
    rows = m.Rows(space, physics, m.collocation_blocks(cf.bl_grid(64, 5, 0.4)))
    beta = np.random.default_rng(4).standard_normal(64)
    A, b = rows.assemble(beta)
    n_lat = len(rows.blocks['left']['s']) + len(rows.blocks['right']['s'])
    assert np.abs(A[-n_lat:]).max() < 1e-13 and np.abs(b[-n_lat:]).max() == 0.0
    A_ex, _ = rows.assemble(beta, exclude=('left', 'right'))
    assert A_ex.shape[0] == A.shape[0] - n_lat


def test_apriori_constants_of_2a():
    rows = m.apriori_constants('viscous', 64, 10)
    by = {r['block']: r for r in rows}
    assert f"{by['interior']['c2_over_c1']:.4f}" == '1.0212'                     # Section 4.2
    assert by['left']['dropped'] == 1                                # S = 1: in the trace space
    assert by['right']['dropped'] == 0 and np.isfinite(by['right']['c1'])  # S = 0 adds nothing (no NaN)
    assert not m.apriori_constants('viscous', 64, 5)[-1]['computable']    # C6 at N/P = 5


def _fake_reference(tmp_path, case):
    x, t = np.linspace(0, 1, 41), np.linspace(0, m.CASES[case]['T'], 31)
    X, T = np.meshgrid(x, t, indexing='ij')
    np.savez(tmp_path / f'bl_fd_ref_{case}_nu0.1.npz', x=x, t=t, u=np.exp(-10 * X) * (1 - T))
    return tmp_path


def test_tiny_runs_end_to_end(tmp_path):
    ref = _fake_reference(tmp_path, 'viscous')
    _fake_reference(tmp_path, 'gravity')
    out = tmp_path / 'out'
    for variant, case in (('cheb', 'viscous'), ('sine', 'gravity')):
        d = m.run(variant, case, 64, 5, out, ref, k_max=4)
        meta = json.loads((d / 'run.json').read_text())
        assert meta['package3']['status'] == 'complete' and meta['b2_check']['max_rel_err_over_run'] < 1e-10
        assert "put lambda on each line" in meta['package3']['weights_note']
        assert len((d / 'constants_by_iterate.csv').read_text().splitlines()) == 5
        assert (d / 'rho_r.csv').exists() and (d / 'beta_last.npy').exists()
    m.run('cheb', 'viscous', 64, 5, out, ref, k_max=4, tag='_k6')
    checks = m.summarize(out, ref)
    assert checks['K4 (cheb)']['passed'] and checks['K4 (sine)']['passed'] and checks['K7 (cheb)']['passed']
    t = (out / m.ITEMS['cheb'] / 'terminal.csv').read_text().splitlines()
    assert len(t) == 2                                               # the K6 rerun is not a result row


def test_the_other_guess_runs_where_the_rule_never_fires(tmp_path, monkeypatch):
    """Section 8.2: a run whose rule never fires is run once more from the other guess."""
    calls = []

    def fake(variant, case, P, r, out_root, reference_dir, guess, threads, tag='', quad_scale=1.0):
        calls.append((variant, case, P, r, guess))
        d = Path(out_root) / m.ITEMS[variant] / m.run_name(case, P, r, guess)
        d.mkdir(parents=True, exist_ok=True)
        return d
    monkeypatch.setattr(m, '_run_process', fake)
    monkeypatch.setattr(m, 'rule_fires', lambda d: 'P256' not in d.name)
    monkeypatch.setattr(m, 'apriori_constants', lambda *a: [{'x': 1}])
    names = m.run_all(tmp_path, tmp_path, sizes=(8, 16), ratios=(5,), workers=3, threads=1)
    assert len(calls) == 8 + 4 and len(names) == 12
    reruns = [c for c in calls if c[2] == 256 and c[4] == m.OTHER_GUESS[m.CASES[c[1]]['guess']]]
    assert len(reruns) == 4                                          # every P = 256 run, from the other guess
    assert calls[0][2] == 256                                        # the largest first


def test_k5_starts_first_k6_last_and_a_failed_run_does_not_stop_the_others(tmp_path, monkeypatch):
    calls = []

    def fake(variant, case, P, r, out_root, reference_dir, guess, threads, tag='', quad_scale=1.0):
        calls.append((variant, case, P, r, guess, tag, quad_scale))
        if (variant, case, P) == ('sine', 'gravity', 64):
            raise RuntimeError('sine gravity_P64 exited 1\nTraceback (most recent call last): ...')
        d = Path(out_root) / m.ITEMS[variant] / m.run_name(case, P, r, guess, tag)
        d.mkdir(parents=True, exist_ok=True)
        return d
    monkeypatch.setattr(m, '_run_process', fake)
    monkeypatch.setattr(m, 'rule_fires', lambda d: True)
    monkeypatch.setattr(m, 'apriori_constants', lambda *a: [{'x': 1}])
    with pytest.raises(m.RunsFailed) as e:
        m.run_all(tmp_path, tmp_path, sizes=(8, 16), ratios=(5,), workers=1, threads=1, with_checks=True)
    assert calls[0][2:] == (1024, 10, 'ic', '_k5', 1.5)               # as long as the largest: it starts with them
    assert calls[-1][2:6] == (256, 10, 'ic', '_k6')
    assert len(calls) == 8 + 2 and len(e.value.done) == 9 and e.value.failed == ['sine gravity_P64 exited 1']
    assert (tmp_path / m.ITEMS['cheb'] / 'constants.csv').exists()  # the rest of run_all still happens


def test_a_failed_run_puts_its_error_in_the_log(tmp_path):
    with pytest.raises(RuntimeError) as e:                          # no reference there: the run fails at once
        m._run_process('cheb', 'viscous', 64, 5, tmp_path, tmp_path / 'no_such_dir', 'ic', 1)
    first, rest = str(e.value).split('\n', 1)
    assert first.startswith('cheb viscous') and 'exited 1' in first and 'Traceback' in rest


def test_the_summary_skips_runs_that_did_not_finish(tmp_path, monkeypatch):
    d = tmp_path / m.ITEMS['cheb']
    for name in ('viscous_P64_NP5_ic', 'viscous_P64_NP5_ic_k6'):
        (d / name).mkdir(parents=True)
        (d / name / 'run.json').write_text(json.dumps({'package3': {'status': 'running'}}))
        (d / name / 'constants_by_iterate.csv').write_text('k,c1\n0,1.0\n')        # written at every iterate
    assert not m._complete(d / 'viscous_P64_NP5_ic') and not m._complete(tmp_path / 'missing')
    monkeypatch.setattr(m, 'delta_table', lambda ref: {})
    checks = m.summarize(tmp_path, tmp_path)
    assert checks['incomplete runs (cheb)'] == {'runs': ['viscous_P64_NP5_ic', 'viscous_P64_NP5_ic_k6'], 'passed': False}
    assert not any(k.startswith('K6') for k in checks)              # not compared with a partial run
