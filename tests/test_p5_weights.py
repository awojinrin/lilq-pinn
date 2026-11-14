"""Package 5, item 1: the norm of X, the Cole-Hopf derivatives, the weight option."""

import json

import numpy as np
import pytest

from experiments import p2_16_certified as p216
from experiments import p3_2_bl_certified as p32
from experiments import p5_1_weights as m
from lilq import certified as cf
from lilq.references import cole_hopf, cole_hopf_derivatives, cole_hopf_ux
from lilq.xnorm import XNorm, gauss_legendre_2d

NU = 0.1


@pytest.mark.parametrize('t', [0.01, 0.3, 1.0])
def test_cole_hopf_derivatives_agree_with_the_existing_formulas(t):
    x = np.linspace(-1, 1, 41)
    d = cole_hopf_derivatives(x, t, NU)
    np.testing.assert_allclose(d['u'], cole_hopf(x, t, NU), atol=1e-12)
    np.testing.assert_allclose(d['u_x'], cole_hopf_ux(x, t, NU), atol=1e-11)


@pytest.mark.parametrize('t', [0.01, 0.3, 1.0])
def test_cole_hopf_derivatives_satisfy_burgers(t):
    """Section 3: u_t + u u_x - nu u_xx = 0 to 1e-8 (relative to the terms)."""
    x = np.linspace(-1, 1, 41)
    d = cole_hopf_derivatives(x, t, NU)
    res = d['u_t'] + d['u'] * d['u_x'] - NU * d['u_xx']
    scale = np.max(np.abs(d['u_t']) + np.abs(d['u'] * d['u_x']) + np.abs(NU * d['u_xx']))
    assert np.max(np.abs(res)) / scale < 1e-8


def test_cole_hopf_u_t_and_u_xx_against_differences():
    """Independent of the PDE: centred differences of u in t and in x."""
    x, t, h = np.linspace(-0.9, 0.9, 13), 0.4, 1e-4
    d = cole_hopf_derivatives(x, t, NU)
    ut = (cole_hopf(x, t + h, NU) - cole_hopf(x, t - h, NU)) / (2 * h)
    uxx = (cole_hopf(x + h, t, NU) - 2 * cole_hopf(x, t, NU) + cole_hopf(x - h, t, NU)) / h ** 2
    np.testing.assert_allclose(d['u_t'], ut, atol=1e-6)
    np.testing.assert_allclose(d['u_xx'], uxx, atol=1e-5)


def test_xnorm_error_and_best_on_a_function_in_the_space():
    x, y, w = gauss_legendre_2d(20, ((0.0, 1.0), (0.0, 1.0)))
    mats = m.scalar_mats('bratu', 4, x, y)
    beta = np.random.default_rng(0).standard_normal(16)
    ref = {c: mats[c] @ beta for c in m.H2}
    xn = XNorm(w, ref, m.H2)
    assert xn.error_of(mats, beta) < 1e-14
    assert xn.best(mats) < 1e-12
    assert xn.best(mats, driver='gelsd') < 1e-12
    np.testing.assert_allclose(xn.error_of(mats, 2 * beta), 1.0, rtol=1e-13)   # error of 2u is ||u||


def test_xnorm_lifting_enters_the_error_and_the_best():
    x, t, w = gauss_legendre_2d(24, ((0.0, 1.0), (0.0, 0.4)))
    space = p32.TrialSpace('sine', 4, 0.4)
    mats, lift = m.bl_mats(space, x, t)
    beta = np.random.default_rng(1).standard_normal(16)
    ref = {c: mats[c] @ beta + lift[c] for c in m.H21}
    xn = XNorm(w, ref, m.H21)
    assert xn.error_of(mats, beta, lift) < 1e-14
    assert xn.best(mats, lift) < 1e-12
    assert xn.error_of(mats, beta) > 1e-3                 # without the lifting the error is not round-off


def test_bratu_mixed_derivative_matches_differences():
    x, y = np.array([0.3, 0.6]), np.array([0.2, 0.7])
    beta = np.random.default_rng(2).standard_normal(25)
    h = 1e-4
    u = lambda a, b: m.scalar_mats('bratu', 5, a, b)['u'] @ beta  # noqa: E731
    fd = (u(x + h, y + h) - u(x + h, y - h) - u(x - h, y + h) + u(x - h, y - h)) / (4 * h * h)
    np.testing.assert_allclose(m.scalar_mats('bratu', 5, x, y)['u_xy'] @ beta, fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('bench, P', [('bratu', 100), ('burgers', 225)])
def test_p2_16_grid_default_is_unchanged(bench, P):
    """The new option leaves the Package 2 grid bit-identical; lam scales only the auxiliary lines."""
    a, b = p216.grid(bench, P, 10), p216.grid(bench, P, 10, lam_aux=p216.LAM_AUX)
    c = p216.grid(bench, P, 10, lam_aux=1e4)
    for name in a['blocks']:
        for i in range(3):
            assert np.array_equal(a['blocks'][name][i], b['blocks'][name][i])
        scale = 1.0 if name == 'interior' else 1e4 / p216.LAM_AUX
        np.testing.assert_allclose(c['blocks'][name][2], scale * a['blocks'][name][2], rtol=1e-15)


def test_p3_2_collocation_blocks_default_is_unchanged():
    g = cf.bl_grid(256, 10, 0.4)
    a, b, c = p32.collocation_blocks(g), p32.collocation_blocks(g, lam_aux=p32.LAM_AUX), p32.collocation_blocks(g, 1e6)
    for name in a:
        assert np.array_equal(a[name][2], b[name][2])
        scale = 1.0 if name == 'interior' else 1e6 / p32.LAM_AUX
        np.testing.assert_allclose(c[name][2], scale * a[name][2], rtol=1e-15)


def test_scalar_run_at_weight_10_is_the_p2_16_system(tmp_path, monkeypatch):
    """At lam = 10 the item-1 rows are P2-16's rows (same A, f, R)."""
    rows = m.ScalarRows('bratu', 25, 10.0)
    ref_rows = p216.Rows('bratu', 5, p216.grid('bratu', 25, 10)['blocks'])
    beta = np.random.default_rng(3).standard_normal(25) * 0.1
    A, f = rows.assemble(beta)
    A0, f0, R0 = p216.assemble(ref_rows, beta)
    assert np.array_equal(A, A0) and np.array_equal(f, f0) and np.array_equal(rows.residual(beta), R0)


def test_terminal_row_gate(tmp_path):
    """An H^{2,1} ratio is reported only where delta_X exceeds three times the reference's accuracy."""
    d = tmp_path / 'run'
    d.mkdir()
    cols = 'k,chi,norm_Rlin_h,num_rank_gelsy,eps_ref\n'
    rows = [(0, 1.0, 1.0, 4, 1.0), (1, 0.01, 0.5, 4, 0.2), (2, 0.01, 0.5, 4, 0.1), (3, 0.01, 0.5, 4, 0.1),
            (4, '', '', '', 0.1)]
    (d / 'iterations.csv').write_text(cols + ''.join(','.join(map(str, r)) + '\n' for r in rows))
    (d / 'errors_by_iterate.csv').write_text('k,eX\n' + ''.join(f'{k},{0.3 if k < 2 else 0.2}\n' for k in range(5)))
    (d / 'run.json').write_text(json.dumps({'package5': {'family': 'bl_cheb_viscous', 'P': 4, 'lam': 10.0,
                                                         'identity_scaled_max': 1e-16}}))
    ok = m.terminal_row(d, dL2=0.05, dX=0.1, gate=0.01)
    assert ok['rule_step'] == 3 and ok['returned_iterate'] == 4   # flags at steps 2 and 3 (n_s = 2)
    assert ok['eX_over_delta'] == pytest.approx(2.0) and ok['x_ratio_reported']
    left_out = m.terminal_row(d, dL2=0.05, dX=0.1, gate=0.05)
    assert left_out['eX_over_delta'] is None and not left_out['x_ratio_reported']
    assert left_out['eL2_over_delta'] == pytest.approx(2.0)


# ---------------------------------------------------------------- the fine reference's fallback

def _fake_reference(out_root, case, p, rule_step=5):
    d = m.reference_dir_bl(out_root) / f'{case}_p{p}'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'run.json').write_text(json.dumps({'package5': {'rule_step': rule_step}}))


def test_choose_keeps_88_when_both_stop_by_the_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'bl_reference_run', lambda *a, **k: pytest.fail('no fallback run expected'))
    for p in (88, 80):
        _fake_reference(tmp_path, 'viscous', p)
    rec = m.choose('viscous', tmp_path, 'refdir')
    assert rec['p_ref'] == 88 and rec['p_neighbour'] == 80 and not rec['fallback']
    assert m.p_ref_of(tmp_path, 'viscous') == 88 and m.p_ref_of(tmp_path, 'viscous', 40) == 40


def test_choose_falls_back_per_case(tmp_path, monkeypatch):
    """The advisor's reply, answer 1: p = 88 fails or does not stop by the rule -> 64 with 56, case by case."""
    ran = []

    def fake_run(case, p, out_root, reference_dir):
        ran.append((case, p))
        _fake_reference(out_root, case, p)
    monkeypatch.setattr(m, 'bl_reference_run', fake_run)
    _fake_reference(tmp_path, 'viscous', 80)
    (m.reference_dir_bl(tmp_path) / 'viscous_p88').mkdir()          # stopped at its time limit: no run.json
    _fake_reference(tmp_path, 'gravity', 80)
    _fake_reference(tmp_path, 'gravity', 88, rule_step=None)         # ran to K_max without the rule
    v = m.choose('viscous', tmp_path, 'refdir')
    g = m.choose('gravity', tmp_path, 'refdir')
    assert v['p_ref'] == 64 and v['fallback'] and 'did not finish' in v['reason']
    assert g['p_ref'] == 64 and 'did not stop by the rule' in g['reason']
    assert ran == [('viscous', 56), ('viscous', 64), ('gravity', 56), ('gravity', 64)]
    for case in ('viscous', 'gravity'):                                 # one file per case (the jobs run side by side)
        assert json.loads((m.reference_dir_bl(tmp_path) / m.CHOSEN.format(case=case)).read_text())['p_ref'] == 64
    assert set(m.chosen_record(tmp_path)) == {'viscous', 'gravity'} and m.p_ref_of(tmp_path, 'gravity') == 64


def test_choose_reports_a_failed_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'bl_reference_run', lambda case, p, out_root, rd: _fake_reference(out_root, case, p, None))
    assert m.main(['choose', '--case', 'viscous', '--out', str(tmp_path), '--reference-dir', 'refdir']) == 1
    rec = m.chosen_record(tmp_path)['viscous']
    assert rec['p_ref'] is None and 'the fallback failed too' in rec['reason']
    with pytest.raises(RuntimeError, match='the fallback failed too'):   # never a silent default
        m.p_ref_of(tmp_path, 'viscous')
    assert m.main(['ref-ok', '--case', 'viscous', '--p-ref', '64', '--out', str(tmp_path)]) == 1


def test_a_size_never_run_is_named_so(tmp_path, monkeypatch):
    """The job runs p = 88 only if p = 80 stopped by the rule; the record says 88 was not run."""
    monkeypatch.setattr(m, 'bl_reference_run', lambda case, p, out_root, rd: _fake_reference(out_root, case, p))
    _fake_reference(tmp_path, 'viscous', 80, rule_step=None)
    rec = m.choose('viscous', tmp_path, 'refdir')
    assert rec['tried'] == {'88': 'p = 88 was not run', '80': 'p = 80 did not stop by the rule within K_max = 60'}
    assert rec['p_ref'] == 64


def test_the_grace_runs_need_each_cases_choice(tmp_path):
    """``runs`` (the Grace job) never falls back to 88 silently: a case without a record stops with the reason."""
    assert m.p_ref_of(tmp_path, 'viscous') == m.P_REF_DEFAULT        # the laptop's 'all' stage: the default
    with pytest.raises(RuntimeError, match='no fine reference was chosen for viscous'):
        m.p_ref_of(tmp_path, 'viscous', require_chosen=True)
    with pytest.raises(RuntimeError, match='no fine reference was chosen'):
        m.main(['runs', '--only', 'bl_cheb_viscous', '--out', str(tmp_path), '--reference-dir', 'refdir'])


def test_the_summary_records_a_case_without_a_reference_and_goes_on(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'bl_reference_run', lambda case, p, out_root, rd: _fake_reference(out_root, case, p, None))
    m.choose('viscous', tmp_path, 'refdir')                             # no usable size
    (tmp_path / m.ITEM / 'bl_cheb_viscous').mkdir(parents=True)
    rows, records = m.summarize(tmp_path, 'refdir', only=['bl_cheb_viscous'])
    assert rows == [] and 'the fallback failed too' in records['bl_viscous']['error']
