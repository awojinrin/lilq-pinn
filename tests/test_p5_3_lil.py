"""Package 5, item 3: the LiL runs (``experiments/p5_3_lil.py``).

The torch residuals against the numpy ones (values and gradients, CPU and,
where present, CUDA); LiL-N L-BFGS through them against item 2's numpy path;
the logged and timed passes of LiL-Q and ``trf``; the square collocation;
C3. Cutoffs set before any comparison (DECISIONS.md, item 3, batch 3)."""

import csv
import json
import math

import numpy as np
import pytest
import torch

import experiments.p5_3_lil as m
from experiments import p2_16_certified as p216
from experiments import p4_1_burgers_gauss as p41

RES_RTOL = 1e-10            # torch residual and gradient against numpy (relative, in norm)
FINAL_LOSS_RTOL = 1e-8      # L-BFGS's final loss against item 2's numpy path (the same minimum)
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
CASES = [('bratu', 25), ('burgers', 36), ('bl', 64)]


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def ref(tmp_path):
    """Stand-in references (the values only enter the error)."""
    x = np.linspace(0, 1, 201)
    np.savez(tmp_path / 'bratu_ref_p48.npz', x=x, y=x, u=np.outer(np.sin(np.pi * x), np.sin(np.pi * x)))
    xb, t = np.linspace(-1, 1, 11), np.linspace(0, 1, 11)
    np.savez(tmp_path / 'burgers_cole_hopf.npz', x=xb, t=t, u=-np.outer(np.sin(np.pi * xb), np.exp(-t)))
    xl, tl = np.linspace(0, 1, 11), np.linspace(0, 0.4, 9)
    np.savez(tmp_path / 'bl_fd_ref_viscous_nu0.1.npz', x=xl, t=tl, u=np.outer(1 - xl, 1 + 0 * tl))
    return tmp_path


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('bench,P', CASES)
def test_the_torch_residual_and_its_gradient_are_the_numpy_ones(ref, bench, P, device):
    """N(beta) in the numpy row order, and the autograd gradient of ||N||^2 equal to
    2 A(beta)^T N(beta), A the analytic Jacobian (for Buckley-Leverett this needs the
    f''(S) S_x term the detached flux derivative drops)."""
    prob = m.Problem(bench, P, ref)
    b = prob.beta0 + 0.05 * np.random.default_rng(0).standard_normal(P)
    R = prob.residual(b)
    bt = torch.tensor(b, device=device, requires_grad=True)
    r = torch.cat([x for _, x in prob.torch_residual(torch.device(device))(bt)])
    (r * r).sum().backward()
    assert _rel(r.detach().cpu().numpy(), R) <= RES_RTOL
    assert _rel(bt.grad.cpu().numpy(), 2 * prob.jacobian(b).T @ R) <= RES_RTOL
    with torch.no_grad():                                   # no graph: the same values
        r0 = torch.cat([x for _, x in prob.torch_residual(torch.device(device))(torch.tensor(b, device=device))])
    assert _rel(r0.cpu().numpy(), R) <= RES_RTOL


def test_the_bratu_residual_alone_is_assembles(ref):
    rows = m.Problem('bratu', 100, ref).rows.rows
    b = 0.1 * np.random.default_rng(2).standard_normal(100)
    A, f, R = p216.assemble(rows, b)
    none_a, none_f, R1 = p216.assemble(rows, b, system=False)
    assert none_a is None and none_f is None and _rel(R1, R) <= 1e-14


def test_the_loss_splits_into_its_blocks(ref):
    prob = m.Problem('bl', 64, ref)
    loss = m.TorchLoss(prob.torch_residual(torch.device('cpu')))
    bt = torch.tensor(prob.beta0, requires_grad=True)
    total, pde, ic, bc = loss(bt)
    r = prob.residual(prob.beta0)
    assert float(total.detach()) == pytest.approx(float(r @ r), rel=1e-12)
    assert float(pde + ic + bc) == pytest.approx(float(total.detach()), rel=1e-12) and loss.param is bt


def test_lbfgs_reaches_item_2s_minimum(ref):
    """L-BFGS through the torch residual and item 2's (numpy residual, analytic
    gradient), Burgers p = 10, from zero, no cap: both stop by the stall rule at the
    same minimum (final losses within FINAL_LOSS_RTOL). The paths themselves agree
    to round-off only until near the minimum, where L-BFGS amplifies the 1e-16
    difference of the two gradients (to 2e-5 near iteration 50 on the laptop; the
    user's decision of 16 November 2026, DECISIONS.md)."""
    from experiments import p5_2_nlls as p52
    from lilq.solvers import solve_lil_n
    prob = m.Problem('burgers', 100, ref)
    old = p52.Problem('burgers', 10, prob.rows, lambda b: {}, {})
    _, _, s2 = solve_lil_n(p52._SquaredResidual(old), np.zeros(100), torch.device('cpu'), max_iterations=m.NO_CAP,
                           R_tol=0.0, verbose=False, stall_rule='f1')
    _, _, s3, *_ = m.solve_lbfgs(prob, 'cpu', budget_s=None, log_error=False)
    assert s2['optimizer_stall'] and s3['optimizer_stall']
    assert abs(s3['final_loss'] - s2['final_loss']) / s2['final_loss'] <= FINAL_LOSS_RTOL


def test_lbfgs_logs_every_10_iterations_and_keeps_those_coefficients(ref, tmp_path):
    prob = m.Problem('burgers', 100, ref)
    beta0 = prob.beta0.copy()
    _, metrics, summary, probe, logged, _ = m.solve_lbfgs(prob, 'cpu', budget_s=None, max_iterations=25)
    assert [r['iteration'] for r in probe.rows] == [0, 10, 20, 25]
    assert len(logged) == len(probe.rows) and np.array_equal(prob.beta0, beta0)   # beta0 is not moved
    for row, b in zip(probe.rows, logged):
        assert row['error'] == pytest.approx(prob.error(b), rel=1e-12)
    assert metrics.off_clock_s > 0


def test_the_budget_ends_an_lbfgs_run(ref, tmp_path):
    rec = m.run_lbfgs(m.Problem('bl', 256, ref), tmp_path, 'cpu', budget_s=0.2)
    assert rec['stopping_reason'] in ('budget', 'optimizer_stall')
    if rec['stopping_reason'] == 'budget':
        assert 0.2 <= rec['time_s'] < 5
    d = m.run_dir_of(tmp_path, 'bl', 256, 'lbfgs')
    rows = _rows(d / 'history.csv')
    assert list(rows[0]) == list(m.HISTORY_COLUMNS)
    assert len(np.load(d / 'betas_logged.npy')) == len(rows)
    rj = json.loads((d / 'run.json').read_text())
    assert rec['last_logged_is_returned'] and rj['timed'] is True and isinstance(rj['iterations'], int)


def test_the_lbfgs_run_leaves_the_default_dtype_alone(ref):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        m.solve_lbfgs(m.Problem('burgers', 36, ref), 'cpu', budget_s=None, max_iterations=3)
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(old)


def test_setup_excludes_the_reference(ref, monkeypatch):
    """setup_s covers the system and the initial coefficients, not the reference loading."""
    import time as _time
    real = p41.load_reference
    monkeypatch.setattr(p41, 'load_reference', lambda d: (_time.sleep(0.3), real(d))[1])
    assert m.Problem('burgers', 36, ref).setup_s < 0.3


def test_lilq_logged_and_timed(ref, tmp_path):
    prob = m.Problem('burgers', 36, ref)
    rec = m.run_lilq(prob, tmp_path)
    d = m.run_dir_of(tmp_path, 'burgers', 36, 'lilq')
    betas, rows = np.load(d / 'betas.npy'), _rows(d / 'history.csv')
    assert rec['returned_iterate'] == rec['rule_step'] + 1 == len(betas) - 1 == len(rows) - 1
    assert rec['timing_matches'] and rec['timing']['max_beta_rel_diff'] <= m.BETA_RTOL
    times = [float(r['time_s']) for r in rows]
    assert times[0] == 0 and all(b > a for a, b in zip(times, times[1:]))
    beta, k_s, _, _ = m.iterate_to_rule(prob.rows, prob.beta0, m.N_S)
    assert k_s == rec['rule_step'] and np.array_equal(beta, betas[-1])
    assert float(rows[-1]['error']) == pytest.approx(prob.error(beta), rel=1e-12)


def test_trf_logs_every_accepted_iterate(ref, tmp_path):
    prob = m.Problem('burgers', 36, ref)
    rec = m.run_trf(prob, tmp_path)
    d = m.run_dir_of(tmp_path, 'burgers', 36, 'trf')
    betas = np.load(d / 'betas.npy')
    assert rec['final_x_is_last_logged'] and rec['njev'] == len(betas) and rec['nfev'] <= m.TRF_MAX_NFEV
    assert rec['timing_matches'] and np.array_equal(betas[0], prob.beta0)
    hist = _rows(d / 'history.csv')
    losses = [float(r['loss']) for r in hist]
    assert all(b <= a for a, b in zip(losses, losses[1:]))          # accepted steps lower the cost
    evals = [int(r['evaluations']) for r in hist]                    # residual evaluations to each iterate
    assert evals[0] == 1 and all(b > a for a, b in zip(evals, evals[1:])) and evals[-1] <= rec['nfev']


def test_the_square_collocation(ref, tmp_path):
    rec = m.run_square(25, tmp_path, ref)
    rows = _rows(m.run_dir_of(tmp_path, 'bratu', 25, 'square') / 'history.csv')
    assert rec['p'] == 7 and rec['free_coefficients'] == 25 and rec['stopping_reason'] == 'tolerance'
    assert len(rows) == rec['k_stop'] + 1 and rec['timing_matches']
    assert all(math.isfinite(float(r['error'])) for r in rows)
    times = [float(r['time_s']) for r in rows]                      # plain floats in the file
    assert times[0] == 0 and all(b > a for a, b in zip(times, times[1:]))


def test_methods_per_device():
    assert m.methods_for('bratu', 'cpu') == ['lilq', 'trf', 'lbfgs', 'square']
    assert m.methods_for('burgers', 'cpu') == ['lilq', 'trf', 'lbfgs']
    assert m.methods_for('bl', 'cuda') == ['lbfgs']


def test_the_warm_up(ref):
    w = m.warm_up('bratu', 'cpu', ref)
    assert set(w) == set(m.METHODS) and all(s > 0 for s in w.values())


def test_a_tag_names_the_process_records_and_is_kept_in_each_run(ref, tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'save_provenance', lambda d: d.mkdir(parents=True, exist_ok=True))
    for tag in ('cpu-lil', 'cpu-bratu'):
        m.main(['run', '--bench', 'bratu', '--out', str(tmp_path), '--reference-dir', str(ref), '--P', '25',
                '--methods', 'lilq', '--no-timing', '--tag', tag])
    root = tmp_path / m.ITEM / 'lil'
    assert sorted(p.name for p in (root / 'warmup').iterdir()) == ['bratu_cpu_cpu-bratu.json', 'bratu_cpu_cpu-lil.json']
    assert sorted(p.name for p in (root / 'provenance').iterdir()) == ['bratu_cpu_cpu-bratu', 'bratu_cpu_cpu-lil']
    assert json.loads((m.run_dir_of(tmp_path, 'bratu', 25, 'lilq') / 'run.json').read_text())['tag'] == 'cpu-bratu'


def test_c3(ref, tmp_path):
    """C3 against a stand-in Package 4 folder written from the run itself, then perturbed."""
    m.run_lilq(m.Problem('burgers', 36, ref), tmp_path, timing=False)
    rows = _rows(m.run_dir_of(tmp_path, 'burgers', 36, 'lilq') / 'history.csv')
    rec = json.loads((m.run_dir_of(tmp_path, 'burgers', 36, 'lilq') / 'run.json').read_text())
    p4 = tmp_path / 'p4' / p41.ITEMS['1a'] / 'p6'
    p4.mkdir(parents=True)

    def write(scale):
        with open(p4 / 'iterations.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['k', 'eps_ref', 'stall_rule_fires'])
            w.writeheader()
            for r in rows:
                k = int(r['iteration'])
                w.writerow({'k': k, 'eps_ref': repr(float(r['error']) * (scale if k == 2 else 1.0)),
                            'stall_rule_fires': k == rec['rule_step']})
    write(1.0)
    res = m.c3(tmp_path, tmp_path / 'p4')
    assert res['passed'] and res['rows'][0]['max_rel_diff_error'] == 0.0
    write(1.0 + 1e-6)
    assert not m.c3(tmp_path, tmp_path / 'p4')['passed']
    res = m.c3(tmp_path, tmp_path / 'nowhere')                       # no Package 4 log: a failed row, no crash
    assert not res['passed'] and 'no Package 4 log' in res['rows'][0]['reason']
