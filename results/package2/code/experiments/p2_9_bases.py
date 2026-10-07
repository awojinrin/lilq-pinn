"""
Package 2, item 6b (P2-9): boundary-conforming bases on Bratu and Buckley-Leverett
=================================================================================

The advisor's instructions of 4 October 2026, Section 9.2. The manuscript's
Section 7.4 recommends sine families in directions with homogeneous
Dirichlet data and Chebyshev elsewhere; the reported Bratu and BL runs
predate it. LiL-Q, K_max = 60, on the paper's grids and weights:

**Bratu** (P = 25, 100, 225; from zero, as the paper):
- ``sin_sin``: sin(i pi x) sin(j pi y), i, j <= p_d (``create_basis_2d('sin_sin')``);
  every basis function vanishes on the boundary, so the paper's weak
  boundary rows are zero rows;
- ``chebyshev_weak``: T_i(2x - 1) T_j(2y - 1), i, j < p_d, with the paper's weak
  boundary rows (lambda_bc = 10): the trial space of Section 6.3 of the
  package, on the paper's random tensor grid (seed 42, N/P about 10) instead
  of CGL.

Everything else is ``run_bratu.paper_setup``. ``eps_ref`` is against the p = 48
reference.

**Buckley-Leverett, both cases** (P = 64, 256, 576, 1,024), with the lifted
sine basis ``lifted_sine_x(1-x)`` (the labels are the advisor's, reply of
5 October, Section 2; run folders ``lifted_sine_x1mx``):

    S = (1 - x) + x (1 - x) sum_ij beta_ij sin(i pi x) phi_j(t),

where phi_j is the paper's mixed Fourier set in t for the viscous case
(``Fourier1D(mode='both')``) and the cosine set in t for gravity
(``Fourier1D(mode='cos')``), i, j = 1..p_d. Both boundary conditions,
S(0, t) = 1 and S(1, t) = 0, then hold identically (the paper's boundary
rows are zero rows); the initial condition stays a weighted row.

The run is the paper's LiL-Q:
- the same quasilinearization, weights and loss, built by the paper's
  ``_make_lil_q_system_fn`` / ``_make_lil_nonlinear_loss_fn``;
- the lifting enters as one extra column with its coefficient fixed at 1,
  moved to the right-hand side: A beta = b - A_L.

The tracker's B2 check, A beta - b against the nonlinear residual,
verifies the algebra on every run.

From both initial guesses, as in B8:
- ``zero``: beta = 0, i.e. S = 1 - x;
- ``ic``: the initial profile extended in time, fitted by least squares
  (``pretrain_lil``, the paper's fit) after subtracting the lifting.

``eps_ref`` is against the finite-difference reference at the paper's
nu = 0.1, refined to 1e-6 as in item 6a
(``reference/bl_fd_ref_<case>_nu0.1.npz``, made here if missing).

**Comparison rows**, not asked for but on the same footing:
- **The paper's bases**, on the same references and machine. Bratu: the
  paper's mixed Fourier, from zero. BL: the paper's basis, both guesses, as
  B8. Their ``basis`` column reads ``paper (<type>)``.
- **BL ``lifted_sine_plain``:** the same lifting with plain sin(i pi x) phi_j(t),
  without the factor x(1 - x). The advisor's reply asks for both, and the
  manuscript will state the plain-sine result.

Why the second one. The factor makes the space a sine series of
w / (x(1 - x)), w = S - (1 - x), which does not vanish at the ends, so it
converges slowly. The best-approximation distance of the viscous reference
is 20-70 times larger than with plain sines (laptop: 5.2e-3 against 1.0e-4
at P = 576; DECISIONS.md, batch 2).

Every row also carries ``delta_P``: the relative L2 distance of the reference
to the trial space on the test grid (a least-squares fit; the lifting
subtracted for the lifted bases), and ``eps_over_delta`` =
eps_ref_stop / delta_P.

**Per run** (``rows.csv``: benchmark, case, basis, P, guess, k_target, k_rule,
eps_ref_stop, eps_ref_60, kappa, rank, plus the stop, the rule's class and the
boundary violation):
- ``k_target``: the first iterate meeting the paper's loss target;
- the stop: that iterate, or 60;
- ``k_rule`` and its A/C class: the termination rule (n_s = 2,
  tau_chi = 0.1, tau_r = 0.01) on the log;
- kappa and rank at the last solve;
- ``bc_max_violation``: the largest |S(0, t) - 1|, |S(1, t)| (BL) or |u| on
  the boundary (Bratu) at k = 60, which must be round-off for the
  boundary-conforming bases.

The iteration is deterministic, so the K_max = 60 run (target 0) contains the
paper rule's stop.

**Outputs** under ``P2_9_bases/``:
- ``<benchmark>_<case>_<basis>_P<P>_<guess>/{run.json, iterations.csv}``;
- ``rows.csv``;
- ``references.json`` (the refinement records of the two BL references).

Usage::

    python experiments/p2_9_bases.py --out <stage root> [--benchmarks bratu bl]
    python experiments/p2_9_bases.py --references-only --out <stage root>
"""

import argparse
import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from lilq.source_lock import current_commit

K_MAX = 60
BRATU_SIZES = (5, 10, 15)
BL_SIZES = (8, 16, 24, 32)
BRATU_BASES = {'sin_sin': 'sin_sin', 'chebyshev_weak': 'chebyshev'}
RULE = {'n_s': 2, 'tau_chi': 0.1, 'tau_r': 0.01}
NU_PAPER = 0.1


# ---------------------------------------------------------------- the lifted sine basis

class LiftedSineBasis:
    """x (1 - x) sin(i pi xi) phi_j(t) (``weighted``, the specified basis) or
    sin(i pi xi) phi_j(t) (not weighted), i, j = 1..n, coefficients i outer
    (the ordering of ``TensorProductBasis2D``); ``mode_t`` 'both' (the mixed
    Fourier set) or 'cos'. The lifting 1 - x is not part of it. Assumes
    x_domain = (0, 1), which is the BL domain."""

    def __init__(self, n, t_domain, mode_t, weighted=True):
        from lilq.basis import Fourier1D
        self.sx = Fourier1D(n, (0.0, 1.0), mode='sin')
        self.ft = Fourier1D(n, t_domain, mode=mode_t)
        self.n_basis = self.sx.n_basis * self.ft.n_basis
        self.weighted = weighted

    def _x(self, x, order):
        x = np.asarray(x, float)
        if not self.weighted:
            return self.sx.evaluate(x) if order == 0 else self.sx.derivative(x, order=order)
        b, b1, b2 = x * (1 - x), 1 - 2 * x, -2.0
        s = self.sx.evaluate(x)
        if order == 0:
            return b[:, None] * s
        s1 = self.sx.derivative(x, order=1)
        if order == 1:
            return b1[:, None] * s + b[:, None] * s1
        s2 = self.sx.derivative(x, order=2)
        return b2 * s + 2 * b1[:, None] * s1 + b[:, None] * s2

    def _t(self, t, order):
        return self.ft.evaluate(np.asarray(t, float)) if order == 0 else self.ft.derivative(np.asarray(t, float), order=order)

    def derivative(self, x, t, dx=0, dy=0):
        X, T = self._x(np.ravel(x), dx), self._t(np.ravel(t), dy)
        return (X[:, :, None] * T[:, None, :]).reshape(len(X), -1)

    def evaluate(self, x, t):
        return self.derivative(x, t)


def lifting(x, dx=0):
    x = np.asarray(x, float)
    return 1.0 - x if dx == 0 else (-np.ones_like(x) if dx == 1 else np.zeros_like(x))


# ---------------------------------------------------------------- Bratu

def bratu_run(basis_label, N, out, ref_dir):
    """One Bratu run, K_max = 60 with the target off; ``basis_label`` a key of
    BRATU_BASES or 'paper'."""
    from experiments.run_bratu import TARGET_LOSSES, paper_setup
    from lilq.iteration_log import IterationLogger
    from problems.bratu import run_lil_q
    config, opt = paper_setup(N)
    basis_type = config.basis_type if basis_label == 'paper' else BRATU_BASES[basis_label]
    config = dataclasses.replace(config, basis_type=basis_type)
    opt = dataclasses.replace(opt, R_tol=0.0, max_quasi_iters_lil=K_MAX)
    label = f'paper ({basis_type})' if basis_label == 'paper' else basis_label
    run_dir = Path(out) / f"bratu_{basis_label}_P{N * N}_zero"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    basis, c, _, _ = run_lil_q(config, opt, verbose=False, iteration_logger=logger,
                               run_json_path=run_dir / 'run.json', reference_npz=Path(ref_dir) / 'bratu_ref_p48.npz')
    logger.to_csv(run_dir / 'iterations.csv')
    g = np.linspace(0, 1, 201)
    edge = np.concatenate([basis.evaluate(g, 0 * g) @ c, basis.evaluate(g, 0 * g + 1) @ c,
                           basis.evaluate(0 * g, g) @ c, basis.evaluate(0 * g + 1, g) @ c])
    ref = np.load(Path(ref_dir) / 'bratu_ref_p48.npz')
    dP = delta_P(basis.evaluate, ref['x'], ref['y'], ref['u'])
    return row('bratu', '', label, N, 'zero', run_dir, TARGET_LOSSES.get(N), float(np.abs(edge).max()), dP)


# ---------------------------------------------------------------- Buckley-Leverett

# the lifted bases: label -> (run folder name, with the factor x(1 - x))
LIFTED = {'lifted_sine_x(1-x)': ('lifted_sine_x1mx', True), 'lifted_sine_plain': ('lifted_sine_plain', False)}


def bl_run(case, basis_label, N, guess, out, ref_dir):
    """One BL run, K_max = 60 with the target off. ``basis_label``
    'lifted_sine_x(1-x)' (the specified basis), 'lifted_sine_plain' or 'paper';
    ``guess`` 'zero' or 'ic'."""
    if basis_label == 'paper':
        return _bl_paper_run(case, N, guess, out, ref_dir)
    from experiments.run_bl import GRAVITY_TARGET_LOSSES, TARGET_LOSSES, paper_setup
    from lilq.collocation import generate_collocation_points_2d
    from lilq.iteration_log import IterationLogger
    from lilq.pretraining import pretrain_lil
    from lilq.provenance import capture_blas_thread_env
    from lilq.run_metadata import build_run_metadata, first_stall_iteration, write_run_json
    from lilq.solvers import solve_lil_q
    import problems.buckley_leverett as bl
    gravity = case == 'gravity'
    config, opt = paper_setup(N, gravity)
    config = dataclasses.replace(config, initial_guess=guess)
    assert tuple(config.x_domain) == (0.0, 1.0) and (config.S_left, config.S_right) == (1.0, 0.0)
    physics = bl.BLPhysics(config)
    T = config.T_final
    folder, weighted = LIFTED[basis_label]
    basis = LiftedSineBasis(N, (0.0, T), 'cos' if gravity else 'both', weighted=weighted)
    # the paper's grid (run_lil_q's call) and matrices
    np.random.seed(config.seed)
    pts = generate_collocation_points_2d(config.x_domain, (0, T), config.N_x, config.N_t, k_ratio=config.k_ratio,
                                         collocation_ratios=(0.9, 0.05, 0.05), has_initial_condition=True,
                                         seed=config.seed, sampling=config.sampling)
    (A_u, A_ux, A_ut, A_uxx, A_ic, A_bl, A_br, ic_t, bl_t, br_t) = bl._prepare_lil_matrices(config, physics, basis, pts)
    col = lambda x, dx=0: lifting(x, dx)[:, None]  # noqa: E731
    xp, xi = pts['x_pde'], pts['x_ic']
    aug = (np.hstack([A_u, col(xp)]), np.hstack([A_ux, col(xp, 1)]), np.hstack([A_ut, 0 * col(xp)]),
           np.hstack([A_uxx, col(xp, 2)]), np.hstack([A_ic, col(xi)]),
           np.hstack([A_bl, col(pts['x_bc_left'])]), np.hstack([A_br, col(pts['x_bc_right'])]))
    n_pde, n_ic = pts['n_pde'], pts['n_ic']
    n_bl, n_br = len(pts['x_bc_left']), len(pts['x_bc_right'])
    args = (*aug, ic_t, bl_t, br_t, physics, opt.lambda_pde, opt.lambda_ic, opt.lambda_bc)
    system_aug = bl._make_lil_q_system_fn(*args, n_pde, n_ic, n_bl, n_br)
    loss_aug = bl._make_lil_nonlinear_loss_fn(*args)
    resid_aug = bl._make_lil_residual_vector_fn(*args, n_pde, n_ic, n_bl, n_br)
    one = lambda beta: np.concatenate([beta, [1.0]])  # noqa: E731

    def system_fn(beta):
        A, b = system_aug(one(beta))
        return A[:, :-1], b - A[:, -1]

    # the initial guess: beta = 0 (S = 1 - x), or the paper's fit of the initial profile minus the lifting
    if guess == 'zero':
        init = np.zeros(basis.n_basis)
    else:
        init, _ = pretrain_lil(basis, lambda x, t: physics.initial_condition(x) - lifting(x), config.x_domain, (0, T),
                               n_grid=opt.pretrain_grid, verbose=False)
    eps = _bl_error_fn(basis, config, Path(ref_dir) / f'bl_fd_ref_{case}_nu0.1.npz')
    run_dir = Path(out) / f'bl_{case}_{folder}_P{N * N}_{guess}'
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    coeffs, _, summary = solve_lil_q(system_fn, lambda beta: loss_aug(one(beta)), init, max_quasi_iters=K_MAX,
                                     R_tol=0.0, verbose=False, iteration_logger=logger,
                                     compute_residual_vector_fn=lambda beta: resid_aug(one(beta)),
                                     n_interior_rows=n_pde, interior_weight=np.sqrt(opt.lambda_pde / n_pde),
                                     test_error_fn=eps)
    logger.to_csv(run_dir / 'iterations.csv')
    thread_env = capture_blas_thread_env()
    write_run_json(run_dir / 'run.json', build_run_metadata(
        N_total=n_pde + n_ic + n_bl + n_br, N_composition={'pde': n_pde, 'ic': n_ic, 'bc_left': n_bl, 'bc_right': n_br},
        P_total=basis.n_basis, P_composition={'u': basis.n_basis},
        row_weights={'pde': float(np.sqrt(opt.lambda_pde / n_pde)), 'ic': float(np.sqrt(opt.lambda_ic / n_ic)),
                     'bc_left': float(np.sqrt(opt.lambda_bc / n_bl)), 'bc_right': float(np.sqrt(opt.lambda_bc / n_br))},
        collocation_construction={'method': 'random-tensor', 'seed': config.seed, 'N_x': config.N_x,
                                  'N_t': config.N_t, 'k_ratio': config.k_ratio},
        basis_description={'family': ('lifted sine: S = (1 - x) + x(1 - x) sum beta_ij sin(i pi x) phi_j(t)'
                                      if basis.weighted else
                                      'lifted plain sine: S = (1 - x) + sum beta_ij sin(i pi x) phi_j(t)'),
                           'phi_t': 'cosine set' if gravity else 'mixed Fourier set', 'modes': [N, N]},
        initial_coefficients='zero (S = 1 - x)' if guess == 'zero' else
        'initial profile minus the lifting, least-squares fitted (pretrain_lil)',
        solver_driver='gelsy', rcond=float(np.finfo(float).eps),
        stopping_rule={'type': 'none: K_max iterations; the paper target is read from the log'}, K_max=K_MAX,
        stopping_reason='iteration_cap', first_stall_iteration=first_stall_iteration(logger.rows), device='cpu',
        thread_count=int(thread_env.get('OMP_NUM_THREADS') or os.cpu_count() or 1),
        b2_check=summary.get('b2_check'), kappa_qr_raw_ratio=summary.get('kappa_qr_raw_ratio')))
    tt = np.linspace(0, T, 201)
    S0 = basis.evaluate(0 * tt, tt) @ coeffs + lifting(0 * tt)
    S1 = basis.evaluate(0 * tt + 1, tt) @ coeffs + lifting(0 * tt + 1)
    viol = float(max(np.abs(S0 - 1).max(), np.abs(S1).max()))
    target = (GRAVITY_TARGET_LOSSES if gravity else TARGET_LOSSES).get(N)
    ref = np.load(Path(ref_dir) / f'bl_fd_ref_{case}_nu0.1.npz')
    dP = delta_P(basis.evaluate, ref['x'], ref['t'], ref['u'], lifted=True)
    return row('bl', case, basis_label, N, guess, run_dir, target, viol, dP)


def _bl_error_fn(basis, config, reference_npz):
    """``eps_u`` and ``maxerr_u`` against Package 1's reference (as the paper's
    log) and ``eps_ref`` against the refined one, for the lifted field."""
    from lilq.references import load_reference
    from problems.buckley_leverett import reference_solution
    (x, t), u_ref, _ = load_reference(reference_npz)
    X, T = np.meshgrid(x, t, indexing='ij')
    Phi, L = basis.evaluate(X.ravel(), T.ravel()), lifting(X.ravel())
    S_p1 = np.asarray(reference_solution(config)).ravel()
    u_ref = u_ref.ravel()

    def errors(beta):
        S = L + Phi @ beta
        return {'eps_u': float(np.linalg.norm(S - S_p1) / np.linalg.norm(S_p1)),
                'maxerr_u': float(np.abs(S - S_p1).max()),
                'eps_ref': float(np.linalg.norm(S - u_ref) / np.linalg.norm(u_ref))}
    return errors


def _bl_paper_run(case, N, guess, out, ref_dir):
    from experiments.run_bl import GRAVITY_TARGET_LOSSES, TARGET_LOSSES, paper_setup
    from lilq.iteration_log import IterationLogger
    from problems.buckley_leverett import run_lil_q
    config, opt = paper_setup(N, case == 'gravity')
    config = dataclasses.replace(config, initial_guess=guess)
    opt = dataclasses.replace(opt, R_tol=0.0, max_quasi_iters_lil=K_MAX)
    run_dir = Path(out) / f'bl_{case}_paper_P{N * N}_{guess}'
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    basis, c, _, _ = run_lil_q(config, opt, verbose=False, iteration_logger=logger, run_json_path=run_dir / 'run.json',
                               reference_npz=Path(ref_dir) / f'bl_fd_ref_{case}_nu0.1.npz')
    logger.to_csv(run_dir / 'iterations.csv')
    tt = np.linspace(0, config.T_final, 201)
    viol = float(max(np.abs(basis.evaluate(0 * tt, tt) @ c - 1).max(), np.abs(basis.evaluate(0 * tt + 1, tt) @ c).max()))
    target = (GRAVITY_TARGET_LOSSES if case == 'gravity' else TARGET_LOSSES).get(N)
    ref = np.load(Path(ref_dir) / f'bl_fd_ref_{case}_nu0.1.npz')
    dP = delta_P(basis.evaluate, ref['x'], ref['t'], ref['u'])
    return row('bl', case, f'paper ({config.basis_type})', N, guess, run_dir, target, viol, dP)


# ---------------------------------------------------------------- rows

def _f(v):
    return float(v) if v not in ('', None) else float('nan')


def delta_P(evaluate, x, y, u, lifted=False):
    """The relative L2 distance of the reference ``u`` (on the grid x by y) to the
    trial space spanned by ``evaluate(x, y)``, by least squares; for a lifted
    basis, the lifting 1 - x is subtracted first."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    Phi = evaluate(X.ravel(), Y.ravel())
    f = u.ravel() - (lifting(X.ravel()) if lifted else 0.0)
    c = np.linalg.lstsq(Phi, f, rcond=None)[0]
    return float(np.linalg.norm(Phi @ c - f) / np.linalg.norm(u))


def row(benchmark, case, basis, N, guess, run_dir, target, bc_violation, dP):
    import experiments.stopping_rule_table as srt
    with open(Path(run_dir) / 'iterations.csv', newline='') as f:
        log = list(csv.DictReader(f))
    d = srt.read_log(Path(run_dir) / 'iterations.csv')
    i, _ = srt.rule_index(d, RULE['n_s'], RULE['tau_chi'], RULE['tau_r'])
    eps = [_f(r['eps_ref']) for r in log]
    k_target = next((int(r['k']) for r in log[1:] if target is not None and _f(r['norm_R_h']) ** 2 < target), None)
    k_stop = k_target if k_target is not None else K_MAX
    last = next(r for r in reversed(log) if r['norm_Rlin_h'] not in ('', None))
    b2 = json.loads((Path(run_dir) / 'run.json').read_text()).get('b2_check') or {}
    return {'benchmark': benchmark, 'case': case, 'basis': basis, 'P': N * N, 'guess': guess,
            'k_target': k_target if k_target is not None else '', 'k_rule': i + 1 if i is not None else '',
            'eps_ref_stop': eps[k_stop], 'eps_ref_60': eps[K_MAX], 'kappa': _f(last['kappa']),
            'rank': int(_f(last['num_rank_svd'])), 'target_mse': target if target is not None else '',
            'stop': 'target' if k_target is not None else 'K_max', 'k_stop': k_stop,
            'rule_class': srt.classify(d, i) if i is not None else 'never',
            'eps_ref_rule': eps[i + 1] if i is not None else '', 'loss_60': _f(log[-1]['norm_R_h']) ** 2,
            'bc_max_violation': bc_violation, 'b2_max_rel_err': b2.get('max_rel_err_over_run', ''),
            'delta_P': dP, 'eps_over_delta': eps[k_stop] / dP}


def ensure_bl_references(out_root):
    """The two nu = 0.1 references (item 6a's refinement); made if missing."""
    import experiments.p2_2_nu_refinement as nr
    records = []
    for case in ('viscous', 'gravity'):
        path = nr.reference_path(Path(out_root) / 'reference', case, NU_PAPER)
        if path.exists():
            records.append({'case': case, 'nu': NU_PAPER, **json.loads(str(np.load(path)['meta']))})
        else:
            records.append(nr.write_reference(out_root, case, NU_PAPER))
    return records


def run_all(out_root, benchmarks=('bratu', 'bl')):
    from lilq.provenance import save_provenance
    out = Path(out_root) / 'P2_9_bases'
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(out_root) / 'reference'
    rows = []
    if 'bratu' in benchmarks:
        if not (ref_dir / 'bratu_ref_p48.npz').exists():
            raise FileNotFoundError(f'{ref_dir / "bratu_ref_p48.npz"}: the Bratu reference (item 1 stage reference)')
        for N in BRATU_SIZES:
            for label in ('sin_sin', 'chebyshev_weak', 'paper'):
                rows.append(bratu_run(label, N, out, ref_dir))
                _print(rows[-1])
    if 'bl' in benchmarks:
        refs = ensure_bl_references(out_root)
        (out / 'references.json').write_text(json.dumps({'references': refs, 'commit': current_commit()}, indent=2,
                                                        default=str))
        for case in ('viscous', 'gravity'):
            for N in BL_SIZES:
                for guess in ('zero', 'ic'):
                    for label in (*LIFTED, 'paper'):
                        rows.append(bl_run(case, label, N, guess, out, ref_dir))
                        _print(rows[-1])
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    save_provenance(out)
    return rows


def _print(r):
    print(f"  {r['benchmark']:5s} {r['case']:7s} {r['basis']:22s} P = {r['P']:4d} {r['guess']:4s}: stop {r['k_stop']:2d} "
          f"({r['stop']}), eps_ref {r['eps_ref_stop']:.2e} / k=60 {r['eps_ref_60']:.2e}, rule {r['k_rule']} "
          f"({r['rule_class']}), kappa {r['kappa']:.1e}, rank {r['rank']}, bc {r['bc_max_violation']:.1e}, "
          f"delta_P {r['delta_P']:.1e}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 2, item 6b: boundary-conforming bases.')
    ap.add_argument('--out', required=True, help='the stage root')
    ap.add_argument('--benchmarks', nargs='+', choices=('bratu', 'bl'), default=['bratu', 'bl'])
    ap.add_argument('--references-only', action='store_true',
                    help="only make the two BL references at the paper's nu = 0.1 (Stage 2's references job)")
    args = ap.parse_args(argv)
    if args.references_only:
        for r in ensure_bl_references(args.out):
            print(f"  BL {r['case']} nu = {r['nu']}: {r['n_intervals']} intervals")
        return
    run_all(args.out, args.benchmarks)


if __name__ == '__main__':
    main()
