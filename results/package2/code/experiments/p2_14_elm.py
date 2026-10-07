"""
Package 2, item 4 (P2-14 and P2-4): the ELM scale sweep, the ELM basis on Kovasznay, the normal-equation control
================================================================================================================

The advisor's instructions of 4 October 2026, Section 7, and his reply of
5 October (Sections 2, 3.1). Untimed.

``sweep`` (Section 7.1, P2-14)
    Table 3's ELM row as a curve: Burgers, the basis study's setting (P = 625
    tanh random features, the equispaced 73 x 73 grid, 50 outer iterations,
    weighted rows; ``experiments/run_burgers_basis_comparison.py``), hidden
    weights and biases U(-sigma, sigma) (``lilq.basis.ELMBasis2D_Uniform``),
    sigma in {0.03, ..., 10}, seeds 0-4. Per run: the final ||R||_h^2,
    ``eps_ref`` against the Cole-Hopf reference, kappa_raw, kappa_retained, the
    SVD rank, the columns ``gelsy`` retains, ||beta||_2, and the iteration at
    which the loss target of Table 2/3 (5e-9 at P = 625) is first met, if it
    is. Two more runs reproduce Table 3's ELM rows (seed 42, sigma =
    sqrt(6/627) and 1/sqrt(2)) as a check of the harness. Writes
    ``P2_14_elm_sweep/sweep.csv`` and ``figures/elm_sweep.{pdf,png}`` (error
    and rank against sigma, median with the min-max band).
``kovasznay`` (Sections 7.2, 7.3, P2-4)
    Kovasznay with one tanh random-feature basis of 600 neurons shared by u,
    v, p (``basis: shared``; 1,800 coefficients), inputs scaled to [-1, 1]^2,
    from zero, on the paper's P = 1,875 layout (5,564 rows), the paper's
    stopping rule, K_max = 60. A short sweep, sigma in {0.3, 1, 3}, seed 0,
    QR, picks the best sigma by the final ||R||_h (not by test error). At
    that sigma, seeds 0-4 are solved by QR (``gelsy``) and by the two
    normal-equation variants of the reply (``problems.kovasznay``):

    - ``normal_shifted``: Cholesky of A^T A + s I, s the smallest of
      {1e-16, ..., 1e-8} x tr(A^T A) / P that factors at the first iteration,
      then fixed;
    - ``normal_eigh``: the pseudo-inverse through the eigendecomposition of
      A^T A, the eigenvalues below P lambda_max eps_mach dropped.

    The same three solvers run on the surrogate, the paper's Chebyshev basis
    at P = 300 on its grid. Check C5 there: both variants agree with QR at
    k = 1 to 1e-8. The result is each variant's error history against QR's
    on the same basis. Writes ``P2_4_elm_kovasznay/``:

    - ``sigma_sweep/``;
    - ``<solver>/seed<s>/`` and ``chebyshev_P300/<solver>/``, each with
      ``run.json`` and ``iterations.csv`` (the normal runs with
      ``kappa_AtA`` and ``shift_or_rank`` per iteration);
    - ``check_c5.json``, ``summary.csv`` and ``figures/normal_vs_qr.{pdf,png}``.

Usage::

    python experiments/p2_14_elm.py sweep --out <stage root> --reference-dir <dir with burgers_cole_hopf.npz> \\
        [--package1 <package1>]
    python experiments/p2_14_elm.py kovasznay --out <stage root>
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from lilq.basis import ELMBasis2D_Uniform
from lilq.iteration_log import IterationLogger, last_solve_row
from lilq.provenance import save_provenance
from lilq.source_lock import current_commit

SIGMAS = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.707, 1, 2, 3, 5, 10)
SEEDS = (0, 1, 2, 3, 4)
TARGET_625 = 5e-9                               # Table 2/3's loss target at P = 625 (TARGET_LOSSES[25])
TABLE3_ROWS = {'elm': np.sqrt(6.0 / 627.0), 'elm_default': 1.0 / np.sqrt(2.0)}   # seed 42
KOV_SIGMAS = (0.3, 1.0, 3.0)
KOV_NEURONS, KOV_LAYOUT_P = 600, 1875


def _e(v):
    return f'{v:.1e}' if isinstance(v, float) else (str(v) or '-')


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- 7.1, the Burgers sweep

def burgers_run(sigma, seed, ref, k_max=50):
    """One run of the basis study's ELM row with U(-sigma, sigma) features."""
    import experiments.run_burgers_basis_comparison as bc
    config = bc.ComparisonConfig(N_x=25, N_t=25, disable_stopping_rule=True, max_quasi_iters=k_max)
    pts = bc.generate_collocation(config)
    basis = ELMBasis2D_Uniform(config.N_x * config.N_t, config.x_domain, (0.0, config.T_final), sigma=sigma, seed=seed)
    logger = IterationLogger()
    res = bc.solve_lilq_burgers_comparison('elm', basis, config, pts, verbose=False, iteration_logger=logger)
    beta = res['coefficients']
    rows = logger.rows
    cond = last_solve_row(rows)
    R2 = [r['norm_R_h'] ** 2 for r in rows]
    (x, t), u_ref = ref
    X, T = np.meshgrid(x, t, indexing='ij')
    u = (basis.evaluate(X.ravel(), T.ravel()) @ beta).reshape(X.shape)
    return {'sigma': sigma, 'seed': seed, 'final_R_h_squared': R2[-1],
            'eps_ref': float(np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)),
            'kappa_raw': cond['kappa_raw'], 'kappa_retained': cond['kappa_retained'],
            'rank_svd': cond['num_rank_svd'], 'rank_gelsy': cond['num_rank_gelsy'],
            'norm_beta': float(np.linalg.norm(beta)),
            'k_target': next((int(r['k']) for r in rows if r['norm_R_h'] ** 2 < TARGET_625), ''),
            'iterations': len(rows) - 1}


def _burgers_reference(reference_dir):
    from lilq.references import load_reference, reference_path
    axes, u, _ = load_reference(reference_path(reference_dir, 'burgers'))
    return axes, u


def sweep(out_root, reference_dir, package1=None, sigmas=SIGMAS, seeds=SEEDS, k_max=50, check_table3=True):
    out = Path(out_root) / 'P2_14_elm_sweep'
    ref = _burgers_reference(reference_dir)
    rows = []
    for sigma in sigmas:
        for seed in seeds:
            r = burgers_run(sigma, seed, ref, k_max)
            rows.append({'set': 'sweep', **r})
            print(f"  sigma {sigma:6.3f} seed {seed}: ||R||^2 {r['final_R_h_squared']:.2e}, eps_ref {r['eps_ref']:.2e}, "
                  f"rank {r['rank_svd']} (gelsy {r['rank_gelsy']})")
    if check_table3:
        table3 = {}
        if package1:
            with open(Path(package1) / 'B_instrumentation' / 'basis_study' / 'table3_basis_study.csv', newline='') as f:
                table3 = {r['basis_key']: r for r in csv.DictReader(f)}
        for key, sigma in TABLE3_ROWS.items():
            r = burgers_run(sigma, 42, ref, k_max)
            t3 = table3.get(key)
            rows.append({'set': f'table3_{key}', **r,
                         'table3_R_h_squared': float(t3['final_R_h_squared']) if t3 else '',
                         'table3_rank_svd': int(t3['num_rank_svd']) if t3 else ''})
    for r in rows:
        r.setdefault('table3_R_h_squared', '')
        r.setdefault('table3_rank_svd', '')
    _write(out / 'sweep.csv', rows)
    save_provenance(out)
    sweep_figure(out, [r for r in rows if r['set'] == 'sweep'])
    return rows


def sweep_figure(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sig = sorted({r['sigma'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, label in ((axes[0], 'eps_ref', r'$\varepsilon_{\mathrm{ref}}$ (Cole-Hopf)'),
                           (axes[1], 'rank_svd', 'numerical rank (SVD), of 625')):
        v = [[float(r[key]) for r in rows if r['sigma'] == s] for s in sig]
        med, lo, hi = [np.median(x) for x in v], [min(x) for x in v], [max(x) for x in v]
        ax.fill_between(sig, lo, hi, alpha=0.25)
        ax.plot(sig, med, 'o-', ms=4)
        ax.set_xscale('log')
        if key == 'eps_ref':
            ax.set_yscale('log')
        ax.set_xlabel(r'hidden-weight scale $\sigma$')
        ax.set_ylabel(label)
        ax.grid(True, which='both', alpha=0.3)
    fig.suptitle('Burgers, P = 625 tanh random features: median over seeds 0-4, min-max band', fontsize=9)
    fig.tight_layout()
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures' / 'elm_sweep.pdf')
    fig.savefig(out / 'figures' / 'elm_sweep.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 7.2, 7.3, Kovasznay

SOLVERS = {'gelsy': 'qr', 'normal_shifted': 'normal_shifted', 'normal_eigh': 'normal_eigh'}   # solver: label


def kovasznay_config(sigma, seed, solver='gelsy', k_max=None):
    from experiments.run_kovasznay import K_RATIO, MAX_ITER, TOL
    from problems.kovasznay import KovasznayConfig
    return KovasznayConfig(N_x=25, N_y=25, k_ratio=K_RATIO, max_iter=MAX_ITER if k_max is None else k_max, tol=TOL,
                           basis_type='elm_uniform', elm_neurons=KOV_NEURONS, elm_sigma=sigma, elm_seed=seed,
                           layout_P=KOV_LAYOUT_P, linear_solver=solver)


def kovasznay_run(config, run_dir, basis='shared'):
    """One run, logged in full. For the normal-equation variants,
    ``iterations.csv`` gains ``kappa_AtA`` and ``shift_or_rank`` on every row
    with a solve (the last row, the final iterate, has none)."""
    from problems.kovasznay import solve_kovasznay
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    r = solve_kovasznay(config, verbose=False, iteration_logger=logger, run_json_path=run_dir / 'run.json')
    logger.to_csv(run_dir / 'iterations.csv')
    h = r['history']
    if config.linear_solver != 'gelsy':
        with open(run_dir / 'iterations.csv', newline='') as f:
            rows = list(csv.DictReader(f))
        for i, row in enumerate(rows):
            row['kappa_AtA'] = h['kappa_AtA'][i] if i < len(h['kappa_AtA']) else ''
            row['shift_or_rank'] = h['normal_shift_or_rank'][i] if i < len(h['normal_shift_or_rank']) else ''
        _write(run_dir / 'iterations.csv', rows)
    save_provenance(run_dir)
    final = logger.rows[-1]
    cond = last_solve_row(logger.rows) if len(logger.rows) > 1 else None
    if r['normal_failed']:
        reason = 'cholesky_failed'
    else:
        reason = 'tolerance' if h['coeff_change'] and h['coeff_change'][-1] < config.tol else 'k_max'
    shift_or_rank = [v for v in h['normal_shift_or_rank'] if v is not None]
    return {'solver': 'qr' if config.linear_solver == 'gelsy' else 'normal', 'variant': SOLVERS[config.linear_solver],
            'basis': basis, 'seed': config.elm_seed if basis == 'shared' else '',
            'sigma': config.elm_sigma if basis == 'shared' else '', 'k_stop': len(h['iteration']), 'reason': reason,
            'eps_u': final['eps_u'], 'eps_v': final['eps_v'], 'eps_p_meanfree': final['eps_p_meanfree'],
            'kappa_A': cond['kappa'] if cond else '', 'rank': cond['num_rank_svd'] if cond else '',
            'final_norm_R_h': final['norm_R_h'],
            # normal_shifted: the factor of tr(A^T A) / P, fixed; normal_eigh: the eigenvalues kept at the last solve
            'shift_or_rank': shift_or_rank[-1] if shift_or_rank else '',
            'kappa_AtA_last': h['kappa_AtA'][-1] if h['kappa_AtA'] else ''}


def _chebyshev_config(solver='gelsy', k_max=None):
    """The surrogate: the paper's Chebyshev basis at P = 300 on its grid."""
    from experiments.run_kovasznay import K_RATIO, MAX_ITER, TOL
    from problems.kovasznay import KovasznayConfig
    return KovasznayConfig(N_x=10, N_y=10, k_ratio=K_RATIO, max_iter=MAX_ITER if k_max is None else k_max, tol=TOL,
                           linear_solver=solver)


def check_c5():
    """C5: on the surrogate, each normal-equation variant agrees with QR at
    k = 1 (the first solve) to 1e-8 relative."""
    from problems.kovasznay import solve_kovasznay

    def theta(solver):
        r = solve_kovasznay(_chebyshev_config(solver, k_max=1), verbose=False, diagnostics=solver != 'gelsy')
        return np.concatenate([r[f'theta_{f}'] for f in 'uvp']), r['history']

    tq, _ = theta('gelsy')
    variants = {}
    for solver in ('normal_shifted', 'normal_eigh'):
        tn, h = theta(solver)
        rel = float(np.linalg.norm(tn - tq) / np.linalg.norm(tq))
        variants[solver] = {'rel_difference_k1': rel, 'shift_or_rank': h['normal_shift_or_rank'][0],
                            'kappa_AtA': h['kappa_AtA'][0], 'passed': rel <= 1e-8}
    return {'check': 'C5', 'basis': 'Chebyshev, P = 300', 'tolerance': 1e-8, 'variants': variants,
            'passed': all(v['passed'] for v in variants.values()), 'commit': current_commit()}


def normal_vs_qr_figure(out, runs):
    """Each variant's error history against QR's: one panel per basis (the
    surrogate, then the ELM seeds)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = list(dict.fromkeys(p for p, _ in runs))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for (p, solver), run_dir in runs.items():
            if p != panel:
                continue
            with open(run_dir / 'iterations.csv', newline='') as f:
                rows = [r for r in csv.DictReader(f) if r['eps_u'] not in ('', None)]
            ax.semilogy([int(r['k']) for r in rows], [float(r['eps_u']) for r in rows], label=SOLVERS[solver])
        ax.set(title=panel, xlabel='k')
    axes[0][0].set_ylabel('relative L2 error of u')
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures' / 'normal_vs_qr.pdf')
    fig.savefig(out / 'figures' / 'normal_vs_qr.png', dpi=150)
    plt.close(fig)


def kovasznay(out_root, sigmas=KOV_SIGMAS, seeds=SEEDS, k_max=None):
    out = Path(out_root) / 'P2_4_elm_kovasznay'
    out.mkdir(parents=True, exist_ok=True)
    c5 = check_c5()
    (out / 'check_c5.json').write_text(json.dumps(c5, indent=2))
    print('  C5 (Chebyshev P = 300, k = 1, against QR): ' + ', '.join(
        f"{s} {v['rel_difference_k1']:.1e} ({v['shift_or_rank']})" for s, v in c5['variants'].items())
        + f" -> {'passed' if c5['passed'] else 'FAILED'}")
    rows, runs = [], {}
    for solver, label in SOLVERS.items():
        run_dir = out / 'chebyshev_P300' / label
        r = kovasznay_run(_chebyshev_config(solver, k_max), run_dir, basis='chebyshev')
        rows.append(r)
        runs[('Chebyshev P = 300', solver)] = run_dir
        print(f"  Chebyshev {label:14s}: {r['k_stop']} iterations ({r['reason']}), E_u {r['eps_u']:.2e}, "
              f"kappa(A) {_e(r['kappa_A'])}, shift or rank {r['shift_or_rank'] or '-'}")
    sweep_rows = [kovasznay_run(kovasznay_config(s, 0, k_max=k_max), out / 'sigma_sweep' / f'sigma{s:g}')
                  for s in sigmas]
    best = min(sweep_rows, key=lambda r: r['final_norm_R_h'])['sigma']
    print('  sigma sweep (seed 0, QR): ' + ', '.join(f"{r['sigma']:g}: ||R||_h {r['final_norm_R_h']:.2e}"
                                                    for r in sweep_rows) + f"; best {best:g}")
    for solver, label in SOLVERS.items():
        for seed in seeds:
            run_dir = out / label / f'seed{seed}'
            r = kovasznay_run(kovasznay_config(best, seed, solver, k_max), run_dir)
            rows.append(r)
            runs[(f'ELM seed {seed}', solver)] = run_dir
            print(f"  {label:14s} seed {seed}: {r['k_stop']} iterations ({r['reason']}), E_u {r['eps_u']:.2e}, "
                  f"kappa(A) {_e(r['kappa_A'])}, rank {r['rank']}, shift or rank {r['shift_or_rank'] or '-'}")
    _write(out / 'summary.csv', rows)
    normal_vs_qr_figure(out, runs)
    _write(out / 'sigma_sweep.csv', [{**r, 'chosen': r['sigma'] == best} for r in sweep_rows])
    save_provenance(out)
    return c5, sweep_rows, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 2, item 4: the ELM sweep, ELM on Kovasznay, the normal equations.")
    ap.add_argument('stage', choices=('sweep', 'kovasznay'))
    ap.add_argument('--out', required=True, help='the stage root')
    ap.add_argument('--reference-dir', default=None, help='the folder with burgers_cole_hopf.npz (sweep)')
    ap.add_argument('--package1', default=None, help="package1, to put Table 3's ELM rows beside the check runs")
    args = ap.parse_args(argv)
    if args.stage == 'sweep':
        sweep(args.out, args.reference_dir, args.package1)
    else:
        kovasznay(args.out)


if __name__ == '__main__':
    main()
