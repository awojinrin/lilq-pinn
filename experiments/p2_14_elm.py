"""
Package 2, item 4 (P2-14 and P2-4): the ELM scale sweep, the ELM basis on Kovasznay, the normal-equation control
================================================================================================================

The advisor's instructions of 4 October 2026, Section 7. Untimed.

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
    v, p (1,800 coefficients), inputs scaled to [-1, 1]^2, from zero, on the
    paper's P = 1,875 layout (5,564 rows), the paper's stopping rule, K_max =
    60. First check C5: on the paper's Chebyshev basis at P = 300 the
    normal-equation and QR solves agree at k = 1 to 1e-8. Then a short sweep,
    sigma in {0.3, 1, 3}, seed 0, QR, the best sigma by the final ||R||_h (not
    by test error); at that sigma, seeds 0-4 solved by QR (``gelsy``) and by
    the normal equations (Cholesky, a shift 1e-16 tr / P only if it fails).
    Writes ``P2_4_elm_kovasznay/{sigma_sweep,qr,normal}/.../{run.json,
    iterations.csv}`` (``normal_eq.csv`` with kappa(A^T A) and the shift per
    iteration for the normal runs), ``check_c5.json`` and ``summary.csv``.

Usage::

    python experiments/p2_14_elm.py sweep --out <stage root> --reference-dir <dir with burgers_cole_hopf.npz> \\
        [--package1 <package1>]
    python experiments/p2_14_elm.py kovasznay --out <stage root>
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

def kovasznay_config(sigma, seed, solver='gelsy', k_max=None):
    from experiments.run_kovasznay import K_RATIO, MAX_ITER, TOL
    from problems.kovasznay import KovasznayConfig
    return KovasznayConfig(N_x=25, N_y=25, k_ratio=K_RATIO, max_iter=MAX_ITER if k_max is None else k_max, tol=TOL,
                           basis_type='elm_uniform', elm_neurons=KOV_NEURONS, elm_sigma=sigma, elm_seed=seed,
                           layout_P=KOV_LAYOUT_P, linear_solver=solver)


def kovasznay_run(config, run_dir):
    from problems.kovasznay import solve_kovasznay
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    r = solve_kovasznay(config, verbose=False, iteration_logger=logger, run_json_path=run_dir / 'run.json')
    logger.to_csv(run_dir / 'iterations.csv')
    save_provenance(run_dir)
    h = r['history']
    if config.linear_solver == 'normal':
        # one row per normal-equation solve attempted (the last may be the failed one)
        _write(run_dir / 'normal_eq.csv', [{'k': k, 'cholesky': s, 'kappa_AtA': c}
                                           for k, (s, c) in enumerate(zip(h['cholesky_shift'], h['kappa_AtA']))])
    final = logger.rows[-1]
    cond = last_solve_row(logger.rows) if len(logger.rows) > 1 else None
    if r['normal_failed']:
        reason = 'cholesky_failed'
    else:
        reason = 'tolerance' if h['coeff_change'] and h['coeff_change'][-1] < config.tol else 'k_max'
    return {'solver': 'qr' if config.linear_solver == 'gelsy' else 'normal', 'seed': config.elm_seed,
            'sigma': config.elm_sigma, 'k_stop': len(h['iteration']), 'reason': reason,
            'eps_u': final['eps_u'], 'eps_v': final['eps_v'], 'eps_p_meanfree': final['eps_p_meanfree'],
            'kappa_A': cond['kappa'] if cond else '', 'rank': cond['num_rank_svd'] if cond else '',
            'final_norm_R_h': final['norm_R_h'],
            'cholesky_status': ';'.join(h['cholesky_shift']) if h['cholesky_shift'] else '',
            'kappa_AtA_last': h['kappa_AtA'][-1] if h['kappa_AtA'] else ''}


def check_c5():
    """C5: on the paper's Chebyshev basis at P = 300 the normal-equation and QR
    solves agree at k = 1 to 1e-8 relative."""
    from experiments.run_kovasznay import K_RATIO, TOL
    from problems.kovasznay import KovasznayConfig, solve_kovasznay
    c = KovasznayConfig(N_x=10, N_y=10, k_ratio=K_RATIO, max_iter=1, tol=TOL)
    q = solve_kovasznay(c, verbose=False, diagnostics=False)
    n = solve_kovasznay(dataclasses.replace(c, linear_solver='normal'), verbose=False)
    tq = np.concatenate([q[f'theta_{f}'] for f in 'uvp'])
    tn = np.concatenate([n[f'theta_{f}'] for f in 'uvp'])
    rel = float(np.linalg.norm(tn - tq) / np.linalg.norm(tq))
    return {'check': 'C5', 'basis': 'Chebyshev, P = 300', 'rel_difference_k1': rel, 'tolerance': 1e-8,
            'cholesky_shift': n['history']['cholesky_shift'][0], 'kappa_AtA': n['history']['kappa_AtA'][0],
            'passed': rel <= 1e-8, 'commit': current_commit()}


def kovasznay(out_root, sigmas=KOV_SIGMAS, seeds=SEEDS, k_max=None):
    out = Path(out_root) / 'P2_4_elm_kovasznay'
    out.mkdir(parents=True, exist_ok=True)
    c5 = check_c5()
    (out / 'check_c5.json').write_text(json.dumps(c5, indent=2))
    print(f"  C5: normal vs QR at k = 1 on Chebyshev P = 300: {c5['rel_difference_k1']:.1e} "
          f"({'passed' if c5['passed'] else 'FAILED'})")
    sweep_rows = [kovasznay_run(kovasznay_config(s, 0, k_max=k_max), out / 'sigma_sweep' / f'sigma{s:g}')
                  for s in sigmas]
    best = min(sweep_rows, key=lambda r: r['final_norm_R_h'])['sigma']
    print('  sigma sweep (seed 0, QR): ' + ', '.join(f"{r['sigma']:g}: ||R||_h {r['final_norm_R_h']:.2e}"
                                                    for r in sweep_rows) + f"; best {best:g}")
    rows = []
    for solver, label in (('gelsy', 'qr'), ('normal', 'normal')):
        for seed in seeds:
            r = kovasznay_run(kovasznay_config(best, seed, solver, k_max), out / label / f'seed{seed}')
            rows.append(r)
            print(f"  {label:6s} seed {seed}: {r['k_stop']} iterations ({r['reason']}), E_u {r['eps_u']:.2e}, "
                  f"kappa(A) {r['kappa_A']}, rank {r['rank']}, Cholesky {r['cholesky_status'] or '-'}")
    _write(out / 'summary.csv', rows)
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
