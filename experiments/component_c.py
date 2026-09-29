"""
Component C: oversampling sweep (Package 1 v2.0 Section 5)
==========================================================

Kovasznay P in {300, 1,200} and Bratu P in {100, 225}, bases as in the
paper; N/P in {1, 1.5, 2, 3, 5, 10, 20}; three point distributions:

* ``paper``: the paper's construction (Kovasznay's equispaced tensor grid,
  Bratu's random-tensor grid with seed 42);
* ``cgl``: Chebyshev-Gauss-Lobatto tensor grid;
* ``random``: uniformly random points, seeds 0-4.

4 x 7 x 7 = 196 LiL-Q runs with the full Section 3.1 log, to the paper's
stopping rule. N/P = 1 is included on purpose and may fail; a failing run
is logged with its traceback. For each target N/P, the density factor
``k_ratio`` is the one whose actual row count comes closest (the paper's
proportions of row types kept); the exact counts are recorded. The paper's
minimum point counts (Kovasznay 10 per direction and per edge; Bratu 5 and
10) are lowered to 1 throughout the sweep, so that small N/P are reachable
and the proportions hold at every ratio.

Untimed (Addendum v2.1 Section 2): the laptop's CPU.

Outputs under ``--root`` (the package's ``C_oversampling/``)::

    runs/<benchmark>_P<P>_r<ratio>_<distribution>[_s<seed>]/{iterations.csv, run.json, solution.pt}
    results/oversampling.csv
    figures/<benchmark>_P<P>.pdf, .csv

Usage::

    python experiments/component_c.py --root <package>/C_oversampling
    python experiments/component_c.py --root ... --quick        # smoke test
"""

import argparse
import csv
import dataclasses
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from lilq.iteration_log import IterationLogger, solve_rows
from lilq.provenance import save_provenance
from lilq.run_metadata import first_stall_iteration
from lilq.saved_models import save_solution
from lilq.source_lock import current_commit

BENCHMARKS = (('kovasznay', 10), ('kovasznay', 20), ('bratu', 10), ('bratu', 15))   # (name, N): P = 3N^2 / N^2
RATIOS = (1, 1.5, 2, 3, 5, 10, 20)
DISTRIBUTIONS = (('paper', None), ('cgl', None)) + tuple(('random', s) for s in range(5))
SWEEP_FLOOR = 1
OUTPUT_DIR = Path(_proj) / 'results' / 'C_oversampling'

COLUMNS = ('benchmark', 'P', 'ratio_nominal', 'ratio_actual', 'N_rows', 'N_distinct', 'k_ratio', 'distribution', 'seed',
           'iterations', 'final_norm_Rlin_h', 'final_norm_Rlin_interior', 'final_norm_R_h', 'kappa',
           'kappa_method', 'num_rank_svd', 'num_rank_gelsy', 'first_stall_iteration',
           'eps_u', 'eps_v', 'eps_p', 'eps_p_meanfree', 't_cum_s', 'stopping', 'commit', 'error')


def n_params(benchmark, N):
    return 3 * N * N if benchmark == 'kovasznay' else N * N


def n_rows(benchmark, N, k, floor=SWEEP_FLOOR):
    """Collocation rows the generator produces at density ``k`` (mirrors
    problems.kovasznay._generate_collocation and
    lilq.collocation.generate_collocation_points_2d; tested against them).
    ``floor=None``: the paper's minimums (Kovasznay 10 and 10; Bratu 5 per
    direction, 10 per edge)."""
    P = n_params(benchmark, N)
    if benchmark == 'kovasznay':                 # ratios (0.6, 0.2, 0.2); 3 rows per interior point,
        dim_floor = bc_floor = 10 if floor is None else floor   # 2 per boundary point, 1 pin
        n_dim = max(math.ceil(math.sqrt(k * 0.6 * P / 3)), dim_floor)
        n_bc = max(math.ceil(k * 0.2 * P / 12), bc_floor)
        return 3 * n_dim ** 2 + 8 * n_bc + 1
    dim_floor, bc_floor = (5, 10) if floor is None else (floor, floor)   # ratios (0.85, 0.15)
    n_dim = max(math.ceil(math.sqrt(k * 0.85 * P)), dim_floor)
    n_bc = max(math.ceil(k * 0.15 * P / 4), bc_floor)
    return n_dim ** 2 + 4 * n_bc


def k_for_ratio(benchmark, N, ratio):
    """The density whose row count is closest to ratio x P (smallest such
    k); at the nominal ratio 1, the smallest density with at least P rows
    (Addendum v2.2 Section 2.9: the closest count gave fewer rows than
    unknowns, 97 for Bratu P = 100)."""
    P = n_params(benchmark, N)
    ks = np.round(np.arange(0.02, 40.0, 0.005), 3)
    if ratio == 1:
        k = float(next(k for k in ks if n_rows(benchmark, N, k) >= P))
        return k, n_rows(benchmark, N, k)
    errors = [abs(n_rows(benchmark, N, k) / P - ratio) for k in ks]
    k = float(ks[int(np.argmin(errors))])
    return k, n_rows(benchmark, N, k)


def _config(benchmark, N, k, distribution, seed):
    if benchmark == 'kovasznay':
        from experiments.run_kovasznay import MAX_ITER, TOL
        from problems.kovasznay import KovasznayConfig
        sampling = {'paper': 'uniform', 'cgl': 'cgl', 'random': 'scattered'}[distribution]
        return KovasznayConfig(N_x=N, N_y=N, k_ratio=k, max_iter=MAX_ITER, tol=TOL, sampling=sampling,
                               seed=42 if seed is None else seed, collocation_floor=SWEEP_FLOOR), None
    from experiments.run_bratu import paper_setup
    config, opt = paper_setup(N)
    sampling = {'paper': 'random', 'cgl': 'cgl', 'random': 'scattered'}[distribution]
    return dataclasses.replace(config, k_ratio=k, sampling=sampling, seed=42 if seed is None else seed,
                               collocation_floor=SWEEP_FLOOR), opt


def run_one(benchmark, N, ratio, distribution, seed, run_dir, quick=False):
    k, rows = k_for_ratio(benchmark, N, ratio)
    config, opt = _config(benchmark, N, k, distribution, seed)
    P = n_params(benchmark, N)
    row = dict(benchmark=benchmark, P=P, ratio_nominal=ratio, ratio_actual=rows / P, N_rows=rows, k_ratio=k,
               distribution=distribution, seed='' if seed is None else seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    if benchmark == 'kovasznay':
        from problems.kovasznay import solve_kovasznay
        if quick:
            config = dataclasses.replace(config, max_iter=3)
        r = solve_kovasznay(config, verbose=False, iteration_logger=logger, run_json_path=run_dir / 'run.json',
                            collocation_path=run_dir / 'collocation.npz')
        save_solution(run_dir, {f: (r[f'basis_{f}'], r[f'theta_{f}']) for f in 'uvp'}, config)
        # The solver stops when the relative coefficient change drops below tol
        # (also possible on the last allowed iteration).
        stopping = 'tolerance' if solve_rows(logger.rows)[-1]['rel_dbeta'] < config.tol else 'K_max'
    else:
        from problems.bratu import run_lil_q
        if quick:
            opt = dataclasses.replace(opt, max_quasi_iters_lil=3)
        basis, c, _metrics, summary = run_lil_q(config, opt, verbose=False, iteration_logger=logger,
                                                run_json_path=run_dir / 'run.json',
                                                collocation_path=run_dir / 'collocation.npz')
        save_solution(run_dir, {'u': (basis, c)}, config, opt)
        stopping = 'target' if summary['converged'] else 'K_max'
    logger.to_csv(run_dir / 'iterations.csv')
    solves, last = solve_rows(logger.rows), logger.rows[-1]
    final = solves[-1]
    row.update(iterations=len(solves), final_norm_Rlin_h=final['norm_Rlin_h'],
               final_norm_Rlin_interior=final['norm_Rlin_interior'], final_norm_R_h=last['norm_R_h'],
               kappa=final['kappa'], kappa_method=final['kappa_method'], num_rank_svd=final['num_rank_svd'],
               num_rank_gelsy=final['num_rank_gelsy'], first_stall_iteration=first_stall_iteration(logger.rows),
               eps_u=last.get('eps_u'), eps_v=last.get('eps_v'), eps_p=last.get('eps_p'),
               eps_p_meanfree=last.get('eps_p_meanfree'), t_cum_s=last['t_cum_s'], stopping=stopping,
               N_distinct=int(np.load(run_dir / 'collocation.npz')['n_distinct']), commit=current_commit())
    save_provenance(run_dir)          # hardware.json per run, not only at the sweep root (Addendum v2.2 2.8.4)
    return row


def run_sweep(root, benchmarks=BENCHMARKS, ratios=RATIOS, distributions=DISTRIBUTIONS, quick=False, verbose=True):
    root = Path(root)
    for sub in ('runs', 'results', 'figures'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    csv_path = root / 'results' / 'oversampling.csv'
    rows = []
    if csv_path.exists():
        with open(csv_path, newline='') as f:
            rows = [r for r in csv.DictReader(f) if not r.get('error')]
    done = {(r['benchmark'], int(r['P']), float(r['ratio_nominal']), r['distribution'], str(r['seed'])) for r in rows}
    save_provenance(root)
    np.random.seed(42)
    for benchmark, N in benchmarks:
        for ratio in ratios:
            for distribution, seed in distributions:
                key = (benchmark, n_params(benchmark, N), float(ratio), distribution, '' if seed is None else str(seed))
                if key in done:
                    continue
                name = f"{benchmark}_P{n_params(benchmark, N)}_r{ratio}_{distribution}" + (f"_s{seed}" if seed is not None else '')
                t0 = time.perf_counter()
                try:
                    row = run_one(benchmark, N, ratio, distribution, seed, root / 'runs' / name, quick=quick)
                except Exception:
                    k, n = k_for_ratio(benchmark, N, ratio)
                    row = dict(benchmark=benchmark, P=n_params(benchmark, N), ratio_nominal=ratio,
                               ratio_actual=n / n_params(benchmark, N), N_rows=n, k_ratio=k,
                               distribution=distribution, seed='' if seed is None else seed,
                               stopping='failure', error=traceback.format_exc())
                rows.append(row)
                _write(csv_path, rows)
                if verbose:
                    print(f"  {name:40s} N/P={row['ratio_actual']:.2f} iters={row.get('iterations', '-')} "
                          f"Rlin={row.get('final_norm_Rlin_h', float('nan')):.2e} "
                          f"kappa={row.get('kappa', float('nan')):.2e} [{row['stopping']}] "
                          f"({time.perf_counter() - t0:.1f}s)", flush=True)
    figures(rows, root / 'figures')
    return csv_path


def _write(path, rows):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _f(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def figures(rows, fig_dir):
    """Per benchmark and P: final ||R_lin||_h and kappa_2 against the actual
    N/P, one curve per distribution (random: median with min-max band)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = Path(fig_dir)
    for benchmark, P in sorted({(r['benchmark'], int(r['P'])) for r in rows}):
        sel = [r for r in rows if r['benchmark'] == benchmark and int(r['P']) == P and not r.get('error')]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        series = []
        for dist, style in (('paper', 'k-o'), ('cgl', 'b-s'), ('random', 'r-^')):
            pts = {}
            for r in sel:
                if r['distribution'] == dist:
                    pts.setdefault(float(r['ratio_nominal']), []).append(r)
            for ax, col in zip(axes, ('final_norm_Rlin_h', 'kappa')):
                xs, med, lo, hi = [], [], [], []
                for ratio in sorted(pts):
                    vals = [_f(r[col]) for r in pts[ratio] if _f(r[col]) is not None]
                    if vals:
                        xs.append(statistics.median(_f(r['ratio_actual']) for r in pts[ratio]))
                        med.append(statistics.median(vals)); lo.append(min(vals)); hi.append(max(vals))
                        series.append(dict(benchmark=benchmark, P=P, distribution=dist, quantity=col,
                                           ratio_actual=xs[-1], median=med[-1], min=lo[-1], max=hi[-1]))
                if xs:
                    ax.plot(xs, med, style, ms=4, label=dist if dist != 'random' else 'random (median, 5 seeds)')
                    if dist == 'random':
                        ax.fill_between(xs, lo, hi, color='r', alpha=0.15)
        for ax, label in zip(axes, (r'final $\|\mathbf{R}_{\mathrm{lin}}\|_h$', r'$\kappa_2(\mathbf{A})$')):
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('N / P (actual)'); ax.set_ylabel(label)
            ax.grid(True, which='both', alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{benchmark}, P = {P}')
        fig.tight_layout()
        fig.savefig(fig_dir / f'{benchmark}_P{P}.pdf')
        plt.close(fig)
        with open(fig_dir / f'{benchmark}_P{P}.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(series[0]) if series else ['benchmark'])
            writer.writeheader()
            writer.writerows(series)


def main():
    ap = argparse.ArgumentParser(description="Component C: oversampling sweep")
    ap.add_argument('--root', default=str(OUTPUT_DIR), help="The package's C_oversampling/ directory.")
    ap.add_argument('--quick', action='store_true', help='Smallest P per benchmark, three ratios, K_max 3.')
    args = ap.parse_args()
    if args.quick:
        path = run_sweep(args.root, benchmarks=(('kovasznay', 10), ('bratu', 10)), ratios=(1, 3, 20),
                         distributions=(('paper', None), ('cgl', None), ('random', 0)), quick=True)
    else:
        path = run_sweep(args.root)
    print(f"Wrote {path}")


if __name__ == '__main__':
    main()
