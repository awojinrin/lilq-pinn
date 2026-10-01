"""
Task B10: Kovasznay on CGL grids with Clenshaw-Curtis weights (Addendum v2.3)
=============================================================================

The advisor's sampling-constant analysis (``S85_Sampling_Constants_Kovasznay``)
found that the paper's equispaced grids, and the equal-weight CGL grids, do not
satisfy the hypothesis of the convergence theorem for the algorithm as run
(Corollary 3), while the same CGL points with Clenshaw-Curtis quadrature
weights do, with a constant of about 1 from N/P = 5-10. B10 runs LiL-Q on such
grids, so that the paper has one experiment the proved theory covers end to
end.

Runs (16, untimed):

* B10-CC: P = 300, 1,200, 1,875 (N = 10, 20, 25) x N/P = 5, 10 x the paper
  and kmax passes, ``weights='clenshaw_curtis'``;
* B10-EQ: P = 1,875 x N/P = 5, 10 x both passes, ``weights='equal'``
  (Component C has the equal-weight CGL runs at P = 300 and 1,200).

Each: Kovasznay, Chebyshev basis, ``sampling='cgl'``, the density chosen by
Component C's rule (``component_c.k_for_ratio``, the row count closest to
N/P x P) with ``collocation_floor = 1``. The paper pass uses the paper's
stopping rule (relative coefficient change 1e-9) with K_max =
``lilq.solvers.LILQ_PAPER_KMAX`` (60; the addendum's text says 20, the
advisor's later follow-up 60 for every LiL-Q paper pass -- a run needing more
than 20 is visible in ``iterations``); the kmax pass runs 60 iterations with
the stopping rule disabled (Addendum v2.2 Section 2.7). Every run keeps the
full Section 3.1 log (``iterations.csv``, ``run.json``), ``collocation.npz``
(with the row weights), the saved model and ``hardware.json``.

Output under ``--out-root`` (the wave's root)::

    B_instrumentation/b10/<run>/...
    B_instrumentation/b10/b10.csv                 one row per run
    B_instrumentation/b10/b10_vs_paper_grid.csv   the paper passes' errors beside
                                                  the paper-grid runs' at the same P
                                                  (with --paper-grid-root)

Usage::

    python experiments/b10_cgl_cc.py --out-root results/wave4 \\
        --paper-grid-root results/wave1/B_instrumentation
"""

import argparse
import csv
import dataclasses
import json
import os
import sys
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
from lilq.solvers import LILQ_PAPER_KMAX
from lilq.source_lock import current_commit

KMAX_PASS_ITERS = 60
RUNS = ([('clenshaw_curtis', N, r) for N in (10, 20, 25) for r in (5, 10)]
        + [('equal', 25, r) for r in (5, 10)])
PASSES = ('paper', 'kmax')
COLUMNS = ('run', 'set', 'P', 'ratio_target', 'ratio_actual', 'N', 'N_distinct', 'k_ratio', 'weights', 'pass',
           'K_max', 'iterations', 'stopping_reason', 'first_stall_iteration', 'final_norm_R_h',
           'final_norm_Rlin_h', 'kappa_raw', 'kappa_retained', 'kappa', 'kappa_method', 'num_rank_qr',
           'num_rank_gelsy', 'num_rank_svd', 'rank_deficient', 'eps_u', 'eps_v', 'eps_p_pin_gauge',
           'eps_p_meanfree', 't_cum_s', 'commit', 'error')
COMPARE_COLUMNS = ('P', 'ratio_target', 'weights', 'b10_eps_u', 'b10_eps_v', 'b10_eps_p_pin_gauge',
                   'b10_eps_p_meanfree', 'b10_iterations', 'paper_grid_run', 'paper_grid_N', 'paper_grid_eps_u',
                   'paper_grid_eps_v', 'paper_grid_eps_p_pin_gauge', 'paper_grid_eps_p_meanfree',
                   'paper_grid_iterations', 'ratio_eps_u_b10_over_paper_grid')


def run_name(weights, N, ratio, pass_):
    return f"kovasznay_P{3 * N * N}_r{ratio}_cgl_{'cc' if weights == 'clenshaw_curtis' else 'eq'}_{pass_}"


def config_for(weights, N, ratio, pass_, smoke=False):
    """The run's configuration and its expected row count."""
    from experiments.component_c import _config, k_for_ratio
    k, rows = k_for_ratio('kovasznay', N, ratio)
    config, _ = _config('kovasznay', N, k, 'cgl', None)
    config = dataclasses.replace(config, weights=weights, max_iter=LILQ_PAPER_KMAX if pass_ == 'paper' else
                                 KMAX_PASS_ITERS, tol=config.tol if pass_ == 'paper' else 0.0)
    if smoke:
        config = dataclasses.replace(config, max_iter=min(config.max_iter, 3))
    return config, k, rows


def run_one(weights, N, ratio, pass_, root, smoke=False):
    from problems.kovasznay import solve_kovasznay
    name = run_name(weights, N, ratio, pass_)
    run_dir = Path(root) / name
    config, k, rows = config_for(weights, N, ratio, pass_, smoke)
    P = 3 * N * N
    row = dict(run=name, set='B10-CC' if weights == 'clenshaw_curtis' else 'B10-EQ', P=P, ratio_target=ratio,
               ratio_actual=rows / P, N=rows, k_ratio=k, weights=weights, **{'pass': pass_}, K_max=config.max_iter,
               commit=current_commit())
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger = IterationLogger()
        r = solve_kovasznay(config, verbose=False, iteration_logger=logger, run_json_path=run_dir / 'run.json',
                            collocation_path=run_dir / 'collocation.npz')
        logger.to_csv(run_dir / 'iterations.csv')
        save_solution(run_dir, {f: (r[f'basis_{f}'], r[f'theta_{f}']) for f in 'uvp'}, config)
        save_provenance(run_dir)
        solves, last = solve_rows(logger.rows), logger.rows[-1]
        final = solves[-1]
        with np.load(run_dir / 'collocation.npz') as z:
            saved_rows, n_distinct = len(z['x']), int(z['n_distinct'])
        if saved_rows != rows:                 # the row-count rule and the solver must agree
            raise RuntimeError(f"N {rows} != {saved_rows} collocation rows saved by the solver")
        rank = final.get('num_rank_qr') or final.get('num_rank_gelsy') or final.get('num_rank_svd')
        reason = 'target' if final['rel_dbeta'] < config.tol else 'iteration_cap'
        row.update(iterations=len(solves), stopping_reason=reason, N_distinct=n_distinct,
                   first_stall_iteration=first_stall_iteration(logger.rows), final_norm_R_h=last['norm_R_h'],
                   final_norm_Rlin_h=final['norm_Rlin_h'], kappa_raw=final.get('kappa_raw'),
                   kappa_retained=final.get('kappa_retained'), kappa=final['kappa'],
                   kappa_method=final['kappa_method'], num_rank_qr=final.get('num_rank_qr'),
                   num_rank_gelsy=final.get('num_rank_gelsy'), num_rank_svd=final.get('num_rank_svd'),
                   rank_deficient=(int(rank) < P) if rank not in (None, '') else '',
                   eps_u=last.get('eps_u'), eps_v=last.get('eps_v'), eps_p_pin_gauge=last.get('eps_p'),
                   eps_p_meanfree=last.get('eps_p_meanfree'), t_cum_s=last['t_cum_s'])
    except Exception:
        (run_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        row.update(stopping_reason='failure', error=traceback.format_exc()[-2000:])
    return row


def compare_with_paper_grid(rows, paper_root):
    """Each B10 paper pass beside the paper-grid (equispaced, k = 4) paper
    pass at the same P (Addendum v2.3 Section 3)."""
    out = []
    for r in rows:
        if r['pass'] != 'paper' or r.get('stopping_reason') == 'failure':
            continue
        path = Path(paper_root) / f"kovasznay_P{r['P']}_cpu_paper"
        s = json.loads((path / 'summary.json').read_text()) if (path / 'summary.json').exists() else {}
        g = lambda k: s.get(k)  # noqa: E731
        eu = g('test_eps_u')
        out.append({'P': r['P'], 'ratio_target': r['ratio_target'], 'weights': r['weights'],
                    'b10_eps_u': r['eps_u'], 'b10_eps_v': r['eps_v'], 'b10_eps_p_pin_gauge': r['eps_p_pin_gauge'],
                    'b10_eps_p_meanfree': r['eps_p_meanfree'], 'b10_iterations': r['iterations'],
                    'paper_grid_run': str(path) if s else '', 'paper_grid_N': '',
                    'paper_grid_eps_u': eu, 'paper_grid_eps_v': g('test_eps_v'),
                    'paper_grid_eps_p_pin_gauge': g('test_eps_p'),
                    'paper_grid_eps_p_meanfree': g('test_eps_p_meanfree'),
                    'paper_grid_iterations': g('iterations'),
                    'ratio_eps_u_b10_over_paper_grid': (float(r['eps_u']) / float(eu))
                    if eu not in (None, '', 0) and r['eps_u'] not in (None, '') else ''})
        if s and (path / 'run.json').exists():
            out[-1]['paper_grid_N'] = json.loads((path / 'run.json').read_text()).get('N_total', '')
    return out


def _write(path, columns, rows):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows([{c: r.get(c) for c in columns} for r in rows])
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Task B10: Kovasznay on CGL grids with Clenshaw-Curtis weights.')
    ap.add_argument('--out-root', required=True, help="The wave's root (results/wave4).")
    ap.add_argument('--paper-grid-root', default=None,
                    help="B_instrumentation folder with the paper-grid Kovasznay runs (wave 1's).")
    ap.add_argument('--smoke', action='store_true', help='P = 300 only, 3 iterations (tests).')
    args = ap.parse_args(argv)
    root = Path(args.out_root) / 'B_instrumentation' / 'b10'
    root.mkdir(parents=True, exist_ok=True)
    save_provenance(root)
    csv_path = root / 'b10.csv'
    rows = [r for r in csv.DictReader(open(csv_path))] if csv_path.exists() else []
    done = {r['run'] for r in rows if r.get('stopping_reason') != 'failure'}
    rows = [r for r in rows if r['run'] in done]
    plan = [(w, N, ratio) for w, N, ratio in RUNS if not args.smoke or N == 10]
    for weights, N, ratio in plan:
        for pass_ in PASSES:
            name = run_name(weights, N, ratio, pass_)
            if name in done:
                print(f"  {name}: done, skipping")
                continue
            row = run_one(weights, N, ratio, pass_, root, args.smoke)
            rows.append(row)
            _write(csv_path, COLUMNS, rows)
            flags = []
            if row.get('stopping_reason') == 'failure':
                flags.append('FAILED (error.txt)')
            elif pass_ == 'paper' and row['stopping_reason'] == 'iteration_cap':
                flags.append('paper pass hit K_max')
            if row.get('rank_deficient') is True:
                flags.append('RANK DEFICIENT')
            print(f"  {name}: {row.get('iterations')} iterations, eps_u {row.get('eps_u')}"
                  + (f"  <- {', '.join(flags)}" if flags else ''), flush=True)
    if args.paper_grid_root:
        _write(root / 'b10_vs_paper_grid.csv', COMPARE_COLUMNS, compare_with_paper_grid(rows, args.paper_grid_root))
    print(f"Wrote {csv_path} ({len(rows)} runs)")


if __name__ == '__main__':
    main()
