"""
Buckley-Leverett: sensitivity to the initial guess (task B8)
============================================================

Computational Package 1, Addendum v2.1, task B8. Both cases (viscous,
gravity) x both initial guesses x P in {64, 256, 576, 1024}, for LiL-Q,
LiL-N, and NiL-N / NiL-Q with seeds 0, 1, 2 -- 128 runs:

* ``zero``: zero coefficients (LiL), a network pretrained to zero (NiL);
* ``ic``: the initial saturation profile extended in time, fitted to the
  basis by least squares (LiL) or pretrained into the network (NiL).

Every run uses the paper's budgets and targets (``experiments/run_bl.py``
``paper_setup``). LiL-Q stops only on its target or at K_max, never on a
stall or a divergence, and keeps its full Section 3.1 log (chi_k, stall
flag) in ``lilq_logs/<case>_<guess>_P<P>/``.

Output ``b8_initial_guess.csv``, one row per run: case, guess, P, method,
seed, iterations, evaluations, final loss, target, target reached,
stopping reason, and for LiL-Q the first stall iteration, whether the
stall flag ever fired, and the chi_k history. Resumable: completed rows are
kept and skipped; a run that raises is logged with its traceback. Every
run's trained model is saved in ``models/<case>_<guess>_P<P>_<method>_s<seed>/``
(``lilq.saved_models``) before its row is written.

Untimed (Addendum Section 2): runs on this machine's CPU by default.

Usage::

    python experiments/b8_initial_guess.py
    python experiments/b8_initial_guess.py --cases gravity --guesses ic --P 64
    python experiments/b8_initial_guess.py --quick        # smoke test
"""

import argparse
import csv
import dataclasses
import os
import sys
import time
import traceback
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from experiments.run_bl import DEFAULT_N_VALUES, paper_setup
from lilq.four_method_log import stopping_fields
from lilq.iteration_log import IterationLogger, solve_rows
from lilq.provenance import save_provenance
from lilq.run_metadata import first_stall_iteration
from lilq.saved_models import save_history, save_network, save_solution
from lilq.source_lock import current_commit
from lilq.solvers import line_search_cap
from problems.buckley_leverett import run_lil_n, run_lil_q, run_nil_n, run_nil_q

CASES = ('viscous', 'gravity')
GUESSES = ('zero', 'ic')
METHODS = ('LiL-Q', 'LiL-N', 'NiL-N', 'NiL-Q')
SEEDS = (0, 1, 2)
OUTPUT_DIR = Path(_proj) / 'results' / 'b8_initial_guess'

COLUMNS = ('case', 'guess', 'P', 'method', 'seed', 'iterations', 'evaluations', 'final_loss',
           'target', 'target_reached', 'stopping_reason', 'iterations_cap', 'evaluations_cap',
           'first_stall_iteration', 'stall_flag_ever', 'chi_history',
           'stall_iteration', 'stall_evaluations', 'stall_time_s', 'wall_s', 'device', 'commit', 'error')


def row_key(row):
    return (row['case'], row['guess'], int(row['P']), row['method'], str(row['seed']))


def _quick(opt):
    return dataclasses.replace(opt, max_iterations=15, max_line_searches=200, max_quasi_iters_nn=2,
                               max_inner_iters_nn=10, max_quasi_iters_lil=3, pretrain_epochs=5)


def run_one(case, guess, N, method, seed, device='cpu', quick=False, log_root=None, model_dir=None):
    """One B8 row (raises on failure; the caller logs it). With ``model_dir``,
    the trained model is saved there (``lilq.saved_models``)."""
    config, opt = paper_setup(N, gravity=(case == 'gravity'))
    config = dataclasses.replace(config, initial_guess=guess)
    if seed != '':
        config = dataclasses.replace(config, init_seed=seed)
    if quick:
        opt = _quick(opt)
    row = dict(case=case, guess=guess, P=N * N, method=method, seed=seed, target=opt.R_tol,
               device=device if method != 'LiL-Q' else 'cpu')

    if method == 'LiL-Q':
        logger = IterationLogger()
        log_dir = Path(log_root) / f'{case}_{guess}_P{N * N}' if log_root else None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
        try:
            basis, c, _metrics, summary = run_lil_q(config, opt, verbose=False, iteration_logger=logger,
                                                    run_json_path=(log_dir / 'run.json') if log_dir else None)
        finally:            # a run that diverges or raises keeps its chi_k history (Addendum v2.2 2.8)
            if log_dir:
                logger.to_csv(log_dir / 'iterations.csv')
        if model_dir:
            save_solution(model_dir, {'u': (basis, c)}, config, opt)
        rows = solve_rows(logger.rows)
        chi = [r['chi'] for r in rows]
        row.update(
            iterations=summary['total_iterations'], evaluations=summary['total_iterations'],
            iterations_cap=opt.max_quasi_iters_lil, evaluations_cap=opt.max_quasi_iters_lil,
            stopping_reason='target' if summary['converged'] else 'iteration_cap',
            first_stall_iteration=first_stall_iteration(logger.rows),
            stall_flag_ever=any(bool(r['stall_flag']) for r in rows),
            chi_history=';'.join('' if c is None or np.isnan(c) else f'{c:.3g}' for c in chi),
        )
    else:
        runner = {'LiL-N': run_lil_n, 'NiL-N': run_nil_n, 'NiL-Q': run_nil_q}[method]
        result = runner(config, opt, device=device, verbose=False)
        summary = result[-1]
        if model_dir:
            if method == 'LiL-N':
                save_solution(model_dir, {'u': (result[0], result[1])}, config, opt)
            else:
                save_network(model_dir, result[0], config, opt)
            save_history(model_dir, result[-2])
        stop = stopping_fields(method, summary, opt)
        row.update(stop)
    row.update(final_loss=summary['final_loss'], target_reached=bool(summary['converged']),
               commit=current_commit())
    return row


def planned_runs(cases=CASES, guesses=GUESSES, sizes=DEFAULT_N_VALUES, methods=METHODS, seeds=SEEDS):
    for case in cases:
        for guess in guesses:
            for N in sizes:
                for method in methods:
                    for seed in (seeds if method.startswith('NiL') else ('',)):
                        yield case, guess, N, method, seed


def _write(path, rows):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run_b8(out_dir, device='cpu', quick=False, verbose=True, **selection):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'b8_initial_guess.csv'
    rows = []
    if csv_path.exists():
        with open(csv_path, newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['stopping_reason'] != 'failure']
    done = {row_key(r) for r in rows}
    save_provenance(out_dir)
    for case, guess, N, method, seed in planned_runs(**selection):
        key = (case, guess, N * N, method, str(seed))
        if key in done:
            continue
        t0 = time.perf_counter()
        try:
            row = run_one(case, guess, N, method, seed, device=device, quick=quick,
                          log_root=out_dir / 'lilq_logs',
                          model_dir=out_dir / 'models' / f"{case}_{guess}_P{N * N}_{method}_s{seed if seed != '' else 'na'}")
        except Exception:
            row = dict(case=case, guess=guess, P=N * N, method=method, seed=seed,
                       stopping_reason='failure', device=device, error=traceback.format_exc())
        row['wall_s'] = time.perf_counter() - t0
        rows.append(row)
        _write(csv_path, rows)
        if verbose:
            status = row['stopping_reason'] if row['stopping_reason'] != 'target' else 'target reached'
            print(f"  {case:8s} {guess:4s} P={N * N:5d} {method:6s} seed={seed!s:2s} "
                  f"iters={row.get('iterations', '-')!s:>6s} loss={row.get('final_loss', float('nan')):.3e} "
                  f"[{status}] ({row['wall_s']:.1f}s)", flush=True)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Task B8: Buckley-Leverett initial-guess sensitivity")
    parser.add_argument('--cases', nargs='+', default=list(CASES), choices=CASES)
    parser.add_argument('--guesses', nargs='+', default=list(GUESSES), choices=GUESSES)
    parser.add_argument('--P', type=int, nargs='+', default=None,
                        help='Subset of P (64 256 576 1024); default all.')
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--device', default='cpu', help="Device for LiL-N/NiL (LiL-Q is scipy on the CPU).")
    parser.add_argument('--quick', action='store_true', help='Smallest P, tiny budgets (smoke test).')
    parser.add_argument('--out-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    sizes = DEFAULT_N_VALUES[:1] if args.quick else (
        [int(round(np.sqrt(p))) for p in args.P] if args.P else DEFAULT_N_VALUES)
    path = run_b8(args.out_dir, device=args.device, quick=args.quick, cases=args.cases,
                  guesses=args.guesses, sizes=sizes, methods=args.methods, seeds=args.seeds)
    print(f"Wrote {path}")


if __name__ == '__main__':
    main()
