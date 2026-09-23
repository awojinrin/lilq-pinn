"""
Component B driver: the logged LiL-Q reruns (Computational_Package_1_v2.md Section 3.3)
=======================================================================================

Runs every LiL-Q configuration of the paper with the full Section 3.1
instrumentation and writes the spec's layout (Section 6)::

    <out-root>/B_instrumentation/<benchmark>_<config>_<device>_<pass>/
        run.json, iterations.csv, summary.json, hardware.json, environment.txt
    <out-root>/B_instrumentation/runs_index.csv     (one row per run folder)

Passes (Section 3.3): ``paper`` uses the paper's stopping rule; ``kmax``
disables it and runs to K_max -- only for the scalar benchmarks and
Kovasznay. Kovasznay runs on CPU and, where a CUDA device exists, GPU.

Every setting comes from the problem's own ``experiments/run_*.py``
constants (the same ones the rest of this codebase uses). A run folder
whose ``summary.json`` exists is complete and is skipped on rerun, so an
interrupted job resumes where it stopped; a run that raises writes
``error.txt`` (with the traceback) and the driver moves on. Runs execute
one at a time, after one untimed warm-up per device.

Usage::

    python experiments/component_b.py --list                 # show the run plan
    python experiments/component_b.py --smoke                # tiny end-to-end check
    python experiments/component_b.py                        # everything
    python experiments/component_b.py --benchmarks beltrami  # a subset (e.g. one cluster job)
"""

import argparse
import csv
import dataclasses
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from lilq.iteration_log import IterationLogger, solve_rows
from lilq.provenance import save_provenance
from lilq.run_metadata import first_stall_iteration

DEFAULT_OUT_ROOT = Path(__file__).resolve().parent.parent / 'results' / 'package1'
BENCHMARKS = ('bratu', 'burgers', 'bl', 'bl_gravity', 'elasticity', 'kovasznay', 'beltrami', 'darcy')
KMAX_BENCHMARKS = ('bratu', 'burgers', 'bl', 'bl_gravity', 'kovasznay')
DARCY_FIELDS = ('S1', 'S2', 'S3', 'SPE10')


@dataclass
class Run:
    benchmark: str
    config_label: str
    device: str
    pass_: str
    execute: Callable[[Path], Dict]  # run_dir -> summary; writes run.json/iterations.csv

    @property
    def name(self) -> str:
        return f"{self.benchmark}_{self.config_label}_{self.device}_{self.pass_}"


def _log_summary(logger: IterationLogger) -> Dict:
    rows = logger.rows
    solves = solve_rows(rows)
    last = rows[-1]
    return {
        'iterations': len(solves),
        'final_norm_R_h': last['norm_R_h'],
        't_cum_s': last['t_cum_s'],
        'first_stall_iteration': first_stall_iteration(rows),
        'kappa_final': solves[-1]['kappa'],
        'kappa_method': solves[-1]['kappa_method'],
        'num_rank_svd_final': solves[-1]['num_rank_svd'],
        'num_rank_gelsy_final': solves[-1]['num_rank_gelsy'],
        **{f'test_{col}': last[col] for col in
           ('eps_u', 'eps_v', 'eps_p', 'eps_p_meanfree', 'maxerr_u', 'maxerr_v', 'maxerr_p')},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run registry
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_runs(benchmark, smoke, passes):
    """Bratu, Burgers, viscous and gravity BL: the same (P, config, opt)
    triples the residual-band figures use (each problem's paper sizes,
    targets and K_max from its experiments/run_*.py)."""
    import experiments.residual_band_figures as rbf
    from problems.bratu import run_lil_q as bratu_q
    from problems.burgers import run_lil_q as burgers_q
    from problems.buckley_leverett import run_lil_q as bl_q
    builders = {
        'bratu': (lambda: rbf._bratu_runs(quick=False), bratu_q),
        'burgers': (lambda: rbf._burgers_runs(quick=False), burgers_q),
        'bl': (lambda: rbf._bl_runs(False, quick=False), bl_q),
        'bl_gravity': (lambda: rbf._bl_runs(True, quick=False), bl_q),
    }
    build, run_lil_q = builders[benchmark]
    triples = build()[:1] if smoke else build()
    runs = []
    for P, config, opt in triples:
        for pass_ in passes:
            # K_max pass: a zero loss target can never be met, so the loop
            # always runs max_quasi_iters_lil iterations.
            run_opt = opt if pass_ == 'paper' else dataclasses.replace(opt, R_tol=0.0)
            if smoke:
                run_opt = dataclasses.replace(run_opt, max_quasi_iters_lil=min(run_opt.max_quasi_iters_lil, 5))

            def execute(run_dir, config=config, run_opt=run_opt, run_lil_q=run_lil_q):
                logger = IterationLogger()
                _basis, _c, _metrics, summary = run_lil_q(
                    config, run_opt, verbose=False, iteration_logger=logger,
                    run_json_path=run_dir / 'run.json')
                logger.to_csv(run_dir / 'iterations.csv')
                return {'final_loss': summary['final_loss'], 'converged': summary['converged'],
                        'R_tol': run_opt.R_tol, 'K_max': run_opt.max_quasi_iters_lil,
                        **_log_summary(logger)}
            runs.append(Run(benchmark, f'P{P}', 'cpu', pass_, execute))
    return runs


def _kovasznay_runs(smoke, passes, devices):
    from experiments.run_kovasznay import DEFAULT_N_VALUES, K_RATIO, MAX_ITER, TOL
    from problems.kovasznay import KovasznayConfig, solve_kovasznay
    runs = []
    for N in (DEFAULT_N_VALUES[:1] if smoke else DEFAULT_N_VALUES):
        for device in devices:
            for pass_ in passes:
                # K_max pass: a zero coefficient-change tolerance is never met.
                config = KovasznayConfig(N_x=N, N_y=N, k_ratio=K_RATIO, max_iter=MAX_ITER,
                                         tol=TOL if pass_ == 'paper' else 0.0,
                                         use_gpu=(device == 'cuda'))

                def execute(run_dir, config=config):
                    logger = IterationLogger()
                    r = solve_kovasznay(config, verbose=False, iteration_logger=logger,
                                        run_json_path=run_dir / 'run.json')
                    logger.to_csv(run_dir / 'iterations.csv')
                    return {'n_outer_iters': r['n_outer_iters'], 'solve_time_total': r['solve_time_total'],
                            'rel_l2_u': r['rel_l2_u'], 'rel_l2_v': r['rel_l2_v'], 'rel_l2_p': r['rel_l2_p'],
                            'tol': config.tol, 'K_max': config.max_iter, **_log_summary(logger)}
                runs.append(Run('kovasznay', f'P{3 * N * N}', device, pass_, execute))
    return runs


def _elasticity_runs(smoke):
    from experiments.run_elasticity import DEFAULT_N_VALUES, K_RATIO
    from problems.elasticity import ElasticityConfig, solve_elasticity
    runs = []
    for N in (DEFAULT_N_VALUES[:1] if smoke else DEFAULT_N_VALUES):
        config = ElasticityConfig(N_x=N, N_y=N, k_ratio=K_RATIO)

        def execute(run_dir, config=config):
            logger = IterationLogger()
            r = solve_elasticity(config, verbose=False, iteration_logger=logger,
                                 run_json_path=run_dir / 'run.json')
            logger.to_csv(run_dir / 'iterations.csv')
            return {k: r[k] for k in ('solve_time_qr', 'solve_time_total', 'pde_mse', 'rel_l2_ux',
                                      'rel_l2_uy', 'rel_l2_sxx', 'rel_l2_syy', 'rel_l2_sxy')} | _log_summary(logger)
        runs.append(Run('elasticity', f'P{2 * N * N}', 'cpu', 'paper', execute))
    return runs


def _beltrami_runs(smoke):
    from experiments.run_beltrami import COLLOC
    from problems.beltrami import BeltramiConfig, solve_beltrami
    if smoke:
        config = BeltramiConfig(N_vel=3, N_p=3, N_x=4, N_y=4, N_z=4, N_t=4, N_bc=3, N_t_bc=3, N_ic=3)
    else:
        config = BeltramiConfig(N_vel=6, N_p=8, **COLLOC[6])
    P = 3 * config.N_vel ** 4 + config.N_p ** 4

    def execute(run_dir, config=config):
        logger = IterationLogger()
        r = solve_beltrami(config, verbose=False, iteration_logger=logger,
                           run_json_path=run_dir / 'run.json')
        logger.to_csv(run_dir / 'iterations.csv')
        return {'n_outer_iters': r['n_outer_iters'], 'solve_time_total': r['solve_time_total'],
                'rel_l2_u': r['rel_l2_u'], 'rel_l2_v': r['rel_l2_v'], 'rel_l2_w': r['rel_l2_w'],
                'rel_l2_p': r['rel_l2_p'], 'snapshots': r['snapshots'], **_log_summary(logger)}
    return [Run('beltrami', f'P{P}', 'cpu', 'paper', execute)]


def _darcy_runs(smoke):
    from experiments.run_darcy import DEFAULT_ORDER
    from problems.darcy import DarcyConfig, DarcyPhysics, solve_lilq_darcy
    order = 6 if smoke else DEFAULT_ORDER
    runs = []
    for field in (DARCY_FIELDS[:1] if smoke else DARCY_FIELDS):
        config = DarcyConfig(ORDER_H=order, ORDER_U=order, ORDER_V=order,
                             perm_file=f'perm_field_{field}.txt')

        def execute(run_dir, config=config):
            logger = IterationLogger()
            r = solve_lilq_darcy(config, DarcyPhysics(config, verbose=False), verbose=False,
                                 iteration_logger=logger, run_json_path=run_dir / 'run.json')
            logger.to_csv(run_dir / 'iterations.csv')
            return {'order': config.ORDER_H, **r['metrics'], **_log_summary(logger)}
        runs.append(Run('darcy', field, 'cpu', 'paper', execute))
    return runs


def build_runs(benchmarks=BENCHMARKS, passes=('paper', 'kmax'), devices=('cpu', 'cuda'),
               smoke=False) -> List[Run]:
    runs = []
    for b in benchmarks:
        b_passes = [p for p in passes if p == 'paper' or b in KMAX_BENCHMARKS]
        if not b_passes:
            continue
        if b in ('bratu', 'burgers', 'bl', 'bl_gravity'):
            runs += _scalar_runs(b, smoke, b_passes)
        elif b == 'kovasznay':
            runs += _kovasznay_runs(smoke, b_passes, devices)
        elif 'paper' in b_passes:
            runs += {'elasticity': _elasticity_runs, 'beltrami': _beltrami_runs,
                     'darcy': _darcy_runs}[b](smoke)
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

_WARMED_UP = set()


def warm_up(device: str) -> None:
    """Section 2: one untimed warm-up before the timed phase, per device."""
    if device in _WARMED_UP:
        return
    from problems.kovasznay import KovasznayConfig, solve_kovasznay
    solve_kovasznay(KovasznayConfig(N_x=4, N_y=4, max_iter=2, use_gpu=(device == 'cuda')), verbose=False)
    _WARMED_UP.add(device)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def execute_run(run: Run, root: Path, fresh=False, verbose=True) -> str:
    """Returns 'done' (already complete), 'ok', or 'failed'."""
    run_dir = root / run.name
    if (run_dir / 'summary.json').exists() and not fresh:
        if verbose:
            print(f"  {run.name}: done, skipping")
        return 'done'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'error.txt').unlink(missing_ok=True)
    warm_up(run.device)
    t0 = time.perf_counter()
    try:
        summary = run.execute(run_dir)
    except Exception:
        (run_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if verbose:
            print(f"  {run.name}: FAILED ({time.perf_counter() - t0:.1f}s) -- see error.txt", flush=True)
        return 'failed'
    wall = time.perf_counter() - t0
    save_provenance(run_dir)
    summary = {'run': run.name, 'benchmark': run.benchmark, 'config': run.config_label,
               'device': run.device, 'pass': run.pass_, 'wall_total_s': wall, **summary}
    # Written last: its presence marks the run complete.
    (run_dir / 'summary.json').write_text(json.dumps(_jsonable(summary), indent=2), encoding='utf-8')
    if verbose:
        print(f"  {run.name}: {summary['iterations']} iterations, t_cum={summary['t_cum_s']:.3f}s, "
              f"wall={wall:.1f}s", flush=True)
    return 'ok'


INDEX_COLUMNS = ('run', 'benchmark', 'config', 'device', 'pass', 'status', 'iterations',
                 'final_norm_R_h', 't_cum_s', 'wall_total_s', 'first_stall_iteration',
                 'kappa_final', 'kappa_method', 'num_rank_gelsy_final',
                 'test_eps_u', 'test_eps_v', 'test_eps_p', 'test_eps_p_meanfree')


def write_index(root: Path) -> Path:
    """One row per run folder under ``root``, complete or failed."""
    rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary_path = run_dir / 'summary.json'
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding='utf-8'))
            rows.append({'status': 'ok', **{c: s.get(c) for c in INDEX_COLUMNS if c != 'status'}})
        elif (run_dir / 'error.txt').exists():
            rows.append({'run': run_dir.name, 'status': 'failed'})
    path = root / 'runs_index.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else v) for k, v in row.items()})
    return path


def main():
    parser = argparse.ArgumentParser(description="Component B (Section 3.3) logged LiL-Q reruns")
    parser.add_argument('--out-root', type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument('--passes', nargs='+', choices=('paper', 'kmax'), default=['paper', 'kmax'])
    parser.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=None,
                        help="Kovasznay devices (default: cpu, plus cuda when available). "
                             "Every other benchmark is CPU-only.")
    parser.add_argument('--configs', nargs='+', default=None,
                        help="Subset of config labels, e.g. P675 P1875 S1.")
    parser.add_argument('--smoke', action='store_true',
                        help='Smallest size per benchmark, capped iterations -- a fast end-to-end check.')
    parser.add_argument('--list', action='store_true', help='Print the run plan and exit.')
    parser.add_argument('--fresh', action='store_true', help='Rerun completed runs too.')
    args = parser.parse_args()

    import torch
    devices = args.devices or (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
    if 'cuda' in devices and not torch.cuda.is_available():
        print("No CUDA device: dropping the Kovasznay GPU runs.")
        devices = [d for d in devices if d != 'cuda']

    runs = build_runs(args.benchmarks, args.passes, devices, smoke=args.smoke)
    if args.configs:
        runs = [r for r in runs if r.config_label in args.configs]

    if args.list:
        for r in runs:
            print(r.name)
        print(f"{len(runs)} runs")
        return

    root = Path(args.out_root) / 'B_instrumentation'
    root.mkdir(parents=True, exist_ok=True)
    save_provenance(Path(args.out_root))
    print(f"{len(runs)} runs -> {root}")

    t0 = time.time()
    status = [execute_run(r, root, fresh=args.fresh) for r in runs]
    index = write_index(root)
    print(f"\nok={status.count('ok')} skipped={status.count('done')} failed={status.count('failed')} "
          f"in {time.time() - t0:.1f}s; index: {index}")


if __name__ == '__main__':
    main()
