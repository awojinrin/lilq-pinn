"""
Clean timing of every quoted LiL-Q time (the advisor's reply to wave 2, Section 3)
==================================================================================

One rule for every LiL-Q time the paper quotes: it comes from a run with the
per-iteration diagnostics off, after a warm-up. For each paper-pass
configuration -- Bratu, Burgers, viscous and gravity BL, elasticity,
Kovasznay (CPU and GPU), Beltrami, the pinned Beltrami run of Section 3.7
(the paper's Beltrami run; the advisor's follow-up of 1 October 2026) and
Darcy -- this runs, on one device:

1. one complete untimed warm-up run of the same configuration;
2. the timed run, with ``diagnostics=False`` (no logger, no conditioning,
   none of the passive work otherwise on the clock; the solver path and the
   coefficients are those of the logged run, ``tests/test_clean_timing.py``);
3. if the timed run is shorter than 1 s, five more, reported as their median
   (Addendum v2.2 Section 2.10), all five recorded.

Every paper pass uses K_max = ``lilq.solvers.LILQ_PAPER_KMAX`` (60). The
logged runs (``component_b.py``, wave 2; the pinned run, wave 1) remain the
source of every monitor value; ``--logged-roots`` puts each logged time
beside the clean one. Output: ``<out-root>/B_instrumentation/clean_timing/
clean_timing.csv``, one row per run and quantity, rewritten after every run
(a resubmitted job skips the runs it has).

Quantities, the same as the logged runs report:
``training_time`` (scalar benchmarks); ``solve_time_total`` (Kovasznay,
Beltrami, pinned Beltrami); ``time_lil_s`` and ``solve_time_qr``
(elasticity); ``total_time`` (Darcy -- with diagnostics off it ends with the
LiL solve; the logged run's also counted the FVM reference and the error
metrics, 0.2-0.3 s).

Usage::

    python experiments/clean_timing.py --devices cpu --out-root results/wave4 \\
        --logged-roots results/wave2/B_instrumentation results/wave1/B_instrumentation
    python experiments/clean_timing.py --devices cuda --benchmarks kovasznay --out-root ...
"""

import argparse
import csv
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from lilq.provenance import save_provenance
from lilq.solvers import LILQ_PAPER_KMAX
from lilq.source_lock import current_commit

BENCHMARKS = ('bratu', 'burgers', 'bl', 'bl_gravity', 'elasticity', 'kovasznay', 'beltrami', 'beltrami_pinned',
              'darcy')
SHORT_RUN_S = 1.0
REPEATS = 5
COLUMNS = ('run', 'benchmark', 'config', 'device', 'quantity', 'clean_time_s', 'clean_time_single_run_s',
           'timing_repeats_s', 'warmup_run_time_s', 'iterations', 'K_max', 'logged_time_s', 'logged_source',
           'commit')


@dataclass
class TimedRun:
    name: str                 # the logged run's folder name (component_b), or 'beltrami_pinned'
    benchmark: str
    config_label: str
    device: str
    k_max: object
    run_once: Callable[[], Dict[str, float]]    # quantities (first: the one the 1 s rule tests) + 'iterations'


def _scalar(benchmark, smoke):
    import experiments.residual_band_figures as rbf
    from problems.bratu import run_lil_q as bratu_q
    from problems.burgers import run_lil_q as burgers_q
    from problems.buckley_leverett import run_lil_q as bl_q
    build, run_lil_q = {'bratu': (lambda: rbf._bratu_runs(quick=False), bratu_q),
                        'burgers': (lambda: rbf._burgers_runs(quick=False), burgers_q),
                        'bl': (lambda: rbf._bl_runs(False, quick=False), bl_q),
                        'bl_gravity': (lambda: rbf._bl_runs(True, quick=False), bl_q)}[benchmark]
    runs = []
    for P, config, opt in (build()[:1] if smoke else build()):
        if smoke:
            opt = dataclasses.replace(opt, max_quasi_iters_lil=min(opt.max_quasi_iters_lil, 5))

        def once(config=config, opt=opt, run_lil_q=run_lil_q):
            s = run_lil_q(config, opt, verbose=False, diagnostics=False)[-1]
            return {'training_time': s['training_time'], 'iterations': s['total_iterations']}
        runs.append(TimedRun(f'{benchmark}_P{P}_cpu_paper', benchmark, f'P{P}', 'cpu', opt.max_quasi_iters_lil, once))
    return runs


def _elasticity(smoke):
    from experiments.run_elasticity import DEFAULT_N_VALUES, K_RATIO
    from problems.elasticity import ElasticityConfig, solve_elasticity
    runs = []
    for N in (DEFAULT_N_VALUES[:1] if smoke else DEFAULT_N_VALUES):
        config = ElasticityConfig(N_x=N, N_y=N, k_ratio=K_RATIO)

        def once(config=config):
            r = solve_elasticity(config, verbose=False, diagnostics=False)
            return {'time_lil_s': r['time_lil_s'], 'solve_time_qr': r['solve_time_qr'], 'iterations': 1}
        runs.append(TimedRun(f'elasticity_P{2 * N * N}_cpu_paper', 'elasticity', f'P{2 * N * N}', 'cpu', '', once))
    return runs


def _kovasznay(smoke, device):
    from experiments.run_kovasznay import DEFAULT_N_VALUES, K_RATIO, MAX_ITER, TOL
    from problems.kovasznay import KovasznayConfig, solve_kovasznay
    runs = []
    for N in (DEFAULT_N_VALUES[:1] if smoke else DEFAULT_N_VALUES):
        config = KovasznayConfig(N_x=N, N_y=N, k_ratio=K_RATIO, max_iter=min(MAX_ITER, 5) if smoke else MAX_ITER,
                                 tol=TOL, use_gpu=(device == 'cuda'))

        def once(config=config):
            r = solve_kovasznay(config, verbose=False, diagnostics=False)
            return {'solve_time_total': r['solve_time_total'], 'iterations': r['n_outer_iters']}
        runs.append(TimedRun(f'kovasznay_P{3 * N * N}_{device}_paper', 'kovasznay', f'P{3 * N * N}', device,
                             config.max_iter, once))
    return runs


def _beltrami(smoke, pinned):
    from experiments.run_beltrami import COLLOC
    from experiments.run_beltrami_pinned import pinned_config
    from problems.beltrami import BeltramiConfig, solve_beltrami
    if smoke:
        config = BeltramiConfig(N_vel=3, N_p=3, N_x=4, N_y=4, N_z=4, N_t=4, N_bc=3, N_t_bc=3, N_ic=3, max_iter=3,
                                n_pressure_pin_levels=3 if pinned else 1)
    else:
        config = pinned_config() if pinned else BeltramiConfig(N_vel=6, N_p=8, max_iter=LILQ_PAPER_KMAX, **COLLOC[6])
    P = 3 * config.N_vel ** 4 + config.N_p ** 4

    def once():
        np.random.seed(42)        # as run_beltrami_pinned; the solve itself is deterministic
        r = solve_beltrami(config, verbose=False, diagnostics=False)
        return {'solve_time_total': r['solve_time_total'], 'iterations': r['n_outer_iters']}
    name = 'beltrami_pinned' if pinned else f'beltrami_P{P}_cpu_paper'
    return [TimedRun(name, 'beltrami_pinned' if pinned else 'beltrami', f'P{P}', 'cpu', config.max_iter, once)]


def _darcy(smoke):
    from experiments.component_b import DARCY_FIELDS
    from experiments.run_darcy import DEFAULT_ORDER
    from problems.darcy import DarcyConfig, DarcyPhysics, solve_lilq_darcy
    order = 6 if smoke else DEFAULT_ORDER
    runs = []
    for field in (DARCY_FIELDS[:1] if smoke else DARCY_FIELDS):
        config = DarcyConfig(ORDER_H=order, ORDER_U=order, ORDER_V=order, perm_file=f'perm_field_{field}.txt')

        def once(config=config):
            r = solve_lilq_darcy(config, DarcyPhysics(config, verbose=False), verbose=False, diagnostics=False)
            return {'total_time': r['metrics']['total_time'], 'iterations': 1}
        runs.append(TimedRun(f'darcy_{field}_cpu_paper', 'darcy', field, 'cpu', '', once))
    return runs


def build_runs(benchmarks=BENCHMARKS, devices=('cpu',), smoke=False) -> List[TimedRun]:
    runs = []
    for b in benchmarks:
        if b in ('bratu', 'burgers', 'bl', 'bl_gravity') and 'cpu' in devices:
            runs += _scalar(b, smoke)
        elif b == 'elasticity' and 'cpu' in devices:
            runs += _elasticity(smoke)
        elif b == 'kovasznay':
            for device in devices:
                runs += _kovasznay(smoke, device)
        elif b in ('beltrami', 'beltrami_pinned') and 'cpu' in devices:
            runs += _beltrami(smoke, b == 'beltrami_pinned')
        elif b == 'darcy' and 'cpu' in devices:
            runs += _darcy(smoke)
    return runs


def logged_time(run: TimedRun, quantity: str, roots):
    """The logged run's reported time for ``quantity`` and where it came
    from: the first of ``roots`` that has it."""
    for root in roots or ():
        root = Path(root)
        if run.benchmark == 'beltrami_pinned':
            path, key = root / 'beltrami_pinned' / 'report.json', 'solver_time_s'
        else:
            path, key = root / run.name / 'summary.json', quantity
        if path.exists():
            value = json.loads(path.read_text()).get(key)
            if isinstance(value, (int, float)):
                return float(value), str(path)
    return None, None


def execute(run: TimedRun, smoke=False, logged_roots=None) -> List[dict]:
    """Warm-up, the timed run, and the repeats of a run under 1 s."""
    import torch
    sync = (lambda: torch.cuda.synchronize()) if run.device == 'cuda' else (lambda: None)  # noqa: E731
    warm = run.run_once()
    sync()
    first = run.run_once()
    sync()
    quantities = [k for k in first if k != 'iterations']
    main = quantities[0]
    repeats = None
    if not smoke and first[main] < SHORT_RUN_S:
        repeats = [run.run_once() for _ in range(REPEATS)]
    rows = []
    for q in quantities:
        times = [r[q] for r in repeats] if repeats else None
        logged, source = logged_time(run, q, logged_roots)
        rows.append({'run': run.name, 'benchmark': run.benchmark, 'config': run.config_label, 'device': run.device,
                     'quantity': q, 'clean_time_s': float(np.median(times)) if times else first[q],
                     'clean_time_single_run_s': first[q], 'timing_repeats_s': json.dumps(times) if times else '',
                     'warmup_run_time_s': warm[q], 'iterations': first['iterations'], 'K_max': run.k_max,
                     'logged_time_s': logged, 'logged_source': source, 'commit': current_commit()})
    return rows


def write(rows, path):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Clean timing of the quoted LiL-Q times.")
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--benchmarks', nargs='+', choices=BENCHMARKS, default=list(BENCHMARKS))
    ap.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=['cpu'])
    ap.add_argument('--logged-roots', nargs='*', default=[],
                    help='B_instrumentation folders of the logged runs, searched in order.')
    ap.add_argument('--smoke', action='store_true', help='Smallest size of each, few iterations (tests).')
    args = ap.parse_args(argv)
    out = Path(args.out_root) / 'B_instrumentation' / 'clean_timing'
    out.mkdir(parents=True, exist_ok=True)
    save_provenance(out)
    path = out / 'clean_timing.csv'
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    done = {r['run'] for r in rows}
    for run in build_runs(args.benchmarks, args.devices, args.smoke):
        if run.name in done:
            print(f"  {run.name}: done, skipping")
            continue
        new = execute(run, args.smoke, args.logged_roots)
        rows += new
        write(rows, path)
        for r in new:
            print(f"  {r['run']} {r['quantity']}: clean {r['clean_time_s']:.4g} s "
                  f"(warm-up {r['warmup_run_time_s']:.4g} s, logged {r['logged_time_s']})", flush=True)
    print(f"Wrote {path} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
