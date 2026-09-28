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


# ─────────────────────────────────────────────────────────────────────────────
# Check B3: GPU-CPU equivalence and same-system solver timing (Section 3.2)
# ─────────────────────────────────────────────────────────────────────────────

EQUIVALENCE_COLUMNS = (
    'P', 'N_rows', 'iterations_cpu', 'iterations_gpu',
    'beta_rel_diff', 'beta_ok', 'rlin_cpu', 'rlin_gpu', 'rlin_rel_diff', 'rlin_ok', 'equivalent',
    'rlin_floor_cpu', 'rlin_floor_gpu', 'rlin_at_floor', 'rlin_ok_amended', 'equivalent_amended',
    't_gelsy_cpu_s', 't_gels_cpu_s', 't_qr_gpu_s', 'timing_repeats',
    'gpu_mem_estimate_bytes', 'gpu_mem_peak_bytes', 'gpu_min_diag_ratio', 'gpu_flagged_iterations',
)


def _median_time(fn, repeats, sync=False):
    import torch
    fn()  # untimed warm-up
    times = []
    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_gpu_equivalence(root: Path, smoke=False, repeats=3, verbose=True) -> Path:
    """Section 3.2 / check B3, every Kovasznay size: the GPU run against
    the CPU ``gelsy`` run (``||beta_GPU - beta_CPU|| / ||beta_CPU|| <= 1e-8``,
    ``||R_lin||_h`` equal to six significant figures -- ``equivalent``, the
    spec's rule; ``equivalent_amended`` waives the six figures when both
    residuals are at Algorithm 1's round-off floor), plus the solve time
    of CPU ``gelsy`` (the paper's), CPU ``gels`` and the GPU QR on the same
    final-iterate system (median of ``repeats`` after a warm-up; the GPU
    time includes the host-to-device copy of A)."""
    import scipy.linalg
    from experiments.run_kovasznay import DEFAULT_N_VALUES, K_RATIO, MAX_ITER, TOL
    from lilq.instrumentation import EPS_MACH
    from problems.kovasznay import (
        KovasznayConfig, _lstsq_cpu_gels, _lstsq_gpu_qr, solve_kovasznay, verify_gpu_cpu_equivalence,
    )
    rows = []
    for N in (DEFAULT_N_VALUES[:1] if smoke else DEFAULT_N_VALUES):
        config = KovasznayConfig(N_x=N, N_y=N, k_ratio=K_RATIO, max_iter=MAX_ITER, tol=TOL)
        eq = verify_gpu_cpu_equivalence(config)
        system = solve_kovasznay(config, verbose=False, return_final_system=True)
        A, b = system['A_final'], system['b_final']
        gpu = eq['gpu_result']['gpu_qr']
        peak = max(r['gpu_mem_peak_bytes'] or 0 for r in solve_rows(eq['gpu_logger'].rows))
        rows.append({
            'P': 3 * N * N, 'N_rows': A.shape[0],
            'iterations_cpu': eq['cpu_result']['n_outer_iters'],
            'iterations_gpu': eq['gpu_result']['n_outer_iters'],
            **{k: eq[k] for k in ('beta_rel_diff', 'beta_ok', 'rlin_cpu', 'rlin_gpu',
                                  'rlin_rel_diff', 'rlin_ok', 'equivalent', 'rlin_floor_cpu',
                                  'rlin_floor_gpu', 'rlin_at_floor', 'rlin_ok_amended',
                                  'equivalent_amended')},
            't_gelsy_cpu_s': _median_time(
                lambda: scipy.linalg.lstsq(A, b, cond=EPS_MACH, lapack_driver='gelsy'), repeats),
            't_gels_cpu_s': _median_time(lambda: _lstsq_cpu_gels(A, b), repeats),
            't_qr_gpu_s': _median_time(lambda: _lstsq_gpu_qr(A, b), repeats, sync=True),
            'timing_repeats': repeats,
            'gpu_mem_estimate_bytes': gpu['mem_estimate_bytes'], 'gpu_mem_peak_bytes': peak,
            'gpu_min_diag_ratio': gpu['min_diag_ratio'],
            'gpu_flagged_iterations': json.dumps(gpu['flagged_iterations']),
        })
        if verbose:
            r = rows[-1]
            if r['equivalent']:
                verdict = 'equivalent'
            elif r['equivalent_amended']:
                verdict = ('NOT EQUIVALENT under the spec rule; equivalent under the amended rule '
                           '(both residuals at the round-off floor)')
            else:
                verdict = 'NOT EQUIVALENT -- stop and report (Section 3.2)'
            print(f"  P={r['P']}: beta diff {r['beta_rel_diff']:.1e}, R_lin diff {r['rlin_rel_diff']:.1e} "
                  f"-> {verdict}; gelsy {r['t_gelsy_cpu_s']:.3f}s, gels {r['t_gels_cpu_s']:.3f}s, "
                  f"GPU QR {r['t_qr_gpu_s']:.3f}s", flush=True)
    path = root / 'gpu_cpu_equivalence.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EQUIVALENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Check B1: reproduction of every LiL-Q table entry (Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────

REPRODUCTION_COLUMNS = ('status', 'benchmark', 'config', 'quantity', 'paper_value', 'rerun_value',
                        'ratio', 'kind', 'tolerance', 'source', 'note')
_STATUS_ORDER = {'violation': 0, 'missing': 1, 'round-off': 2, 'reported': 3, 'ok': 4}
# A relative error at or below this on both sides is round-off: the factor-2
# test says nothing there (e.g. elasticity's 1e-16 against the paper's 1e-15).
ROUND_OFF_ERROR = 1e-13


def reproduction_check(root: Path) -> Path:
    """Compare every manuscript LiL-Q entry (``experiments/paper_values.py``)
    with the CPU paper-pass rerun; write ``reproduction_check.csv`` with the
    discrepancies first. Violations are reported, never fixed (Section 3.3)."""
    from experiments.paper_values import NOTES, PAPER_VALUES, TOLERANCE_FACTOR
    rows = []
    for pv in PAPER_VALUES:
        summary_path = root / f'{pv.benchmark}_{pv.config}_cpu_paper' / 'summary.json'
        row = {'benchmark': pv.benchmark, 'config': pv.config, 'quantity': pv.quantity,
               'paper_value': pv.value, 'kind': pv.kind, 'source': pv.source,
               'note': '; '.join(n for n in (pv.note, NOTES.get(pv.benchmark, '')) if n)}
        if not summary_path.exists():
            rows.append({**row, 'status': 'missing', 'tolerance': ''})
            continue
        value = pv.rerun(json.loads(summary_path.read_text(encoding='utf-8')))
        ratio = value / pv.value if value is not None and pv.value else None
        if pv.kind in ('iterations', 'rank'):
            tolerance, ok = 'equal', value == pv.value
        elif pv.kind == 'time':
            tolerance, ok = 'none (report only)', None
        else:
            factor = TOLERANCE_FACTOR[pv.kind]
            tolerance = f'factor {factor:g}'
            ok = value is not None and value > 0 and 1 / factor <= ratio <= factor
        status = 'reported' if ok is None else ('ok' if ok else 'violation')
        if (status == 'violation' and pv.kind == 'error' and value is not None
                and max(value, pv.value) <= ROUND_OFF_ERROR):
            status = 'round-off'
        rows.append({**row, 'status': status, 'rerun_value': value, 'ratio': ratio, 'tolerance': tolerance})
    rows.sort(key=lambda r: _STATUS_ORDER[r['status']])
    path = root / 'reproduction_check.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REPRODUCTION_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in REPRODUCTION_COLUMNS})
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Section 6's reference/ and code/ folders
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_README = """Test grids and reference fields (Computational_Package_1_v2.md Section 2).
None of these points is used for collocation. Arrays are on tensor grids,
indexed [x, y] / [x, t] / [x, y, z, t] (numpy meshgrid indexing='ij').

kovasznay.npz    x (301), y (401) uniform on [-0.5,1] x [-0.5,1.5]; exact u, v, p.
bratu.npz        x, y (201 each) uniform on [0,1]^2. No closed-form solution:
                 the test error is the mean square of the PDE residual there.
burgers.npz      x (201) on [-1,1], t (201) on [0,1]: PDE-residual grid (our choice;
                 Section 2 names none).
bl.npz, bl_gravity.npz   x (201) on [0,1], t (201) on [0,T]: PDE-residual grids (our choice).
elasticity.npz   x, y (200 each) on [0,1]^2 (the grid behind the paper's Table 7);
                 exact u_x, u_y.
beltrami.npz     x, y, z (21 each) on [-1,1], t (11) on [0,1] -- the paper's error grid;
                 exact u, v, w, p. Snapshot errors use the same x, y, z at t = 0, 0.25,
                 0.5, 0.75, 1.
darcy_<field>.npz   cell centres x (60), y (220) on [0,1]^2 (normalized) and the FVM
                 pressure (psi) the LiL-Q pressure is compared against, per field.
"""


def save_reference(out_root: Path) -> Path:
    """Write every test grid, and the reference field where one exists, to
    ``<out_root>/reference/`` (Section 2: "Save the grids and reference
    fields under reference/"). The grids come from each problem's own
    ``TEST_GRID``/error code, so they are the ones the errors used."""
    import problems.bratu as bratu
    import problems.burgers as burgers
    import problems.buckley_leverett as bl
    import problems.elasticity as elasticity
    import problems.kovasznay as kov
    import problems.beltrami as bel
    import problems.darcy as darcy
    ref = Path(out_root) / 'reference'
    ref.mkdir(parents=True, exist_ok=True)

    kc = kov.KovasznayConfig()
    kp = kov.KovasznayPhysics(kc)
    x = np.linspace(*kc.x_domain, kov.TEST_GRID[0]); y = np.linspace(*kc.y_domain, kov.TEST_GRID[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    np.savez_compressed(ref / 'kovasznay.npz', x=x, y=y, u=kp.exact_u(X, Y), v=kp.exact_v(X, Y), p=kp.exact_p(X, Y))

    bc = bratu.BratuConfig()
    np.savez_compressed(ref / 'bratu.npz', x=np.linspace(*bc.x_domain, bratu.TEST_GRID[0]),
                        y=np.linspace(*bc.y_domain, bratu.TEST_GRID[1]))
    uc = burgers.BurgersConfig()
    np.savez_compressed(ref / 'burgers.npz', x=np.linspace(*uc.x_domain, burgers.TEST_GRID[0]),
                        t=np.linspace(0.0, uc.T_final, burgers.TEST_GRID[1]))
    for name, cfg in (('bl', bl.BLConfig()), ('bl_gravity', bl.BLConfig.with_gravity())):
        np.savez_compressed(ref / f'{name}.npz', x=np.linspace(*cfg.x_domain, bl.TEST_GRID[0]),
                            t=np.linspace(0.0, cfg.T_final, bl.TEST_GRID[1]),
                            S=bl.reference_solution(cfg))   # finite-difference reference, [i_x, j_t]

    ec = elasticity.ElasticityConfig()
    ep = elasticity.ElasticityPhysics(ec)
    x = np.linspace(*ec.x_domain, elasticity.TEST_GRID[0]); y = np.linspace(*ec.y_domain, elasticity.TEST_GRID[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    np.savez_compressed(ref / 'elasticity.npz', x=x, y=y, u_x=ep.exact_ux(X, Y), u_y=ep.exact_uy(X, Y))

    bp = bel.BeltramiPhysics(bel.BeltramiConfig())
    axes = [np.linspace(*bp.x_domain, 21), np.linspace(*bp.y_domain, 21),
            np.linspace(*bp.z_domain, 21), np.linspace(*bp.t_domain, 11)]
    G = np.meshgrid(*axes, indexing='ij')
    np.savez_compressed(ref / 'beltrami.npz', x=axes[0], y=axes[1], z=axes[2], t=axes[3],
                        u=bp.exact_u(*G), v=bp.exact_v(*G), w=bp.exact_w(*G), p=bp.exact_p(*G))

    for field in DARCY_FIELDS:
        dc = darcy.DarcyConfig(perm_file=f'perm_field_{field}.txt')
        dp = darcy.DarcyPhysics(dc, verbose=False)
        np.savez_compressed(ref / f'darcy_{field}.npz',
                            x=(np.arange(dc.NX_CELLS) + 0.5) / dc.NX_CELLS,
                            y=(np.arange(dc.NY_CELLS) + 0.5) / dc.NY_CELLS,
                            P_fvm=darcy.solve_fvm(dp))

    (ref / 'README.txt').write_text(REFERENCE_README, encoding='utf-8')
    return ref


CODE_DIRS = ('lilq', 'problems', 'experiments', 'scripts', 'tests')


def save_code(out_root: Path) -> Path:
    """Section 6's ``code/``: the commit hash and any diff (``PROVENANCE.json``,
    from git in a checkout or the bundle's own record on the cluster) plus
    the source that produced the results -- including the new scripts
    (figures, monitors, GPU path) the spec asks for."""
    import shutil
    from lilq.provenance import capture_git_info
    repo = Path(__file__).resolve().parent.parent
    code = Path(out_root) / 'code'
    code.mkdir(parents=True, exist_ok=True)
    git = capture_git_info(repo)
    (code / 'PROVENANCE.json').write_text(json.dumps(git, indent=2), encoding='utf-8')
    if git.get('diff'):
        (code / 'uncommitted.diff').write_text(git['diff'], encoding='utf-8')
    for d in CODE_DIRS:
        shutil.copytree(repo / d, code / d, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for f in ('DECISIONS.md', 'requirements.txt', 'pyproject.toml'):
        if (repo / f).exists():
            shutil.copy2(repo / f, code / f)
    return code


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
    parser.add_argument('--gpu-equivalence', action='store_true',
                        help='Run check B3 (Section 3.2) instead of the run plan; needs a CUDA device. '
                             'Writes gpu_cpu_equivalence.csv; exits non-zero if any size fails.')
    parser.add_argument('--reproduction-check', action='store_true',
                        help='Check B1 from the completed runs under --out-root (no solving): '
                             'writes reproduction_check.csv, discrepancies first.')
    parser.add_argument('--save-reference', action='store_true',
                        help="Write the test grids and reference fields to <out-root>/reference/ (Section 2).")
    parser.add_argument('--save-code', action='store_true',
                        help="Write the commit, any diff and the source to <out-root>/code/ (Section 6).")
    args = parser.parse_args()

    if args.save_reference or args.save_code:
        if args.save_reference:
            print(f"Wrote {save_reference(Path(args.out_root))}")
        if args.save_code:
            print(f"Wrote {save_code(Path(args.out_root))}")
        return

    if args.reproduction_check:
        root = Path(args.out_root) / 'B_instrumentation'
        path = reproduction_check(root)
        with open(path, newline='') as f:
            status = [r['status'] for r in csv.DictReader(f)]
        print(f"Wrote {path}: " + ", ".join(f"{s}={status.count(s)}" for s in _STATUS_ORDER))
        return

    import torch
    if args.gpu_equivalence:
        if not torch.cuda.is_available():
            sys.exit("--gpu-equivalence needs a CUDA device.")
        root = Path(args.out_root) / 'B_instrumentation'
        root.mkdir(parents=True, exist_ok=True)
        save_provenance(Path(args.out_root))
        path = run_gpu_equivalence(root, smoke=args.smoke)
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        strict = [r['P'] for r in rows if r['equivalent'] != 'True']
        amended = [r['P'] for r in rows if r['equivalent_amended'] != 'True']
        print(f"Wrote {path}: spec rule "
              + (f"fails at P={strict}" if strict else "passes at every size")
              + "; amended rule " + (f"fails at P={amended}" if amended else "passes at every size"))
        # The amended rule decides whether to stop (DECISIONS.md, 2026-09-24);
        # spec-rule failures are reported in the CSV, not treated as fatal.
        sys.exit(1 if amended else 0)

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
