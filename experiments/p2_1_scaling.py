"""
Package 2, item 7 (P2-1): scaling of LiL-Q in P and N, on the CPU and the A100
=============================================================================

The advisor's instructions of 4 October 2026, Section 10, and checks C8'
and C9 (Section 12.3).

**Series:**
- ``kovasznay_P`` (N/P about 3): p_d in {25, 30, 35, 40, 50}, i.e.
  P = 1,875, 2,700, 3,675, 4,800, 7,500. The paper's grid formula
  (``run_kovasznay.K_RATIO``), stopping rule and K_max.
- ``kovasznay_N`` (P = 1,200, p_d = 20): N/P in {3, 5, 10, 20} on the paper's
  grid type (equispaced). The density per target ratio is chosen exactly as
  in Component C (``component_c.k_for_ratio`` and ``_config``), whose runs
  already give the errors; here the point is time and memory.

- ``beltrami`` (the pinned configuration of Section 6.7):
  - N_vel = 6, 7, 8, with N_p = 8, 9, 10: P = 7,984, 13,764, 22,288;
  - interior grids 8^4, 10^4, 11^4 (Section 10);
  - the boundary and initial rows and the pressure pins follow the paper's
    run's rule for the respective grid (``beltrami_config``): N_ic = the
    grid, N_bc = N_t_bc = the grid - 2, one pressure pin per temporal level
    (N_p pins). Section 10's "eight pressure pins" is the paper's count; the
    pins close the N_p-dimensional pressure null space, so N_p pins are
    needed at N_p = 9, 10 (DECISIONS.md, batch 4);
  - N_vel = 6 is exactly the paper's run
    (``run_beltrami_pinned.pinned_config``);
  - errors as the paper: u, v, w, p and the pin-gauge p at t = 1 and at the
    other snapshot times, and the time-averaged relative L2 errors.
  - A size that does not fit in the A100's 40 GB is recorded as
    ``did_not_fit``, with the error, the system's size and the bytes of A
    (Section 10: report the size at which it does not fit).

**Protocol per (series, size, device)**, in a fresh process of its own, since
peak host memory is a per-process maximum:
1. one untimed warm-up run;
2. the GPU peak-memory counter reset (A100);
3. one timed run with the diagnostics off. Section 10's protocol for this
   item: a single timing, not the median of three; disclosed in
   ``run.json``;
4. the peak host memory (the process's maximum resident set, read before
   anything else is allocated) and the GPU peak
   (``torch.cuda.max_memory_allocated``, Package 1's accounting);
5. off the clock: the errors on the paper's 301 x 401 test grid
   (``problems.kovasznay.make_test_error_fn``, the function behind
   package1's ``test_eps_*``), and kappa and the numerical rank of the last
   iteration's system by SVD.

**Records:**
- ``iterations.csv``: per iteration, t_assemble_s and t_solve_s (and the GPU
  transfer parts and peak);
- ``run.json``: total time (the solver's ``solve_time_total``, the quantity
  of package1's clean timing), the iterations, peak memory, kappa, rank,
  errors, threads and hardware.

**Summary** (``summarize``):
- ``scaling.csv``;
- the fitted exponents of total time against P (expected about 3) and
  against N (expected about 1) per device, by least squares on the log-log
  points;
- **check C8'** at the paper-size points, Kovasznay P = 1,875 and Beltrami
  P = 7,984 (the pinned run):
  - CPU: every error equals package1's to 10 digits (relative 1e-10), and
    the time is within 15% of package1's clean time;
  - A100 (Kovasznay only: the paper has no Beltrami GPU point): the errors
    equal package1's GPU errors to 10 digits;
  - Grace reproduced the Kovasznay point bit for bit between waves 1 and 2,
    so 10 digits is attainable there; a laptop is not expected to;
- **check C9's records:** threads, the warm-up, the node's exclusivity
  (provenance per job);
- ``figures/scaling.{pdf,png}``: time and memory against P and N, log-log, both devices.

**Outputs** under ``P2_1_scaling/``:
- ``<problem>/<series>_<size>_<device>/{run.json, iterations.csv}``;
- ``scaling.csv``, ``check_c8prime.json``;
- ``figures/``, ``provenance_<device>_<job id>/``.

Usage::

    python experiments/p2_1_scaling.py series --series kovasznay_P kovasznay_N beltrami --device cpu --out <stage root>
    python experiments/p2_1_scaling.py summarize --out <stage root> --package1 <package1>
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from lilq.source_lock import current_commit

SERIES = {
    'kovasznay_P': {'problem': 'kovasznay', 'sizes': (25, 30, 35, 40, 50), 'axis': 'P'},   # p_d
    'kovasznay_N': {'problem': 'kovasznay', 'sizes': (3, 5, 10, 20), 'axis': 'N'},         # N/P at P = 1,200
    'beltrami': {'problem': 'beltrami', 'sizes': (6, 7, 8), 'axis': 'P'},                   # N_vel
}
BELTRAMI = {6: (8, 8), 7: (9, 10), 8: (10, 11)}       # N_vel: (N_p, interior grid per dimension), Section 10
N_SERIES_PD = 20
PAPER_POINTS = {'kovasznay_P': 25, 'beltrami': 6}
C8_DIGITS = 1e-10
C8_TIME = 0.15
ERROR_KEYS = ('eps_u', 'eps_v', 'eps_p', 'eps_p_meanfree')
SNAPSHOT_FIELDS = ('u', 'v', 'w', 'p', 'p_pin_gauge')
BELTRAMI_ERRORS = ('rel_l2_u', 'rel_l2_v', 'rel_l2_w', 'rel_l2_p', 'rel_l2_p_pin_gauge')


def run_name(series, size, device):
    return f'{series}_{size}_{device}'


def kovasznay_config(series, size, device):
    from experiments.run_kovasznay import K_RATIO, MAX_ITER, TOL
    from problems.kovasznay import KovasznayConfig
    import dataclasses
    if series == 'kovasznay_P':
        config = KovasznayConfig(N_x=size, N_y=size, k_ratio=K_RATIO, max_iter=MAX_ITER, tol=TOL)
    else:
        import experiments.component_c as cc
        k, _ = cc.k_for_ratio('kovasznay', N_SERIES_PD, size)
        config = cc._config('kovasznay', N_SERIES_PD, k, 'paper', None)[0]
    return dataclasses.replace(config, use_gpu=(device == 'cuda'))


def beltrami_config(n_vel, device):
    """The paper's pinned run's rule at N_vel: N_p and the interior grid from
    Section 10, N_ic = the grid, N_bc = N_t_bc = the grid - 2, N_p pins. At
    N_vel = 6 this is ``run_beltrami_pinned.pinned_config()`` exactly."""
    import dataclasses
    from experiments.run_beltrami_pinned import DEFAULT_BASIS
    from lilq.solvers import LILQ_PAPER_KMAX
    from problems.beltrami import BeltramiConfig
    n_p, g = BELTRAMI[n_vel]
    config = BeltramiConfig(N_vel=n_vel, N_p=n_p, basis_type=DEFAULT_BASIS, n_pressure_pin_levels=n_p,
                            max_iter=LILQ_PAPER_KMAX, N_x=g, N_y=g, N_z=g, N_t=g, N_bc=g - 2, N_t_bc=g - 2, N_ic=g)
    return dataclasses.replace(config, use_gpu=(device == 'cuda'))


def beltrami_rows(config):
    """Rows of the Beltrami system: 4 equations at the interior points, the 3
    velocity components on the 6 faces and at t = 0, and the pins."""
    return (4 * config.N_x * config.N_y * config.N_z * config.N_t + 3 * 6 * config.N_bc ** 2 * config.N_t_bc
            + 3 * config.N_ic ** 3 + config.n_pressure_pin_levels)


def config_for(series, size, device):
    return beltrami_config(size, device) if SERIES[series]['problem'] == 'beltrami' else \
        kovasznay_config(series, size, device)


def peak_host_bytes():
    """The process's peak resident set so far: ``ru_maxrss`` on Linux (KiB),
    the peak working set elsewhere."""
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except ImportError:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, 'peak_wset', 0) or info.rss)


# ---------------------------------------------------------------- one run (its own process)

def _solve(problem, config, **kw):
    if problem == 'beltrami':
        from problems.beltrami import solve_beltrami
        return solve_beltrami(config, verbose=False, diagnostics=False, **kw)
    from problems.kovasznay import solve_kovasznay
    return solve_kovasznay(config, verbose=False, diagnostics=False, **kw)


def _errors(problem, config, r):
    """The paper's error measures for the problem, off the clock."""
    if problem == 'beltrami':
        out = {k: float(r[k]) for k in BELTRAMI_ERRORS}
        for snap in r['snapshots']:
            for f in SNAPSHOT_FIELDS:
                out[f't{snap["t"]:g}_{f}'] = float(snap[f])
        return out
    from problems.kovasznay import KovasznayPhysics, make_test_error_fn
    theta = np.concatenate([r['theta_u'], r['theta_v'], r['theta_p']])
    e = make_test_error_fn(KovasznayPhysics(config), r['basis_u'], r['basis_v'], r['basis_p'])(theta)
    return {k: float(e[k]) for k in ERROR_KEYS}


def run_one(series, size, device, out_root):
    """Warm-up, the timed run, the memory readings, then the errors and the
    conditioning off the clock. Writes the run folder; returns run.json. On
    the A100, a system that does not fit is recorded as ``did_not_fit``."""
    problem = SERIES[series]['problem']
    config = config_for(series, size, device)
    run_dir = Path(out_root) / 'P2_1_scaling' / problem / run_name(series, size, device)
    run_dir.mkdir(parents=True, exist_ok=True)
    cuda = device == 'cuda'
    base = {'series': series, 'problem': problem, 'size': size, 'device': device,
            'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'slurm_job': os.environ.get('SLURM_JOB_ID'),
            'host': os.environ.get('HOSTNAME') or os.environ.get('COMPUTERNAME'), 'commit': current_commit()}
    if cuda:
        import torch
        base['gpu_name'] = torch.cuda.get_device_name(0)
        base['gpu_total_bytes'] = int(torch.cuda.get_device_properties(0).total_memory)
    try:
        t0 = time.perf_counter()
        warm = _solve(problem, config)
        warm_s = time.perf_counter() - t0
        del warm
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = _solve(problem, config, return_final_system=True)
        wall_s = time.perf_counter() - t0
    except Exception as exc:                       # out of GPU memory: Section 10 asks for the size
        if not (cuda and 'out of memory' in str(exc).lower()):
            raise
        P = config.N_vel ** 4 * 3 + config.N_p ** 4 if problem == 'beltrami' else 3 * config.N_x ** 2
        N = beltrami_rows(config) if problem == 'beltrami' else None
        record = {**base, 'status': 'did_not_fit', 'P': P, 'N': N, 'A_bytes': 8 * N * P if N else None,
                  'error': f'{type(exc).__name__}: {str(exc)[:500]}'}
        (run_dir / 'run.json').write_text(json.dumps(record, indent=2))
        return record
    host_peak = peak_host_bytes()
    gpu_peak = int(torch.cuda.max_memory_allocated()) if cuda else None
    # off the clock
    h = r['history']
    errors = _errors(problem, config, r)
    A = r.pop('A_final')
    N_rows, P = A.shape
    s = np.linalg.svd(A, compute_uv=False)
    kappa = float(s[0] / s[-1])
    rank = int((s > max(A.shape) * np.finfo(float).eps * s[0]).sum())
    del A
    rows = []
    for i, k in enumerate(h['iteration']):
        row = {'k': k, 't_assemble_s': h['t_assemble'][i], 't_solve_s': h['t_solve'][i],
               'iteration_time_s': h['solve_time'][i], 'rel_coeff_change': h['coeff_change'][i]}
        if cuda:
            parts = h['gpu_parts'][i] if problem == 'beltrami' else \
                {key: r['gpu_qr'][key][i] for key in ('h2d_s', 'qr_solve_s', 'd2h_s')}
            row.update(h2d_s=parts['h2d_s'], qr_solve_s=parts['qr_solve_s'], d2h_s=parts['d2h_s'],
                       gpu_mem_peak_bytes=h['gpu_mem_peak_bytes'][i])
        rows.append(row)
    with open(run_dir / 'iterations.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    extra = ({'N_vel': config.N_vel, 'N_p': config.N_p, 'grid': config.N_x, 'N_bc': config.N_bc,
              'N_t_bc': config.N_t_bc, 'N_ic': config.N_ic, 'pins': config.n_pressure_pin_levels}
             if problem == 'beltrami' else {'p_d': config.N_x, 'k_ratio': config.k_ratio})
    record = {
        **base, 'status': 'ok', 'P': int(P), 'N': int(N_rows), 'N_over_P': N_rows / P, **extra,
        'iterations': len(h['iteration']), 'stop': 'tolerance' if h['coeff_change'][-1] < config.tol else 'K_max',
        'K_max': config.max_iter, 'tol': config.tol,
        'total_time_s': r['solve_time_total'], 'wall_time_s': wall_s, 'warmup_time_s': warm_s,
        't_assemble_total_s': float(sum(h['t_assemble'])), 't_solve_total_s': float(sum(h['t_solve'])),
        'peak_host_bytes': host_peak, 'peak_gpu_bytes': gpu_peak, 'A_bytes': 8 * int(N_rows) * int(P),
        'kappa': kappa, 'rank': rank, **errors,
        'timing_protocol': 'one untimed warm-up, then one timed run with the diagnostics off (Section 10: a single '
                           'timing, not the median of three)',
    }
    (run_dir / 'run.json').write_text(json.dumps(record, indent=2))
    return record


def run_series(series_list, device, out_root, sizes=None):
    """Each size in a fresh process (``one``); a finished run is skipped."""
    from lilq.provenance import save_provenance
    # one folder per job: the two CPU jobs may run at the same time
    save_provenance(Path(out_root) / 'P2_1_scaling' / f"provenance_{device}_{os.environ.get('SLURM_JOB_ID', 'local')}")
    for series in series_list:
        for size in sizes or SERIES[series]['sizes']:
            run_dir = Path(out_root) / 'P2_1_scaling' / SERIES[series]['problem'] / run_name(series, size, device)
            if (run_dir / 'run.json').exists():
                print(f'  {run_name(series, size, device)}: done, skipping', flush=True)
                continue
            subprocess.run([sys.executable, __file__, 'one', '--series', series, '--size', str(size),
                            '--device', device, '--out', str(out_root)], check=True)
            r = json.loads((run_dir / 'run.json').read_text())
            if r['status'] == 'did_not_fit':
                print(f"  {run_name(series, size, device)}: P = {r['P']}, N = {r['N']}: does not fit on the GPU "
                      f"(A alone {r['A_bytes'] / 1e9:.1f} GB): {r['error'][:120]}", flush=True)
                continue
            gpu = f", GPU peak {r['peak_gpu_bytes'] / 1e9:.2f} GB" if r['peak_gpu_bytes'] else ''
            print(f"  {run_name(series, size, device)}: P = {r['P']}, N = {r['N']} (N/P {r['N_over_P']:.2f}), "
                  f"{r['iterations']} iterations, {r['total_time_s']:.2f} s (assembly {r['t_assemble_total_s']:.2f}, "
                  f"solve {r['t_solve_total_s']:.2f}), host peak {r['peak_host_bytes'] / 1e9:.2f} GB{gpu}, "
                  f"kappa {r['kappa']:.1e}, rank {r['rank']}, "
                  + (f"E_u {r['eps_u']:.2e}" if r['problem'] == 'kovasznay' else
                     f"u(t=1) {r['t1_u']:.2e}, p(t=1) {r['t1_p']:.2e}"), flush=True)


# ---------------------------------------------------------------- summary

def _runs(out_root, ok_only=True):
    runs = [json.loads(p.read_text()) for p in sorted((Path(out_root) / 'P2_1_scaling').glob('*/*/run.json'))]
    return [r for r in runs if r.get('status', 'ok') == 'ok'] if ok_only else runs


def exponent(xs, ys):
    """Least-squares slope of log y against log x (None with fewer than 2 points)."""
    if len(xs) < 2:
        return None
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def _clean_times(B):
    clean = {}
    with open(B / 'clean_timing' / 'clean_timing.csv', newline='') as f:
        for r in csv.DictReader(f):
            if r['clean_time_s']:
                clean[(r['run'], r['quantity'])] = float(r['clean_time_s'])
    return clean


def _package1_errors(B, problem, device, P):
    """``(name, iterations, {key: value})`` of package1's paper point, in this
    module's key names; None where package1 has no such run."""
    if problem == 'kovasznay':
        name = f'kovasznay_P{P}_{device}_paper'
        if not (B / name / 'summary.json').exists():
            return None
        p1 = json.loads((B / name / 'summary.json').read_text())
        return name, p1['n_outer_iters'], {k: p1[f'test_{k}'] for k in ERROR_KEYS}
    if device != 'cpu':
        return None                                 # the paper has no Beltrami GPU point
    name = 'beltrami_pinned'
    p1 = json.loads((B / name / 'report.json').read_text())
    errors = {'rel_l2_p_pin_gauge': p1['rel_l2_p_pin_gauge']}
    for snap in p1['snapshots']:
        for f in SNAPSHOT_FIELDS:
            errors[f't{snap["t"]:g}_{f}'] = snap[f]
    return name, p1['n_outer_iters'], errors


def check_c8prime(runs, package1):
    """C8' at the paper-size points: every recorded error to 10 digits; the CPU
    time within 15% of package1's clean time."""
    B = Path(package1) / 'B_instrumentation'
    clean = _clean_times(B)
    out = []
    for series, size in PAPER_POINTS.items():
        for device in ('cpu', 'cuda'):
            run = next((r for r in runs if (r['series'], r['size'], r['device']) == (series, size, device)), None)
            p1 = _package1_errors(B, SERIES[series]['problem'], device, run['P']) if run else None
            if run is None or p1 is None:
                continue
            name, iters, errors = p1
            diffs = {k: abs(run[k] - v) / abs(v) for k, v in errors.items()}
            row = {'point': name, 'device': device, 'errors_rel_diff': diffs,
                   'errors_pass': all(d <= C8_DIGITS for d in diffs.values()),
                   'iterations': run['iterations'], 'iterations_package1': iters}
            if device == 'cpu':
                ref = clean.get((name, 'solve_time_total'))
                row.update(time_s=run['total_time_s'], package1_time_s=ref,
                           time_ratio=run['total_time_s'] / ref if ref else None,
                           time_pass=bool(ref and abs(run['total_time_s'] / ref - 1) <= C8_TIME))
                row['passed'] = row['errors_pass'] and row['time_pass']
            else:
                row['passed'] = row['errors_pass']
            out.append(row)
    return {'check': "C8' (item 7)", 'error_tolerance': C8_DIGITS, 'time_tolerance': C8_TIME, 'points': out,
            'passed': bool(out) and all(r['passed'] for r in out), 'commit': current_commit()}


def summarize(out_root, package1=None):
    runs = _runs(out_root)
    out = Path(out_root) / 'P2_1_scaling'
    cols = ['series', 'problem', 'size', 'device', 'status', 'P', 'N', 'N_over_P', 'p_d', 'N_vel', 'N_p', 'grid',
            'iterations', 'stop', 'total_time_s', 't_assemble_total_s', 't_solve_total_s', 'warmup_time_s',
            'peak_host_bytes', 'peak_gpu_bytes', 'A_bytes', 'gpu_name', 'kappa', 'rank', *ERROR_KEYS,
            *BELTRAMI_ERRORS, 't1_u', 't1_v', 't1_w', 't1_p', 't1_p_pin_gauge', 'error', 'commit']
    with open(out / 'scaling.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(_runs(out_root, ok_only=False))
    fits = []
    for series, spec in SERIES.items():
        for device in ('cpu', 'cuda'):
            g = sorted((r for r in runs if r['series'] == series and r['device'] == device), key=lambda r: r[spec['axis']])
            if len(g) >= 2:
                x = [r[spec['axis']] for r in g]
                fits.append({'series': series, 'device': device, 'against': spec['axis'], 'points': len(g),
                             'exponent_total_time': exponent(x, [r['total_time_s'] for r in g]),
                             'exponent_solve_time': exponent(x, [r['t_solve_total_s'] for r in g]),
                             'exponent_assembly_time': exponent(x, [r['t_assemble_total_s'] for r in g])})
    if fits:
        with open(out / 'exponents.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(fits[0]))
            w.writeheader()
            w.writerows(fits)
    c8 = None
    if package1:
        c8 = check_c8prime(runs, package1)
        (out / 'check_c8prime.json').write_text(json.dumps(c8, indent=2))
    figures(out, runs)
    return runs, fits, c8


def figures(out, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(SERIES), figsize=(4.5 * len(SERIES), 7), squeeze=False)
    for j, (series, spec) in enumerate(SERIES.items()):
        for device, mk in (('cpu', 'o'), ('cuda', 's')):
            g = sorted((r for r in runs if r['series'] == series and r['device'] == device), key=lambda r: r[spec['axis']])
            if not g:
                continue
            x = [r[spec['axis']] for r in g]
            label = {'cpu': 'CPU', 'cuda': 'A100'}[device]
            axes[0, j].loglog(x, [r['total_time_s'] for r in g], mk + '-', ms=4, label=f'{label}, total')
            axes[0, j].loglog(x, [r['t_solve_total_s'] for r in g], mk + ':', ms=3, label=f'{label}, solve')
            axes[1, j].loglog(x, [r['peak_host_bytes'] / 1e9 for r in g], mk + '-', ms=4, label=f'{label}, host')
            if device == 'cuda':
                axes[1, j].loglog(x, [r['peak_gpu_bytes'] / 1e9 for r in g], mk + ':', ms=3, label='A100, device')
        what = {'kovasznay_P': 'P (N/P about 3)', 'kovasznay_N': 'N (P = 1,200)', 'beltrami': 'P'}[series]
        name = spec['problem'].capitalize()
        axes[0, j].set(xlabel=what, ylabel='time (s)', title=f'{name}: time against {spec["axis"]}')
        axes[1, j].set(xlabel=what, ylabel='peak memory (GB)', title=f'{name}: memory against {spec["axis"]}')
    for ax in axes.ravel():
        ax.grid(True, which='both', alpha=0.3)
        if ax.has_data():
            ax.legend(fontsize=7)
    fig.tight_layout()
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures' / 'scaling.pdf')
    fig.savefig(out / 'figures' / 'scaling.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 2, item 7: scaling in P and N.')
    ap.add_argument('stage', choices=('one', 'series', 'summarize'))
    ap.add_argument('--out', required=True, help='the stage root')
    ap.add_argument('--series', nargs='+', choices=tuple(SERIES), default=list(SERIES))
    ap.add_argument('--size', type=float, help='one: p_d (P series) or N/P (N series)')
    ap.add_argument('--sizes', type=float, nargs='+', help='series: a subset of the sizes')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--package1', help="summarize: package1, for check C8'")
    args = ap.parse_args(argv)
    as_size = lambda v: int(v) if float(v).is_integer() else v  # noqa: E731
    if args.stage == 'one':
        run_one(args.series[0], as_size(args.size), args.device, args.out)
        return
    if args.stage == 'series':
        run_series(args.series, args.device, args.out, [as_size(v) for v in args.sizes] if args.sizes else None)
        return
    _, fits, c8 = summarize(args.out, args.package1)
    for f in fits:
        print(f"  {f['series']:12s} {f['device']:4s}: time ~ {f['against']}^{f['exponent_total_time']:.2f} "
              f"(solve {f['exponent_solve_time']:.2f}, assembly {f['exponent_assembly_time']:.2f}; {f['points']} points)")
    if c8:
        for p in c8['points']:
            print(f"  C8' {p['point']}: errors to 1e-10: {p['errors_pass']} (max {max(p['errors_rel_diff'].values()):.1e})"
                  + (f", time {p['time_s']:.2f} s against {p['package1_time_s']:.2f} s: {p['time_pass']}"
                     if p['device'] == 'cpu' else ''))


if __name__ == '__main__':
    main()
