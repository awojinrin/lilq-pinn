"""
Package 2, item 7 (P2-1): scaling of LiL-Q in P and N, on the CPU and the A100
=============================================================================

The advisor's instructions of 4 October 2026, Section 10, and checks C8'
and C9 (Section 12.3). This batch: Kovasznay. Beltrami is the next.

**Series:**
- ``kovasznay_P`` (N/P about 3): p_d in {25, 30, 35, 40, 50}, i.e.
  P = 1,875, 2,700, 3,675, 4,800, 7,500. The paper's grid formula
  (``run_kovasznay.K_RATIO``), stopping rule and K_max.
- ``kovasznay_N`` (P = 1,200, p_d = 20): N/P in {3, 5, 10, 20} on the paper's
  grid type (equispaced). The density per target ratio is chosen exactly as
  in Component C (``component_c.k_for_ratio`` and ``_config``), whose runs
  already give the errors; here the point is time and memory.

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
- **check C8'** at the paper-size point (P = 1,875):
  - CPU: E_u, E_v, E_p and the mean-free E_p equal package1's to 10 digits
    (relative 1e-10), and the time is within 15% of package1's clean time;
  - A100: the errors equal package1's GPU errors to 10 digits;
  - Grace reproduced these bit for bit between waves 1 and 2, so 10 digits
    is attainable there; a laptop is not expected to;
- **check C9's records:** threads, the warm-up, the node's exclusivity
  (provenance per job);
- ``figures/``: time and memory against P and N, log-log, both devices.

**Outputs** under ``P2_1_scaling/``:
- ``kovasznay/<series>_<size>_<device>/{run.json, iterations.csv}``;
- ``scaling.csv``, ``check_c8prime.json``;
- ``figures/``, ``provenance_<device>/``.

Usage::

    python experiments/p2_1_scaling.py series --series kovasznay_P kovasznay_N --device cpu --out <stage root>
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
}
N_SERIES_PD = 20
PAPER_POINT = ('kovasznay_P', 25)
C8_DIGITS = 1e-10
C8_TIME = 0.15
ERROR_KEYS = ('eps_u', 'eps_v', 'eps_p', 'eps_p_meanfree')


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

def run_one(series, size, device, out_root):
    """Warm-up, the timed run, the memory readings, then the errors and the
    conditioning off the clock. Writes the run folder; returns run.json."""
    from problems.kovasznay import KovasznayPhysics, make_test_error_fn, solve_kovasznay
    config = kovasznay_config(series, size, device)
    run_dir = Path(out_root) / 'P2_1_scaling' / SERIES[series]['problem'] / run_name(series, size, device)
    run_dir.mkdir(parents=True, exist_ok=True)
    cuda = device == 'cuda'
    if cuda:
        import torch
    t0 = time.perf_counter()
    warm = solve_kovasznay(config, verbose=False, diagnostics=False)
    warm_s = time.perf_counter() - t0
    del warm
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    r = solve_kovasznay(config, verbose=False, diagnostics=False, return_final_system=True)
    wall_s = time.perf_counter() - t0
    host_peak = peak_host_bytes()
    gpu_peak = int(torch.cuda.max_memory_allocated()) if cuda else None
    # off the clock
    h = r['history']
    physics = KovasznayPhysics(config)
    theta = np.concatenate([r['theta_u'], r['theta_v'], r['theta_p']])
    errors = make_test_error_fn(physics, r['basis_u'], r['basis_v'], r['basis_p'])(theta)
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
            g = r['gpu_qr']
            row.update(h2d_s=g['h2d_s'][i], qr_solve_s=g['qr_solve_s'][i], d2h_s=g['d2h_s'][i],
                       gpu_mem_peak_bytes=h['gpu_mem_peak_bytes'][i])
        rows.append(row)
    with open(run_dir / 'iterations.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gpu_name = None
    if cuda:
        gpu_name = torch.cuda.get_device_name(0)
    record = {
        'series': series, 'problem': SERIES[series]['problem'], 'size': size, 'device': device,
        'P': int(P), 'N': int(N_rows), 'N_over_P': N_rows / P, 'p_d': config.N_x, 'k_ratio': config.k_ratio,
        'iterations': len(h['iteration']), 'stop': 'tolerance' if h['coeff_change'][-1] < config.tol else 'K_max',
        'K_max': config.max_iter, 'tol': config.tol,
        'total_time_s': r['solve_time_total'], 'wall_time_s': wall_s, 'warmup_time_s': warm_s,
        't_assemble_total_s': float(sum(h['t_assemble'])), 't_solve_total_s': float(sum(h['t_solve'])),
        'peak_host_bytes': host_peak, 'peak_gpu_bytes': gpu_peak, 'gpu_name': gpu_name,
        'kappa': kappa, 'rank': rank, **{k: float(errors[k]) for k in ERROR_KEYS},
        'timing_protocol': 'one untimed warm-up, then one timed run with the diagnostics off (Section 10: a single '
                           'timing, not the median of three)',
        'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'slurm_job': os.environ.get('SLURM_JOB_ID'), 'host': os.environ.get('HOSTNAME') or os.environ.get('COMPUTERNAME'),
        'commit': current_commit(),
    }
    (run_dir / 'run.json').write_text(json.dumps(record, indent=2))
    return record


def run_series(series_list, device, out_root, sizes=None):
    """Each size in a fresh process (``one``); a finished run is skipped."""
    from lilq.provenance import save_provenance
    save_provenance(Path(out_root) / 'P2_1_scaling' / f'provenance_{device}')
    for series in series_list:
        for size in sizes or SERIES[series]['sizes']:
            run_dir = Path(out_root) / 'P2_1_scaling' / SERIES[series]['problem'] / run_name(series, size, device)
            if (run_dir / 'run.json').exists():
                print(f'  {run_name(series, size, device)}: done, skipping', flush=True)
                continue
            subprocess.run([sys.executable, __file__, 'one', '--series', series, '--size', str(size),
                            '--device', device, '--out', str(out_root)], check=True)
            r = json.loads((run_dir / 'run.json').read_text())
            gpu = f", GPU peak {r['peak_gpu_bytes'] / 1e9:.2f} GB" if r['peak_gpu_bytes'] else ''
            print(f"  {run_name(series, size, device)}: P = {r['P']}, N = {r['N']} (N/P {r['N_over_P']:.2f}), "
                  f"{r['iterations']} iterations, {r['total_time_s']:.2f} s (assembly {r['t_assemble_total_s']:.2f}, "
                  f"solve {r['t_solve_total_s']:.2f}), host peak {r['peak_host_bytes'] / 1e9:.2f} GB{gpu}, "
                  f"kappa {r['kappa']:.1e}, rank {r['rank']}, E_u {r['eps_u']:.2e}", flush=True)


# ---------------------------------------------------------------- summary

def _runs(out_root):
    return [json.loads(p.read_text()) for p in sorted((Path(out_root) / 'P2_1_scaling').glob('*/*/run.json'))]


def exponent(xs, ys):
    """Least-squares slope of log y against log x (None with fewer than 2 points)."""
    if len(xs) < 2:
        return None
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def check_c8prime(runs, package1):
    """C8' at the paper-size point: errors to 10 digits on both devices, the
    CPU time within 15% of package1's clean time."""
    B = Path(package1) / 'B_instrumentation'
    clean = {}
    with open(B / 'clean_timing' / 'clean_timing.csv', newline='') as f:
        for r in csv.DictReader(f):
            if r['clean_time_s']:
                clean[(r['run'], r['quantity'])] = float(r['clean_time_s'])
    out = []
    series, size = PAPER_POINT
    for device in ('cpu', 'cuda'):
        run = next((r for r in runs if (r['series'], r['size'], r['device']) == (series, size, device)), None)
        if run is None:
            continue
        name = f"kovasznay_P{run['P']}_{device}_paper"
        p1 = json.loads((B / name / 'summary.json').read_text())
        diffs = {k: abs(run[k] - p1[f'test_{k}']) / abs(p1[f'test_{k}']) for k in ERROR_KEYS}
        row = {'point': name, 'device': device, 'errors_rel_diff': diffs,
               'errors_pass': all(d <= C8_DIGITS for d in diffs.values()),
               'iterations': run['iterations'], 'iterations_package1': p1['n_outer_iters']}
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
    cols = ['series', 'problem', 'size', 'device', 'P', 'N', 'N_over_P', 'p_d', 'iterations', 'stop', 'total_time_s',
            't_assemble_total_s', 't_solve_total_s', 'warmup_time_s', 'peak_host_bytes', 'peak_gpu_bytes', 'gpu_name',
            'kappa', 'rank', *ERROR_KEYS, 'commit']
    with open(out / 'scaling.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(runs)
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
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
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
        what = 'P (N/P about 3)' if spec['axis'] == 'P' else 'N (P = 1,200)'
        axes[0, j].set(xlabel=what, ylabel='time (s)', title=f'Kovasznay: time against {spec["axis"]}')
        axes[1, j].set(xlabel=what, ylabel='peak memory (GB)', title=f'Kovasznay: memory against {spec["axis"]}')
    for ax in axes.ravel():
        ax.grid(True, which='both', alpha=0.3)
        if ax.has_data():
            ax.legend(fontsize=7)
    fig.tight_layout()
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures' / 'scaling_kovasznay.pdf')
    fig.savefig(out / 'figures' / 'scaling_kovasznay.png', dpi=150)
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
