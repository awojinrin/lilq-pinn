"""
Package 3, checks K0 and K8 (the advisor's instructions of 8 October 2026, Section 7)
====================================================================================

**K0** (on the Grace node, before B3 level 2 is launched). ``scipy.linalg.lstsq(...,
lapack_driver='gelsy')`` and the column-pivoted QR on a random 600,000 x 7,984 matrix:
4.79e9 entries, above 2^32; 38 GB. B3 level 2's system is 665,331 x 7,984 (5.3e9).
For a consistent right-hand side b = A x0, both must run, and the solution must
recover x0 with the residual at round-off.
- The pivoted QR goes through ``scipy.linalg.qr_multiply(..., pivoting=True)``,
  which applies Q^T to b inside LAPACK, so Q (another 38 GB) is never formed.
- Pass: both relative residuals ||A x - b|| / ||b|| and both relative errors
  ||x - x0|| / ||x0|| are at most 1e-12. A Gaussian 600,000 x 7,984 matrix has
  kappa about 1.26, so round-off is about 1e-15.

**K8** (after all code changes, at the Package 3 commit). The paper's runs
reproduce ``package1`` to all digits: the iteration count, and ``norm_R_h`` and
``norm_Rlin_h`` at every row of ``iterations.csv``.
- **The paper's Beltrami run:** the pinned run of Section 3.7 (equispaced,
  P = 7,984), ``solve_beltrami(run_beltrami_pinned.pinned_config())`` with its
  logger, as ``run_beltrami_pinned`` runs it (``package1/B_instrumentation/
  beltrami_pinned``, 48 threads).
- **The paper's viscous BL run at P = 576:** component B's paper pass,
  ``problems.buckley_leverett.run_lil_q`` at ``run_bl.paper_setup(24, False)``
  (``bl_P576_cpu_paper``, 48 threads).

Each writes ``<out>/P3_checks/k0.json`` or ``k8.json``; K8's reruns go in
``P3_checks/k8/<run>/``.

Usage::

    python experiments/p3_k0_k8.py k0 --out <root> [--m 600000 --n 7984]
    python experiments/p3_k0_k8.py k8 --out <root> --package1 <package1>
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np
import scipy.linalg as sla

from lilq.instrumentation import EPS_MACH
from lilq.provenance import capture_blas_thread_env
from lilq.source_lock import current_commit

K0_SHAPE = (600_000, 7_984)
K0_TOL = 1e-12


def k0(m=K0_SHAPE[0], n=K0_SHAPE[1], seed=0):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    A = rng.standard_normal((m, n))
    x0 = rng.standard_normal(n)
    b = A @ x0
    out = {'check': 'K0', 'shape': [m, n], 'entries': m * n, 'above_2_32': m * n > 2 ** 32, 'GB': A.nbytes / 1e9,
           'generate_s': time.perf_counter() - t0}
    nb, nx = float(np.linalg.norm(b)), float(np.linalg.norm(x0))
    t0 = time.perf_counter()
    x, _, rank, _ = sla.lstsq(A, b, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)
    out['gelsy'] = {'seconds': time.perf_counter() - t0, 'rank': int(rank),
                    'rel_residual': float(np.linalg.norm(A @ x - b)) / nb, 'rel_error': float(np.linalg.norm(x - x0)) / nx}
    del x
    t0 = time.perf_counter()
    qtb, R, piv = sla.qr_multiply(A, b[None, :], mode='right', pivoting=True)     # b Q, i.e. (Q^T b)^T; Q never formed
    y = sla.solve_triangular(R[:n, :n], qtb.ravel()[:n])
    x = np.empty(n)
    x[piv] = y
    out['pivoted_qr'] = {'seconds': time.perf_counter() - t0, 'via': 'scipy.linalg.qr_multiply(pivoting=True)',
                         'rel_residual': float(np.linalg.norm(A @ x - b)) / nb,
                         'rel_error': float(np.linalg.norm(x - x0)) / nx}
    out['tolerance'] = K0_TOL
    out['passed'] = all(out[k][q] <= K0_TOL for k in ('gelsy', 'pivoted_qr') for q in ('rel_residual', 'rel_error'))
    out['threads'] = capture_blas_thread_env().get('OMP_NUM_THREADS')
    out['scipy'] = __import__('scipy').__version__
    out['commit'] = current_commit()
    return out


# ---------------------------------------------------------------- K8

def _log(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def compare_with_package1(here, package1_run):
    """Every row of package1's iterations.csv: the same number of rows, and
    norm_R_h and norm_Rlin_h equal as written (all digits)."""
    a, b = _log(here), _log(package1_run)
    diffs = [{'k': rb['k'], 'column': c, 'here': ra.get(c), 'package1': rb.get(c)}
             for ra, rb in zip(a, b) for c in ('norm_R_h', 'norm_Rlin_h') if ra.get(c) != rb.get(c)]
    return {'rows_here': len(a), 'rows_package1': len(b), 'differences': diffs[:20],
            'passed': len(a) == len(b) and not diffs}


def k8_beltrami(out_dir):
    from experiments.run_beltrami_pinned import pinned_config
    from lilq.iteration_log import IterationLogger
    from problems.beltrami import solve_beltrami
    np.random.seed(42)                            # as run_beltrami_pinned (NumPy only; the solve does not draw)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    solve_beltrami(pinned_config(), verbose=False, iteration_logger=logger, run_json_path=out_dir / 'run.json')
    logger.to_csv(out_dir / 'iterations.csv')
    return out_dir / 'iterations.csv'


def k8_bl(out_dir):
    from experiments.run_bl import paper_setup
    from lilq.iteration_log import IterationLogger
    from problems.buckley_leverett import run_lil_q
    config, opt = paper_setup(24, False)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    run_lil_q(config, opt, verbose=False, iteration_logger=logger, run_json_path=out_dir / 'run.json')
    logger.to_csv(out_dir / 'iterations.csv')
    return out_dir / 'iterations.csv'


def k8(out_root, package1):
    root = Path(out_root) / 'P3_checks' / 'k8'
    B = Path(package1) / 'B_instrumentation'
    res = {}
    for name, fn, ref in (('beltrami_pinned', k8_beltrami, B / 'beltrami_pinned' / 'iterations.csv'),
                          ('bl_P576_cpu_paper', k8_bl, B / 'bl_P576_cpu_paper' / 'iterations.csv')):
        t0 = time.perf_counter()
        here = fn(root / name)
        res[name] = {**compare_with_package1(here, ref), 'seconds': time.perf_counter() - t0, 'package1': str(ref)}
    return {'check': 'K8', 'runs': res, 'passed': all(r['passed'] for r in res.values()),
            'threads': capture_blas_thread_env().get('OMP_NUM_THREADS'), 'commit': current_commit()}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 3, checks K0 and K8.')
    ap.add_argument('check', choices=('k0', 'k8'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--package1')
    ap.add_argument('--m', type=int, default=K0_SHAPE[0])
    ap.add_argument('--n', type=int, default=K0_SHAPE[1])
    args = ap.parse_args(argv)
    out = Path(args.out) / 'P3_checks'
    out.mkdir(parents=True, exist_ok=True)
    result = k0(args.m, args.n) if args.check == 'k0' else k8(args.out, args.package1)
    (out / f'{args.check}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=1)[:1500])
    if args.check == 'k8':
        for name, r in result['runs'].items():
            print(f"  {name}: {'identical' if r['passed'] else 'DIFFERS'} ({r['rows_here']} rows; package1 {r['rows_package1']})")
    print(f"{args.check.upper()} {'passed' if result['passed'] else 'FAILED'}")
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
