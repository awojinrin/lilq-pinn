"""
Darcy: LiL and NiL pressures against the finite-volume solution (task B9)
=========================================================================

Computational Package 1, Addendum v2.1, task B9. For each field (S1, S2,
S3, SPE10) and for LiL and NiL, on the native 60 x 220 cell centres::

    delta_FV = ||p_h - p_FV||_2 / ||p_FV - p_bot||_2,   p_bot = 3,000 psi

and, for the record, the residual of the TPFA linear system at the FVM
solution (a direct solve: it should be at round-off). This replaces the
withdrawn FVM "continuity" entries of Table 14; no np.gradient divergence
is computed. ``fvm_rel_L2`` (normalized by ||p_FV||, which the 3,000 psi
offset makes 2-4x smaller) is kept in the CSV only for comparison.

NiL (``DarcyPINN``, float64) runs one row per seed; it is expensive
(150,000 Adam epochs by default), so it only runs with ``--nil``.

Usage::

    python experiments/darcy_fv_comparison.py                    # LiL, all four fields
    python experiments/darcy_fv_comparison.py --nil --seeds 0 1 2
    python experiments/darcy_fv_comparison.py --fields S1 --nil --nil-epochs 200   # smoke
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from lilq.provenance import save_provenance
from problems.darcy import (
    DarcyConfig, DarcyPhysics, delta_fv, run_nil_n_darcy, solve_fvm, solve_lilq_darcy, tpfa_residual,
)

FIELDS = ('S1', 'S2', 'S3', 'SPE10')
ORDER = 32  # the paper's (experiments/run_darcy.py DEFAULT_ORDER)
DATA_DIR = Path(_proj) / 'data' / 'spe10'
OUTPUT_DIR = Path(_proj) / 'results' / 'darcy_fv_comparison'

COLUMNS = ('field', 'method', 'seed', 'delta_fv', 'rel_l2_vs_abs_pressure', 'max_abs_err_psi',
           'tpfa_residual', 'tpfa_residual_rel', 'time_s', 'final_loss', 'dtype')


def _config(field, order):
    return DarcyConfig(ORDER_H=order, ORDER_U=order, ORDER_V=order, perm_file=f'perm_field_{field}.txt')


def _row(field, method, seed, P_h, P_fvm, p_bot, tpfa, time_s, final_loss, dtype):
    return {
        'field': field, 'method': method, 'seed': seed,
        'delta_fv': delta_fv(P_h, P_fvm, p_bot),
        'rel_l2_vs_abs_pressure': float(np.linalg.norm(P_h - P_fvm) / np.linalg.norm(P_fvm)),
        'max_abs_err_psi': float(np.abs(P_h - P_fvm).max()),
        **tpfa, 'time_s': time_s, 'final_loss': final_loss, 'dtype': dtype,
    }


def compare_field(field, order=ORDER, nil_seeds=(), nil_epochs=150000, verbose=True):
    """Rows for one field: LiL, then NiL once per seed in ``nil_seeds``."""
    config = _config(field, order)
    physics = DarcyPhysics(config, verbose=False)
    P_fvm = solve_fvm(physics)
    tpfa = tpfa_residual(physics, P_fvm)
    p_bot = config.P_BOTTOM

    t0 = time.perf_counter()
    lil = solve_lilq_darcy(config, physics, verbose=False)
    rows = [_row(field, 'LiL', '', lil['P_lil'], P_fvm, p_bot, tpfa,
                 time.perf_counter() - t0, '', 'float64')]
    if verbose:
        print(f"  {field} LiL: delta_FV = {rows[-1]['delta_fv']:.3e}  "
              f"(TPFA residual {tpfa['tpfa_residual_rel']:.1e} relative)", flush=True)

    for seed in nil_seeds:
        nil = run_nil_n_darcy(config, physics, max_epochs=nil_epochs, seed=seed, verbose=False)
        rows.append(_row(field, 'NiL', seed, nil['fields']['P'], P_fvm, p_bot, tpfa,
                         nil['training_time'], nil['final_loss'], 'float64'))
        if verbose:
            print(f"  {field} NiL seed {seed}: delta_FV = {rows[-1]['delta_fv']:.3e}  "
                  f"({nil['training_time']:.0f} s)", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Task B9: Darcy pressures vs. the FVM solution")
    parser.add_argument('--fields', nargs='+', default=list(FIELDS))
    parser.add_argument('--order', type=int, default=ORDER)
    parser.add_argument('--nil', action='store_true', help='Also train NiL (expensive).')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--nil-epochs', type=int, default=150000)
    parser.add_argument('--out-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    rows = []
    for field in args.fields:
        if not (DATA_DIR / f'perm_field_{field}.txt').exists():
            print(f"  [SKIP] perm_field_{field}.txt not found in {DATA_DIR}")
            continue
        rows += compare_field(field, args.order, args.seeds if args.nil else (), args.nil_epochs)
        with open(out_dir / 'darcy_fv_comparison.csv', 'w', newline='') as f:  # after every field
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    save_provenance(out_dir)
    print(f"Wrote {out_dir / 'darcy_fv_comparison.csv'} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
