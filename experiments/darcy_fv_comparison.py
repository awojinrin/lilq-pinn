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
import json
import csv
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from lilq.blas_threads import pin_torch  # first: sets the BLAS threads before numpy/scipy load
import numpy as np
import torch

from lilq.provenance import save_provenance
from lilq.saved_models import save_solution
from lilq.source_lock import current_commit
from problems.darcy import (
    DarcyConfig, DarcyPhysics, delta_fv, run_nil_n_darcy, solve_fvm, solve_lilq_darcy, tpfa_residual,
)

pin_torch()   # PyTorch's threads = the BLAS allocation

FIELDS = ('S1', 'S2', 'S3', 'SPE10')
ORDER = 32  # the paper's (experiments/run_darcy.py DEFAULT_ORDER)
DATA_DIR = Path(_proj) / 'data' / 'spe10'
OUTPUT_DIR = Path(_proj) / 'results' / 'darcy_fv_comparison'

# allocation and resumed_at record what each time ran on (the advisor's
# follow-up of 1 October 2026, item 4: the paper quotes the NiL times).
COLUMNS = ('field', 'method', 'seed', 'delta_fv', 'rel_l2_vs_abs_pressure', 'max_abs_err_psi',
           'tpfa_residual', 'tpfa_residual_rel', 'time_s', 'final_loss', 'dtype', 'n_params',
           'allocation', 'resumed_at', 'commit')


def _config(field, order):
    return DarcyConfig(ORDER_H=order, ORDER_U=order, ORDER_V=order, perm_file=f'perm_field_{field}.txt')


def allocation_label(scheduler=None, gpu_name=None):
    """``'<cores> cores + <n> x <GPU>'`` (or ``'<cores> cores'``) of the job a
    time was measured in, then how it held its node: ``', exclusive'``
    (``--exclusive``), ``', whole node'`` (every core and all the memory,
    the ``timed`` class) or ``', shared'`` (other jobs could run on the node:
    the advisor's reply on wave 3, item 2.6). From a ``hardware.json``
    ``scheduler`` record, or this process's environment. ``'local'`` outside
    a scheduler."""
    if scheduler is None:
        from lilq.provenance import capture_scheduler_info
        scheduler = capture_scheduler_info()
    if not scheduler or not scheduler.get('slurm'):
        return 'local'
    cores = (scheduler.get('job') or {}).get('NumCPUs') or scheduler.get('SLURM_CPUS_PER_TASK') \
        or scheduler.get('SLURM_CPUS_ON_NODE')
    gpus = [g for g in str(scheduler.get('SLURM_JOB_GPUS') or '').split(',') if g != '']
    if gpu_name is None and gpus:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU'
    label = f"{cores} cores" + (f" + {len(gpus)} x {gpu_name}" if gpus else '')
    if scheduler.get('exclusive'):
        return label + ', exclusive'
    return label + (', whole node' if scheduler.get('holds_whole_node') else ', shared')


def _row(field, method, seed, P_h, P_fvm, p_bot, tpfa, time_s, final_loss, dtype, n_params=None,
         allocation=None, resumed_at=None):
    return {
        'field': field, 'method': method, 'seed': seed,
        'delta_fv': delta_fv(P_h, P_fvm, p_bot),
        'rel_l2_vs_abs_pressure': float(np.linalg.norm(P_h - P_fvm) / np.linalg.norm(P_fvm)),
        'max_abs_err_psi': float(np.abs(P_h - P_fvm).max()),
        **tpfa, 'time_s': time_s, 'final_loss': final_loss, 'dtype': dtype,
        'n_params': n_params, 'allocation': allocation,
        'resumed_at': json.dumps(resumed_at) if resumed_at is not None else '', 'commit': current_commit(),
    }


def compare_field(field, order=ORDER, nil_seeds=(), nil_epochs=150000, verbose=True, model_root=None,
                  nil_dtypes=('float64', 'float32')):
    """Rows for one field: LiL, then NiL once per seed in ``nil_seeds`` and
    per precision in ``nil_dtypes``. The NiL network is the manuscript's
    (Table 13: three networks of 2 hidden layers x 32, 3,555 parameters;
    ``problems.darcy`` defaults). float64 is the result (v2.0 Section 2);
    float32, the manuscript's precision, gives the value Addendum v2.2's 20%
    rule compares with (report both where delta_FV differs by more than 20%).

    With ``model_root``, the LiL solution is saved in ``<model_root>/LiL_<field>/``
    and each NiL run keeps ``<model_root>/NiL_<field>_s<seed>_<dtype>/``: a checkpoint
    every 5,000 epochs while it trains (a rerun resumes from it), then the
    trained networks (``network.pt``, ``problems.darcy.load_darcy_pinn``),
    which a rerun loads instead of training again."""
    config = _config(field, order)
    physics = DarcyPhysics(config, verbose=False)
    P_fvm = solve_fvm(physics)
    tpfa = tpfa_residual(physics, P_fvm)
    p_bot = config.P_BOTTOM

    t0 = time.perf_counter()
    lil = solve_lilq_darcy(config, physics, verbose=False)
    if model_root is not None:
        save_solution(Path(model_root) / f'LiL_{field}',
                      {'h_tilde': (lil['basis_h_tilde'], lil['c_h_tilde']),
                       'u': (lil['basis_u'], lil['c_u']), 'v': (lil['basis_v'], lil['c_v'])},
                      config, extra={'P_lil': lil['P_lil'], 'P_fvm': P_fvm})
    n_lil = sum(len(lil[k]) for k in ('c_h_tilde', 'c_u', 'c_v'))
    allocation = allocation_label()
    rows = [_row(field, 'LiL', '', lil['P_lil'], P_fvm, p_bot, tpfa,
                 time.perf_counter() - t0, '', 'float64', n_lil, allocation=allocation)]
    if verbose:
        print(f"  {field} LiL: delta_FV = {rows[-1]['delta_fv']:.3e}  "
              f"(TPFA residual {tpfa['tpfa_residual_rel']:.1e} relative)", flush=True)

    for seed in nil_seeds:
        for dtype in nil_dtypes:
            nil = run_nil_n_darcy(config, physics, max_epochs=nil_epochs, seed=seed, verbose=False,
                                  dtype=getattr(torch, dtype),
                                  model_dir=Path(model_root) / f'NiL_{field}_s{seed}_{dtype}' if model_root else None)
            pinn = nil['pinn']
            n_nil = sum(q.numel() for net in (pinn.net_P, pinn.net_U, pinn.net_V) for q in net.parameters())
            rows.append(_row(field, 'NiL', seed, np.asarray(nil['fields']['P'], dtype=np.float64), P_fvm, p_bot,
                             tpfa, nil['training_time'], nil['final_loss'], dtype, n_nil,
                             allocation=allocation, resumed_at=nil.get('resumed_at', [])))
            if verbose:
                print(f"  {field} NiL seed {seed} {dtype}: delta_FV = {rows[-1]['delta_fv']:.3e}  "
                      f"({nil['training_time']:.0f} s)", flush=True)
    return rows


def annotate(darcy_fv_root):
    """Fill ``allocation`` and ``resumed_at`` in every job's table under
    ``darcy_fv_root`` (``<root>/<field>_s<seed>/darcy_fv_comparison.csv``)
    where they are empty -- tables written before these columns existed
    (wave 3): the allocation from the job's ``hardware.json``, the resumes
    from each NiL run's ``models/NiL_<field>_s<seed>_<dtype>/network.pt``.
    Returns the number of rows filled."""
    filled = 0
    for table in sorted(Path(darcy_fv_root).glob('*/darcy_fv_comparison.csv')):
        job = table.parent
        with open(table, newline='') as f:
            rows = list(csv.DictReader(f))
        hw = job / 'hardware.json'
        hardware = json.loads(hw.read_text()) if hw.exists() else {}
        gpu = ((hardware.get('gpu') or {}).get('gpus') or [{}])[0].get('name')
        label = allocation_label(hardware.get('scheduler') or {}, gpu) if hardware else ''
        for r in rows:
            if not r.get('allocation'):
                r['allocation'] = label
                filled += 1
            if r['method'] == 'NiL' and not r.get('resumed_at'):
                net = job / 'models' / f"NiL_{r['field']}_s{r['seed']}_{r['dtype']}" / 'network.pt'
                if net.exists():
                    saved = torch.load(net, map_location='cpu', weights_only=False)
                    r['resumed_at'] = json.dumps(saved.get('resumed_at', []))
        # A new file swapped in, never the table rewritten in place: in place,
        # a hard-linked copy would change its original too (the advisor's
        # reply on wave 3, Section 1, item 2).
        tmp = table.with_name(table.name + '.tmp')
        with open(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows([{c: r.get(c, '') for c in COLUMNS} for r in rows])
        os.replace(tmp, table)
    return filled


def main():
    parser = argparse.ArgumentParser(description="Task B9: Darcy pressures vs. the FVM solution")
    parser.add_argument('--fields', nargs='+', default=list(FIELDS))
    parser.add_argument('--order', type=int, default=ORDER)
    parser.add_argument('--nil', action='store_true', help='Also train NiL (expensive).')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--nil-epochs', type=int, default=150000)
    parser.add_argument('--nil-dtypes', nargs='+', default=['float64', 'float32'], choices=['float64', 'float32'])
    parser.add_argument('--out-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--annotate', type=str, default=None,
                        help='Instead of running: fill allocation and resumed_at in the job tables under '
                             'this darcy_fv folder (wave 3).')
    args = parser.parse_args()
    if args.annotate:
        print(f"Filled {annotate(args.annotate)} rows under {args.annotate}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    rows = []
    for field in args.fields:
        if not (DATA_DIR / f'perm_field_{field}.txt').exists():
            print(f"  [SKIP] perm_field_{field}.txt not found in {DATA_DIR}")
            continue
        rows += compare_field(field, args.order, args.seeds if args.nil else (), args.nil_epochs,
                              model_root=out_dir / 'models', nil_dtypes=args.nil_dtypes)
        with open(out_dir / 'darcy_fv_comparison.csv', 'w', newline='') as f:  # after every field
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    save_provenance(out_dir)
    print(f"Wrote {out_dir / 'darcy_fv_comparison.csv'} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
