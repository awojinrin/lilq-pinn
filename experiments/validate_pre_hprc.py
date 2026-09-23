"""
Pre-HPRC Validation Run
==========================

A comprehensive, small-scale sanity sweep across every problem in this
codebase, run locally before committing to the full experiment matrix on
the HPRC cluster. This is Phase 0 validation tooling -- not a
Computational_Package_1_v2.md deliverable -- and asks three questions:

    1. Does everything run to completion without crashing or producing
       NaN/Inf?
    2. Do the results look scientifically reasonable (converges, or at
       least moves in the right direction; losses/errors in a sane
       range)?
    3. How long does each size actually take, to project full-scale
       HPRC runtime and catch any efficiency red flags before spending
       cluster time on them?

Scope, deliberately not full-paper-scale:

- Bratu/Burgers/BL (viscous + gravity) -- the four-method (NiL-N, NiL-Q,
  LiL-N, LiL-Q) nonconvex problems: the two *smallest* paper sizes only,
  at the real paper budgets (iteration caps, R_tol) imported directly
  from each problem's own experiments/run_*.py -- reduced scale, not
  reduced rigor.
- Kovasznay, Elasticity: LiL-Q only, direct solves, cheap even at every
  paper size -- run the full N-sweep.
- Darcy: LiL-Q (the paper's ORDER_H/U/V=32, cheap -- a single linear
  solve) run in full; NiL-N (PINN) only smoke-tested at a small epoch
  count (the real 150,000-epoch budget is not a "smaller run").
- Beltrami: one small smoke config (N_vel=4) plus the real paper config
  (N_vel=6, N_p=8, P_total=7,984) -- the latter is included in full
  because its own runtime (~300s, per this project's prior empirical
  findings) is short enough to just run.

No plotting/checkpoint-saving side effects (unlike the run_*.py
scripts) -- this calls each problem's run_nil_n/run_nil_q/run_lil_n/
run_lil_q/solve_* functions directly, so wall-clock time here is spent
on the actual solves, not on Beltrami's 3D cube rendering or similar.

Usage::

    python experiments/validate_pre_hprc.py
    python experiments/validate_pre_hprc.py --skip beltrami darcy_pinn
"""

import sys
import os
import argparse
import json
import time
import traceback
import math
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from lilq.utils import set_seed, clear_gpu_memory, DEVICE

RESULTS_PATH = Path(__file__).resolve().parent.parent / 'results' / 'pre_hprc_validation.json'


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────

def _finite_and_sane(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True  # non-numeric fields (strings, None) aren't a sanity concern here


def _run_one(problem, method, size_label, fn, *args, **kwargs):
    """Run one solve, catching exceptions, timing it, and extracting a
    minimal sanity check (finite final_loss, finite rel_l2 errors if
    present) from whatever summary/dict the function returns. Never
    raises -- a failed run is a *result*, the whole point of this script.
    """
    print(f"\n{'-' * 70}\n{problem} | {method} | {size_label}\n{'-' * 70}")
    set_seed(42)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        summary = _extract_summary(result)
        final_loss = summary.get('final_loss')
        rel_l2_fields = {k: v for k, v in summary.items()
                         if k.startswith('rel_l2') and isinstance(v, (int, float))}
        numeric_checks = [final_loss] + list(rel_l2_fields.values())
        sane = all(_finite_and_sane(v) for v in numeric_checks if v is not None)
        record = {
            'problem': problem, 'method': method, 'size': size_label,
            'status': 'ok', 'elapsed_s': elapsed,
            'final_loss': final_loss,
            'converged': summary.get('converged'),
            'iterations': summary.get('total_iterations') or summary.get('n_outer_iters'),
            'rel_l2': rel_l2_fields,
            'finite_loss': sane,
        }
        status_str = "OK" if sane else "** NaN/Inf **"
        rel_l2_str = (' rel_l2=' + ', '.join(f'{k.replace("rel_l2_","")}={v:.3e}'
                                             for k, v in rel_l2_fields.items())) if rel_l2_fields else ''
        print(f"  -> {status_str}  loss={final_loss}  "
              f"converged={summary.get('converged')}  time={elapsed:.2f}s{rel_l2_str}")
        return record
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  -> ** FAILED ** ({type(e).__name__}: {e})  time={elapsed:.2f}s")
        traceback.print_exc()
        return {
            'problem': problem, 'method': method, 'size': size_label,
            'status': 'FAILED', 'elapsed_s': elapsed,
            'error': f"{type(e).__name__}: {e}",
        }
    finally:
        clear_gpu_memory()


def _extract_summary(result):
    """Every run_*/solve_* function returns either a plain dict
    (LiL-Q-only problems) or a tuple ending in a summary dict (NiL-N/
    NiL-Q/LiL-N/LiL-Q, per lilq.multiseed's own documented convention)."""
    if isinstance(result, dict):
        return result
    return result[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Group A: Bratu / Burgers / BL -- four methods, two smallest paper sizes
# ─────────────────────────────────────────────────────────────────────────────

def validate_bratu(records, n_sizes=2):
    from problems.bratu import (
        BratuConfig, BratuOptConfig, run_nil_n, run_nil_q, run_lil_n, run_lil_q,
    )
    from experiments.run_bratu import (
        DEFAULT_N_VALUES, DEFAULT_LAMBDA, DEFAULT_BASIS, DEFAULT_K_RATIO,
        TARGET_LOSSES, MAX_ITERATIONS, MAX_LINE_SEARCHES,
        MAX_QUASI_ITERS, MAX_LBFGS_PER_QUASI_ITER,
    )
    runners = {'NiL-N': run_nil_n, 'NiL-Q': run_nil_q, 'LiL-N': run_lil_n, 'LiL-Q': run_lil_q}
    for N in DEFAULT_N_VALUES[:n_sizes]:
        config = BratuConfig(lambda_=DEFAULT_LAMBDA, N_x=N, N_y=N,
                             k_ratio=DEFAULT_K_RATIO, basis_type=DEFAULT_BASIS)
        opt = BratuOptConfig(
            max_iterations=MAX_ITERATIONS.get(N, 10000),
            max_line_searches=MAX_LINE_SEARCHES.get(N, 30000),
            R_tol=TARGET_LOSSES.get(N, 1e-4),
            max_quasi_iters_nn=MAX_QUASI_ITERS,
            max_inner_iters_nn=MAX_LBFGS_PER_QUASI_ITER.get(N, 300),
        )
        for method, runner in runners.items():
            kwargs = dict(device=DEVICE, verbose=False) if method != 'LiL-Q' else dict(verbose=False)
            records.append(_run_one('bratu', method, f'N={N} (P={N*N})', runner, config, opt, **kwargs))


def validate_burgers(records, n_sizes=2):
    from problems.burgers import (
        BurgersConfig, BurgersOptConfig, run_nil_n, run_nil_q, run_lil_n, run_lil_q,
    )
    from experiments.run_burgers import (
        DEFAULT_N_VALUES, DEFAULT_BASIS, VISCOSITY, T_FINAL, K_RATIO,
        TARGET_LOSSES, MAX_LBFGS_ITERS, MAX_LINE_SEARCHES,
        MAX_QUASI_ITERS, MAX_LBFGS_PER_QUASI,
    )
    runners = {'NiL-N': run_nil_n, 'NiL-Q': run_nil_q, 'LiL-N': run_lil_n, 'LiL-Q': run_lil_q}
    for N in DEFAULT_N_VALUES[:n_sizes]:
        config = BurgersConfig(N_x=N, N_t=N, viscosity=VISCOSITY, T_final=T_FINAL,
                               basis_type=DEFAULT_BASIS, k_ratio=K_RATIO)
        opt = BurgersOptConfig(
            max_iterations=MAX_LBFGS_ITERS.get(N, 10000),
            max_line_searches=MAX_LINE_SEARCHES.get(N, 100000),
            R_tol=TARGET_LOSSES.get(N, 1e-4),
            max_quasi_iters_nn=MAX_QUASI_ITERS,
            max_inner_iters_nn=MAX_LBFGS_PER_QUASI.get(N, 300),
        )
        for method, runner in runners.items():
            kwargs = dict(device=DEVICE, verbose=False) if method != 'LiL-Q' else dict(verbose=False)
            records.append(_run_one('burgers', method, f'N={N} (P={N*N})', runner, config, opt, **kwargs))


def validate_bl(records, gravity, n_sizes=2):
    import dataclasses
    from problems.buckley_leverett import (
        BLConfig, BLOptConfig, run_nil_n, run_nil_q, run_lil_n, run_lil_q,
    )
    from experiments.run_bl import (
        DEFAULT_N_VALUES, DEFAULT_BASIS, GRAVITY_BASIS, K_RATIO,
        TARGET_LOSSES, MAX_LBFGS_ITERS, MAX_LBFGS_PER_QUASI, MAX_QUASI_ITERS,
        GRAVITY_TARGET_LOSSES, GRAVITY_MAX_QUASI_ITERS, GRAVITY_MAX_LBFGS_PER_QUASI,
    )
    runners = {'NiL-N': run_nil_n, 'NiL-Q': run_nil_q, 'LiL-N': run_lil_n, 'LiL-Q': run_lil_q}
    label = 'bl_gravity' if gravity else 'bl'
    targets = GRAVITY_TARGET_LOSSES if gravity else TARGET_LOSSES
    quasi_iters = GRAVITY_MAX_QUASI_ITERS if gravity else MAX_QUASI_ITERS
    inner_per_quasi = GRAVITY_MAX_LBFGS_PER_QUASI if gravity else MAX_LBFGS_PER_QUASI
    basis_type = GRAVITY_BASIS if gravity else DEFAULT_BASIS
    for N in DEFAULT_N_VALUES[:n_sizes]:
        base = BLConfig.with_gravity() if gravity else BLConfig()
        config = dataclasses.replace(base, N_x=N, N_t=N, basis_type=basis_type, k_ratio=K_RATIO)
        # max_line_searches left unset -- BLOptConfig derives it (see
        # tests/test_bl_experiment_runner_config.py / DECISIONS.md).
        opt = BLOptConfig(
            max_iterations=MAX_LBFGS_ITERS.get(N, 10000),
            R_tol=targets.get(N, 1e-3),
            max_quasi_iters_nn=quasi_iters,
            max_inner_iters_nn=inner_per_quasi.get(N, 200),
        )
        for method, runner in runners.items():
            kwargs = dict(device=DEVICE, verbose=False) if method != 'LiL-Q' else dict(verbose=False)
            records.append(_run_one(label, method, f'N={N} (P={N*N})', runner, config, opt, **kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# Group B: LiL-Q-only, cheap -- full sweeps
# ─────────────────────────────────────────────────────────────────────────────

def validate_kovasznay(records):
    from problems.kovasznay import KovasznayConfig, solve_kovasznay
    from experiments.run_kovasznay import DEFAULT_N_VALUES, DEFAULT_RE, DEFAULT_BASIS, K_RATIO, MAX_ITER, TOL
    for N in DEFAULT_N_VALUES:
        config = KovasznayConfig(N_x=N, N_y=N, Re=DEFAULT_RE, basis_type=DEFAULT_BASIS,
                                 k_ratio=K_RATIO, max_iter=MAX_ITER, tol=TOL)

        def _wrap(config=config):
            r = solve_kovasznay(config, verbose=False)
            # Matches experiments/run_kovasznay.py's own summary convention.
            return {'final_loss': r['pde_mse'], 'converged': True,
                   'n_outer_iters': r['n_outer_iters'],
                   'rel_l2_u': r['rel_l2_u'], 'rel_l2_v': r['rel_l2_v'], 'rel_l2_p': r['rel_l2_p']}

        records.append(_run_one('kovasznay', 'LiL-Q', f'N={N} (P={3*N*N})', _wrap))


def validate_elasticity(records):
    from problems.elasticity import ElasticityConfig, solve_elasticity
    from experiments.run_elasticity import DEFAULT_N_VALUES, K_RATIO
    for N in DEFAULT_N_VALUES:
        config = ElasticityConfig(N_x=N, N_y=N, k_ratio=K_RATIO)

        def _wrap(config=config):
            r = solve_elasticity(config, verbose=False)
            return {'final_loss': r['pde_mse'], 'converged': True,
                   'rel_l2_ux': r['rel_l2_ux'], 'rel_l2_uy': r['rel_l2_uy']}

        records.append(_run_one('elasticity', 'LiL-Q', f'N={N} (P~{2*N*N})', _wrap))


def validate_darcy(records, run_pinn=True, pinn_epochs=1000):
    from problems.darcy import DarcyConfig, DarcyPhysics, solve_lilq_darcy, run_nil_n_darcy
    from experiments.run_darcy import DEFAULT_FIELDS, DEFAULT_ORDER

    for field_name in DEFAULT_FIELDS:
        perm_file = f'perm_field_{field_name}.txt'
        data_dir = Path(__file__).resolve().parent.parent / 'data' / 'spe10'
        if not (data_dir / perm_file).exists():
            print(f"  [SKIP] Darcy {field_name}: {perm_file} not found")
            continue

        config = DarcyConfig(ORDER_H=DEFAULT_ORDER, ORDER_U=DEFAULT_ORDER,
                             ORDER_V=DEFAULT_ORDER, perm_file=perm_file)
        physics = DarcyPhysics(config, verbose=False)

        def _lilq_wrap(config=config, physics=physics, verbose=False):
            r = solve_lilq_darcy(config, physics, verbose=verbose)
            return {'final_loss': r['metrics']['residual_norm'],
                   'rel_l2_fvm': r['metrics']['fvm_rel_L2'], 'converged': True}

        records.append(_run_one('darcy', 'LiL-Q', f'{field_name} (order={DEFAULT_ORDER})', _lilq_wrap))

        if run_pinn:
            def _pinn_wrap(config=config, physics=physics):
                r = run_nil_n_darcy(config, physics, max_epochs=pinn_epochs, verbose=False)
                return {'final_loss': r['final_loss'], 'converged': True}

            records.append(_run_one('darcy_pinn', 'NiL-N',
                                   f'{field_name} (epochs={pinn_epochs}, SMOKE-TEST budget)', _pinn_wrap))


def _beltrami_wrap(config):
    from problems.beltrami import solve_beltrami
    r = solve_beltrami(config, verbose=False)
    return {'final_loss': r['pde_mse'], 'converged': True,
           'n_outer_iters': r['n_outer_iters'],
           'rel_l2_u': r.get('rel_l2_u', float('nan')), 'rel_l2_v': r.get('rel_l2_v', float('nan')),
           'rel_l2_w': r.get('rel_l2_w', float('nan')), 'rel_l2_p': r.get('rel_l2_p', float('nan'))}


def validate_beltrami(records, include_full=True):
    from problems.beltrami import BeltramiConfig
    from experiments.run_beltrami import COLLOC, DEFAULT_BASIS

    # Small smoke config.
    cs = COLLOC[4]
    small_config = BeltramiConfig(N_vel=4, N_p=5, basis_type=DEFAULT_BASIS, **cs)
    records.append(_run_one('beltrami', 'LiL-Q', 'N_vel=4,N_p=5 (smoke)',
                           _beltrami_wrap, small_config))

    if include_full:
        cs6 = COLLOC[6]
        full_config = BeltramiConfig(N_vel=6, N_p=8, basis_type=DEFAULT_BASIS, **cs6)
        records.append(_run_one('beltrami', 'LiL-Q', 'N_vel=6,N_p=8 (paper P=7984)',
                               _beltrami_wrap, full_config))


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(records):
    print(f"\n\n{'=' * 78}\nPRE-HPRC VALIDATION SUMMARY\n{'=' * 78}")
    print(f"{'Problem':<14} {'Method':<8} {'Size':<28} {'Status':<8} {'Time(s)':>8}  Loss")
    print('-' * 78)
    failures = []
    nan_flags = []
    for r in records:
        if r['status'] == 'FAILED':
            failures.append(r)
            print(f"{r['problem']:<14} {r['method']:<8} {r['size']:<28} "
                 f"{'FAILED':<8} {r['elapsed_s']:>8.2f}  {r['error']}")
        else:
            loss_str = f"{r['final_loss']:.4e}" if isinstance(r['final_loss'], (int, float)) else str(r['final_loss'])
            flag = ''
            if r.get('finite_loss') is False:
                flag = '  ** NaN/Inf **'
                nan_flags.append(r)
            print(f"{r['problem']:<14} {r['method']:<8} {r['size']:<28} "
                 f"{'ok':<8} {r['elapsed_s']:>8.2f}  {loss_str}{flag}")

    print('-' * 78)
    total_time = sum(r['elapsed_s'] for r in records)
    print(f"Total: {len(records)} runs, {total_time:.1f}s wall-clock, "
         f"{len(failures)} failed, {len(nan_flags)} NaN/Inf")
    if failures:
        print("\nFAILURES:")
        for r in failures:
            print(f"  - {r['problem']}/{r['method']}/{r['size']}: {r['error']}")
    if nan_flags:
        print("\nNaN/Inf FINAL LOSS:")
        for r in nan_flags:
            print(f"  - {r['problem']}/{r['method']}/{r['size']}")
    if not failures and not nan_flags:
        print("\nNo failures, no NaN/Inf -- clean pass.")


def main():
    parser = argparse.ArgumentParser(description="Pre-HPRC validation sweep")
    parser.add_argument('--skip', nargs='+', default=[],
                        choices=['bratu', 'burgers', 'bl', 'bl_gravity', 'kovasznay',
                                'elasticity', 'darcy', 'darcy_pinn', 'beltrami', 'beltrami_full'])
    parser.add_argument('--n-sizes', type=int, default=2,
                        help='Number of smallest paper sizes to test for the 4-method problems')
    args = parser.parse_args()

    records = []
    t_start = time.time()

    if 'bratu' not in args.skip:
        validate_bratu(records, n_sizes=args.n_sizes)
    if 'burgers' not in args.skip:
        validate_burgers(records, n_sizes=args.n_sizes)
    if 'bl' not in args.skip:
        validate_bl(records, gravity=False, n_sizes=args.n_sizes)
    if 'bl_gravity' not in args.skip:
        validate_bl(records, gravity=True, n_sizes=args.n_sizes)
    if 'kovasznay' not in args.skip:
        validate_kovasznay(records)
    if 'elasticity' not in args.skip:
        validate_elasticity(records)
    if 'darcy' not in args.skip:
        validate_darcy(records, run_pinn='darcy_pinn' not in args.skip)
    if 'beltrami' not in args.skip:
        validate_beltrami(records, include_full='beltrami_full' not in args.skip)

    print_report(records)
    print(f"\nGrand total wall-clock: {time.time() - t_start:.1f}s")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == '__main__':
    main()
