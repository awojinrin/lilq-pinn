"""
Darcy / SPE10 — Experiment Runner
====================================

FVM (reference) + LiL-Q + optional NiL-N (PINN). Saves checkpoints and plots.

Usage::

    python experiments/run_darcy.py
    python experiments/run_darcy.py --fields S1 S2 S3 SPE10
    python experiments/run_darcy.py --order 32 --skip-pinn
"""

import sys, os, time, argparse
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np

from lilq.utils import set_seed
from problems.darcy import (
    DarcyConfig, DarcyPhysics,
    solve_lilq_darcy, evaluate_fields, run_nil_n_darcy,
)

from experiments.exp_utils import (
    make_experiment_dir, make_figures_dir, experiment_base_dir,
    save_master_results, save_summary_json, plot_solution_field,
)

DEFAULT_FIELDS = ['S3']
DEFAULT_ORDER = 32


def run_for_field(field_name, order, run_pinn, pinn_epochs, verbose=True):
    perm_file = f'perm_field_{field_name}.txt'
    config = DarcyConfig(ORDER_H=order, ORDER_U=order, ORDER_V=order,
                         perm_file=perm_file)

    from pathlib import Path as _P
    data_dir = _P(__file__).resolve().parent.parent / 'data' / 'spe10'
    if not (data_dir / perm_file).exists():
        print(f"  [SKIP] {perm_file} not found in {data_dir}")
        return {}

    physics = DarcyPhysics(config, verbose=verbose)

    n_dir = make_experiment_dir('darcy', f'order{order}', field_name)
    fig_dir = make_figures_dir('darcy', f'order{order}')
    results = {'field': field_name, 'order': order}

    # ── LiL-Q (includes FVM reference) ──
    if verbose:
        print(f"\n{'='*60}\nDARCY {field_name} — LiL-Q (order={order})\n{'='*60}")

    lilq_result = solve_lilq_darcy(config, physics, verbose=verbose)
    m = lilq_result['metrics']

    results['LiL-Q'] = {
        'method': 'LiL-Q',
        'solve_time': m['solve_time'],
        'training_time': m['total_time'],
        'final_loss': m['residual_norm'],
        'fvm_rmse_psi': m['fvm_rmse_psi'],
        'fvm_rel_L2': m['fvm_rel_L2'],
        'n_coefficients': m['n_coefficients'],
        'mse_Dx_nd': m['mse_Dx_nd'],
        'mse_Dy_nd': m['mse_Dy_nd'],
        'mse_CE_nd': m['mse_CE_nd'],
        'converged': True,
    }

    save_summary_json(n_dir / 'lil_q_summary.json', results['LiL-Q'])
    np.savez(n_dir / 'lil_q_checkpoint.npz',
             c_h_tilde=lilq_result['c_h_tilde'],
             c_u=lilq_result['c_u'], c_v=lilq_result['c_v'])

    # Plot pressure field
    fields = evaluate_fields(lilq_result, physics, n_eval_x=180, n_eval_y=660)
    plot_solution_field(
        fields['X'], fields['Y'], fields['P'],
        f'Darcy LiL-Q Pressure ({field_name})',
        fig_dir, f'darcy_{field_name}_pressure_lilq',
        cmap='jet', xlabel='x (ft)', ylabel='y (ft)', clabel='P (psi)')

    # Velocity fields
    plot_solution_field(
        fields['X'], fields['Y'], fields['ux'],
        f'Darcy LiL-Q x-velocity ({field_name})',
        fig_dir, f'darcy_{field_name}_ux',
        cmap='jet', xlabel='x (ft)', ylabel='y (ft)', clabel='$u_x$')
    plot_solution_field(
        fields['X'], fields['Y'], fields['uy'],
        f'Darcy LiL-Q y-velocity ({field_name})',
        fig_dir, f'darcy_{field_name}_uy',
        cmap='jet', xlabel='x (ft)', ylabel='y (ft)', clabel='$u_y$')

    # Save FVM reference + LiL pressure for comparison
    np.savetxt(n_dir / f'pressure_fvm_{field_name}.txt',
               lilq_result['P_fvm'].ravel(), fmt='%.6e')
    np.savetxt(n_dir / f'pressure_lilq_{field_name}.txt',
               fields['P'].ravel(), fmt='%.6e')

    # ── NiL-N (PINN) ──
    if run_pinn:
        if verbose:
            print(f"\n{'='*60}\nDARCY {field_name} — NiL-N (epochs={pinn_epochs})\n{'='*60}")

        pinn_result = run_nil_n_darcy(config, physics,
                                      max_epochs=pinn_epochs, verbose=verbose)
        results['NiL-N'] = {
            'method': 'NiL-N',
            'training_time': pinn_result['training_time'],
            'final_loss': pinn_result['final_loss'],
            'converged': True,
        }
        save_summary_json(n_dir / 'nil_n_summary.json', results['NiL-N'])

    return results


def main():
    parser = argparse.ArgumentParser(description="Darcy/SPE10 Experiments")
    parser.add_argument('--fields', type=str, nargs='+', default=DEFAULT_FIELDS)
    parser.add_argument('--order', type=int, default=DEFAULT_ORDER)
    parser.add_argument('--skip-pinn', action='store_true')
    parser.add_argument('--pinn-epochs', type=int, default=150000)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    set_seed(42)
    verbose = not args.quiet
    print(f"Darcy Experiments | order={args.order}, fields={args.fields}")

    all_results = {}
    t0 = time.time()

    for fname in args.fields:
        result = run_for_field(fname, args.order, not args.skip_pinn,
                               args.pinn_epochs, verbose)
        if result:
            all_results[fname] = result

    # Summary
    print(f"\n{'='*60}\nDARCY SUMMARY\n{'='*60}")
    for fname, r in all_results.items():
        print(f"\n  {fname}:")
        if 'LiL-Q' in r:
            lq = r['LiL-Q']
            print(f"    LiL-Q: {lq['n_coefficients']} DOFs, "
                  f"time={lq['training_time']:.3f}s, "
                  f"FVM rel_L2={lq['fvm_rel_L2']:.4e}")
        if 'NiL-N' in r:
            pn = r['NiL-N']
            print(f"    NiL-N: time={pn['training_time']:.1f}s, "
                  f"final_loss={pn['final_loss']:.4e}")

    print(f"\nTotal: {time.time()-t0:.1f}s")

    base_dir = experiment_base_dir('darcy', f'order{args.order}')
    save_master_results(base_dir / 'darcy_master_results.json', all_results,
                        {'fields': args.fields, 'order': args.order})


if __name__ == '__main__':
    main()
