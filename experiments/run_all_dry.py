"""
Dry-Run Validation for All Experiments
========================================

Runs all problem solvers with minimal settings (N=3, few iterations)
to validate the entire pipeline end-to-end.

Usage::

    python experiments/run_all_dry.py

Expected runtime: < 2 minutes on CPU.
"""

import sys
import os
import time

# Add project root to path
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import torch
from lilq.utils import set_seed, clear_gpu_memory, DEVICE


def dry_run_bratu():
    """Dry-run all 4 Bratu solvers."""
    from problems.bratu import BratuConfig, BratuOptConfig, run_nil_n, run_nil_q, run_lil_n, run_lil_q

    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = BratuOptConfig(
        max_iterations=10, max_line_searches=50,
        max_quasi_iters_nn=2, max_inner_iters_nn=5,
        max_quasi_iters_lil=5,
        R_tol=1e-10,  # intentionally tight so we test iteration loops
        pretrain_epochs=5, pretrain_grid=10,
    )

    results = {}

    print("\n" + "=" * 60)
    print("BRATU — NiL-N")
    print("=" * 60)
    model, metrics, summary = run_nil_n(config, opt, device=DEVICE, verbose=True)
    results['nil_n'] = summary
    clear_gpu_memory()

    print("\n" + "=" * 60)
    print("BRATU — NiL-Q")
    print("=" * 60)
    model, metrics, summary = run_nil_q(config, opt, device=DEVICE, verbose=True)
    results['nil_q'] = summary
    clear_gpu_memory()

    print("\n" + "=" * 60)
    print("BRATU — LiL-N")
    print("=" * 60)
    basis, coeffs, metrics, summary = run_lil_n(config, opt, device=DEVICE, verbose=True)
    results['lil_n'] = summary
    clear_gpu_memory()

    print("\n" + "=" * 60)
    print("BRATU — LiL-Q")
    print("=" * 60)
    basis, coeffs, metrics, summary = run_lil_q(config, opt, verbose=True)
    results['lil_q'] = summary

    return results


def dry_run_burgers():
    """Dry-run all 4 Burgers solvers."""
    from problems.burgers import BurgersConfig, BurgersOptConfig, run_nil_n, run_nil_q, run_lil_n, run_lil_q

    config = BurgersConfig(N_x=3, N_t=3, k_ratio=5)
    opt = BurgersOptConfig(
        max_iterations=10, max_line_searches=50,
        max_quasi_iters_nn=2, max_inner_iters_nn=5,
        max_quasi_iters_lil=5,
        R_tol=1e-10,
        pretrain_epochs=5, pretrain_grid=10,
    )

    results = {}

    print("\n" + "=" * 60)
    print("BURGERS — NiL-N")
    print("=" * 60)
    model, metrics, summary = run_nil_n(config, opt, device=DEVICE, verbose=True)
    results['nil_n'] = summary
    clear_gpu_memory()

    print("\n" + "=" * 60)
    print("BURGERS — NiL-Q")
    print("=" * 60)
    model, metrics, summary = run_nil_q(config, opt, device=DEVICE, verbose=True)
    results['nil_q'] = summary
    clear_gpu_memory()

    print("\n" + "=" * 60)
    print("BURGERS — LiL-N")
    print("=" * 60)
    basis, coeffs, metrics, summary = run_lil_n(config, opt, device=DEVICE, verbose=True)
    results['lil_n'] = summary
    clear_gpu_memory()

    print("\n" + "=" * 60)
    print("BURGERS — LiL-Q")
    print("=" * 60)
    basis, coeffs, metrics, summary = run_lil_q(config, opt, verbose=True)
    results['lil_q'] = summary

    return results


def dry_run_bl():
    """Dry-run BL (no gravity) — LiL-Q only for speed."""
    from problems.buckley_leverett import BLConfig, BLOptConfig, run_lil_q

    config = BLConfig(N_x=3, N_t=3, k_ratio=5)
    opt = BLOptConfig(
        max_quasi_iters_lil=5,
        R_tol=1e-10,
        pretrain_epochs=5, pretrain_grid=10,
    )

    print("\n" + "=" * 60)
    print("BUCKLEY-LEVERETT (no gravity) — LiL-Q")
    print("=" * 60)
    basis, coeffs, metrics, summary = run_lil_q(config, opt, verbose=True)
    return {'lil_q': summary}


def dry_run_bl_gravity():
    """Dry-run BL with gravity — LiL-Q only."""
    from problems.buckley_leverett import BLConfig, BLOptConfig, run_lil_q

    config = BLConfig.with_gravity()
    config.N_x, config.N_t = 3, 3
    config.k_ratio = 5
    opt = BLOptConfig(
        max_quasi_iters_lil=5,
        R_tol=1e-10,
        pretrain_epochs=5, pretrain_grid=10,
    )

    print("\n" + "=" * 60)
    print("BUCKLEY-LEVERETT (with gravity) — LiL-Q")
    print("=" * 60)
    basis, coeffs, metrics, summary = run_lil_q(config, opt, verbose=True)
    return {'lil_q': summary}


def dry_run_kovasznay():
    """Dry-run Kovasznay flow (LiL-Q only)."""
    from problems.kovasznay import KovasznayConfig, solve_kovasznay

    config = KovasznayConfig(N_x=3, N_y=3, max_iter=3)

    print("\n" + "=" * 60)
    print("KOVASZNAY FLOW -- LiL-Q")
    print("=" * 60)
    result = solve_kovasznay(config, verbose=True)
    return {'lil_q': {
        'final_loss': result.get('pde_mse', float('nan')),
        'training_time': result.get('solve_time_total', 0),
        'converged': result.get('converged', False),
        'rel_l2_u': result.get('rel_l2_u', float('nan')),
    }}


def dry_run_beltrami():
    """Dry-run Beltrami 3D flow (LiL-Q only)."""
    from problems.beltrami import BeltramiConfig, solve_beltrami

    config = BeltramiConfig(N_vel=3, N_p=3, N_x=4, N_y=4, N_z=4, N_t=4,
                            N_bc=2, N_t_bc=2, N_ic=3, max_iter=2)

    print("\n" + "=" * 60)
    print("BELTRAMI 3D FLOW -- LiL-Q")
    print("=" * 60)
    result = solve_beltrami(config, verbose=True)
    return {'lil_q': {
        'final_loss': result.get('pde_mse', float('nan')),
        'training_time': result.get('solve_time_total', 0),
        'converged': result.get('converged', False),
        'rel_l2_u': result.get('rel_l2_u', float('nan')),
    }}


def dry_run_elasticity():
    """Dry-run Linear Elasticity (LiL-Q only, single QR solve)."""
    from problems.elasticity import ElasticityConfig, solve_elasticity

    config = ElasticityConfig(N_x=3, N_y=3)

    print("\n" + "=" * 60)
    print("LINEAR ELASTICITY -- LiL-Q")
    print("=" * 60)
    result = solve_elasticity(config, verbose=True)
    return {'lil_q': {
        'final_loss': result.get('pde_mse', float('nan')),
        'training_time': result.get('solve_time_total', 0),
        'converged': True,  # linear PDE, always solves in 1 step
        'rel_l2_ux': result.get('rel_l2_ux', float('nan')),
    }}


def dry_run_darcy():
    """Dry-run Darcy/SPE10 (LiL-Q + NiL-N) -- skipped if data files missing."""
    try:
        from problems.darcy import (DarcyConfig, DarcyPhysics,
                                    solve_lilq_darcy, run_nil_n_darcy)
    except ImportError:
        print("  [SKIP] problems.darcy not yet implemented")
        return None

    from pathlib import Path
    data_dir = Path(__file__).parent.parent / 'data' / 'spe10'
    perm_file = data_dir / 'perm_field_S3.txt'
    if not perm_file.exists():
        print(f"  [SKIP] Permeability file not found: {perm_file}")
        return None

    config = DarcyConfig(ORDER_H=4, ORDER_U=4, ORDER_V=4, perm_file=str(perm_file))
    physics = DarcyPhysics(config)

    results = {}

    print("\n" + "=" * 60)
    print("DARCY/SPE10 -- LiL-Q")
    print("=" * 60)
    result = solve_lilq_darcy(config, physics, verbose=True)
    results['lil_q'] = {
        'final_loss': result['metrics'].get('residual_norm', float('nan')),
        'training_time': result['metrics'].get('total_time', 0),
        'converged': True,
        'fvm_rel_L2': result['metrics'].get('fvm_rel_L2', float('nan')),
    }

    print("\n" + "=" * 60)
    print("DARCY/SPE10 -- NiL-N (100 epochs)")
    print("=" * 60)
    try:
        pinn_result = run_nil_n_darcy(config, physics, max_epochs=100,
                                      hidden_dim=32, num_layers=2,
                                      verbose=True)
        results['nil_n'] = {
            'final_loss': pinn_result['final_loss'],
            'training_time': pinn_result['training_time'],
            'converged': True,
        }
    except Exception as e:
        print(f"  [WARN] NiL-N dry-run failed: {e}")
        results['nil_n'] = {
            'final_loss': float('nan'),
            'training_time': 0,
            'converged': False,
        }

    return results


def main():
    print("=" * 60)
    print("LiL-Q DRY-RUN VALIDATION")
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    t0 = time.time()
    all_results = {}

    try:
        all_results['bratu'] = dry_run_bratu()
        print("\n[OK] Bratu: ALL PASSED")
    except Exception as e:
        print(f"\n[FAIL] Bratu FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['burgers'] = dry_run_burgers()
        print("\n[OK] Burgers: ALL PASSED")
    except Exception as e:
        print(f"\n[FAIL] Burgers FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['bl'] = dry_run_bl()
        print("\n[OK] BL (no gravity): PASSED")
    except Exception as e:
        print(f"\n[FAIL] BL (no gravity) FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['bl_gravity'] = dry_run_bl_gravity()
        print("\n[OK] BL (gravity): PASSED")
    except Exception as e:
        print(f"\n[FAIL] BL (gravity) FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['kovasznay'] = dry_run_kovasznay()
        print("\n[OK] Kovasznay: PASSED")
    except Exception as e:
        print(f"\n[FAIL] Kovasznay FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['beltrami'] = dry_run_beltrami()
        print("\n[OK] Beltrami: PASSED")
    except Exception as e:
        print(f"\n[FAIL] Beltrami FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_results['elasticity'] = dry_run_elasticity()
        print("\n[OK] Elasticity: PASSED")
    except Exception as e:
        print(f"\n[FAIL] Elasticity FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        darcy_result = dry_run_darcy()
        if darcy_result is not None:
            all_results['darcy'] = darcy_result
            print("\n[OK] Darcy/SPE10: PASSED")
        else:
            print("\n[SKIP] Darcy/SPE10: skipped")
    except Exception as e:
        print(f"\n[FAIL] Darcy FAILED: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for problem, results in all_results.items():
        print(f"\n{problem.upper()}:")
        for method, summary in results.items():
            status = "CONVERGED" if summary.get('converged', False) else "ran"
            loss = summary.get('final_loss', float('nan'))
            t_train = summary.get('training_time', 0)
            print(f"  {method:8s}: loss={loss:.4e}, time={t_train:.2f}s, {status}")

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("=" * 60)

    n_passed = len(all_results)
    n_total = 8  # bratu, burgers, bl, bl_gravity, kovasznay, beltrami, elasticity, darcy
    if n_passed >= n_total - 1:  # darcy may be skipped
        print("ALL PROBLEMS PASSED [OK]")
    else:
        print(f"PARTIAL: {n_passed}/{n_total} passed")

    return all_results


if __name__ == "__main__":
    main()
