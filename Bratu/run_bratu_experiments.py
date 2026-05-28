"""
Bratu Equation - Complete Experiment Runner (Modular LiL)
============================================================

This script:
1. Runs all 4 methods for specified N values
2. Allows selection of LiL basis type (Chebyshev, Fourier, ELM, mixed)
3. Saves a MASTER RESULTS FILE containing all data needed for plotting
4. Provides STANDALONE PLOTTING that loads from master file (no re-running)
5. Generates TWO sets of convergence plots (iterations AND line searches)

BASIS SELECTION:
    Set LIL_BASIS_TYPE to any of:
        'chebyshev'  : Chebyshev(x) x Chebyshev(y)
        'sin_sin'    : Sin(x) x Sin(y)  [satisfies homogeneous Dirichlet BCs]
        'cos_cos'    : Cos(x) x Cos(y)
        'fourier'    : Full Fourier (cos+sin) x (cos+sin)
        'cheb_sin'   : Chebyshev(x) x Sin(y)
        'sin_cheb'   : Sin(x) x Chebyshev(y)
        'cos_sin'    : Cos(x) x Sin(y)
        'sin_cos'    : Sin(x) x Cos(y)
        'elm'        : Extreme Learning Machine
    Or define a custom basis in create_custom_basis().

MASTER RESULTS FILE: bratu_master_results.json
- Contains all metrics history for each method/N combination
- Can be loaded to regenerate any plot without re-running experiments

Usage:
    # Run experiments and generate plots
    python run_bratu_experiments.py

    # Just regenerate plots from saved results (no experiments)
    python run_bratu_experiments.py --plot-only

    # Specify custom results file
    python run_bratu_experiments.py --plot-only --results-file path/to/results.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import torch

# Ensure float64 precision
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=16)

# Import from the core module
from bratu_core import (
    BratuPhysics, DiscretizationConfig, OptimizationConfig,
    solve_nonlinear_pinn, solve_quasilinear_pinn,
    solve_nonlinear_lil, solve_quasilinear_lil,
    evaluate_nn_solution, evaluate_nn_residual,
    evaluate_lil_solution, evaluate_lil_residual,
    create_bratu_basis,
    set_seed, clear_gpu_memory, DEVICE, MLP,
    calculate_hidden_dim
)

# Import from lil_basis for custom basis construction
from lilq_shared.basis import (
    Chebyshev1D, Fourier1D, TensorProductBasis2D,
    ELMBasis2D, AugmentedBasis1D
)

from lilq_shared.style import *
import lilq_shared.style as pub_style
from bratu_plots_refactored import *


# =============================================================================
# CONFIGURATION
# =============================================================================

N_VALUES = [5, 10, 15]
LAMBDA = 6.2

TARGET_LOSSES = {
    5: 2.5e-1,
    10: 1e-4,
    15: 2.5e-7
}

K_RATIO = 10
COLLOCATION_RATIOS = (0.85, 0.15)

MAX_ITERATIONS = {5: 5000, 10: 7500, 15: 10000}
MAX_LINE_SEARCHES = {5: 24000, 10: 30000, 15: 30000}
MAX_QUASI_ITERS = 25
MAX_LBFGS_PER_QUASI_ITER = {5: 300, 10: 300, 15: 400}
N_HIDDEN_LAYERS = 2

PRETRAIN_EPOCHS = 1500
PRETRAIN_TOL = 1e-5


# =============================================================================
# LiL BASIS CONFIGURATION
# =============================================================================

# Choose the LiL basis type for all LiL experiments.
# Options: 'chebyshev', 'sin_sin', 'cos_cos', 'fourier',
#          'cheb_sin', 'sin_cheb', 'cheb_cos', 'cos_cheb',
#          'cos_sin', 'sin_cos', 'elm', or 'custom'
LIL_BASIS_TYPE = "fourier"

# ELM-specific settings (only used when LIL_BASIS_TYPE = 'elm')
ELM_SEED = 42
ELM_ACTIVATION = 'tanh'


def create_custom_basis(N_x, N_y, x_domain, y_domain):
    """
    Define a custom basis here for maximum flexibility.

    This function is called when LIL_BASIS_TYPE = 'custom'.
    Modify the body to construct any basis you like.

    Examples:
        # Chebyshev(x) x Augmented-Sin(y)
        basis_x = Chebyshev1D(N_x - 1, x_domain)
        basis_y = AugmentedBasis1D(
            Fourier1D(N_y, y_domain, mode='sin'),
            include_constant=True, include_linear=True
        )
        return TensorProductBasis2D(basis_x, basis_y)

        # ELM with specific settings
        return ELMBasis2D(n_hidden=200, domain_x=x_domain, domain_y=y_domain,
                          activation='sigmoid', seed=123)
    """
    # Default custom: Chebyshev x Chebyshev (modify as needed)
    basis_x = Chebyshev1D(N_x - 1, x_domain)
    basis_y = Chebyshev1D(N_y - 1, y_domain)
    return TensorProductBasis2D(basis_x, basis_y)


def get_basis_for_experiment(N_x, N_y, physics):
    """
    Create the LiL basis for a given experiment configuration.

    Routes to either the built-in create_bratu_basis() or the
    user-defined create_custom_basis() depending on LIL_BASIS_TYPE.
    """
    x_domain = physics.x_domain
    y_domain = physics.y_domain

    if LIL_BASIS_TYPE == 'custom':
        return create_custom_basis(N_x, N_y, x_domain, y_domain)
    else:
        return create_bratu_basis(
            LIL_BASIS_TYPE, N_x, N_y,
            x_domain, y_domain,
            elm_seed=ELM_SEED,
            elm_activation=ELM_ACTIVATION
        )


# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(f"bratu_experiments_{LIL_BASIS_TYPE}")
MASTER_RESULTS_FILE = "bratu_master_results.json"

from lilq_shared.style import METHOD_CONFIG, METHOD_ORDER

N_COLORS = {5: '#1f77b4', 10: '#ff7f0e', 15: '#2ca02c'}

import lilq_shared.style as pub_style


# =============================================================================
# MASTER RESULTS FILE I/O
# =============================================================================

def save_master_results(all_results, output_dir):
    """Save all experiment results to a master JSON file."""
    output_dir = Path(output_dir)

    master_data = {
        'config': {
            'n_values': N_VALUES,
            'lambda': LAMBDA,
            'target_losses': TARGET_LOSSES,
            'k_ratio': K_RATIO,
            'collocation_ratios': COLLOCATION_RATIOS,
            'max_iterations': MAX_ITERATIONS,
            'max_line_searches': MAX_LINE_SEARCHES,
            'pretrain_epochs': PRETRAIN_EPOCHS,
            'pretrain_tol': PRETRAIN_TOL,
            'lil_basis_type': LIL_BASIS_TYPE,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': {}
    }

    for N in N_VALUES:
        if N not in all_results:
            continue

        master_data['results'][str(N)] = {}

        for method in METHOD_ORDER:
            if all_results[N].get(method) is None:
                continue

            result = all_results[N][method]
            metrics = result['metrics']

            if hasattr(metrics, 'to_dict'):
                metrics_data = metrics.to_dict()
            elif hasattr(metrics, 'data'):
                metrics_data = {k: list(v) for k, v in metrics.data.items()}
            else:
                metrics_data = metrics

            master_data['results'][str(N)][method] = {
                'metrics': metrics_data,
                'summary': result['summary'],
                'n_params': result['n_params'],
                'converged': result['converged'],
                'final_loss': result['final_loss'],
                'total_iterations': result['summary'].get('total_iterations', 0),
                'total_line_searches': result['summary'].get('total_line_searches', 0),
                'pretrain_loss': result.get('pretrain_loss', 0.0)
            }

    master_path = output_dir / MASTER_RESULTS_FILE
    with open(master_path, 'w') as f:
        json.dump(master_data, f, indent=2)

    print(f"\nMaster results saved to: {master_path}")
    return master_path


def load_master_results(results_path):
    """Load master results from JSON file."""
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, 'r') as f:
        master_data = json.load(f)

    print(f"Loaded results from: {results_path}")
    print(f"  Timestamp: {master_data['config'].get('timestamp', 'unknown')}")
    print(f"  N values: {master_data['config']['n_values']}")
    print(f"  LiL basis: {master_data['config'].get('lil_basis_type', 'unknown')}")

    all_results = {}

    for N_str, methods_data in master_data['results'].items():
        N = int(N_str)
        all_results[N] = {}

        for method, data in methods_data.items():
            all_results[N][method] = {
                'metrics': data['metrics'],
                'summary': data['summary'],
                'n_params': data['n_params'],
                'converged': data['converged'],
                'final_loss': data['final_loss'],
                'pretrain_loss': data.get('pretrain_loss', 0.0)
            }

    return all_results, master_data['config']


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(N, R_tol, output_dir):
    """Run all 4 methods for a single N value."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENTS FOR N = {N}")
    print(f"Target loss tolerance: {R_tol:.2e}")
    print(f"LiL basis type: {LIL_BASIS_TYPE}")
    print(f"{'='*80}")

    results = {}
    exp_dir = output_dir / f"N_{N}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)

    physics = BratuPhysics(lambda_=LAMBDA, x_domain=(0.0, 1.0), y_domain=(0.0, 1.0))

    discretization = DiscretizationConfig(
        N_x=N, N_y=N, k_ratio=K_RATIO,
        collocation_ratios=COLLOCATION_RATIOS,
        domain_sampling="random", boundary_sampling="random"
    )

    optimization = OptimizationConfig(
        n_epochs_adam=0,
        max_iterations=MAX_ITERATIONS[N],
        max_line_searches=MAX_LINE_SEARCHES[N],
        max_quasi_iters=MAX_QUASI_ITERS,
        n_epochs_lbfgs_per_iter=MAX_LBFGS_PER_QUASI_ITER[N],
        R_tol=R_tol,
        regularization_weight=1e-10,
        lambda_pde=1.0, lambda_bc=10.0,
        n_hidden_layers=N_HIDDEN_LAYERS,
        pretrain_epochs=PRETRAIN_EPOCHS,
        pretrain_tol=PRETRAIN_TOL
    )

    # Create the LiL basis for this experiment
    basis = get_basis_for_experiment(N, N, physics)
    print(f"\nLiL Basis: {basis}")
    print(f"  Total basis functions: {basis.n_basis}")

    # -------------------------------------------------------------------------
    # Method 1: Nonlinear LiL (LiL-N)
    # -------------------------------------------------------------------------
    print(f"\n{'-'*40}\nMethod 1: Nonlinear LiL (LiL-N)\n{'-'*40}")
    try:
        basis_obj, coeffs, metrics, summary = solve_nonlinear_lil(
            physics, discretization, optimization, basis, device, verbose=True)
        results['nl_lil'] = {
            'basis': basis_obj, 'coefficients': coeffs,
            'metrics': metrics, 'summary': summary,
            'final_loss': summary['final_loss'],
            'converged': summary['converged'],
            'n_params': basis_obj.n_basis,
            'pretrain_loss': summary.get('pretrain_loss', 0.0)
        }
        metrics.to_dataframe().to_csv(exp_dir / 'nl_lil_metrics.csv', index=False)
        with open(exp_dir / 'nl_lil_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        save_lil_checkpoint(basis_obj, coeffs, metrics, summary,
                            exp_dir / 'nl_lil_checkpoint.npz')
    except Exception as e:
        print(f"Nonlinear LiL failed: {e}")
        import traceback; traceback.print_exc()
        results['nl_lil'] = None

    clear_gpu_memory()

    # -------------------------------------------------------------------------
    # Method 2: Quasilinear LiL (LiL-Q)
    # -------------------------------------------------------------------------
    # Recreate basis to ensure clean state (especially for ELM seed)
    basis = get_basis_for_experiment(N, N, physics)

    print(f"\n{'-'*40}\nMethod 2: Quasilinear LiL (LiL-Q)\n{'-'*40}")
    try:
        basis_obj, coeffs, metrics, summary = solve_quasilinear_lil(
            physics, discretization, optimization, basis, verbose=True)
        results['ql_lil'] = {
            'basis': basis_obj, 'coefficients': coeffs,
            'metrics': metrics, 'summary': summary,
            'final_loss': summary['final_loss'],
            'converged': summary['converged'],
            'n_params': basis_obj.n_basis,
            'pretrain_loss': summary.get('pretrain_loss', 0.0)
        }
        metrics.to_dataframe().to_csv(exp_dir / 'ql_lil_metrics.csv', index=False)
        with open(exp_dir / 'ql_lil_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        save_lil_checkpoint(basis_obj, coeffs, metrics, summary,
                            exp_dir / 'ql_lil_checkpoint.npz')
    except Exception as e:
        print(f"Quasilinear LiL failed: {e}")
        import traceback; traceback.print_exc()
        results['ql_lil'] = None

    # -------------------------------------------------------------------------
    # Method 3: Standard PINN (NiL-N)
    # -------------------------------------------------------------------------
    print(f"\n{'-'*40}\nMethod 3: Standard PINN (NiL-N)\n{'-'*40}")
    try:
        model, metrics, summary = solve_nonlinear_pinn(
            physics, discretization, optimization, device, verbose=True)
        results['std_pinn'] = {
            'model': model, 'metrics': metrics, 'summary': summary,
            'final_loss': summary['final_loss'],
            'converged': summary['converged'],
            'n_params': sum(p.numel() for p in model.parameters()),
            'pretrain_loss': summary.get('pretrain_loss', 0.0)
        }
        metrics.to_dataframe().to_csv(exp_dir / 'std_pinn_metrics.csv', index=False)
        with open(exp_dir / 'std_pinn_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        torch.save(model.state_dict(), exp_dir / 'std_pinn_model.pt')
    except Exception as e:
        print(f"Standard PINN failed: {e}")
        import traceback; traceback.print_exc()
        results['std_pinn'] = None

    clear_gpu_memory()

    # -------------------------------------------------------------------------
    # Method 4: Quasilinear PINN (NiL-Q)
    # -------------------------------------------------------------------------
    print(f"\n{'-'*40}\nMethod 4: Quasilinear PINN (NiL-Q)\n{'-'*40}")
    try:
        model, metrics, summary = solve_quasilinear_pinn(
            physics, discretization, optimization, device, verbose=True)
        results['ql_pinn'] = {
            'model': model, 'metrics': metrics, 'summary': summary,
            'final_loss': summary['final_loss'],
            'converged': summary['converged'],
            'n_params': sum(p.numel() for p in model.parameters()),
            'pretrain_loss': summary.get('pretrain_loss', 0.0)
        }
        metrics.to_dataframe().to_csv(exp_dir / 'ql_pinn_metrics.csv', index=False)
        with open(exp_dir / 'ql_pinn_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        torch.save(model.state_dict(), exp_dir / 'ql_pinn_model.pt')
    except Exception as e:
        print(f"Quasilinear PINN failed: {e}")
        import traceback; traceback.print_exc()
        results['ql_pinn'] = None

    results['physics'] = physics
    results['discretization'] = discretization

    return results


def run_all_experiments():
    """Run experiments for all N values and save master results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for N in N_VALUES:
        R_tol = TARGET_LOSSES.get(N, 1e-4)
        results = run_single_experiment(N, R_tol, OUTPUT_DIR)
        all_results[N] = results

    save_master_results(all_results, OUTPUT_DIR)

    return all_results


# =============================================================================
# VISUALIZATION HELPER FUNCTIONS
# =============================================================================

def get_metrics_arrays(metrics):
    """Extract arrays from metrics (handles both object and dict formats)."""
    if hasattr(metrics, 'data'):
        data = metrics.data
    else:
        data = metrics

    iterations = np.array(data.get('iteration', []))
    func_evals = np.array(data.get('n_func_evals', []))
    losses = np.array(data.get('loss', []))

    return iterations, func_evals, losses


def save_solution_panel(X, Y, U, filename, vmin=None, vmax=None, cmap='turbo'):
    """Save a single solution contour panel as PDF."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

    if vmin is None: vmin = U.min()
    if vmax is None: vmax = U.max()

    pcm = ax.pcolormesh(X, Y, U, vmin=vmin, vmax=vmax, cmap=cmap, shading='gouraud')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_aspect('equal')

    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename.name}")


def save_residual_panel(X, Y, residual, filename, symmetric_clim=None):
    """Save a single residual contour panel as PDF."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)

    max_abs = np.abs(residual).max()
    vmax = symmetric_clim if symmetric_clim is not None else max_abs
    vmin = -vmax

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    pcm = ax.pcolormesh(X, Y, residual, cmap='RdBu_r', norm=norm, shading='gouraud')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_aspect('equal')

    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(f'max = {max_abs:.2e}', fontsize=9, pad=3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename.name} (max |res| = {max_abs:.2e})")
    return max_abs



# =============================================================================
# SOLUTION AND RESIDUAL PANELS
# =============================================================================

def generate_solution_residual_panels(all_results, N, output_dir):
    """Generate solution and residual contour panels for a given N."""
    if N not in all_results:
        return

    output_dir = Path(output_dir)
    physics = all_results[N].get('physics')
    discretization = all_results[N].get('discretization')

    if physics is None or discretization is None:
        print(f"  Skipping panels for N={N} (no physics/discretization objects)")
        return

    all_data = {}

    for method_key in METHOD_ORDER:
        if all_results[N].get(method_key) is None:
            continue

        result = all_results[N][method_key]

        if method_key in ['std_pinn', 'ql_pinn']:
            model = result.get('model')
            if model is None:
                continue
            X_sol, Y_sol, U = evaluate_nn_solution(model, physics, discretization)
            X_res, Y_res, residual = evaluate_nn_residual(model, physics, discretization)
        else:
            basis_obj = result.get('basis')
            coeffs = result.get('coefficients')
            if basis_obj is None or coeffs is None:
                continue
            X_sol, Y_sol, U = evaluate_lil_solution(basis_obj, coeffs, physics, discretization)
            X_res, Y_res, residual = evaluate_lil_residual(basis_obj, coeffs, physics, discretization)

        all_data[method_key] = {
            'X_sol': X_sol, 'Y_sol': Y_sol, 'U': U,
            'X_res': X_res, 'Y_res': Y_res, 'residual': residual
        }

    if not all_data:
        return

    u_min = min(all_data[k]['U'].min() for k in all_data)
    u_max = max(all_data[k]['U'].max() for k in all_data)
    global_residual_clim = max(np.abs(all_data[k]['residual']).max() for k in all_data)

    print(f"  N = {N}: Solution [{u_min:.4f}, {u_max:.4f}], Max res {global_residual_clim:.2e}")

    for method_key in METHOD_ORDER:
        if method_key not in all_data:
            continue

        config = METHOD_CONFIG[method_key]
        data = all_data[method_key]

        sol_filename = output_dir / f"Bratu_{config['filename_prefix']}_N{N}_sol.pdf"
        save_solution_panel(data['X_sol'], data['Y_sol'], data['U'],
                           sol_filename, vmin=u_min, vmax=u_max)

        sol_filename2 = output_dir / f"Bratu_{config['filename_prefix']}_N{N}_sol.png"
        save_solution_panel(data['X_sol'], data['Y_sol'], data['U'],
                           sol_filename2, vmin=u_min, vmax=u_max)

        res_filename = output_dir / f"Bratu_{config['filename_prefix']}_N{N}_res.pdf"
        save_residual_panel(data['X_res'], data['Y_res'], data['residual'],
                           res_filename, symmetric_clim=global_residual_clim)


# =============================================================================
# MASTER FIGURE GENERATION
# =============================================================================

def generate_all_figures(all_results, output_dir, skip_panels=False):
    """Generate all publication figures."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION FIGURES")
    print(f"LiL basis: {LIL_BASIS_TYPE}")
    print("=" * 80)

    # Import the refactored plotting functions
    from bratu_plots_refactored import (
        plot_convergence_by_method,
        plot_convergence_by_size,
        plot_solution_fields,
        plot_solution_fields_single_N,
    )

    print("\n[1/4] Convergence histories (by method, 2x2)...")
    plot_convergence_by_method(all_results, N_VALUES, TARGET_LOSSES,
                               figures_dir, n_colors=N_COLORS)

    print("\n[2/4] Convergence histories (by size)...")
    plot_convergence_by_size(all_results, N_VALUES, TARGET_LOSSES,
                             figures_dir)

    if not skip_panels:
        print("\n[3/4] Solution fields (combined grid)...")
        from bratu_core import evaluate_nn_solution, evaluate_lil_solution
        plot_solution_fields(all_results, N_VALUES, figures_dir,
                             evaluate_nn_solution, evaluate_lil_solution)
        
        for N in N_VALUES:
            plot_solution_fields_single_N(all_results, N, figures_dir,
                                  evaluate_nn_solution, evaluate_lil_solution)
    else:
        print("\n[3/4] Skipping solution panels (--plot-only mode)")

    print(f"\n{'=' * 80}")
    print(f"All figures saved to: {figures_dir}")
    print(f"{'=' * 80}")


def print_results_summary(all_results):
    """Print summary table."""
    print("\n" + "="*120)
    print(f"RESULTS SUMMARY (LiL basis: {LIL_BASIS_TYPE})")
    print("="*120)

    header = f"{'N':>5} | {'Method':<25} | {'Params':>8} | {'Iterations':>12} | {'Line Searches':>14} | {'Final Loss':>12} | {'Conv':>6}"
    print(header)
    print("-" * len(header))

    for N in N_VALUES:
        if N not in all_results:
            continue

        for method_key in METHOD_ORDER:
            if all_results[N].get(method_key) is None:
                print(f"{N:>5} | {METHOD_CONFIG[method_key]['label']:<25} | {'FAILED':>8}")
                continue

            result = all_results[N][method_key]
            summary = result.get('summary', {})

            n_params = result.get('n_params', summary.get('n_params', 0))
            total_iters = summary.get('total_iterations', 0)
            total_evals = summary.get('total_line_searches', 0)
            final_loss = result.get('final_loss', summary.get('final_loss', 0))
            converged = result.get('converged', summary.get('converged', False))

            print(f"{N:>5} | {METHOD_CONFIG[method_key]['label']:<25} | "
                  f"{n_params:>8} | {total_iters:>12} | {total_evals:>14} | "
                  f"{final_loss:>12.2e} | {'Yes' if converged else 'No':>6}")

        print("-" * len(header))


# =============================================================================
# STANDALONE PLOTTING FUNCTION
# =============================================================================

def plot_from_saved_results(results_file, output_dir=None):
    """Load saved results and regenerate all plots."""
    all_results, config = load_master_results(results_file)

    if output_dir is None:
        output_dir = Path(results_file).parent

    print_results_summary(all_results)
    generate_all_figures(all_results, output_dir, skip_panels=True)

    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bratu equation experiments with modular LiL basis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments with default Chebyshev basis
  python run_bratu_experiments.py

  # Just regenerate plots from saved results
  python run_bratu_experiments.py --plot-only

  # Use custom results file
  python run_bratu_experiments.py --plot-only --results-file path/to/results.json

To change the LiL basis, edit LIL_BASIS_TYPE at the top of this file.
        """
    )

    parser.add_argument('--plot-only', action='store_true',
                       help='Only generate plots from saved results (no experiments)')
    parser.add_argument('--results-file', type=str, default=None,
                       help='Path to master results JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures')

    args = parser.parse_args()

    if args.plot_only:
        results_file = args.results_file
        if results_file is None:
            results_file = OUTPUT_DIR / MASTER_RESULTS_FILE

        output_dir = args.output_dir if args.output_dir else Path(results_file).parent

        print("="*80)
        print("PLOT-ONLY MODE: Loading saved results")
        print("="*80)

        all_results = plot_from_saved_results(results_file, output_dir)

    else:
        print("="*80)
        print("BRATU EQUATION EXPERIMENTS (MODULAR LiL)")
        print("="*80)
        print(f"\nN values: {N_VALUES}")
        print(f"Lambda: {LAMBDA}")
        print(f"Target losses: {TARGET_LOSSES}")
        print(f"LiL basis type: {LIL_BASIS_TYPE}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Device: {DEVICE}")

        all_results = run_all_experiments()
        print_results_summary(all_results)
        generate_all_figures(all_results, OUTPUT_DIR, skip_panels=False)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
