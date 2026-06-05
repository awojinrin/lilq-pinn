"""
================================================================================
BURGERS EQUATION - LiL-Q BASIS FUNCTION COMPARATIVE STUDY
================================================================================

Compares multiple LiL-Q networks with different basis function choices
on the viscous Burgers equation.

PDE:  u_t + u * u_x - nu * u_xx = 0
Domain:  x in [-1, 1],  t in [0, T_final]
IC:   u(x, 0) = -sin(pi * x)
BC:   u(-1, t) = 0,  u(1, t) = 0

ASYMMETRY:
    Unlike Bratu (symmetric x and y with identical BCs), Burgers has
    fundamentally different physics in each dimension:
      - x-direction: homogeneous Dirichlet BCs -> Sin basis natural fit
      - t-direction: IC at t=0 is NON-ZERO, no constraint at t=T -> needs
        a complete basis (constant, polynomial) to represent IC

    This makes MIXED bases (e.g., Sin(x) x Cheb(t)) physically meaningful,
    and pure symmetric bases (Sin x Sin, Cos x Cos) suspect.

BASIS FUNCTIONS COMPARED:
    1. Cheb(x) x Cheb(t)         -- general purpose baseline
    2. Sin(x) x Cheb(t)          -- respects x-Dirichlet, flexible in t
    3. Sin(x) x Sin(t)           -- enforces u=0 at BOTH boundaries AND t=0,t=T
    4. Cos(x) x Cheb(t)          -- cos(0)=1, cannot enforce Dirichlet in x
    5. AugSin(x) x Cheb(t)       -- augmented sin in x, Chebyshev in t
    6. Fourier(both,x) x Cheb(t) -- full Fourier in x, Chebyshev in t
    7. Fourier(both,x) x Fourier(both,t) -- full Fourier in both directions
    8. ELM (tanh)                 -- random nonlinear basis

Usage::

    python experiments/run_burgers_basis_comparison.py
    python experiments/run_burgers_basis_comparison.py --N 16 --viscosity 0.1
    python experiments/run_burgers_basis_comparison.py --bases sin_cheb elm

================================================================================
"""

import sys
import os
import numpy as np
import scipy.linalg
import time
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

# Ensure project root is importable
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from lilq.basis import (
    Chebyshev1D, Fourier1D, TensorProductBasis2D,
    ELMBasis2D, AugmentedBasis1D,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ComparisonConfig:
    """Master configuration for the Burgers basis comparison."""

    # Physics
    viscosity: float = 0.1
    T_final: float = 1.0
    x_domain: Tuple[float, float] = (-1.0, 1.0)

    # Discretization -- N_x = N_t = 16 gives 256 coefficients for all bases
    N_x: int = 16
    N_t: int = 16
    k_ratio: int = 10
    collocation_ratios: Tuple[float, float, float] = (0.85, 0.05, 0.10)
    seed: int = 42

    # LiL-Q optimisation
    lambda_pde: float = 1.0
    lambda_ic: float = 10.0
    lambda_bc: float = 10.0
    R_tol: float = 1e-5
    max_quasi_iters: int = 50
    stagnation_window: int = 5
    stagnation_rtol: float = 1e-3

    # Output
    output_dir: str = "results/burgers_basis_comparison"


# ---------------------------------------------------------------------------
# Visual identity -- colours/markers consistent with the Bratu comparison.
# The "identity" is the distinguishing x-basis type:
#   Cheb -> red,  Sin -> teal,  Cos -> steel blue,  AugSin -> orange,
#   Fourier -> dark teal,  ELM -> purple.
# Sin(x) x Cheb(t) is new (no Bratu analogue) -> rose (#D81B60).
# ---------------------------------------------------------------------------
BASIS_CONFIGS = {
    'cheb_cheb': {
        'label': r'Cheb$(x)$ $\times$ Cheb$(t)$',
        'short_label': 'Cheb x Cheb',
        'color': '#E63946',    # red  (same as Bratu Cheb)
        'marker': 'o',
    },
    'sin_cheb': {
        'label': r'Sin$(x)$ $\times$ Cheb$(t)$',
        'short_label': 'Sin x Cheb',
        'color': '#D81B60',    # rose -- NEW, no Bratu analogue
        'marker': 'h',
    },
    'sin_sin': {
        'label': r'Sin$(x)$ $\times$ Sin$(t)$',
        'short_label': 'Sin x Sin',
        'color': '#2A9D8F',    # teal  (same as Bratu Sin)
        'marker': 'D',
    },
    'cos_cheb': {
        'label': r'Cos$(x)$ $\times$ Cheb$(t)$',
        'short_label': 'Cos x Cheb',
        'color': '#457B9D',    # steel blue  (same as Bratu Cos)
        'marker': 's',
    },
    'augsin_cheb': {
        'label': r'AugSin$(x)$ $\times$ Cheb$(t)$',
        'short_label': 'AugSin x Cheb',
        'color': '#F4A261',    # orange  (same as Bratu AugSin)
        'marker': '^',
    },
    'fourier_cheb': {
        'label': r'Fourier$(x)$ $\times$ Cheb$(t)$',
        'short_label': 'Fourier x Cheb',
        'color': '#264653',    # dark teal  (same as Bratu Fourier)
        'marker': 'P',
    },
    'fourier_fourier': {
        'label': r'Fourier$(x)$ $\times$ Fourier$(t)$',
        'short_label': 'Fourier x Fourier',
        'color': '#1B9E77',    # distinct green
        'marker': 'X',
    },
    'elm': {
        'label': 'ELM (tanh)',
        'short_label': 'ELM',
        'color': '#6A0572',    # purple  (same as Bratu ELM)
        'marker': 'v',
    },
}


# =============================================================================
# BASIS FACTORY
# =============================================================================

def create_comparison_basis(
    key: str,
    N_x: int,
    N_t: int,
    x_domain: Tuple[float, float],
    t_domain: Tuple[float, float],
    seed: int = 42,
):
    """
    Build a 2D basis for the Burgers problem.

    Convention: dimension-0 = x (spatial), dimension-1 = t (temporal).
    All bases target N_x * N_t total coefficients.

    Returns
    -------
    basis : TensorProductBasis2D or ELMBasis2D
    n_coeffs : int
    description : str
    """
    if key == 'cheb_cheb':
        basis = TensorProductBasis2D(
            Chebyshev1D(order=N_x - 1, domain=x_domain),
            Chebyshev1D(order=N_t - 1, domain=t_domain))
        desc = f"T_0..{N_x-1}(x) x T_0..{N_t-1}(t)"

    elif key == 'sin_cheb':
        basis = TensorProductBasis2D(
            Fourier1D(n_modes=N_x, domain=x_domain, mode='sin'),
            Chebyshev1D(order=N_t - 1, domain=t_domain))
        desc = f"sin(1..{N_x})(x) x T_0..{N_t-1}(t)"

    elif key == 'sin_sin':
        basis = TensorProductBasis2D(
            Fourier1D(n_modes=N_x, domain=x_domain, mode='sin'),
            Fourier1D(n_modes=N_t, domain=t_domain, mode='sin'))
        desc = f"sin(1..{N_x})(x) x sin(1..{N_t})(t)"

    elif key == 'cos_cheb':
        basis = TensorProductBasis2D(
            Fourier1D(n_modes=N_x, domain=x_domain, mode='cos'),
            Chebyshev1D(order=N_t - 1, domain=t_domain))
        desc = f"cos(0..{N_x-1})(x) x T_0..{N_t-1}(t)"

    elif key == 'augsin_cheb':
        n_sin = N_x - 2
        if n_sin < 1:
            n_sin = 1
        bx = AugmentedBasis1D(
            Fourier1D(n_modes=n_sin, domain=x_domain, mode='sin'),
            include_constant=True, include_linear=True)
        basis = TensorProductBasis2D(
            bx, Chebyshev1D(order=N_t - 1, domain=t_domain))
        desc = f"[1, x, sin(1..{n_sin})](x) x T_0..{N_t-1}(t)"

    elif key == 'fourier_cheb':
        basis = TensorProductBasis2D(
            Fourier1D(n_modes=N_x, domain=x_domain, mode='both'),
            Chebyshev1D(order=N_t - 1, domain=t_domain))
        desc = f"cos+sin({N_x})(x) x T_0..{N_t-1}(t)"

    elif key == 'fourier_fourier':
        basis = TensorProductBasis2D(
            Fourier1D(n_modes=N_x, domain=x_domain, mode='both'),
            Fourier1D(n_modes=N_t, domain=t_domain, mode='both'))
        desc = f"cos+sin({N_x})(x) x cos+sin({N_t})(t)"

    elif key == 'elm':
        n_hidden = N_x * N_t
        basis = ELMBasis2D(
            n_hidden=n_hidden, domain_x=x_domain, domain_y=t_domain,
            activation='tanh', seed=seed)
        desc = f"ELM(tanh, {n_hidden} neurons)"

    else:
        raise ValueError(f"Unknown basis key: {key}")

    return basis, basis.n_basis, desc


# =============================================================================
# COLLOCATION POINTS
# =============================================================================

def generate_collocation(config: ComparisonConfig) -> Dict:
    """Generate PDE, IC, and BC collocation points."""
    np.random.seed(config.seed)

    x_min, x_max = config.x_domain
    t_max = config.T_final
    n_coeffs = config.N_x * config.N_t
    r = config.collocation_ratios
    nr = [ri / sum(r) for ri in r]

    # PDE interior (meshgrid)
    n_pde = config.k_ratio * nr[0] * n_coeffs
    n_dim = math.ceil(np.sqrt(n_pde))
    eps = 1e-5
    xp = np.linspace(x_min + eps, x_max - eps, n_dim, dtype=np.float64)
    tp = np.linspace(eps, t_max - eps, n_dim, dtype=np.float64)
    Xp, Tp = np.meshgrid(xp, tp, indexing='ij')
    pts_pde = np.column_stack([Xp.ravel(), Tp.ravel()]).astype(np.float64)

    # IC: u(x, 0)
    n_ic = max(math.ceil(config.k_ratio * nr[1] * n_coeffs), 10)
    x_ic = np.linspace(x_min, x_max, n_ic, dtype=np.float64)
    pts_ic = np.column_stack([x_ic, np.zeros(n_ic, dtype=np.float64)])

    # BC: u(-1, t) = 0,  u(1, t) = 0
    n_bc = max(math.ceil(config.k_ratio * nr[2] * n_coeffs / 2), 10)
    t_bc = np.linspace(0, t_max, n_bc, dtype=np.float64)
    pts_bc_left = np.column_stack([np.full(n_bc, x_min), t_bc])
    pts_bc_right = np.column_stack([np.full(n_bc, x_max), t_bc])

    return {
        'pts_pde': pts_pde, 'pts_ic': pts_ic,
        'pts_bc_left': pts_bc_left, 'pts_bc_right': pts_bc_right,
        'x_ic': x_ic,
        'n_pde': pts_pde.shape[0], 'n_ic': n_ic, 'n_bc': n_bc,
    }


# =============================================================================
# GENERIC LiL-Q SOLVER FOR BURGERS
# =============================================================================

def _bmm(basis, pts, dx=0, dy=0):
    """Build basis matrix helper."""
    return (basis.evaluate(pts[:, 0], pts[:, 1]) if (dx == 0 and dy == 0)
            else basis.derivative(pts[:, 0], pts[:, 1], dx=dx, dy=dy))


def solve_lilq_burgers_comparison(
    basis_key: str,
    basis,
    config: ComparisonConfig,
    pts: Dict,
    verbose: bool = True,
) -> Dict:
    """
    Solve the Burgers equation via LiL-Q with an arbitrary basis.

    Quasilinearisation of  u_t + u u_x - nu u_xx = 0  gives:
        u_t + u_prev u_x + u_x_prev u - nu u_xx = u_prev u_x_prev
    which is linear in the new u.

    Returns
    -------
    dict with keys: coefficients, metrics, summary, basis_key
    """
    n_coeffs = basis.n_basis
    label = BASIS_CONFIGS[basis_key]['short_label']
    if verbose:
        print(f"\n{'='*70}")
        print(f"  LiL-Q: {label}")
        print(f"  Coefficients: {n_coeffs}")
        print(f"{'='*70}")

    # ---- build basis matrices ----
    t0 = time.time()
    A_u   = _bmm(basis, pts['pts_pde'], 0, 0)
    A_ux  = _bmm(basis, pts['pts_pde'], 1, 0)
    A_ut  = _bmm(basis, pts['pts_pde'], 0, 1)
    A_uxx = _bmm(basis, pts['pts_pde'], 2, 0)
    A_ic  = _bmm(basis, pts['pts_ic'],  0, 0)
    A_bcL = _bmm(basis, pts['pts_bc_left'],  0, 0)
    A_bcR = _bmm(basis, pts['pts_bc_right'], 0, 0)
    t_build = time.time() - t0
    if verbose:
        print(f"  Matrix build: {t_build:.3f}s   "
              f"PDE={pts['n_pde']}  IC={pts['n_ic']}  BC={pts['n_bc']}")

    ic_target = -np.sin(np.pi * pts['x_ic'])

    # ---- pretrain to u = 0 ----
    n_fd = max(10, int(np.sqrt(2 * n_coeffs)))
    xf = np.linspace(config.x_domain[0], config.x_domain[1], n_fd)
    tf = np.linspace(0, config.T_final, n_fd)
    Xf, Tf = np.meshgrid(xf, tf, indexing='ij')
    pf = np.column_stack([Xf.ravel(), Tf.ravel()])
    A_fit = _bmm(basis, pf, 0, 0)
    target_zero = np.zeros(pf.shape[0], dtype=np.float64)
    beta, _, _, _ = scipy.linalg.lstsq(
        A_fit, target_zero, lapack_driver='gelsy')
    beta = beta.astype(np.float64)
    pretrain_mse = float(np.mean((A_fit @ beta - target_zero) ** 2))
    if verbose:
        print(f"  Pre-train to u=0: MSE = {pretrain_mse:.6e}")

    nu = config.viscosity
    lp, li, lb = config.lambda_pde, config.lambda_ic, config.lambda_bc
    n_pde, n_ic, n_bc = pts['n_pde'], pts['n_ic'], pts['n_bc']

    metrics = {
        'iteration': [], 'loss': [],
        'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
    }

    t_start = time.time()

    def nonlinear_residual(b):
        return (A_ut @ b) + (A_u @ b) * (A_ux @ b) - nu * (A_uxx @ b)

    def record(it, b):
        pr = nonlinear_residual(b)
        pde_m = float(np.mean(pr**2))
        ic_m  = float(np.mean((A_ic @ b - ic_target)**2))
        bc_m  = float(np.mean((A_bcL @ b)**2) + np.mean((A_bcR @ b)**2))
        tot   = lp * pde_m + li * ic_m + lb * bc_m
        metrics['iteration'].append(it)
        metrics['loss'].append(tot)
        metrics['pde_loss'].append(pde_m)
        metrics['ic_loss'].append(ic_m)
        metrics['bc_loss'].append(bc_m)
        return tot, pde_m, ic_m, bc_m

    tot, pde_m, ic_m, bc_m = record(0, beta)
    if verbose:
        print(f"  Iter   0: total={tot:.6e}  pde={pde_m:.6e}  "
              f"ic={ic_m:.6e}  bc={bc_m:.6e}")

    converged, stagnated = False, False
    best_loss, no_improve = tot, 0
    it = 0

    # ---- quasilinearization loop ----
    for qi in range(config.max_quasi_iters):
        it = qi + 1

        u_prev  = A_u  @ beta
        ux_prev = A_ux @ beta

        # Linearised PDE operator
        A_lin = (A_ut
                 + u_prev.reshape(-1, 1)  * A_ux
                 + ux_prev.reshape(-1, 1) * A_u
                 - nu * A_uxx)
        b_lin = u_prev * ux_prev

        # Weighted stack with IC / BC
        wp = np.sqrt(lp / n_pde)
        wi = np.sqrt(li / n_ic)
        wb = np.sqrt(lb / n_bc)

        A_stack = np.vstack([wp * A_lin, wb * A_bcL, wb * A_bcR, wi * A_ic])
        b_stack = np.concatenate([wp * b_lin,
                                  wb * np.zeros(n_bc),
                                  wb * np.zeros(n_bc),
                                  wi * ic_target])

        # QR solve via least squares
        beta, _, _, _ = scipy.linalg.lstsq(
            A_stack, b_stack, lapack_driver='gelsy')
        beta = beta.astype(np.float64)

        tot, pde_m, ic_m, bc_m = record(it, beta)

        if verbose and (it <= 10 or it % 5 == 0):
            print(f"  Iter {it:3d}: total={tot:.6e}  pde={pde_m:.6e}  "
                  f"ic={ic_m:.6e}  bc={bc_m:.6e}")

        # Convergence check
        if tot < config.R_tol:
            converged = True
            if verbose:
                print(f"  ** Converged at iteration {it} **")
            break

        # Stagnation check
        if tot < best_loss * (1.0 - config.stagnation_rtol):
            best_loss, no_improve = tot, 0
        else:
            no_improve += 1
        if no_improve >= config.stagnation_window:
            stagnated = True
            if verbose:
                print(f"  ** Stagnated at iteration {it} "
                      f"(best={best_loss:.6e}) **")
            break

    total_time = time.time() - t_start
    summary = {
        'basis_key': basis_key,
        'n_coefficients': n_coeffs,
        'n_iterations': it,
        'final_loss': tot,
        'final_pde_loss': pde_m,
        'final_ic_loss': ic_m,
        'final_bc_loss': bc_m,
        'converged': converged,
        'stagnated': stagnated,
        'total_time': total_time,
        'build_time': t_build,
    }
    return {
        'coefficients': beta,
        'metrics': metrics,
        'summary': summary,
        'basis_key': basis_key,
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_on_grid(basis, coeffs, config, n_eval=200):
    """Return X, T, U, residual arrays on a uniform grid."""
    x = np.linspace(config.x_domain[0], config.x_domain[1], n_eval)
    t = np.linspace(0, config.T_final, n_eval)
    X, T = np.meshgrid(x, t, indexing='ij')
    pts = np.column_stack([X.ravel(), T.ravel()])

    u    = _bmm(basis, pts, 0, 0) @ coeffs
    u_t  = _bmm(basis, pts, 0, 1) @ coeffs
    u_x  = _bmm(basis, pts, 1, 0) @ coeffs
    u_xx = _bmm(basis, pts, 2, 0) @ coeffs
    res  = u_t + u * u_x - config.viscosity * u_xx

    return X, T, u.reshape(n_eval, n_eval), res.reshape(n_eval, n_eval)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_all_bases(
    config: ComparisonConfig,
    basis_keys: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """Run all bases and return results dict + collocation points."""
    if basis_keys is None:
        basis_keys = list(BASIS_CONFIGS.keys())

    pts = generate_collocation(config)
    t_dom = (0.0, config.T_final)

    if verbose:
        print("=" * 70)
        print("BURGERS EQUATION - LiL-Q BASIS FUNCTION COMPARISON")
        print("=" * 70)
        print(f"  nu = {config.viscosity},  T = {config.T_final}")
        print(f"  x in [{config.x_domain[0]}, {config.x_domain[1]}]")
        print(f"  N_x={config.N_x}, N_t={config.N_t}  =>  "
              f"target coefficients = {config.N_x * config.N_t}")
        print(f"  PDE pts={pts['n_pde']}, IC pts={pts['n_ic']}, "
              f"BC pts={pts['n_bc']}")
        print(f"  R_tol={config.R_tol}, max_iters={config.max_quasi_iters}")

    results = {}
    for bk in basis_keys:
        try:
            basis, nc, desc = create_comparison_basis(
                bk, config.N_x, config.N_t, config.x_domain, t_dom, config.seed)
            target = config.N_x * config.N_t
            tag = "" if nc == target else f"  [NOTE: {nc} != {target}]"
            if verbose:
                print(f"\n  Created: {desc}  ({nc} coefficients){tag}")
            r = solve_lilq_burgers_comparison(bk, basis, config, pts, verbose)
            r['basis'] = basis
            r['description'] = desc
            results[bk] = r
        except Exception as e:
            print(f"\n  !! {bk} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[bk] = None
    return results, pts


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_results(results, config, out):
    """Save metrics and summaries as JSON."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = {
        'config': {
            'viscosity': config.viscosity,
            'T_final': config.T_final,
            'N_x': config.N_x,
            'N_t': config.N_t,
            'k_ratio': config.k_ratio,
            'R_tol': config.R_tol,
            'max_quasi_iters': config.max_quasi_iters,
            'stagnation_window': config.stagnation_window,
            'stagnation_rtol': config.stagnation_rtol,
            'lambda_pde': config.lambda_pde,
            'lambda_ic': config.lambda_ic,
            'lambda_bc': config.lambda_bc,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': {
            bk: ({'metrics': r['metrics'], 'summary': r['summary'],
                  'description': r.get('description', '')} if r else None)
            for bk, r in results.items()
        },
    }
    p = out / 'burgers_basis_results.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {p}")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(results, config):
    """Formatted table of results (ASCII-only)."""
    print("\n" + "=" * 140)
    print("RESULTS SUMMARY")
    print("=" * 140)
    h = (f"{'Basis':<25} | {'Coeffs':>7} | {'Iters':>6} | {'Status':>10} | "
         f"{'Total Loss':>12} | {'PDE Loss':>12} | {'IC Loss':>12} | "
         f"{'BC Loss':>12} | {'Time':>8}")
    print(h)
    print("-" * len(h))
    for bk, r in results.items():
        if r is None:
            print(f"{BASIS_CONFIGS[bk]['short_label']:<25} | "
                  f"{'FAILED':>7}")
            continue
        s = r['summary']
        st = ('Converged' if s['converged']
              else ('Stagnated' if s.get('stagnated') else 'Max iters'))
        print(f"{BASIS_CONFIGS[bk]['short_label']:<25} | "
              f"{s['n_coefficients']:>7} | {s['n_iterations']:>6} | "
              f"{st:>10} | {s['final_loss']:>12.4e} | "
              f"{s['final_pde_loss']:>12.4e} | {s['final_ic_loss']:>12.4e} | "
              f"{s['final_bc_loss']:>12.4e} | {s['total_time']:>7.3f}s")
    print("=" * 140 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Burgers LiL-Q basis comparison')
    ap.add_argument('--N', type=int, default=25,
                    help='Basis size per direction (default: 25)')
    ap.add_argument('--viscosity', type=float, default=0.1,
                    help='Viscosity nu (default: 0.1)')
    ap.add_argument('--T', type=float, default=1.0,
                    help='Final time (default: 1.0)')
    ap.add_argument('--R-tol', type=float, default=1e-5,
                    help='Convergence tolerance (default: 1e-5)')
    ap.add_argument('--max-iters', type=int, default=50,
                    help='Maximum quasilinear iterations (default: 50)')
    ap.add_argument('--output-dir', type=str, default='results/burgers_basis_comparison',
                    help='Output directory (default: results/burgers_basis_comparison)')
    ap.add_argument('--bases', type=str, nargs='+', default=None,
                    help='Subset of bases to run (default: all)')
    args = ap.parse_args()

    config = ComparisonConfig(
        viscosity=args.viscosity,
        T_final=args.T,
        N_x=args.N,
        N_t=args.N,
        R_tol=args.R_tol,
        max_quasi_iters=args.max_iters,
        output_dir=args.output_dir,
    )

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results, pts = run_all_bases(
        config, basis_keys=args.bases, verbose=True)
    print_summary(results, config)
    save_results(results, config, out)

    print(f"\nAll outputs: {out}\nDONE.")


if __name__ == '__main__':
    main()
