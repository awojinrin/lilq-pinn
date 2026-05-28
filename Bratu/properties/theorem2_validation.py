"""
Numerical validation of Theorem 2 (residual bounds with Gamma ||delta u||)
for the Bratu LiL-Q experiment.

Validates at iterate u^(n) (n >= 1) using r^(n-1), J_(n-1), delta u^(n-1) from the
previous quasilinear step — matching the manuscript's same-iterate statement.

Usage:
    python theorem2_validation.py --N 5 10 15
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import scipy.linalg

# Injection for shared and relative packages
_PROP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_PROP_DIR, '../..')))
sys.path.insert(0, _PROP_DIR)

from lilq_shared.theorem2 import (
    Theorem2Tracker,
    apply_theorem2_at_iterate,
    check_theorem2_violations,
    save_theorem2_json,
    summarize_gamma,
)

from typing import Dict
from Bratu.bratu_core import (
    BratuPhysics,
    DiscretizationConfig,
    OptimizationConfig,
    create_fourier_basis_2d,
)

def generate_collocation_points(physics: BratuPhysics,
                                disc: DiscretizationConfig) -> Dict:
    """Generate PDE interior and BC collocation points."""
    np.random.seed(disc.seed)

    n_coefs = disc.N_x * disc.N_y
    ratios = disc.collocation_ratios
    normalized_ratios = [r / sum(ratios) for r in ratios]

    x_min, x_max = physics.x_domain
    y_min, y_max = physics.y_domain

    n_pde = disc.k_ratio * normalized_ratios[0] * n_coefs
    n_pde_dim = int(np.ceil(np.sqrt(n_pde)))

    eps = 1e-5
    if disc.domain_sampling == "random":
        x_pde = np.random.uniform(x_min + eps, x_max - eps,
                                  n_pde_dim).astype(np.float64)
        y_pde = np.random.uniform(y_min + eps, y_max - eps,
                                  n_pde_dim).astype(np.float64)
    else:
        x_pde = np.linspace(x_min + eps, x_max - eps,
                            n_pde_dim, dtype=np.float64)
        y_pde = np.linspace(y_min + eps, y_max - eps,
                            n_pde_dim, dtype=np.float64)

    pts_pde = np.stack(
        np.meshgrid(x_pde, y_pde), axis=-1
    ).reshape([-1, 2]).astype(np.float64)

    n_bc = int(np.ceil(disc.k_ratio * normalized_ratios[1] * n_coefs / 4))
    if disc.boundary_sampling == "random":
        x_bc = np.random.uniform(x_min, x_max, n_bc).astype(np.float64)
        y_bc = np.random.uniform(y_min, y_max, n_bc).astype(np.float64)
    else:
        x_bc = np.linspace(x_min, x_max, n_bc, dtype=np.float64)
        y_bc = np.linspace(y_min, y_max, n_bc, dtype=np.float64)

    pts_bc = np.vstack([
        np.column_stack([np.zeros(n_bc, dtype=np.float64), y_bc]),
        np.column_stack([np.ones(n_bc, dtype=np.float64), y_bc]),
        np.column_stack([x_bc, np.zeros(n_bc, dtype=np.float64)]),
        np.column_stack([x_bc, np.ones(n_bc, dtype=np.float64)])
    ])

    return {'pts_pde': pts_pde, 'pts_bc': pts_bc,
            'n_pde': pts_pde.shape[0], 'n_bc': pts_bc.shape[0]}

def bratu_remainder(lambda_: float, u_k: np.ndarray,
                    delta_u_k: np.ndarray) -> np.ndarray:
    """Compute the exact Taylor remainder for the Bratu nonlinearity."""
    return lambda_ * np.exp(u_k) * (np.expm1(delta_u_k) - delta_u_k)


def solve_lilq_theorem2(
    physics: BratuPhysics,
    disc: DiscretizationConfig,
    opt: OptimizationConfig,
    gamma: float | None = None,
    verbose: bool = True,
) -> tuple[object, Theorem2Tracker, dict]:
    """LiL-Q with Theorem 2 tracking (same-iterate indexing)."""
    np.random.seed(disc.seed)

    if verbose:
        print("=" * 70)
        print("THEOREM 2 VALIDATION — BRATU (LiL-Q)")
        print("=" * 70)

    solver = create_fourier_basis_2d(
        disc.N_x, disc.N_y, physics.x_domain, physics.y_domain,
        mode_x="both", mode_y="both",
    )
    n_coefs = solver.n_basis

    beta = np.random.uniform(-1.0, 1.0, n_coefs).astype(np.float64) * 1e-4
    points = generate_collocation_points(physics, disc)
    pts_pde, pts_bc = points["pts_pde"], points["pts_bc"]
    M_pde, M_bc = points["n_pde"], points["n_bc"]

    x_pde, y_pde = pts_pde[:, 0], pts_pde[:, 1]
    x_bc, y_bc = pts_bc[:, 0], pts_bc[:, 1]

    A_u = solver.evaluate(x_pde, y_pde)
    A_uxx = solver.derivative(x_pde, y_pde, dx=2, dy=0)
    A_uyy = solver.derivative(x_pde, y_pde, dx=0, dy=2)
    A_bc = solver.evaluate(x_bc, y_bc)
    A_lap = A_uxx + A_uyy

    weight_pde = np.sqrt(opt.lambda_pde / M_pde)
    weight_bc = np.sqrt(opt.lambda_bc / M_bc)

    tracker = Theorem2Tracker()

    def nonlinear_vec(b: np.ndarray) -> np.ndarray:
        u = A_u @ b
        N_pde = A_lap @ b + physics.lambda_ * np.exp(u)
        N_bc = A_bc @ b
        return np.concatenate([weight_pde * N_pde, weight_bc * N_bc])

    # n = 0
    N0 = nonlinear_vec(beta)
    tracker.nonlinear_res.append(float(np.linalg.norm(N0)))

    prev_r_full = None
    prev_J_full = None
    prev_delta_beta = None
    prev_delta_u_norm = None
    prev_w_norm = None

    eps_L, eps_U = np.inf, 0.0
    gamma_used = float(gamma) if gamma is not None else 0.0

    for k in range(opt.max_quasi_iters):
        # --- Theorem 2 at iterate u^(k) before solve (k >= 1) ---
        if k >= 1 and prev_r_full is not None:
            N_k = nonlinear_vec(beta)
            apply_theorem2_at_iterate(
                tracker,
                N_vec=N_k,
                r_vec=prev_r_full,
                J_mat=prev_J_full,
                delta_beta=prev_delta_beta,
                delta_u_norm=prev_delta_u_norm,
                eps_L=eps_L,
                eps_U=eps_U,
                gamma=gamma_used,
                predictive_tight_upper=(
                    float(np.linalg.norm(prev_r_full)) + prev_w_norm
                    if prev_w_norm is not None else None
                ),
            )

        beta_k = beta.copy()
        u_k = A_u @ beta_k
        exp_u_k = np.exp(u_k)

        J_pde = A_lap + physics.lambda_ * (exp_u_k.reshape(-1, 1) * A_u)
        b_pde = physics.lambda_ * exp_u_k * (u_k - 1)
        J_full = np.vstack([weight_pde * J_pde, weight_bc * A_bc])
        b_full = np.concatenate([weight_pde * b_pde, weight_bc * np.zeros(M_bc)])

        try:
            sig_max = float(scipy.linalg.svdvals(J_full)[0])
        except Exception:
            sig_max = 1.0

        beta_kp1 = scipy.linalg.lstsq(J_full, b_full, lapack_driver="gelsy")[0]
        beta = beta_kp1

        delta_beta = beta_kp1 - beta_k
        delta_u = A_u @ delta_beta
        delta_u_norm = float(np.linalg.norm(delta_u))

        r_pde = J_pde @ beta_kp1 - b_pde
        r_bc = A_bc @ beta_kp1
        r_full = np.concatenate([weight_pde * r_pde, weight_bc * r_bc])

        w_pde = bratu_remainder(physics.lambda_, u_k, delta_u)
        w_full = np.concatenate([weight_pde * w_pde, np.zeros(M_bc)])
        prev_w_norm = float(np.linalg.norm(w_full))

        r_norm = float(np.linalg.norm(r_full))
        eps_L = min(eps_L, r_norm)
        eps_U = max(eps_U, r_norm)

        if gamma is None:
            du_safe = max(delta_u_norm, 1e-30)
            ratio = float(np.linalg.norm(J_full @ delta_beta)) / du_safe
            gamma_used = max(gamma_used, ratio)

        prev_r_full = r_full.copy()
        prev_J_full = J_full.copy()
        prev_delta_beta = delta_beta.copy()
        prev_delta_u_norm = delta_u_norm
        tracker.sigma_max_J.append(sig_max)

        if float(np.mean((A_lap @ beta + physics.lambda_ * np.exp(A_u @ beta)) ** 2)) < opt.R_tol:
            if verbose:
                print(f"  Converged at step k={k}")
            break

    # Final iterate n = K+1
    if prev_r_full is not None:
        N_final = nonlinear_vec(beta)
        apply_theorem2_at_iterate(
            tracker,
            N_vec=N_final,
            r_vec=prev_r_full,
            J_mat=prev_J_full,
            delta_beta=prev_delta_beta,
            delta_u_norm=prev_delta_u_norm,
            eps_L=eps_L,
            eps_U=eps_U,
            gamma=gamma_used,
            predictive_tight_upper=(
                float(np.linalg.norm(prev_r_full)) + prev_w_norm
                if prev_w_norm is not None else None
            ),
        )

    gamma_info = summarize_gamma(
        np.array(tracker.nprime_du_norm),
        np.array(tracker.delta_u_norm),
        np.array(tracker.sigma_max_J),
    )
    gamma_info["gamma_used_in_bounds"] = float(gamma_used)

    violations = check_theorem2_violations(tracker)

    summary = {
        "problem": "Bratu",
        "n_params": n_coefs,
        "eps_L": float(eps_L),
        "eps_U": float(eps_U),
        "gamma_used": float(gamma_used),
        "final_nonlinear_res": float(tracker.nonlinear_res[-1]),
        "final_identity_rel_err": float(tracker.identity_rel_err[-1]),
        "thm2_lower_violations": len(violations["lower"]),
        "thm2_upper_violations": len(violations["upper"]),
    }

    return solver, tracker, summary, violations, gamma_info


def run_experiment(
    N: int,
    output_dir: str | Path | None = None,
    gamma: float | None = None,
    verbose: bool = True,
):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "bratu_experiments_fourier"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics = BratuPhysics(lambda_=6.2)
    disc = DiscretizationConfig(N_x=N, N_y=N)
    opt = OptimizationConfig(max_quasi_iters=10, R_tol=1e-14)

    solver, tracker, summary, violations, gamma_info = solve_lilq_theorem2(
        physics, disc, opt, gamma=gamma, verbose=verbose
    )

    print(f"\n  THEOREM 2 SUMMARY (N={N}):")
    print(f"    eps_L = {summary['eps_L']:.6e}, eps_U = {summary['eps_U']:.6e}")
    print(f"    Gamma used in bounds = {summary['gamma_used']:.6e}")
    print(f"    Final ||N|| = {summary['final_nonlinear_res']:.6e}")
    print(f"    Theorem 2 band violations: lower={summary['thm2_lower_violations']}, "
          f"upper={summary['thm2_upper_violations']}")

    save_theorem2_json(
        output_dir / f"theorem2_bratu_N{N}.json",
        tracker, summary, violations, gamma_info,
    )

    return solver, tracker, summary


def main():
    parser = argparse.ArgumentParser(description="Theorem 2 validation — Bratu")
    parser.add_argument("--N", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--gamma", type=float, default=None,
                        help="Fixed Gamma for (A4). If omitted, uses running max ||N'du||/||du||.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    for n in args.N:
        run_experiment(n, output_dir=args.output_dir, gamma=args.gamma)

    print("\nDONE")


if __name__ == "__main__":
    main()
