"""
Quick Start: Bratu Equation — LiL-Q Solver
=============================================

Solves the 2D Bratu equation u_xx + u_yy + lambda*exp(u) = 0 on [0,1]^2
using the LiL-Q (quasilinear basis expansion) method.

Usage::

    python examples/run_bratu_lilq.py
    python examples/run_bratu_lilq.py --N 10 --basis fourier
"""

import sys
import os
import argparse
import time

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from problems.bratu import (
    BratuConfig, BratuOptConfig, run_lil_q,
    evaluate_lil_solution, evaluate_lil_residual,
)

# Per-N target losses from the original experiments
TARGET_LOSSES = {5: 2.5e-1, 10: 1e-4, 15: 2.5e-7}


def main():
    parser = argparse.ArgumentParser(description="Bratu LiL-Q quick start")
    parser.add_argument('--N', type=int, default=15)
    parser.add_argument('--basis', type=str, default='fourier',
                        choices=['chebyshev', 'fourier', 'sin_sin'])
    parser.add_argument('--lambda_', type=float, default=6.2)
    args = parser.parse_args()

    R_tol = TARGET_LOSSES.get(args.N, 1e-4)

    config = BratuConfig(
        N_x=args.N, N_y=args.N,
        lambda_=args.lambda_,
        basis_type=args.basis,
        k_ratio=10,
    )
    opt = BratuOptConfig(
        max_quasi_iters_lil=25,
        R_tol=R_tol,
    )

    print(f"Bratu LiL-Q: N={args.N}, P={args.N**2} DOFs, "
          f"lambda={args.lambda_}, basis={args.basis}, R_tol={R_tol:.1e}")
    print("-" * 50)

    t0 = time.time()
    basis, coefficients, metrics, summary = run_lil_q(config, opt, verbose=True)
    elapsed = time.time() - t0

    print(f"\nFinal loss:     {summary['final_loss']:.4e}")
    print(f"Outer iters:    {summary['total_iterations']}")
    print(f"Elapsed time:   {elapsed:.3f}s")
    print(f"Converged:      {summary['converged']}")


if __name__ == "__main__":
    main()
