"""
Quick Start: Burgers Equation — LiL-Q Solver
===============================================

Solves u_t + u*u_x - nu*u_xx = 0 on [-1,1] x [0,T] using LiL-Q.

Usage::

    python examples/run_burgers_lilq.py
    python examples/run_burgers_lilq.py --N 15
"""

import sys
import os
import argparse
import time

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from problems.burgers import BurgersConfig, BurgersOptConfig, run_lil_q

# Per-N target losses from the original experiments
TARGET_LOSSES = {5: 6e-2, 10: 1e-3, 15: 2e-5, 20: 1e-7, 25: 5e-9}


def main():
    parser = argparse.ArgumentParser(description="Burgers LiL-Q quick start")
    parser.add_argument('--N', type=int, default=15)
    parser.add_argument('--basis', type=str, default='sin_fourier')
    args = parser.parse_args()

    R_tol = TARGET_LOSSES.get(args.N, 1e-4)

    config = BurgersConfig(
        N_x=args.N, N_t=args.N,
        viscosity=0.1,
        basis_type=args.basis,
        k_ratio=10,
    )
    opt = BurgersOptConfig(
        max_quasi_iters_lil=20,
        R_tol=R_tol,
    )

    print(f"Burgers LiL-Q: N={args.N}, P={args.N**2} DOFs, "
          f"nu=0.1, basis={args.basis}, R_tol={R_tol:.1e}")
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
