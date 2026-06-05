"""
Quick Start: Buckley-Leverett — LiL-Q Solver
===============================================

Solves the Buckley-Leverett conservation law using LiL-Q.
Supports both standard (N_g=0) and gravity-driven (N_g=-5) cases.

Usage::

    python examples/run_bl_lilq.py
    python examples/run_bl_lilq.py --N 16
    python examples/run_bl_lilq.py --gravity
"""

import sys
import os
import argparse
import time

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from problems.buckley_leverett import BLConfig, BLOptConfig, run_lil_q

# Per-N target losses from the original experiments
TARGET_LOSSES = {8: 8.5e-2, 16: 1.5e-2, 24: 7.5e-4, 32: 7.5e-5}


def main():
    parser = argparse.ArgumentParser(description="BL LiL-Q quick start")
    parser.add_argument('--N', type=int, default=16)
    parser.add_argument('--gravity', action='store_true',
                        help="Use gravity configuration (N_g=-5)")
    parser.add_argument('--basis', type=str, default='fourier')
    args = parser.parse_args()

    R_tol = TARGET_LOSSES.get(args.N, 1e-3)

    if args.gravity:
        config = BLConfig.with_gravity()
        config.N_x = args.N
        config.N_t = args.N
        config.basis_type = args.basis
        label = "BL-gravity"
    else:
        config = BLConfig(N_x=args.N, N_t=args.N, basis_type=args.basis)
        label = "BL (no gravity)"

    opt = BLOptConfig(max_quasi_iters_lil=50, R_tol=R_tol)

    print(f"{label} LiL-Q: N={args.N}, basis={args.basis}, R_tol={R_tol:.1e}")
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
