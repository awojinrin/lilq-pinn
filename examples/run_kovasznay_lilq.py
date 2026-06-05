"""
Quick Start: Kovasznay Flow — LiL-Q Solver
=============================================

Solves steady 2D incompressible Navier-Stokes (Kovasznay flow)
using the multi-field LiL-Q solver.

Usage::

    python examples/run_kovasznay_lilq.py
    python examples/run_kovasznay_lilq.py --N 15 --Re 40
"""

import sys
import os
import argparse
import time

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from problems.kovasznay import KovasznayConfig, solve_kovasznay


def main():
    parser = argparse.ArgumentParser(description="Kovasznay LiL-Q quick start")
    parser.add_argument('--N', type=int, default=15)
    parser.add_argument('--Re', type=float, default=40.0)
    parser.add_argument('--basis', type=str, default='chebyshev')
    args = parser.parse_args()

    config = KovasznayConfig(
        N_x=args.N, N_y=args.N,
        Re=args.Re,
        basis_type=args.basis,
        k_ratio=4,
        max_iter=20,
        tol=1e-9,
    )

    print(f"Kovasznay LiL-Q: N={args.N}, Re={args.Re}, basis={args.basis}")
    print("-" * 50)

    t0 = time.time()
    result = solve_kovasznay(config, verbose=True)
    elapsed = time.time() - t0

    print(f"\nOuter iters:    {result['n_outer_iters']}")
    print(f"Elapsed time:   {elapsed:.3f}s")
    print(f"rel L2 (u):     {result['rel_l2_u']:.3e}")
    print(f"rel L2 (v):     {result['rel_l2_v']:.3e}")
    print(f"rel L2 (p):     {result['rel_l2_p']:.3e}")


if __name__ == "__main__":
    main()
