"""
Quick Start: Linear Elasticity — LiL-Q Solver
================================================

Solves 2D plane-strain linear elasticity (Haghighat et al., CMAME 2021)
using a single QR-factorisation step (linear PDE).

Usage::

    python examples/run_elasticity_lilq.py
    python examples/run_elasticity_lilq.py --N 20
"""

import sys
import os
import argparse
import time

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from problems.elasticity import ElasticityConfig, solve_elasticity


def main():
    parser = argparse.ArgumentParser(description="Elasticity LiL-Q quick start")
    parser.add_argument('--N', type=int, default=15)
    args = parser.parse_args()

    config = ElasticityConfig(N_x=args.N, N_y=args.N, k_ratio=10)

    print(f"Elasticity LiL-Q: N={args.N}, P={args.N**2} per field")
    print("-" * 50)

    t0 = time.time()
    result = solve_elasticity(config, verbose=True)
    elapsed = time.time() - t0

    print(f"\nDOFs:           {result['n_params']}")
    print(f"QR time:        {result['solve_time_qr']:.4f}s")
    print(f"Total time:     {elapsed:.3f}s")
    print(f"rel L2 (ux):    {result['rel_l2_ux']:.3e}")
    print(f"rel L2 (uy):    {result['rel_l2_uy']:.3e}")


if __name__ == "__main__":
    main()
