"""
Package 2, item 8 (P2-10): a manufactured elasticity solution outside the span of the basis
==========================================================================================

The advisor's instructions of 4 October 2026, Section 11, and his reply of
5 October (Section 3.2). The paper's elasticity case (Section 6.5) has its
exact solution in the span of the basis, so it only checks assembly and
solve. This runs the same operator, domain, bases, boundary-condition types
and P (50 .. 1,250) with two manufactured solutions outside the span of both
bases (delta_P > 0), one solve per P each (``solve_elasticity`` with the
manufactured physics; the paper's case is unchanged, bit for bit):

- ``compatible`` (the reply's replacement; the result):

      u_x = e^{cos(pi x)} e^{cos(2 pi y)} sin(pi y) / 10,
      u_y = sin(pi x) e^{cos(2 pi x)} y e^y / 20

  (``CompatibleManufacturedElasticityPhysics``). It matches the bases'
  symmetries and meets every boundary condition with the paper's data types
  (the lateral faces traction-free), so delta_P falls geometrically with P
  and error / delta_P is the quantity of interest.
- ``specified`` (the instructions' solution, kept as a negative result):

      u_x = y (1 - y) e^{xy} / 10,    u_y = x (1 - x) y e^{x+y} / 20

  (``ManufacturedElasticityPhysics``). It breaks the symmetry built into the
  bases (below), and its errors stall while delta_P falls.

Per solution and P:
- the relative L2 errors of u_x and u_y against the exact solution on the
  200 x 200 test grid, and the stresses';
- delta_P, the relative discrete L2 distance of the exact solution to the
  trial space on the same grid (a least-squares fit per field), per field and
  for (u_x, u_y) together;
- the ratio error / delta_P;
- kappa and the numerical rank of the system (SVD; ``iterations.csv``).

Also per P, ``lateral_sxx_trial_max`` and ``lateral_sxx_required_max``: the
largest |sigma_xx| any trial function can have on the lateral faces, and the
largest the manufactured data asks for (zero for ``compatible``). With the
paper's bases (u_x: cosine in x; u_y: sine in x), every trial function has d(u_x)/dx = 0 and
d(u_y)/dy = 0 on x = 0 and 1, so sigma_xx is zero there for every
coefficient vector: the lateral traction rows are zero rows, and the
``specified`` (non-zero) traction cannot be met. The same parity makes the
second derivatives the equations are collocated on unresolvable near the
faces (DECISIONS.md, Package 2, Stage 2, batch 2a).

Writes ``P2_10_elasticity_manufactured/rows.csv`` (column ``solution``) and,
per solution and P, ``<solution>/P<P>/`` with ``run.json`` and
``iterations.csv``.

Usage::

    python experiments/p2_10_elasticity_manufactured.py --out <stage root>
"""

import argparse
import csv
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from experiments.run_elasticity import DEFAULT_N_VALUES, K_RATIO
from lilq.iteration_log import IterationLogger
from lilq.provenance import save_provenance
from problems.elasticity import (TEST_GRID, CompatibleManufacturedElasticityPhysics, ElasticityConfig,
                                 ManufacturedElasticityPhysics, solve_elasticity)

SOLUTIONS = {'compatible': CompatibleManufacturedElasticityPhysics, 'specified': ManufacturedElasticityPhysics}


def best_approximation(basis, exact, xs, ys):
    """The trial space's least-squares fit of ``exact`` on the tensor grid
    xs x ys: ``(||fit - exact||, ||exact||)``."""
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    x, y, f = X.ravel(), Y.ravel(), exact(X, Y).ravel()
    B = basis.evaluate(x, y)
    c = np.linalg.lstsq(B, f, rcond=None)[0]
    return float(np.linalg.norm(B @ c - f)), float(np.linalg.norm(f))


def lateral_traction(basis_u, basis_v, physics, n=101):
    """``(max |sigma_xx| of any unit trial function, max |required sigma_xx|)``
    on the lateral faces: the largest entry of the sigma_xx rows, and of the data."""
    y = np.linspace(*physics.y_domain, n)
    trial, required = 0.0, 0.0
    for xe in physics.x_domain:
        x = np.full_like(y, xe)
        rows = np.hstack([physics.C11 * basis_u.derivative(x, y, dx=1, dy=0),
                          physics.C12 * basis_v.derivative(x, y, dx=0, dy=1)])
        trial = max(trial, float(np.abs(rows).max()))
        required = max(required, float(np.abs(physics.traction_lateral_sxx(x, y)).max()))
    return trial, required


def one(N, out_dir, solution='compatible'):
    config = ElasticityConfig(N_x=N, N_y=N, k_ratio=K_RATIO)
    physics = SOLUTIONS[solution](config)
    run_dir = Path(out_dir) / solution / f'P{2 * N * N}'
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    r = solve_elasticity(config, verbose=False, iteration_logger=logger, run_json_path=run_dir / 'run.json',
                         physics=physics)
    logger.to_csv(run_dir / 'iterations.csv')
    xs, ys = np.linspace(*config.x_domain, TEST_GRID[0]), np.linspace(*config.y_domain, TEST_GRID[1])
    du, nu = best_approximation(r['basis_u'], physics.exact_ux, xs, ys)
    dv, nv = best_approximation(r['basis_v'], physics.exact_uy, xs, ys)
    eu, ev = r['rel_l2_ux'], r['rel_l2_uy']
    e_both = float(np.sqrt((eu * nu) ** 2 + (ev * nv) ** 2) / np.hypot(nu, nv))
    d_both = float(np.hypot(du, dv) / np.hypot(nu, nv))
    row = logger.rows[0]
    trial_sxx, required_sxx = lateral_traction(r['basis_u'], r['basis_v'], physics)
    return {'solution': solution, 'N': N, 'P': r['n_params'], 'P_u': r['basis_u'].n_basis, 'P_v': r['basis_v'].n_basis,
            'eps_ux': eu, 'eps_uy': ev, 'eps_both': e_both,
            'delta_ux': du / nu, 'delta_uy': dv / nv, 'delta_both': d_both,
            'ratio_ux': eu / (du / nu), 'ratio_uy': ev / (dv / nv), 'ratio_both': e_both / d_both,
            'kappa': row['kappa'], 'kappa_method': row['kappa_method'], 'rank_svd': row['num_rank_svd'],
            'rank_gelsy': row['num_rank_gelsy'], 'eps_sxx': r['rel_l2_sxx'], 'eps_syy': r['rel_l2_syy'],
            'eps_sxy': r['rel_l2_sxy'], 'pde_mse': r['pde_mse'],
            'lateral_sxx_trial_max': trial_sxx, 'lateral_sxx_required_max': required_sxx}


def run(out_root, sizes=DEFAULT_N_VALUES, solutions=tuple(SOLUTIONS)):
    out = Path(out_root) / 'P2_10_elasticity_manufactured'
    rows = [one(N, out, solution) for solution in solutions for N in sizes]
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_provenance(out)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 2, item 8: the manufactured elasticity solution.")
    ap.add_argument('--out', required=True, help='the stage root')
    ap.add_argument('--solutions', nargs='+', choices=list(SOLUTIONS), default=list(SOLUTIONS))
    args = ap.parse_args(argv)
    for r in run(args.out, solutions=args.solutions):
        print(f"{r['solution']:10s} P = {r['P']:5d}: error u_x {r['eps_ux']:.2e} u_y {r['eps_uy']:.2e} | delta_P {r['delta_ux']:.2e} "
              f"{r['delta_uy']:.2e} | ratio {r['ratio_ux']:.2f} {r['ratio_uy']:.2f} | kappa {r['kappa']:.1e} "
              f"rank {r['rank_svd']} of {r['P']}")


if __name__ == '__main__':
    main()
