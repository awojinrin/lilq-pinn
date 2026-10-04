"""
Package 2, Section 6.2: the Burgers reference (Cole-Hopf) and its checks
========================================================================

``burgers``: the Cole-Hopf solution of the Burgers problem of Section 6.3
(nu = 0.1, u(x, 0) = -sin(pi x), u(+-1, t) = 0) on the 201 x 201 test grid of
``problems.burgers`` ([-1, 1] x [0, 1]), by adaptive quadrature of the
Cole-Hopf integrals (``lilq.references.cole_hopf``, relative tolerance 1e-13).
Two checks are recorded with it:

* **Basdevant et al. (1986):** u_x(0, 1.6037/pi) = -152.00516 at nu = 0.01/pi
  (check C4: the integrator);
* **an independent evaluation:** the cosine series of the Cole-Hopf transform
  (``cole_hopf_series``) on the whole grid at nu = 0.1.

Writes ``reference/burgers_cole_hopf.npz`` (x, t, u, meta) and
``reference/burgers_cole_hopf_checks.json``. The Bratu reference is written by
``experiments/p2_3_classical.py reference``.

Usage::

    python experiments/p2_references.py burgers --out <package2_results>
"""

import argparse
import json
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np

from lilq import references as R
from lilq.source_lock import current_commit


def burgers(out_dir):
    from problems.burgers import TEST_GRID, BurgersConfig
    cfg = BurgersConfig()
    x = np.linspace(*cfg.x_domain, TEST_GRID[0])
    t = np.linspace(0.0, cfg.T_final, TEST_GRID[1])
    b = R.BASDEVANT
    ux = float(R.cole_hopf_ux(np.array([b['x']]), b['t'], b['nu'])[0])
    u = R.burgers_reference(x, t, cfg.viscosity)
    series = np.stack([R.cole_hopf_series(x, tj, cfg.viscosity) for tj in t], axis=1)
    checks = {
        'basdevant': {'nu': b['nu'], 'x': b['x'], 't': b['t'], 'u_x': ux, 'published': b['u_x'],
                      'abs_difference': abs(ux - b['u_x']),
                      'agrees_to_published_digits': f"{ux:.5f}" == f"{b['u_x']:.5f}"},
        'series': {'max_abs_difference': float(np.abs(u - series).max()),
                   'rel_l2_difference': R.rel_l2(series, u)},
        'boundary_max_abs': float(max(np.abs(u[0]).max(), np.abs(u[-1]).max())),
        'initial_condition_max_abs_difference': float(np.abs(u[:, 0] + np.sin(np.pi * x)).max()),
    }
    meta = {'problem': 'Burgers, u_t + u u_x = nu u_xx on (-1,1) x (0,1], u(x,0) = -sin(pi x), u(+-1,t) = 0',
            'nu': cfg.viscosity, 'method': 'Cole-Hopf, adaptive quadrature (scipy quad_vec), relative tolerance 1e-13',
            'grid': '201 x 201 on [-1,1] x [0,1], u[i, j] at (x[i], t[j])', 'checks': checks, 'commit': current_commit()}
    ref_dir = Path(out_dir) / 'reference'
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ref_dir / 'burgers_cole_hopf.npz', x=x, t=t, u=u, meta=np.array(json.dumps(meta)))
    (ref_dir / 'burgers_cole_hopf_checks.json').write_text(json.dumps(meta, indent=2))
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 2: the Burgers Cole-Hopf reference.")
    ap.add_argument('stage', choices=('burgers',))
    ap.add_argument('--out', required=True, help='package2_results')
    args = ap.parse_args(argv)
    c = burgers(args.out)
    print(f"Basdevant: u_x = {c['basdevant']['u_x']:.7f} against {c['basdevant']['published']} "
          f"({'agrees' if c['basdevant']['agrees_to_published_digits'] else 'DIFFERS'})")
    print(f"series: max |difference| {c['series']['max_abs_difference']:.1e}; boundary {c['boundary_max_abs']:.1e}")


if __name__ == '__main__':
    main()
