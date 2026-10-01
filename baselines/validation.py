"""
Validation residual of a Component A model (the advisor's reply to wave 2, 1 October 2026)
=========================================================================================

Selecting by final training loss chose, in both families, the configuration
that generalizes worst (wave 2): a network that fits its collocation points
without learning the solution. Component A now selects by the residual on
points no run trained on -- F1's unweighted loss
(:func:`baselines.f1_pinn.unweighted_loss`) evaluated on a fixed fresh set:

* the x-momentum, y-momentum and continuity mean squares on 20,000 uniform
  interior points (``torch.Generator().manual_seed(20261001)``, drawn on the
  CPU in float64, so the points do not depend on the device), in chunks of
  2,500;
* for F1 runs with soft boundary conditions, the boundary mean square of u
  and v on 400 points per face (``f1_pinn.boundary_points``); hard-BC F1
  runs and every F2 run satisfy the boundary data exactly and have no such
  term.

No exact solution is used beyond the boundary data, which the training uses
too. This is the design of the wave-2 analysis
(``results/wave2/analysis/validation_residual.py`` on branch
``wave2-results``), which the advisor adopted as the protocol. On wave 2's
saved models it reproduces that analysis's values bit for bit on the GPU and
to round-off on the CPU (DECISIONS.md, 2026-10-01).
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

import baselines.f1_pinn as f1
from baselines.lm_kovasznay import coons, ell, g_u, g_v

SEED = 20261001
N_INTERIOR = 20000
N_BC_PER_FACE = 400
CHUNK = 2500
PROTOCOL = {'criterion': "F1's unweighted loss on fresh points", 'seed': SEED, 'n_interior': N_INTERIOR,
            'n_boundary_per_face': N_BC_PER_FACE, 'interior_distribution': 'uniform on [-0.5, 1] x [-0.5, 1.5]',
            'boundary_term': 'soft-BC F1 runs only', 'dtype': 'float64'}
VALIDATION_FILE = 'validation.json'

_POINTS: Dict[str, tuple] = {}


def validation_points(device) -> tuple:
    """``(xy_interior, xy_boundary)`` of the protocol, on ``device``."""
    key = str(device)
    if key not in _POINTS:
        gen = torch.Generator().manual_seed(SEED)
        _POINTS[key] = (f1.interior_points(N_INTERIOR, gen, torch.float64, torch.device(device)),
                        f1.boundary_points(N_BC_PER_FACE, torch.float64, torch.device(device)))
    return _POINTS[key]


def residual_terms(fields_fn: Callable, soft_bc: bool, device) -> Dict[str, float]:
    """The protocol's mean squares for ``fields_fn: (N x 2) -> (N x 3)``."""
    xy_int, xy_bc = validation_points(device)
    sums = {'xmom': 0.0, 'ymom': 0.0, 'cont': 0.0}
    for chunk in xy_int.split(CHUNK):
        r1, r2, r3 = f1.ns_residuals(fields_fn, chunk)
        for k, r in zip(sums, (r1, r2, r3)):
            sums[k] += float((r.detach() ** 2).sum())
    terms = {k: v / len(xy_int) for k, v in sums.items()}
    if soft_bc:
        with torch.no_grad():
            out = fields_fn(xy_bc)
            ue, ve, _ = f1.exact(xy_bc[:, 0], xy_bc[:, 1])
            terms['bc'] = float(((out[:, 0] - ue) ** 2).mean() + ((out[:, 1] - ve) ** 2).mean())
    return terms


def f2_fields(model) -> Callable:
    """(u, v, p) of an F2 network with its hard boundary conditions, with
    gradients (``lm_kovasznay.predict_f2`` is the no-grad version)."""
    def fn(xy):
        out = model(xy)
        x, y = xy[:, 0], xy[:, 1]
        L = ell(x, y)
        return torch.stack([coons(g_u, x, y) + L * out[:, 0], coons(g_v, x, y) + L * out[:, 1], out[:, 2]], 1)
    return fn


def family_of(run_dir: Path, run: dict) -> str:
    family = (run.get('config') or {}).get('family') or Path(run_dir).name[:2]
    if family not in ('F1', 'F2'):
        raise ValueError(f"{run_dir}: cannot tell the family (F1 or F2)")
    return family


def compute(run_dir, device='cpu') -> dict:
    """The validation residual of the model saved in ``run_dir``:
    ``{'val_residual', 'val_terms', 'val_protocol'}``. A run without a saved
    model (it failed) gets ``inf``."""
    from lilq.saved_models import load_f1, load_f2
    run_dir = Path(run_dir)
    run = json.loads((run_dir / 'run.json').read_text())
    family = family_of(run_dir, run)
    if family == 'F1':
        if not (run_dir / 'model.pt').exists():
            return {'val_residual': math.inf, 'val_terms': None, 'val_protocol': PROTOCOL}
        model, _ = load_f1(run_dir, device=device)
        terms = residual_terms(model.double(), (run.get('config') or {}).get('bc') == 'soft', device)
    else:
        if not (run_dir / 'theta.pt').exists():
            return {'val_residual': math.inf, 'val_terms': None, 'val_protocol': PROTOCOL}
        model, _ = load_f2(run_dir, device=device)
        terms = residual_terms(f2_fields(model), False, device)
    total = sum(terms.values())
    return {'val_residual': total if math.isfinite(total) else math.inf, 'val_terms': terms,
            'val_protocol': PROTOCOL}


def ensure(run_dir, device='cpu', record_in_run_json: bool = False) -> float:
    """The run's validation residual, computed once. Where it is kept: in
    ``run.json`` for runs made by this code (``record_in_run_json``), and in
    ``validation.json`` beside it for older runs, whose ``run.json`` is left
    as it was written. Either is read back before computing again."""
    run_dir = Path(run_dir)
    run_path = run_dir / 'run.json'
    run = json.loads(run_path.read_text())
    if 'val_residual' in run:
        return float(run['val_residual'])
    side = run_dir / VALIDATION_FILE
    if side.exists():
        return float(json.loads(side.read_text())['val_residual'])
    result = compute(run_dir, device)
    if record_in_run_json:
        run.update(result)
        run_path.write_text(json.dumps(run, indent=2, default=str))
    else:
        side.write_text(json.dumps(result, indent=2, default=str))
    return float(result['val_residual'])
