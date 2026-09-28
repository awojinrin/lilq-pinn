"""
Saved models: every trained solution of a production run, reloadable
====================================================================

Every run of the cluster jobs leaves its trained model next to its logs, so
fields, error maps and figures can be recomputed later without training
again. Two kinds of file, both written when the run ends (outside every
timed phase) and before the run's completion marker, so a run marked
complete always has its model:

``solution.pt`` (LiL-Q, LiL-N)
    ``{'format', 'fields': {name: {'basis', 'coefficients'}}, 'config',
    'opt', 'extra'}``: the basis objects, the coefficient vectors, and the
    run's configuration objects. :func:`load_solution`; evaluate a field with
    :func:`evaluate_field`.
``network.pt`` (NiL-N, NiL-Q, the Darcy PINN)
    ``{'format', 'model', 'state_dict', 'config', 'opt', 'extra'}``: the
    network itself (a CPU copy), its ``state_dict``, and the configuration
    objects. :func:`load_network`.

The Component A baselines keep their own files (``baselines/``: F1's
``model.pt`` state dict and F2's ``theta.pt`` parameter vector, rebuilt from
the configuration in ``run.json``; :func:`load_f1`, :func:`load_f2`).

Long runs also checkpoint during training (:func:`save_checkpoint`,
:func:`load_checkpoint`; the Darcy PINN), so a crash or a walltime kill
resumes from the last checkpoint instead of starting over.

All files are ``torch.save`` pickles of this repository's classes: load them
with the code at the commit recorded in the run's ``hardware.json``. They
load on a machine without a GPU (``map_location='cpu'``). Every write goes
to a temporary file first and is then renamed, so a crash never leaves a
truncated file.
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

FORMAT_VERSION = 1
SOLUTION_FILE = 'solution.pt'
NETWORK_FILE = 'network.pt'

PathLike = Union[str, Path]


def _atomic_save(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def _load(path: Path):
    return torch.load(path, map_location='cpu', weights_only=False)


def _resolve(path: PathLike, default_name: str) -> Path:
    path = Path(path)
    return path / default_name if path.is_dir() else path


def save_solution(run_dir: PathLike, fields: Dict[str, Any], config=None, opt=None,
                  extra: Optional[Dict[str, Any]] = None, filename: str = SOLUTION_FILE) -> Path:
    """Save a LiL solution. ``fields`` maps each field's name to
    ``(basis, coefficients)`` -- one entry (``'u'``) for a scalar problem."""
    payload = {
        'format': FORMAT_VERSION,
        'fields': {name: {'basis': basis, 'coefficients': np.asarray(coef, dtype=np.float64).copy()}
                   for name, (basis, coef) in fields.items()},
        'config': config, 'opt': opt, 'extra': extra or {},
    }
    return _atomic_save(payload, Path(run_dir) / filename)


def load_solution(path: PathLike) -> Dict[str, Any]:
    """A saved LiL solution (the run directory or the file itself)."""
    return _load(_resolve(path, SOLUTION_FILE))


def evaluate_field(solution: Dict[str, Any], name: str, *coords) -> np.ndarray:
    """Field ``name`` of a loaded solution at the points ``coords`` (one
    array per coordinate, all the same shape)."""
    entry = solution['fields'][name]
    shape = np.shape(coords[0])
    flat = [np.ravel(c).astype(np.float64) for c in coords]
    return (entry['basis'].evaluate(*flat) @ entry['coefficients']).reshape(shape)


def save_network(run_dir: PathLike, model: torch.nn.Module, config=None, opt=None,
                 extra: Optional[Dict[str, Any]] = None, filename: str = NETWORK_FILE) -> Path:
    """Save a trained network (a CPU copy; the original is not moved)."""
    cpu_model = copy.deepcopy(model).cpu().eval()
    payload = {
        'format': FORMAT_VERSION, 'model': cpu_model,
        'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'config': config, 'opt': opt, 'extra': extra or {},
    }
    return _atomic_save(payload, Path(run_dir) / filename)


def load_network(path: PathLike) -> Dict[str, Any]:
    """A saved network (the run directory or the file itself), on the CPU
    and in eval mode, with its configuration."""
    return _load(_resolve(path, NETWORK_FILE))


def save_checkpoint(path: PathLike, state: Dict[str, Any]) -> Path:
    """A mid-training checkpoint (atomic; overwrites the previous one)."""
    return _atomic_save({'format': FORMAT_VERSION, **state}, Path(path))


def load_checkpoint(path: PathLike) -> Optional[Dict[str, Any]]:
    """The checkpoint at ``path``, or ``None`` when there is none."""
    path = Path(path)
    return _load(path) if path.exists() else None


def load_f1(run_dir: PathLike, device='cpu'):
    """Component A, F1: the trained :class:`baselines.f1_pinn.F1Model` of a
    run directory, rebuilt from its ``run.json`` configuration."""
    import json
    from baselines.f1_pinn import F1Model
    run_dir = Path(run_dir)
    run = json.loads((run_dir / 'run.json').read_text())
    c = run['config']
    state = torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=True)
    dtype = next(iter(state.values())).dtype
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = F1Model(c['width'], c['depth'], c['m'], c['sigma_ff'], c['trunk'], c['bc'])
    finally:
        torch.set_default_dtype(old)
    model.load_state_dict(state)
    return model.to(device).eval(), run


def load_f2(run_dir: PathLike, device='cpu'):
    """Component A, F2: ``(model, run)`` -- the
    :class:`baselines.lm_kovasznay.FourierMLP` of a run directory with its
    trained parameter vector loaded, and ``run.json``. The model gives the
    raw outputs; :func:`predict_f2` applies the hard boundary conditions."""
    import json
    from baselines.lm_kovasznay import FourierMLP
    run_dir = Path(run_dir)
    run = json.loads((run_dir / 'run.json').read_text())
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = FourierMLP(run['width'], run['depth'], run['m'], run['sigma_ff'], run['seed'])
    finally:
        torch.set_default_dtype(old)
    theta = torch.load(run_dir / 'theta.pt', map_location='cpu', weights_only=True)
    torch.nn.utils.vector_to_parameters(theta.to(torch.float64), model.parameters())
    return model.to(device).eval(), run


def predict_f2(model, xy: torch.Tensor) -> torch.Tensor:
    """(u, v, p) of a loaded F2 model at points ``xy`` (N x 2), with the
    hard boundary conditions of :mod:`baselines.lm_kovasznay` applied."""
    from baselines.lm_kovasznay import coons, ell, g_u, g_v
    with torch.no_grad():
        out = model(xy)
        x, y = xy[:, 0], xy[:, 1]
        L = ell(x, y)
        return torch.stack([coons(g_u, x, y) + L * out[:, 0], coons(g_v, x, y) + L * out[:, 1], out[:, 2]], 1)
