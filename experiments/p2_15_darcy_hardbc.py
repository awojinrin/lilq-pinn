"""
Package 2, item 5 (P2-15): the Darcy network baseline with hard Dirichlet conditions
=====================================================================================

The advisor's instructions of 4 October 2026, Section 8, as his reply of
5 October (Section 1.1) settles them. The paper's Darcy NiL baseline (Table 15)
differs from LiL in the boundary treatment, which confounds the LiL/NiL gap.
This run removes that confound and changes nothing else. It differs from the
paper's NiL baseline in two things only:

1. **The pressure output is lifted.** The Dirichlet data hold identically:

       h* = y* + omega(y*) NN_h,   P = h* (P_top - P_bot) + P_bot,
       omega = 4 y* (1 - y*)   (unit maximum, zero on the top and bottom faces),

   with y* = y / L_y and y* the LiL lifting function (``solve_lilq_darcy``:
   P = (y* + h_tilde*) DELTA_P + P_BOTTOM).
2. **The optimizer is Levenberg-Marquardt** with item 2's F2 damping
   (``experiments.p2_8_lm_networks.lm``).

Everything else is the paper's run (``problems.darcy.DarcyPINN``):
- **The networks:** three, h*, u* and v*, each 2 hidden layers x 32 SiLU
  (3,555 parameters), float64, built after ``torch.manual_seed(seed)`` in the
  paper's order. That is PyTorch's default initialization, so the same
  weights as the paper's run for each seed.
- **Inputs and scales:** the normalized inputs and the output scales
  (u*, v* times V_SCALE).
- **The residuals,** at the 60 x 220 cell centres: Darcy-x, Darcy-y and
  continuity, in the paper's normalization, each a mean square weighted 50.
- **The lateral no-flow rows:** U = 0 at the 220 lateral cell-face centres
  of each lateral face, each face a mean square weighted 20. These are the
  paper's own lateral rows, so Section 8's -K dh/dn rows are not introduced
  (the reply, 1.1).
- **No Dirichlet rows.** The paper's top and bottom rows vanish identically
  under the lifting (P = P_bot and P_top exactly) and are left out.

**Check C3.** At the start, r . r equals the paper's own loss
(``DarcyPINN._compute_loss``) evaluated on the lifted network to 1e-12
relative, and that loss's Dirichlet terms are zero.

**LM** (item 2's ``lm``): F2's damping (mu0 1e-3, x0.2 on an accepted step and
x5 on a rejected one, the diagonal floored at 1e-12), and tolerances 0 with
one restart (the target is 0, so a run ends at a cap or at a second lost
step). The caps are 2,000 iterations or 30 min of training time.
``eps_ref`` is logged every iteration, off the clock: delta_FV of the
pressure against the finite-volume reference at the cell centres.
Seeds 0, 1, 2; fields S1, S2, S3, SPE10.

**Where.** FASTER, a shared A100 (untimed). The time is reported with the
hardware and is not compared with the manuscript's Grace times.

**Outputs** under ``P2_15_darcy_hardbc/``:
- ``<field>_seed<s>/``: ``run.json``, ``log.csv``,
  ``pressure_field.npz`` (P, P_FV and the cell centres), ``network.pt``
  (the three networks) and the job's ``hardware.json``;
- ``darcy_hardbc_rows.csv``: ``field, seed, k_stop, reason, delta_FV,
  max_diff_psi, time_s, device``, then the final loss, C3, and with
  ``--package1`` the paper's soft-BC Adam NiL and LiL delta_FV of the same
  field and seed.

Usage::

    python experiments/p2_15_darcy_hardbc.py run --field S1 --seed 0 --out <stage root> [--device cuda]
    python experiments/p2_15_darcy_hardbc.py summarize --out <stage root> [--package1 <package1>]
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from lilq.blas_threads import pin_torch  # first: sets the BLAS threads before numpy/scipy load
import numpy as np
import torch
from torch.func import jacrev, vmap

from experiments.p2_8_lm_networks import DIAG_FLOOR, MU0, MU_DOWN, MU_MAX, MU_UP, lm
from lilq.provenance import capture_blas_thread_env, save_provenance
from lilq.source_lock import current_commit
from problems.darcy import DarcyConfig, DarcyPhysics, DarcyPINN, delta_fv, solve_fvm

pin_torch()

FIELDS = ('S1', 'S2', 'S3', 'SPE10')
SEEDS = (0, 1, 2)
MAX_ITERATIONS = 2000
WALL_CAP_S = 30 * 60.0
W_PDE, W_BC = 50.0, 20.0          # DarcyPINN._compute_loss's weights (C3 checks the loss they give)
C3_TOL = 1e-12
METHOD = 'NiL-N hard-BC (LM)'
LOG_COLUMNS = ('iteration', 'n_func_evals', 'loss', 'darcy_x', 'darcy_y', 'continuity', 'bc_lr', 'wall_time',
               'mu', 'n_jacobian_evals', 'accepted', 'restarts', 'eps_ref')
ROW_COLUMNS = ('field', 'seed', 'k_stop', 'reason', 'delta_FV', 'max_diff_psi', 'time_s', 'device')


def config_for(field):
    return DarcyConfig(perm_file=f'perm_field_{field}.txt')


class HardBCDarcyPINN(DarcyPINN):
    """The paper's network with the pressure output lifted:
    P = (y* + omega(y*) NN_h) DELTA_P + P_BOTTOM, omega = 4 y* (1 - y*)."""

    def _get_P(self, x, y):
        xn, yn = self._norm_input(x, y)
        ys = y.to(self.dtype) / self.physics.LY
        h = self.net_P(torch.cat([xn, yn], dim=1))
        return (ys + 4.0 * ys * (1.0 - ys) * h) * self.physics.DELTA_P + self.config.P_BOTTOM


# ---------------------------------------------------------------- the networks as functions of theta

def _mlp(ps, X):
    """A SiLU MLP (``lilq.nn.MLP``) from its parameter list: the output and its
    gradient with respect to the (normalized) inputs, ``(N,), (N, 2)``."""
    a = X
    da = torch.eye(2, dtype=X.dtype, device=X.device).expand(X.shape[0], 2, 2)
    L = len(ps) // 2
    for k in range(L):
        W, b = ps[2 * k], ps[2 * k + 1]
        z = a @ W.T + b
        dz = torch.einsum('ij,njd->nid', W, da)
        if k < L - 1:
            s = torch.sigmoid(z)
            a, da = z * s, (s * (1 + z * (1 - s)))[..., None] * dz
        else:
            a, da = z, dz
    return a[:, 0], da[:, 0, :]


class DarcyResidual:
    """The hard-BC loss as r . r (the paper's weights), its Jacobian, and the
    components ``log.csv`` records (``lm``'s ``log_columns`` hook)."""

    log_columns = LOG_COLUMNS

    def __init__(self, pinn):
        self.p = pinn
        ph = pinn.physics
        self.shapes = [q.shape for net in (pinn.net_P, pinn.net_U, pinn.net_V) for q in net.parameters()]
        n_layers = len(list(pinn.net_P.parameters()))
        self.split = (n_layers, 2 * n_layers)
        dev, dt = pinn.device, pinn.dtype
        t = lambda a: torch.as_tensor(a, dtype=dt, device=dev)  # noqa: E731
        self.X_int = torch.cat([pinn.xpde, pinn.ypde], 1).detach()
        self.sqrt_K = pinn.sqrt_Kx.detach()[:, 0]
        self.X_lat = [torch.cat([pinn.xleft, pinn.yleft], 1).detach(), torch.cat([pinn.xright, pinn.yright], 1).detach()]
        self.n_int, self.n_lat = len(self.X_int), [len(X) for X in self.X_lat]
        self.w_int = math.sqrt(W_PDE / self.n_int)
        self.w_lat = [math.sqrt(W_BC / n) for n in self.n_lat]
        self.XS, self.YS, self.LY = pinn.X_SCALE, pinn.Y_SCALE, float(ph.LY)
        self.DP, self.PB = float(ph.DELTA_P), float(pinn.config.P_BOTTOM)
        self.VS, self.CE, self.norm = pinn.V_SCALE, pinn.CE_SCALE, pinn.sqrt_K0 * pinn.dP_scale
        self._t = t

        def point_int(theta, X, sk):
            dx, dy, ce = self._interior(theta, X[None], sk[None])
            return torch.stack([dx[0], dy[0], ce[0]])

        def point_lat(theta, X):
            return self._lateral(theta, X[None])[0]
        self._j_int = vmap(jacrev(point_int), (None, 0, 0))
        self._j_lat = vmap(jacrev(point_lat), (None, 0))

    def theta0(self):
        return torch.cat([q.detach().reshape(-1) for net in (self.p.net_P, self.p.net_U, self.p.net_V)
                          for q in net.parameters()])

    def _unflat(self, theta):
        out, i = [], 0
        for s in self.shapes:
            out.append(theta[i:i + s.numel()].reshape(s))
            i += s.numel()
        a, b = self.split
        return out[:a], out[a:b], out[b:]

    def _norm(self, X):
        return torch.stack([X[:, 0] / self.XS - 1.0, X[:, 1] / self.YS - 1.0], 1)

    def _interior(self, theta, X, sk):
        """The unweighted Darcy-x, Darcy-y and continuity residuals (the paper's)."""
        pP, pU, pV = self._unflat(theta)
        Xn = self._norm(X)
        h, dh = _mlp(pP, Xn)
        u, du = _mlp(pU, Xn)
        v, dv = _mlp(pV, Xn)
        ys = X[:, 1] / self.LY
        om, dom = 4.0 * ys * (1.0 - ys), 4.0 * (1.0 - 2.0 * ys) / self.LY
        Px = self.DP * om * dh[:, 0] / self.XS
        Py = self.DP * (1.0 / self.LY + dom * h + om * dh[:, 1] / self.YS)
        U, V = u * self.VS, v * self.VS
        Ux, Vy = du[:, 0] / self.XS * self.VS, dv[:, 1] / self.YS * self.VS
        return ((U / sk + sk * Px) / self.norm, (V / sk + sk * Py) / self.norm, (Ux + Vy) / self.CE)

    def _lateral(self, theta, X):
        """U / V_SCALE on a lateral face (the paper's no-flow rows)."""
        _, pU, _ = self._unflat(theta)
        u, _ = _mlp(pU, self._norm(X))
        return u

    def pressure(self, theta, X):
        pP, _, _ = self._unflat(theta)
        h, _ = _mlp(pP, self._norm(X))
        ys = X[:, 1] / self.LY
        return (ys + 4.0 * ys * (1.0 - ys) * h) * self.DP + self.PB

    def vector(self, theta):
        dx, dy, ce = self._interior(theta, self.X_int, self.sqrt_K)
        lat = [w * self._lateral(theta, X) for w, X in zip(self.w_lat, self.X_lat)]
        return torch.cat([self.w_int * dx, self.w_int * dy, self.w_int * ce] + lat)

    def jacobian(self, theta):
        Ji = self._j_int(theta, self.X_int, self.sqrt_K)              # (n, 3, n_theta)
        J = [self.w_int * Ji[:, 0], self.w_int * Ji[:, 1], self.w_int * Ji[:, 2]]
        J += [w * self._j_lat(theta, X) for w, X in zip(self.w_lat, self.X_lat)]
        return torch.cat(J)

    def log_values(self, theta):
        dx, dy, ce = self._interior(theta, self.X_int, self.sqrt_K)
        lat = sum(float(torch.mean(self._lateral(theta, X) ** 2)) for X in self.X_lat)
        return {'darcy_x': repr(W_PDE * float(torch.mean(dx ** 2))), 'darcy_y': repr(W_PDE * float(torch.mean(dy ** 2))),
                'continuity': repr(W_PDE * float(torch.mean(ce ** 2))), 'bc_lr': repr(W_BC * lat)}


# ---------------------------------------------------------------- one run

def run(field, seed, out_root, device=None, max_iterations=MAX_ITERATIONS, wall_cap_s=WALL_CAP_S):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run_dir = Path(out_root) / 'P2_15_darcy_hardbc' / f'{field}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    config = config_for(field)
    physics = DarcyPhysics(config, verbose=False)
    P_fv = solve_fvm(physics)
    pinn = HardBCDarcyPINN(config, physics, device=device, dtype=torch.float64, seed=seed)
    res = DarcyResidual(pinn)
    theta = res.theta0()

    # C3: r . r against the paper's loss on the lifted network; its Dirichlet terms vanish
    paper_loss, parts = pinn._compute_loss()
    r0 = res.vector(theta)
    c3 = {'paper_loss_iteration_0': float(paper_loss.detach()), 'lm_r_dot_r_iteration_0': float(r0 @ r0),
          'paper_dirichlet_terms': [parts['bc_bot'], parts['bc_top']]}
    c3['rel_diff'] = abs(c3['lm_r_dot_r_iteration_0'] - c3['paper_loss_iteration_0']) / c3['paper_loss_iteration_0']
    c3['passed'] = bool(c3['rel_diff'] <= C3_TOL and parts['bc_bot'] == 0 and parts['bc_top'] == 0)

    x_c = (np.arange(config.NX_CELLS) + 0.5) * (physics.LX / config.NX_CELLS)
    y_c = (np.arange(config.NY_CELLS) + 0.5) * (physics.LY / config.NY_CELLS)
    Xg, Yg = np.meshgrid(x_c, y_c, indexing='ij')
    cells = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], 1), dtype=torch.float64, device=device)

    def pressure(theta):
        return res.pressure(theta, cells).detach().cpu().numpy().reshape(Xg.shape)

    def eps_ref(theta):
        return delta_fv(pressure(theta), P_fv, config.P_BOTTOM)

    theta, summary = lm(res, theta, 0.0, eps_ref, run_dir / 'log.csv', max_iterations, wall_cap_s)
    P = pressure(theta)
    np.savez(run_dir / 'pressure_field.npz', P=P, P_fv=P_fv, x=x_c, y=y_c)
    with torch.no_grad():
        for net, ps in zip((pinn.net_P, pinn.net_U, pinn.net_V), res._unflat(theta)):
            for q, v in zip(net.parameters(), ps):
                q.copy_(v)
    from lilq.saved_models import save_checkpoint
    save_checkpoint(run_dir / 'network.pt', {'networks': pinn.network_state(), 'field': field,
                                             'lifting': 'P = (y* + 4 y*(1 - y*) NN_h) DELTA_P + P_BOTTOM'})
    record = {
        'field': field, 'seed': seed, 'method': METHOD, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'k_stop': summary['total_iterations'], 'reason': summary['stopping_reason'],
        'delta_FV': delta_fv(P, P_fv, config.P_BOTTOM), 'max_diff_psi': float(np.abs(P - P_fv).max()),
        'time_s': summary['training_time'], 'final_loss': summary['final_loss'], 'c3': c3,
        'n_theta': int(theta.numel()), 'n_rows': int(r0.numel()),
        'differs_from_the_paper_nil': ['the pressure output is lifted (hard Dirichlet conditions)',
                                       'Levenberg-Marquardt instead of Adam'],
        'network': '3 x MLP(2, 32, 1, 2 hidden layers, SiLU), float64, torch.manual_seed(seed), the paper\'s order',
        'loss': {'interior': 'Darcy-x, Darcy-y, continuity: each 50 x mean square (the paper\'s normalization)',
                 'lateral': 'U / V_SCALE = 0 on each lateral face: 20 x mean square (the paper\'s rows)',
                 'dirichlet': 'none: identically satisfied by the lifting'},
        'optimizer': {'name': 'Levenberg-Marquardt (F2)', 'mu0': MU0, 'mu_up': MU_UP, 'mu_down': MU_DOWN,
                      'mu_max': MU_MAX, 'diag_floor': DIAG_FLOOR, 'stall_rule': 'tolerances 0, one restart'},
        'caps': {'iterations': max_iterations, 'wall_time_s': wall_cap_s},
        'summary': summary, 'thread_env': capture_blas_thread_env(), 'torch_threads': torch.get_num_threads(),
        'commit': current_commit()}
    (run_dir / 'run.json').write_text(json.dumps(record, indent=2, default=str))
    save_provenance(run_dir)
    return record


# ---------------------------------------------------------------- the rows

def _paper_rows(package1):
    """``{(field, seed): delta_FV}`` of the paper's NiL (float64) and ``{field: delta_FV}`` of LiL."""
    nil, lil = {}, {}
    for p in sorted(Path(package1).glob('B_instrumentation/darcy_fv/*/darcy_fv_comparison.csv')):
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                if r['method'] == 'NiL' and r['dtype'] == 'float64':
                    nil[(r['field'], int(r['seed']))] = float(r['delta_fv'])
                elif r['method'] == 'LiL':
                    lil[r['field']] = float(r['delta_fv'])
    return nil, lil


def summarize(out_root, package1=None):
    out = Path(out_root) / 'P2_15_darcy_hardbc'
    runs = [json.loads(p.read_text()) for p in sorted(out.glob('*_seed*/run.json'))]
    nil, lil = _paper_rows(package1) if package1 else ({}, {})
    rows = []
    for r in runs:
        row = {k: r[k] for k in ROW_COLUMNS}
        row.update(final_loss=r['final_loss'], c3_rel_diff=r['c3']['rel_diff'], c3_passed=r['c3']['passed'],
                   gpu=r['gpu'], restarts=r['summary'].get('lm_restarts'),
                   delta_FV_paper_nil_soft_bc_adam=nil.get((r['field'], r['seed']), ''),
                   delta_FV_lil=lil.get(r['field'], ''))
        rows.append(row)
    if rows:
        with open(out / 'darcy_hardbc_rows.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 2, item 5: the Darcy network with hard Dirichlet conditions.')
    ap.add_argument('stage', choices=('run', 'summarize'))
    ap.add_argument('--out', required=True, help='the stage root')
    ap.add_argument('--field', choices=FIELDS)
    ap.add_argument('--seed', type=int, choices=SEEDS)
    ap.add_argument('--array-task', type=int, help='0-11: field = FIELDS[task // 3], seed = task %% 3')
    ap.add_argument('--device', default=None)
    ap.add_argument('--package1', default=None)
    ap.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS)
    ap.add_argument('--wall-cap-s', type=float, default=WALL_CAP_S)
    args = ap.parse_args(argv)
    if args.stage == 'summarize':
        for r in summarize(args.out, args.package1):
            print(f"  {r['field']:5s} seed {r['seed']}: {r['k_stop']} iterations ({r['reason']}), "
                  f"delta_FV {r['delta_FV']:.3e} (paper NiL {r['delta_FV_paper_nil_soft_bc_adam'] or '-'}), "
                  f"max {r['max_diff_psi']:.1f} psi, {r['time_s']:.0f} s")
        return
    if args.array_task is not None:
        field, seed = FIELDS[args.array_task // len(SEEDS)], SEEDS[args.array_task % len(SEEDS)]
    else:
        field, seed = args.field, args.seed
    if field is None or seed is None:
        ap.error('run needs --field and --seed, or --array-task')
    done = Path(args.out) / 'P2_15_darcy_hardbc' / f'{field}_seed{seed}' / 'run.json'
    if done.exists():
        print(f'  {done.parent}: done, skipping')
        return
    r = run(field, seed, args.out, args.device, args.max_iterations, args.wall_cap_s)
    print(f"  {field} seed {seed}: {r['k_stop']} iterations ({r['reason']}), delta_FV {r['delta_FV']:.3e}, "
          f"max {r['max_diff_psi']:.1f} psi, {r['time_s']:.0f} s on {r['gpu'] or r['device']}; C3 "
          f"{'passed' if r['c3']['passed'] else 'FAILED'} ({r['c3']['rel_diff']:.1e})")


if __name__ == '__main__':
    main()
