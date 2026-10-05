"""
Package 2, item 2 (P2-8): NiL-N trained by Levenberg-Marquardt
==============================================================

The advisor's instructions of 4 October 2026, Section 5. The NiL-N
formulation of the four-method comparison, with only the optimizer
changed: Levenberg-Marquardt with the F2 damping schedule
(``baselines/lm_kovasznay.py``) instead of L-BFGS. The method is labelled
**NiL-N (LM)**.

**The same runs, by construction.** Each benchmark's own ``run_nil_n`` is
called with the four-method configuration (``paper_setup``) and the seed
(``init_seed``, collocation seed 42), with its ``solve_nil_n`` replaced by
the LM trainer here. Everything before the optimizer is therefore the
Package 1 code:
- the network: tanh, two hidden layers of width h with h^2 + 5h >= P;
- the pretraining fit to the zero initial guess;
- the collocation set, the boundary and initial data, and the loss weights;
- the target MSE.

**The loss as a residual vector.** LM needs r(theta) with
loss = r . r. The rows are:
- the interior residual times sqrt(lambda_pde / n_pde);
- each boundary or initial line's residual times sqrt(lambda / n_line).

So r . r is the NiL loss: for Bratu, lambda_bc times the sum of the four
per-edge mean squares; for Burgers and BL, the initial line and the two
lateral lines. The derivatives the residual needs come from an explicit
Taylor-mode pass through the tanh layers (``fields``), and the Jacobian
from ``torch.func``. **Check C3** compares r . r at iteration 0 with the
L-BFGS objective of ``solve_nil_n``, evaluated on the same network by the
Package 1 residual functions; they must agree to 1e-12 relative. The
LiL-normalized boundary term is logged beside the NiL one:
- **Bratu:** all boundary points in one mean square;
- **Burgers and BL:** the same as the NiL term, since LiL also weights
  each line by its own count.

**LM** (F2):
- the step solves (J^T J + mu diag(J^T J)) dtheta = -J^T r by Cholesky,
  with the diagonal floored at 1e-12;
- mu0 = 1e-3; mu is multiplied by 0.2 on an accepted step and by 5 on a
  rejected step (or on a failed factorization);
- a step is accepted if and only if the loss decreases.

**Stopping:**
- the target, checked after each step;
- 2,000 LM iterations;
- a 15-min training wall time, excluding the per-iteration ``eps_ref``,
  as F2 excludes its test errors;
- a stall under the stall controls' rule (``stall_rule='f1'``:
  tolerances 0, one restart).

For LM, a step is lost when mu exceeds 1e16 without an accepted step. The
restart resets mu to mu0; a second loss in a row is ``optimizer_stall``. F2's
own 20-step stagnation tolerance is a tolerance and is therefore off.

**Logging** (``log.csv``):
- the history columns of the Package 1 four-method runs (``iteration,
  n_func_evals, loss, pde_loss, ic_loss, bc_loss, wall_time``), where
  ``loss`` is r . r, the quantity tested against the target;
- then ``bc_loss_lil, loss_lil, mu, n_jacobian_evals, accepted, restarts``;
- ``eps_ref`` (against the Section 6.2 reference; BL against
  ``problems.buckley_leverett.reference_solution``) at every iteration.

**Outputs** under ``P2_8_lm_networks/``:
- ``<benchmark>_P<P>_seed<s>/{run.json, log.csv, network.pt}``;
- ``provenance/<benchmark>_<device>/``: each job's hardware and environment;
- ``four_method_lm_rows.csv``: one row per run, in the columns of
  ``four_method_tables.csv``, plus the LM fields;
- ``four_method_lm_medians.csv``: medians over the seeds with the tables'
  stopping markers, beside the NiL-N (L-BFGS) medians of ``package1``;
- ``figures/``: loss and ``eps_ref`` against time, one panel per size,
  with NiL-N (L-BFGS) from ``package1`` on the same axes.

**Device.** CPU node. A configuration that the wall-time cap ends before the
target and before 2,000 iterations is rerun on an A100 with the same cap,
all three seeds (``--device cuda``; ``gpu-list`` names them).

Usage::

    python experiments/p2_8_lm_networks.py run --benchmark bratu --out <stage root> --reference-dir <dir> \\
        [--P 25 100] [--seeds 0 1 2] [--device cpu] [--package1 <package1>]
    python experiments/p2_8_lm_networks.py summarize --out <stage root> --package1 <package1> \
        --lbfgs-errors <stage 1>/P2_12_reference_errors/scalar_reference_errors.csv \
        <stage 1>/G1_release/bl_reference_errors_networks.csv
    python experiments/p2_8_lm_networks.py gpu-list --out <stage root>
"""

import argparse
import contextlib
import csv
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from unittest import mock

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np
import torch
from torch.func import grad, jacrev, vmap

from lilq.four_method_log import FOUR_METHOD_CSV_COLUMNS
from lilq.provenance import capture_blas_thread_env, save_provenance
from lilq.source_lock import current_commit

METHOD = 'NiL-N (LM)'
SEEDS = (0, 1, 2)
MAX_ITERATIONS = 2000
WALL_CAP_S = 15 * 60.0
MU0, MU_UP, MU_DOWN, MU_MAX, DIAG_FLOOR = 1e-3, 5.0, 0.2, 1e16, 1e-12
C3_TOL = 1e-12
BENCHMARKS = ('bratu', 'burgers', 'bl')
LOG_COLUMNS = ('iteration', 'n_func_evals', 'loss', 'pde_loss', 'ic_loss', 'bc_loss', 'wall_time',
               'bc_loss_lil', 'loss_lil', 'mu', 'n_jacobian_evals', 'accepted', 'restarts', 'eps_ref')


# ---------------------------------------------------------------- the benchmarks

def setup(bench, P):
    """``(module, config, opt)`` of the four-method run at P."""
    n = int(round(math.sqrt(P)))
    if bench == 'bratu':
        import problems.bratu as mod
        from experiments.run_bratu import paper_setup
        return (mod, *paper_setup(n))
    if bench == 'burgers':
        import problems.burgers as mod
        from experiments.run_burgers import paper_setup
        return (mod, *paper_setup(n))
    import problems.buckley_leverett as mod
    from experiments.run_bl import paper_setup
    return (mod, *paper_setup(n, False))


def sizes(bench):
    if bench == 'bratu':
        from experiments.run_bratu import DEFAULT_N_VALUES
    elif bench == 'burgers':
        from experiments.run_burgers import DEFAULT_N_VALUES
    else:
        from experiments.run_bl import DEFAULT_N_VALUES
    return [n * n for n in DEFAULT_N_VALUES]


def pde_function(bench, config):
    """``(u, du, ddu) -> interior residual``, du = (u_x, u_y), ddu = (u_xx, u_yy),
    the formula of the benchmark's ``_compute_pde_residual_nn``."""
    if bench == 'bratu':
        lam = config.lambda_
        return lambda u, du, ddu: ddu[..., 0] + ddu[..., 1] + lam * torch.exp(u)
    if bench == 'burgers':
        nu = config.viscosity
        return lambda u, du, ddu: du[..., 1] + u * du[..., 0] - nu * ddu[..., 0]
    from problems.buckley_leverett import BLPhysics
    physics = BLPhysics(config)
    dflux = vmap(grad(lambda s: physics.flux(s)))
    return lambda u, du, ddu: du[..., 1] + dflux(u.reshape(-1)).reshape(u.shape) * du[..., 0] + physics.D * ddu[..., 0]


def boundary_blocks(bench, bc_data):
    """``[(name, kind, X, target)]``: the boundary and initial lines of the
    benchmark's ``_compute_bc_residual_nn``, values only."""
    col = lambda a: a.reshape(-1, 1)  # noqa: E731
    xt = lambda x, t: torch.cat([col(x), col(t)], 1).detach()  # noqa: E731
    if bench == 'bratu':
        return [(e, 'bc', xt(bc_data[f'x_{e}'], bc_data[f'y_{e}']), None) for e in ('left', 'right', 'bottom', 'top')]
    blocks = [('initial', 'ic', xt(bc_data['x_ic'], bc_data['t_ic']), bc_data['ic_target'].reshape(-1))]
    for side in ('left', 'right'):
        target = bc_data.get(f'bc_{side}_target')
        blocks.append((side, 'bc', xt(bc_data[f'x_{side}'], bc_data[f't_{side}']),
                       None if target is None else target.reshape(-1)))
    return blocks


# ---------------------------------------------------------------- the network, Taylor mode

class Net:
    """A tanh MLP (``lilq.nn.MLP``) as a function of its flat parameter
    vector: values, and first and second input derivatives per direction."""

    def __init__(self, model):
        self.shapes = [p.shape for p in model.parameters()]
        acts = [m for m in model.network if not isinstance(m, torch.nn.Linear)]
        if not all(isinstance(a, torch.nn.Tanh) for a in acts) or len(acts) != len(self.shapes) // 2 - 1:
            raise ValueError('Net expects Linear/Tanh layers with a linear output')

    def unflat(self, theta):
        out, i = [], 0
        for s in self.shapes:
            k = s.numel()
            out.append(theta[i:i + k].reshape(s))
            i += k
        return out

    def fields(self, theta, X):
        """``u (N,), du (N, 2), ddu (N, 2)`` at the points ``X (N, 2)``."""
        ps = self.unflat(theta)
        n = X.shape[0]
        a = X
        da = torch.eye(2, dtype=X.dtype, device=X.device).expand(n, 2, 2)
        dda = torch.zeros(n, 2, 2, dtype=X.dtype, device=X.device)
        L = len(ps) // 2
        for k in range(L):
            W, b = ps[2 * k], ps[2 * k + 1]
            z = a @ W.T + b
            dz = torch.einsum('ij,njd->nid', W, da)
            ddz = torch.einsum('ij,njd->nid', W, dda)
            if k < L - 1:
                s = torch.tanh(z)
                sp = 1 - s * s
                a, da, dda = s, sp[..., None] * dz, sp[..., None] * ddz - 2 * (s * sp)[..., None] * dz * dz
            else:
                a, da, dda = z, dz, ddz
        return a[:, 0], da[:, 0, :], dda[:, 0, :]

    def values(self, theta, X):
        ps = self.unflat(theta)
        a = X
        for k in range(len(ps) // 2):
            a = a @ ps[2 * k].T + ps[2 * k + 1]
            if k < len(ps) // 2 - 1:
                a = torch.tanh(a)
        return a[:, 0]


class Residual:
    """The NiL-N loss as r . r, its components and its Jacobian."""

    def __init__(self, net, pde, X_int, blocks, lambdas):
        self.net, self.pde, self.X = net, pde, X_int
        self.blocks = blocks
        lp, lb, li = lambdas
        self.w_pde = math.sqrt(lp / len(X_int))
        self.w = [math.sqrt((li if kind == 'ic' else lb) / len(X)) for _, kind, X, _ in blocks]
        self.lambdas = lambdas
        self.n_bc = sum(len(X) for _, kind, X, _ in blocks if kind == 'bc')

        def point_pde(theta, x):
            u, du, ddu = net.fields(theta, x[None])
            return pde(u, du, ddu)[0]

        def point_value(theta, x):
            return net.values(theta, x[None])[0]
        self._j_pde = vmap(jacrev(point_pde), (None, 0))
        self._j_val = vmap(jacrev(point_value), (None, 0))

    def raw(self, theta):
        """The unweighted interior residual and each line's residual."""
        u, du, ddu = self.net.fields(theta, self.X)
        lines = []
        for _, _, X, target in self.blocks:
            v = self.net.values(theta, X)
            lines.append(v if target is None else v - target)
        return self.pde(u, du, ddu), lines

    def vector(self, theta):
        pde, lines = self.raw(theta)
        return torch.cat([self.w_pde * pde] + [w * r for w, r in zip(self.w, lines)])

    def jacobian(self, theta):
        J = [self.w_pde * self._j_pde(theta, self.X)]
        J += [w * self._j_val(theta, X) for w, (_, _, X, _) in zip(self.w, self.blocks)]
        return torch.cat(J)

    def components(self, theta):
        """``loss, pde, ic, bc, bc_lil, loss_lil`` (the NiL and LiL normalizations)."""
        lp, lb, li = self.lambdas
        pde, lines = self.raw(theta)
        pde_l = float(torch.mean(pde ** 2))
        ic = sum(float(torch.mean(r ** 2)) for r, (_, k, _, _) in zip(lines, self.blocks) if k == 'ic')
        bc = sum(float(torch.mean(r ** 2)) for r, (_, k, _, _) in zip(lines, self.blocks) if k == 'bc')
        bc_lil = sum(float(torch.sum(r ** 2)) for r, (_, k, _, _) in zip(lines, self.blocks) if k == 'bc') / self.n_bc
        lines_per_bc = sum(1 for _, k, _, _ in self.blocks if k == 'bc')
        if lines_per_bc == 2:                     # Burgers and BL: LiL also weights each line by its own count
            bc_lil = bc
        return (lp * pde_l + lb * bc + li * ic, pde_l, ic, bc, bc_lil, lp * pde_l + lb * bc_lil + li * ic)


# ---------------------------------------------------------------- references

def reference(bench, config, reference_dir):
    """``(axes, field)`` on the benchmark's test grid."""
    if bench == 'bl':
        from experiments.network_reference_errors import bl_reference
        return bl_reference(config)
    from lilq.references import load_reference, reference_path
    axes, u, _ = load_reference(reference_path(reference_dir, bench))
    return axes, u


def eps_ref_function(net, ref, device):
    (x, y), u = ref
    X, Y = np.meshgrid(x, y, indexing='ij')
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64, device=device)
    target = torch.tensor(u.ravel(), dtype=torch.float64, device=device)
    norm = torch.linalg.norm(target)
    return lambda theta: float(torch.linalg.norm(net.values(theta, pts) - target) / norm)


# ---------------------------------------------------------------- LM

def lm(res, theta, target, eps_ref, log_path, max_iterations=MAX_ITERATIONS, wall_cap_s=WALL_CAP_S):
    """The F2 Levenberg-Marquardt loop with the item's stopping rules.
    Returns ``(theta, summary)``; writes ``log.csv`` as it goes."""
    columns = list(LOG_COLUMNS)
    fh = open(log_path, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=columns)
    w.writeheader()
    mu = MU0
    r = res.vector(theta)
    loss = float(r @ r)
    n_evals, n_jac, restarts, restarted = 1, 0, 0, False
    excluded = 0.0
    t0 = time.perf_counter()

    def log(it, accepted):
        nonlocal excluded
        t = time.perf_counter() - t0 - excluded
        te = time.perf_counter()
        comp = res.components(theta)
        e = eps_ref(theta)
        excluded += time.perf_counter() - te
        w.writerow(dict(zip(columns, (it, n_evals, repr(loss), repr(comp[1]), repr(comp[2]), repr(comp[3]),
                                      f'{t:.6f}', repr(comp[4]), repr(comp[5]), repr(mu), n_jac, int(accepted),
                                      restarts, repr(e)))))
        fh.flush()
        return t, e

    log(0, True)
    it, reason, stall = 0, None, None
    while reason is None:
        J = res.jacobian(theta)
        n_jac += 1
        H = J.T @ J
        g = J.T @ r
        d = torch.diagonal(H).clamp_min(DIAG_FLOOR)
        accepted, lost, out_of_time = False, False, False
        while not accepted:
            A = H.clone()
            A.diagonal().add_(mu * d)
            Lc, info = torch.linalg.cholesky_ex(A)
            if info.item() == 0:
                step = -torch.cholesky_solve(g.unsqueeze(1), Lc).squeeze(1)
                r_new = res.vector(theta + step)
                n_evals += 1
                loss_new = float(r_new @ r_new)
                if loss_new < loss:
                    theta, r, loss = theta + step, r_new, loss_new
                    mu *= MU_DOWN
                    accepted = True
                    break
            mu *= MU_UP
            if mu > MU_MAX:
                lost = True
                break
            if time.perf_counter() - t0 - excluded > wall_cap_s:
                out_of_time = True
                break
        it += 1
        t, _ = log(it, accepted)
        if accepted:
            restarted = False
        if not math.isfinite(loss):
            reason = 'failure'
        elif loss < target:
            reason = 'target'
        elif lost and restarted:
            reason = 'optimizer_stall'
            stall = {'iteration': it, 'evaluations': n_evals, 'time_s': t}
        elif lost:
            mu, restarted = MU0, True
            restarts += 1
        if reason is None and (out_of_time or t > wall_cap_s):
            reason = 'wall_time_cap'
        if reason is None and it >= max_iterations:
            reason = 'iteration_cap'
    fh.close()
    return theta, {'total_iterations': it, 'total_line_searches': n_evals, 'n_jacobian_evals': n_jac,
                   'training_time': time.perf_counter() - t0 - excluded, 'final_loss': loss,
                   'converged': reason == 'target', 'stopping_reason': reason, 'mu_final': mu,
                   'lm_restarts': restarts, 'stall': stall}


# ---------------------------------------------------------------- one run

def c3_reference_loss(pde_fn, bc_fn, model, x_pde, y_pde, bc_data, lambda_pde, lambda_bc, lambda_ic):
    """The L-BFGS objective of ``lilq.solvers.solve_nil_n`` on the given network."""
    residual = pde_fn(model, x_pde, y_pde)
    pde_loss = torch.nn.MSELoss()(residual, torch.zeros_like(residual))
    bc_loss, ic_loss = bc_fn(model, bc_data)
    total = lambda_pde * pde_loss + lambda_bc * bc_loss
    if ic_loss is not None:
        total = total + lambda_ic * ic_loss
    return float(total.detach())


def run(bench, P, seed, reference_dir, out_root, device='cpu', package1=None,
        max_iterations=MAX_ITERATIONS, wall_cap_s=WALL_CAP_S):
    torch.set_default_dtype(torch.float64)
    mod, config, opt = setup(bench, P)
    config = dataclasses.replace(config, init_seed=seed)
    run_dir = Path(out_root) / 'P2_8_lm_networks' / f'{bench}_P{P}_seed{seed}{"_cuda" if device != "cpu" else ""}'
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    captured = {}

    def lm_solver(pde_fn, bc_fn, model, x_pde, y_pde, bc_data, lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
                  R_tol=1e-4, **_):
        start = c3_reference_loss(pde_fn, bc_fn, model, x_pde, y_pde, bc_data, lambda_pde, lambda_bc, lambda_ic)
        net = Net(model)
        X = torch.cat([x_pde.reshape(-1, 1), y_pde.reshape(-1, 1)], 1).detach()
        res = Residual(net, pde_function(bench, config), X, boundary_blocks(bench, bc_data),
                       (lambda_pde, lambda_bc, lambda_ic))
        theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        r0 = res.vector(theta)
        lm_start = float(r0 @ r0)
        eps = eps_ref_function(net, reference(bench, config, reference_dir), dev)
        theta, summary = lm(res, theta, R_tol, eps, run_dir / 'log.csv', max_iterations, wall_cap_s)
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(theta, model.parameters())
        captured.update(c3={'lbfgs_objective_iteration_0': start, 'lm_r_dot_r_iteration_0': lm_start,
                            'rel_diff': abs(lm_start - start) / abs(start),
                            'passed': bool(abs(lm_start - start) <= C3_TOL * abs(start))},
                        eps_ref_final=eps(theta), n_theta=int(theta.numel()), n_rows=int(r0.numel()), R_tol=R_tol,
                        lambdas=[lambda_pde, lambda_bc, lambda_ic])
        return model, None, summary

    t0 = time.perf_counter()
    with mock.patch.object(mod, 'solve_nil_n', lm_solver):
        model, _, summary = mod.run_nil_n(config, opt, device=dev, verbose=False)
    wall_total = time.perf_counter() - t0
    from lilq.saved_models import save_network
    save_network(run_dir, model, config, opt)
    p1 = package1_start(package1, bench, P, seed) if package1 else None
    record = {
        'benchmark': bench, 'P': P, 'method': METHOD, 'seed': seed, 'collocation_seed': config.seed,
        'device': device, 'wall_total_s': wall_total, 'commit': current_commit(),
        'optimizer': {'name': 'Levenberg-Marquardt (F2)', 'mu0': MU0, 'mu_up': MU_UP, 'mu_down': MU_DOWN,
                      'mu_max': MU_MAX, 'diag_floor': DIAG_FLOOR, 'stall_rule': 'f1 (tolerances 0, one restart)'},
        'caps': {'iterations': max_iterations, 'wall_time_s': wall_cap_s},
        'thread_env': capture_blas_thread_env(), 'torch_threads': torch.get_num_threads(),
        'pretrain_loss': summary.get('pretrain_loss'), **{k: v for k, v in summary.items() if k != 'pretrain_loss'},
        **captured, 'package1_start': p1,
    }
    (run_dir / 'run.json').write_text(json.dumps(record, indent=2, default=str))
    return record


def package1_start(package1, bench, P, seed):
    """The iteration-0 loss of the Package 1 NiL-N (L-BFGS) run(s) of this seed,
    for the record (another device's pretraining need not agree to 12 digits)."""
    out = {}
    for dev in ('cpu', 'cuda'):
        h = Path(package1) / 'B_instrumentation' / 'four_method_jobs' / f'{bench}_{"gpu" if dev == "cuda" else "cpu"}' \
            / 'models' / f'{bench}_P{P}_NiL-N_s{seed}_{dev}' / 'history.csv'
        if h.exists():
            with open(h) as fh:
                out[dev] = float(next(csv.DictReader(fh))['loss'])
    return out or None


# ---------------------------------------------------------------- summaries

def _runs(out_root):
    rows = []
    for p in sorted((Path(out_root) / 'P2_8_lm_networks').glob('*_seed*/run.json')):
        rows.append(json.loads(p.read_text()))
    return rows


def _markers(reasons):
    """The tables' markers (``experiments/release_tables._markers``), plus ``W`` when
    most runs ended at the wall-time cap."""
    n, hit = len(reasons), reasons.count('target')
    marks = []
    if reasons.count('iteration_cap') * 2 > n:
        marks.append('*')
    if reasons.count('optimizer_stall') * 2 > n:
        marks.append('dagger')
    if reasons.count('wall_time_cap') * 2 > n:
        marks.append('W')
    if n == 3 and 0 < hit < 3:
        marks.append(f'{hit}/3')
    return ' '.join(marks)


def lbfgs_reference_errors(paths):
    """``{(benchmark, P, seed): eps_ref_final}`` of the NiL-N (L-BFGS) runs of the
    tables (GPU), from the Stage-1 reference-error CSVs: ``scalar_reference_errors.csv``
    (Bratu, Burgers) and ``bl_reference_errors_networks.csv`` (BL)."""
    out = {}
    for path in paths or ():
        with open(path) as fh:
            for r in csv.DictReader(fh):
                bench = r.get('benchmark') or r.get('case')
                if r['method'] == 'NiL-N' and r['device'] == 'cuda' and r['seed'] not in ('', None):
                    out[(bench, int(r['P']), int(r['seed']))] = float(r['eps_ref_final'])
    return out


def summarize(out_root, package1=None, lbfgs_errors=None):
    out = Path(out_root) / 'P2_8_lm_networks'
    runs = _runs(out_root)
    lbfgs_eps = lbfgs_reference_errors(lbfgs_errors)
    extra = ['n_jacobian_evals', 'eps_ref_final', 'c3_rel_diff', 'c3_passed', 'mu_final', 'lm_restarts', 'pretrain_loss']
    with open(out / 'four_method_lm_rows.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(FOUR_METHOD_CSV_COLUMNS) + extra, extrasaction='ignore')
        w.writeheader()
        for r in runs:
            st = r.get('stall') or {}
            w.writerow({**r, 'training_time_s': r['training_time'], 'iterations_cap': r['caps']['iterations'],
                        'line_searches_cap': '', 'stall_iteration': st.get('iteration'),
                        'stall_evaluations': st.get('evaluations'), 'stall_time_s': st.get('time_s'),
                        'loss_history_every_10': '', 'variant': '', 'c3_rel_diff': r['c3']['rel_diff'],
                        'c3_passed': r['c3']['passed']})
    lbfgs = _lbfgs_rows(package1) if package1 else []
    med = []
    for bench, P, device in sorted({(r['benchmark'], r['P'], r['device']) for r in runs}):
        g = [r for r in runs if (r['benchmark'], r['P'], r['device']) == (bench, P, device)]
        lb = [r for r in lbfgs if r['benchmark'] == bench and int(r['P']) == P and r['method'] == 'NiL-N'
              and r['device'] == 'cuda' and not r['variant']]
        med.append({
            'benchmark': bench, 'P': P, 'device': device, 'target_mse': g[0]['R_tol'], 'n_seeds': len(g),
            'iterations_median': statistics.median(r['total_iterations'] for r in g),
            'markers': _markers([r['stopping_reason'] for r in g]),
            'time_s_median': statistics.median(r['training_time'] for r in g),
            'final_loss_median': statistics.median(r['final_loss'] for r in g),
            'eps_ref_median': statistics.median(r['eps_ref_final'] for r in g),
            'reasons': ' '.join(r['stopping_reason'] for r in g),
            'lbfgs_iterations_median': statistics.median(int(r['total_iterations']) for r in lb) if lb else '',
            'lbfgs_markers': _markers([r['stopping_reason'] for r in lb]) if lb else '',
            'lbfgs_time_s_median': statistics.median(float(r['training_time_s']) for r in lb) if lb else '',
            'lbfgs_final_loss_median': statistics.median(float(r['final_loss']) for r in lb) if lb else '',
            'lbfgs_eps_ref_median': statistics.median(e) if (e := [lbfgs_eps[(bench, P, int(r['seed']))] for r in lb
                                                                   if (bench, P, int(r['seed'])) in lbfgs_eps]) else '',
            'lbfgs_device': 'cuda (the tables\' runs)' if lb else ''})
    if med:
        with open(out / 'four_method_lm_medians.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(med[0]))
            w.writeheader()
            w.writerows(med)
    figures(out, runs, package1, lbfgs_eps)
    return runs, med


def _lbfgs_rows(package1):
    p = Path(package1) / 'B_instrumentation' / 'four_method_tables.csv'
    if not p.exists():
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def figures(out, runs, package1=None, lbfgs_eps=None):
    """Loss and eps_ref against training time, one panel per size: NiL-N (LM),
    and NiL-N (L-BFGS, the tables' GPU runs) from package1 -- its loss history,
    and its final eps_ref as a marker at its final time (only the final models
    were saved)."""
    lbfgs_eps = lbfgs_eps or {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out / 'figures').mkdir(exist_ok=True)
    for bench in sorted({r['benchmark'] for r in runs}):
        Ps = sorted({r['P'] for r in runs if r['benchmark'] == bench})
        fig, axes = plt.subplots(2, len(Ps), figsize=(3.4 * len(Ps), 6), squeeze=False)
        for j, P in enumerate(Ps):
            for r in [r for r in runs if r['benchmark'] == bench and r['P'] == P]:
                d = out / f"{bench}_P{P}_seed{r['seed']}{'_cuda' if r['device'] != 'cpu' else ''}"
                with open(d / 'log.csv') as fh:
                    log = list(csv.DictReader(fh))
                t = [float(x['wall_time']) for x in log]
                axes[0, j].loglog(t[1:], [float(x['loss']) for x in log][1:], 'C0-', lw=0.9,
                                  label='NiL-N (LM)' if r['seed'] == 0 else None)
                axes[1, j].loglog(t[1:], [float(x['eps_ref']) for x in log][1:], 'C0-', lw=0.9)
                if package1:
                    h = Path(package1) / 'B_instrumentation' / 'four_method_jobs' / f'{bench}_gpu' / 'models' \
                        / f"{bench}_P{P}_NiL-N_s{r['seed']}_cuda" / 'history.csv'
                    if h.exists():
                        with open(h) as fh:
                            hist = list(csv.DictReader(fh))
                        axes[0, j].loglog([float(x['wall_time']) for x in hist][1:], [float(x['loss']) for x in hist][1:],
                                          'C1-', lw=0.9, label='NiL-N (L-BFGS, GPU)' if r['seed'] == 0 else None)
                        e = lbfgs_eps.get((bench, P, int(r['seed'])))
                        if e is not None:
                            axes[1, j].loglog([float(hist[-1]['wall_time'])], [e], 'C1o', ms=4)
            axes[0, j].axhline(next(x['R_tol'] for x in runs if x['benchmark'] == bench and x['P'] == P),
                               color='k', ls=':', lw=0.8)
            axes[0, j].set(title=f'{bench}, P = {P}', xlabel='training time (s)', ylabel='loss')
            axes[1, j].set(xlabel='training time (s)', ylabel='eps_ref')
            for ax in axes[:, j]:
                ax.grid(True, which='both', alpha=0.3)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / 'figures' / f'{bench}_loss_eps_ref.pdf')
        fig.savefig(out / 'figures' / f'{bench}_loss_eps_ref.png', dpi=130)
        plt.close(fig)


def gpu_list(out_root):
    """The configurations the wall-time cap ended on the CPU before the target
    and before the iteration cap: rerun all three seeds on an A100."""
    runs = [r for r in _runs(out_root) if r['device'] == 'cpu']
    return sorted({(r['benchmark'], r['P']) for r in runs if r['stopping_reason'] == 'wall_time_cap'})


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 2, item 2: NiL-N trained by Levenberg-Marquardt.')
    ap.add_argument('stage', choices=('run', 'summarize', 'gpu-list'))
    ap.add_argument('--out', required=True, help='the stage root')
    ap.add_argument('--benchmark', choices=BENCHMARKS)
    ap.add_argument('--P', type=int, nargs='+', help='default: every paper size')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--reference-dir', help='the Bratu and Burgers references')
    ap.add_argument('--package1', help='package1 (the NiL-N (L-BFGS) runs)')
    ap.add_argument('--lbfgs-errors', nargs='+', help='summarize: the Stage-1 reference-error CSVs of the L-BFGS runs')
    args = ap.parse_args(argv)
    if args.stage == 'gpu-list':
        for bench, P in gpu_list(args.out):
            print(f'{bench} {P}')
        return
    if args.stage == 'summarize':
        _, med = summarize(args.out, args.package1, args.lbfgs_errors)
        for m in med:
            print(f"  {m['benchmark']:7s} P = {m['P']:4d} {m['device']}: {m['iterations_median']} its "
                  f"[{m['markers']}], {m['time_s_median']:.1f} s, loss {m['final_loss_median']:.2e}, "
                  f"eps_ref {m['eps_ref_median']:.2e}")
        return
    # one folder per job: the three CPU jobs run at the same time
    save_provenance(Path(args.out) / 'P2_8_lm_networks' / 'provenance' / f'{args.benchmark}_{args.device}')
    for P in args.P or sizes(args.benchmark):
        for seed in args.seeds:
            done = Path(args.out) / 'P2_8_lm_networks' / f'{args.benchmark}_P{P}_seed{seed}{"_cuda" if args.device != "cpu" else ""}' / 'run.json'
            if done.exists():                     # a resubmitted job keeps the finished runs
                print(f'  {args.benchmark:7s} P = {P:4d} seed {seed}: done, skipping', flush=True)
                continue
            r = run(args.benchmark, P, seed, args.reference_dir, args.out, args.device, args.package1)
            print(f"  {args.benchmark:7s} P = {P:4d} seed {seed}: {r['stopping_reason']} at {r['total_iterations']} "
                  f"its, {r['training_time']:.1f} s, loss {r['final_loss']:.3e}, eps_ref {r['eps_ref_final']:.3e}, "
                  f"C3 {r['c3']['rel_diff']:.1e}", flush=True)


if __name__ == '__main__':
    main()
