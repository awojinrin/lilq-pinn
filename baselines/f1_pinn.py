"""
F1: modern first-order PINN for Kovasznay flow (Package 1 v2.0 Section 4.2)
===========================================================================

Fully connected ``tanh`` network(s) for (u, v, p), optionally with a
Fourier-feature embedding ``x -> [cos(2 pi B x), sin(2 pi B x)]`` (B
Gaussian, standard deviation sigma_FF, m features; m = 0 turns it off):

* one shared trunk with three outputs, or three separate networks;
* Dirichlet u, v either hard, ``u = u_bd + l(x) u~`` with the transfinite
  (Coons) interpolant of the boundary data and the normalized face-distance
  product (the same construction as F2's reference, ``baselines.lm_kovasznay``),
  or soft, with weight lambda_bc on 400 boundary points per face;
* loss: mean squares of the two momentum residuals and continuity at N_int
  uniform random interior points (plus the soft boundary term);
* optional gradient-norm loss balancing every 100 Adam iterations,
  lambda^_i = sum_k ||grad L_k|| / ||grad L_i||, moving average 0.9
  (Wang, Sankaran, Wang & Perdikaris, arXiv:2308.08468, Section 3); the
  weights are frozen during L-BFGS;
* Adam with a 1,000-iteration linear warm-up to eta and decay x0.9 every
  2,000 iterations, for T_Adam iterations; then L-BFGS (strong Wolfe,
  history 50, calls of up to 500 iterations and 625 evaluations, PyTorch's
  convergence tolerances off) until the budget;
* optional resampling of the interior points every 1,000 Adam iterations.

No pressure condition: the pressure is determined up to a constant, and
baselines are compared with the mean-free pressure error (Section 2).
The exact solution is used only for boundary data and test errors.

The budget is wall-clock from the first optimizer step, enforced inside an
L-BFGS call as well: a call interrupted by the budget keeps the lowest-loss
point it evaluated, logged at the time it was evaluated (Addendum v2.2
Section 2.5). A run ends on the budget, on a non-finite loss (``failure``),
or on the family's criterion: an L-BFGS call that does not lower the loss
is followed by one call with a fresh optimizer, and the run ends only if
that one does not lower it either. ``tolerance_grad`` and
``tolerance_change`` are 0: PyTorch's defaults are absolute, and loss
scales differ about 100-fold across configurations.

Each call runs in pieces of 10 iterations with the optimizer state carried
over, so ``log.csv`` has a row every 10 L-BFGS iterations (v2.0 Section
4.4); every point is evaluated once (``lilq.solvers.LBFGSObjective``), so the
piece boundaries cost nothing and change nothing. Test errors are computed
off the clock. Logs follow the Section 6 baseline ``log.csv`` format;
``run.json`` holds the configuration, seed, n_theta, peak GPU memory, end
reason, and ``final_loss_unweighted`` -- momentum and continuity mean
squares plus the soft boundary mean square with weight 1 -- on which
Component A ranks and selects (Section 2.5: the weighted loss is not
comparable across configurations).

``precision='adam32'`` runs Adam in float32 and L-BFGS in float64 (the
Section 4.5 sensitivity run); the default is float64 throughout.
"""

import csv
import json
import math
import os
import time
import traceback

import numpy as np
import torch

from baselines.lm_kovasznay import NU, X0, X1, Y0, Y1, coons, ell, exact, g_u, g_v
from lilq.solvers import LBFGSObjective

LOG_COLUMNS = ("iter", "phase", "t_cum_s", "loss_total", "loss_xmom", "loss_ymom", "loss_cont",
               "loss_bc", "w_xmom", "w_ymom", "w_cont", "w_bc", "lr_or_mu", "grad_norm",
               "eps_u", "eps_v", "eps_p", "eps_p_meanfree")
TERMS = ("xmom", "ymom", "cont", "bc")

WARMUP_ITERS = 1000
DECAY_EVERY, DECAY_RATE = 2000, 0.9
BALANCE_EVERY, BALANCE_ALPHA = 100, 0.9
RESAMPLE_EVERY = 1000
LOG_EVERY = 100
N_BC_PER_FACE = 400
LBFGS_CALL_ITERS, LBFGS_CALL_EVALS = 500, 625    # one call: PyTorch's max_iter=500 and its default max_eval
LBFGS_LOG_EVERY = 10                             # a log.csv row every 10 L-BFGS iterations


def unweighted_loss(terms):
    """Momentum and continuity mean squares, plus the soft boundary mean
    square with weight 1: comparable across configurations, unlike the
    balanced or lambda_bc-weighted training loss."""
    return float(sum(float(v.detach()) if torch.is_tensor(v) else float(v) for v in terms.values()))


def new_lbfgs(params):
    """L-BFGS for one piece of a call: tolerances off, so only the budget,
    the iteration and evaluation limits and the family's criterion stop it."""
    return torch.optim.LBFGS(params, lr=1.0, max_iter=LBFGS_LOG_EVERY, max_eval=LBFGS_CALL_EVALS,
                             tolerance_grad=0.0, tolerance_change=0.0, history_size=50,
                             line_search_fn="strong_wolfe")


def adam_lr(it, eta):
    """Learning rate at Adam iteration ``it`` (0-based): linear warm-up to eta
    over 1,000 iterations, times 0.9 every 2,000 iterations."""
    return eta * min(1.0, (it + 1) / WARMUP_ITERS) * DECAY_RATE ** (it // DECAY_EVERY)


def balanced_weights(weights, grad_norms, alpha=BALANCE_ALPHA):
    """One gradient-norm balancing update: lambda^_i = sum_k g_k / g_i, then
    lambda_i <- alpha lambda_i + (1 - alpha) lambda^_i."""
    total = sum(grad_norms.values())
    return {k: alpha * weights[k] + (1 - alpha) * total / max(g, 1e-30) for k, g in grad_norms.items()}


class FourierNet(torch.nn.Module):
    """[cos(2 pi B x), sin(2 pi B x)] (or x itself, m = 0) -> tanh MLP -> n_out."""

    def __init__(self, width, depth, m, sigma_ff, n_out):
        super().__init__()
        self.m = m
        if m > 0:
            self.register_buffer("B", sigma_ff * torch.randn(m, 2))    # fixed, not trained
        n_in = 2 * m if m > 0 else 2
        layers = [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, n_out)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy):
        if self.m > 0:
            z = 2.0 * math.pi * xy @ self.B.T
            xy = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        return self.net(xy)


class F1Model(torch.nn.Module):
    """(u, v, p) at points ``xy`` (N x 2), with the hard boundary conditions
    applied when ``bc == 'hard'``."""

    def __init__(self, width, depth, m, sigma_ff, trunk='shared', bc='hard'):
        super().__init__()
        self.bc = bc
        if trunk == 'shared':
            self.nets = torch.nn.ModuleList([FourierNet(width, depth, m, sigma_ff, 3)])
        else:
            self.nets = torch.nn.ModuleList([FourierNet(width, depth, m, sigma_ff, 1) for _ in range(3)])

    def raw(self, xy):
        return torch.cat([net(xy) for net in self.nets], dim=1)

    def forward(self, xy):
        out = self.raw(xy)
        if self.bc != 'hard':
            return out
        x, y = xy[:, 0], xy[:, 1]
        L = ell(x, y)
        u = coons(g_u, x, y) + L * out[:, 0]
        v = coons(g_v, x, y) + L * out[:, 1]
        return torch.stack([u, v, out[:, 2]], dim=1)


def ns_residuals(fields_fn, xy):
    """(x-momentum, y-momentum, continuity) residuals at ``xy`` by autograd,
    for any ``fields_fn: (N x 2) -> (N x 3)``."""
    xy = xy.detach().requires_grad_(True)
    out = fields_fn(xy)
    u, v = out[:, 0], out[:, 1]
    grad = lambda f: torch.autograd.grad(f.sum(), xy, create_graph=True)[0]  # noqa: E731
    gu, gv, gp = grad(u), grad(v), grad(out[:, 2])
    ux, uy, vx, vy = gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1]
    lap_u = grad(ux)[:, 0] + grad(uy)[:, 1]
    lap_v = grad(vx)[:, 0] + grad(vy)[:, 1]
    r1 = u * ux + v * uy + gp[:, 0] - NU * lap_u
    r2 = u * vx + v * vy + gp[:, 1] - NU * lap_v
    return r1, r2, ux + vy


def interior_points(n, gen, dtype, device):
    u01 = torch.rand(n, 2, generator=gen, dtype=dtype)
    return torch.stack([X0 + (X1 - X0) * u01[:, 0], Y0 + (Y1 - Y0) * u01[:, 1]], 1).to(device)


def boundary_points(n_per_face, dtype, device):
    s = torch.linspace(0.0, 1.0, n_per_face, dtype=dtype)
    xs, ys = X0 + (X1 - X0) * s, Y0 + (Y1 - Y0) * s
    faces = [torch.stack([torch.full_like(ys, X0), ys], 1), torch.stack([torch.full_like(ys, X1), ys], 1),
             torch.stack([xs, torch.full_like(xs, Y0)], 1), torch.stack([xs, torch.full_like(xs, Y1)], 1)]
    return torch.cat(faces).to(device)


def loss_terms(model, xy_int, xy_bc):
    r1, r2, r3 = ns_residuals(model, xy_int)
    terms = {"xmom": (r1 ** 2).mean(), "ymom": (r2 ** 2).mean(), "cont": (r3 ** 2).mean()}
    if xy_bc is not None:
        out = model(xy_bc)
        ue, ve, _ = exact(xy_bc[:, 0], xy_bc[:, 1])
        terms["bc"] = ((out[:, 0] - ue) ** 2).mean() + ((out[:, 1] - ve) ** 2).mean()
    return terms


def test_errors(model, device, dtype, nx=301, ny=401, chunk=20000):
    """eps_u, eps_v, eps_p (as is), eps_p mean-free, on the Section 2 grid."""
    xs = torch.linspace(X0, X1, nx, dtype=dtype, device=device)
    ys = torch.linspace(Y0, Y1, ny, dtype=dtype, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    with torch.no_grad():
        out = torch.cat([model(xy[i:i + chunk]) for i in range(0, xy.shape[0], chunk)])
    ue, ve, pe = exact(xy[:, 0], xy[:, 1])
    rel = lambda a, b: (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()  # noqa: E731
    p = out[:, 2]
    return rel(out[:, 0], ue), rel(out[:, 1], ve), rel(p, pe), rel(p - p.mean(), pe - pe.mean())


class _BudgetReached(Exception):
    """Raised inside the L-BFGS closure once the wall-clock budget is spent."""


def f1_train(config, seed, budget_s, out_dir, device="cpu", precision="float64",
             test_every=10, max_adam_iters=None, max_lbfgs_calls=None):
    """Train one F1 configuration (a dict with the Section 4.3 search keys).
    ``max_adam_iters`` / ``max_lbfgs_calls`` cap the phases for tests only."""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)
    dtype = torch.float32 if precision == "adam32" else torch.float64
    torch.manual_seed(seed)
    np.random.seed(seed)
    old_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)             # initialize in the run's own precision
    try:
        model = F1Model(config["width"], config["depth"], config["m"], config["sigma_ff"],
                        config["trunk"], config["bc"]).to(device)
    finally:
        torch.set_default_dtype(old_default)
    n_theta = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(seed)
    xy_int = interior_points(config["n_int"], gen, dtype, device)
    xy_bc = boundary_points(N_BC_PER_FACE, dtype, device) if config["bc"] == "soft" else None
    weights = {"xmom": 1.0, "ymom": 1.0, "cont": 1.0}
    if xy_bc is not None:
        weights["bc"] = float(config["lambda_bc"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run = dict(config=config, seed=seed, device=str(device), precision=precision, n_theta=n_theta,
               budget_s=budget_s, torch=torch.__version__, threads=torch.get_num_threads())
    log_f = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(LOG_COLUMNS)
    clock = {"start": None, "excluded": 0.0}
    n_logged = [0]

    def elapsed():
        return time.perf_counter() - clock["start"] - clock["excluded"]

    def write_row(it, phase, terms, lr, grad_norm, t=None, force_eps=False):
        """``t``: the time to log (default now); ``force_eps``: test errors
        whatever ``test_every`` says (the last row of a run)."""
        t = elapsed() if t is None else t
        eps = [""] * 4
        if force_eps or n_logged[0] % test_every == 0:
            t0 = time.perf_counter()
            eps = list(test_errors(model, device, dtype))
            clock["excluded"] += time.perf_counter() - t0
        n_logged[0] += 1
        vals = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in terms.items()}
        total = sum(weights[k] * vals[k] for k in vals)
        log.writerow([it, phase, f"{t:.3f}", repr(total)]
                     + [repr(vals[k]) if k in vals else "" for k in TERMS]
                     + [repr(weights[k]) if k in weights else "" for k in TERMS]
                     + [repr(lr), repr(grad_norm), *eps])
        log_f.flush()
        return total

    def weighted(terms):
        return sum(weights[k] * terms[k] for k in terms)

    end_reason, it, adam_iters, lbfgs_calls, lbfgs_iters, lbfgs_restarts = "budget", 0, 0, 0, 0, 0
    objective = None
    params = list(model.parameters())
    try:
        adam = torch.optim.Adam(params, lr=adam_lr(0, config["eta"]))
        t_adam = config["t_adam"] if max_adam_iters is None else min(config["t_adam"], max_adam_iters)
        clock["start"] = time.perf_counter()
        while it < t_adam and elapsed() < budget_s:
            if config["resample"] and it > 0 and it % RESAMPLE_EVERY == 0:
                xy_int = interior_points(config["n_int"], gen, dtype, device)
            for group in adam.param_groups:
                group["lr"] = adam_lr(it, config["eta"])
            terms = loss_terms(model, xy_int, xy_bc)
            if config["balancing"] and it % BALANCE_EVERY == 0:
                norms = {k: torch.linalg.norm(torch.cat([g.reshape(-1) for g in torch.autograd.grad(
                    t, params, retain_graph=True, allow_unused=True) if g is not None])).item()
                         for k, t in terms.items()}
                weights = balanced_weights(weights, norms)
            loss = weighted(terms)
            if not torch.isfinite(loss):
                end_reason = "failure: non-finite loss"
                break
            adam.zero_grad()
            loss.backward()
            if it % LOG_EVERY == 0:
                gn = torch.linalg.norm(torch.cat([p.grad.reshape(-1) for p in params])).item()
                write_row(it, "adam", terms, adam.param_groups[0]["lr"], gn)
            adam.step()
            it += 1
        adam_iters = it

        if end_reason == "budget" and elapsed() < budget_s:
            if dtype != torch.float64:                  # adam32: L-BFGS in float64
                dtype = torch.float64
                model = model.to(dtype)
                params = list(model.parameters())
                xy_int = xy_int.to(dtype)
                xy_bc = xy_bc.to(dtype) if xy_bc is not None else None

            def evaluate():
                terms = loss_terms(model, xy_int, xy_bc)
                return weighted(terms), {k: float(v.detach()) for k, v in terms.items()}

            objective = LBFGSObjective(params, evaluate)
            best = {"loss": math.inf, "params": None, "t": None}

            def closure():
                # The budget holds inside a call too (a call can run 500
                # iterations): past it, stop at the next evaluation.
                if elapsed() >= budget_s:
                    raise _BudgetReached
                loss = objective.closure()
                t = elapsed()
                if t > budget_s:        # finished past the budget: not a point reached within it
                    raise _BudgetReached
                value = float(loss)
                if value < best["loss"]:
                    best.update(loss=value, params=[q.detach().clone() for q in params], t=t)
                return loss

            def log_current(t=None, force_eps=False):
                """A log row at the current point; its loss, terms and gradient
                come from the evaluation L-BFGS already made there."""
                value = float(objective.closure())
                parts = objective.value()[1]
                gn = torch.linalg.norm(torch.cat([q.grad.reshape(-1) for q in params])).item()
                write_row(adam_iters + lbfgs_iters, "lbfgs", parts, 1.0, gn, t=t, force_eps=force_eps)
                return value

            lbfgs, last, restarted = new_lbfgs(params), None, False
            while elapsed() < budget_s and (max_lbfgs_calls is None or lbfgs_calls < max_lbfgs_calls):
                # One call: up to 500 iterations and 625 evaluations, in pieces of 10.
                call_iters, call_evals0, budget_hit = 0, objective.n_evals, False
                while call_iters < LBFGS_CALL_ITERS:
                    remaining = LBFGS_CALL_EVALS - (objective.n_evals - call_evals0)
                    if remaining < 1:
                        break
                    lbfgs.param_groups[0]["max_eval"] = remaining
                    lbfgs.param_groups[0]["max_iter"] = min(LBFGS_LOG_EVERY, LBFGS_CALL_ITERS - call_iters)
                    n0 = lbfgs.state[params[0]].get("n_iter", 0)
                    try:
                        lbfgs.step(closure)
                    except _BudgetReached:
                        # Interrupted mid-call, the parameters may sit at a line-search
                        # trial point: keep the lowest-loss point L-BFGS evaluated
                        # (none yet: the call never evaluated, parameters untouched),
                        # logged at the time it was evaluated, inside the budget.
                        budget_hit = True
                        if best["params"] is not None:
                            with torch.no_grad():
                                for q, bp in zip(params, best["params"]):
                                    q.copy_(bp)
                    done = lbfgs.state[params[0]].get("n_iter", 0) - n0
                    call_iters += done
                    lbfgs_iters += done
                    if budget_hit:
                        total_now = log_current(t=best["t"], force_eps=True)
                        break
                    total_now = log_current()
                    if not math.isfinite(total_now) or done < lbfgs.param_groups[0]["max_iter"]:
                        break                     # non-finite, or L-BFGS stopped inside the piece
                lbfgs_calls += 1
                it = adam_iters + lbfgs_iters
                if not math.isfinite(total_now):
                    end_reason = "failure: non-finite loss"
                    break
                if budget_hit:
                    break
                if last is not None and total_now >= last:
                    if restarted:
                        end_reason = "criterion: L-BFGS made no progress, also after a restart"
                        break
                    lbfgs, restarted = new_lbfgs(params), True     # one retry with a fresh optimizer
                    lbfgs_restarts += 1
                    continue
                last, restarted = total_now, False
    except Exception:
        end_reason = "failure"
        run["traceback"] = traceback.format_exc()
    finally:
        log_f.close()

    wall = elapsed() if clock["start"] else 0.0
    try:
        eu, ev, ep, epm = test_errors(model, device, dtype)
        final_terms = loss_terms(model, xy_int, xy_bc)
        final_loss = float(weighted(final_terms).detach())
        final_unweighted = unweighted_loss(final_terms)
    except Exception:
        eu = ev = ep = epm = final_loss = final_unweighted = None
    run.update(end_reason=end_reason, iterations=it, adam_iters=adam_iters, lbfgs_calls=lbfgs_calls,
               lbfgs_iters=lbfgs_iters, lbfgs_restarts=lbfgs_restarts,
               lbfgs_evaluations=objective.n_evals if objective is not None else 0,
               wall_s=wall, eps_u=eu, eps_v=ev, eps_p=ep, eps_p_meanfree=epm, final_loss=final_loss,
               final_loss_unweighted=final_unweighted, weights=weights,
               peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
    # The model before run.json: run.json marks the run complete, so a run
    # marked complete always has its model (lilq.saved_models.load_f1).
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2, default=str)
    return run


# The Section 8 check A1: a plain PINN -- Fourier features off, balancing off,
# soft boundary conditions with lambda_bc = 10 -- must reach eps_u <= 1e-3 in
# the full GPU budget. The package leaves the rest open; these are standard
# choices, recorded in the tuning log.
A1_CONFIG = dict(id="A1_plain", family="F1", width=128, depth=4, trunk="shared", m=0, sigma_ff=1.0,
                 bc="soft", lambda_bc=10.0, balancing=False, eta=1e-3, t_adam=20000, n_int=8000,
                 resample=False)
