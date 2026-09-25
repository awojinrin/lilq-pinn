"""
Reference Levenberg-Marquardt (LM) trainer for a Kovasznay-flow PINN
=====================================================================

Package 1 (v2.1), Component A, baseline family F2.

What this file provides
-----------------------
* A Fourier-feature tanh network for (u, v, p) with hard Dirichlet enforcement
  of u and v through a transfinite (Coons) interpolant of the boundary data.
* The stacked, weighted residual vector r(theta) of the steady incompressible
  Navier-Stokes equations at interior collocation points, plus one pressure
  pin row (the same gauge as LiL-Q).
* The Jacobian J = dr/dtheta assembled with torch.func (jacrev over a vmapped,
  functional residual), in row chunks to bound peak memory.
* The damped Gauss-Newton (LM) step
      (J^T J + mu * diag(J^T J)) dtheta = -J^T r
  solved by Cholesky, with the damping update of Package 1 Section 4.2:
  mu0 = 1e-3, mu *= 0.2 on an accepted step, mu *= 5 on a rejected step,
  a step is accepted iff the sum of squared residuals decreases.
* Stopping by a wall-clock budget or when the relative loss decrease over the
  last 20 accepted steps is below 1e-10.
* A CSV log with the columns of Package 1 Section 6 (baseline log.csv).

What it deliberately does not do
--------------------------------
It does not tune anything. Width, depth, the number of Fourier features m, the
Fourier scale sigma_FF, and N_int are arguments; the random search of Section 4.3
chooses them. The exact solution is used only for the boundary data and for
the test errors, never in the interior loss.

Precision is float64 throughout. Everything runs on the device given by
--device (cpu or cuda).

Usage (small smoke test, a few minutes on a laptop CPU):
    python lm_kovasznay_reference.py --width 20 --depth 2 --m 16 --n_int 400 \
        --budget_s 120 --out smoke

The only external dependencies are torch >= 2.1 and numpy.

Provenance in this repository
-----------------------------
Supplied with Addendum v2.1 (``lm_reference/lm_kovasznay_reference.py``).
Changes, none of them to the algorithm (step equation, damping schedule,
acceptance rule, stopping rule):
* ``torch.set_default_dtype(torch.float64)`` moved from import time into
  ``lm_train`` and ``main``, so importing this module does not change
  PyTorch's default precision for the rest of the process.
* The test errors (post-hoc reading only) are computed off the clock:
  their time is excluded from ``t_cum_s``, the budget and ``wall_s``, as
  for F1, so neither family's budget pays for evaluation.
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import functional_call, jacfwd, jacrev, vmap

# ----------------------------------------------------------------------------
# Problem: Kovasznay flow, Re = 40, Omega = [-0.5, 1] x [-0.5, 1.5]
# ----------------------------------------------------------------------------
RE = 40.0
NU = 1.0 / RE
LAM = 1.0 / (2.0 * NU) - math.sqrt(1.0 / (4.0 * NU ** 2) + 4.0 * math.pi ** 2)
X0, X1, Y0, Y1 = -0.5, 1.0, -0.5, 1.5


def exact(x, y):
    """Exact (u, v, p). Works for numpy arrays and torch tensors."""
    lib = torch if isinstance(x, torch.Tensor) else np
    e = lib.exp(LAM * x)
    u = 1.0 - e * lib.cos(2.0 * math.pi * y)
    v = LAM / (2.0 * math.pi) * e * lib.sin(2.0 * math.pi * y)
    p = 0.5 * (1.0 - lib.exp(2.0 * LAM * x))
    return u, v, p


# ----------------------------------------------------------------------------
# Hard boundary conditions for u and v
# ----------------------------------------------------------------------------
def coons(g, x, y):
    """Transfinite (Coons) interpolant on the rectangle of the boundary data g.

    g(x, y) is only ever evaluated on the four edges, so the interior values of
    the exact solution never enter the network output.
    """
    s = (x - X0) / (X1 - X0)
    t = (y - Y0) / (Y1 - Y0)
    x0 = torch.full_like(x, X0); x1 = torch.full_like(x, X1)
    y0 = torch.full_like(y, Y0); y1 = torch.full_like(y, Y1)
    edges = ((1 - s) * g(x0, y) + s * g(x1, y)
             + (1 - t) * g(x, y0) + t * g(x, y1))
    corners = ((1 - s) * (1 - t) * g(x0, y0) + s * (1 - t) * g(x1, y0)
               + (1 - s) * t * g(x0, y1) + s * t * g(x1, y1))
    return edges - corners


def g_u(x, y):
    return exact(x, y)[0]


def g_v(x, y):
    return exact(x, y)[1]


def ell(x, y):
    """Normalized product of the four face distances; zero on the boundary."""
    s = (x - X0) / (X1 - X0)
    t = (y - Y0) / (Y1 - Y0)
    return 16.0 * s * (1 - s) * t * (1 - t)


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
class FourierMLP(torch.nn.Module):
    """[cos(2 pi B x), sin(2 pi B x)] -> tanh MLP -> (u~, v~, p).

    B is a fixed Gaussian matrix (not trained). Shared trunk, three outputs.
    """

    def __init__(self, width, depth, m, sigma_ff, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("B", sigma_ff * torch.randn(m, 2, generator=gen))
        torch.manual_seed(seed)
        layers = [torch.nn.Linear(2 * m, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 3)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy):                       # xy: (..., 2)
        z = 2.0 * math.pi * xy @ self.B.T
        return self.net(torch.cat([torch.cos(z), torch.sin(z)], dim=-1))


def fields(model, params, xy):
    """(u, v, p) at a single point xy of shape (2,), hard BCs applied."""
    out = functional_call(model, params, (xy,))
    x, y = xy[0], xy[1]
    L = ell(x, y)
    u = coons(g_u, x, y) + L * out[0]
    v = coons(g_v, x, y) + L * out[1]
    p = out[2]
    return torch.stack([u, v, p])


# ----------------------------------------------------------------------------
# Residual r(theta)
# ----------------------------------------------------------------------------
class Residual:
    """Stacked weighted residual: interior (x-mom, y-mom, continuity) rows
    scaled by sqrt(w_int / n_int), plus one pressure pin row scaled by
    sqrt(w_pin). The loss is 0.5 * ||r||^2."""

    def __init__(self, model, xy_int, pin_xy, w_int=1.0, w_pin=1.0):
        self.model = model
        self.names = [n for n, _ in model.named_parameters()]
        self.shapes = [p.shape for _, p in model.named_parameters()]
        self.sizes = [p.numel() for _, p in model.named_parameters()]
        self.buffers = dict(model.named_buffers())
        self.xy_int = xy_int
        self.pin_xy = pin_xy
        self.pin_val = exact(pin_xy[0], pin_xy[1])[2]
        self.s_int = math.sqrt(w_int / xy_int.shape[0])
        self.s_pin = math.sqrt(w_pin)

    # flat parameter vector <-> dict
    def unflatten(self, theta):
        out, i = {}, 0
        for n, shp, k in zip(self.names, self.shapes, self.sizes):
            out[n] = theta[i:i + k].view(shp)
            i += k
        out.update(self.buffers)
        return out

    def point_residual(self, theta, xy):
        params = self.unflatten(theta)
        f = lambda z: fields(self.model, params, z)
        J = jacfwd(f)(xy)                         # (3, 2): d(u,v,p)/d(x,y)
        H = jacfwd(jacfwd(f))(xy)                 # (3, 2, 2)
        u, v, _ = f(xy)
        ux, uy, vx, vy = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
        px, py = J[2, 0], J[2, 1]
        lap_u = H[0, 0, 0] + H[0, 1, 1]
        lap_v = H[1, 0, 0] + H[1, 1, 1]
        r1 = u * ux + v * uy + px - NU * lap_u
        r2 = u * vx + v * vy + py - NU * lap_v
        r3 = ux + vy
        return torch.stack([r1, r2, r3])

    def interior(self, theta, xy_chunk):
        return self.s_int * vmap(lambda z: self.point_residual(theta, z))(xy_chunk).reshape(-1)

    def pin(self, theta):
        p = fields(self.model, self.unflatten(theta), self.pin_xy)[2]
        return (self.s_pin * (p - self.pin_val)).reshape(1)

    def vector(self, theta, chunk):
        parts = [self.interior(theta, self.xy_int[i:i + chunk])
                 for i in range(0, self.xy_int.shape[0], chunk)]
        parts.append(self.pin(theta))
        return torch.cat(parts)

    def jacobian(self, theta, chunk):
        blocks = [jacrev(self.interior, argnums=0)(theta, self.xy_int[i:i + chunk])
                  for i in range(0, self.xy_int.shape[0], chunk)]
        blocks.append(jacrev(self.pin)(theta))
        return torch.cat(blocks, dim=0)


# ----------------------------------------------------------------------------
# Test errors (Package 1 Section 2)
# ----------------------------------------------------------------------------
def test_errors(model, res, theta, device, nx=301, ny=401, chunk=20000):
    xs = torch.linspace(X0, X1, nx, device=device)
    ys = torch.linspace(Y0, Y1, ny, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    params = res.unflatten(theta)
    with torch.no_grad():
        out = torch.cat([vmap(lambda z: fields(model, params, z))(xy[i:i + chunk])
                         for i in range(0, xy.shape[0], chunk)])
    ue, ve, pe = exact(xy[:, 0], xy[:, 1])
    rel = lambda a, b: (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()
    pm, pem = out[:, 2] - out[:, 2].mean(), pe - pe.mean()
    return rel(out[:, 0], ue), rel(out[:, 1], ve), rel(out[:, 2], pe), rel(pm, pem)


# ----------------------------------------------------------------------------
# Levenberg-Marquardt
# ----------------------------------------------------------------------------
def lm_train(args):
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    model = FourierMLP(args.width, args.depth, args.m, args.sigma_ff, args.seed).to(device)
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).to(device)
    n_theta = theta.numel()
    if n_theta > args.max_params:
        raise ValueError(f"n_theta = {n_theta} exceeds {args.max_params}")

    gen = torch.Generator().manual_seed(args.seed)
    u01 = torch.rand(args.n_int, 2, generator=gen)
    xy_int = torch.stack([X0 + (X1 - X0) * u01[:, 0], Y0 + (Y1 - Y0) * u01[:, 1]], 1).to(device)
    pin_xy = torch.tensor([X0, Y0], device=device)     # corner pin, as in LiL-Q
    res = Residual(model, xy_int, pin_xy, args.w_int, args.w_pin)

    n_rows = 3 * args.n_int + 1
    mem_est = 8 * (n_rows * n_theta + 2 * n_theta * n_theta)
    os.makedirs(args.out, exist_ok=True)
    run = dict(vars(args), n_theta=n_theta, n_rows=n_rows, mem_estimate_bytes=mem_est,
               torch=torch.__version__, threads=torch.get_num_threads())
    print(f"n_theta = {n_theta}, rows = {n_rows}, J + H memory ~ {mem_est / 1e9:.2f} GB")

    f = open(os.path.join(args.out, "log.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["iter", "phase", "t_cum_s", "loss_total", "loss_xmom", "loss_ymom",
                "loss_cont", "loss_bc", "w_xmom", "w_ymom", "w_cont", "w_bc",
                "lr_or_mu", "grad_norm", "eps_u", "eps_v", "eps_p", "eps_p_meanfree",
                "accepted", "n_evals"])

    def split_loss(r):
        ri = r[:-1].reshape(-1, 3)
        return [0.5 * float((ri[:, j] ** 2).sum()) for j in range(3)] + [0.5 * float(r[-1] ** 2)]

    mu = args.mu0
    r = res.vector(theta, args.chunk)
    loss = 0.5 * float(r @ r)
    history = [loss]
    n_evals = 1
    t0 = time.perf_counter()
    excluded = 0.0                              # test-error time, off the clock
    end_reason = "budget"
    it = 0
    while True:
        J = res.jacobian(theta, args.chunk)
        g = J.T @ r
        H = J.T @ J
        d = torch.diagonal(H).clamp_min(args.diag_floor)
        accepted = False
        while not accepted:
            A = H + mu * torch.diag(d)
            Lc, info = torch.linalg.cholesky_ex(A)
            if info.item() != 0:                  # not positive definite: damp more
                mu *= 5.0
                continue
            step = -torch.cholesky_solve(g.unsqueeze(1), Lc).squeeze(1)
            r_new = res.vector(theta + step, args.chunk)
            n_evals += 1
            loss_new = 0.5 * float(r_new @ r_new)
            if loss_new < loss:
                theta, r, loss = theta + step, r_new, loss_new
                mu *= 0.2
                accepted = True
            else:
                mu *= 5.0
            if mu > 1e16:
                end_reason = "mu_overflow"
                break
            if time.perf_counter() - t0 - excluded > args.budget_s:
                break
        it += 1
        t = time.perf_counter() - t0 - excluded
        t_te = time.perf_counter()
        eu, ev, ep, epm = test_errors(model, res, theta, device) if it % args.test_every == 0 else [""] * 4
        excluded += time.perf_counter() - t_te
        comp = split_loss(r)
        w.writerow([it, "lm", f"{t:.3f}", repr(loss), *map(repr, comp),
                    args.w_int, args.w_int, args.w_int, args.w_pin,
                    repr(mu), repr(float(torch.linalg.norm(g))), eu, ev, ep, epm, int(accepted), n_evals])
        f.flush()
        if accepted:
            history.append(loss)
        if end_reason == "mu_overflow":
            break
        if len(history) > 20 and (history[-21] - history[-1]) / history[-21] < 1e-10:
            end_reason = "stagnation_20"
            break
        if t > args.budget_s:
            end_reason = "budget"
            break
        if args.max_steps and it >= args.max_steps:
            end_reason = "max_steps"
            break

    wall_s = time.perf_counter() - t0 - excluded
    eu, ev, ep, epm = test_errors(model, res, theta, device)
    run.update(end_reason=end_reason, steps=it, n_evals=n_evals, final_loss=loss,
               eps_u=eu, eps_v=ev, eps_p=ep, eps_p_meanfree=epm,
               wall_s=wall_s,
               peak_gpu_bytes=(torch.cuda.max_memory_allocated() if device.type == "cuda" else None))
    f.close()
    json.dump(run, open(os.path.join(args.out, "run.json"), "w"), indent=2)
    torch.save(theta.cpu(), os.path.join(args.out, "theta.pt"))
    print(json.dumps({k: run[k] for k in ("end_reason", "steps", "final_loss", "eps_u", "eps_v", "eps_p_meanfree", "wall_s")}, indent=2))
    return run


def main():
    torch.set_default_dtype(torch.float64)
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--sigma_ff", type=float, default=1.0)
    ap.add_argument("--n_int", type=int, default=2000)
    ap.add_argument("--w_int", type=float, default=1.0)
    ap.add_argument("--w_pin", type=float, default=1.0)
    ap.add_argument("--mu0", type=float, default=1e-3)
    ap.add_argument("--diag_floor", type=float, default=1e-12)
    ap.add_argument("--budget_s", type=float, default=3600.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--max_params", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=256, help="interior points per Jacobian block")
    ap.add_argument("--test_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="lm_run")
    lm_train(ap.parse_args())


if __name__ == "__main__":
    main()
