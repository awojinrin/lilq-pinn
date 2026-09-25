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
* Speed and memory, same step (Addendum v2.1 Section 6 allows it; the
  finite-difference and A2 checks still pass, and tests compare against the
  reference's own evaluation): the interior residual of a whole block of
  points is computed in one batched Taylor-mode pass (the value, first
  derivatives and the two second derivatives the Laplacian needs, propagated
  exactly through the tanh layers; the boundary interpolant's
  parameter-independent derivatives computed once), instead of nested
  forward-mode passes point by point; J^T J and J^T r are accumulated row
  block by row block, so the full Jacobian is never stored; the damping is
  added to the diagonal in place instead of through a dense ``diag``.
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


def taylor_forward(model, params, xy):
    """The network's raw outputs (u~, v~, p) at points ``xy`` (N x 2) and their
    derivatives d/dx, d/dy, d2/dx2, d2/dy2, in one batched pass (the Laplacian
    needs no mixed derivatives). Exact for FourierMLP's layers: the Fourier
    embedding, Linear, and tanh (tanh' = 1 - tanh^2, tanh'' = -2 tanh tanh').
    Returns five (N x 3) tensors."""
    B = params['B']
    z = 2.0 * math.pi * xy @ B.T                       # (N, m)
    kx, ky = 2.0 * math.pi * B[:, 0], 2.0 * math.pi * B[:, 1]
    c, s = torch.cos(z), torch.sin(z)
    h = torch.stack([torch.cat([c, s], 1),
                     torch.cat([-s * kx, c * kx], 1), torch.cat([-s * ky, c * ky], 1),
                     torch.cat([-c * kx ** 2, -s * kx ** 2], 1),
                     torch.cat([-c * ky ** 2, -s * ky ** 2], 1)])
    for idx, layer in enumerate(model.net):
        if isinstance(layer, torch.nn.Linear):
            W, b = params[f'net.{idx}.weight'], params[f'net.{idx}.bias']
            h = h @ W.T                                  # value and derivatives: the same linear map
            h = torch.cat([h[:1] + b, h[1:]], 0)         # the bias only shifts the value
        elif isinstance(layer, torch.nn.Tanh):
            a, ax, ay, axx, ayy = h
            t = torch.tanh(a)
            t1 = 1.0 - t * t
            t2 = -2.0 * t * t1
            h = torch.stack([t, t1 * ax, t1 * ay, t2 * ax * ax + t1 * axx, t2 * ay * ay + t1 * ayy])
        else:
            raise TypeError(f"taylor_forward does not handle {type(layer).__name__}")
    return h[0], h[1], h[2], h[3], h[4]


def boundary_geometry(xy):
    """The parameter-independent parts of the hard boundary conditions at
    ``xy`` -- the Coons interpolants of u and v, and the distance factor l --
    with the derivatives the residual needs, computed once."""
    def geo(z):
        x, y = z[0], z[1]
        return torch.stack([coons(g_u, x, y), coons(g_v, x, y), ell(x, y)])
    val = vmap(geo)(xy)
    jac = vmap(jacfwd(geo))(xy)                          # (N, 3, 2)
    hes = vmap(jacfwd(jacfwd(geo)))(xy)                  # (N, 3, 2, 2)
    out = {}
    for i, name in enumerate(('cu', 'cv', 'l')):
        out[name] = val[:, i]
        out[name + '_x'], out[name + '_y'] = jac[:, i, 0], jac[:, i, 1]
        out[name + '_xx'], out[name + '_yy'] = hes[:, i, 0, 0], hes[:, i, 1, 1]
    return out


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
        self.geometry = boundary_geometry(xy_int)

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

        # One nested forward-mode pass gives the values, the first and the
        # second derivatives together (the reference made three passes:
        # f, jacfwd(f) and jacfwd(jacfwd(f)); same quantities).
        def f_aux(z):
            out = f(z)
            return out, out

        def jac_aux(z):
            J, out = jacfwd(f_aux, has_aux=True)(z)
            return J, (J, out)

        H, (J, out) = jacfwd(jac_aux, has_aux=True)(xy)   # H (3,2,2), J (3,2), out (3,)
        u, v = out[0], out[1]
        ux, uy, vx, vy = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
        px, py = J[2, 0], J[2, 1]
        lap_u = H[0, 0, 0] + H[0, 1, 1]
        lap_v = H[1, 0, 0] + H[1, 1, 1]
        r1 = u * ux + v * uy + px - NU * lap_u
        r2 = u * vx + v * vy + py - NU * lap_v
        r3 = ux + vy
        return torch.stack([r1, r2, r3])

    def interior_reference(self, theta, xy_chunk):
        """The reference's per-point evaluation (kept for the equivalence tests)."""
        return self.s_int * vmap(lambda z: self.point_residual(theta, z))(xy_chunk).reshape(-1)

    def interior(self, theta, rows):
        """Weighted interior residual at the points ``self.xy_int[rows]`` (a
        slice), in one batched Taylor-mode pass: same values as
        :meth:`interior_reference`, far fewer kernels."""
        params = self.unflatten(theta)
        out, dx, dy, dxx, dyy = taylor_forward(self.model, params, self.xy_int[rows])
        G = {k: v[rows] for k, v in self.geometry.items()}

        def bc_field(i, c):
            # u = C_u + l u~ (and v): the product rule for its derivatives.
            f, fx, fy, fxx, fyy = out[:, i], dx[:, i], dy[:, i], dxx[:, i], dyy[:, i]
            val = G[c] + G['l'] * f
            vx = G[c + '_x'] + G['l_x'] * f + G['l'] * fx
            vy = G[c + '_y'] + G['l_y'] * f + G['l'] * fy
            lap = (G[c + '_xx'] + G[c + '_yy'] + (G['l_xx'] + G['l_yy']) * f
                   + 2.0 * (G['l_x'] * fx + G['l_y'] * fy) + G['l'] * (fxx + fyy))
            return val, vx, vy, lap

        u, ux, uy, lap_u = bc_field(0, 'cu')
        v, vx, vy, lap_v = bc_field(1, 'cv')
        px, py = dx[:, 2], dy[:, 2]                       # p = p~: no boundary factor
        r = torch.stack([u * ux + v * uy + px - NU * lap_u,
                         u * vx + v * vy + py - NU * lap_v,
                         ux + vy], dim=1)
        return self.s_int * r.reshape(-1)

    def pin(self, theta):
        p = fields(self.model, self.unflatten(theta), self.pin_xy)[2]
        return (self.s_pin * (p - self.pin_val)).reshape(1)

    def vector(self, theta, chunk):
        parts = [self.interior(theta, slice(i, i + chunk))
                 for i in range(0, self.xy_int.shape[0], chunk)]
        parts.append(self.pin(theta))
        return torch.cat(parts)

    def normal_equations(self, theta, r, chunk):
        """J^T J and J^T r for the residual ``r = self.vector(theta, chunk)``,
        accumulated one row block at a time: the full Jacobian is never held
        (the reference formed J, then J^T J and J^T r; same quantities up to
        the order of floating-point summation)."""
        n = theta.numel()
        H = torch.zeros(n, n, dtype=theta.dtype, device=theta.device)
        g = torch.zeros_like(theta)
        for i in range(0, self.xy_int.shape[0], chunk):
            Jc = jacrev(self.interior, argnums=0)(theta, slice(i, i + chunk))
            H.addmm_(Jc.T, Jc)
            g.addmv_(Jc.T, r[3 * i:3 * i + Jc.shape[0]])
        Jp = jacrev(self.pin)(theta)
        H.addmm_(Jp.T, Jp)
        g.addmv_(Jp.T, r[-1:])
        return H, g

    def jacobian(self, theta, chunk):
        """The full Jacobian (for the finite-difference check; training uses
        :meth:`normal_equations`)."""
        blocks = [jacrev(self.interior, argnums=0)(theta, slice(i, i + chunk))
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
    # J^T J, its damped copy and the Cholesky factor, plus one row block of J
    # (the full J is never formed; see Residual.normal_equations).
    mem_est = 8 * (3 * n_theta * n_theta + 3 * args.chunk * n_theta)
    os.makedirs(args.out, exist_ok=True)
    run = dict(vars(args), n_theta=n_theta, n_rows=n_rows, mem_estimate_bytes=mem_est,
               torch=torch.__version__, threads=torch.get_num_threads())
    print(f"n_theta = {n_theta}, rows = {n_rows}, normal-equation memory ~ {mem_est / 1e9:.2f} GB")

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
        H, g = res.normal_equations(theta, r, args.chunk)
        d = torch.diagonal(H).clamp_min(args.diag_floor)
        accepted = False
        while not accepted:
            A = H.clone()
            A.diagonal().add_(mu * d)             # H + mu diag(d), without a dense diag(d)
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
