"""
Package 5, item 2: LiL-N by nonlinear least squares
===================================================

LiL-N and LiL-Q minimize the same weighted collocation residual N(beta) over
the same trial space. Section 4 asks whether general nonlinear least-squares
solvers reach LiL-Q's point, and at what cost.

**Problems** (Package 4's, unchanged):
- Burgers: sine x Chebyshev on the tensor Gauss-Legendre set
  (``p4_1_burgers_gauss.Rows``), p = 10, 18, 26, 30; the relative L2 error on
  the 201 x 201 grid against Package 4's delta_P;
- Poisson-Boltzmann around three inclusions, seed 0, on the shipped Halton
  points (``p4_2_pb_inclusions.lilq_set``), p = 10, 18, 22; the H2 error (the
  headline, as in the advisor's script) and the L2 error, each against its
  delta_P.

**Solvers**, all from zero coefficients:
- ``trf``, ``lm``, ``dogbox``: ``scipy.optimize.least_squares`` with the
  analytic Jacobian, the linearized collocation matrix A(beta) (the Newton
  identity: N(beta + d) = N(beta) + A(beta) d + O(d^2)); xtol = ftol = gtol =
  1e-12; at most 200 evaluations (the advisor's script; his text gives none);
- ``lbfgs``: the four-method runs' L-BFGS (``lilq.solvers.solve_lil_n``:
  PyTorch, history 100, strong Wolfe) with tolerances 0 (``stall_rule='f1'``,
  the four-method stall control) and no target, to 2,000 iterations, on the
  loss ||N||^2 with the gradient 2 A^T N;
- ``lbfgs_scipy``: the advisor's ``L-BFGS-B`` (maxcor 100, ftol = gtol = 0,
  2,000 iterations, 40,000 evaluations) on ||N||^2 / 2, beside ``lbfgs``,
  since his expected values come from it;
- ``lilq``: Algorithm 1 to the termination rule (tau_chi 0.1, tau_r 0.01,
  n_s 2), returning the iterate after the rule's step, as Package 4's timings.

**Timing** (the reading of Section 4, DECISIONS.md): Package 4's protocol
(``p4_common.timed``: an untimed warm-up, then the median of three, each the
median of five if under 1 s) on one whole node. The run that gives the
coefficients is the warm-up. The clock covers the solver
from zero coefficients to its return; the collocation set and the basis
matrices are built before it, as in the advisor's script. The errors are
evaluated after the clock.
The two L-BFGS solvers are timed by their one logged pass (``ONE_PASS``): the
advisor's reply to the cost check, 14 November 2026, answer 2. Their runs end
at the 2,000-iteration cap, so one pass is their time; the least-squares
solvers and LiL-Q keep the protocol.

**Output** under ``<out>/P5_2_nlls/``: ``<problem>_p<p>/<solver>/result.json``
and ``beta.npy`` (LiL-Q also ``betas.npy``, every iterate);
``terminal.csv``; ``compare_expected.csv`` against
``expected_nlls_liln.json``.

Usage::

    python experiments/p5_2_nlls.py run --out <root> --reference-dir <dir> --points <dir> [--problem burgers --p 10] [--no-timing]
    python experiments/p5_2_nlls.py summarize --out <root> [--expected expected_nlls_liln.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np
import scipy
from scipy.optimize import least_squares, minimize

from experiments import p4_1_burgers_gauss as p41
from experiments import p4_2_pb_inclusions as p42
from experiments.p4_common import N_S, iterate_to_rule, timed, to_float, write_csv
from lilq.provenance import capture_blas_thread_env
from problems import poisson_boltzmann as pb

ITEM = 'P5_2_nlls'
SIZES = {'burgers': (10, 18, 26, 30), 'pb': (10, 18, 22)}
PB_SEED = 0
LS_METHODS = ('trf', 'lm', 'dogbox')
LS_TOL = 1e-12
LS_MAX_NFEV = 200
LBFGS_MAX_IT = 2000
LBFGS_SCIPY_MAXCOR = 100
SOLVERS = ('lilq',) + LS_METHODS + ('lbfgs', 'lbfgs_scipy')
ONE_PASS = ('lbfgs', 'lbfgs_scipy')      # timed by their logged pass (the advisor's reply, 14 November 2026)
HEADLINE = {'burgers': 'eL2', 'pb': 'eH2'}


# ---------------------------------------------------------------- the problems

class Problem:
    """One system of Package 4: ``rows`` (``assemble(beta) -> (A, f)``, ``residual(beta)``),
    ``errors(beta) -> {name: value}`` and ``delta`` (the same names)."""

    def __init__(self, name, p, rows, errors, delta):
        self.name, self.p, self.rows = name, p, rows
        self.P = p * p
        self.errors, self.delta = errors, delta

    def residual(self, beta):
        return self.rows.residual(beta)

    def jacobian(self, beta):
        return self.rows.assemble(beta)[0]


def burgers(p, reference_dir):
    space = p41.Space(p)
    errs = p41.Errors(p41.load_reference(reference_dir))
    return Problem('burgers', p, p41.Rows(space), lambda b: {'eL2': errs.error(space, b)},
                   {'eL2': errs.delta_P(space)})


def poisson_boltzmann(p, points_dir, norms=None):
    norms = norms or pb.Norms()
    S = p42.lilq_set(p42.Halton(points_dir), p, PB_SEED)
    d0, d2 = norms.best(p)
    return Problem('pb', p, S, lambda b: dict(zip(('eL2', 'eH2'), norms.errors(b, p))), {'eL2': d0, 'eH2': d2})


# ---------------------------------------------------------------- the solvers: (beta, counts, status)

def solve_lilq(prob, on_iterate=None):
    beta, k_s, _, _ = iterate_to_rule(prob.rows, np.zeros(prob.P), N_S, on_iterate=on_iterate)
    return beta, {'solves': None if k_s is None else k_s + 1}, 'rule fired' if k_s is not None else 'K_max'


def solve_ls(prob, method):
    s = least_squares(prob.residual, np.zeros(prob.P), jac=prob.jacobian, method=method,
                      xtol=LS_TOL, ftol=LS_TOL, gtol=LS_TOL, max_nfev=LS_MAX_NFEV)
    return s.x, {'nfev': int(s.nfev), 'njev': None if s.njev is None else int(s.njev)}, \
        f'{int(s.status)}: {s.message}'


class _SquaredResidual:
    """||N(beta)||^2 as a torch scalar with the analytic gradient 2 A(beta)^T N(beta)."""

    def __init__(self, prob):
        import torch

        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, beta):
                b = beta.detach().cpu().numpy()
                r = prob.residual(b)
                ctx.save_for_backward(torch.from_numpy(2.0 * (prob.jacobian(b).T @ r)))
                return beta.new_tensor(float(r @ r))

            @staticmethod
            def backward(ctx, grad):
                (g,) = ctx.saved_tensors
                return grad * g
        self.fn = Fn

    def __call__(self, beta):
        loss = self.fn.apply(beta)
        return loss, loss.detach(), 0.0, loss.new_tensor(0.0)


def solve_lbfgs(prob):
    import torch
    from lilq.solvers import solve_lil_n
    beta, _, summary = solve_lil_n(_SquaredResidual(prob), np.zeros(prob.P), torch.device('cpu'),
                                   max_iterations=LBFGS_MAX_IT, R_tol=0.0, verbose=False, stall_rule='f1')
    status = 'stall (f1)' if summary['optimizer_stall'] else (
        'iteration cap' if summary['total_iterations'] >= LBFGS_MAX_IT else 'stopped')
    return beta, {'iterations': summary['total_iterations'], 'evaluations': summary['total_line_searches'],
                  'restarts': summary['lbfgs_restarts']}, status


def solve_lbfgs_scipy(prob):
    def fun(b):
        r = prob.residual(b)
        return 0.5 * float(r @ r), prob.jacobian(b).T @ r
    m = minimize(fun, np.zeros(prob.P), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=LBFGS_MAX_IT, maxfun=20 * LBFGS_MAX_IT, ftol=0, gtol=0,
                              maxcor=LBFGS_SCIPY_MAXCOR))
    return m.x, {'iterations': int(m.nit), 'evaluations': int(m.nfev)}, f'{int(m.status)}: {m.message}'


def solver(name):
    if name == 'lilq':
        return solve_lilq
    if name in LS_METHODS:
        return lambda prob: solve_ls(prob, name)
    return {'lbfgs': solve_lbfgs, 'lbfgs_scipy': solve_lbfgs_scipy}[name]


def _count(counts):
    """The count the timing runs must repeat (a deterministic run gives the same)."""
    return next((v for v in counts.values() if v is not None), None)


# ---------------------------------------------------------------- one run

def run_dir_of(out_root, problem, p, name):
    return Path(out_root) / ITEM / f'{problem}_p{p}' / name


def run(prob, name, out_root, timing=True):
    fn = solver(name)
    iterates = [np.zeros(prob.P)] if name == 'lilq' else None     # Section 2: every iterate of a LiL run
    t0 = time.perf_counter()
    beta, counts, status = solve_lilq(prob, iterates.append) if iterates is not None else fn(prob)
    first_s = time.perf_counter() - t0
    out = {'problem': prob.name, 'p': prob.p, 'P': prob.P, 'solver': name, **counts, 'status': status}
    if timing and name in ONE_PASS:
        out.update(time_s=first_s, timing_counts=[_count(counts)], timing_matches=True,
                   timing="one logged pass (the advisor's reply to the cost check, 14 November 2026)")
    elif timing:
        def once():
            s = time.perf_counter()
            _, c, _ = fn(prob)
            return time.perf_counter() - s, _count(c)
        tj = timed(once, warmup=(first_s, _count(counts)))      # the run above is the warm-up
        out.update(time_s=tj['median_s'], timing_spread=tj['spread'], timing_counts=tj['solves'],
                   timing_matches=tj['solves'] == ([_count(counts)] if _count(counts) is not None else []),
                   timing=tj)
    else:
        out.update(time_s=first_s, timing='single untimed run (laptop)')
    errs = prob.errors(beta)
    out.update(norm_N=float(np.linalg.norm(prob.residual(beta))), **errs,
               **{f'delta_{k}': prob.delta[k] for k in errs},
               **{f'{k}_over_delta': errs[k] / prob.delta[k] for k in errs},
               thread_env=capture_blas_thread_env(), scipy_version=scipy.__version__)
    d = run_dir_of(out_root, prob.name, prob.p, name)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'beta.npy', beta)
    if iterates is not None:
        np.save(d / 'betas.npy', np.array(iterates))                  # beta^(0..K)
    (d / 'result.json').write_text(json.dumps(out, indent=2, default=str))
    return out


def run_all(out_root, reference_dir, points_dir, problems=None, sizes=None, solvers=SOLVERS, timing=True):
    norms = None
    for problem in problems or SIZES:
        for p in sizes or SIZES[problem]:
            if problem == 'burgers':
                prob = burgers(p, reference_dir)
            else:
                norms = norms or pb.Norms()
                prob = poisson_boltzmann(p, points_dir, norms)
            for name in solvers:
                r = run(prob, name, out_root, timing)
                h = HEADLINE[problem]
                print(f"{problem} p={p} {name}: {r['time_s']:.3g} s, ||N|| {r['norm_N']:.4g}, {h} {r[h]:.4g} "
                      f"({r[h + '_over_delta']:.3g} x delta), {r['status']}", flush=True)


# ---------------------------------------------------------------- the summary

TERMINAL_KEYS = ('problem', 'p', 'P', 'solver', 'solves', 'nfev', 'njev', 'iterations', 'evaluations', 'restarts',
                 'status', 'time_s', 'timing_spread', 'timing_matches', 'norm_N', 'eL2', 'delta_eL2', 'eL2_over_delta',
                 'eH2', 'delta_eH2', 'eH2_over_delta')


def summarize(out_root, expected=None):
    root = Path(out_root) / ITEM
    rows = []
    for f in sorted(root.glob('*_p*/*/result.json'), key=lambda f: (f.parent.parent.name.split('_p')[0],
                                                                       int(f.parent.parent.name.split('_p')[1]),
                                                                       SOLVERS.index(f.parent.name))):
        r = json.loads(f.read_text())
        rows.append({k: r.get(k, '') for k in TERMINAL_KEYS})
    write_csv(root / 'terminal.csv', rows)
    cmp_rows = compare_expected(rows, expected) if expected else []
    if cmp_rows:
        write_csv(root / 'compare_expected.csv', cmp_rows)
    return rows, cmp_rows


EXPECTED_SOLVER = {'lilq': 'lilq', 'trf': 'trf', 'lm': 'lm', 'dogbox': 'dogbox', 'lbfgs_scipy': 'lbfgs_2000'}


def compare_expected(rows, expected_path):
    """Our values beside the advisor's (``expected_nlls_liln.json``): the error, ||N|| and the counts."""
    exp = {}
    for e in json.loads(Path(expected_path).read_text()):
        problem, p = e['case'].split()
        exp[('pb' if problem == 'pb' else 'burgers', int(p[1:]))] = e
    out = []
    for r in rows:
        e = exp.get((r['problem'], int(r['p'])))
        if e is None:
            continue
        theirs = e.get(EXPECTED_SOLVER.get(r['solver'], ''), None)
        h = HEADLINE[r['problem']]
        row = {'problem': r['problem'], 'p': r['p'], 'solver': r['solver'], 'error': r[h],
               'error_expected': theirs['err'] if theirs else '', 'norm_N': r['norm_N'],
               'norm_N_expected': theirs['N'] if theirs else ''}
        if theirs:
            row['error_rel_diff'] = abs(to_float(r[h]) - theirs['err']) / theirs['err']
            row['norm_N_rel_diff'] = abs(to_float(r['norm_N']) - theirs['N']) / theirs['N']
            for ours, key in (('solves', 'solves'), ('nfev', 'nfev'), ('njev', 'njev'), ('iterations', 'it'),
                              ('evaluations', 'nfev')):
                if r.get(ours) not in ('', None) and key in theirs:
                    row[f'{ours}'], row[f'{ours}_expected'] = r[ours], theirs[key]
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 5, item 2: LiL-N by nonlinear least squares.')
    ap.add_argument('stage', choices=('run', 'summarize'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir', help="Burgers' Cole-Hopf reference (burgers_cole_hopf.npz)")
    ap.add_argument('--points', help="Package 4's shipped Halton points (halton_seed0.npz)")
    ap.add_argument('--problem', choices=tuple(SIZES))
    ap.add_argument('--p', type=int)
    ap.add_argument('--solver', choices=SOLVERS)
    ap.add_argument('--no-timing', action='store_true', help='one run per solver, no protocol (laptop checks)')
    ap.add_argument('--expected', help='expected_nlls_liln.json')
    a = ap.parse_args(argv)
    if a.stage == 'run':
        run_all(a.out, a.reference_dir, a.points, [a.problem] if a.problem else None, [a.p] if a.p else None,
                (a.solver,) if a.solver else SOLVERS, timing=not a.no_timing)
    rows, cmp_rows = summarize(a.out, a.expected)
    print(f'{len(rows)} runs; {len(cmp_rows)} beside the expected values')
    return 0


if __name__ == '__main__':
    sys.exit(main())
