"""
Package 5, item 3: cost at a common accuracy -- the LiL runs
============================================================

Section 6 of the Package 5 instructions, the methods on LiL-Q's trial space,
beside the network runs of ``p5_3_networks``:

- **LiL-Q**: Algorithm 1 to the termination rule (tau_chi 0.1, tau_r 0.01,
  n_s 2), returning the iterate after the rule's step;
- **LiL-N, trust region** (``trf``): ``scipy.optimize.least_squares`` with the
  analytic Jacobian, as in item 2 (tolerances 1e-12, at most 200 evaluations;
  "the trust-region runs end by their own criteria");
- **LiL-N, L-BFGS**: the four-method runs' PyTorch L-BFGS
  (``lilq.solvers.solve_lil_n``) with tolerances 0 (``stall_rule='f1'``), no
  target and no cap, to the 600 s budget (``lilq.metrics.RunProbe``), on the
  loss ||N(beta)||^2. N is the torch residual of each system
  (``torch_residual``), the same code on the CPU node and the A100, its
  gradient by autograd (the user's decision, 16 November 2026);
- **the square collocation** (Bratu only): ``P2_3_classical``'s SQ-CGL Newton
  at p = 7, 12, 17 (25, 100, 225 free coefficients), settings unchanged.

**The systems** (the advisor's reply of 15 November 2026, answer 1):
- Bratu: P2-16's tensor Chebyshev on the CGL set at N/P = 10 with the
  boundary weight 10^4 (item 1's ``ScalarRows``), P = 25, 100, 225, from zero;
- Burgers: Package 4's sine x Chebyshev on the Gauss-Legendre set
  (``p4_1_burgers_gauss.Rows``), p = 6, 10, ..., 30, from zero;
- Buckley-Leverett, viscous: Package 3's lifted sine x Chebyshev with the
  boundary and initial weight 10^4 (item 1's ``bl_setup``), P = 64, 256, 576,
  1,024, from Package 3's initial guess (the fit of the initial condition).
LiL-N starts where LiL-Q starts: it minimizes the same objective.

**The error** is the relative L2 error of the network runs (Bratu: the p = 48
spectral reference; Burgers: Cole-Hopf; Buckley-Leverett: the stored
finite-difference reference), each on its 201 x 201 grid.

**The clocks.** Every clock starts at the solver from the initial
coefficients; the collocation set, the basis matrices and (L-BFGS) the move
to the device are built before it and timed beside it (``setup_s``).
- LiL-Q, ``trf``, square collocation, "the usual protocol": one logged pass
  stores every iterate (LiL-Q, ``trf``: at the Jacobian's evaluations, which
  scipy makes at every accepted iterate) and serves as the warm-up; then three
  timed passes record a timestamp at every iterate and nothing else, each the
  median of five if a pass is under 1 s (``p4_common``'s protocol). The time
  of an iterate is the median of its timestamps. Every timed pass must give
  the logged pass's iterate count and final coefficients (to ``BETA_RTOL``).
  The square collocation keeps P2-3's measure: the sum of the assembly and
  solve times of the Newton steps before the iterate.
- LiL-N L-BFGS: one logged pass carries the timing (Section 6), the error
  every 10 iterations off the clock (``RunProbe``). The coefficients at every
  logged iterate are stored (``betas_logged.npy``): every iterate of a 600 s
  L-BFGS run would be up to some 10^5 vectors.
- Each job first warms up every method on its device, untimed (``warm_up``).

**Outputs** under ``<out>/P5_3_cost/lil/``:
``<bench>_P<P>/<method>_<device>/``: ``run.json``; ``history.csv``
(``iteration, evaluations, time_s, loss, error``, as the network runs);
``betas.npy`` (every iterate; L-BFGS ``betas_logged.npy``); ``beta.npy``;
LiL-L-BFGS also ``metrics.csv`` (the loss at every iteration).
``checks/c3_burgers.json``: C3, LiL-Q on Burgers against Package 4.
``warmup/`` and ``provenance/``: one record per process, ``<bench>_<device>[_<tag>]``
(``--tag``: two processes of a job, or two jobs, never write the same record).

Usage::

    python experiments/p5_3_lil.py run --bench burgers --out <root> --reference-dir <dir> \\
        [--P 36 100] [--methods lilq trf lbfgs square] [--device cpu] [--budget 600] [--tag cpu-bratu]
    python experiments/p5_3_lil.py c3 --out <root> --package4 <Package 4 results root>
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np
import scipy
from scipy.optimize import least_squares

from experiments import p2_16_certified as p216
from experiments import p3_2_bl_certified as p32
from experiments import p4_1_burgers_gauss as p41
from experiments import p5_1_weights as p51
from experiments.p4_common import N_S, REPEATS, SHORT_REPEATS, SHORT_RUN_S, iterate_to_rule, read_csv, write_csv
from experiments.p5_3_networks import BUDGET_S, HISTORY_COLUMNS, ITEM, NO_CAP, finished, write_history
from lilq.provenance import OPENMP_KINDS, capture_blas_thread_env, job_dir, require_openmp, save_provenance
from lilq.source_lock import current_commit

LAM = 1e4                                          # the advisor's reply of 15 November 2026, answer 1
SIZES = {'bratu': (25, 100, 225), 'burgers': tuple(p * p for p in p41.SIZES), 'bl': (64, 256, 576, 1024)}
SQUARE_P = {25: 7, 100: 12, 225: 17}               # SQ-CGL: (p - 2)^2 free coefficients
METHODS = ('lilq', 'trf', 'lbfgs', 'square')
LABELS = {'lilq': 'LiL-Q', 'trf': 'LiL-N (trf)', 'lbfgs': 'LiL-N (L-BFGS)', 'square': 'square collocation (SQ-CGL)'}
DEVICE_METHODS = ('lbfgs',)                        # the only LiL method that runs on the A100 (Section 6)
TRF_TOL = 1e-12                                    # item 2's
TRF_MAX_NFEV = 200
EVERY = 10                                         # L-BFGS: the error every 10 iterations (Section 6)
BETA_RTOL = 1e-10                                  # a timed pass's final coefficients against the logged pass's
C3_RTOL = 1e-8                                     # C3: the errors per iterate against Package 4's
WARMUP_ITERATIONS = 5


def isqrt(P):
    p = int(round(math.sqrt(P)))
    assert p * p == P, P
    return p


# ---------------------------------------------------------------- the systems

class Problem:
    """One LiL system of item 3: ``residual``, ``jacobian`` (numpy, the analytic
    Jacobian of item 2), ``error``, ``beta0``, ``torch_residual(device)``."""

    def __init__(self, bench, P, reference_dir, lam=LAM):
        """``setup_s``: the system and the initial coefficients (the method's setup);
        the reference and the error evaluator are built outside it."""
        self.bench, self.P, self.p = bench, P, isqrt(P)
        t0 = time.perf_counter()
        if bench == 'bratu':
            self.rows = p51.ScalarRows('bratu', P, lam)
            self.beta0, self.guess = np.zeros(P), 'zero'
            self.setup_s = time.perf_counter() - t0
            self.error = p216.error_fn('bratu', self.p, p216.load_reference('bratu', reference_dir))
            self._torch = lambda dev: p216.torch_residual(self.rows.rows, dev)
            self.weight = lam
            self.space = 'tensor Chebyshev on the CGL set, N/P = 10 (P2-16)'
        elif bench == 'burgers':
            space = p41.Space(self.p)
            self.rows = p41.Rows(space)
            self.beta0, self.guess = np.zeros(P), 'zero'
            self.setup_s = time.perf_counter() - t0
            errors = p41.Errors(p41.load_reference(reference_dir))
            self.error = lambda b: errors.error(space, b)
            self._torch = self.rows.torch_residual
            self.weight = p41.LAM0
            self.space = 'sine x Chebyshev on the Gauss-Legendre set, 3p per direction (Package 4)'
        elif bench == 'bl':
            space, physics, _, self.rows = p51.bl_setup('sine', 'viscous', P, lam)
            self.guess = p32.CASES['viscous']['guess']
            self.beta0 = p32.initial_guess(space, physics, self.guess)
            self.setup_s = time.perf_counter() - t0
            e = p32.error_fn(space, p32.load_reference(reference_dir, 'viscous'))
            self.error = lambda b: e(b)['eps_ref']
            self._torch = self.rows.torch_residual
            self.weight = lam
            self.space = 'lifted sine x Chebyshev on the CGL set, N/P = 10 (Package 3 split)'
        else:
            raise ValueError(bench)

    def residual(self, beta):
        return self.rows.residual(beta)

    def jacobian(self, beta):
        return self.rows.assemble(beta)[0]

    def loss(self, beta):
        r = self.residual(beta)
        return float(r @ r)

    def torch_residual(self, device):
        return self._torch(device)


class TorchLoss:
    """``compute_loss_fn`` of ``solve_lil_n``: ||N(beta)||^2 from a torch residual,
    split into its interior, initial and boundary parts. Keeps the parameter it is
    called with (``param``), so the error can be read at the current iterate."""

    def __init__(self, residual):
        self.residual = residual
        self.param = None

    def __call__(self, beta):
        self.param = beta
        sq = {name: (r * r).sum() for name, r in self.residual(beta)}
        total = sum(sq.values())
        pde = sq.get('interior', total * 0)
        ic = sq.get('initial', total * 0)
        return total, pde.detach(), ic.detach(), (total - pde - ic).detach()


def run_dir_of(out_root, bench, P, method, device='cpu'):
    return Path(out_root) / ITEM / 'lil' / f'{bench}_P{P}' / f'{method}_{device}'


# ---------------------------------------------------------------- the usual protocol, per iterate

def protocol(pass_fn, logged_count, logged_beta):
    """Three timed passes after the logged one (the warm-up), each the median
    of SHORT_REPEATS passes if a pass is under SHORT_RUN_S. ``pass_fn()``
    returns ``(stamps, total_s, beta)``, the stamps one per iterate. Returns the
    per-iterate median stamps, the median total, and the checks."""
    def one():
        stamps, total, beta = pass_fn()
        counts.add(len(stamps))
        scale = float(np.max(np.abs(logged_beta))) or 1.0
        diffs.append(float(np.max(np.abs(np.asarray(beta) - logged_beta))) / scale)
        return stamps, total

    counts, diffs, timings = set(), [], []
    for _ in range(REPEATS):
        stamps, total = one()
        if total < SHORT_RUN_S:
            reps = [one() for _ in range(SHORT_REPEATS)]
            if len({len(r[0]) for r in reps}) == 1:
                stamps = [statistics.median(r[0][i] for r in reps) for i in range(len(reps[0][0]))]
            total = statistics.median(r[1] for r in reps)
            timings.append({'total_s': total, 'runs_s': [r[1] for r in reps]})
        else:
            timings.append({'total_s': total, 'runs_s': [total]})
        timings[-1]['stamps'] = stamps
    same = counts == {logged_count} and max(diffs) <= BETA_RTOL
    n = logged_count
    med = [statistics.median(t['stamps'][i] for t in timings) for i in range(n)] if same else None
    return {'stamps': med, 'total_s': statistics.median(t['total_s'] for t in timings),
            'timings': [{k: v for k, v in t.items() if k != 'stamps'} for t in timings],
            'spread': max(t['total_s'] for t in timings) / min(t['total_s'] for t in timings) - 1,
            'iterate_counts': sorted(counts), 'logged_count': logged_count,
            'max_beta_rel_diff': max(diffs), 'beta_rtol': BETA_RTOL, 'matches_logged_pass': same}


def lilq_pass(prob, keep=False):
    """One LiL-Q pass from beta0: ``(stamps, total_s, beta, betas, k_s)``; the
    stamps of iterates 0 .. k_s + 1, the betas only if ``keep``."""
    betas = [prob.beta0.copy()] if keep else None
    stamps = [0.0]
    t0 = time.perf_counter()

    def seen(b):
        stamps.append(time.perf_counter() - t0)
        if keep:
            betas.append(np.array(b, copy=True))
    beta, k_s, _, _ = iterate_to_rule(prob.rows, prob.beta0, N_S, on_iterate=seen)
    total = time.perf_counter() - t0
    return stamps, total, beta, betas, k_s


def trf_pass(prob, keep=False):
    """One ``trf`` pass: a stamp (and, if ``keep``, the coefficients) at every
    Jacobian evaluation, i.e. at the start and at every accepted iterate."""
    betas = [] if keep else None
    stamps, nfev, evals = [], [0], []
    t0 = time.perf_counter()

    def fun(b):
        nfev[0] += 1
        return prob.residual(b)

    def jac(b):
        stamps.append(time.perf_counter() - t0)
        if keep:
            betas.append(np.array(b, copy=True))
            evals.append(nfev[0])
        return prob.jacobian(b)
    s = least_squares(fun, prob.beta0, jac=jac, method='trf', xtol=TRF_TOL, ftol=TRF_TOL,
                      gtol=TRF_TOL, max_nfev=TRF_MAX_NFEV)
    total = time.perf_counter() - t0
    if keep:
        s.evaluations_at_iterates = evals            # residual evaluations up to each accepted iterate
    return stamps, total, s, betas


def _history(betas, stamps, prob, evaluations=None):
    rows = []
    for k, b in enumerate(betas):
        rows.append({'iteration': k, 'evaluations': k if evaluations is None else evaluations[k],
                     'time_s': stamps[k] if stamps is not None else math.nan,
                     'loss': prob.loss(b), 'error': float(prob.error(b))})
    return rows


def _save(d, record, rows, betas=None, beta=None, tag=''):
    d.mkdir(parents=True, exist_ok=True)
    write_history(d / 'history.csv', rows)
    if betas is not None:
        np.save(d / 'betas.npy', np.array(betas))
    if beta is not None:
        np.save(d / 'beta.npy', np.asarray(beta))
    record.update(commit=current_commit(), thread_env=capture_blas_thread_env(), scipy_version=scipy.__version__,
                  tag=tag, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    (d / 'run.json').write_text(json.dumps(record, indent=2, default=_plain))
    return record


def _plain(o):
    """JSON for numpy scalars (as numbers) and anything else (as text)."""
    return o.item() if isinstance(o, np.generic) else str(o)


def _base(prob, method, device='cpu'):
    return {'benchmark': prob.bench, 'P': prob.P, 'p': prob.p, 'method': LABELS[method], 'method_key': method,
            'device': device, 'trial_space': prob.space, 'boundary_weight': prob.weight,
            'initial_coefficients': prob.guess, 'setup_s': prob.setup_s}


def run_lilq(prob, out_root, timing=True, tag=''):
    stamps, total, beta, betas, k_s = lilq_pass(prob, keep=True)          # the logged pass (and the warm-up)
    record = _base(prob, 'lilq')
    record.update(rule_step=k_s, returned_iterate=None if k_s is None else k_s + 1,
                  stopping_reason='rule fired' if k_s is not None else 'K_max', iterates=len(betas) - 1,
                  logged_pass_s=total)
    if timing:
        t = protocol(lambda: lilq_pass(prob)[:3], len(stamps), betas[-1])
        record.update(time_s=t['total_s'], timing=t, timing_matches=t['matches_logged_pass'])
        stamps = t['stamps']
    else:
        record.update(time_s=total, timing='the logged pass only (untimed run)', timing_matches=None)
    record['timed'] = bool(timing)
    rows = _history(betas, stamps, prob)
    record.update(final_loss=rows[-1]['loss'], final_error=rows[-1]['error'])
    return _save(run_dir_of(out_root, prob.bench, prob.P, 'lilq'), record, rows, betas, beta, tag=tag)


def run_trf(prob, out_root, timing=True, tag=''):
    stamps, total, s, betas = trf_pass(prob, keep=True)
    record = _base(prob, 'trf')
    record.update(status=int(s.status), message=s.message, nfev=int(s.nfev), njev=int(s.njev),
                  stopping_reason=f'{int(s.status)}: {s.message}', iterates=len(betas) - 1,
                  logged_pass_s=total, tolerances=TRF_TOL, max_nfev=TRF_MAX_NFEV,
                  final_x_is_last_logged=bool(np.array_equal(s.x, betas[-1])))
    if timing:
        t = protocol(lambda: (lambda o: (o[0], o[1], o[2].x))(trf_pass(prob)), len(stamps), betas[-1])
        record.update(time_s=t['total_s'], timing=t, timing_matches=t['matches_logged_pass'])
        stamps = t['stamps']
    else:
        record.update(time_s=total, timing='the logged pass only (untimed run)', timing_matches=None)
    record['timed'] = bool(timing)
    rows = _history(betas, stamps, prob, evaluations=s.evaluations_at_iterates)
    record.update(final_loss=rows[-1]['loss'], final_error=rows[-1]['error'])
    return _save(run_dir_of(out_root, prob.bench, prob.P, 'trf'), record, rows, betas, s.x, tag=tag)


def solve_lbfgs(prob, device='cpu', budget_s=BUDGET_S, max_iterations=None, log_error=True):
    """LiL-N by L-BFGS from beta0 on ``device``. Returns ``(beta, metrics, summary,
    probe, betas_logged, setup_s)``."""
    import torch
    from lilq.metrics import RunProbe, _synchronize
    from lilq.solvers import solve_lil_n
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)          # for this run only (restored below)
    try:
        dev = torch.device(device)
        _synchronize()
        t0 = time.perf_counter()
        loss = TorchLoss(prob.torch_residual(dev))
        _synchronize()
        setup_s = time.perf_counter() - t0
        logged = []

        def error():
            # loss.param is the optimizer's own parameter tensor, at the accepted point after a step
            b = loss.param.detach().cpu().numpy().copy()
            logged.append(b)
            return float(prob.error(b))
        probe = RunProbe(error if log_error else None, every=EVERY, budget_s=budget_s,
                         max_iterations=max_iterations)
        beta, metrics, summary = solve_lil_n(loss, prob.beta0.copy(), dev, max_iterations=NO_CAP,
                                             max_line_searches=NO_CAP, R_tol=0.0, verbose=False,
                                             stall_rule='f1', probe=probe)
    finally:
        torch.set_default_dtype(old_dtype)
    return beta, metrics, summary, probe, logged, setup_s


def _stopping(summary):
    if not math.isfinite(summary['final_loss']):
        return 'failure'
    if summary.get('budget_reached'):
        return 'budget'
    if summary.get('optimizer_stall'):
        return 'optimizer_stall'
    return 'iteration_cap'


def run_lbfgs(prob, out_root, device='cpu', budget_s=BUDGET_S, tag=''):
    import torch
    beta, metrics, summary, probe, logged, setup_s = solve_lbfgs(prob, device, budget_s)
    record = _base(prob, 'lbfgs', device)
    record.update(setup_s=prob.setup_s + setup_s, device_setup_s=setup_s, budget_s=budget_s,
                  stopping_reason=_stopping(summary), iterations=summary['total_iterations'],
                  evaluations=summary['total_line_searches'], restarts=summary['lbfgs_restarts'],
                  time_s=summary['training_time'], training_time_s=summary['training_time'],
                  error_time_s=metrics.off_clock_s, error_every=EVERY, final_loss=summary['final_loss'],
                  stall={k: summary.get(k) for k in ('stall_iteration', 'stall_evaluations', 'stall_time_s')},
                  timing='one logged pass (Section 6), the error off the clock', torch_threads=torch.get_num_threads(),
                  gradient='autograd through the torch residual', timed=True)
    rows = probe.rows
    record['final_error'] = rows[-1]['error'] if rows else None
    record['last_logged_is_returned'] = bool(logged) and bool(np.array_equal(logged[-1], beta))
    d = run_dir_of(out_root, prob.bench, prob.P, 'lbfgs', device)
    d.mkdir(parents=True, exist_ok=True)
    metrics.to_dataframe().to_csv(d / 'metrics.csv', index=False, float_format='%.17g')
    np.save(d / 'betas_logged.npy', np.array(logged))
    return _save(d, record, rows, beta=beta, tag=tag)


def run_square(P, out_root, reference_dir, timing=True, tag=''):
    """SQ-CGL Newton (P2-3, settings unchanged) at (p - 2)^2 = P free coefficients: the
    error at every Newton step, the time of an iterate the median over the timed passes of
    the assembly and solve times summed over the steps before it."""
    import baselines.square_chebyshev as sc
    from experiments.p2_3_classical import REF_P, TEST_AXIS
    p = SQUARE_P[P]
    with np.load(Path(reference_dir) / f'bratu_ref_p{REF_P}.npz') as z:
        if not (np.allclose(z['x'], TEST_AXIS) and np.allclose(z['y'], TEST_AXIS)):
            raise ValueError('the Bratu reference is not on the 201 x 201 test grid of P2-3')
        ref = z['u'].copy()
    d = sc.newton('SQ-CGL', p, reference=(TEST_AXIS, ref))                # the logged pass (and the warm-up)
    errors = [h['err'] for h in d['history']]

    def stamps_of(history):
        steps = [h for h in history if h.get('t_solve_s') is not None]
        return [0.0] + [float(t) for t in np.cumsum([h['t_assemble_s'] + h['t_solve_s'] for h in steps])]

    record = {'benchmark': 'bratu', 'P': P, 'p': p, 'method': LABELS['square'], 'method_key': 'square',
              'device': 'cpu', 'free_coefficients': d['free_coeff'], 'P_trial': d['P_trial'], 'N': d['N'],
              'k_stop': d['k_stop'], 'stopping_reason': d['stop'], 'tolerance': sc.TOL, 'K_max': sc.K_MAX,
              'setup_s': d['t_setup_s'], 'kappa': d['kappa'], 'rank': d['rank'], 'iterates': len(errors) - 1}
    if timing:
        def one():
            r = sc.newton('SQ-CGL', p, diagnostics=False)
            s = stamps_of(r['history'])
            return s, s[-1], r['beta']
        t = protocol(one, len(errors), d['beta'])
        record.update(time_s=t['total_s'], timing=t, timing_matches=t['matches_logged_pass'])
        stamps = t['stamps']
    else:
        stamps = stamps_of(d['history'])
        record.update(time_s=stamps[-1], timing='the logged pass only (untimed run)', timing_matches=None)
    record['timed'] = bool(timing)
    rows = [{'iteration': k, 'evaluations': k, 'time_s': stamps[k] if stamps else math.nan,
             'loss': h.get('norm_R', math.nan) ** 2 if h.get('norm_R') is not None else math.nan, 'error': e}
            for k, (h, e) in enumerate(zip(d['history'], errors))]
    record.update(loss='||R||^2 of the square system at the iterate (the last iterate: not evaluated)',
                  final_error=errors[-1])
    return _save(run_dir_of(out_root, 'bratu', P, 'square'), record, rows, beta=d['beta'], tag=tag)


# ---------------------------------------------------------------- the warm-up

def warm_up(bench, device, reference_dir, methods=METHODS):
    """Every method on the device, untimed, before a job's first run, at the
    smallest size: LiL-Q and ``trf`` a full pass, L-BFGS WARMUP_ITERATIONS
    iterations with the error log on, the square collocation one Newton solve."""
    out = {}
    P = SIZES[bench][0]
    prob = Problem(bench, P, reference_dir)
    for method in methods:
        t = time.perf_counter()
        if method == 'lilq' and device == 'cpu':
            lilq_pass(prob)
        elif method == 'trf' and device == 'cpu':
            trf_pass(prob)
        elif method == 'lbfgs':
            solve_lbfgs(prob, device, budget_s=None, max_iterations=WARMUP_ITERATIONS)
        elif method == 'square' and bench == 'bratu' and device == 'cpu':
            import baselines.square_chebyshev as sc
            sc.newton('SQ-CGL', SQUARE_P[P], diagnostics=False)
        else:
            continue
        out[method] = time.perf_counter() - t
    return out


# ---------------------------------------------------------------- C3

def c3(out_root, package4):
    """C3: LiL-Q on Burgers equals Package 4 (item 1a's logged runs): the same rule
    step, and the error of every iterate to the returned one within C3_RTOL."""
    rows = []
    for P in SIZES['burgers']:
        p = isqrt(P)
        d = run_dir_of(out_root, 'burgers', P, 'lilq')
        if not (d / 'run.json').exists():
            continue
        rec = json.loads((d / 'run.json').read_text())
        new = [float(r['error']) for r in read_csv(d / 'history.csv')]
        p4 = Path(package4) / p41.ITEMS['1a'] / f'p{p}'
        if not (p4 / 'iterations.csv').exists():
            rows.append({'p': p, 'P': P, 'rule_step': rec['rule_step'], 'passed': False,
                         'reason': f'no Package 4 log at {p4}'})
            continue
        old_rows = read_csv(p4 / 'iterations.csv')
        old = {int(r['k']): float(r['eps_ref']) for r in old_rows if r.get('eps_ref') not in ('', None)}
        p4_step = next((int(r['k']) for r in old_rows if str(r.get('stall_rule_fires')).lower() == 'true'), None)
        its = [k for k in range(len(new)) if k in old]
        worst = max((abs(new[k] - old[k]) / abs(old[k]) for k in its), default=math.inf)
        rows.append({'p': p, 'P': P, 'rule_step': rec['rule_step'], 'package4_rule_step': p4_step,
                     'iterates_compared': len(its), 'max_rel_diff_error': worst, 'rtol': C3_RTOL,
                     'passed': bool(rec['rule_step'] == p4_step and worst <= C3_RTOL)})
    out = Path(out_root) / ITEM / 'lil' / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    result = {'check': 'C3: LiL-Q values of Burgers equal Package 4', 'rows': rows,
              'passed': bool(rows) and all(r['passed'] for r in rows)}
    (out / 'c3_burgers.json').write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------- CLI

def run_one(bench, P, method, out_root, reference_dir, device='cpu', budget_s=BUDGET_S, timing=True, tag=''):
    if method == 'square':
        return run_square(P, out_root, reference_dir, timing, tag)
    prob = Problem(bench, P, reference_dir)
    if method == 'lilq':
        return run_lilq(prob, out_root, timing, tag)
    if method == 'trf':
        return run_trf(prob, out_root, timing, tag)
    return run_lbfgs(prob, out_root, device, budget_s, tag)


def methods_for(bench, device, methods=METHODS):
    out = [m for m in methods if device == 'cpu' or m in DEVICE_METHODS]
    return [m for m in out if m != 'square' or bench == 'bratu']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--bench', required=True, choices=tuple(SIZES))
    r.add_argument('--out', required=True)
    r.add_argument('--reference-dir', required=True)
    r.add_argument('--P', type=int, nargs='*')
    r.add_argument('--methods', nargs='*', default=list(METHODS), choices=METHODS)
    r.add_argument('--device', default='cpu')
    r.add_argument('--budget', type=float, default=BUDGET_S)
    r.add_argument('--no-timing', action='store_true', help='the logged pass only (laptop rehearsal)')
    r.add_argument('--tag', default='', help='this process in its job (e.g. cpu-bratu): names its warm-up and '
                   'provenance records and is kept in each run.json')
    r.add_argument('--require-openmp', choices=OPENMP_KINDS,
                   help="stop before any run unless the process loaded this OpenMP library ('module' on Grace)")
    c = sub.add_parser('c3')
    c.add_argument('--out', required=True)
    c.add_argument('--package4', required=True)
    args = ap.parse_args(argv)

    if args.cmd == 'c3':
        res = c3(args.out, args.package4)
        print(json.dumps(res, indent=2))
        return
    if args.require_openmp:
        require_openmp(args.require_openmp)       # before any run or record
    root = Path(args.out) / ITEM / 'lil'
    name = f'{args.bench}_{args.device}' + (f'_{args.tag}' if args.tag else '')
    prov = job_dir(root / 'provenance' / name)      # a resubmitted job writes beside the first's
    save_provenance(prov)
    name = prov.name
    methods = methods_for(args.bench, args.device, args.methods)
    todo = [(P, m) for P in args.P or SIZES[args.bench] for m in methods]
    done = [(P, m) for P, m in todo if finished(
        run_dir_of(args.out, args.bench, P, m, args.device),
        **({'timed': True, 'budget_s': args.budget} if m == 'lbfgs' else {'timed': not args.no_timing}))]
    if done:
        print(f'{len(done)} of {len(todo)} runs already done here (run.json of this commit): not rerun', flush=True)
        todo = [t for t in todo if t not in done]
    if not todo:
        return
    w = warm_up(args.bench, args.device, args.reference_dir, [m for m in methods if any(m == t[1] for t in todo)])
    (root / 'warmup').mkdir(parents=True, exist_ok=True)
    (root / 'warmup' / f'{name}.json').write_text(json.dumps(
        {'seconds': w, 'P': SIZES[args.bench][0], 'lbfgs_iterations': WARMUP_ITERATIONS}, indent=2))
    for P, method in todo:
        rec = run_one(args.bench, P, method, args.out, args.reference_dir, args.device, args.budget,
                      not args.no_timing, args.tag)
        print(f"{args.bench} P={P} {rec['method']} {args.device}: {rec['stopping_reason']}, "
              f"{rec['time_s']:.3g} s, error {rec['final_error']:.3e}", flush=True)


if __name__ == '__main__':
    main()
