"""
Package 5, item 3: cost at a common accuracy -- the network runs
================================================================

Section 6 of the Package 5 instructions. NiL-N (L-BFGS), NiL-Q (L-BFGS) and
NiL-N (Levenberg-Marquardt) on Bratu, Burgers and viscous Buckley-Leverett,
seeds 0, 1, 2, each run to a wall-clock budget of 600 s, with the error against
the reference logged along the run and its time off the clock.

**The same runs as the four-method tables, by construction.** As in P2-8
(``p2_8_lm_networks``), each benchmark's own ``run_nil_n`` / ``run_nil_q`` is
called with the four-method configuration (``paper_setup``) and the seed, with
its solver replaced by a wrapper here. Everything before the optimizer is the
Package 1 code: the network (tanh, two hidden layers, h^2 + 5h >= P), the
pretraining, the collocation set, the boundary and initial data and the loss
weights. The wrapper changes only the stopping:
- no target (``R_tol = 0``) and tolerances zero (``stall_rule='f1'``: tolerances
  0, one restart, then two steps in a row that do not lower the loss end the
  run as ``optimizer_stall``);
- no iteration, evaluation or outer-iteration cap: the budget is the only limit
  (``lilq.metrics.RunProbe``). NiL-Q keeps its inner cap per linearization (the
  number of L-BFGS steps before it relinearizes), which is part of the method;
- a non-finite loss ends a run as ``failure``, as in the four-method runs.

Levenberg-Marquardt is P2-8's (F2 damping, ``p2_8_lm_networks.lm``), with the
target 0, no iteration cap and the wall-clock cap set to the budget. It logs the
error at every iteration, off the clock, as in P2-8.

**The error** is the four-method runs' ``eps_ref``: the relative L2 error on the
benchmark's 201 x 201 test grid against the Section 6.2 reference (Bratu,
Burgers) or the stored finite-difference reference (BL; ``reference``), through
``p2_8_lm_networks.eps_ref_function``. The LiL runs use the same references. It is
logged at iteration 0, every 10 L-BFGS iterations (NiL-Q: its inner
iterations, counted over the run), every LM iteration, and at the last iterate.

**The clock** (``time_s``) starts at the optimizer's first iteration, as the
four-method tables' ``training_time``. Pretraining and the setup before the
optimizer are timed beside it (``pretrain_s``, ``setup_s``) and are not on it.
Each job first warms up every method on its device, untimed (``warm_up``).

**Outputs** under ``<out>/P5_3_cost/networks/``:
- ``<bench>_P<P>/<method>_s<seed>_<device>/``: ``run.json``; ``history.csv``
  (``iteration, evaluations, time_s, loss, error``: the history file of the
  instructions); the loss at every iteration (``metrics.csv``, L-BFGS, or LM's
  ``log.csv``); ``network.pt``;
- ``warmup/`` and ``provenance/``: one record per process, ``<bench>_<device>[_<tag>]``
  (``--tag``: two processes of a job, or two jobs, never write the same record);
- ``checks/solo_<bench>_P<P>_<method>_s<seed>_<device>/`` and ``checks/pairing.json``: the
  A100 set's paired-against-solo check (``solo``, ``pairing``).

Checks: C1 (``c1``) against the Package 1 (v2.0.0) histories; C2 and C4
(``c2_c4``): one run of a fixed length with and without the error log.

Usage::

    python experiments/p5_3_networks.py run --bench burgers --out <root> --reference-dir <dir> \\
        [--P 25 100] [--methods nil_n nil_q lm] [--seeds 0 1 2] [--device cpu] [--budget 600] \\
        [--part 0/2] [--tag gpu-burgers-0]
    python experiments/p5_3_networks.py solo --bench burgers --P 625 --method nil_n --out <root> \\
        --reference-dir <dir> [--device cuda] [--budget 60]
    python experiments/p5_3_networks.py pairing --out <root>
    python experiments/p5_3_networks.py checks --bench burgers --out <root> --reference-dir <dir> [--device cpu]
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
from unittest import mock

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import torch

from experiments import p2_8_lm_networks as p28
from lilq.metrics import RunProbe, _synchronize
from lilq.provenance import capture_blas_thread_env, save_provenance
from lilq.source_lock import current_commit

ITEM = 'P5_3_cost'
BUDGET_S = 600.0
SEEDS = (0, 1, 2)
METHODS = ('nil_n', 'nil_q', 'lm')
LABELS = {'nil_n': 'NiL-N (L-BFGS)', 'nil_q': 'NiL-Q (L-BFGS)', 'lm': 'NiL-N (LM)'}
EVERY = {'nil_n': 10, 'nil_q': 10, 'lm': 1}
NO_CAP = 10 ** 9                       # iterations, evaluations, outer iterations: the budget is the limit
HISTORY_COLUMNS = ('iteration', 'evaluations', 'time_s', 'loss', 'error')
WARMUP_ITERATIONS = 5                  # per method, smallest size, untimed, before a job's first run
C1_ITERATIONS = 100                    # the instructions: "over the first 100 iterations"
C1_RTOL = 1e-6                         # set before any comparison (DECISIONS.md, 2026-11-15)
C4_RTOL = 0.05                         # the instructions: "within 5%"
C2C4_ITERATIONS = {'nil_n': 300, 'lm': 30}
SOLO_BUDGET_S = 60.0                   # the solo check's run: long enough for a stable clock, cheap
PAIR_RTOL = 0.05                       # the plan (16 November 2026): a paired run within 5% of the same run alone


def sizes(bench):
    return p28.sizes(bench)


def reference(bench, config, reference_dir):
    """``(axes, field)`` on the 201 x 201 test grid: one reference per problem for every
    method of item 3. Bratu and Burgers: the four-method runs' (``p2_8_lm_networks.reference``).
    Buckley-Leverett: the stored finite-difference reference ``bl_fd_ref_viscous_nu0.1.npz``
    of Packages 2 and 3, which the LiL runs use, in place of the one the four-method runs
    compute (``reference_solution``; the same grid, 1.5e-7 apart, relative)."""
    if bench == 'bl':
        from experiments.p3_2_bl_certified import load_reference
        return load_reference(reference_dir, 'viscous')
    return p28.reference(bench, config, reference_dir)


def run_dir_of(out_root, bench, P, method, seed, device):
    return Path(out_root) / ITEM / 'networks' / f'{bench}_P{P}' / f'{method}_s{seed}_{device}'


# ---------------------------------------------------------------- one run

def _flat(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()


def _stopping(summary, budget_s):
    if not math.isfinite(summary['final_loss']):
        return 'failure'
    if summary.get('budget_reached'):
        return 'budget'
    if summary.get('optimizer_stall'):
        return 'optimizer_stall'
    return 'iteration_cap'          # only the short runs of the warm-up and of C2/C4 have a cap


def _timed_pretraining(mod, record):
    real = mod.pretrain_nn

    def timed(*args, **kwargs):
        _synchronize()
        t = time.perf_counter()
        out = real(*args, **kwargs)
        _synchronize()
        record['pretrain_s'] = time.perf_counter() - t
        return out
    return timed


def solve(bench, P, method, seed, reference_dir, device='cpu', budget_s=BUDGET_S, max_iterations=None,
          log_error=True, log_path=None):
    """One run. Returns ``(model, record, rows, metrics)``: ``rows`` the history
    (``HISTORY_COLUMNS``), ``metrics`` the L-BFGS ``MetricsTracker`` (None for LM,
    whose per-iteration log is written to ``log_path``)."""
    torch.set_default_dtype(torch.float64)
    mod, config, opt = p28.setup(bench, P)
    config = dataclasses.replace(config, init_seed=seed)
    dev = torch.device(device)
    cap = max_iterations or NO_CAP
    ref = reference(bench, config, reference_dir) if log_error else None
    got = {}

    def error_fn_of(model):
        if not log_error:
            return None
        eps = p28.eps_ref_function(p28.Net(model), ref, dev)
        return lambda: eps(_flat(model))

    def lbfgs(real):
        def wrapped(*args, **kwargs):
            got['solver_start'] = time.perf_counter()
            model = args[2] if method == 'nil_n' else args[3]      # (pde, bc, model, ...) / (pde, lin, bc, model, ...)
            probe = RunProbe(error_fn_of(model), every=EVERY[method], budget_s=budget_s,
                             max_iterations=max_iterations)  # a short run's cap: L-BFGS iterations, NiL-Q's inner ones
            kwargs.update(max_line_searches=NO_CAP, R_tol=0.0, stall_rule='f1', probe=probe, verbose=False)
            if method == 'nil_q':
                kwargs.update(max_quasi_iters=NO_CAP)    # the inner cap per linearization is kept
            else:
                kwargs.update(max_iterations=NO_CAP)
            got['inner_iters'] = kwargs.get('max_inner_iters')
            out = real(*args, **kwargs)
            got.update(probe=probe, solver_end=time.perf_counter())
            return out
        return wrapped

    def lm_solver(pde_fn, bc_fn, model, x_pde, y_pde, bc_data, lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0, **_):
        got['solver_start'] = time.perf_counter()
        net = p28.Net(model)
        X = torch.cat([x_pde.reshape(-1, 1), y_pde.reshape(-1, 1)], 1).detach()
        res = p28.Residual(net, p28.pde_function(bench, config), X, p28.boundary_blocks(bench, bc_data),
                           (lambda_pde, lambda_bc, lambda_ic))
        theta = _flat(model)
        eps = p28.eps_ref_function(net, ref, dev) if log_error else (lambda _theta: math.nan)
        theta, summary = p28.lm(res, theta, 0.0, eps, log_path, cap, budget_s if budget_s is not None else math.inf)
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(theta, model.parameters())
        summary['budget_reached'] = summary['stopping_reason'] == 'wall_time_cap'
        summary['optimizer_stall'] = summary['stopping_reason'] == 'optimizer_stall'
        got['solver_end'] = time.perf_counter()
        return model, None, summary

    record = {}
    patches = [mock.patch.object(mod, 'pretrain_nn', _timed_pretraining(mod, record))]
    if method == 'lm':
        patches.append(mock.patch.object(mod, 'solve_nil_n', lm_solver))
        runner = mod.run_nil_n
    else:
        name = 'solve_nil_n' if method == 'nil_n' else 'solve_nil_q'
        patches.append(mock.patch.object(mod, name, lbfgs(getattr(mod, name))))
        runner = mod.run_nil_n if method == 'nil_n' else mod.run_nil_q
    t0 = time.perf_counter()
    with patches[0], patches[1]:
        model, metrics, summary = runner(config, opt, device=dev, verbose=False)
    wall_total = time.perf_counter() - t0

    if method == 'lm':
        with open(log_path) as fh:
            rows = [{'iteration': int(r['iteration']), 'evaluations': int(r['n_func_evals']),
                     'time_s': float(r['wall_time']), 'loss': float(r['loss']), 'error': float(r['eps_ref'])}
                    for r in csv.DictReader(fh)]
        off_clock = summary['off_clock_s']
    else:
        rows = got['probe'].rows
        off_clock = metrics.off_clock_s
    training = float(summary['training_time'])
    record.update({
        'benchmark': bench, 'P': P, 'method': LABELS[method], 'method_key': method, 'seed': seed,
        'collocation_seed': config.seed, 'device': device, 'budget_s': budget_s,
        'iteration_cap': max_iterations, 'error_logged': bool(log_error), 'error_every': EVERY[method],
        'stopping_reason': _stopping(summary, budget_s),
        'iterations': int(summary['total_iterations']), 'evaluations': int(summary['total_line_searches']),
        'training_time_s': training,
        'error_time_s': off_clock,
        'solver_wall_s': got['solver_end'] - got['solver_start'],
        'setup_s': got['solver_start'] - t0 - record.get('pretrain_s', 0.0),
        'wall_total_s': wall_total,
        'final_loss': float(summary['final_loss']),
        'final_error': rows[-1]['error'] if rows else None,
        'n_params': int(sum(p.numel() for p in model.parameters())),
        'pretrain_loss': summary.get('pretrain_loss'),
        'restarts': summary.get('lbfgs_restarts', summary.get('lm_restarts')),
        'nil_q_outer_iterations': summary.get('n_quasi_iters'), 'nil_q_inner_cap': got.get('inner_iters'),
        'stall': {k: summary.get(k) for k in ('stall_iteration', 'stall_evaluations', 'stall_time_s')}
        if method != 'lm' else summary.get('stall'),
        'thread_env': capture_blas_thread_env(), 'torch_threads': torch.get_num_threads(),
    })
    if method == 'lm':
        record['lm'] ={'mu_final': summary['mu_final'], 'n_jacobian_evals': summary['n_jacobian_evals']}
    return model, record, rows, metrics


def write_history(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(r[k])) if isinstance(r[k], float) else r[k]) for k in HISTORY_COLUMNS})


def run(bench, P, method, seed, reference_dir, out_root, device='cpu', budget_s=BUDGET_S, tag='', run_dir=None):
    """One run into its folder (``run_dir_of``, or ``run_dir``: the solo check's). ``tag`` names the
    process in its job and is kept in run.json."""
    d = Path(run_dir) if run_dir is not None else run_dir_of(out_root, bench, P, method, seed, device)
    d.mkdir(parents=True, exist_ok=True)
    model, record, rows, metrics = solve(bench, P, method, seed, reference_dir, device, budget_s,
                                         log_path=d / 'log.csv')
    write_history(d / 'history.csv', rows)
    if metrics is not None:
        metrics.to_dataframe().to_csv(d / 'metrics.csv', index=False, float_format='%.17g')
    from lilq.saved_models import save_network
    mod, config, opt = p28.setup(bench, P)
    save_network(d, model, dataclasses.replace(config, init_seed=seed), opt)
    record.update(commit=current_commit(), tag=tag)
    (d / 'run.json').write_text(json.dumps(record, indent=2, default=str))
    return record


# ---------------------------------------------------------------- the warm-up

def warm_up(bench, device, reference_dir, methods=METHODS, scratch=None, tag=''):
    """Every method on the device, untimed, before a job's first run: WARMUP_ITERATIONS
    iterations at the smallest size, error logging on (the PyTorch and CUDA first-call
    costs, BLAS thread start-up; the lesson of item 2's re-time). Returns the seconds."""
    out = {}
    for method in methods:
        t = time.perf_counter()
        log = Path(scratch or '.') / f'.warmup_{bench}_{method}_{device}{_suffix(tag)}.csv'
        solve(bench, sizes(bench)[0], method, 0, reference_dir, device, budget_s=None,
              max_iterations=WARMUP_ITERATIONS, log_path=log)
        if log.exists():
            log.unlink()
        out[method] = time.perf_counter() - t
    return out


# ---------------------------------------------------------------- checks

def _package1_history(package1, bench, P, method, seed, device):
    name = {'nil_n': 'NiL-N', 'nil_q': 'NiL-Q'}[method]
    tag = 'gpu' if device != 'cpu' else 'cpu'
    return Path(package1) / 'B_instrumentation' / 'four_method_jobs' / f'{bench}_{tag}' / 'models' / \
        f'{bench}_P{P}_{name}_s{seed}_{"cuda" if device != "cpu" else "cpu"}' / 'history.csv'


def c1(run_dir, package1):
    """C1: the loss over iterations 0..C1_ITERATIONS against the Package 1 (v2.0.0)
    run of the same benchmark, size, method, seed and device. None if there is none."""
    rec = json.loads((Path(run_dir) / 'run.json').read_text())
    if rec['method_key'] not in ('nil_n', 'nil_q'):
        return None
    ref = _package1_history(package1, rec['benchmark'], rec['P'], rec['method_key'], rec['seed'], rec['device'])
    if not ref.exists():
        return None
    with open(ref) as fh:
        old = {int(r['iteration']): float(r['loss']) for r in csv.DictReader(fh)}
    with open(Path(run_dir) / 'metrics.csv') as fh:
        new = {int(r['iteration']): float(r['loss']) for r in csv.DictReader(fh)}
    its = [i for i in range(C1_ITERATIONS + 1) if i in old and i in new]
    worst = max(abs(new[i] - old[i]) / abs(old[i]) for i in its)
    return {'benchmark': rec['benchmark'], 'P': rec['P'], 'method': rec['method'], 'seed': rec['seed'],
            'device': rec['device'], 'iterations_compared': len(its), 'last_iteration': max(its),
            'max_rel_diff': worst, 'rtol': C1_RTOL, 'passed': bool(worst <= C1_RTOL), 'package1_history': str(ref)}


def c2_c4(bench, P, method, reference_dir, out_root, device='cpu', iterations=None):
    """C2 and C4 on one run of a fixed length (no budget), with and without the
    error log: the same iterates (C2, bit for bit) and the same clock (C4,
    within C4_RTOL). The run without the log goes first."""
    n = iterations or C2C4_ITERATIONS[method]
    d = Path(out_root) / ITEM / 'networks' / 'checks' / f'c2_c4_{bench}_P{P}_{method}_{device}'
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for logged in (False, True):
        model, rec, rows, metrics = solve(bench, P, method, 0, reference_dir, device, budget_s=None,
                                          max_iterations=n, log_error=logged,
                                          log_path=d / f'log_{"with" if logged else "without"}.csv')
        loss = metrics.data['loss'] if metrics is not None else \
            [r['loss'] for r in rows]
        out[logged] = (_flat(model).cpu(), loss, rec)
    (p0, l0, r0), (p1, l1, r1) = out[False], out[True]
    t0, t1 = r0['training_time_s'], r1['training_time_s']
    result = {'benchmark': bench, 'P': P, 'method': LABELS[method], 'device': device, 'iterations': n,
              'c2_same_parameters': bool(torch.equal(p0, p1)), 'c2_same_losses': l0 == l1,
              'time_without_s': t0, 'time_with_s': t1, 'error_time_s': r1['error_time_s'],
              'time_rel_diff': abs(t1 - t0) / t0, 'c4_rtol': C4_RTOL}
    result['c2_passed'] = result['c2_same_parameters'] and result['c2_same_losses']
    result['c4_passed'] = result['time_rel_diff'] <= C4_RTOL
    (d / 'result.json').write_text(json.dumps(result, indent=2, default=str))
    return result


# ---------------------------------------------------------------- paired against solo (the A100 set)

def _suffix(tag):
    return f'_{tag}' if tag else ''


def solo_dir_of(out_root, bench, P, method, seed, device):
    return Path(out_root) / ITEM / 'networks' / 'checks' / f'solo_{bench}_P{P}_{method}_s{seed}_{device}'


def pairing(out_root):
    """The A100 set runs two at a time on a two-A100 node (the user's decision, 16 November 2026).
    Each solo check (``solo``: the same run, the node's other GPU idle, for SOLO_BUDGET_S) against
    its paired run: the clock at the last iteration both logged, paired / solo - 1, within
    PAIR_RTOL. Writes ``checks/pairing.json``; returns its record."""
    rows = []
    for d in sorted((Path(out_root) / ITEM / 'networks' / 'checks').glob('solo_*')):
        solo = json.loads((d / 'run.json').read_text())
        paired = run_dir_of(out_root, solo['benchmark'], solo['P'], solo['method_key'], solo['seed'], solo['device'])
        row = {'benchmark': solo['benchmark'], 'P': solo['P'], 'method': solo['method'], 'seed': solo['seed'],
               'device': solo['device'], 'solo_tag': solo.get('tag')}
        if not (paired / 'history.csv').exists():
            rows.append(row | {'passed': None, 'reason': f'no paired run at {paired.name}'})
            continue
        a, b = (_times(x / 'history.csv') for x in (d, paired))
        common = sorted(set(a) & set(b) - {0})
        if not common:
            rows.append(row | {'passed': None, 'reason': 'no logged iteration in common'})
            continue
        k = common[-1]
        ratio = b[k] / a[k] - 1.0
        rows.append(row | {'iteration': k, 'time_solo_s': a[k], 'time_paired_s': b[k], 'paired_over_solo_minus_1': ratio,
                           'rtol': PAIR_RTOL, 'paired_tag': json.loads((paired / 'run.json').read_text()).get('tag'),
                           'passed': bool(abs(ratio) <= PAIR_RTOL)})
    out = Path(out_root) / ITEM / 'networks' / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    verdicts = [r['passed'] for r in rows]
    result = {'check': 'a paired A100 run against the same run alone', 'rows': rows,
              'passed': (all(v is True for v in verdicts) if verdicts else None)}
    (out / 'pairing.json').write_text(json.dumps(result, indent=2))
    return result


def _times(path):
    with open(path) as fh:
        return {int(r['iteration']): float(r['time_s']) for r in csv.DictReader(fh)}


def part_of(runs, part):
    """``part`` 'i/n': every n-th of ``runs`` from the i-th (round robin, so that two processes of a
    job get about the same number of runs of each size)."""
    i, n = (int(x) for x in part.split('/'))
    if not 0 <= i < n:
        raise ValueError(f'--part {part}: need 0 <= i < n')
    return runs[i::n]


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--bench', required=True, choices=p28.BENCHMARKS)
    r.add_argument('--out', required=True)
    r.add_argument('--reference-dir', required=True)
    r.add_argument('--P', type=int, nargs='*')
    r.add_argument('--methods', nargs='*', default=list(METHODS), choices=METHODS)
    r.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    r.add_argument('--device', default='cpu')
    r.add_argument('--budget', type=float, default=BUDGET_S)
    r.add_argument('--part', default='0/1', help="'i/n': this process's share of the runs (round robin)")
    r.add_argument('--tag', default='', help='this process in its job: names its warm-up and provenance '
                   'records and is kept in each run.json')
    so = sub.add_parser('solo', help='the paired-against-solo check: one run, the other GPU idle')
    so.add_argument('--bench', required=True, choices=p28.BENCHMARKS)
    so.add_argument('--out', required=True)
    so.add_argument('--reference-dir', required=True)
    so.add_argument('--P', type=int, required=True)
    so.add_argument('--method', required=True, choices=METHODS)
    so.add_argument('--seed', type=int, default=0)
    so.add_argument('--device', default='cuda')
    so.add_argument('--budget', type=float, default=SOLO_BUDGET_S)
    so.add_argument('--tag', default='')
    pa = sub.add_parser('pairing', help='compare the solo checks with their paired runs')
    pa.add_argument('--out', required=True)
    c = sub.add_parser('checks')
    c.add_argument('--bench', required=True, choices=p28.BENCHMARKS)
    c.add_argument('--out', required=True)
    c.add_argument('--reference-dir', required=True)
    c.add_argument('--P', type=int)
    c.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)

    root = Path(args.out) / ITEM / 'networks'
    if args.cmd == 'pairing':
        print(json.dumps(pairing(args.out), indent=2), flush=True)
        return
    tag = getattr(args, 'tag', '')
    name = f'{args.bench}_{args.device}{_suffix(tag)}'
    save_provenance(root / 'provenance' / name)
    if args.cmd == 'run':
        methods = args.methods
        todo = part_of([(P, m, s) for P in args.P or sizes(args.bench) for m in methods for s in args.seeds], args.part)
        w = warm_up(args.bench, args.device, args.reference_dir, sorted({m for _, m, _ in todo}), scratch=root, tag=tag)
        (root / 'warmup').mkdir(parents=True, exist_ok=True)
        (root / 'warmup' / f'{name}.json').write_text(json.dumps(
            {'seconds': w, 'iterations': WARMUP_ITERATIONS, 'P': sizes(args.bench)[0], 'part': args.part}, indent=2))
        for P, method, seed in todo:
            rec = run(args.bench, P, method, seed, args.reference_dir, args.out, args.device, args.budget, tag)
            print(f"{args.bench} P={P} {rec['method']} s{seed} {args.device}: {rec['iterations']} it, "
                  f"{rec['training_time_s']:.1f} s, {rec['stopping_reason']}, loss {rec['final_loss']:.3e}, "
                  f"error {rec['final_error']:.3e}", flush=True)
    elif args.cmd == 'solo':
        warm_up(args.bench, args.device, args.reference_dir, (args.method,), scratch=root, tag=tag)
        rec = run(args.bench, args.P, args.method, args.seed, args.reference_dir, args.out, args.device, args.budget,
                  tag, run_dir=solo_dir_of(args.out, args.bench, args.P, args.method, args.seed, args.device))
        print(f"solo {args.bench} P={args.P} {rec['method']} s{args.seed} {args.device}: {rec['iterations']} it in "
              f"{rec['training_time_s']:.1f} s", flush=True)
    else:
        warm_up(args.bench, args.device, args.reference_dir, ('nil_n', 'lm'), scratch=root, tag=tag)
        P = args.P or sizes(args.bench)[-1]
        for method in ('nil_n', 'lm'):
            res = c2_c4(args.bench, P, method, args.reference_dir, args.out, args.device)
            print(json.dumps(res, indent=2), flush=True)


if __name__ == '__main__':
    main()
