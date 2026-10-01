"""
Four-method Tables (Computational_Package_1_v2.md Section 3.4)
======================================================================

Reruns NiL-N, NiL-Q, and LiL-N -- the three L-BFGS-trained methods, as
opposed to LiL-Q, which already has its own full Section 3.1 log (see
``experiments/residual_band_figures.py`` and each problem's
``run_json_path`` wiring) -- across Bratu, Burgers, and both
Buckley-Leverett variants, at every P from the paper, seeds 0/1/2 for
NiL-N/NiL-Q (LiL-N: one deterministic run, per the spec's own exemption
-- same reasoning as ``lilq.multiseed``), on GPU as in the paper plus an
additional CPU run for the largest P per benchmark.

Every P/target-loss/iteration-and-line-search-budget value is imported
directly from that problem's own ``experiments/run_*.py`` (not
duplicated), same principle as ``residual_band_figures.py`` -- this
script cannot silently drift from the "paper settings"/"repository code
and budgets" (Section 3.3/3.4) already established there.

Output: ``four_method_tables.csv``, one row per (benchmark, P, method,
seed, device) -- see ``lilq.four_method_log`` for the exact schema.

This is a substantially larger compute job than
``residual_band_figures.py``: NiL-N/NiL-Q/LiL-N are real L-BFGS training
loops with iteration caps up to 10,000-20,000, run at up to 3 seeds each,
across ~15 (benchmark, P) combinations, on top of an extra CPU pass for
the largest size per benchmark. Use ``--quick`` for a fast correctness
smoke test (1 size, 1 seed, tiny caps); the full run (no flags) is a
real, possibly long-running job -- run it deliberately, not by default.

Stall control (the advisor's reply to wave 1, item 2.5): ``--controls``
reruns, after this job's own runs, every run of this job that ended on
``optimizer_stall`` -- same seed, starting point, budgets and caps, with
tolerances 0 and F1's restart-once rule (``stall_rule='f1'``) -- as a row
with ``variant = f1_stall_rule``, its model and history saved like any
other. ``--controls-from`` does the same for the stalled runs in earlier
tables (wave 1's Bratu), and ``--controls-only`` skips this job's own runs.
A control runs on its original's device, among this job's devices. The
merged ``four_method_tables.csv`` keeps the tables' own runs; the controls
go to ``four_method_controls.csv`` beside it.

Usage::

    python experiments/four_method_tables.py --quick
    python experiments/four_method_tables.py
    python experiments/four_method_tables.py --problems bratu --passes primary --controls-only \
        --controls-from results/wave1/B_instrumentation/four_method_jobs/bratu_gpu
"""

import sys
import os
import argparse
import dataclasses
import time
import traceback
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np
import torch

from lilq.utils import DEVICE, clear_gpu_memory
from lilq.four_method_log import (
    CONTROL_VARIANT, FourMethodLogger, compare_histories, row_key, stopping_fields, subsample_loss_history,
)
from lilq.provenance import save_provenance
from lilq.saved_models import save_history, save_network, save_solution
from lilq.source_lock import current_commit
from lilq.solvers import line_search_cap

from problems.bratu import (
    BratuConfig, BratuOptConfig,
    run_nil_n as bratu_nil_n, run_nil_q as bratu_nil_q, run_lil_n as bratu_lil_n,
)
from problems.burgers import (
    run_nil_n as burgers_nil_n, run_nil_q as burgers_nil_q, run_lil_n as burgers_lil_n,
)
from problems.buckley_leverett import (
    run_nil_n as bl_nil_n, run_nil_q as bl_nil_q, run_lil_n as bl_lil_n,
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'four_method_tables'
DEFAULT_SEEDS = (0, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-problem run configuration -- sizes/targets/budgets pulled straight
# from the existing experiment scripts (Section 3.3/3.4's "paper
# settings"/"repository code and budgets").
# ─────────────────────────────────────────────────────────────────────────────

def _bratu_runs(quick=False):
    from experiments.run_bratu import DEFAULT_N_VALUES, paper_setup
    N_values = DEFAULT_N_VALUES[:1] if quick else DEFAULT_N_VALUES
    return [(N * N, *paper_setup(N)) for N in N_values]


def _burgers_runs(quick=False):
    from experiments.run_burgers import DEFAULT_N_VALUES, paper_setup
    N_values = DEFAULT_N_VALUES[:1] if quick else DEFAULT_N_VALUES
    return [(N * N, *paper_setup(N)) for N in N_values]


def _bl_runs(gravity, quick=False):
    from experiments.run_bl import DEFAULT_N_VALUES, paper_setup
    N_values = DEFAULT_N_VALUES[:1] if quick else DEFAULT_N_VALUES
    return [(N * N, *paper_setup(N, gravity)) for N in N_values]


# label, run-list builder, NiL-N fn, NiL-Q fn, LiL-N fn
PROBLEMS = [
    ('bratu', _bratu_runs, bratu_nil_n, bratu_nil_q, bratu_lil_n),
    ('burgers', _burgers_runs, burgers_nil_n, burgers_nil_q, burgers_lil_n),
    ('bl', lambda quick: _bl_runs(False, quick), bl_nil_n, bl_nil_q, bl_lil_n),
    ('bl_gravity', lambda quick: _bl_runs(True, quick), bl_nil_n, bl_nil_q, bl_lil_n),
]


def _apply_quick_budgets(opt, max_iterations=15, max_line_searches=200,
                         max_quasi_iters_nn=2, max_inner_iters_nn=10):
    """Shrink an opt config's budgets for --quick smoke testing, without
    touching R_tol (keeping the real target means the run genuinely
    exercises "did not converge, hit a cap" -- the interesting path --
    rather than trivially converging on a target loosened along with
    everything else)."""
    return dataclasses.replace(
        opt,
        max_iterations=max_iterations, max_line_searches=max_line_searches,
        max_quasi_iters_nn=max_quasi_iters_nn, max_inner_iters_nn=max_inner_iters_nn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run + log
# ─────────────────────────────────────────────────────────────────────────────

def model_dir_name(row):
    """``<benchmark>_P<P>_<method>_s<seed>_<device>[_<variant>]``, the folder
    of a row's saved model."""
    seed = 'na' if row['seed'] in (None, '') else row['seed']
    variant = f"_{row['variant']}" if row.get('variant') else ''
    return f"{row['benchmark']}_P{row['P']}_{row['method']}_s{seed}_{row['device']}{variant}"


def _run_and_log(logger, benchmark, P, config, opt, method_name, runner,
                 seeds, devices, verbose=True, csv_path=None, skip_keys=frozenset(),
                 variant=None, original=None):
    """One row per (seed, device). ``seeds=None`` means "run once at
    config.seed, log seed as None" -- LiL-N's deterministic case.

    ``seed`` is the network-initialization seed (``init_seed``); the
    collocation set stays on ``config.seed`` for every row. A run that
    raises is logged as a ``failure`` row with its traceback and the sweep
    continues. With ``csv_path``, the whole CSV is rewritten after every
    row, so a killed job loses at most the run in progress. Rows whose key
    is in ``skip_keys`` (already completed, on resume) are not rerun.

    Every completed run's trained model is saved under
    ``<csv_path's folder>/models/<benchmark>_P<P>_<method>_s<seed>_<device>/``
    (``lilq.saved_models``: ``network.pt`` for NiL-N/NiL-Q, ``solution.pt``
    for LiL-N), after the run's clock stops and before its row is written.

    ``variant``/``original``: a stall control's row (:func:`run_controls`),
    compared with the row of the run it controls.
    """
    actual_seeds = list(seeds) if seeds else [None]

    for device in devices:
        for seed in actual_seeds:
            base = dict(benchmark=benchmark, P=P, method=method_name, seed=seed,
                        collocation_seed=config.seed, device=str(device), variant=variant)
            if row_key(base) in skip_keys:
                if verbose:
                    print(f"    {method_name:6s} seed={seed} device={device} -- done, skipping")
                continue
            run_config = dataclasses.replace(config, init_seed=seed) if seed is not None else config

            t0 = time.perf_counter()
            try:
                result = runner(run_config, opt, device=device, verbose=False)
            except Exception:
                elapsed = time.perf_counter() - t0
                logger.record(**base, wall_total_s=elapsed, stopping_reason='failure',
                              error=traceback.format_exc())
                if verbose:
                    print(f"    {method_name:6s} seed={seed} device={device} FAILED "
                          f"({elapsed:.1f}s) -- traceback logged")
                if csv_path is not None:
                    logger.to_csv(csv_path)
                clear_gpu_memory()
                continue
            elapsed = time.perf_counter() - t0
            metrics, summary = result[-2], result[-1]
            save_error = {}
            if csv_path is not None:
                model_dir = Path(csv_path).parent / 'models' / model_dir_name(base)
                try:       # a failed save is recorded in the row; it does not stop the sweep
                    if method_name == 'LiL-N':
                        save_solution(model_dir, {'u': (result[0], result[1])}, run_config, opt)
                    else:
                        save_network(model_dir, result[0], run_config, opt)
                    save_history(model_dir, metrics)
                except Exception:
                    save_error = {'error': 'model not saved: ' + traceback.format_exc()}

            stop = stopping_fields(method_name, summary, opt)
            history = subsample_loss_history(metrics.to_dict())
            control = {}
            if original is not None:
                same_start, start_diff, departs = compare_histories(original.get('loss_history_every_10'), history)
                control = dict(lbfgs_restarts=summary.get('lbfgs_restarts'), same_start=same_start,
                               start_loss_rel_diff=start_diff, departs_at_iteration=departs,
                               original_total_iterations=original.get('total_iterations'),
                               original_final_loss=original.get('final_loss'))

            logger.record(
                **base,
                total_iterations=summary['total_iterations'],
                total_line_searches=summary['total_line_searches'],
                training_time_s=summary['training_time'],
                wall_total_s=elapsed,
                final_loss=summary['final_loss'],
                converged=summary['converged'], stopping_reason=stop['stopping_reason'],
                iterations_cap=stop['iterations_cap'], line_searches_cap=stop['evaluations_cap'],
                stall_iteration=stop['stall_iteration'], stall_evaluations=stop['stall_evaluations'],
                stall_time_s=stop['stall_time_s'],
                loss_history_every_10=history,
                commit=current_commit(),
                **control,
                **save_error,
            )
            if csv_path is not None:
                logger.to_csv(csv_path)

            if verbose:
                status = "CONVERGED" if summary['converged'] else stop['stopping_reason']
                seed_label = f"seed={seed}" if seed is not None else "seed=n/a"
                if variant:
                    seed_label += f" [{variant}]"
                print(f"    {method_name:6s} {seed_label:10s} device={device!s:6s} "
                      f"iters={summary['total_iterations']:6d} "
                      f"loss={summary['final_loss']:.3e} [{status}] ({elapsed:.1f}s)", flush=True)

            clear_gpu_memory()


_WARMED_UP = set()


def warm_up(device, verbose=True):
    """Section 2: one untimed warm-up before the timed phase. Runs a tiny
    NiL-N and LiL-N solve on ``device`` once per process, so CUDA context
    creation, cuBLAS/autograd kernel setup, and first-call allocation
    don't land in the first timed run."""
    key = str(device)
    if key in _WARMED_UP:
        return
    from problems.bratu import BratuConfig, BratuOptConfig
    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = BratuOptConfig(max_iterations=5, max_line_searches=100, pretrain_epochs=5)
    t0 = time.perf_counter()
    bratu_nil_n(config, opt, device=device, verbose=False)
    bratu_lil_n(config, opt, device=device, verbose=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    clear_gpu_memory()
    _WARMED_UP.add(key)
    if verbose:
        print(f"  warm-up on {key}: {time.perf_counter() - t0:.1f}s (untimed)")


def run_problem(benchmark, runs_fn, nil_n_fn, nil_q_fn, lil_n_fn, logger,
                quick=False, seeds=DEFAULT_SEEDS, verbose=True,
                P_values=None, methods=('NiL-N', 'NiL-Q', 'LiL-N'),
                passes=('primary', 'cpu'), csv_path=None, skip_keys=frozenset()):
    """``passes``: ``primary`` runs every size on ``DEVICE`` (the GPU when
    present, as in the paper); ``cpu`` adds the spec's extra CPU run at the
    largest size. ``P_values``/``methods``/``passes`` select a subset, so a
    sweep can be split across independent jobs."""
    runs = runs_fn(quick)
    cpu = torch.device('cpu')
    runners = {'NiL-N': (nil_n_fn, seeds), 'NiL-Q': (nil_q_fn, seeds), 'LiL-N': (lil_n_fn, None)}
    for i, (P, config, opt) in enumerate(runs):
        if P_values and P not in P_values:
            continue
        if quick:
            opt = _apply_quick_budgets(opt)
        is_largest = (i == len(runs) - 1)
        devices = []
        if 'primary' in passes:
            devices.append(DEVICE)
        # The CPU pass at the largest size -- unless the primary pass already
        # ran there on the CPU in this call. On a CPU-only node (job 21 on a
        # timed-cpu node, Addendum v2.2 Section 4.1) DEVICE is the CPU, and a
        # 'cpu'-only call must still run.
        if 'cpu' in passes and is_largest and not ('primary' in passes and str(DEVICE) == 'cpu'):
            devices.append(cpu)
        if not devices:
            continue
        for device in devices:
            warm_up(device, verbose=verbose)

        if verbose:
            print(f"\n  P={P} (devices={[str(d) for d in devices]})")

        for method_name in methods:
            runner, method_seeds = runners[method_name]
            if method_seeds and quick:
                method_seeds = method_seeds[:1]
            _run_and_log(logger, benchmark, P, config, opt, method_name, runner,
                         seeds=method_seeds, devices=devices, verbose=verbose,
                         csv_path=csv_path, skip_keys=skip_keys)


def control_candidates(rows, devices, problems=None, P_values=None):
    """The runs to control: the tables' own runs (no ``variant``) that ended
    on ``optimizer_stall``, on one of ``devices``, optionally only of
    ``problems`` and sizes ``P_values``; each once."""
    devs = {str(d) for d in devices}
    out, seen = [], set()
    for r in rows:
        if (r.get('variant') or r.get('stopping_reason') != 'optimizer_stall' or str(r['device']) not in devs
                or (problems and r['benchmark'] not in problems) or (P_values and int(r['P']) not in P_values)):
            continue
        if row_key(r) not in seen:
            seen.add(row_key(r))
            out.append(r)
    return out


def run_controls(originals, logger, quick=False, verbose=True, csv_path=None, skip_keys=frozenset()):
    """One stall control per row of ``originals`` (the advisor's reply to
    wave 1, item 2.5): the same benchmark, size, method, seed, device,
    budgets and caps -- the configuration rebuilt from the problem's paper
    settings, and checked against the row's collocation seed -- with
    ``stall_rule='f1'``. Logged as ``variant = f1_stall_rule``; a control
    already in ``skip_keys`` is not rerun."""
    problems = {p[0]: p for p in PROBLEMS}
    for orig in originals:
        benchmark, P, method = orig['benchmark'], int(orig['P']), orig['method']
        _, runs_fn, nil_n_fn, nil_q_fn, lil_n_fn = problems[benchmark]
        # Every paper size (not the quick subset), so any original is found.
        match = [(c, o) for p, c, o in runs_fn(False) if p == P]
        problem = (f"no {benchmark} configuration with P={P}" if not match else
                   f"{benchmark} P={P}: collocation seed {orig['collocation_seed']} is not the "
                   f"configuration's {match[0][0].seed}"
                   if orig.get('collocation_seed') not in (None, '')
                   and int(orig['collocation_seed']) != match[0][0].seed else None)
        if problem:
            # Logged as a failed control; the other controls still run.
            base = {k: orig.get(k) for k in ('benchmark', 'P', 'method', 'seed', 'collocation_seed', 'device')}
            if row_key({**base, 'variant': CONTROL_VARIANT}) not in skip_keys:
                logger.record(**base, variant=CONTROL_VARIANT, stopping_reason='failure', error=problem)
                if csv_path is not None:
                    logger.to_csv(csv_path)
            print(f"  control not run: {problem}")
            continue
        config, opt = match[0]
        if quick:
            opt = _apply_quick_budgets(opt)
        opt = dataclasses.replace(opt, stall_rule='f1')
        runner = {'NiL-N': nil_n_fn, 'NiL-Q': nil_q_fn, 'LiL-N': lil_n_fn}[method]
        seed = orig.get('seed')
        device = torch.device(str(orig['device']))
        warm_up(device, verbose=verbose)
        if verbose:
            print(f"\n  control of {benchmark} P={P} {method} seed={seed} device={device} "
                  f"(original: {orig.get('total_iterations')} iterations, optimizer_stall)")
        _run_and_log(logger, benchmark, P, config, opt, method, runner,
                     seeds=None if seed in (None, '') else [int(seed)], devices=[device],
                     verbose=verbose, csv_path=csv_path, skip_keys=skip_keys,
                     variant=CONTROL_VARIANT, original=orig)


def merge_csvs(paths, out_path):
    """Combine per-job CSVs (split runs of this script) into one; a run
    key appearing in two inputs is an error -- the jobs overlapped. The
    tables' own runs go to ``out_path``; stall controls, when there are
    any, to ``four_method_controls.csv`` beside it."""
    merged, controls = FourMethodLogger(), FourMethodLogger()
    seen = set()
    for path in paths:
        if not Path(path).exists():
            print(f"  merge: no CSV at {path} (that job produced nothing) -- skipped")
            continue
        for row in FourMethodLogger.from_csv(path, drop_failures=False).rows:
            key = row_key(row)
            if key in seen:
                raise ValueError(f"run {key} appears in more than one input ({path})")
            seen.add(key)
            (controls if row.get('variant') else merged).record(**row)
    merged.to_csv(out_path)
    if len(controls):
        controls.to_csv(Path(out_path).with_name('four_method_controls.csv'))
    return merged, controls


def lilq_rows(b_root, out_path=None):
    """The four-method tables' LiL-Q rows (the advisor's follow-up of 1
    October 2026, item 1): time and iterations from the clean-timing runs
    (``<b_root>/clean_timing/clean_timing.csv``: warm-up, diagnostics off,
    K_max = 60), final loss and convergence from the logged paper pass of the
    same configuration (``<b_root>/<benchmark>_P<P>_cpu_paper/summary.json``;
    the clean run follows the same solver path), in this table's schema.
    LiL-Q runs on the CPU, once (deterministic). With ``out_path`` the rows
    are written there (``four_method_lilq.csv``)."""
    import csv as _csv
    import json as _json
    path = Path(b_root) / 'clean_timing' / 'clean_timing.csv'
    if not path.exists():
        return []
    rows = FourMethodLogger()
    with open(path, newline='') as f:
        clean = [r for r in _csv.DictReader(f)
                 if r['benchmark'] in ('bratu', 'burgers', 'bl', 'bl_gravity') and r['quantity'] == 'training_time']
    for r in clean:
        s_path = Path(b_root) / r['run'] / 'summary.json'
        s = _json.loads(s_path.read_text()) if s_path.exists() else {}
        converged = s.get('converged')
        rows.record(benchmark=r['benchmark'], P=int(r['config'].lstrip('P')), method='LiL-Q', seed=None,
                    device=r['device'], total_iterations=int(r['iterations']),
                    total_line_searches=int(r['iterations']), training_time_s=float(r['clean_time_s']),
                    final_loss=s.get('final_loss'), converged=converged,
                    stopping_reason='target' if converged else ('iteration_cap' if converged is False else None),
                    iterations_cap=int(r['K_max']) if r['K_max'] else None, commit=r['commit'])
    if out_path is not None:
        rows.to_csv(out_path)
    return rows.rows


def main():
    parser = argparse.ArgumentParser(description="Section 3.4 four-method tables")
    parser.add_argument('--quick', action='store_true',
                        help='Smoke-test budgets (1 size, 1 seed, tiny iteration/line-search '
                             'caps) instead of the full paper reruns -- this is a real, '
                             'possibly long-running job without this flag.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory. Jobs running at the same time must each '
                             'use their own (the CSV is rewritten after every run).')
    parser.add_argument('--problems', type=str, nargs='+', default=None,
                        choices=['bratu', 'burgers', 'bl', 'bl_gravity'],
                        help='Subset of problems to run (default: all four).')
    parser.add_argument('--P', type=int, nargs='+', default=None,
                        help='Subset of parameter counts P (default: every paper size).')
    parser.add_argument('--methods', type=str, nargs='+', default=['NiL-N', 'NiL-Q', 'LiL-N'],
                        choices=['NiL-N', 'NiL-Q', 'LiL-N'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS),
                        help='Network-init seeds for NiL-N/NiL-Q (default 0 1 2).')
    parser.add_argument('--passes', type=str, nargs='+', default=['primary', 'cpu'],
                        choices=['primary', 'cpu'],
                        help="'primary': every size on the GPU (CPU if none); "
                             "'cpu': extra CPU run at the largest size per benchmark.")
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore an existing CSV in --out-dir instead of resuming from it.')
    parser.add_argument('--controls', action='store_true',
                        help="After this job's runs, run a stall control (tolerances 0, F1's "
                             "restart-once rule) for each of them that ended on optimizer_stall.")
    parser.add_argument('--controls-from', type=str, nargs='+', default=None,
                        help='Also control the stalled runs in the four_method_tables.csv of these '
                             "job directories (an earlier wave's), on this job's devices.")
    parser.add_argument('--controls-only', action='store_true',
                        help="Skip this job's own runs; run only the controls.")
    parser.add_argument('--lilq-from-clean-timing', action='store_true',
                        help="Instead of running: write --out-dir's four_method_lilq.csv, LiL-Q's rows from "
                             "the clean-timing runs under --out-dir (package1's B_instrumentation).")
    parser.add_argument('--merge-from', type=str, nargs='+', default=None,
                        help='Instead of running: combine the four_method_tables.csv of these '
                             'job directories into --out-dir (failure rows kept).')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'four_method_tables.csv'

    if args.lilq_from_clean_timing:
        rows = lilq_rows(out_dir, out_dir / 'four_method_lilq.csv')
        print(f"Wrote {len(rows)} LiL-Q rows to {out_dir / 'four_method_lilq.csv'}")
        return
    if args.merge_from:
        merged, controls = merge_csvs([Path(d) / 'four_method_tables.csv' for d in args.merge_from], out_path)
        print(f"Merged {len(merged)} rows from {len(args.merge_from)} job directories into {out_path}"
              + (f"; {len(controls)} stall controls into four_method_controls.csv" if len(controls) else ''))
        return
    save_provenance(out_dir)

    problems = PROBLEMS
    if args.problems:
        problems = [p for p in PROBLEMS if p[0] in args.problems]

    if out_path.exists() and not args.fresh:
        logger = FourMethodLogger.from_csv(out_path)
        print(f"Resuming: {len(logger)} completed rows in {out_path} will be skipped "
              f"(failures are rerun).")
    else:
        logger = FourMethodLogger()
    skip_keys = frozenset(logger.completed_keys())
    t_start = time.perf_counter()

    if not args.controls_only:
        for benchmark, runs_fn, nil_n_fn, nil_q_fn, lil_n_fn in problems:
            print(f"\n{'=' * 60}\n{benchmark}\n{'=' * 60}")
            run_problem(benchmark, runs_fn, nil_n_fn, nil_q_fn, lil_n_fn, logger,
                        quick=args.quick, seeds=tuple(args.seeds), P_values=args.P,
                        methods=args.methods, passes=args.passes,
                        csv_path=out_path, skip_keys=skip_keys)

    if args.controls or args.controls_from:
        # This job's devices: the primary pass's (the GPU when present) and the CPU pass's.
        devices = ([DEVICE] if 'primary' in args.passes else []) + (
            [torch.device('cpu')] if 'cpu' in args.passes else [])
        names = [p[0] for p in problems]
        sources = [logger.rows] if args.controls else []
        sources += [FourMethodLogger.from_csv(Path(d) / 'four_method_tables.csv').rows
                    for d in (args.controls_from or [])]
        originals = control_candidates([r for rows in sources for r in rows], devices, names, args.P)
        print(f"\n{'=' * 60}\nstall controls: {len(originals)} runs ended on optimizer_stall\n{'=' * 60}")
        run_controls(originals, logger, quick=args.quick, csv_path=out_path, skip_keys=skip_keys)

    logger.to_csv(out_path)
    n_fail = sum(1 for r in logger.rows if r['stopping_reason'] == 'failure')
    print(f"\nWrote {len(logger)} rows to {out_path} ({n_fail} failures)")
    print(f"Total elapsed: {time.perf_counter() - t_start:.1f}s")


if __name__ == '__main__':
    main()
