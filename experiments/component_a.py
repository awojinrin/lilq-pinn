"""
Component A: strong baselines on Kovasznay flow (Package 1 v2.0 Section 4)
==========================================================================

Stages, each resumable (a run whose ``run.json`` exists is not rerun) and
each run appended to ``tuning_log.md`` in the order tried::

    search          save the 24 + 24 configurations (generator seed 12345)
    screen          every configuration of a family, seed 0, 10-minute budget
    select          rank the screening runs by final training loss (F1: unweighted); keep the top 3
    full            the top 3 x seeds 0-4, 60-minute budget; then the
                    representative (lowest median training loss) and the best
                    test errors among the 15 runs
    cpu             the representative, CPU, seeds 0-4, full budget
    float32         F1's representative with Adam in float32, L-BFGS in float64, seed 0
    a1              check A1: the plain PINN must reach eps_u <= 1e-3 in the full budget
    f2-jacobian     check F2's Jacobian against central differences on this node
    a2              check A2: two F2 runs with the same seed agree to 12 digits over 20 steps

Layout under ``--root`` (the package's ``A_calibration/``)::

    search/F1_configs.json, F2_configs.json
    screening/<config>_s0/{log.csv, run.json}, screening/<family>_selection.json
    full/<config>_s<seed>/..., full/<family>_representative.json
    full_cpu/..., float32/..., checks/{a1,a2,f2_jacobian}.json
    tuning_log.md

Budgets are wall-clock from the first optimizer step, never stopped on test
error; selection uses training loss only (Section 4.3).

Usage::

    python experiments/component_a.py search --root <package>/A_calibration
    python experiments/component_a.py screen --family F1 --device cuda --root ...
"""

import argparse
import csv
import datetime
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from lilq.blas_threads import pin_torch  # first: sets the BLAS threads before numpy/scipy load
import torch

pin_torch()   # PyTorch's threads = the BLAS allocation

import baselines.f1_pinn as f1
import baselines.lm_kovasznay as lm
from baselines.search import save_search
from lilq.provenance import save_provenance
from lilq.source_lock import current_commit

FAMILIES = ('F1', 'F2')
SCREEN_BUDGET_S = 600.0
FULL_BUDGET_S = 3600.0
FULL_SEEDS = (0, 1, 2, 3, 4)
TOP_K = 3
A1_TARGET = 1e-3

TUNING_LOG_HEADER = """# Component A tuning log

Every configuration tried, in order, and every manual intervention with its
reason (Package 1 v2.0 Section 7, item 4). Implementation choices where the
package is open are recorded in the repository's DECISIONS.md (2026-09-24,
"Component A, F1 ..." and "Component A, F2 ..."): the gradient-norm balancing
form, the learning-rate schedule's reference point, no pressure pin in F1,
the L-BFGS stopping criterion, test errors off the clock for both families,
the check-A1 configuration, and the hard/soft draw. Changed by Addendum v2.2
Section 2.5 (DECISIONS.md, 2026-09-29): F1's L-BFGS runs with its
tolerances at 0; a call that does not lower the loss is followed by one call
with a fresh optimizer, and the run ends only if that call does not lower it
either (our reading of v2.0 Section 4.3's "the family's own criterion");
F1 configurations are ranked and selected on the unweighted final loss.

## Runs

"""


def _log(root, line):
    path = Path(root) / 'tuning_log.md'
    try:                                   # atomic: concurrent jobs never truncate each other
        with open(path, 'x', encoding='utf-8') as f:
            f.write(TUNING_LOG_HEADER)
    except FileExistsError:
        pass
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line.rstrip() + '\n')


def load_configs(root, family):
    return json.loads((Path(root) / 'search' / f'{family}_configs.json').read_text())['configs']


def run_one(root, stage, family, config, seed, budget_s, out_dir, device, precision='float64', test_every=1):
    """Train one configuration unless its run.json exists; returns run.json's content."""
    out_dir = Path(out_dir)
    if (out_dir / 'run.json').exists():
        return json.loads((out_dir / 'run.json').read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    if family == 'F1':
        try:
            run = f1.f1_train(config, seed, budget_s, out_dir, device=device, precision=precision,
                              test_every=test_every)
        except Exception:                  # e.g. out of memory building the network
            run = dict(config=config, seed=seed, device=device, precision=precision, budget_s=budget_s,
                       end_reason='failure', final_loss=None, traceback=traceback.format_exc())
            (out_dir / 'run.json').write_text(json.dumps(run, indent=2, default=str))
    else:
        args = argparse.Namespace(width=config['width'], depth=config['depth'], m=config['m'],
                                  sigma_ff=config['sigma_ff'], n_int=config['n_int'], w_int=1.0, w_pin=1.0,
                                  mu0=1e-3, diag_floor=1e-12, budget_s=budget_s, max_steps=0,
                                  max_params=20000, chunk=256, test_every=test_every, seed=seed,
                                  device=device, out=str(out_dir))
        try:
            run = lm.lm_train(args)
        except Exception:
            run = dict(vars(args), end_reason='failure', final_loss=None, traceback=traceback.format_exc())
        run.update(config=config, seed=seed)
        (out_dir / 'run.json').write_text(json.dumps(run, indent=2, default=str))
    if 'commit' not in run:           # the code's commit in every run.json (Addendum v2.2 2.8.4)
        run['commit'] = current_commit()
        (out_dir / 'run.json').write_text(json.dumps(run, indent=2, default=str))
    save_provenance(out_dir)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    fmt = lambda v: f'{v:.3e}' if isinstance(v, (int, float)) else str(v)  # noqa: E731
    _log(root, f"- {stamp} | {stage} | {family} {config['id']} seed {seed} | {device}, {precision}, "
               f"budget {budget_s:.0f} s | end: {run.get('end_reason')} | final loss {fmt(run.get('final_loss'))} (unweighted {fmt(run.get('final_loss_unweighted'))})"
               f" | eps_u {fmt(run.get('eps_u'))}")
    return run


def _final_loss(run):
    """The selection loss (Addendum v2.2 Section 2.5): F1's unweighted final
    loss (``final_loss_unweighted``: its balanced or lambda_bc-weighted loss
    is not comparable across configurations); F2's final loss, whose row
    weights are the same for every configuration."""
    v = run.get('final_loss_unweighted', run.get('final_loss'))
    return v if isinstance(v, (int, float)) else float('inf')


def screen(root, family, device, budget_s=SCREEN_BUDGET_S, configs=None):
    configs = configs or load_configs(root, family)
    return {c['id']: run_one(root, 'screen', family, c, 0, budget_s,
                             Path(root) / 'screening' / f"{c['id']}_s0", device)
            for c in configs}


def select(root, family):
    runs = []
    for c in load_configs(root, family):
        path = Path(root) / 'screening' / f"{c['id']}_s0" / 'run.json'
        if not path.exists():
            raise FileNotFoundError(f"screening run missing: {path}")
        runs.append((c['id'], _final_loss(json.loads(path.read_text()))))
    ranking = sorted(runs, key=lambda r: r[1])
    selection = {'selection_loss': 'final_loss_unweighted' if family == 'F1' else 'final_loss',
                 'ranking': [{'id': i, 'loss': l} for i, l in ranking],
                 'top': [i for i, _ in ranking[:TOP_K]]}
    (Path(root) / 'screening' / f'{family}_selection.json').write_text(json.dumps(selection, indent=2))
    return selection


def full(root, family, device, budget_s=FULL_BUDGET_S, seeds=FULL_SEEDS):
    selection = json.loads((Path(root) / 'screening' / f'{family}_selection.json').read_text())
    by_id = {c['id']: c for c in load_configs(root, family)}
    runs = {(cid, s): run_one(root, 'full', family, by_id[cid], s, budget_s,
                              Path(root) / 'full' / f'{cid}_s{s}', device)
            for cid in selection['top'] for s in seeds}
    medians = {cid: statistics.median(_final_loss(runs[(cid, s)]) for s in seeds) for cid in selection['top']}
    rep = min(medians, key=medians.get)
    best = {k: min((r.get(k) for r in runs.values() if isinstance(r.get(k), (int, float))), default=None)
            for k in ('eps_u', 'eps_v', 'eps_p', 'eps_p_meanfree')}
    summary = {'representative': rep, 'representative_config': by_id[rep], 'median_final_loss': medians,
               'best_test_errors_over_all_full_runs': best, 'seeds': list(seeds)}
    (Path(root) / 'full' / f'{family}_representative.json').write_text(json.dumps(summary, indent=2))
    return summary


def _representative(root, family):
    return json.loads((Path(root) / 'full' / f'{family}_representative.json').read_text())['representative_config']


def cpu_reruns(root, family, budget_s=FULL_BUDGET_S, seeds=FULL_SEEDS):
    c = _representative(root, family)
    return [run_one(root, 'cpu', family, c, s, budget_s, Path(root) / 'full_cpu' / f"{c['id']}_s{s}", 'cpu')
            for s in seeds]


def float32_run(root, device, budget_s=FULL_BUDGET_S):
    c = _representative(root, 'F1')
    return run_one(root, 'float32', 'F1', c, 0, budget_s, Path(root) / 'float32' / f"{c['id']}_s0",
                   device, precision='adam32')


def check_a1(root, device, budget_s=FULL_BUDGET_S):
    run = run_one(root, 'check A1', 'F1', f1.A1_CONFIG, 0, budget_s, Path(root) / 'checks' / 'A1_s0', device)
    result = {'eps_u': run.get('eps_u'), 'target': A1_TARGET, 'budget_s': budget_s,
              'passed': isinstance(run.get('eps_u'), float) and run['eps_u'] <= A1_TARGET}
    (Path(root) / 'checks' / 'a1.json').write_text(json.dumps(result, indent=2))
    return result


def check_f2_jacobian(root, device, n_params=16, h=1e-6):
    """The advisor's finite-difference check of F2's Jacobian, on this node
    and device (Addendum v2.1 Section 8), at the smoke-run size."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dev = torch.device(device)
        model = lm.FourierMLP(24, 2, 16, 1.0, 0).to(dev)
        theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        gen = torch.Generator().manual_seed(0)
        u01 = torch.rand(50, 2, generator=gen)
        xy = torch.stack([lm.X0 + (lm.X1 - lm.X0) * u01[:, 0], lm.Y0 + (lm.Y1 - lm.Y0) * u01[:, 1]], 1).to(dev)
        res = lm.Residual(model, xy, torch.tensor([lm.X0, lm.Y0], device=dev))
        J = res.jacobian(theta, 25)
        idx = torch.randperm(theta.numel(), generator=gen)[:n_params]
        worst = 0.0
        for i in idx:
            e = torch.zeros_like(theta); e[i] = h
            fd = (res.vector(theta + e, 25) - res.vector(theta - e, 25)) / (2 * h)
            worst = max(worst, ((fd - J[:, i]).abs().max() / (J[:, i].abs().max() + 1e-30)).item())
    finally:
        torch.set_default_dtype(old)
    result = {'device': str(device), 'max_relative_fd_error': worst, 'columns_checked': n_params,
              'passed': worst < 1e-6,
              'gpu': torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None}
    out = Path(root) / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'f2_jacobian.json').write_text(json.dumps(result, indent=2))
    return result


def check_a2(root, device, steps=20):
    """Check A2 (Section 8): two F2 runs with the same seed agree to 12
    significant digits over ``steps`` steps, on this node and device. A
    moderate size that cannot be fitted exactly (an exact fit ends early)."""
    histories = []
    for name in ('a', 'b'):
        out = Path(root) / 'checks' / 'a2' / name
        args = argparse.Namespace(width=24, depth=2, m=16, sigma_ff=1.0, n_int=600, w_int=1.0, w_pin=1.0,
                                  mu0=1e-3, diag_floor=1e-12, budget_s=3600.0, max_steps=steps,
                                  max_params=20000, chunk=200, test_every=10 ** 6, seed=0, device=device,
                                  out=str(out))
        lm.lm_train(args)
        with open(out / 'log.csv', newline='') as f:
            histories.append([float(r['loss_total']) for r in csv.DictReader(f)])
    a, b = histories
    worst = max((abs(x - y) / max(abs(x), 1e-300) for x, y in zip(a, b)), default=float('inf'))
    result = {'device': str(device), 'steps': [len(a), len(b)], 'max_relative_difference': worst,
              'bit_identical': a == b, 'passed': len(a) == len(b) == steps and worst <= 1e-12}
    (Path(root) / 'checks' / 'a2.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description="Component A: Kovasznay baselines")
    ap.add_argument('stage', choices=['search', 'screen', 'select', 'full', 'cpu', 'float32', 'a1', 'a2', 'f2-jacobian'])
    ap.add_argument('--root', required=True, help="The package's A_calibration/ directory.")
    ap.add_argument('--family', choices=FAMILIES)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--budget-s', type=float, default=None,
                    help='Override the stage budget (smoke tests only; the package fixes 600 / 3600 s).')
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    needs_family = args.stage in ('screen', 'select', 'full', 'cpu')
    if needs_family and not args.family:
        ap.error(f"{args.stage} needs --family")
    if args.budget_s is not None:
        _log(root, f"- manual intervention: stage {args.stage} run with budget {args.budget_s:.0f} s "
                   "instead of the package's (smoke test)")
    budget = lambda default: args.budget_s if args.budget_s is not None else default  # noqa: E731

    if args.stage == 'search':
        print(f"Saved {save_search(root / 'search')}")
    elif args.stage == 'screen':
        screen(root, args.family, args.device, budget(SCREEN_BUDGET_S))
    elif args.stage == 'select':
        print(json.dumps(select(root, args.family)['top']))
    elif args.stage == 'full':
        print(json.dumps(full(root, args.family, args.device, budget(FULL_BUDGET_S)), indent=2))
    elif args.stage == 'cpu':
        cpu_reruns(root, args.family, budget(FULL_BUDGET_S))
    elif args.stage == 'float32':
        float32_run(root, args.device, budget(FULL_BUDGET_S))
    elif args.stage == 'a1':
        result = check_a1(root, args.device, budget(FULL_BUDGET_S))
        print(json.dumps(result))
        sys.exit(0 if result['passed'] else 1)       # a gate: Component A waits for it (Addendum v2.2 2.6)
    elif args.stage == 'a2':
        result = check_a2(root, args.device)
        print(json.dumps(result))
        sys.exit(0 if result['passed'] else 1)
    elif args.stage == 'f2-jacobian':
        result = check_f2_jacobian(root, args.device)
        print(json.dumps(result))
        sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
