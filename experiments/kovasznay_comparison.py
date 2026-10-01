"""
Kovasznay: LiL-Q against the Component A baselines (Package 1 v2.0 Section 4.6)
===============================================================================

Reads LiL-Q's Section 3.3 Kovasznay runs (``B_instrumentation/kovasznay_P<P>_
<device>_paper/summary.json``: test-grid errors) with their time from the
clean-timing runs (``B_instrumentation/clean_timing/clean_timing.csv``:
warm-up, then diagnostics off -- the rule for every quoted LiL-Q time, the
advisor's reply to wave 2, Section 3; the logged ``solve_time_total`` is
kept beside it, and used only where there is no clean time), and each baseline
family's representative runs (``A_calibration/full/`` on the GPU,
``A_calibration/full_cpu/`` on the CPU), and writes:

``results/kovasznay_comparison.csv``
    One row per (family, device): the best eps_u, eps_v and mean-free eps_p
    reached within the budget (per seed the minimum over its log within the budget; median,
    min and max over seeds), peak GPU memory, and for each of the five
    LiL-Q eps_u levels on the same device the time-to-accuracy (median over
    the seeds that reached it, and how many did) with the iteration count at
    that time. LiL-Q has one row per (P, device).
``results/time_to_accuracy.csv``
    The same, per seed.
``figures/eps_u_vs_time.pdf`` and ``.csv``
    eps_u against wall-clock time, log-log, one curve per seed per family,
    the five LiL-Q results as points, one panel per device.

Usage::

    python experiments/kovasznay_comparison.py --package <package1_results>
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

KOVASZNAY_P = (75, 300, 675, 1200, 1875)
DEVICES = {'gpu': ('cuda', 'full'), 'cpu': ('cpu', 'full_cpu')}   # label: (LiL-Q device, baseline dir)
FAMILIES = ('F1', 'F2')
EPS = ('eps_u', 'eps_v', 'eps_p_meanfree')


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_clean_times(b_root):
    """``{run: solve_time_total}`` from the clean-timing table, if any."""
    path = Path(b_root) / 'clean_timing' / 'clean_timing.csv'
    if not path.exists():
        return {}
    with open(path, newline='') as f:
        return {r['run']: float(r['clean_time_s']) for r in csv.DictReader(f)
                if r['quantity'] == 'solve_time_total' and r['clean_time_s'] not in ('', None)}


def load_lilq(b_root, device):
    out = {}
    clean = load_clean_times(b_root)
    for P in KOVASZNAY_P:
        name = f'kovasznay_P{P}_{device}_paper'
        path = Path(b_root) / name / 'summary.json'
        if path.exists():
            s = json.loads(path.read_text())
            out[P] = {'eps_u': s.get('test_eps_u'), 'eps_v': s.get('test_eps_v'),
                      'eps_p_meanfree': s.get('test_eps_p_meanfree'), 'eps_p': s.get('test_eps_p'),
                      'time_s': clean.get(name, s.get('solve_time_total')),
                      'time_s_logged': s.get('solve_time_total'),
                      'time_source': 'clean' if name in clean else 'logged',
                      'iterations': s.get('iterations')}
    return out


def load_seed_runs(a_root, family, stage_dir):
    rep_path = Path(a_root) / 'full' / f'{family}_representative.json'
    if not rep_path.exists():
        return None, []
    rep = json.loads(rep_path.read_text())['representative']
    runs = []
    for run_dir in sorted((Path(a_root) / stage_dir).glob(f'{rep}_s*')):
        if not (run_dir / 'run.json').exists():
            continue
        run = json.loads((run_dir / 'run.json').read_text())
        budget = _num(run.get('budget_s')) or float('inf')
        with open(run_dir / 'log.csv', newline='') as f:
            # Within the budget only: an LM step (or the last F1 evaluation)
            # can finish just past it.
            rows = [r for r in csv.DictReader(f) if float(r['t_cum_s']) <= budget]
        curve = [(float(r['t_cum_s']), int(float(r['iter'])), _num(r['eps_u']))
                 for r in rows if _num(r['eps_u']) is not None]
        final = [_num(run.get(k)) for k in EPS] if (_num(run.get('wall_s')) or 0.0) <= budget else [None] * 3
        best = {k: min([v for v in [_num(r.get(k)) for r in rows] + [f] if v is not None], default=None)
                for k, f in zip(EPS, final)}
        runs.append({'seed': run.get('seed'), 'curve': curve, 'best': best,
                     'peak_gpu_bytes': run.get('peak_gpu_bytes')})
    return rep, runs


def time_to_accuracy(curve, level):
    """(time, iteration) of the first logged eps_u <= level, or (None, None)."""
    for t, it, e in curve:
        if e <= level:
            return t, it
    return None, None


def _stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    return statistics.median(vals), min(vals), max(vals)


def build(package, out_root=None):
    package = Path(package)
    b_root, a_root = package / 'B_instrumentation', package / 'A_calibration'
    out_root = Path(out_root) if out_root else a_root
    (out_root / 'results').mkdir(parents=True, exist_ok=True)
    (out_root / 'figures').mkdir(parents=True, exist_ok=True)

    summary_rows, tta_rows, curves = [], [], []
    for label, (lil_device, stage_dir) in DEVICES.items():
        lilq = load_lilq(b_root, lil_device)
        for P, r in lilq.items():
            summary_rows.append({'family': 'LiL-Q', 'device': label, 'config': f'P={P}', 'n_seeds': 1,
                                 **{f'{k}_median': r[k] for k in EPS}, 'time_s': r['time_s'],
                                 'time_s_logged': r['time_s_logged'], 'time_source': r['time_source']})
            curves.append({'family': 'LiL-Q', 'device': label, 'seed': '', 'P': P,
                           't': r['time_s'], 'eps_u': r['eps_u']})
        for family in FAMILIES:
            rep, runs = load_seed_runs(a_root, family, stage_dir)
            if not runs:
                continue
            row = {'family': family, 'device': label, 'config': rep, 'n_seeds': len(runs)}
            for k in EPS:
                row[f'{k}_median'], row[f'{k}_min'], row[f'{k}_max'] = _stats([r['best'][k] for r in runs])
            peaks = [r['peak_gpu_bytes'] for r in runs if r['peak_gpu_bytes']]
            row['peak_gpu_bytes_max'] = max(peaks) if peaks else None
            for P, lil in lilq.items():
                hits = [time_to_accuracy(r['curve'], lil['eps_u']) for r in runs]
                for r, (t, it) in zip(runs, hits):
                    tta_rows.append({'family': family, 'device': label, 'config': rep, 'seed': r['seed'],
                                     'lilq_P': P, 'lilq_eps_u': lil['eps_u'],
                                     'time_s': t if t is not None else 'not reached',
                                     'iteration': it if it is not None else ''})
                reached = [(t, it) for t, it in hits if t is not None]
                row[f'tta_P{P}_level'] = lil['eps_u']
                row[f'tta_P{P}_reached'] = f'{len(reached)}/{len(runs)}'
                row[f'tta_P{P}_time_median'] = statistics.median(t for t, _ in reached) if reached else 'not reached'
                row[f'tta_P{P}_iter_median'] = statistics.median(it for _, it in reached) if reached else ''
            summary_rows.append(row)
            for r in runs:
                curves += [{'family': family, 'device': label, 'seed': r['seed'], 'P': '', 't': t, 'eps_u': e}
                           for t, _, e in r['curve']]

    def write(path, rows):
        cols = []
        for row in rows:
            cols += [c for c in row if c not in cols]
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

    write(out_root / 'results' / 'kovasznay_comparison.csv', summary_rows)
    write(out_root / 'results' / 'time_to_accuracy.csv', tta_rows)
    write(out_root / 'figures' / 'eps_u_vs_time.csv', curves)
    _figure(curves, out_root / 'figures' / 'eps_u_vs_time.pdf')
    return out_root


def _figure(curves, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'F1': 'tab:blue', 'F2': 'tab:orange'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, label in zip(axes, DEVICES):
        for family in FAMILIES:
            seeds = sorted({c['seed'] for c in curves if c['family'] == family and c['device'] == label},
                           key=str)
            for i, seed in enumerate(seeds):
                pts = [(c['t'], c['eps_u']) for c in curves
                       if c['family'] == family and c['device'] == label and c['seed'] == seed and c['t'] > 0]
                if pts:
                    ax.plot(*zip(*pts), color=colors[family], alpha=0.7, lw=1,
                            label=family if i == 0 else None)
        lil = [c for c in curves if c['family'] == 'LiL-Q' and c['device'] == label
               and c['t'] and c['eps_u']]
        if lil:
            ax.plot([c['t'] for c in lil], [c['eps_u'] for c in lil], 'k*', ms=10, label='LiL-Q')
            for c in lil:
                ax.annotate(f"P={c['P']}", (c['t'], c['eps_u']), textcoords='offset points',
                            xytext=(5, 3), fontsize=7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('wall-clock time (s)')
        ax.set_title(label.upper())
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel(r'$\epsilon_u$ (relative $L^2$, test grid)')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Section 4.6: LiL-Q vs. the Component A baselines")
    ap.add_argument('--package', required=True, help='package1_results/ (holds B_instrumentation and A_calibration)')
    print(f"Wrote {build(ap.parse_args().package)}")


if __name__ == '__main__':
    main()
