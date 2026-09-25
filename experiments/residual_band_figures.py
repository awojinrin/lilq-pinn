"""
Residual-band Figures (Computational_Package_1_v2.md Section 3.5)
======================================================================

Regenerates the four figures that compare ||R_lin^(k)||_h and
||R^(k+1)||_h against k for each P (Bratu, Burgers, Buckley-Leverett
viscous, Buckley-Leverett gravity), from real LiL-Q reruns logged with
the shared Section 3.1 instrumentation (``lilq.iteration_log``). Per the
spec: y-axis labeled ||.||_h (not "MSE" -- the present repository has no
such figures at all, corrected or otherwise), log scale, one panel per
P, both curves, with chi_k in a second row of panels (the spec's stated
alternative to a secondary axis -- chosen here because chi_k's dynamic
range, log-scale by nature, does not share an axis cleanly with the two
already-log-scale residual curves).

Every problem/size/stopping-rule choice below is imported directly from
that problem's own ``experiments/run_*.py`` (not duplicated as separate
constants here) so this script is guaranteed to use the exact same
"paper settings" (Section 3.3) as the rest of this codebase already
does, with no risk of silently drifting from them.

Usage::

    python experiments/residual_band_figures.py
    python experiments/residual_band_figures.py --quick   # small smoke-test sizes
"""

import sys
import os
import csv
import argparse
import shutil
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lilq.iteration_log import IterationLogger, solve_rows
from problems.bratu import run_lil_q as run_bratu_lil_q
from problems.burgers import run_lil_q as run_burgers_lil_q
from problems.buckley_leverett import run_lil_q as run_bl_lil_q


RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'residual_band_figures'


# ─────────────────────────────────────────────────────────────────────────────
# Per-problem run configuration -- sizes/targets/budgets pulled straight
# from the existing experiment scripts (Section 3.3's "paper settings").
# Neither LiL-Q pretraining step (``pretrain_lil``, shared by all three
# problems) takes an epoch count -- it's a least-squares fit, not an
# epoch-based optimizer -- so ``pretrain_epochs`` from those scripts is
# irrelevant here and intentionally not imported.
# ─────────────────────────────────────────────────────────────────────────────

def _bratu_runs(quick=False):
    from experiments.run_bratu import DEFAULT_N_VALUES, paper_setup
    N_values = DEFAULT_N_VALUES[:2] if quick else DEFAULT_N_VALUES
    return [(N * N, *paper_setup(N)) for N in N_values]


def _burgers_runs(quick=False):
    from experiments.run_burgers import DEFAULT_N_VALUES, paper_setup
    N_values = DEFAULT_N_VALUES[:2] if quick else DEFAULT_N_VALUES
    return [(N * N, *paper_setup(N)) for N in N_values]


def _bl_runs(gravity, quick=False):
    from experiments.run_bl import DEFAULT_N_VALUES, paper_setup
    N_values = DEFAULT_N_VALUES[:2] if quick else DEFAULT_N_VALUES
    return [(N * N, *paper_setup(N, gravity)) for N in N_values]


# label, title, run-list builder, LiL-Q entry point
PROBLEMS = [
    ('bratu', 'Bratu', _bratu_runs, run_bratu_lil_q),
    ('burgers', 'Burgers', _burgers_runs, run_burgers_lil_q),
    ('bl', 'Buckley–Leverett (viscous)', lambda quick: _bl_runs(False, quick), run_bl_lil_q),
    ('bl_gravity', 'Buckley–Leverett (gravity)', lambda quick: _bl_runs(True, quick), run_bl_lil_q),
]


# ─────────────────────────────────────────────────────────────────────────────
# Run + log + plot
# ─────────────────────────────────────────────────────────────────────────────

def run_and_log(label, runs_fn, run_fn, out_dir, quick=False, verbose=True):
    """Run LiL-Q at every configured P for one problem, with full Section
    3.1 instrumentation, saving each run's own iterations.csv (the "CSV
    behind each panel" the spec asks for) and returning {P: IterationLogger}.
    """
    loggers = {}
    for P, config, opt in runs_fn(quick):
        logger = IterationLogger()
        if verbose:
            print(f"  P={P} ...", end=' ', flush=True)
        run_fn(config, opt, verbose=False, iteration_logger=logger)
        loggers[P] = logger
        logger.to_csv(out_dir / f"{label}_P{P}_iterations.csv")
        if verbose:
            n_solves = len(solve_rows(logger.rows))
            final_norm = logger.rows[-1]['norm_R_h'] if len(logger) else float('nan')
            print(f"{n_solves} iterations, final ||R||_h={final_norm:.3e}")
    return loggers


class _LoggedRows:
    """A read-back ``iterations.csv`` with the ``.rows`` interface of
    ``IterationLogger`` (all that ``plot_residual_bands`` uses)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)


def _parse_cell(value):
    if value == '':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return float(value)
    except ValueError:
        return value


def read_iterations_csv(path):
    with open(path, newline='') as f:
        rows = [{k: _parse_cell(v) for k, v in row.items()} for row in csv.DictReader(f)]
    for row in rows:
        row['k'] = int(row['k'])
    return _LoggedRows(rows)


def load_from_logs(label, logs_root, pass_='paper'):
    """``{P: rows}`` for one problem from the Section 3.3 driver's run
    folders (``<logs_root>/B_instrumentation/<label>_P<P>_cpu_<pass_>/``)."""
    loggers = {}
    prefix = f'{label}_P'
    for run_dir in sorted((Path(logs_root) / 'B_instrumentation').glob(f'{prefix}*_cpu_{pass_}')):
        P_label = run_dir.name[len(prefix):].split('_')[0]
        if not P_label.isdigit() or not (run_dir / 'iterations.csv').exists():
            continue  # e.g. 'bl_P64' must not pick up 'bl_gravity_P64'
        loggers[int(P_label)] = read_iterations_csv(run_dir / 'iterations.csv')
    return loggers


def plot_residual_bands(label, title, loggers, out_dir):
    """One figure per problem: top row = ||R_lin^(k)||_h & ||R^(k+1)||_h
    (log scale) vs k, one panel per P; bottom row = chi_k, same columns.
    """
    Ps = sorted(loggers.keys())
    n = len(Ps)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * max(n, 1), 6), squeeze=False)

    for col, P in enumerate(Ps):
        # Row k holds ||R_lin^(k)||_h and chi_k; ||R^(k+1)||_h is the next
        # row's norm_R_h (the terminal row supplies ||R^(K)||_h).
        rows = loggers[P].rows
        solves = solve_rows(rows)
        k = [r['k'] for r in solves]
        r_lin = [r['norm_Rlin_h'] for r in solves]
        r_next = [rows[i + 1]['norm_R_h'] for i in range(len(solves))]
        chi = [r['chi'] for r in solves]

        ax_top = axes[0, col]
        ax_top.semilogy(k, r_lin, 'o-', ms=3, lw=1.2, color='#457B9D',
                         label=r'$\|\mathbf{R}_{\mathrm{lin}}^{(k)}\|_h$')
        ax_top.semilogy(k, r_next, 's-', ms=3, lw=1.2, color='#E63946',
                         label=r'$\|\mathbf{R}^{(k+1)}\|_h$')
        ax_top.set_title(f'$P={P}$', fontsize=10)
        ax_top.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax_top.set_ylabel(r'$\|\cdot\|_h$')
            ax_top.legend(fontsize=7, loc='best')

        ax_bot = axes[1, col]
        ax_bot.semilogy(k, chi, 'd-', ms=3, lw=1.2, color='#6a3d9a')
        ax_bot.set_xlabel('$k$')
        for ax in (ax_top, ax_bot):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_bot.grid(True, alpha=0.3, which='both')
        if col == 0:
            ax_bot.set_ylabel(r'$\chi_k$')

    fig.suptitle(f'{title} — residual bands', fontsize=12)
    fig.tight_layout()

    pdf_path = out_dir / f"{label}_residual_bands.pdf"
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(out_dir / f"{label}_residual_bands.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Section 3.5 residual-band figures")
    parser.add_argument('--quick', action='store_true',
                        help='Smoke-test sizes (first 2 P per problem) instead of the full paper sweep')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: results/residual_band_figures/)')
    parser.add_argument('--from-logs', type=str, default=None,
                        help="Plot the Section 3.3 logs under this package root (experiments/"
                             "component_b.py's --out-root) instead of rerunning -- Section 3.5's "
                             "'from the LiL-Q logs of 3.3'.")
    parser.add_argument('--pass', dest='pass_', choices=('paper', 'kmax'), default='paper',
                        help="Which Section 3.3 pass to plot with --from-logs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import lilq.style  # noqa: F401
    except ImportError:
        pass

    for label, title, runs_fn, run_fn in PROBLEMS:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        if args.from_logs:
            loggers = load_from_logs(label, args.from_logs, args.pass_)
            if not loggers:
                print(f"  no {args.pass_}-pass logs for {label} under {args.from_logs}; skipping")
                continue
            for P in loggers:  # the CSV behind each panel, next to the figure
                run_dir = Path(args.from_logs) / 'B_instrumentation' / f'{label}_P{P}_cpu_{args.pass_}'
                shutil.copyfile(run_dir / 'iterations.csv', out_dir / f"{label}_P{P}_iterations.csv")
        else:
            loggers = run_and_log(label, runs_fn, run_fn, out_dir, quick=args.quick)
        pdf_path = plot_residual_bands(label, title, loggers, out_dir)
        print(f"  Saved {pdf_path}")

    print(f"\nAll figures and CSVs written to {out_dir}")


if __name__ == '__main__':
    main()
