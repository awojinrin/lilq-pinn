"""
Package 2, item 6a (P2-2): Buckley-Leverett nu-refinement
=========================================================

The advisor's instructions of 4 October 2026, Section 9.1, and checks C7
and C8 (Section 12.3).

**Runs.** The viscous Buckley-Leverett problem of Section 6.4 (N_g = 0, M = 0.5,
T = 0.4, S_0 = e^{-10x}, the initial-condition guess) with the viscosity
nu = |D| changed:
- nu in {0.1, 0.05, 0.02, 0.01} at P = 576 and 1,024, and P = 1,600
  (p_x = p_t = 40) at nu <= 0.02;
- the gravity case (N_g = -5, cosine x Fourier) at nu in {0.05, 0.02},
  P = 576 and 1,024, under the skip rule of Section 1 (it runs while 80 SU
  remain under item 6's cap; it costs minutes).

Everything else is the paper's configuration (``experiments/run_bl.py
paper_setup``): basis, random tensor grid (seed 42, N/P about 9.8), weights,
initial guess. nu < 0.01 is never run (Section 9.1, Section 12.4).

**Two passes per configuration**, as Package 1's:
- ``paper``: the paper's stopping rule (the loss target of
  ``run_bl.TARGET_LOSSES`` / ``GRAVITY_TARGET_LOSSES``), K_max = 60;
- ``kmax``: the target set to 0, so the iteration runs K_max = 60 steps,
  with the full ``iterations.csv``.

P = 1,600 has no paper target, so it has the kmax pass only, and its stop
is k = 60.

**References** (``reference``, check C7). At each (case, nu), the
finite-difference reference of ``problems.buckley_leverett.reference_solution``
is refined from Package 1's 4,000 intervals by doubling, until two
successive solutions agree to 1e-6 in relative L2 on the 201 x 201 test
grid. The finer of the pair is the reference:
``reference/bl_fd_ref_<case>_nu<nu>.npz`` (``x, t, u, meta``, the
``lilq.references`` format). The resolutions and differences go in
``reference/bl_fd_refinement.json``.

**Per configuration** (``nu_refinement.csv``):
- the iteration at which the paper's loss target is first met (if it is);
- ``eps_ref`` at the stop and at k = 60;
- kappa and the numerical rank at the last solve;
- the termination rule (Section 5.7, defaults n_s = 2, tau_chi = 0.1,
  tau_r = 0.01): the iterate it returns, and its classification, ``A``
  (approximation-limited) or ``C`` (conditioning/round-off-limited), as in
  ``experiments/stopping_rule_table.py``;
- the overshoot min S and max S - 1 on the test grid at the stop and at
  k = 60 (Gibbs onset).

**Check C8.** At nu = 0.1, P = 576 and 1,024, each pass's ``norm_R_h``
equals package1's (``bl_P<P>_cpu_<pass>``) at every k to 1e-10 relative.

**Outputs** under ``P2_2_nu_refinement/``:
- ``<case>_P<P>_nu<nu>_<pass>/{run.json, iterations.csv, solution.pt}``;
- ``nu_refinement.csv``, ``check_c7.json``, ``check_c8.json``;
- ``figures/nu_refinement.{pdf,png}``.

Usage::

    python experiments/p2_2_nu_refinement.py reference --out <stage root>
    python experiments/p2_2_nu_refinement.py runs --out <stage root> --package1 <package1> [--cases viscous gravity]
"""

import argparse
import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from lilq.source_lock import current_commit

NUS = {'viscous': (0.1, 0.05, 0.02, 0.01), 'gravity': (0.05, 0.02)}
SIZES = {'viscous': (24, 32), 'gravity': (24, 32)}        # modes per direction: P = 576, 1,024
EXTRA_SIZE, EXTRA_NU_MAX = 40, 0.02                       # P = 1,600 at nu <= 0.02 (viscous)
NU_MIN = 0.01
K_MAX = 60
FIRST_INTERVALS, MAX_INTERVALS, AGREEMENT = 4000, 64000, 1e-6
C8_TOLERANCE = 1e-10
RULE = {'n_s': 2, 'tau_chi': 0.1, 'tau_r': 0.01}


def nu_label(nu):
    return f'{nu:g}'


def configs(case):
    """``[(N, nu)]`` of a case, in run order."""
    out = [(N, nu) for nu in NUS[case] for N in SIZES[case]]
    if case == 'viscous':
        out += [(EXTRA_SIZE, nu) for nu in NUS[case] if nu <= EXTRA_NU_MAX]
    if any(nu < NU_MIN for _, nu in out):
        raise ValueError('nu < 0.01 is not run (Section 9.1)')
    return out


def setup(case, N, nu):
    """``(config, opt, target)``: the paper's configuration at N with D = -nu;
    ``target`` is the paper's loss target, or None (no paper target at N)."""
    from experiments.run_bl import GRAVITY_TARGET_LOSSES, TARGET_LOSSES, paper_setup
    config, opt = paper_setup(N, case == 'gravity')
    config = dataclasses.replace(config, D_coef=-nu)
    target = (GRAVITY_TARGET_LOSSES if case == 'gravity' else TARGET_LOSSES).get(N)
    assert opt.max_quasi_iters_lil == K_MAX
    return config, opt, target


def reference_path(ref_dir, case, nu):
    return Path(ref_dir) / f'bl_fd_ref_{case}_nu{nu_label(nu)}.npz'


# ---------------------------------------------------------------- references (C7)

def refine(config, first=FIRST_INTERVALS, last=MAX_INTERVALS, agreement=AGREEMENT):
    """Double the finite-difference resolution from ``first`` until two
    successive solutions agree to ``agreement`` (relative L2 on the test
    grid). Returns ``(field, record)``; the field is the finer of the pair."""
    from problems.buckley_leverett import reference_solution
    steps, prev, n = [], None, first
    while n <= last:
        u = np.array(reference_solution(config, n))
        # SciPy's BDF can warn about an invalid value on a trial step it then rejects;
        # the accepted solution must be finite and a saturation (round-off aside)
        if not np.isfinite(u).all() or u.min() < -1e-12 or u.max() > 1 + 1e-12:
            raise RuntimeError(f'reference at {n} intervals is not a finite saturation in [0, 1]')
        diff = None if prev is None else float(np.linalg.norm(u - prev) / np.linalg.norm(u))
        steps.append({'n_intervals': n, 'rel_diff_to_previous': diff})
        if diff is not None and diff < agreement:
            return u, {'n_intervals': n, 'steps': steps, 'passed': True}
        prev, n = u, 2 * n
    return prev, {'n_intervals': n // 2, 'steps': steps, 'passed': False}


def write_references(out_root, cases=('viscous', 'gravity')):
    from lilq.references import save_npz_atomic, write_text_atomic
    from problems.buckley_leverett import TEST_GRID
    ref_dir = Path(out_root) / 'reference'
    ref_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        for nu in NUS[case]:
            config, _, _ = setup(case, SIZES[case][0], nu)
            u, rec = refine(config)
            x = np.linspace(*config.x_domain, TEST_GRID[0])
            t = np.linspace(0.0, config.T_final, TEST_GRID[1])
            meta = {'problem': f'Buckley-Leverett, {case}, nu = {nu_label(nu)} (D = {config.D_coef:g}), N_g = '
                               f'{config.N_g:g}, M = {config.M_param:g}, T = {config.T_final:g}',
                    'method': 'finite differences (problems.buckley_leverett.reference_solution), refined by '
                              f'doubling from {FIRST_INTERVALS} intervals until two successive solutions agree '
                              f'to {AGREEMENT:g}', 'grid': '201 x 201, u[i, j] at (x[i], t[j])',
                    **rec, 'commit': current_commit()}
            save_npz_atomic(reference_path(ref_dir, case, nu), x=x, t=t, u=u, meta=np.array(json.dumps(meta)))
            records.append({'case': case, 'nu': nu, **rec})
            last = rec['steps'][-1]['rel_diff_to_previous']
            print(f"  {case:7s} nu = {nu_label(nu):5s}: {rec['n_intervals']} intervals, last difference "
                  f"{last:.2e} ({'C7 passed' if rec['passed'] else 'C7 FAILED'})", flush=True)
    c7 = {'check': 'C7 (item 6)', 'agreement': AGREEMENT, 'passed': all(r['passed'] for r in records),
          'references': records, 'commit': current_commit()}
    write_text_atomic(ref_dir / 'bl_fd_refinement.json', json.dumps(c7, indent=2))
    return c7


# ---------------------------------------------------------------- runs

def run_one(case, N, nu, pass_, out, ref_dir):
    """One pass of one configuration; returns its run folder."""
    from lilq.iteration_log import IterationLogger
    from lilq.saved_models import save_solution
    from problems.buckley_leverett import run_lil_q
    config, opt, target = setup(case, N, nu)
    if pass_ == 'kmax':
        opt = dataclasses.replace(opt, R_tol=0.0, max_quasi_iters_lil=K_MAX)
    elif target is None:
        raise ValueError(f'no paper target at N = {N}: kmax pass only')
    run_dir = Path(out) / f'{case}_P{N * N}_nu{nu_label(nu)}_{pass_}'
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    basis, coeffs, _, summary = run_lil_q(config, opt, verbose=False, iteration_logger=logger,
                                          run_json_path=run_dir / 'run.json',
                                          reference_npz=reference_path(ref_dir, case, nu))
    logger.to_csv(run_dir / 'iterations.csv')
    save_solution(run_dir, {'u': (basis, coeffs)}, config, opt)
    return run_dir, basis, coeffs, summary


def _log(run_dir):
    with open(Path(run_dir) / 'iterations.csv', newline='') as f:
        return list(csv.DictReader(f))


def _f(v):
    return float(v) if v not in ('', None) else float('nan')


def overshoot(basis, coeffs, config):
    from lilq.test_errors import tensor_grid_values
    from problems.buckley_leverett import TEST_GRID
    x = np.linspace(*config.x_domain, TEST_GRID[0])
    t = np.linspace(0.0, config.T_final, TEST_GRID[1])
    S = tensor_grid_values(basis, coeffs, [x, t])
    return float(S.min()), float(S.max() - 1.0)


def summarize_config(case, N, nu, kmax_dir, kmax_fit, paper=None):
    """One row of ``nu_refinement.csv`` from the kmax pass (and the paper pass)."""
    import experiments.stopping_rule_table as srt
    config, _, target = setup(case, N, nu)
    log = _log(kmax_dir)
    d = srt.read_log(Path(kmax_dir) / 'iterations.csv')
    i, _ = srt.rule_index(d, RULE['n_s'], RULE['tau_chi'], RULE['tau_r'])
    eps = [_f(r['eps_ref']) for r in log]
    k_target = next((int(r['k']) for r in log[1:] if target is not None and _f(r['norm_R_h']) ** 2 < target), None)
    # the last row with a solve (the terminal row k = K has none; read from the CSV, its cells are empty)
    last = next(r for r in reversed(log) if r['norm_Rlin_h'] not in ('', None))
    min60, over60 = overshoot(*kmax_fit, config)
    row = {'case': case, 'nu': nu, 'P': N * N, 'target_mse': target if target is not None else '',
           'k_target': k_target if k_target is not None else '',
           'eps_ref_60': eps[-1], 'kappa': _f(last['kappa']), 'kappa_retained': _f(last['kappa_retained']),
           'rank': int(_f(last['num_rank_svd'])), 'P_trial': N * N,
           'k_rule': i + 1 if i is not None else '', 'rule_class': srt.classify(d, i) if i is not None else 'never',
           'eps_ref_rule': eps[i + 1] if i is not None else '',
           'min_S_60': min60, 'max_S_minus_1_60': over60}
    if paper is not None:
        p_dir, p_fit = paper
        p_log = _log(p_dir)
        min_s, over = overshoot(*p_fit, config)
        row.update(k_stop=int(p_log[-1]['k']), stop='target' if _f(p_log[-1]['norm_R_h']) ** 2 < target else 'K_max',
                   eps_ref_stop=_f(p_log[-1]['eps_ref']), min_S_stop=min_s, max_S_minus_1_stop=over)
    else:
        row.update(k_stop=K_MAX, stop='K_max (no paper target at this P)', eps_ref_stop=eps[-1],
                   min_S_stop=min60, max_S_minus_1_stop=over60)
    return row


def _max_rel(a_rows, b_rows):
    return max(abs(_f(a['norm_R_h']) - _f(b['norm_R_h'])) / abs(_f(b['norm_R_h'])) for a, b in zip(a_rows, b_rows))


def check_c8(out, package1, sizes=(24, 32), tol=C8_TOLERANCE):
    """C8: at nu = 0.1 each pass's norm_R_h equals package1's at every k to
    ``tol`` (pass/fail exactly as Section 12.3 states). Recorded beside it,
    so that the result can be read:

    - ``same_stop`` and the relative differences of the loss and of
      ``eps_u`` at the last row;
    - ``package1_paper_vs_kmax``: how closely package1's own two passes (from
      different Grace jobs) agree over the rows they share. That is
      Grace's run-to-run reproducibility of the same configuration, the floor
      of any rerun's agreement. At kappa about 1e16-1e17 it is about 1e-4,
      not 1e-10 (DECISIONS.md, batch 1)."""
    rows = []
    for N in sizes:
        p1 = {pass_: _log(Path(package1) / 'B_instrumentation' / f'bl_P{N * N}_cpu_{pass_}')
              for pass_ in ('paper', 'kmax')}
        n_paper = len(p1['paper'])
        p1_self = _max_rel(p1['paper'], p1['kmax'][:n_paper])
        for pass_ in ('paper', 'kmax'):
            new = _log(Path(out) / f'viscous_P{N * N}_nu0.1_{pass_}')
            old = p1[pass_]
            same_k = [r['k'] for r in new] == [r['k'] for r in old]
            rel = _max_rel(new, old)
            rows.append({'run': f'bl_P{N * N}_cpu_{pass_}', 'rows': len(new), 'rows_package1': len(old),
                         'same_iterations': same_k, 'max_rel_diff_norm_R_h': rel,
                         'passed': bool(same_k and rel <= tol),
                         'same_stop': new[-1]['k'] == old[-1]['k'],
                         'final_loss_rel_diff': abs(_f(new[-1]['norm_R_h']) ** 2 - _f(old[-1]['norm_R_h']) ** 2)
                         / _f(old[-1]['norm_R_h']) ** 2,
                         'final_eps_u_rel_diff': abs(_f(new[-1]['eps_u']) - _f(old[-1]['eps_u'])) / _f(old[-1]['eps_u']),
                         'package1_paper_vs_kmax': p1_self})
    return {'check': 'C8 (item 6)', 'tolerance': tol, 'passed': all(r['passed'] for r in rows), 'runs': rows,
            'commit': current_commit()}


def run_all(out_root, package1=None, cases=('viscous', 'gravity'), ref_dir=None):
    from lilq.provenance import save_provenance
    out = Path(out_root) / 'P2_2_nu_refinement'
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(ref_dir) if ref_dir else Path(out_root) / 'reference'
    for case in cases:
        for nu in NUS[case]:
            if not reference_path(ref_dir, case, nu).exists():
                raise FileNotFoundError(f'{reference_path(ref_dir, case, nu)}: run the reference stage first')
    rows = []
    for case in cases:
        for N, nu in configs(case):
            k_dir, k_basis, k_coeffs, _ = run_one(case, N, nu, 'kmax', out, ref_dir)
            paper = None
            if setup(case, N, nu)[2] is not None:
                p_dir, p_basis, p_coeffs, _ = run_one(case, N, nu, 'paper', out, ref_dir)
                paper = (p_dir, (p_basis, p_coeffs))
            r = summarize_config(case, N, nu, k_dir, (k_basis, k_coeffs), paper)
            rows.append(r)
            print(f"  {case:7s} P = {r['P']:4d} nu = {nu_label(nu):5s}: stop k = {r['k_stop']} ({r['stop']}), "
                  f"eps_ref {r['eps_ref_stop']:.3e} / k=60 {r['eps_ref_60']:.3e}, rule {r['k_rule']} "
                  f"({r['rule_class']}), kappa {r['kappa']:.1e}, rank {r['rank']}, min S {r['min_S_stop']:.4f}, "
                  f"max S-1 {r['max_S_minus_1_stop']:.4f}", flush=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'nu_refinement.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    c7 = json.loads((ref_dir / 'bl_fd_refinement.json').read_text())
    (out / 'check_c7.json').write_text(json.dumps(c7, indent=2))
    if package1 and 'viscous' in cases:
        c8 = check_c8(out, package1)
        (out / 'check_c8.json').write_text(json.dumps(c8, indent=2))
        print(f"  C8: {'passed' if c8['passed'] else 'FAILED'} (max relative difference "
              f"{max(r['max_rel_diff_norm_R_h'] for r in c8['runs']):.1e}; package1's own two passes agree to "
              f"{max(r['package1_paper_vs_kmax'] for r in c8['runs']):.1e})")
    save_provenance(out)
    figure(out, rows)
    return rows


def figure(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for case, mk in (('viscous', 'o'), ('gravity', 's')):
        for P in sorted({r['P'] for r in rows if r['case'] == case}):
            g = sorted((r for r in rows if r['case'] == case and r['P'] == P), key=lambda r: r['nu'])
            nu = [r['nu'] for r in g]
            axes[0].loglog(nu, [r['eps_ref_stop'] for r in g], mk + '-', ms=4, label=f'{case}, P = {P}')
            axes[1].loglog(nu, [max(-r['min_S_stop'], r['max_S_minus_1_stop'], 1e-16) for r in g], mk + '-', ms=4)
    axes[0].set(xlabel=r'$\nu$', ylabel='eps_ref at the stop', title='error against the refined reference')
    axes[1].set(xlabel=r'$\nu$', ylabel='max(-min S, max S - 1)', title='overshoot on the test grid (stop)')
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures' / 'nu_refinement.pdf')
    fig.savefig(out / 'figures' / 'nu_refinement.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 2, item 6a: Buckley-Leverett nu-refinement.')
    ap.add_argument('stage', choices=('reference', 'runs'))
    ap.add_argument('--out', required=True, help='the stage root')
    ap.add_argument('--package1', help='package1 (check C8)')
    ap.add_argument('--cases', nargs='+', choices=('viscous', 'gravity'), default=['viscous', 'gravity'])
    args = ap.parse_args(argv)
    if args.stage == 'reference':
        c7 = write_references(args.out, args.cases)
        print(f"C7 {'passed' if c7['passed'] else 'FAILED'}")
        return
    run_all(args.out, args.package1, args.cases)


if __name__ == '__main__':
    main()
