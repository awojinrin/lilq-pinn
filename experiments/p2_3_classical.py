"""
Package 2, item 1 (P2-3): the classical baseline on Bratu -- references and the port check
=========================================================================================

Stage 1 of the advisor's instructions of 4 October 2026 (Sections 3, 4.1,
6.2, 12.3 C1 and C2), with ``baselines/square_chebyshev.py``:

``reference``
    The Bratu reference, SQ-CGL at p = 48, and the p = 64 solution for its
    error estimate, each stored with its coefficient vector and its field on
    the 201 x 201 test grid (``reference/bratu_ref_p48.npz``,
    ``bratu_ref_p64.npz``), and ``bratu_reference_error.json``: the relative
    discrete L2 distance between the two on the test grid (check C2: it must
    be below every reported Bratu error by a factor of 10 or more).
``check-pilot``
    Check C1: the port against the pilot's ``rows.json`` at p = 12, 20
    (LS-hard-CGL-1.5) and p = 14, 22 (SQ-CGL), to three digits, on the
    pilot's 101 x 101 error grid against the same p = 48 reference; and the
    sanity run: LS-hard with N = P on the interior CGL points of the
    12-point grid (p = 10) reproduces SQ-CGL at p = 12 to round-off.
    Writes ``check_c1.csv`` and ``check_c1.json``.
``kovasznay-pilot``
    Section 4.2, step 1: the square P_N - P_{N-2} Kovasznay system
    (``baselines/square_kovasznay.py``) at p_d = 10 and 15, with the pressure
    pin at the corner and, for comparison, at the interior CGL point nearest
    it: rank, kappa_2 (SVD, at the final iterate and at zero), the Newton
    iteration count from zero, and the errors on the 301 x 401 test grid.
    Writes ``kovasznay/pilot.json`` and ``kovasznay/pilot.md``.

Stage 2 (Sections 4.1, 4.2 step 2, 4.4; the advisor's Stage-1 reply, Section 4):

``bratu``
    Every Bratu method and size of Section 4.1: SQ-CGL at p = 7, 12, 17 and
    6, 8, ..., 24; LS-hard-CGL-1.5 at p = 5, 10, 15 and 6, 8, ..., 24;
    LS-weakMS-CGL-3 (lambda_bc = 10) at p = 5, 10, 15. Per run: one
    diagnostic run (the per-iteration error against the p = 48 reference,
    the plateau iterate, kappa and rank by SVD), then the timing protocol --
    one untimed warm-up, then the median of three clean runs with the
    diagnostics off, assembly and solve timed separately, the time to the
    plateau iterate and to the stop. Then the paper's LiL-Q at P = 25, 100,
    225 under the same protocol (the same-day reference; its time must agree
    with package1's to 15%). Writes ``bratu/rows.csv``,
    ``bratu/history_<method>_p<p>.csv``, ``bratu/timing_lilq.csv`` and
    ``bratu/run.json``.
``kovasznay``
    Section 4.2, step 2: the square system with the corner pin at p_d = 10,
    15, 20, 25 (K_max = 60), the same diagnostics and timing protocol, the
    errors E_u, E_v, E_p in the pin gauge and mean-free, and the unknowns
    split into velocity and pressure (the advisor's Stage-1 reply, Section
    4, item 1). Then, the same day, the paper's LiL-Q (P = 75 .. 1,875) and
    wave 4's Clenshaw-Curtis CGL runs at N/P = 5 (B10). Writes
    ``kovasznay/rows.csv``, ``kovasznay/history_SQ-PNPN2_pd<p>.csv``,
    ``kovasznay/timing_lilq.csv`` and ``kovasznay/run.json``.
``figures``
    The work-precision figure of Section 4.4 (error against free
    coefficients and against time, one panel each per problem), from the
    stage outputs, the Stage-1 reference errors of the LiL-Q runs and
    package1. Writes ``figures/work_precision.{pdf,png}``.

Usage::

    python experiments/p2_3_classical.py reference --out <package2_results>
    python experiments/p2_3_classical.py check-pilot --pilot-rows rows.json --out <package2_results>
    python experiments/p2_3_classical.py kovasznay-pilot --out <package2_results>
    python experiments/p2_3_classical.py bratu --out <stage root> --package1 <package1>
    python experiments/p2_3_classical.py kovasznay --out <stage root> --package1 <package1>
    python experiments/p2_3_classical.py figures --out <stage root> --package1 <package1> \\
        --lilq-errors <stage 1>/P2_12_reference_errors/scalar_reference_errors.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy: the timing stages)
import numpy as np

import baselines.square_chebyshev as sc
from lilq.source_lock import current_commit

TEST_AXIS = np.linspace(0, 1, 201)        # the 201 x 201 test grid on [0, 1]^2
PILOT_AXIS = np.linspace(0, 1, 101)       # the pilot's error grid
REF_P, CHECK_P = 48, 64
C1_CASES = (('LS-hard-CGL-1.5', 12), ('LS-hard-CGL-1.5', 20), ('SQ-CGL', 14), ('SQ-CGL', 22))


def write_reference(out_dir):
    ref_dir = Path(out_dir) / 'reference'
    ref_dir.mkdir(parents=True, exist_ok=True)
    fields = {}
    for p in (REF_P, CHECK_P):
        beta, space, u, res = sc.reference(p, TEST_AXIS)
        fields[p] = u
        np.savez_compressed(ref_dir / f'bratu_ref_p{p}.npz', coefficients=beta, p=p, x=TEST_AXIS, y=TEST_AXIS, u=u,
                            meta=np.array(json.dumps({
                                'problem': 'Bratu, lambda = 6.2, u = 0 on the boundary of (0,1)^2, lower branch',
                                'method': f'SQ-CGL at p = {p}: square Chebyshev collocation on the {p} x {p} CGL grid, '
                                          'Newton from zero, LU', 'basis': 'T_i(2x-1) T_j(2y-1), i, j < p, x-index outer',
                                'newton_iterations': res['k_stop'], 'commit': current_commit()})))
    err = float(np.linalg.norm(fields[REF_P] - fields[CHECK_P]) / np.linalg.norm(fields[CHECK_P]))
    record = {'reference': f'SQ-CGL p = {REF_P}', 'check': f'SQ-CGL p = {CHECK_P}', 'test_grid': '201 x 201 on [0,1]^2',
              'reference_error': err, 'max_abs_difference': float(np.abs(fields[REF_P] - fields[CHECK_P]).max()),
              'commit': current_commit()}
    (ref_dir / 'bratu_reference_error.json').write_text(json.dumps(record, indent=2))
    return record


def check_pilot(pilot_rows, out_dir):
    with open(pilot_rows) as f:
        pilot = {(r['m'], r['p']): r for r in json.load(f)}
    _, _, ref, _ = sc.reference(REF_P, PILOT_AXIS)
    rows = []
    for label, p in C1_CASES:
        r = sc.newton(label, p, reference=(PILOT_AXIS, ref))
        pv = pilot[(label, p)]
        rel = abs(r['err_final'] - pv['Eend']) / pv['Eend']
        rows.append({'method': label, 'p': p, 'free_coeff': r['free_coeff'], 'N': r['N'], 'N_pilot': pv['N'],
                     'err_final': r['err_final'], 'err_pilot': pv['Eend'], 'rel_difference': rel,
                     'three_digits': f"{r['err_final']:.2e}" == f"{pv['Eend']:.2e}",
                     'kappa': r['kappa'], 'kappa_pilot': pv['kappa'], 'k_stop': r['k_stop'], 'k_plateau': r['k_plateau']})
    sq = sc.newton('SQ-CGL', 12)
    space, layout = sc.TrialSpace('hard', 10), sc.hard_layout(10, None, m=12)
    rws, beta = sc.Rows(space, layout), np.zeros(space.n)
    for _ in range(sc.K_MAX):
        A, f, _ = sc.assemble(rws, beta)
        new = sc.solve_linear(A, f, square=False)
        done = np.linalg.norm(new - beta) < sc.TOL * np.linalg.norm(new)
        beta = new
        if done:
            break
    diff = float(np.abs(space.grid_values(beta, PILOT_AXIS) - sq['space'].grid_values(sq['beta'], PILOT_AXIS)).max())
    sanity = {'LS_hard_p': 10, 'grid_points_per_direction': 12, 'N': layout.N, 'P': space.n,
              'SQ_CGL_p': 12, 'max_abs_difference': diff, 'round_off': diff < 1e-12}
    out = Path(out_dir) / 'P2_3_classical'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'check_c1.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = {'check': 'C1', 'error_grid': '101 x 101 (the pilot\'s)', 'reference': f'SQ-CGL p = {REF_P}',
              'cases': rows, 'sanity': sanity,
              'passed': all(r['three_digits'] for r in rows) and sanity['round_off'], 'commit': current_commit()}
    (out / 'check_c1.json').write_text(json.dumps(result, indent=2))
    return result


PILOT_SIZES = (10, 15)
PILOT_KEYS = ('p_d', 'pin', 'pin_point', 'n_unknowns', 'n_equations', 'P_velocity', 'P_pressure', 'rank', 'kappa',
              'rank_at_zero', 'kappa_at_zero', 'k_stop', 'stop', 'eps_u', 'eps_v', 'eps_p', 'eps_p_meanfree')


def kovasznay_pilot(out_dir):
    import baselines.square_kovasznay as sk
    rows = []
    for p_d in PILOT_SIZES:
        for pin in ('corner', 'interior'):
            r = sk.newton(p_d, pin)
            rows.append({**{k: r[k] for k in PILOT_KEYS}, 'rel_dbeta': [h['rel_dbeta'] for h in r['history']]})
    full = {r['p_d']: r['rank'] == r['n_unknowns'] for r in rows if r['pin'] == 'corner'}
    result = {'section': '4.2, step 1', 'rows': rows, 'full_rank_with_corner_pin': full,
              'decision_input': 'full rank with the corner pin at every size' if all(full.values())
              else 'rank-deficient with the corner pin', 'commit': current_commit()}
    out = Path(out_dir) / 'P2_3_classical' / 'kovasznay'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pilot.json').write_text(json.dumps(result, indent=2))
    lines = ['# Kovasznay square-system pilot (Package 2, Section 4.2, step 1)', '',
             'Square spectral collocation, P_N - P_{N-2}: velocity in the tensor Chebyshev space of p_d modes per '
             'direction, pressure in p_d - 2; momentum and continuity at the (p_d - 2)^2 interior CGL points, the '
             'velocity Dirichlet data at the 4 p_d - 4 boundary CGL points; one continuity row (at the interior point '
             'nearest (-0.5, -0.5)) replaced by the pressure pin. Re = 40; Newton from zero; stop at relative '
             'coefficient change < 1e-9, K_max = 60; errors on the 301 x 401 test grid. Rank and kappa_2 by SVD of the '
             'Jacobian at the final iterate (rank threshold n eps sigma_max).', '',
             '| p_d | unknowns = equations | pin | rank | kappa_2 | rank, kappa_2 at zero | Newton iterations | '
             'eps_u | eps_v | eps_p (pin gauge) | eps_p (mean-free) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['p_d']} | {r['n_unknowns']} | {r['pin']} {tuple(round(c, 4) for c in r['pin_point'])} | "
                     f"{r['rank']} | {r['kappa']:.2e} | {r['rank_at_zero']}, {r['kappa_at_zero']:.2e} | {r['k_stop']} | "
                     f"{r['eps_u']:.2e} | {r['eps_v']:.2e} | {r['eps_p']:.2e} | {r['eps_p_meanfree']:.2e} |")
    lines += ['', f"**Result:** {result['decision_input']}. The interior pin (the fallback) is shown for comparison; it "
              'changes only the pressure gauge (the velocity and mean-free pressure errors are identical).', '',
              f"Commit `{current_commit()}`; `python experiments/p2_3_classical.py kovasznay-pilot`."]
    (out / 'pilot.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


# ---------------------------------------------------------------- Stage 2: runs and timing

REPEATS = 3                     # the median of three clean runs, after one warm-up (Section 4.1, Timing)
SAME_DAY_TOLERANCE = 0.15       # the paper's LiL-Q times must agree with package1's to 15%
BRATU_RUNS = ([('SQ-CGL', p) for p in sorted({7, 12, 17} | set(range(6, 25, 2)))]
              + [('LS-hard-CGL-1.5', p) for p in sorted({5, 10, 15} | set(range(6, 25, 2)))]
              + [('LS-weakMS-CGL-3', p) for p in (5, 10, 15)])
KOVASZNAY_SIZES = (10, 15, 20, 25)
B10_SAME_DAY = (10, 20, 25)     # wave 4's Clenshaw-Curtis CGL runs at N/P = 5 (P = 300, 1,200, 1,875)


def plateau(errors, share=sc.PLATEAU):
    """The first iterate whose error is within ``share`` of the final one."""
    return next(i for i, e in enumerate(errors) if e <= (1 + share) * errors[-1])


def iteration_times(history, k_stop, k_plateau):
    """Assembly and solve time summed over the solves to the stop, and the
    time to the plateau iterate (the first ``k_plateau`` solves)."""
    solves = [h for h in history if h.get('t_solve_s') is not None]
    n = k_stop if k_stop is not None else len(solves)
    ta = sum(h['t_assemble_s'] for h in solves[:n])
    ts = sum(h['t_solve_s'] for h in solves[:n])
    tp = sum(h['t_assemble_s'] + h['t_solve_s'] for h in solves[:k_plateau])
    return {'t_assemble_s': ta, 't_solve_s': ts, 't_plateau_s': tp, 't_stop_s': ta + ts}


def timed(run, k_plateau, repeats=REPEATS):
    """One untimed warm-up, then ``repeats`` clean runs (``run()`` returns the
    solver's result with the diagnostics off). Each quantity is the median of
    its ``repeats`` values; the clean runs must stop where the diagnostic run
    did."""
    run()
    results = [run() for _ in range(repeats)]
    per_run = [iteration_times(r['history'], r['k_stop'], k_plateau) for r in results]
    out = {q: float(np.median([t[q] for t in per_run])) for q in per_run[0]}
    out['t_stop_runs_s'] = json.dumps([t['t_stop_s'] for t in per_run])
    out['k_stop_clean'] = sorted({r['k_stop'] for r in results})
    return out


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _run_json(out, stage, extra):
    from lilq.provenance import save_provenance
    save_provenance(out)
    threads = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    (out / 'run.json').write_text(json.dumps({
        'stage': stage, 'commit': current_commit(), 'device': 'cpu', 'threads': threads,
        'slurm_job': os.environ.get('SLURM_JOB_ID'), 'timing': f'one untimed warm-up, then the median of {REPEATS} '
        'clean runs (diagnostics off); assembly and solve timed separately', **extra}, indent=2))


def _load_reference(reference_dir, out_dir):
    ref = Path(reference_dir) if reference_dir else Path(out_dir) / 'reference'
    if not (ref / f'bratu_ref_p{REF_P}.npz').exists():
        write_reference(ref.parent)
    return {p: np.load(ref / f'bratu_ref_p{p}.npz')['u'] for p in (REF_P, CHECK_P)}


def _package1_clean_times(package1):
    path = Path(package1) / 'B_instrumentation' / 'clean_timing' / 'clean_timing.csv'
    with open(path, newline='') as f:
        return {(r['run'], r['quantity']): float(r['clean_time_s']) for r in csv.DictReader(f) if r['clean_time_s']}


def same_day_lilq(runs, quantity, package1=None, repeats=REPEATS):
    """The paper's LiL-Q under the same protocol, beside package1's clean time."""
    p1 = _package1_clean_times(package1) if package1 else {}
    rows = []
    for run in runs:
        run.run_once()                                              # warm-up
        times = [run.run_once() for _ in range(repeats)]
        t = float(np.median([x[quantity] for x in times]))
        ref = p1.get((run.name, quantity))
        rows.append({'run': run.name, 'quantity': quantity, 'iterations': times[0]['iterations'],
                     'time_s': t, 'time_runs_s': json.dumps([x[quantity] for x in times]),
                     'package1_time_s': ref if ref is not None else '',
                     'ratio_to_package1': t / ref if ref else '',
                     'within_15pct': abs(t / ref - 1) <= SAME_DAY_TOLERANCE if ref else ''})
    return rows


def bratu_stage(out_dir, package1=None, reference_dir=None, runs=BRATU_RUNS, repeats=REPEATS, same_day=True):
    out = Path(out_dir) / 'P2_3_classical' / 'bratu'
    ref = _load_reference(reference_dir, out_dir)
    rows = []
    for label, p in runs:
        d = sc.newton(label, p, reference=(TEST_AXIS, ref[REF_P]))
        t = timed(lambda: sc.newton(label, p, diagnostics=False), d['k_plateau'], repeats)
        assert t['k_stop_clean'] == [d['k_stop']], (label, p, t['k_stop_clean'], d['k_stop'])
        hist = [{k: h.get(k) for k in ('k', 'err', 'norm_R', 'norm_Rlin', 'rel_dbeta', 'chi', 't_assemble_s',
                                       't_solve_s')} for h in d['history']]
        _write_csv(out / f'history_{label}_p{p}.csv', hist)
        rows.append({'method': label, 'p': p, 'free_coeff': d['free_coeff'], 'P_trial': d['P_trial'], 'N': d['N'],
                     'k_stop': d['k_stop'], 'stop': d['stop'], 'k_plateau': d['k_plateau'],
                     'err_ref48': d['err_final'], 'err_ref64': sc.error(d['space'], d['beta'], TEST_AXIS, ref[CHECK_P]),
                     't_assemble_s': t['t_assemble_s'], 't_solve_s': t['t_solve_s'], 't_plateau_s': t['t_plateau_s'],
                     't_stop_s': t['t_stop_s'], 't_stop_runs_s': t['t_stop_runs_s'], 'kappa': d['kappa'],
                     'rank': d['rank']})
        print(f"  {label:16s} p = {p:2d}: {d['k_stop']} iterations, plateau {d['k_plateau']}, "
              f"error {d['err_final']:.2e}, {t['t_stop_s']:.4f} s")
    _write_csv(out / 'rows.csv', rows)
    timing = []
    if same_day:
        import experiments.clean_timing as ct
        timing = same_day_lilq(ct._scalar('bratu', smoke=False), 'training_time', package1, repeats)
        _write_csv(out / 'timing_lilq.csv', timing)
    _run_json(out, 'bratu', {'section': '4.1', 'K_max': sc.K_MAX, 'tol': sc.TOL, 'reference': f'SQ-CGL p = {REF_P}',
                             'error_grid': '201 x 201', 'lambda_bc_weakMS': 10.0, 'runs': len(rows)})
    return rows, timing


def kovasznay_stage(out_dir, package1=None, sizes=KOVASZNAY_SIZES, repeats=REPEATS, same_day=True, b10=B10_SAME_DAY):
    import baselines.square_kovasznay as sk
    out = Path(out_dir) / 'P2_3_classical' / 'kovasznay'
    rows = []
    for p_d in sizes:
        d = sk.newton(p_d, 'corner')
        k_plat = plateau([h['eps_u'] for h in d['history']] + [d['eps_u']])
        t = timed(lambda: sk.newton(p_d, 'corner', diagnostics=False), k_plat, repeats)
        assert t['k_stop_clean'] == [d['k_stop']], (p_d, t['k_stop_clean'], d['k_stop'])
        _write_csv(out / f'history_SQ-PNPN2_pd{p_d}.csv',
                   [{k: h.get(k) for k in ('k', 'eps_u', 'eps_v', 'eps_p', 'eps_p_meanfree', 'rel_dbeta',
                                           't_assemble_s', 't_solve_s')} for h in d['history']])
        rows.append({'method': 'SQ-PNPN2', 'p_d': p_d, 'P_velocity': d['P_velocity'], 'P_pressure': d['P_pressure'],
                     'free_coeff': d['n_unknowns'], 'velocity_per_component': p_d * p_d,
                     'k_stop': d['k_stop'], 'stop': d['stop'], 'k_plateau': k_plat,
                     'eps_u': d['eps_u'], 'eps_v': d['eps_v'], 'eps_p': d['eps_p'], 'eps_p_meanfree': d['eps_p_meanfree'],
                     't_assemble_s': t['t_assemble_s'], 't_solve_s': t['t_solve_s'], 't_plateau_s': t['t_plateau_s'],
                     't_stop_s': t['t_stop_s'], 't_stop_runs_s': t['t_stop_runs_s'], 'kappa': d['kappa'],
                     'rank': d['rank'], 'pin': 'corner'})
        print(f"  p_d = {p_d}: {d['n_unknowns']} unknowns ({d['P_velocity']} + {d['P_pressure']}), rank {d['rank']}, "
              f"{d['k_stop']} iterations, E_u {d['eps_u']:.2e}, {t['t_stop_s']:.3f} s")
    _write_csv(out / 'rows.csv', rows)
    timing = []
    if same_day:
        import experiments.clean_timing as ct
        timing = same_day_lilq(ct._kovasznay(False, 'cpu'), 'solve_time_total', package1, repeats)
        timing += same_day_lilq(_b10_runs(b10), 'solve_time_total', None, repeats)
        _write_csv(out / 'timing_lilq.csv', timing)
    _run_json(out, 'kovasznay', {'section': '4.2, step 2', 'K_max': sk.K_MAX, 'tol': sk.TOL, 'pin': 'corner',
                                 'error_grid': '301 x 401', 'runs': len(rows)})
    return rows, timing


def _b10_runs(sizes):
    """Wave 4's Clenshaw-Curtis CGL Kovasznay runs at N/P = 5 (B10), as timed runs."""
    import experiments.clean_timing as ct
    from experiments.b10_cgl_cc import config_for, run_name
    from problems.kovasznay import solve_kovasznay
    runs = []
    for N in sizes:
        config, _, _ = config_for('clenshaw_curtis', N, 5, 'paper')

        def once(config=config):
            r = solve_kovasznay(config, verbose=False, diagnostics=False)
            return {'solve_time_total': r['solve_time_total'], 'iterations': r['n_outer_iters']}
        runs.append(ct.TimedRun(run_name('clenshaw_curtis', N, 5, 'paper'), 'kovasznay', f'P{3 * N * N}', 'cpu',
                                config.max_iter, once))
    return runs


def _csv_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def work_precision_data(out_dir, package1, lilq_errors):
    """``{problem: {series: [(free coefficients, time s, error), ...]}}``. Bratu:
    error against the p = 48 reference (the LiL-Q runs': Stage 1's eps_ref).
    Kovasznay: E_u (LiL-Q and B10 from package1, which these timings rerun)."""
    root = Path(out_dir) / 'P2_3_classical'
    data = {'bratu': {}, 'kovasznay': {}}
    for r in _csv_rows(root / 'bratu' / 'rows.csv'):
        data['bratu'].setdefault(r['method'], []).append((int(r['free_coeff']), float(r['t_stop_s']), float(r['err_ref48'])))
    t_bratu = {t['run']: float(t['time_s']) for t in _csv_rows(root / 'bratu' / 'timing_lilq.csv')}
    for r in _csv_rows(lilq_errors):
        if r['benchmark'] == 'bratu' and r['method'] == 'LiL-Q':
            P = int(r['P'])
            data['bratu'].setdefault('LiL-Q (paper)', []).append((P, t_bratu[f'bratu_P{P}_cpu_paper'], float(r['eps_ref_final'])))
    for r in _csv_rows(root / 'kovasznay' / 'rows.csv'):
        data['kovasznay'].setdefault(r'SQ $P_N$-$P_{N-2}$', []).append((int(r['free_coeff']), float(r['t_stop_s']), float(r['eps_u'])))
    t_kov = {t['run']: float(t['time_s']) for t in _csv_rows(root / 'kovasznay' / 'timing_lilq.csv')}
    B = Path(package1) / 'B_instrumentation'
    for P in (75, 300, 675, 1200, 1875):
        run = f'kovasznay_P{P}_cpu_paper'
        if run in t_kov:
            e = json.loads((B / run / 'summary.json').read_text())['test_eps_u']
            data['kovasznay'].setdefault('LiL-Q (paper)', []).append((P, t_kov[run], float(e)))
    for r in _csv_rows(B / 'b10' / 'b10.csv'):
        if r['run'] in t_kov:
            data['kovasznay'].setdefault('LiL-Q, CGL + CC, N/P = 5', []).append((int(r['P']), t_kov[r['run']], float(r['eps_u'])))
    for d in data.values():
        for v in d.values():
            v.sort()
    return data


def work_precision_figure(out_dir, package1, lilq_errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = work_precision_data(out_dir, package1, lilq_errors)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    titles = {'bratu': 'Bratu (error against the p = 48 reference)', 'kovasznay': 'Kovasznay ($E_u$)'}
    for i, prob in enumerate(('bratu', 'kovasznay')):
        for label, pts in data[prob].items():
            P, t, e = zip(*pts)
            axes[i, 0].loglog(P, e, 'o-', ms=4, label=label)
            axes[i, 1].loglog(t, e, 'o-', ms=4, label=label)
        axes[i, 0].set_xlabel('free coefficients (unknowns)')
        axes[i, 1].set_xlabel('time to the stop (s, median of 3, CPU)')
        for ax in axes[i]:
            ax.set_ylabel('relative $L^2$ error')
            ax.set_title(titles[prob], fontsize=9)
            ax.grid(True, which='both', alpha=0.3)
        axes[i, 0].legend(fontsize=7)
    fig.tight_layout()
    out = Path(out_dir) / 'P2_3_classical' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'work_precision.pdf')
    fig.savefig(out / 'work_precision.png', dpi=150)
    plt.close(fig)
    (out / 'work_precision_data.json').write_text(json.dumps(data, indent=1))
    return out / 'work_precision.pdf'


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 2, item 1: the classical baselines (P2-3).")
    ap.add_argument('stage', choices=('reference', 'check-pilot', 'kovasznay-pilot', 'bratu', 'kovasznay', 'figures'))
    ap.add_argument('--out', required=True, help='package2_results (Stage 1) or the stage root (Stage 2)')
    ap.add_argument('--pilot-rows', default=None, help="the pilot's rows.json (check-pilot)")
    ap.add_argument('--package1', default=None, help="package1, for the same-day comparison and the figure")
    ap.add_argument('--reference-dir', default=None, help='the Bratu references (default: <out>/reference, made if absent)')
    ap.add_argument('--lilq-errors', default=None, help="Stage 1's scalar_reference_errors.csv (figures)")
    args = ap.parse_args(argv)
    if args.stage == 'bratu':
        rows, timing = bratu_stage(args.out, args.package1, args.reference_dir)
        for t in timing:
            print(f"  LiL-Q {t['run']}: {t['time_s']:.4f} s, package1 {t['package1_time_s']}, within 15%: {t['within_15pct']}")
        return
    if args.stage == 'kovasznay':
        rows, timing = kovasznay_stage(args.out, args.package1)
        for t in timing:
            print(f"  LiL-Q {t['run']}: {t['time_s']:.4f} s, package1 {t['package1_time_s']}, within 15%: {t['within_15pct']}")
        return
    if args.stage == 'figures':
        print(work_precision_figure(args.out, args.package1, args.lilq_errors))
        return
    if args.stage == 'reference':
        r = write_reference(args.out)
        print(f"Bratu reference p = {REF_P}; its distance to p = {CHECK_P}: {r['reference_error']:.3e}")
    elif args.stage == 'kovasznay-pilot':
        r = kovasznay_pilot(args.out)
        for row in r['rows']:
            print(f"p_d = {row['p_d']} ({row['pin']} pin): rank {row['rank']} of {row['n_unknowns']}, kappa "
                  f"{row['kappa']:.2e}, {row['k_stop']} Newton iterations, eps_u {row['eps_u']:.2e}")
        print(r['decision_input'])
    else:
        r = check_pilot(args.pilot_rows, args.out)
        for c in r['cases']:
            print(f"{c['method']:16s} p = {c['p']:2d}: {c['err_final']:.6e} against the pilot's {c['err_pilot']:.6e} "
                  f"(relative difference {c['rel_difference']:.1e})")
        s = r['sanity']
        print(f"sanity: LS-hard N = P = {s['P']} against SQ-CGL p = 12, max |difference| {s['max_abs_difference']:.1e}")
        print(f"C1 {'passed' if r['passed'] else 'FAILED'}")


if __name__ == '__main__':
    main()
