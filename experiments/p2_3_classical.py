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

Usage::

    python experiments/p2_3_classical.py reference --out <package2_results>
    python experiments/p2_3_classical.py check-pilot --pilot-rows rows.json --out <package2_results>
    python experiments/p2_3_classical.py kovasznay-pilot --out <package2_results>
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 2, item 1: Bratu references and the port check.")
    ap.add_argument('stage', choices=('reference', 'check-pilot', 'kovasznay-pilot'))
    ap.add_argument('--out', required=True, help='package2_results')
    ap.add_argument('--pilot-rows', default=None, help="the pilot's rows.json (check-pilot)")
    args = ap.parse_args(argv)
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
