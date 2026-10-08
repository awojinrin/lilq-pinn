"""
Package 3, checks K1 and K2 (the advisor's instructions of 8 October 2026, Section 7)
=====================================================================================

**K1:**
- the Kronecker interior constants (``lilq.certified``) reproduce the
  advisor's expected values for every grid of items 1 and 2a, to the printed
  digits of Sections 3.3 and 4.2;
- for Beltrami B1 level 1 and BL P = 64, N/P = 10, the direct constants
  (SVD of the weighted evaluation matrix) agree with the Kronecker ones to
  1e-10;
- the block routine reproduces ``expected_constants.py``'s P2-16 Burgers-line
  values.

Given ``--expected-script``, every value is also compared with the script's
own functions at full precision. The script's lines use its default of 400
Gauss-Legendre points, so ours do here too.

**K2:** Clenshaw-Curtis exactness (``lilq.certified.cc_exactness``) on every
grid of items 1 and 2a:
- Beltrami: the interior (4D), the six faces and the initial slab (3D);
- BL, both cases: the interior, the initial line, and the rule each lateral
  line is cut from. A lateral line is the (n + 1)-point rule on [0, T]
  without t = 0, so its own weights sum to 1 - w_0, as P2-16; that sum is
  recorded.

Writes ``<out>/P3_checks/k1.json`` and ``k2.json``.

Usage::

    python experiments/p3_checks.py k1 --out <root> [--expected-script <path to expected_constants.py>]
    python experiments/p3_checks.py k2 --out <root>
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from lilq import certified as cf
from lilq.source_lock import current_commit

BELTRAMI_INTERVALS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))
# Section 3.3 and the script's printout: (N_vel, N_p, M) -> (c1^2, c2^2, c2/c1)
BELTRAMI_EXPECTED = {
    (4, 5, 13): ('0.905968', '1.007056', '1.0543'), (4, 5, 16): ('0.993686', '1.000000', '1.0032'),
    (5, 6, 15): ('0.858803', '1.010079', '1.0845'), (5, 6, 18): ('0.980958', '1.000972', '1.0102'),
    (6, 8, 16): ('0.674459', '1.016545', '1.2277'), (6, 8, 19): ('0.938615', '1.004885', '1.0347'),
}
BELTRAMI_SIZES = {'B1': (4, 5, (13, 16)), 'B2': (5, 6, (15, 18)), 'B3': (6, 8, (16, 19))}
# Section 4.2: (P, N/P, m) -> c2/c1, None where not computable (m = 3 at N/P = 20 only)
BL_EXPECTED = {
    (64, 5, 2): None, (64, 10, 2): '1.0212', (64, 20, 2): '1.0000', (64, 20, 3): '1.0255',
    (256, 5, 2): '3.6637', (256, 10, 2): '1.0111', (256, 20, 2): '1.0000', (256, 20, 3): '1.0164',
    (576, 5, 2): '1.5540', (576, 10, 2): '1.0088', (576, 20, 2): '1.0000', (576, 20, 3): '1.0141',
    (1024, 5, 2): '1.3977', (1024, 10, 2): '1.0077', (1024, 20, 2): '1.0000', (1024, 20, 3): '1.0130',
}
BL_M = {(64, 5): 17, (64, 10): 24, (64, 20): 34, (256, 5): 34, (256, 10): 48, (256, 20): 68,
        (576, 5): 51, (576, 10): 72, (576, 20): 102, (1024, 5): 68, (1024, 10): 96, (1024, 20): 136}
BL_T = {'viscous': 0.4, 'gravity': 0.175}
# the script's Burgers-line check: P -> (c1, c2); and the initial line at P = 625
BURGERS_LINES = {25: ('0.8638', '1.0051'), 100: ('0.9596', '1.0004'), 225: ('0.9826', '1.0001')}
BURGERS_INITIAL = ('1.0000', '1.0000')
P2_16 = {('bratu', 5, 15): '1.041395', ('bratu', 10, 30): '1.016546', ('bratu', 15, 44): '1.014894',
         ('burgers', 10, 22): '3.209607'}
DIRECT_TOL = 1e-10


def printed(value, text):
    """Whether ``value`` shows as ``text`` at text's number of decimals."""
    if text is None or value is None:
        return text is None and value is None
    return f'{value:.{len(text.split(".")[1])}f}' == text


def load_script(path):
    spec = importlib.util.spec_from_file_location('expected_constants', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def k1(script=None):
    rows, max_script_diff = [], 0.0

    def vs_script(ours, theirs):
        nonlocal max_script_diff
        if theirs is None or ours is None:
            return None
        d = abs(ours - theirs)
        max_script_diff = max(max_script_diff, d)
        return d
    for (bench, p, M), text in P2_16.items():         # the script's own check against P2-16
        k = cf.interior_constants_kronecker(M, 2 * p + 1, 2)
        rows.append({'what': 'P2-16 interior', 'case': f'{bench} p_d={p} M={M}', 'value': k['c2_over_c1'],
                     'expected': text, 'printed_match': printed(k['c2_over_c1'], text),
                     'script_diff': vs_script(k['c2_over_c1'], script.ratio(M, 2 * p + 1, 2) if script else None)})
    for size, (nv, npr, Ms) in BELTRAMI_SIZES.items():
        for level, M in enumerate(Ms, 1):
            k = cf.interior_constants_kronecker(M, 2 * nv + 1, 4)
            e = BELTRAMI_EXPECTED[(nv, npr, M)]
            sd = None
            if script:
                lo, hi = script.gram_1d_extremes(M, 2 * nv + 1)
                sd = max(vs_script(k['c1_squared'], lo ** 4), vs_script(k['c2_squared'], hi ** 4))
            rows.append({'what': 'item 1 interior', 'case': f'{size} level {level} M={M}',
                         'c1_squared': k['c1_squared'], 'c2_squared': k['c2_squared'], 'value': k['c2_over_c1'],
                         'expected': list(e), 'script_diff': sd,
                         'printed_match': printed(k['c1_squared'], e[0]) and printed(k['c2_squared'], e[1])
                         and printed(k['c2_over_c1'], e[2])})
    for (P, r, m), text in BL_EXPECTED.items():
        p = int(round(math.sqrt(P)))
        g = cf.bl_grid(P, r, BL_T['viscous'])
        k = cf.interior_constants_kronecker(g['M'], m * p + 1, 2)
        theirs = script.ratio(g['M'], m * p + 1, 2) if script else None
        rows.append({'what': 'item 2a interior', 'case': f'P={P} N/P={r} m={m} M={g["M"]}', 'value': k['c2_over_c1'],
                     'expected': text, 'M_expected': BL_M[(P, r)],
                     'printed_match': printed(k['c2_over_c1'], text) and g['M'] == BL_M[(P, r)],
                     'script_diff': vs_script(k['c2_over_c1'], theirs)})
    for P, (c1t, c2t) in BURGERS_LINES.items():         # lateral lines: (n_b + 1)-point rule on [0, 1] without t = 0
        p, nb = int(round(math.sqrt(P))), math.ceil(0.025 * 10 * P)
        t, w = cf.cgl_rule(0.0, 1.0, nb + 1)
        b = cf.block_constants(t[1:], w[1:], [(0.0, 1.0)], p, n_gl=400)
        sd = None
        if script:
            s1, s2 = script.block_constants((script.cgl(nb + 1)[1:] + 1) / 2, script.cc_weights(nb + 1)[1:], p)
            sd = max(vs_script(b['c1'], s1), vs_script(b['c2'], s2))
        rows.append({'what': 'block (P2-16 Burgers lateral line)', 'case': f'P={P}', 'c1': b['c1'], 'c2': b['c2'],
                     'expected': [c1t, c2t], 'dropped': b['dropped'], 'script_diff': sd,
                     'printed_match': printed(b['c1'], c1t) and printed(b['c2'], c2t)})
    ni = math.ceil(0.05 * 10 * 625)
    x, w = cf.cgl_rule(-1.0, 1.0, ni)
    datum = lambda z: -np.sin(np.pi * z[:, 0])  # noqa: E731
    b = cf.block_constants(x, w, [(-1.0, 1.0)], 25, data=datum, n_gl=400)
    sd = None
    if script:
        s1, s2 = script.block_constants(script.cgl(ni), script.cc_weights(ni), 25, data=lambda z: -np.sin(np.pi * z),
                                        a=-1.0, b=1.0)
        sd = max(vs_script(b['c1'], s1), vs_script(b['c2'], s2))
    rows.append({'what': 'block (P2-16 Burgers initial line, datum -sin(pi x))', 'case': 'P=625 N/P=10',
                 'c1': b['c1'], 'c2': b['c2'], 'expected': list(BURGERS_INITIAL), 'dropped': b['dropped'],
                 'script_diff': sd, 'printed_match': printed(b['c1'], BURGERS_INITIAL[0])
                 and printed(b['c2'], BURGERS_INITIAL[1])})
    # direct against Kronecker
    direct = []
    t0 = time.perf_counter()
    g = cf.beltrami_grid(13, BELTRAMI_INTERVALS)['blocks']['interior']
    d = cf.interior_constants_direct(g['points'], g['w'], BELTRAMI_INTERVALS, 9)
    k = cf.interior_constants_kronecker(13, 9, 4)
    direct.append({'case': 'Beltrami B1 level 1 (M = 13, q = 9, 28,561 x 6,561)', 'direct': d,
                   'kronecker': {key: k[key] for key in ('c1', 'c2', 'c2_over_c1')},
                   'seconds': time.perf_counter() - t0})
    for case, T in BL_T.items():
        g = cf.bl_grid(64, 10, T)
        b = g['blocks']['interior']
        d = cf.interior_constants_direct(b['points'], b['w'], ((0.0, 1.0), (0.0, T)), 17)
        k = cf.interior_constants_kronecker(g['M'], 17, 2)
        direct.append({'case': f'BL {case} P = 64, N/P = 10 (M = {g["M"]}, q = 17)', 'direct': d,
                       'kronecker': {key: k[key] for key in ('c1', 'c2', 'c2_over_c1')}})
    for dct in direct:
        dct['rel_diff'] = max(abs(dct['direct'][key] - dct['kronecker'][key]) / dct['kronecker'][key]
                              for key in ('c1', 'c2'))
        dct['passed'] = dct['rel_diff'] < DIRECT_TOL
    passed = all(r['printed_match'] for r in rows) and all(d['passed'] for d in direct)
    return {'check': 'K1', 'passed': passed, 'rows': rows, 'direct_vs_kronecker': direct,
            'direct_tolerance': DIRECT_TOL, 'script': None if script is None else str(script.__file__),
            'max_abs_diff_to_script': max_script_diff if script else None, 'commit': current_commit()}


def k2():
    grids = []
    for size, (nv, npr, Ms) in BELTRAMI_SIZES.items():
        for level, M in enumerate(Ms, 1):
            g = cf.beltrami_grid(M, BELTRAMI_INTERVALS)
            for name, b in g['blocks'].items():
                iv = [BELTRAMI_INTERVALS[i] for i in b['free']]
                r = cf.cc_exactness(b['points'][:, list(b['free'])], b['w'], iv, b['Ms'])
                grids.append({'item': 1, 'grid': f'{size} level {level} (M = {M})', 'block': name, **r})
    for case, T in BL_T.items():
        for (P, r_), M in BL_M.items():
            g = cf.bl_grid(P, r_, T)
            for name, b in g['blocks'].items():
                iv = [((0.0, 1.0), (0.0, T))[i] for i in b['free']]
                if name in ('left', 'right'):           # the rule the line is cut from
                    t, w = cf.cgl_rule(0.0, T, b['Ms'][0] + 1)
                    res = cf.cc_exactness(t[:, None], w, iv, (b['Ms'][0] + 1,))
                    res['block_weight_sum'] = float(b['w'].sum())
                    res['note'] = 'the (n + 1)-point rule; the block omits its t = 0 node'
                else:
                    res = cf.cc_exactness(b['points'][:, list(b['free'])], b['w'], iv, b['Ms'])
                grids.append({'item': '2a', 'grid': f'{case} P = {P}, N/P = {r_} (M = {M})', 'block': name, **res})
    worst = max(max(g['integral_degree_M_minus_1_err_over_L2_norm'], g['norm_degree_half_rel_err']) for g in grids)
    return {'check': 'K2', 'passed': all(g['passed'] for g in grids), 'n_rules': len(grids), 'worst_error': worst,
            'tolerance': cf.CC_EXACTNESS_TOL, 'rules': grids, 'commit': current_commit()}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 3, checks K1 and K2.')
    ap.add_argument('check', choices=('k1', 'k2'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected-script', help="the advisor's expected_constants.py (K1)")
    args = ap.parse_args(argv)
    out = Path(args.out) / 'P3_checks'
    out.mkdir(parents=True, exist_ok=True)
    result = k1(load_script(args.expected_script) if args.expected_script else None) if args.check == 'k1' else k2()
    (out / f'{args.check}.json').write_text(json.dumps(result, indent=2))
    if args.check == 'k1':
        for r in result['rows']:
            print(f"  {r['what']:48s} {r['case']:28s} {'ok' if r['printed_match'] else 'MISMATCH'}"
                  + (f"  (script diff {r['script_diff']:.1e})" if r.get('script_diff') is not None else ''))
        for d in result['direct_vs_kronecker']:
            print(f"  direct vs Kronecker, {d['case']}: {d['rel_diff']:.1e}")
    else:
        print(f"  {result['n_rules']} rules, worst error {result['worst_error']:.1e}")
    print(f"{args.check.upper()} {'passed' if result['passed'] else 'FAILED'}")
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
