"""
The manuscript's tables, regenerated from ``package1`` (Package 2, Part 1: the release, gate G1)
================================================================================================

Many tables of Section 6 and Appendix B of the manuscript were typed from
the package CSVs (medians over seeds, stopping-reason markers, ratios). This
module regenerates each of them from ``package1`` and checks the result
against ``main.tex``, so that every number has a reproducible path:

- ``build(package, out)`` writes one CSV per table, ``<label>.csv``, cell by
  cell in the manuscript's layout, at full precision, with the markers.
- ``check(tex, tables)`` reads the same tables out of ``main.tex`` and
  compares every checked cell: a number agrees when it lies within half a
  unit of the last digit printed; markers (``*``, ``dagger``, ``r/3``, ...)
  must be the same set; text cells must match after LaTeX is stripped.

Rows of published results (other papers' solvers) are not checked; they
carry no number of ours.

Usage::

    python experiments/release_tables.py --package <package1> --tex <main.tex> --out <G1_release>/tables
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


# ---------------------------------------------------------------- cells

@dataclass(frozen=True)
class Cell:
    """One table cell: the numbers it shows (in order), its markers, or its text.
    ``markers=None`` leaves the markers unchecked; ``values=None`` leaves the numbers unchecked."""
    values: tuple = None
    markers: frozenset = None
    text: str = None


def num(*values, markers=()):
    return Cell(values=tuple(values), markers=frozenset(markers))


def txt(text):
    return Cell(text=text)


SKIP = None          # a cell that is not checked (a heading repeated, a published value)


def render(cell):
    if cell is None:
        return ''
    if cell.text is not None:
        return cell.text
    s = ' '.join(repr(float(v)) if isinstance(v, float) else str(v) for v in (cell.values or ()))
    return s + (f" [{','.join(sorted(cell.markers))}]" if cell.markers else '')


# ---------------------------------------------------------------- reading main.tex

def norm(text):
    """LaTeX stripped to compare labels: no math, braces, spacing or font commands."""
    t = re.sub(r'\\(mbox|textsc|textbf|textit|emph|mathrm|text|texttt)\b', '', text)
    t = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}', '', t)
    t = re.sub(r'\\(,|;|!| )', '', t)
    t = t.replace('\\ ', '').replace('$', '').replace('{', '').replace('}', '').replace('~', '')
    return re.sub(r'\s+', '', t)


# Tables whose data rows sit between a \cmidrule under the header and the first \midrule.
BODY_BETWEEN = {'tab:darcy_training': ('\\cmidrule(lr){3-4}', '\\midrule')}


def _table_body(tex, label):
    i = tex.index('\\label{' + label + '}')
    if label in BODY_BETWEEN:
        start, end = BODY_BETWEEN[label]
        a = tex.index(start, i) + len(start)
        return tex[a:tex.index(end, a)]
    long_start = tex.rfind('\\begin{longtable}', 0, i)
    table_start = tex.rfind('\\begin{table', 0, i)
    if long_start > table_start:
        return tex[tex.index('\\endlastfoot', i) + len('\\endlastfoot'):tex.index('\\end{longtable}', i)]
    body = tex[i:tex.index('\\end{table', i)]
    return body[body.index('\\midrule') + len('\\midrule'):body.rindex('\\bottomrule')]


def _expand_multicolumn(row):
    out = []
    for cell in row.split('&'):
        m = re.match(r'\s*\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$', cell, re.S)
        if m:
            out += [m.group(2)] + [''] * (int(m.group(1)) - 1)
        else:
            out.append(cell)
    return out


def tex_rows(tex, label):
    """The data rows of a table of ``main.tex``, each a list of cell strings."""
    rows = []
    for raw in _table_body(tex, label).split('\\\\'):
        r = re.sub(r'\\(addlinespace|cmidrule)(\[[^\]]*\])?(\([^)]*\))?(\{[^}]*\})?', '', raw)
        r = re.sub(r'^\s*\[[^\]]*\]', '', r.replace('\\midrule', ''))
        if '&' in r:
            rows.append(_expand_multicolumn(r))
    return rows


SCI = re.compile(r'(\d+(?:\.\d+)?)\s*\\times\s*10\^\{(-?\d+)\}')
MARK = re.compile(r'\^\{([^{}]*)\}|\^(\\[A-Za-z]+|\*)')
PLAIN = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)')


def parse_cell(cell):
    """``(numbers, markers)`` of a cell; each number is ``(value, tolerance)``,
    the tolerance half a unit of the last digit printed."""
    c = re.sub(r'\\phantom\{[^}]*\}', '', cell)
    c = c.replace('{,}', ',').replace('{\\times}', '\\times').replace('{-}', '-').replace('{=}', '=')
    c = c.replace('$', '').replace('{\\sim}', '').replace('\\sim', '')
    c = re.sub(r'(?<=\d),(?=\d{3})', '', c)
    found = []
    for m in SCI.finditer(c):
        mant, exp = m.group(1), int(m.group(2))
        dec = len(mant.split('.')[1]) if '.' in mant else 0
        found.append((m.start(), float(mant) * 10.0 ** exp, 0.5 * 10.0 ** (exp - dec)))
    c = SCI.sub(lambda m: ' ' * len(m.group(0)), c)
    markers = set()
    for m in MARK.finditer(c):
        for tok in (m.group(1) or m.group(2)).split(','):
            tok = tok.replace('\\,', '').replace('\\', '').strip()
            if tok:
                markers.add({'ast': '*'}.get(tok, tok))
    c = MARK.sub(lambda m: ' ' * len(m.group(0)), c).replace('--', ' ')
    for m in PLAIN.finditer(c):
        v = m.group(1)
        dec = len(v.split('.')[1]) if '.' in v else 0
        found.append((m.start(), float(v), 0.5 * 10.0 ** (-dec)))
    return [(v, tol) for _, v, tol in sorted(found)], markers


def compare_cell(expected, tex_cell):
    """``None`` when the cell agrees, else a short reason."""
    if expected is None:
        return None
    if expected.text is not None:
        return None if norm(tex_cell) == norm(expected.text) else f"text {norm(tex_cell)!r} != {norm(expected.text)!r}"
    numbers, markers = parse_cell(tex_cell)
    if expected.values is not None:
        if len(numbers) != len(expected.values):
            return f"{len(numbers)} numbers in the manuscript, {len(expected.values)} regenerated"
        for (v, tol), e in zip(numbers, expected.values):
            if e is None or isinstance(e, float) and math.isnan(e):
                return f"no regenerated value for {v}"
            if abs(v - e) > tol * (1 + 1e-9) + 1e-300:
                return f"{v:g} (manuscript) vs {e:.6g} (regenerated)"
    if expected.markers is not None and set(expected.markers) != markers:
        return f"markers {sorted(markers)} != {sorted(expected.markers)}"
    return None


# ---------------------------------------------------------------- the tables

TABLES = {}


def table(label, key=None):
    """Register a table builder ``f(package) -> (columns, rows)``. ``key``: the
    number of leading cells that identify a row (rows are matched by key,
    blank key cells repeat the row above); ``None`` matches rows in order."""
    def register(f):
        TABLES[label] = (f, key)
        return f
    return register


def _csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _median(xs):
    return statistics.median(xs)


TARGETS = {}


def targets(bench):
    """The four-method target MSE by P (the paper's TARGET_LOSSES, keyed by modes per direction)."""
    if bench not in TARGETS:
        if bench == 'bratu':
            from experiments.run_bratu import TARGET_LOSSES as t
        elif bench == 'burgers':
            from experiments.run_burgers import TARGET_LOSSES as t
        elif bench == 'bl':
            from experiments.run_bl import TARGET_LOSSES as t
        else:
            from experiments.run_bl import GRAVITY_TARGET_LOSSES as t
        TARGETS[bench] = {n * n: v for n, v in t.items()}
    return TARGETS[bench]


FOUR = ('NiL-N', 'NiL-Q', 'LiL-N')
DEVICE = {'cuda': 'GPU', 'cpu': 'CPU'}
STOP = {'target': 'T', 'iteration_cap': 'B', 'optimizer_stall': 'S'}
BENCH_LABEL = {'bratu': 'Bratu', 'burgers': 'Burgers', 'bl': 'BL ($N_g{=}0$)', 'bl_gravity': 'BL ($N_g{=}{-}5$)'}


def _four_method_runs(package):
    b = Path(package) / 'B_instrumentation'
    runs = [r for r in _csv(b / 'four_method_tables.csv') if not r['variant']]
    return runs, _csv(b / 'four_method_lilq.csv')


def _markers(group):
    """The main tables' markers: r/3 when 1 or 2 of 3 seeds reached the target;
    ``*`` when most runs ended at the iteration budget; ``dagger`` at a stall."""
    reasons = [r['stopping_reason'] for r in group]
    n, hit = len(reasons), reasons.count('target')
    marks = set()
    if reasons.count('iteration_cap') * 2 > n:
        marks.add('*')
    if reasons.count('optimizer_stall') * 2 > n:
        marks.add('dagger')
    if n == 3 and 0 < hit < 3:
        marks.add(f'{hit}/3')
    return marks


def four_method_table(package, bench):
    runs, lilq = _four_method_runs(package)
    rows = []
    for P in sorted({int(r['P']) for r in runs if r['benchmark'] == bench}):
        its, times = [], []
        for m in FOUR:
            g = [r for r in runs if r['benchmark'] == bench and int(r['P']) == P and r['method'] == m
                 and r['device'] == 'cuda']
            its.append(num(_median([int(r['total_iterations']) for r in g]), markers=_markers(g)))
            times.append(num(_median([float(r['training_time_s']) for r in g])))
        [q] = [r for r in lilq if r['benchmark'] == bench and int(r['P']) == P]
        its.append(num(int(q['total_iterations'])))
        times.append(num(float(q['training_time_s'])))
        rows.append([num(P), num(targets(bench)[P])] + its + times)
    cols = ['P', 'target_mse'] + [f'iterations_{m}' for m in FOUR + ('LiL-Q',)] + [f'time_s_{m}' for m in FOUR + ('LiL-Q',)]
    return cols, rows


table('tab:bratu_results')(lambda pkg: four_method_table(pkg, 'bratu'))
table('tab:burgers_iterations')(lambda pkg: four_method_table(pkg, 'burgers'))
table('tab:bl_case1_iterations')(lambda pkg: four_method_table(pkg, 'bl'))
table('tab:bl_case2_iterations')(lambda pkg: four_method_table(pkg, 'bl_gravity'))


@table('tab:four_method_runs', key=5)
def four_method_runs(package):
    runs, lilq = _four_method_runs(package)
    rows = []
    for r in runs:
        rows.append([txt(BENCH_LABEL[r['benchmark']]), num(int(r['P'])), txt(r['method']), txt(DEVICE[r['device']]),
                     txt(r['seed'] if r['method'] != 'LiL-N' else '--'),
                     num(int(r['total_iterations'])), num(int(r['iterations_cap'])), num(int(r['total_line_searches'])),
                     num(float(r['training_time_s'])), num(float(r['final_loss'])),
                     num(targets(r['benchmark'])[int(r['P'])]), txt(STOP[r['stopping_reason']])])
    for q in lilq:
        rows.append([txt(BENCH_LABEL[q['benchmark']]), num(int(q['P'])), txt('LiL-Q'), txt(DEVICE[q['device']]), txt('--'),
                     num(int(q['total_iterations'])), num(int(q['iterations_cap'])), num(),
                     num(float(q['training_time_s'])), num(float(q['final_loss'])),
                     num(targets(q['benchmark'])[int(q['P'])]), txt(STOP[q['stopping_reason']])])
    cols = ['benchmark', 'P', 'method', 'device', 'seed', 'iterations', 'budget', 'evaluations', 'time_s',
            'final_mse', 'target_mse', 'stop']
    return cols, rows


@table('tab:stall_controls')
def stall_controls(package):
    c = _csv(Path(package) / 'B_instrumentation' / 'four_method_controls.csv')

    def row(label, device, group):
        ratios = sorted(float(r['final_loss']) / float(r['original_final_loss']) for r in group)
        reasons = [r['stopping_reason'] for r in group]
        span = (ratios[0],) if ratios[0] == ratios[-1] else (ratios[0], ratios[-1])
        Ps = sorted({int(r['P']) for r in group})
        return [txt(label), txt(device) if device else SKIP, num(*Ps) if device else SKIP, num(len(group)),
                num(reasons.count('target')), num(reasons.count('iteration_cap')), num(_median(ratios)), num(*span)]

    rows = []
    for bench in ('bratu', 'burgers', 'bl', 'bl_gravity'):
        for dev in ('cuda', 'cpu'):
            g = [r for r in c if r['benchmark'] == bench and r['device'] == dev]
            label = BENCH_LABEL[bench].replace('$', '')
            rows.append(row(label, DEVICE[dev], g))
    rows.append(row('All', '', c))
    return ['benchmark', 'device', 'P', 'controls', 'target', 'budget', 'ratio_median', 'ratio_range'], rows


# ---------------------------------------------------------------- LiL-Q tables of Section 6

def _b(package, *parts):
    return Path(package, 'B_instrumentation', *parts)


def _json(path):
    return json.loads(Path(path).read_text())


def _clean(package):
    """Option B's clean times: ``{(run, quantity): seconds}``."""
    return {(r['run'], r['quantity']): float(r['clean_time_s'])
            for r in _csv(_b(package, 'clean_timing', 'clean_timing.csv')) if r['clean_time_s']}


def _log(path):
    return _csv(path)


def _f(x):
    return float(x) if x not in ('', None) else None


BASIS_ROWS = ('sin_cheb', 'sin_fourier', 'augsin_cheb', 'fourier_cheb', 'fourier_fourier', 'cheb_cheb',
              'elm_default', 'elm', 'cos_cheb', 'sin_sin')


@table('tab:basis_configs')
def basis_configs(package):
    rows = {r['basis_key']: r for r in _csv(_b(package, 'basis_study', 'table3_basis_study.csv'))}
    out = [[SKIP, SKIP, SKIP, num(float(rows[k]['final_R_h_squared'])), num(float(rows[k]['kappa'])),
            num(int(rows[k]['num_rank_svd']))] for k in BASIS_ROWS]
    return ['configuration', 'spatial', 'temporal', 'mse_k50', 'kappa', 'rank_svd'], out


@table('tab:bl_reference_errors')
def bl_reference_errors(package):
    """The LiL-Q saturation's error against the finite-difference reference
    (``eps_u`` of the logs): at the paper pass's stop, its minimum over the
    K_max = 60 pass and where, and at k = 60."""
    rows = []
    lilq = {(r['benchmark'], int(r['P'])): r for r in _four_method_runs(package)[1]}
    for case, label in (('bl', '$N_g = 0$'), ('bl_gravity', '$N_g = -5$')):
        for P in (64, 256, 576, 1024):
            stop = int(lilq[(case, P)]['total_iterations'])
            log = [(int(r['k']), float(r['eps_u'])) for r in
                   _log(_b(package, 'residual_band_figures', 'kmax', f'{case}_P{P}_iterations.csv')) if r['eps_u']]
            e = dict(log)
            k_min, e_min = min(log, key=lambda t: t[1])
            rows.append([txt(label) if P == 64 else SKIP, num(P), num(stop), num(e[stop]), num(e_min, k_min),
                         num(e[max(e)])])
    return ['case', 'P', 'stop', 'E_S_stop', 'min_E_S_and_k', 'E_S_k60'], rows


ELASTICITY_P = (50, 200, 450, 800, 1250)


@table('tab:elasticity_results')
def elasticity_results(package):
    clean, rows = _clean(package), []
    for P in ELASTICITY_P:
        run = f'elasticity_P{P}_cpu_paper'
        s = _json(_b(package, run, 'summary.json'))
        rows.append([num(int(round((P / 2) ** 0.5))), num(P), num(clean[(run, 'solve_time_qr')])]
                    + [num(s[f'rel_l2_{f}']) for f in ('ux', 'uy', 'sxx', 'syy', 'sxy')])
    return ['p_d', 'P', 'qr_time_s', 'E_ux', 'E_uy', 'E_sxx', 'E_syy', 'E_sxy'], rows


@table('tab:elasticity_comparison')
def elasticity_comparison(package):
    clean, rows = _clean(package), [None]                            # SciANN: published
    for P in (50, 800, 1250):
        run = f'elasticity_P{P}_cpu_paper'
        s = _json(_b(package, run, 'summary.json'))
        rows.append([SKIP, num(P), num(1), num(clean[(run, 'time_lil_s')]), num(s['rel_l2_ux']), num(s['rel_l2_uy'])])
    return ['method', 'P', 'qr_solves', 'total_time_s', 'E_ux', 'E_uy'], rows


KOVASZNAY_P = (75, 300, 675, 1200, 1875)


@table('tab:kovasznay_results')
def kovasznay_results(package):
    clean, rows = _clean(package), []
    for P in KOVASZNAY_P:
        run = f'kovasznay_P{P}_cpu_paper'
        s = _json(_b(package, run, 'summary.json'))
        rows.append([num(int(round((P / 3) ** 0.5))), num(P), num(int(s['iterations'])),
                     num(clean[(run, 'solve_time_total')]),
                     num(s['test_eps_u']), num(s['test_eps_v']), num(s['test_eps_p_meanfree'])])
    return ['p_d', 'P', 'iterations', 'runtime_cpu_s', 'E_u', 'E_v', 'E_p_meanfree'], rows


@table('tab:kovasznay_comparison')
def kovasznay_comparison(package):
    cmp = {(r['family'], r['device'], r['config']): r
           for r in _csv(Path(package, 'A_calibration', 'results', 'kovasznay_comparison.csv'))}
    rows = [None] * 9                                                # published rows
    for fam, cfg in (('F1', 'F1_18'), ('F2', 'F2_05')):
        r = cmp[(fam, 'gpu', cfg)]
        n_theta = _json(Path(package, 'A_calibration', 'full', f'{cfg}_s0', 'run.json'))['n_theta']
        # training ("~4,600 LM steps": the median is 4,608) and runtime (the 60-minute budget) are not checked
        rows.append([SKIP, num(n_theta), SKIP, SKIP, num(float(r['eps_u_median'])), num(float(r['eps_v_median'])),
                     num(float(r['eps_p_meanfree_median']))])
    for P in (675, 1200, 1875):
        c, g = cmp[('LiL-Q', 'cpu', f'P={P}')], cmp[('LiL-Q', 'gpu', f'P={P}')]
        s = _json(_b(package, f'kovasznay_P{P}_cpu_paper', 'summary.json'))
        rows.append([SKIP, num(P), num(int(s['iterations'])), num(float(c['time_s']), float(g['time_s'])),
                     num(float(c['eps_u_median'])), num(float(c['eps_v_median'])),
                     num(float(c['eps_p_meanfree_median']))])
    return ['method', 'P', 'training', 'runtime_cpu_gpu_s', 'E_u', 'E_v', 'E_p_meanfree'], rows


def _beltrami(package):
    """The eight-pin run (Section 6.7): per-snapshot errors from its report, the
    whole-domain errors from its log's last iterate."""
    rep = _json(_b(package, 'beltrami_pinned', 'report.json'))
    last = _log(_b(package, 'beltrami_pinned', 'iterations.csv'))[-1]
    return rep, last


@table('tab:beltrami_results')
def beltrami_results(package):
    """Snapshots and the whole-domain u, v, p errors from the eight-pin run.
    Its log has no ``eps_w`` column, so the whole-domain E_w is the one-pin
    paper pass's ``rel_l2_w`` (``beltrami_P7984_cpu_paper/summary.json``):
    the two runs' whole-domain velocity errors agree to round-off, which is
    asserted for u and v."""
    rep, last = _beltrami(package)
    one_pin = _json(_b(package, 'beltrami_P7984_cpu_paper', 'summary.json'))
    for f in ('u', 'v'):
        assert abs(float(last[f'eps_{f}']) / one_pin[f'rel_l2_{f}'] - 1) < 1e-9, f
    whole = {'u': float(last['eps_u']), 'v': float(last['eps_v']), 'w': one_pin['rel_l2_w'], 'p': float(last['eps_p'])}
    snaps = rep['snapshots']
    rows = [[SKIP] + [num(100 * s[f]) for s in snaps] + [num(100 * whole[f])] for f in ('u', 'v', 'w', 'p')]
    return ['field'] + [f"t={s['t']}" for s in snaps] + ['entire_domain'], rows


@table('tab:beltrami_comparison')
def beltrami_comparison(package):
    rep, _ = _beltrami(package)
    t1 = [s for s in rep['snapshots'] if s['t'] == 1.0][0]
    rows = [None] * 5 + [[SKIP, num(rep['P_total']), num(rep['n_outer_iters'])]
                         + [num(100 * t1[f]) for f in ('u', 'v', 'w', 'p')]]
    return ['method', 'P', 'qr_solves', 'E_u_pct', 'E_v_pct', 'E_w_pct', 'E_p_pct'], rows


DARCY_FIELDS = ('S1', 'S2', 'S3', 'SPE10')


@table('tab:darcy_training', key=1)
def darcy_training(package):
    """Table 14, keyed by its left-hand label. The network size, precision and
    iteration budget come from the saved float64 networks; the LiL modes and
    coefficients from the paper pass; the activation and the loss weights from
    ``problems/darcy.py`` (``DarcyPINN``), the code the runs used. The output
    and optimizer entries are descriptions, not numbers, and are not checked."""
    import torch
    rows = _csv(_b(package, 'darcy_fv_comparison.csv'))
    [n_nil] = {int(r['n_params']) for r in rows if r['method'] == 'NiL'}
    [n_lil] = {int(r['n_params']) for r in rows if r['method'] == 'LiL'}
    nets = [torch.load(p, map_location='cpu', weights_only=False)
            for p in sorted(_b(package, 'darcy_fv').glob('*/models/NiL_*_float64/network.pt'))]
    meta = {(n['networks']['hidden_dim'], n['networks']['num_layers'], n['networks']['dtype'], n['max_epochs'])
            for n in nets}
    assert len(nets) == 12 and len(meta) == 1, meta
    [(width, depth, dtype, epochs)] = meta
    order = _json(_b(package, 'darcy_S1_cpu_paper', 'summary.json'))['order']
    src = (_proj / 'problems' / 'darcy.py').read_text(encoding='utf-8')
    act = re.search(r'act = nn\.(\w+)\(\)', src).group(1)
    w_pde, w_bc = map(float, re.search(r'W_PDE, W_BC = ([\d.]+), ([\d.]+)', src).groups())
    out = [[txt('Hidden layers'), num(depth), SKIP, num(order)],
           [txt('Neurons per layer'), num(width), SKIP, num(n_lil)],
           [txt('Activation'), txt(act), SKIP, SKIP],
           [txt('Output ($h^*$)'), SKIP, SKIP, SKIP],
           [txt('Total parameters'), num(n_nil), SKIP, SKIP],
           [txt('Optimizer'), SKIP, SKIP, SKIP],
           [txt('Iterations'), num(epochs), SKIP, SKIP],
           [txt('Loss weights'), num(w_pde, w_bc), SKIP, SKIP],
           [txt('Precision'), txt('\\texttt{' + dtype.replace('torch.', '') + '}'), SKIP, SKIP]]
    return ['nil_setting', 'nil_value', 'lil_setting', 'lil_value'], out


@table('tab:darcy_results', key=2)
def darcy_results(package):
    rows = _csv(_b(package, 'darcy_fv_comparison.csv'))
    clean, out = _clean(package), []
    for field in DARCY_FIELDS:
        nil = [r for r in rows if r['field'] == field and r['method'] == 'NiL' and r['dtype'] == 'float64']
        [lil] = [r for r in rows if r['field'] == field and r['method'] == 'LiL']
        s = _json(_b(package, f'darcy_{field}_cpu_paper', 'summary.json'))
        out.append([txt(field), txt('NiL'), num(_median([float(r['delta_fv']) for r in nil])),
                    num(_median([float(r['max_abs_err_psi']) for r in nil])),
                    num(_median([float(r['time_s']) for r in nil]), markers={'a'})])
        out.append([txt(field), txt('LiL'), num(float(lil['delta_fv'])), num(float(lil['max_abs_err_psi'])),
                    num(clean[(f'darcy_{field}_cpu_paper', 'total_time')])])
        out.append([txt(field), txt('FVM'), num(), num(), num(s['fvm_time'])])
    return ['case', 'method', 'delta_fv', 'max_abs_diff_psi', 'runtime_s'], out


def _kappas(package, run):
    """``(final kappa, max_k kappa / min_k kappa over k >= 1, rank at the last logged kappa)``."""
    log = [r for r in _log(_b(package, run, 'iterations.csv')) if r['kappa']]
    ks = [float(r['kappa']) for r in log if int(r['k']) >= 1] or [float(log[-1]['kappa'])]
    rank = log[-1]['num_rank_svd'] or log[-1]['num_rank_gelsy']
    return float(log[-1]['kappa']), max(ks) / min(ks), int(rank)


@table('tab:condition_numbers')
def condition_numbers(package):
    rows = []
    for bench, Ps in (('bratu', (25, 100, 225)), ('burgers', (25, 100, 225, 400, 625)),
                      ('bl', (64, 256, 576, 1024)), ('bl_gravity', (64, 256, 576, 1024))):
        ks = [_kappas(package, f'{bench}_P{P}_cpu_paper') for P in Ps]
        Pmax = Ps[-1]
        rank = ks[-1][2]
        rank_cell = txt('full (=P)') if rank == Pmax else num(rank, Pmax - rank, markers={'*'} if bench == 'bl' else ())
        rows.append([SKIP, num(Ps[0], Pmax), num(ks[0][0]), num(ks[-1][0]), num(max(k[1] for k in ks)), rank_cell])
    ks = [_kappas(package, f'elasticity_P{P}_cpu_paper') for P in ELASTICITY_P]
    rows.append([SKIP, num(ELASTICITY_P[0], ELASTICITY_P[-1]), num(ks[0][0]), num(ks[-1][0]), SKIP,
                 txt('full (=P)') if ks[-1][2] == ELASTICITY_P[-1] else num(ks[-1][2])])
    ks = [_kappas(package, f'kovasznay_P{P}_cpu_paper') for P in KOVASZNAY_P]
    rows.append([SKIP, num(KOVASZNAY_P[0], KOVASZNAY_P[-1]), num(ks[0][0]), num(ks[-1][0]),
                 num(max(k[1] for k in ks)), txt('full (=P)') if ks[-1][2] == KOVASZNAY_P[-1] else num(ks[-1][2])])
    ks = [_kappas(package, f'darcy_{f}_cpu_paper') for f in DARCY_FIELDS]
    rows.append([SKIP, num(3169), num(min(k[0] for k in ks), max(k[0] for k in ks)), SKIP, SKIP,
                 txt('full (=P)') if all(k[2] == 3169 for k in ks) else SKIP])
    rep, _ = _beltrami(package)
    kb = _kappas(package, 'beltrami_pinned')
    rows.append([SKIP, num(rep['P_total']), SKIP, num(rep['kappa']), num(kb[1]),
                 txt('full (=P)') if rep['full_column_rank'] else SKIP])
    return ['problem', 'P_range', 'kappa_min_P', 'kappa_max_P', 'iteration_stability', 'rank_max_P'], rows


STOPPING_RUNS = (('bratu', (25, 100, 225)), ('burgers', (25, 100, 225, 400, 625)), ('bl', (64, 256, 576, 1024)),
                 ('bl_gravity', (64, 256, 576, 1024)), ('kovasznay', KOVASZNAY_P), ('beltrami', (7984,)))


def _stopping(package, n_s, tau_chi, tau_r):
    from experiments.stopping_rule_table import table16
    rows, summary = table16(_b(package), [n_s], tau_chi, tau_r)
    return {r['run']: r for r in rows if int(r['n_s']) == n_s}, summary[n_s] if n_s in summary else summary[str(n_s)]


@table('tab:stopping_demo')
def stopping_demo(package):
    by_run, _ = _stopping(package, 2, 0.1, 0.01)
    rows = []
    for bench, Ps in STOPPING_RUNS:
        for P in Ps:
            r = by_run[f'{bench}_P{P}']
            rows.append([SKIP, num(P), num(int(r['paper_stop'])), num(int(r['returned'])), txt(r['class']),
                         num(float(r['R_over_min'])), num(float(r['E_over_paper_stop']))])
    return ['problem', 'P', 'stop', 'returned', 'type', 'R_ratio', 'error_ratio'], rows


STOPPING_VARIANTS = ((0.1, 0.1, 1), (0.1, 0.1, 2), (0.05, 0.05, 1), (0.2, 0.2, 1), (0.1, 0.01, 1), (0.1, 0.01, 2))


@table('tab:stopping_variants')
def stopping_variants(package):
    rows = []
    for tau_chi, tau_r, n_s in STOPPING_VARIANTS:
        by_run, s = _stopping(package, n_s, tau_chi, tau_r)
        g64, g256 = by_run['bl_gravity_P64'], by_run['bl_gravity_P256']
        rows.append([SKIP, num(s['early']), num(s['errstop']), num(int(g64['returned'])), num(float(g64['R_over_min'])),
                     num(int(g256['returned'])), num(float(g256['R_over_min']))])
    return ['variant', 'early_stops', 'error_stops', 'k_P64', 'R_ratio_P64', 'k_P256', 'R_ratio_P256'], rows


# ---------------------------------------------------------------- build and check

def build(package, out, labels=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    built = {}
    for label, (f, _key) in TABLES.items():
        if labels and label not in labels:
            continue
        cols, rows = f(package)
        built[label] = rows
        with open(out / f"{label.replace(':', '_')}.csv", 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                if r is not None:
                    w.writerow([render(c) for c in r])
    return built


def check(tex, built):
    """One record per checked cell: ``(label, row, column, manuscript, regenerated, problem)``."""
    records = []
    for label, rows in built.items():
        key = TABLES[label][1]
        trows = tex_rows(tex, label)
        if key is None:
            pairs = list(zip(trows, rows))
            if len(trows) != len(rows):
                records.append((label, '', '', f'{len(trows)} rows', f'{len(rows)} rows', 'row count differs'))
        else:
            index = {tuple(norm(c.text) if c and c.text is not None else render(c) for c in r[:key]): r
                     for r in rows if r is not None}
            pairs, last = [], [''] * key
            for t in trows:
                k = [norm(c) or last[i] for i, c in enumerate(t[:key])]
                last = k
                k = tuple(norm(c) if isinstance(c, str) else c for c in k)
                k = tuple(x if not re.fullmatch(r'[\d,]+', x) else x.replace(',', '') for x in k)
                pairs.append((t, index.get(k)))
                if k not in index:
                    records.append((label, '|'.join(k), '', 'row', '', 'no regenerated row'))
            for r in rows:
                if r is not None:
                    k = tuple(norm(c.text) if c and c.text is not None else render(c) for c in r[:key])
                    if not any(p[1] is r for p in pairs):
                        records.append((label, '|'.join(k), '', '', 'row', 'row missing from the manuscript'))
        for i, (t, r) in enumerate(pairs):
            if r is None:
                continue
            if len(t) != len(r):
                records.append((label, i, '', f'{len(t)} cells', f'{len(r)} cells', 'cell count differs'))
                continue
            for j, (tc, rc) in enumerate(zip(t, r)):
                if rc is None or (key is not None and j < key and not norm(tc)):
                    continue                     # a blank key cell repeats the row above (\multirow)
                problem = compare_cell(rc, tc)
                records.append((label, i, j, re.sub(r'\s+', ' ', tc).strip(), render(rc), problem or ''))
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate the manuscript's tables from package1 and check them.")
    ap.add_argument('--package', required=True)
    ap.add_argument('--tex', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tables', nargs='*', default=None)
    args = ap.parse_args(argv)
    built = build(args.package, args.out, args.tables)
    tex = Path(args.tex).read_text(encoding='utf-8')
    records = check(tex, built)
    out = Path(args.out)
    with open(out / 'release_check.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['table', 'row', 'column', 'manuscript', 'regenerated', 'problem'])
        w.writerows(records)
    summary = {}
    for label in built:
        mine = [r for r in records if r[0] == label]
        summary[label] = {'checked': len(mine), 'problems': sum(1 for r in mine if r[5])}
    (out / 'release_check.json').write_text(json.dumps(summary, indent=2))
    for label, s in summary.items():
        print(f"{label:28s} checked {s['checked']:4d}  problems {s['problems']}")
    bad = [r for r in records if r[5]]
    for r in bad[:40]:
        print('  ', r)
    print('all agree' if not bad else f'{len(bad)} problems')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
