"""
The manuscript's LiL-Q table entries, for check B1 (Section 3.3)
=================================================================

Transcribed from the revised manuscript (``main.pdf``; its table
numbers, which run one ahead of the computational package's for some
tables). Each entry names the rerun it is compared against and how:

- ``iterations``, ``rank``: must be equal.
- ``error``, ``residual``: within a factor of 2.
- ``condition``: within a factor of 10.
- ``time``: reported side by side, no tolerance (B1 sets none; the
  hardware and load differ from the paper's runs).

Iterations are LiL-Q outer iterations (number of linear solves).
Beltrami errors are fractions (the table's percentages / 100).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class PaperValue:
    benchmark: str
    config: str       # the Component B config label, e.g. 'P225' or 'S3'
    quantity: str
    value: float
    kind: str         # iterations | rank | error | residual | condition | time
    source: str
    rerun: Callable[[Dict], Optional[float]]  # summary.json (paper pass, CPU) -> value
    note: str = ''


def _key(name):
    return lambda s: s.get(name)


def _snapshot(field, t):
    return lambda s: next(x[field] for x in s['snapshots'] if abs(x['t'] - t) < 1e-12)


_ITERS = _key('iterations')
_TIME = _key('t_cum_s')
_KAPPA = _key('kappa_final')


def _rank(s):
    return s['num_rank_svd_final'] if s.get('num_rank_svd_final') is not None else s.get('num_rank_gelsy_final')


def _four_method(benchmark, source, rows):
    out = []
    for P, iters, runtime in rows:
        out.append(PaperValue(benchmark, f'P{P}', 'LiL-Q iterations', iters, 'iterations', source, _ITERS))
        out.append(PaperValue(benchmark, f'P{P}', 'LiL-Q runtime (s)', runtime, 'time', source, _TIME))
    return out


PAPER_VALUES = (
    _four_method('bratu', 'Table 2', [(25, 2, 0.0004), (100, 3, 0.032), (225, 3, 0.14)])
    + _four_method('burgers', 'Table 3', [(25, 2, 0.0005), (100, 3, 0.036), (225, 3, 0.12),
                                          (400, 4, 0.51), (625, 4, 1.26)])
    + _four_method('bl', 'Table 5', [(64, 11, 0.052), (256, 4, 0.26), (576, 4, 1.33), (1024, 4, 4.16)])
    + _four_method('bl_gravity', 'Table 6', [(64, 9, 0.05), (256, 7, 0.49), (576, 10, 3.16), (1024, 8, 9.18)])
)

# Table 7: linear elasticity (P = 2 pd^2), QR solve time and relative L2 errors.
for P, qr, eux, euy, sxx, syy, sxy in [
        (50, 0.007, 6.5e-16, 1.0e-15, 2.4e-16, 3.7e-16, 8.6e-16),
        (200, 0.029, 7.1e-16, 9.3e-16, 4.3e-16, 6.7e-16, 9.0e-16),
        (450, 0.106, 1.2e-15, 9.1e-16, 1.1e-15, 1.1e-15, 1.4e-15),
        (800, 0.283, 1.6e-15, 1.6e-15, 1.4e-15, 1.0e-15, 1.2e-15),
        (1250, 1.03, 3.0e-15, 2.2e-15, 2.6e-15, 2.2e-15, 2.3e-15)]:
    PAPER_VALUES += (
        PaperValue('elasticity', f'P{P}', 'QR solve time (s)', qr, 'time', 'Table 7', _key('solve_time_qr')),
        PaperValue('elasticity', f'P{P}', 'E_ux', eux, 'error', 'Table 7', _key('rel_l2_ux'),
                   'machine-precision level: a factor-2 miss is round-off, not a reproduction failure'),
        PaperValue('elasticity', f'P{P}', 'E_uy', euy, 'error', 'Table 7', _key('rel_l2_uy')),
        PaperValue('elasticity', f'P{P}', 'E_sxx', sxx, 'error', 'Table 7', _key('rel_l2_sxx')),
        PaperValue('elasticity', f'P{P}', 'E_syy', syy, 'error', 'Table 7', _key('rel_l2_syy')),
        PaperValue('elasticity', f'P{P}', 'E_sxy', sxy, 'error', 'Table 7', _key('rel_l2_sxy')),
    )

# Table 9: Kovasznay (paper's own 200 x 200 error grid).
for P, iters, runtime, eu, ev, ep in [
        (75, 16, 0.06, 3.8e-1, 1.2e0, 8.8e-1), (300, 10, 0.22, 2.9e-2, 1.3e-1, 4.8e-1),
        (675, 6, 0.67, 2.0e-5, 1.3e-4, 8.1e-4), (1200, 6, 1.87, 7.2e-9, 2.6e-8, 1.4e-7),
        (1875, 6, 5.58, 7.2e-13, 5.3e-12, 1.3e-11)]:
    PAPER_VALUES += (
        PaperValue('kovasznay', f'P{P}', 'iterations', iters, 'iterations', 'Table 9', _key('n_outer_iters')),
        PaperValue('kovasznay', f'P{P}', 'runtime (s)', runtime, 'time', 'Table 9', _TIME),
        PaperValue('kovasznay', f'P{P}', 'E_u', eu, 'error', 'Table 9', _key('rel_l2_u')),
        PaperValue('kovasznay', f'P{P}', 'E_v', ev, 'error', 'Table 9', _key('rel_l2_v')),
        PaperValue('kovasznay', f'P{P}', 'E_p', ep, 'error', 'Table 9', _key('rel_l2_p')),
    )

# Table 11: Beltrami, P = 7,984 (u, v, w errors identical by symmetry; w omitted).
PAPER_VALUES += (
    PaperValue('beltrami', 'P7984', 'iterations', 4, 'iterations', 'Table 11 caption', _key('n_outer_iters')),
    PaperValue('beltrami', 'P7984', 'wall time (s)', 300.0, 'time', 'Table 11 caption ("about 5 min")',
               _key('solve_time_total')),
    PaperValue('beltrami', 'P7984', 'E_u entire domain', 0.000345, 'error', 'Table 11', _key('rel_l2_u')),
    PaperValue('beltrami', 'P7984', 'E_v entire domain', 0.000345, 'error', 'Table 11', _key('rel_l2_v')),
    PaperValue('beltrami', 'P7984', 'E_p entire domain', 0.002610, 'error', 'Table 11', _key('rel_l2_p')),
)
for t, eu, ep in [(0.0, 0.014, 0.146), (0.25, 0.041, 0.282), (0.5, 0.037, 0.315),
                  (0.75, 0.038, 0.451), (1.0, 0.034, 0.752)]:
    PAPER_VALUES += (
        PaperValue('beltrami', 'P7984', f'E_u t={t:.2f}', eu / 100, 'error', 'Table 11', _snapshot('u', t)),
        PaperValue('beltrami', 'P7984', f'E_p t={t:.2f}', ep / 100, 'error', 'Table 11', _snapshot('p', t)),
    )

# Table 14: Darcy LiL rows -- residual MSEs in physical units (the code's
# mse_*_phys; they match the table to the digit for SPE10) and runtime.
for field, dx, dy, ce, runtime in [
        ('S1', 2.20e-5, 9.47e-5, 1.67e-12, 24), ('S2', 6.25e-5, 1.45e-4, 5.47e-12, 23),
        ('S3', 1.97e-1, 1.36e-1, 1.30e-8, 23), ('SPE10', 7.11e-1, 5.23, 8.94e-8, 24)]:
    PAPER_VALUES += (
        PaperValue('darcy', field, 'Darcy-x MSE', dx, 'residual', 'Table 14', _key('mse_Dx_phys')),
        PaperValue('darcy', field, 'Darcy-y MSE', dy, 'residual', 'Table 14', _key('mse_Dy_phys')),
        PaperValue('darcy', field, 'Continuity MSE', ce, 'residual', 'Table 14', _key('mse_CE_phys')),
        PaperValue('darcy', field, 'runtime (s)', runtime, 'time', 'Table 14', _key('total_time')),
    )

# Table 15: conditioning at the smallest and largest P (final iterate) and
# numerical rank at the largest P. Darcy's row is a range over the four
# fields; Beltrami's kappa is the retained-part ratio (sigma_1/sigma_7977).
for bench, p_min, k_min, p_max, k_max, rank in [
        ('bratu', 25, 1.7e2, 225, 1.4e9, 225), ('burgers', 25, 8.9e1, 625, 6.8e9, 625),
        ('bl', 64, 6.9e3, 1024, 3.5e16, 934), ('bl_gravity', 64, 7.8e3, 1024, 4.5e16, 936),
        ('elasticity', 50, 1.1e2, 1250, 4.4e4, 1250), ('kovasznay', 75, 7.4e1, 1875, 1.7e5, 1875)]:
    PAPER_VALUES += (
        PaperValue(bench, f'P{p_min}', 'kappa (final iterate)', k_min, 'condition', 'Table 15', _KAPPA),
        PaperValue(bench, f'P{p_max}', 'kappa (final iterate)', k_max, 'condition', 'Table 15', _KAPPA),
        PaperValue(bench, f'P{p_max}', 'numerical rank', rank, 'rank', 'Table 15', _rank),
    )
PAPER_VALUES += (
    PaperValue('darcy', 'S1', 'kappa (final iterate)', 2.6e6, 'condition', 'Table 15 (low end of the 4-field range)', _KAPPA),
    PaperValue('darcy', 'SPE10', 'kappa (final iterate)', 1.3e7, 'condition', 'Table 15 (high end of the 4-field range)', _KAPPA),
    PaperValue('beltrami', 'P7984', 'kappa (retained part)', 1.2e4, 'condition', 'Table 15', _KAPPA),
    PaperValue('beltrami', 'P7984', 'numerical rank', 7977, 'rank', 'Table 15', _rank),
)

TOLERANCE_FACTOR = {'error': 2.0, 'residual': 2.0, 'condition': 10.0}

# Known, documented reasons a B1 comparison is expected to differ.
NOTES = {
    'bl_gravity': 'basis changed to cos_fourier and loss targets retargeted (DECISIONS.md) -- '
                  'a deliberate deviation, not expected to reproduce',
}
