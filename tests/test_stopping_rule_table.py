"""The manuscript's revised termination rule (the advisor's reply to wave 4,
item 2(b)): both stall-detector conditions at n_s consecutive steps, tau_chi
= 0.1, tau_r = 0.01, n_s = 2; the stall_rule_fires log column; and
experiments/stopping_rule_table.py, which recomputes Table 16 and the
held-out evaluation from existing logs.

On the real wave 1 logs (not in the repository), the script reproduces the
advisor's ns_eval.py row by row: 66 Table 16 rows and 576 held-out rows,
no difference; for n_s = 2, 69 of 192 held-out runs fire, 76 are censored,
and the returned residuals are within 0.039% of the smallest (DECISIONS.md)."""

import csv
import json

import numpy as np
import pytest

import lilq.iteration_log as il
from lilq.instrumentation import (
    DEFAULT_N_S, DEFAULT_TAU_CHI, DEFAULT_TAU_R, stall_flags, stall_rule_index,
)


def _ns_eval_fire(chi, rl, ns, tc=0.1, tr=0.01):
    """The advisor's ns_eval.py ``fire_ns``, verbatim in substance: the
    reference the rule is checked against."""
    b = np.zeros(len(chi), bool)
    for i in range(1, len(chi)):
        b[i] = np.isfinite(chi[i]) and chi[i] <= tc and abs(rl[i] - rl[i - 1]) <= tr * rl[i]
    for i in range(1, len(chi)):
        if i - ns + 1 >= 1 and b[i - ns + 1:i + 1].all():
            return i
    return None


def test_defaults_are_the_manuscripts():
    assert (DEFAULT_TAU_CHI, DEFAULT_TAU_R, DEFAULT_N_S) == (0.1, 0.01, 2)


@pytest.mark.parametrize('seed', range(40))
def test_rule_matches_the_advisors_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 30))
    chi = rng.choice([0.01, 0.05, 0.5, np.nan], size=n, p=[0.4, 0.3, 0.2, 0.1])
    rl = np.cumprod(rng.choice([1.0, 0.999, 0.995, 0.9, 0.5], size=n))
    for ns in (1, 2, 3):
        assert stall_rule_index(chi, rl, n_s=ns) == _ns_eval_fire(chi, rl, ns)


def test_persistence_and_censoring():
    chi = np.array([np.nan, 0.05, 0.5, 0.05, 0.05, 0.05])
    rl = np.ones(6)
    assert list(stall_flags(chi, rl)) == [False, True, False, True, True, True]
    assert stall_rule_index(chi, rl, n_s=1) == 1
    assert stall_rule_index(chi, rl, n_s=2) == 4          # steps 3 and 4: returns u^(5)
    assert stall_rule_index(chi, rl, n_s=3) == 5
    assert stall_rule_index(chi[:5], rl[:5], n_s=3) is None   # censored: the steps run out
    with pytest.raises(ValueError):
        stall_rule_index(chi, rl, n_s=0)


def test_logged_column_is_the_n_s_rule_beside_the_single_step_flag(monkeypatch):
    """The tracker logs stall_flag (one step) and stall_rule_fires (n_s
    consecutive steps); a scripted flag sequence checks the persistence."""
    script = iter([True, True, False, True, True, True])
    monkeypatch.setattr(il, '_stall_flag', lambda *a, **k: next(script))
    rng = np.random.default_rng(0)
    A, b = rng.standard_normal((12, 4)), rng.standard_normal(12)
    for n_s, expected in ((2, [False, False, True, False, False, True, True]),
                          (3, [False, False, False, False, False, False, True])):
        script = iter([True, True, False, True, True, True])
        monkeypatch.setattr(il, '_stall_flag', lambda *a, **k: next(script))
        tracker = il.LilQDiagnosticsTracker(n_s=n_s)
        beta = np.zeros(4)
        rows = []
        for k in range(7):
            rows.append(tracker.step(k=k, A_stacked=A, b_stacked=b, beta_prev=beta, beta_new=beta + 0.1,
                                     total_loss=0.5, rank_gelsy=4, t_assemble_s=0.0, t_solve_s=0.0,
                                     is_final_iterate=False))
            beta = beta + 0.1
        assert [r['stall_flag'] for r in rows] == [False, True, True, False, True, True, True]
        assert [r['stall_rule_fires'] for r in rows] == expected


def _log(path, chi, rl, R, e, roundoff=None, kappa=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    K = len(chi)                                   # solve rows 0..K-1, terminal row K
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['k', 'chi', 'norm_Rlin_h', 'norm_R_h', 'eps_u', 'roundoff_ratio',
                                          'kappa', 'kappa_retained'])
        w.writeheader()
        for k in range(K + 1):
            w.writerow({'k': k, 'chi': chi[k] if k < K else '', 'norm_Rlin_h': rl[k] if k < K else '',
                        'norm_R_h': R[k], 'eps_u': e[k],
                        'roundoff_ratio': (roundoff or [1.0] * K)[k] if k < K else '',
                        'kappa': (kappa or [10.0] * K)[k] if k < K else '', 'kappa_retained': ''})


def test_table16_and_held_out_from_logs(tmp_path):
    import experiments.stopping_rule_table as srt
    b = tmp_path / 'B'
    chi = [np.nan, 0.5, 0.05, 0.05, 0.05, 0.05]
    rl = [1.0, 0.5, 0.2, 0.2, 0.2, 0.2]
    R = [1.0, 0.5, 0.21, 0.2, 0.2, 0.2, 0.2]
    e = [1.0, 0.1, 0.01, 0.002, 0.001, 0.001, 0.001]
    _log(b / 'run_P1_cpu_kmax' / 'iterations.csv', chi, rl, R, e)
    _log(b / 'run_P1_cpu_paper' / 'iterations.csv', chi[:4], rl[:4], R[:5], e[:5])
    rows, summary = srt.table16(b, (1, 2), 0.1, 0.01, runs=('run_P1',))
    r1, r2 = rows
    # flags at k = 3, 4, 5 (k = 2 changes ||R_lin|| by 60%): n_s = 1 fires at 3, n_s = 2 at 4
    assert (r1['fires_at'], r1['returned'], r2['fires_at'], r2['returned']) == (3, 4, 4, 5)
    assert r1['paper_stop'] == 4 and r1['E_over_paper_stop'] == pytest.approx(0.001 / 0.001)
    assert r1['class'] == 'A' and summary[2]['early'] == 0 and summary[2]['R_max'] == pytest.approx(1.0)

    runs = tmp_path / 'C'
    _log(runs / 'kovasznay_P300_r5_cgl' / 'iterations.csv', chi, rl, R, e)
    _log(runs / 'kovasznay_P300_r3_paper' / 'iterations.csv', chi, rl, R, e)        # excluded
    late = [np.nan, 0.5, 0.5, 0.5, 0.5, 0.05]                                      # met at the last step only
    _log(runs / 'bratu_P25_r5_cgl' / 'iterations.csv', late, [1.0] * 6, R, e)
    h_rows, h_sum = srt.held_out(runs, (1, 2), 0.1, 0.01)
    assert h_sum[2]['runs'] == 2 and h_sum[2]['fired'] == 1 and h_sum[2]['censored'] == 1
    assert h_sum[1]['fired'] == 2 and h_sum[1]['censored'] == 0
    assert {r['run'] for r in h_rows} == {'kovasznay_P300_r5_cgl', 'bratu_P25_r5_cgl'}


def test_cli_writes_the_tables(tmp_path):
    import experiments.stopping_rule_table as srt
    b = tmp_path / 'B'
    chi = [np.nan, 0.05, 0.05, 0.05]
    for name in srt.TABLE16_RUNS:
        _log(b / f'{name}_cpu_kmax' / 'iterations.csv', chi, [1.0] * 4, [1.0] * 5, [1.0] * 5)
        _log(b / f'{name}_cpu_paper' / 'iterations.csv', chi[:2], [1.0] * 2, [1.0] * 3, [1.0] * 3)
    srt.main(['--b-root', str(b), '--out-dir', str(tmp_path / 'out')])
    s = json.loads((tmp_path / 'out' / 'stopping_rule_summary.json').read_text())
    assert s['tau_r'] == 0.01 and s['n_s'] == [1, 2, 3] and s['table16']['2']['never_fires'] == 0
    assert len(list(csv.DictReader(open(tmp_path / 'out' / 'stopping_rule_table16.csv')))) == 3 * 22
