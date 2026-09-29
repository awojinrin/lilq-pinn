"""The four-method stall control (the advisor's reply to wave 1, item 2.5):
tolerances 0 and F1's restart-once rule, run as its own row beside the
original."""

import dataclasses

import numpy as np
import pytest
import torch

import experiments.four_method_tables as fmt
from lilq.four_method_log import CONTROL_VARIANT, FourMethodLogger, compare_histories, row_key
from lilq.solvers import solve_lil_n


def _quadratic(scale):
    def loss_fn(beta):
        r = scale * ((beta - 3.0) ** 2).sum()
        return r, r, 0.0, r * 0
    return loss_fn


def test_pytorch_rule_stalls_on_a_small_gradient_and_the_f1_rule_does_not():
    """The published rule's absolute tolerance_grad (1e-8) makes a step a
    no-op once the gradient is that small, whatever the loss scale; with
    tolerances 0 the same run keeps going."""
    start = np.array([1.0, -2.0])
    loss_fn = _quadratic(1e-10)                    # gradient ~1e-9 at the start
    _, _, pt = solve_lil_n(loss_fn, start.copy(), torch.device('cpu'), max_iterations=50, R_tol=-1.0,
                           verbose=False)
    _, _, f1 = solve_lil_n(loss_fn, start.copy(), torch.device('cpu'), max_iterations=50, R_tol=-1.0,
                           verbose=False, stall_rule='f1')
    assert pt['optimizer_stall'] and pt['total_iterations'] == 1 and pt['stall_rule'] == 'pytorch'
    assert f1['stall_rule'] == 'f1' and f1['final_loss'] < 1e-3 * pt['final_loss']


def test_f1_rule_restarts_once_then_stops():
    """A step that does not lower the loss gets one retry with a fresh
    optimizer; a second one in a row ends the run on optimizer_stall."""
    def flat(beta):                                  # zero gradient: every step is a no-op
        r = (beta * 0).sum() + 1.0
        return r, r, 0.0, r * 0
    _, _, s = solve_lil_n(flat, np.zeros(2), torch.device('cpu'), max_iterations=50, R_tol=-1.0,
                          verbose=False, stall_rule='f1')
    assert s['optimizer_stall'] and s['lbfgs_restarts'] == 1 and s['total_iterations'] == 2


def test_unknown_stall_rule_is_refused():
    with pytest.raises(ValueError, match='stall_rule'):
        solve_lil_n(_quadratic(1.0), np.zeros(2), torch.device('cpu'), max_iterations=2, verbose=False,
                    stall_rule='tolerant')


@pytest.mark.parametrize('method', ['NiL-N', 'NiL-Q', 'LiL-N'])
def test_every_method_runs_under_the_f1_rule(method):
    from experiments.run_bratu import paper_setup
    from problems.bratu import run_lil_n, run_nil_n, run_nil_q
    config, opt = paper_setup(5)
    opt = dataclasses.replace(fmt._apply_quick_budgets(opt), stall_rule='f1')
    runner = {'NiL-N': run_nil_n, 'NiL-Q': run_nil_q, 'LiL-N': run_lil_n}[method]
    summary = runner(config, opt, device=torch.device('cpu'), verbose=False)[-1]
    assert summary['stall_rule'] == 'f1' and summary['lbfgs_restarts'] >= 0
    assert np.isfinite(summary['final_loss'])


def test_old_rows_key_as_the_tables_own_runs():
    """Wave 1's CSV has no variant column: its rows are the tables' own runs."""
    row = dict(benchmark='bratu', P=100, method='NiL-N', seed='0', device='cuda')
    assert row_key(row) == row_key({**row, 'variant': ''}) != row_key({**row, 'variant': CONTROL_VARIANT})


def test_history_comparison():
    original = [[0, 1.0, 0.0], [10, 0.5, 0.1], [20, 0.25, 0.2]]
    assert compare_histories(original, [[0, 1.0, 0.0], [10, 0.5, 0.1], [20, 0.2, 0.2], [30, 0.1, 0.3]]) == (True, 0.0, 20)
    assert compare_histories(original, original + [[30, 0.1, 0.3]]) == (True, 0.0, None)
    same, rel, departs = compare_histories(original, [[0, 1.1, 0.0]])
    assert (same, departs) == (False, 0) and rel == pytest.approx(0.1)
    assert compare_histories(None, original) == (None, None, None)


def test_controls_rerun_stalled_runs_from_the_same_start(tmp_path):
    """End to end: a table's stalled rows (here relabelled, on quick budgets)
    get one control each on their own device, from the same seed -- the
    same loss at iteration 0 -- logged beside them, model saved, and split
    into four_method_controls.csv by the merge."""
    job = tmp_path / 'bratu_cpu'
    logger = FourMethodLogger()
    (P, config, opt), = fmt._bratu_runs(quick=True)
    opt = fmt._apply_quick_budgets(opt)
    for method, runner, seeds in (('NiL-N', fmt.bratu_nil_n, [1]), ('LiL-N', fmt.bratu_lil_n, None)):
        fmt._run_and_log(logger, 'bratu', P, config, opt, method, runner, seeds=seeds,
                         devices=[torch.device('cpu')], verbose=False, csv_path=job / 'four_method_tables.csv')
    rows = logger.rows
    for r in rows:
        r['stopping_reason'] = 'optimizer_stall'
    earlier = FourMethodLogger()
    for r in rows:
        earlier.record(**r)
    earlier.to_csv(job / 'four_method_tables.csv')

    originals = fmt.control_candidates(FourMethodLogger.from_csv(job / 'four_method_tables.csv').rows,
                                       [torch.device('cpu')], ['bratu'])
    assert len(originals) == 2
    assert fmt.control_candidates(rows, [torch.device('cuda')]) == []          # another device's rows
    out = tmp_path / 'wave2_bratu_cpu' / 'four_method_tables.csv'
    new = FourMethodLogger()
    fmt.run_controls(originals, new, quick=True, verbose=False, csv_path=out)
    controls = new.rows
    assert [c['variant'] for c in controls] == [CONTROL_VARIANT] * 2
    for c, o in zip(controls, originals):
        assert (c['method'], str(c['seed'] or ''), c['device']) == (o['method'], str(o['seed'] or ''), o['device'])
        assert c['same_start'] is True and c['start_loss_rel_diff'] == 0.0
        assert c['original_total_iterations'] == o['total_iterations']
        assert (out.parent / 'models' / fmt.model_dir_name(c) / 'history.csv').exists()
        assert fmt.model_dir_name(c).endswith(CONTROL_VARIANT)

    again = FourMethodLogger.from_csv(out)                                  # resume: nothing rerun
    fmt.run_controls(originals, again, quick=True, verbose=False, csv_path=out,
                     skip_keys=frozenset(again.completed_keys()))
    assert len(again) == 2

    merged, merged_controls = fmt.merge_csvs([job / 'four_method_tables.csv', out], tmp_path / 'm.csv')
    assert len(merged) == 2 and len(merged_controls) == 2
    assert (tmp_path / 'four_method_controls.csv').exists()
