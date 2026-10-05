"""Package 2, item 6a (P2-2): the Buckley-Leverett nu-refinement. The run list
and configurations against Section 9.1, the reference refinement (C7), the
eps_ref option of the BL solver (no change without it), one small end-to-end
pass pair with its row, and check C8."""

import csv
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import experiments.p2_2_nu_refinement as m

REPO = Path(__file__).resolve().parents[1]


def test_run_list_is_section_9_1():
    v = m.configs('viscous')
    assert sorted((N * N, nu) for N, nu in v) == sorted(
        [(P, nu) for nu in (0.1, 0.05, 0.02, 0.01) for P in (576, 1024)] + [(1600, 0.02), (1600, 0.01)])
    assert sorted((N * N, nu) for N, nu in m.configs('gravity')) == [(576, 0.02), (576, 0.05), (1024, 0.02),
                                                                     (1024, 0.05)]
    assert min(nu for _, nu in v) == 0.01                      # never nu < 0.01


@pytest.mark.parametrize('case', ['viscous', 'gravity'])
def test_only_the_viscosity_changes(case):
    from experiments.run_bl import paper_setup
    config, opt, target = m.setup(case, 24, 0.02)
    paper_config, paper_opt = paper_setup(24, case == 'gravity')
    assert config.D_coef == -0.02 and dataclasses.replace(config, D_coef=paper_config.D_coef) == paper_config
    assert opt == paper_opt and opt.max_quasi_iters_lil == 60 and target == paper_opt.R_tol
    assert m.setup('viscous', 40, 0.01)[2] is None              # no paper target at P = 1,600


def test_refine_stops_at_agreement_and_rejects_bad_fields(monkeypatch):
    import problems.buckley_leverett as bl
    base = np.linspace(0, 0.5, 50)                         # a saturation, so the range guard lets it through
    monkeypatch.setattr(bl, 'reference_solution', lambda config, n: base * (1 + 1.0 / n ** 2))
    u, rec = m.refine(None, first=100, last=6400, agreement=1e-6)
    # relative change between n/2 and n is 3/n^2: 1.2e-6 at n = 1,600, 2.9e-7 at n = 3,200
    assert rec['passed'] and rec['n_intervals'] == 3200 and np.allclose(u, base * (1 + 1 / 3200 ** 2))
    assert [st['n_intervals'] for st in rec['steps']] == [100, 200, 400, 800, 1600, 3200]
    _, rec = m.refine(None, first=100, last=200, agreement=1e-6)
    assert not rec['passed']
    monkeypatch.setattr(bl, 'reference_solution', lambda config, n: base * np.nan)
    with pytest.raises(RuntimeError):
        m.refine(None, first=100, last=200)


def _small(tmp_path):
    """A coarse reference (the solver's own, 200 intervals) for nu = 0.1."""
    from problems.buckley_leverett import TEST_GRID, reference_solution
    config, _, _ = m.setup('viscous', 8, 0.1)
    ref = tmp_path / 'reference'
    ref.mkdir()
    x = np.linspace(*config.x_domain, TEST_GRID[0])
    t = np.linspace(0, config.T_final, TEST_GRID[1])
    np.savez(m.reference_path(ref, 'viscous', 0.1), x=x, t=t, u=np.array(reference_solution(config, 200)))
    return ref


def test_reference_npz_changes_nothing_but_adds_eps_ref(tmp_path):
    from lilq.iteration_log import IterationLogger
    from problems.buckley_leverett import run_lil_q
    config, opt, _ = m.setup('viscous', 8, 0.1)
    opt = dataclasses.replace(opt, max_quasi_iters_lil=3, R_tol=0.0)
    a, b = IterationLogger(), IterationLogger()
    _, ca, _, _ = run_lil_q(config, opt, verbose=False, iteration_logger=a)
    _, cb, _, _ = run_lil_q(config, opt, verbose=False, iteration_logger=b,
                            reference_npz=m.reference_path(_small(tmp_path), 'viscous', 0.1))
    assert np.array_equal(ca, cb)
    assert [r['norm_R_h'] for r in a.rows] == [r['norm_R_h'] for r in b.rows]
    assert a.rows[-1].get('eps_ref') in (None, '') and 0 < b.rows[-1]['eps_ref'] < 1


def test_small_pass_pair_row_and_check_c8(tmp_path):
    ref = _small(tmp_path)
    out = tmp_path / 'P2_2_nu_refinement'
    k_dir, *k_fit, _ = m.run_one('viscous', 8, 0.1, 'kmax', out, ref)
    p_dir, *p_fit, _ = m.run_one('viscous', 8, 0.1, 'paper', out, ref)
    row = m.summarize_config('viscous', 8, 0.1, k_dir, tuple(k_fit), (p_dir, tuple(p_fit)))
    with open(k_dir / 'iterations.csv') as f:
        log = list(csv.DictReader(f))
    assert int(log[-1]['k']) == 60 and row['eps_ref_60'] == float(log[-1]['eps_ref'])
    assert row['stop'] in ('target', 'K_max') and row['k_stop'] <= 60
    if row['stop'] == 'target':
        assert row['k_target'] == row['k_stop']             # the two passes share their iterates
    assert row['rule_class'] in ('A', 'C', 'never') and row['rank'] <= 64
    assert (k_dir / 'run.json').exists() and (p_dir / 'solution.pt').exists()
    # C8 against a "package1" holding these same logs passes; a perturbed copy fails
    p1 = tmp_path / 'p1' / 'B_instrumentation'
    for pass_, d in (('paper', p_dir), ('kmax', k_dir)):
        shutil.copytree(d, p1 / f'bl_P64_cpu_{pass_}')
    assert m.check_c8(out, tmp_path / 'p1', sizes=(8,))['passed']
    path = p1 / 'bl_P64_cpu_kmax' / 'iterations.csv'
    rows = list(csv.DictReader(open(path)))
    rows[5]['norm_R_h'] = repr(float(rows[5]['norm_R_h']) * (1 + 1e-8))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c8 = m.check_c8(out, tmp_path / 'p1', sizes=(8,))
    assert not c8['passed'] and json.dumps(c8)


def test_item_6a_job():
    text = (REPO / 'scripts' / 'cluster' / 'package2' / 'p2s2_nu_refinement.slurm').read_text()
    assert '# lilq-resources: timed-cpu' in text and 'p2_2_nu_refinement.py reference' in text   # C8: package1's 48 threads
    assert 'p2_2_nu_refinement.py runs' in text and '--package1 "$RESULTS/package1_v2.0.0/package1"' in text
