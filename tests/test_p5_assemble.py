"""Package 5's assembly (experiments/p5_assemble.py) on a small fake of Grace's tree. The three items'
summaries, the Beltrami comparison and the checks are stand-ins (the real ones read full run folders
and the delivered packages); they rewrite and write files as the real ones do."""

import json
import subprocess
from pathlib import Path

import pytest

import experiments.p5_assemble as m

REPO = Path(__file__).resolve().parents[1]
BESIDE = 'P3_1_beltrami_certified/beside_delivered.csv'
T1 = 'family,P,lam,delta_X,eX_over_delta,eX\r\nbratu,25,10.0,0.05,1.8,0.09\r\n'
REFS = {'burgers_pde_check': {'max_abs_pde_residual': 2.46e-12, 'max_term_scale': 4.6, 'relative': 5.35e-13,
                              'tol': 1e-8, 'passed': True},
        'bl_viscous': {'p_ref': 88, 'h21_diff_neighbour': 0.0028408400106777}}


def _git(*a):
    out = subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


needs_git = pytest.mark.skipif(_git('rev-parse', 'HEAD~1') is None, reason='needs a git checkout with history')


def _fake_grace(root, commit):
    g = root / 'package5'
    w = lambda rel, text: (g / rel).parent.mkdir(parents=True, exist_ok=True) or (g / rel).write_bytes(text.encode())  # noqa: E731
    j = lambda rel, obj: w(rel, json.dumps(obj, indent=2) + '\n')  # noqa: E731
    j('COMMIT', {'commit': commit, 'tree_hash': 'abc', 'source': 'bundle', 'locked_at_utc': 'then'})
    w('P5_1_weights/terminal.csv', T1)
    j('P5_1_weights/references.json', REFS)
    w('P3_1_beltrami_certified/terminal.csv', 'size,level\r\nB1,1\r\n')
    w('P5_2_nlls/terminal.csv', 'problem,p,solver,eL2\r\nburgers,10,lilq,1e-05\r\n')
    w('P5_4_pb/terminal_4b.csv', 'P,sigma,draw,eH2\r\n100,2.0,0,1e-05\r\n')
    cpu = {'model': 'Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz', 'logical_cores': 48, 'hostname': 'c080'}
    pools = [{'internal_api': 'flexiblas', 'version': '3.4.4', 'num_threads': 48, 'filepath': '/sw/libflexiblas.so.3'},
             {'internal_api': 'openblas', 'version': '0.3.27', 'num_threads': 48,
              'filepath': '/sw/lib/libopenblas_skylakexp-r0.3.27.so'}]
    j('provenance/timed/hardware.json', {'cpu': cpu, 'threads': {'threadpools': pools},
                                         'scheduler': {'SLURM_JOB_ID': '42', 'SLURM_JOB_NAME': 'lilq-p5-timed',
                                                       'SLURM_JOB_PARTITION': 'medium'}})
    w('provenance/timed/environment.txt', 'numpy.show_config()\n')
    w('slurm_logs/lilq-p5-preflight.41.out', 'provenance lock OK: commit x\njob 41 on c302: 24 threads, 24 cores '
                                             'usable, OverSubscribe=OK\nnumpy 1.26.4 scipy 1.13.1\ncpu-mem billing: 24 '
                                             'per hour (expected 24)\ninputs OK (11 files)\n861 passed, 35 skipped\n'
                                             'PREFLIGHT OK\n')
    w('slurm_logs/lilq-p5-timed.42.out', 'provenance lock OK\njob 42 on c080: 48 threads, 48 cores usable, '
                                         'OverSubscribe=NO\n')
    w('sacct.txt', 'JobID JobName Partition AllocTRES Elapsed State ExitCode\n---- ---- ----\n'
                   '41 lilq-p5-preflight medium billing=24,cpu=24,mem=96G 00:11:00 COMPLETED 0:0\n'
                   '42 lilq-p5-timed medium billing=48,cpu=48 00:35:00 COMPLETED 0:0\n')
    w('su_per_job.csv', 'job_id,job_name,alloc_tres,elapsed_h,state,su_per_hour,su,note\n'
                        '41,lilq-p5-preflight,cpu=24,0.1839,COMPLETED,24,4.4,\n'
                        '42,lilq-p5-timed,cpu=48,0.5836,COMPLETED,48,28.0,\n')
    w('report_log.txt', 'p5 report\n')
    adv = root / 'advisor'
    adv.mkdir()
    for name in m.ADVISOR_FILES:
        (adv / name).write_text('{}')
    extra = root / 'extra.txt'
    extra.write_text('43|lilq-p5-report|billing=24,cpu=24,mem=32G,node=1|150|COMPLETED\n')
    return g, adv, extra


@pytest.fixture
def stand_ins(monkeypatch):
    """The summaries rewrite Grace's tables (item 1's delta_X moved by round-off; ``state['change']``
    moves another column) and references.json (the PDE residual moved; ``state['refs']`` moves another
    number); each writes its comparison file; C1 to C3 pass unless ``state['checks']`` says otherwise."""
    import experiments.p5_1_weights as item1
    import experiments.p5_2_nlls as item2
    import experiments.p5_4_pb as item4
    import experiments.p5_checks as checks
    state = {'change': None, 'refs': None, 'checks': {'C1': True, 'C2': True, 'C3': True}}

    def s1(out, reference_dir, expected_bratu=None, expected_bl=None, **k):
        t = T1.replace('0.05,', '0.05000000000005,')                   # 1e-12 relative: round-off in delta_X
        if state['change']:
            t = t.replace(*state['change'])
        (Path(out) / 'P5_1_weights' / 'terminal.csv').write_bytes(t.encode())
        refs = json.loads(json.dumps(REFS))
        refs['burgers_pde_check'].update(max_abs_pde_residual=2.37e-12, relative=5.15e-13)
        if state['refs']:
            refs['bl_viscous']['h21_diff_neighbour'] = state['refs']
        (Path(out) / 'P5_1_weights' / 'references.json').write_text(json.dumps(refs, indent=2))
        (Path(out) / 'P5_1_weights' / 'compare_expected.csv').write_text('family\nbratu\n')
        return [], refs

    def s2(out, expected=None):
        (Path(out) / 'P5_2_nlls' / 'compare_expected.csv').write_text('problem\nburgers\n')
        return [], []

    def s4(out_root, expected_rf=None):
        (Path(out_root) / 'P5_4_pb' / 'compare_expected_4b.csv').write_text('P\n100\n')
        return {}

    def beside(p5_root, p3_root):
        (Path(p5_root) / BESIDE).write_text('size\nB1\n')
        return [{}, {}]

    def collect(out_root, p2, p3, p4):
        (Path(out_root) / 'checks.json').write_text(json.dumps({'summary': state['checks'], 'checks': {}}))
        return dict(state['checks'])

    monkeypatch.setattr(item1, 'summarize', s1)
    monkeypatch.setattr(item1, 'beltrami_beside_delivered', beside)
    monkeypatch.setattr(item2, 'summarize', s2)
    monkeypatch.setattr(item4, 'summarize', s4)
    monkeypatch.setattr(checks, 'collect', collect)
    return state


@needs_git
def test_the_assembly_and_its_records(tmp_path, stand_ins):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    out = tmp_path / 'out'
    summary, total = m.assemble(g, out, tmp_path, tmp_path, tmp_path, tmp_path, adv, extra)
    assert summary == {'C1': True, 'C2': True, 'C3': True}
    assert b'\r' not in (out / 'checks.json').read_bytes()
    for rel in ('P5_1_weights/terminal.csv', 'P5_1_weights/references.json'):
        assert (out / rel).read_bytes() == (g / rel).read_bytes()     # Grace's kept
    for rel in ('P5_1_weights/compare_expected.csv', 'P5_2_nlls/compare_expected.csv',
                'P5_4_pb/compare_expected_4b.csv', BESIDE):
        assert (out / rel).exists()
    rec = json.loads((out / m.RECORD).read_text())
    assert rec['largest_relative_difference_grace_vs_assembly']['P5_1_weights/terminal.csv delta_X'] == \
        pytest.approx(1e-12, rel=1e-3)
    r = rec['references_json_roundoff']['/burgers_pde_check/max_abs_pde_residual']
    assert r['grace'] == 2.46e-12 and r['assembly'] == 2.37e-12
    log = (out / 'assembly_log.txt').read_text()
    assert 'written with the advisor\'s files: P5_1_weights/compare_expected.csv' in log
    su = (out / 'su.csv').read_text()
    assert total == pytest.approx(4.4 + 28.0 + 1.0) and 'from sacct after it ended' in su
    hw = json.loads((out / 'hardware.json').read_text())
    assert 'job 42 (lilq-p5-timed) on c080' in hw['cpu_source'] and hw['jobs']['41']['node'] == 'c302'
    env = (out / 'environment.txt').read_text()
    assert '861 passed, 35 skipped' in env and 'at 4 threads each' in env
    dec = (out / 'DECISIONS_package5.md').read_text()
    assert dec.startswith('# DECISIONS.md: the Package 5 entries') and '-- Package 4' not in dec
    prov = json.loads((out / 'provenance.json').read_text())
    assert set(prov['advisor_files']) == set(m.ADVISOR_FILES)
    assert not any((out / name).exists() for name in m.ADVISOR_FILES)   # the advisor's files are not copied
    m.assemble(g, out, tmp_path, tmp_path, tmp_path, tmp_path, adv, extra)   # a second time: the same result


@needs_git
def test_a_table_that_differs_outside_delta_P_stops_the_assembly(tmp_path, stand_ins):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    stand_ins['change'] = (',1.8,', ',1.81,')                          # eX_over_delta: a delta column, recorded
    m.assemble(g, tmp_path / 'out', tmp_path, tmp_path, tmp_path, tmp_path, adv, extra)
    stand_ins['change'] = (',0.09\r\n', ',0.0901\r\n')                   # eX: not delta_P
    with pytest.raises(SystemExit, match="differ from Grace's"):
        m.assemble(g, tmp_path / 'out2', tmp_path, tmp_path, tmp_path, tmp_path, adv, extra)


@needs_git
def test_references_json_may_differ_only_in_the_pde_residual(tmp_path, stand_ins):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    stand_ins['refs'] = 0.0029                                          # the H^{2,1} difference: not round-off
    with pytest.raises(SystemExit, match='h21_diff_neighbour'):
        m.assemble(g, tmp_path / 'out', tmp_path, tmp_path, tmp_path, tmp_path, adv, extra)


@needs_git
def test_a_failed_check_is_the_exit_code_not_a_stop(tmp_path, stand_ins):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    stand_ins['checks'] = {'C1': True, 'C2': False, 'C3': True}
    args = ['--grace', str(g), '--out', str(tmp_path / 'out'), '--reference-dir', str(tmp_path), '--package2',
            str(tmp_path), '--package3', str(tmp_path), '--package4', str(tmp_path), '--advisor', str(adv)]
    assert m.main(args) == 1 and (tmp_path / 'out' / 'assembly_log.txt').exists()


@needs_git
def test_the_assembly_refuses_runs_from_another_line(tmp_path, stand_ins):
    g, adv, extra = _fake_grace(tmp_path, '0' * 40)
    with pytest.raises(SystemExit, match='not an ancestor'):
        m.assemble(g, tmp_path / 'out', tmp_path, tmp_path, tmp_path, tmp_path, adv, extra)


# ---------------------------------------------------------------- the re-time (stage p5t)

RUN = {'problem': 'burgers', 'p': 10, 'P': 100, 'solver': 'lbfgs', 'iterations': 138, 'evaluations': 222,
       'restarts': 2, 'status': 'stall (f1)', 'time_s': 30.9, 'timing_counts': [138], 'norm_N': 0.0323,
       'eL2': 0.00399, 'delta_eL2': 0.00199}


def _fake_retime(root, g, commit, change=None):
    """The timed job's run in Grace's tree, and the re-time's tree (stage p5t) of the same run."""
    import numpy as np
    beta = np.linspace(-0.6, 0.6, 7)
    for base, extra in ((g, {}), (root / 'package5_retime', {'time_s': 0.37, 'warmup_s': 59.7})):
        run = base / m.RETIMED
        run.mkdir(parents=True, exist_ok=True)
        (run / 'result.json').write_text(json.dumps(RUN | extra | ((change or {}) if extra else {})))
        np.save(run / 'beta.npy', beta)
    r = root / 'package5_retime'
    (r / 'COMMIT').write_text(json.dumps({'commit': commit, 'tree_hash': 'def', 'source': 'bundle',
                                          'locked_at_utc': 'later'}))
    (r / 'provenance' / 'timed').mkdir(parents=True)
    (r / 'provenance' / 'timed' / 'hardware.json').write_text(json.dumps({
        'cpu': {'hostname': 'c057'}, 'scheduler': {'SLURM_JOB_ID': '44', 'SLURM_JOB_NAME': 'lilq-p5t-retime',
                                                   'SLURM_JOB_PARTITION': 'medium', 'holds_whole_node': True}}))
    return r


@needs_git
def test_the_retime_sits_beside_the_timed_jobs_run(tmp_path, stand_ins):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    r = _fake_retime(tmp_path, g, _git('rev-parse', 'HEAD~1'))
    extra.write_text(extra.read_text() + '44|lilq-p5t-retime|billing=48,cpu=48,mem=360G,node=1|291|COMPLETED\n')
    out = tmp_path / 'out'
    _, total = m.assemble(g, out, tmp_path, tmp_path, tmp_path, tmp_path, adv, extra, r)
    assert total == pytest.approx(4.4 + 28.0 + 1.0 + 48 * 291 / 3600, abs=0.05)
    assert (out / m.RETIME / m.RETIMED / 'result.json').read_bytes() == (r / m.RETIMED / 'result.json').read_bytes()
    assert json.loads((out / m.RETIMED / 'result.json').read_text())['time_s'] == 30.9      # Grace's kept
    row = (out / m.RETIME_TABLE).read_text().splitlines()
    assert row[0].startswith('problem,p,solver,iterations,time_s_timed_job,')
    assert row[1].startswith('burgers,10,lbfgs,138,30.9,') and ',0.37,' in row[1] and ',59.7,5,44,c057,0.0' in row[1]
    assert json.loads((out / 'hardware.json').read_text())['retime']['node'] == 'c057'
    assert json.loads((out / 'provenance.json').read_text())['retime']['job'] == '44'
    m.assemble(g, out, tmp_path, tmp_path, tmp_path, tmp_path, adv, extra)        # without it: its folder goes
    assert not (out / m.RETIME).exists()


@needs_git
@pytest.mark.parametrize('change', [{'iterations': 139}, {'eL2': 0.00398}])
def test_a_retime_of_a_different_run_stops_the_assembly(tmp_path, stand_ins, change):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    r = _fake_retime(tmp_path, g, _git('rev-parse', 'HEAD~1'), change)
    with pytest.raises(SystemExit, match='the timed job had'):
        m.assemble(g, tmp_path / 'out', tmp_path, tmp_path, tmp_path, tmp_path, adv, extra, r)


# ---------------------------------------------------------------- Addendum 2 (stage p5a)

A2_TERMINAL = 'run,rule_step,eL2,delta_L2,eL2_over_delta\r\nA_cheb_P256_nu0.1_lam10000,7,0.0202,0.0114,1.77\r\n'


def _fake_a2(root, commit, k2=True):
    """Addendum 2's tree: every run's files, its summary and checks, its job's record."""
    import experiments.p5_a2_plateau as a2m
    a = root / 'package5_a2'
    item = a / a2m.ITEM
    for r in a2m.runs():
        d = item / a2m.run_name(*r)
        d.mkdir(parents=True)
        for f in m.A2_RUN_FILES:
            (d / f).write_text('x')
    (item / 'terminal.csv').write_bytes(A2_TERMINAL.encode())
    (item / 'stretches.csv').write_bytes(b'run,first,last,steps\r\nA_cheb_P256_nu0.1_lam10000,4,7,3\r\n')
    (item / 'trig_delta_P.json').write_text(json.dumps({'P': 1024, 'gelsy': {'delta_L2': 0.000897, 'rank_L2': 1024}}))
    (item / 'item1_references.json').write_text(json.dumps({'p_ref': 88, 'n_gl': 176, 'h21_diff_neighbour': 0.0130}))
    (item / 'checks.json').write_text(json.dumps({'K1': {'pass': True, 'rows': []}, 'K2': {'pass': k2, 'iterates': 61}}))
    (a / 'COMMIT').write_text(json.dumps({'commit': commit, 'tree_hash': 'a2', 'source': 'bundle', 'locked_at_utc': 'x'}))
    (a / 'provenance' / 'a2').mkdir(parents=True)
    (a / 'provenance' / 'a2' / 'hardware.json').write_text(json.dumps({
        'cpu': {'hostname': 'c119'}, 'scheduler': {'SLURM_JOB_ID': '45', 'SLURM_JOB_NAME': 'lilq-p5a-a2',
                                                   'SLURM_JOB_PARTITION': 'medium'}}))
    (a / 'slurm_logs').mkdir()
    (a / 'slurm_logs' / 'lilq-p5a-a2.40.out').write_text('the job that timed out\n')
    log = root / 'lilq-p5a-a2.45.out'
    log.write_text('PARTS A TO C AND K1 DONE\n')
    return a, log


@pytest.fixture
def a2_stand_ins(monkeypatch):
    """The summary rewrites terminal.csv (delta_L2 by round-off; ``state['change']`` moves another cell) and
    the JSON files (trig delta_P by round-off; ``state['refs']`` moves the H^{2,1} gate); 3a writes one folder."""
    import experiments.p5_a2_plateau as a2m
    state = {'change': None, 'refs': None, 'seen': {}}

    def summarize(out_root, reference_dir, item1_root, delivered):
        item = Path(out_root) / a2m.ITEM
        t = A2_TERMINAL.replace(',0.0114,', ',0.011400000000001,')
        if state['change']:
            t = t.replace(*state['change'])
        (item / 'terminal.csv').write_bytes(t.encode())
        (item / 'trig_delta_P.json').write_text(json.dumps({'P': 1024, 'gelsy': {'delta_L2': 0.000897000000001,
                                                                                  'rank_L2': 1024}}))
        (item / 'item1_references.json').write_text(json.dumps(
            {'p_ref': 88, 'n_gl': 176, 'h21_diff_neighbour': state['refs'] or 0.0130}))
        state['seen'] = {'item1_root': Path(item1_root), 'delivered': Path(delivered)}

    def all_updates(package5, package4):
        (Path(package5) / a2m.ITEM / 'updates_package4').mkdir(parents=True, exist_ok=True)
        return {'item1': 96}

    monkeypatch.setattr(a2m, 'summarize', summarize)
    monkeypatch.setattr(a2m, 'all_updates', all_updates)
    return state


@needs_git
def test_addendum_2_is_copied_checked_and_counted(tmp_path, stand_ins, a2_stand_ins):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    a, joblog = _fake_a2(tmp_path, _git('rev-parse', 'HEAD~1'))
    extra.write_text(extra.read_text() + '40|lilq-p5a-a2|billing=24,cpu=24,mem=96G,node=1|3671|TIMEOUT\n'
                                         '45|lilq-p5a-a2|billing=24,cpu=24,mem=96G,node=1|954|COMPLETED\n')
    out = tmp_path / 'out'
    summary, total = m.assemble(g, out, tmp_path, tmp_path / 'p2', tmp_path, tmp_path, adv, extra, a2=a,
                                a2_logs=[joblog])
    assert summary == {'C1': True, 'C2': True, 'C3': True, 'K1': True, 'K2': True}
    assert total == pytest.approx(4.4 + 28.0 + 1.0 + 24 * (3671 + 954) / 3600, abs=0.1)   # each job's SU to 0.1
    item = out / 'P5_A2_initial_profile'
    for name in ('terminal.csv', 'trig_delta_P.json', 'item1_references.json'):          # Grace's kept
        assert (item / name).read_bytes() == (a / 'P5_A2_initial_profile' / name).read_bytes()
    assert (out / m.A2 / 'slurm_logs' / 'lilq-p5a-a2.45.out').exists() and (out / m.A2 / 'COMMIT').exists()
    assert not (out / m.A2 / 'P5_A2_initial_profile').exists()                         # the runs: Section 6's path
    assert (out / 'P5_A2_initial_profile' / 'updates_package4').is_dir()                  # Section 3a, in out
    assert a2_stand_ins['seen'] == {'item1_root': out, 'delivered': tmp_path / 'p2' / m.A2_DELIVERED}
    rec = json.loads((out / m.A2_RECORD).read_text())['largest_relative_difference_grace_vs_assembly']
    assert rec['P5_A2_initial_profile/terminal.csv delta_L2'] == pytest.approx(8.8e-14, rel=0.05)
    assert rec['trig_delta_P.json /gelsy/delta_L2'] == pytest.approx(1.1e-12, rel=0.05)
    assert json.loads((out / 'hardware.json').read_text())['addendum2']['node'] == 'c119'
    assert json.loads((out / 'provenance.json').read_text())['addendum2']['job'] == '45'
    m.assemble(g, out, tmp_path, tmp_path / 'p2', tmp_path, tmp_path, adv, extra)      # without it: its folders go
    assert not (out / m.A2).exists() and not (out / 'P5_A2_initial_profile').exists()
    assert not (out / m.A2_RECORD).exists()


@needs_git
@pytest.mark.parametrize('change, refs, match', [
    ((',7,', ',8,'), None, "differ from Grace's"),                                       # the rule step
    (None, 0.0131, 'h21_diff_neighbour'),                                                # the gate, in JSON
])
def test_an_addendum_2_summary_that_differs_stops_the_assembly(tmp_path, stand_ins, a2_stand_ins, change, refs, match):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    a, _ = _fake_a2(tmp_path, _git('rev-parse', 'HEAD~1'))
    a2_stand_ins['change'], a2_stand_ins['refs'] = change, refs
    with pytest.raises(SystemExit, match=match):
        m.assemble(g, tmp_path / 'out', tmp_path, tmp_path, tmp_path, tmp_path, adv, extra, a2=a)


@needs_git
def test_a_failed_k2_is_the_exit_code_and_a_missing_run_stops(tmp_path, stand_ins, a2_stand_ins):
    g, adv, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    a, _ = _fake_a2(tmp_path, _git('rev-parse', 'HEAD~1'), k2=False)
    summary, _ = m.assemble(g, tmp_path / 'out', tmp_path, tmp_path, tmp_path, tmp_path, adv, extra, a2=a)
    assert summary['K2'] is False
    (a / 'P5_A2_initial_profile' / 'D_trig_P1024_nu0.1_lam10' / 'betas.npy').unlink()
    with pytest.raises(SystemExit, match='1 run files missing'):
        m.assemble(g, tmp_path / 'out2', tmp_path, tmp_path, tmp_path, tmp_path, adv, extra, a2=a)
