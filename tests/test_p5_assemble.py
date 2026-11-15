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
