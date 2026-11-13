"""Package 4's assembly (experiments/p4_assemble.py) on a small fake of Grace's tree. The two items'
summaries are stand-ins (the real ones read full run folders); check collection is the real one."""

import json
import subprocess
from pathlib import Path

import pytest

import experiments.p4_assemble as m

REPO = Path(__file__).resolve().parents[1]
GRACE_SUMMARY = {'M0': True, 'M2': True, 'M3': True, 'M4': True, 'M5': True, 'M9': False, 'M6': None, 'M7': True}


def _git(*a):
    out = subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


needs_git = pytest.mark.skipif(_git('rev-parse', 'HEAD~1') is None, reason='needs a git checkout with history')

T2 = 'P,seed,delta_L2,delta_H2,eL2,eL2_over_delta\r\n' + ''.join(
    f'{P},{s},{d!r},{10 * d!r},{3 * d!r},3.0\r\n' for P, d in ((484, 6e-6), (676, 1.5e-7)) for s in (0, 1))
T1 = 'p,P,solves_to_stop,delta_P,error\r\n10,100,6,0.001,0.002\r\n'


def _fake_grace(root, commit):
    g = root / 'package4'
    w = lambda rel, text: (g / rel).parent.mkdir(parents=True, exist_ok=True) or (g / rel).write_bytes(text.encode())  # noqa: E731
    j = lambda rel, obj: w(rel, json.dumps(obj, indent=2) + '\n')  # noqa: E731
    j('COMMIT', {'commit': commit, 'tree_hash': 'abc', 'source': 'bundle', 'locked_at_utc': 'then'})
    j('checks.json', {'summary': GRACE_SUMMARY, 'checks': {}})
    j('checks_item1.json', {'M2': {'pass': True}, 'M3': {'pass': True}, 'M4': {'pass': True}, 'M5': {'pass': True},
                            'M9_item1': {'pass': False}})
    j('P4_2_pb_inclusions/checks_item2.json', {'M3': {'pass': True}, 'M6': {'pass': None}, 'M7': {'pass': True},
                                               'M9_item2': {'pass': True}})
    j('P4_checks/m0.json', {'check': 'M0', 'pass': True, 'runs': {}})
    w('P4_1a_burgers_gauss/terminal.csv', T1)
    w('P4_1b_burgers_enrichment/summary.csv', 'variant,tolerance\r\nE1,1e-05\r\n')
    w('P4_1c_burgers_constants/constants.csv', 'p,c1_min\r\n10,0.99\r\n')
    w('P4_2_pb_inclusions/lilq/terminal.csv', T2)
    cpu = {'model': 'Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz', 'logical_cores': 48, 'hostname': 'c098'}
    pools = [{'internal_api': 'flexiblas', 'version': '3.4.4', 'num_threads': 48, 'filepath': '/sw/libflexiblas.so.3'},
             {'internal_api': 'openblas', 'version': '0.3.27', 'num_threads': 48,
              'filepath': '/sw/lib/libopenblas_skylakexp-r0.3.27.so'}]
    j('provenance/timed/hardware.json', {'cpu': cpu, 'threads': {'threadpools': pools},
                                         'scheduler': {'SLURM_JOB_ID': '12', 'SLURM_JOB_NAME': 'lilq-p4-timed',
                                                       'SLURM_JOB_PARTITION': 'medium'}})
    w('provenance/timed/environment.txt', 'numpy.show_config()\n')
    w('slurm_logs/lilq-p4-preflight.11.out', 'provenance lock OK: commit x\njob 11 on c1: 24 threads, 24 cores usable, '
                                             'OverSubscribe=OK\nno GPU visible\nnumpy 1.26.4 scipy 1.13.1\n'
                                             'inputs OK (6 files)\n807 passed, 31 skipped\nPREFLIGHT OK\n')
    w('slurm_logs/lilq-p4-timed.12.out', 'provenance lock OK\njob 12 on c098: 48 threads, 48 cores usable, OverSubscribe=NO\n')
    w('sacct.txt', 'JobID JobName Partition AllocTRES Elapsed State ExitCode\n---- ---- ----\n'
                   '11 lilq-p4-preflight medium billing=24,cpu=24 00:10:00 COMPLETED 0:0\n'
                   '12 lilq-p4-timed medium billing=48,cpu=48 00:06:00 COMPLETED 0:0\n')
    w('su_per_job.csv', 'job_id,job_name,alloc_tres,elapsed_h,state,su_per_hour,su,note\n'
                        '11,lilq-p4-preflight,cpu=24,0.1667,COMPLETED,24,4.0,\n'
                        '12,lilq-p4-timed,cpu=48,0.1,COMPLETED,48,4.8,\n')
    w('report_log.txt', 'p4 report\n')
    adv = root / 'advisor'
    adv.mkdir()
    (adv / 'expected_burgers.json').write_text('{"direct": {}}')
    (adv / 'expected_pb.json').write_text(json.dumps({'best': {'484': [6.0e-6, 6e-5], '676': [1.5e-7, 1.5e-6]}}))
    extra = root / 'extra.txt'
    extra.write_text('13|lilq-p4-report|billing=24,cpu=24,mem=32G,node=1|300|COMPLETED\n')
    return g, adv / 'expected_burgers.json', adv / 'expected_pb.json', extra


@pytest.fixture
def summaries(monkeypatch):
    """Stand-ins for the two summaries: they rewrite Grace's tables as the real ones do (CRLF; item 2's
    delta_P moved by round-off, ``state['change']`` moves another column) and write each item's
    checks with the advisor's files (``state['item2']``: item 2's verdicts)."""
    import experiments.p4_1_burgers_gauss as item1
    import experiments.p4_2_pb_inclusions as item2
    state = {'change': None, 'item2': {'M6': True, 'M8': True}}

    def s1(out, reference_dir, expected, sizes=None):
        (Path(out) / 'P4_1a_burgers_gauss' / 'terminal.csv').write_bytes(T1.encode())
        Path(out, 'checks_item1.json').write_text(json.dumps(
            {'M1': {'pass': True}, 'M2': {'pass': True}, 'M3': {'pass': True}, 'M4': {'pass': True},
             'M5': {'pass': True}, 'M9_item1': {'pass': False}}))   # write_text: CRLF on Windows, as the real one
        return [], {}

    def s2(out, expected=None):
        t = T2.replace('6e-06', '6.000006e-06')                     # 1e-6 relative: round-off in delta_P
        if state['change']:
            t = t.replace(*state['change'])
        (Path(out) / 'P4_2_pb_inclusions' / 'lilq' / 'terminal.csv').write_bytes(t.encode())
        Path(out, 'P4_2_pb_inclusions', 'checks_item2.json').write_text(json.dumps(
            {'M3': {'pass': True}, 'M6': {'pass': state['item2']['M6']}, 'M7': {'pass': True},
             'M8': {'pass': state['item2']['M8']}, 'M9_item2': {'pass': True}}))
        return [], {}

    monkeypatch.setattr(item1, 'summarize', s1)
    monkeypatch.setattr(item2, 'summarize', s2)
    return state


@needs_git
def test_the_assembly_and_its_records(tmp_path, summaries):
    g, eb, ep, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    out = tmp_path / 'out'
    summary, total = m.assemble(g, out, tmp_path, eb, ep, extra)
    assert summary == {'M0': True, 'M1': True, 'M2': True, 'M3': True, 'M4': True, 'M5': True, 'M6': True,
                       'M7': True, 'M8': True, 'M9': False}
    for rel in m.CHECK_FILES:                      # Grace's kept beside; the new ones in LF
        assert (out / m._grace_name(rel)).read_bytes() == (g / rel).read_bytes()
        assert b'\r' not in (out / rel).read_bytes()
    assert json.loads((out / 'checks.json').read_text())['summary'] == summary
    for rel in ('P4_2_pb_inclusions/lilq/terminal.csv', 'P4_1a_burgers_gauss/terminal.csv'):
        assert (out / rel).read_bytes() == (g / rel).read_bytes()     # the tables stay Grace's
    rec = json.loads((out / 'delta_P_across_machines.json').read_text())
    assert rec['item2_by_P']['484']['grace']['delta_L2'] == 6e-6
    assert rec['item2_by_P']['484']['assembly_machine']['delta_L2'] == 6.000006e-6
    assert rec['item2_by_P']['484']['advisor'] == {'delta_L2': 6.0e-6, 'delta_H2': 6e-5}
    assert rec['item2_by_P']['484']['delta_L2_spread'] == pytest.approx(1e-6)
    assert rec['item2_by_P']['676']['delta_L2_spread'] == 0
    assert rec['largest_relative_difference_grace_vs_assembly'] == {
        'lilq/terminal.csv delta_L2': pytest.approx(1e-6, rel=1e-5), 'lilq/terminal.csv delta_H2': 0.0,
        'lilq/terminal.csv eL2_over_delta': 0.0}
    su = (out / 'su.csv').read_bytes().decode()
    assert total == pytest.approx(4.0 + 4.8 + 2.0) and 'from sacct after it ended' in su and '\r' not in su
    hw = json.loads((out / 'hardware.json').read_text())
    assert hw['cpu']['hostname'] == 'c098' and 'job 12 (lilq-p4-timed) on c098' in hw['cpu_source']
    assert hw['jobs']['12']['threads'] == 48 and hw['jobs']['11']['node'] == 'c1'
    env = (out / 'environment.txt').read_text()
    assert 'numpy 1.26.4 scipy 1.13.1' in env and '807 passed, 31 skipped' in env and 'no GPU' not in env
    assert 'OpenBLAS 0.3.27 (libopenblas_skylakexp-r0.3.27.so), through FlexiBLAS 3.4.4, at 48 threads' in env
    assert 'loaded in the timed job (provenance/timed/hardware.json)' in env
    dec = (out / 'DECISIONS_package4.md').read_text()
    assert dec.startswith('# DECISIONS.md: the Package 4 entries')
    assert '## 2026-11-12 -- Package 4: the Grace submission' in dec and '-- Package 3' not in dec
    prov = json.loads((out / 'provenance.json').read_text())
    assert prov['runs']['commit'] == _git('rev-parse', 'HEAD~1') and len(prov['commits_between']) == 1
    assert set(prov['advisor_files']) == {'expected_burgers.json', 'expected_pb.json'}
    assert not (out / 'expected_pb.json').exists()                   # the advisor's files are not copied
    m.assemble(g, out, tmp_path, eb, ep, extra)                        # a second time: the same result
    assert (out / 'checks_grace.json').read_bytes() == (g / 'checks.json').read_bytes()


@needs_git
def test_a_table_that_differs_outside_delta_P_stops_the_assembly(tmp_path, summaries):
    g, eb, ep, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    summaries['change'] = (',3.0\r\n', ',3.0001\r\n')                # eL2_over_delta: a delta_P column, recorded
    m.assemble(g, tmp_path / 'out', tmp_path, eb, ep, extra)
    summaries['change'] = ('484,0,', '484,1,')                        # seed: not delta_P
    with pytest.raises(SystemExit, match='differ from Grace'):
        m.assemble(g, tmp_path / 'out2', tmp_path, eb, ep, extra)


@needs_git
def test_a_verdict_unlike_grace_stops_the_assembly(tmp_path, summaries, monkeypatch):
    g, eb, ep, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    summaries['item2'] = {'M6': False, 'M8': True}                   # M6 had no verdict on Grace: allowed
    summary, _ = m.assemble(g, tmp_path / 'out', tmp_path, eb, ep, extra)
    assert summary['M6'] is False
    grace = json.loads((g / 'checks.json').read_text())
    grace['summary']['M9'] = True                                   # Grace said M9 passed; the assembly says not
    (g / 'checks.json').write_text(json.dumps(grace))
    with pytest.raises(SystemExit, match="verdicts differ from Grace's"):
        m.assemble(g, tmp_path / 'out2', tmp_path, eb, ep, extra)


@needs_git
def test_the_assembly_refuses_runs_from_another_line(tmp_path, summaries):
    g, eb, ep, extra = _fake_grace(tmp_path, '0' * 40)
    with pytest.raises(SystemExit, match='not an ancestor'):
        m.assemble(g, tmp_path / 'out', tmp_path, eb, ep, extra)


def test_the_package_3_entries_stay_package_3s():
    from experiments.assemble_common import decisions
    d3, d4 = decisions(3), decisions(4)
    assert '-- Package 4' not in d3 and '-- Package 3' not in d4
    assert d3.count('\n## ') > 5 and d4.count('\n## ') >= 3
