"""Package 3's assembly (experiments/p3_assemble.py) on a small fake of Grace's tree."""

import json
import subprocess
from pathlib import Path

import pytest

import experiments.p3_assemble as m

REPO = Path(__file__).resolve().parents[1]


def _git(*a):
    out = subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


needs_git = pytest.mark.skipif(_git('rev-parse', 'HEAD~1') is None, reason='needs a git checkout with history')


def _fake_grace(root, commit):
    g = root / 'package3'
    w = lambda rel, text: (g / rel).parent.mkdir(parents=True, exist_ok=True) or (g / rel).write_text(text)  # noqa: E731
    j = lambda rel, obj: w(rel, json.dumps(obj))  # noqa: E731
    j('COMMIT', {'commit': commit, 'tree_hash': 'abc', 'source': 'bundle', 'locked_at_utc': 'then'})
    j('P3_checks/k0.json', {'passed': True, 'shape': [6, 2], 'gelsy': {'rel_residual': 1e-15, 'rel_error': 1e-15},
                            'pivoted_qr': {'rel_residual': 1e-15, 'rel_error': 1e-15}})
    j('P3_checks/k1.json', {'passed': True})
    j('P3_checks/k2.json', {'passed': True, 'n_rules': 144, 'worst_error': 4e-14})
    j('P3_checks/k8.json', {'passed': True, 'runs': {'beltrami_pinned': {'passed': True, 'rows_here': 5}}})
    j('P3_1_beltrami_certified/checks_item1.json', {'K3': {'passed': True}, 'K6': {'passed': True},
                                                    'K7': {'passed': True},
                                                    'K4': {'passed': True, 'max_rel_err_at_k1': 3e-14, 'rule': 'k = 1'}})
    j('checks_item2.json', {'K4 (cheb)': {'passed': True, 'max_rel_err_at_k1': 4e-15},
                            'K4 (sine)': {'passed': True, 'max_rel_err_at_k1': 3e-15}, 'K7 (cheb)': {'passed': True},
                            'K7 (sine)': {'passed': True}, 'K5 (BL)': {'passed': True}, 'K6 (BL)': {'passed': True}})
    for d in ('P3_2a_bl_chebyshev_certified', 'P3_2b_bl_lifted_sine_cheb'):
        w(f'{d}/terminal.csv', 'x\n1\n')
    j('P3_3_affine_certificates/checks_item3.json', {'assembly': {'all_bitwise': True}, 'K5': {'passed': True},
                                                     'resolves': {'darcy S1': {'identical': True}}})
    rows = 'k,norm_R_h,norm_Rlin_h,eps_ref\n' + ''.join(f'{k},{1 / (k + 1)!r},{0.5 / (k + 1)!r},{0.1 / (k + 1)!r}\n'
                                                       for k in range(6))
    w('P3_4_burgers_large_P/burgers_P625/iterations.csv', rows)
    w('P3_4_burgers_large_P/burgers_P625/losses.csv',
      'iterate,loss\n' + ''.join(f"{k},{0.0625 if k == 4 else 1.0}\n" for k in range(6)))
    j('P3_4_burgers_large_P/checks_item4.json', {'L1': {'passed': False, 'differences': [{'k': 4}]},
                                                 'L2': {'passed': True}, 'L3': {'passed': True}, 'L4': {'passed': True}})
    w('slurm_logs/lilq-p3-preflight.11.out', 'provenance lock OK: commit x\njob 11 on c1: 24 threads, 24 cores usable, '
                                             'OverSubscribe=OK\ntorch 2.10.0 OK\nnumpy 1.26.4 scipy 1.13.1\n'
                                             '700 passed, 2 skipped\nPREFLIGHT OK\n')
    w('slurm_logs/lilq-p3-k0.12.out', 'provenance lock OK\njob 12 on c2: 48 threads, 48 cores usable, OverSubscribe=NO\n')
    w('sacct.txt', 'JobID JobName Partition AllocTRES Elapsed State ExitCode\n---- ---- ----\n'
                   '11 lilq-p3-preflight medium billing=24,cpu=24 00:09:00 COMPLETED 0:0\n'
                   '12 lilq-p3-k0 medium billing=48,cpu=48 01:00:00 COMPLETED 0:0\n')
    w('su_per_job.csv', 'job_id,job_name,alloc_tres,elapsed_h,state,su_per_hour,su,note\n'
                        '11,lilq-p3-preflight,cpu=24,0.15,COMPLETED,24,3.6,\n12,lilq-p3-k0,cpu=48,1.0,COMPLETED,48,48.0,\n')
    w('report_log.txt', 'p3 report\n')
    p2 = root / 'P2_12' / 'B_instrumentation' / 'burgers_P625_cpu_paper'
    p2.mkdir(parents=True)
    (p2 / 'iterations.csv').write_text('\n'.join(rows.splitlines()[:5] + ['4,0.25,,0.02']) + '\n')   # k = 4: terminal
    extra = root / 'extra.txt'
    extra.write_text('13|lilq-p3-report|billing=24,cpu=24,mem=32G,node=1|360|COMPLETED\n')
    return g, root / 'P2_12', extra


@needs_git
def test_the_assembly_and_its_records(tmp_path):
    g, p2, extra = _fake_grace(tmp_path, _git('rev-parse', 'HEAD~1'))
    out = tmp_path / 'out'
    checks, total = m.assemble(g, out, p2, extra)
    assert all(v['passed'] for v in checks.values()) and total == pytest.approx(3.6 + 48.0 + 2.4)
    l1 = json.loads((out / 'P3_4_burgers_large_P' / 'checks_item4.json').read_text())['L1']
    assert l1['passed'] and not l1['grace']['passed'] and l1['reevaluated_at'] == _git('rev-parse', 'HEAD')
    assert json.loads((out / 'checks.json').read_text())['L1 (Addendum 1)']['passed']
    su = (out / 'su.csv').read_bytes().decode()
    assert '\r' not in su and su.count('\n') == 4 and 'its own charge' in su
    hw = json.loads((out / 'hardware.json').read_text())
    assert hw['jobs']['12'] == {'log': 'lilq-p3-k0.12.out', 'node': 'c2', 'threads': 48, 'cores_usable': 48,
                                'oversubscribe': 'NO', 'name': 'lilq-p3-k0', 'partition': 'medium',
                                'alloc_tres': 'billing=48,cpu=48', 'elapsed': '01:00:00'}
    assert hw['cpu_model'] is None and '6248R' in hw['cpu_model_note']
    env = (out / 'environment.txt').read_text()
    assert 'numpy 1.26.4 scipy 1.13.1' in env and '700 passed, 2 skipped' in env and 'GCC/13.3.0' in env
    dec = (out / 'DECISIONS_package3.md').read_text()
    assert '## 2026-10-08 -- Package 3, batch 0' in dec and 'Package 2, Stage 2' not in dec.split('---', 1)[1]
    prov = json.loads((out / 'provenance.json').read_text())
    assert prov['runs']['commit'] == _git('rev-parse', 'HEAD~1') and len(prov['commits_between']) == 1
    m.assemble(g, out, p2, extra)                                                  # a second time: Grace's verdict kept
    l1 = json.loads((out / 'P3_4_burgers_large_P' / 'checks_item4.json').read_text())['L1']
    assert not l1['grace']['passed'] and 'grace' not in l1['grace']


@needs_git
def test_the_assembly_refuses_runs_from_another_line(tmp_path):
    g, p2, extra = _fake_grace(tmp_path, '0' * 40)
    with pytest.raises(SystemExit):
        m.assemble(g, tmp_path / 'out', p2, extra)
