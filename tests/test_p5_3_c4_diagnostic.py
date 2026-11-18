"""Package 5, item 3: the C4 diagnostic (``experiments/p5_3_c4_diagnostic.py``).

The wait policy each kind needs; a process's records; the summary's verdicts as fixed before the job."""

import json

import numpy as np
import pytest
import torch

import experiments.p5_3_c4_diagnostic as c4d


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def ref(tmp_path):
    xb, t = np.linspace(-1, 1, 11), np.linspace(0, 1, 11)
    d = tmp_path / 'ref'
    d.mkdir()
    np.savez(d / 'burgers_cole_hopf.npz', x=xb, t=t, u=-np.outer(np.sin(np.pi * xb), np.exp(-t)))
    return d


def test_the_kinds_and_their_wait_policy():
    assert c4d.KINDS['active'] == ('numpy', 'ACTIVE') and c4d.EXPECTED_LIBRARY['active'] == 'module'
    c4d.check_wait_policy('numpy', {})
    c4d.check_wait_policy('torch', {})
    c4d.check_wait_policy('active', {'OMP_WAIT_POLICY': 'ACTIVE'})
    with pytest.raises(SystemExit, match='needs OMP_WAIT_POLICY=ACTIVE'):
        c4d.check_wait_policy('active', {})
    with pytest.raises(SystemExit, match=r'needs OMP_WAIT_POLICY=\(unset\), not ACTIVE'):
        c4d.check_wait_policy('numpy', {'OMP_WAIT_POLICY': 'ACTIVE'})
    with pytest.raises(SystemExit, match='not PASSIVE'):
        c4d.check_wait_policy('torch', {'OMP_WAIT_POLICY': 'PASSIVE'})


def test_the_constants_are_the_reruns():
    import experiments.p5_3_networks as n3
    assert (c4d.BENCH, c4d.P, c4d.METHOD, c4d.RERUN_C4_ITERATIONS, c4d.REPEATS) == ('bl', 1024, 'lm', 213, 3)
    assert c4d.RTOL == n3.C4_RTOL and c4d.WARMUP_METHODS == ('nil_n', 'lm')


def test_a_cpu_process_times_the_pair_and_records_its_state(ref, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(c4d, 'BENCH', 'burgers')
    monkeypatch.setattr(c4d, 'P', 25)
    monkeypatch.setattr(c4d, 'RERUN_C4_ITERATIONS', 3)
    monkeypatch.setattr(c4d, 'REPEATS', 2)
    monkeypatch.delenv('OMP_WAIT_POLICY', raising=False)
    out = tmp_path / 'out'
    c4d.main(['run', '--kind', 'numpy', '--round', '1', '--device', 'cpu', '--out', str(out),
              '--reference-dir', str(ref)])
    d = c4d.root_of(out) / 'cpu_numpy_r1'
    env = json.loads((d / 'environment.json').read_text())
    assert env['kind'] == 'numpy' and env['round'] == 1 and env['environment']['OMP_WAIT_POLICY'] is None
    assert json.loads((d / 'process_r1.json').read_text())['first'] == 'numpy'
    res = json.loads((d / 'result.json').read_text())
    assert res['iterations'] == 3 and res['pilot'] is None and len(res['timings']) == 2 and res['c2_passed']
    assert list((c4d.root_of(out) / 'provenance').iterdir())
    assert 'cpu_numpy_r1: 3 iterations' in capsys.readouterr().out
    rec = c4d._process(d)
    assert rec['library'] == env['library'] and len(rec['pairs']) == 2 and rec['omp_wait_policy'] is None


def test_an_active_process_stops_without_its_wait_policy(ref, tmp_path, monkeypatch):
    monkeypatch.delenv('OMP_WAIT_POLICY', raising=False)
    with pytest.raises(SystemExit, match='needs OMP_WAIT_POLICY=ACTIVE'):
        c4d.main(['run', '--kind', 'active', '--round', '0', '--device', 'cpu', '--out', str(tmp_path),
                  '--reference-dir', str(ref)])
    assert not c4d.root_of(tmp_path).exists()          # nothing written


def _fake(root, device, kind, round_, diffs, library=None, policy='default', c2=True, iterations=213):
    d = root / c4d.label_of(device, kind, round_)
    d.mkdir(parents=True)
    pol = c4d.KINDS[kind][1] if policy == 'default' else policy
    (d / 'environment.json').write_text(json.dumps(
        {'kind': kind, 'device': device, 'round': round_, 'library': library or c4d.EXPECTED_LIBRARY[kind],
         'environment': {'OMP_WAIT_POLICY': pol, 'OMP_NUM_THREADS': None}}))
    (d / 'result.json').write_text(json.dumps(
        {'iterations': iterations, 'c2_passed': c2, 'error_time_s': 3.0,
         'timings': [{'time_without_s': 10.0, 'time_with_s': 10.0 * (1 + x), 'rel_diff_with_vs_without': x}
                     for x in diffs]}))


def _all(root, module, wheel, active, gpu):
    for r in range(3):
        _fake(root, 'cpu', 'numpy', r, module)
        _fake(root, 'cpu', 'torch', r, wheel)
        _fake(root, 'cpu', 'active', r, active)
    for r in range(2):
        _fake(root, 'cuda', 'numpy', r, gpu, iterations=400)


def test_the_summary_library_dependent_leak_from_sleeping_threads_a100_clean(tmp_path):
    root = c4d.root_of(tmp_path)
    _all(root, [0.17, 0.19, 0.04], [0.02, 0.01, -0.01], [0.01, 0.00, 0.02], [0.01, -0.02, 0.00])
    res = c4d.summary(tmp_path)
    g = res['groups']
    assert g['cpu numpy (module)']['pairs'] == 9 and g['cpu numpy (module)']['median'] == pytest.approx(0.17)
    assert g['cpu numpy (module)']['verdict'] == 'leaks' and g['cpu numpy (module)']['pairs_outside_rtol'] == 6
    assert g['cpu torch (wheel)']['verdict'] == 'clean'
    assert res['findings'] == ['the leak depends on the library (module leaks, wheel clean)',
                               'ACTIVE removes it: the threads going to sleep are the cause']
    assert res['a100_set_may_go'] is True and res['records_as_expected']
    assert json.loads((root / 'summary.json').read_text())['a100_set_may_go'] is True


def test_the_summary_both_libraries_leak_and_the_a100_leaks(tmp_path):
    root = c4d.root_of(tmp_path)
    _all(root, [0.17] * 3, [0.08] * 3, [0.16] * 3, [0.09, 0.07, 0.02])
    res = c4d.summary(tmp_path)
    assert res['findings'] == ['the log leaks with either library',
                               'ACTIVE does not remove it: not the threads going to sleep']
    assert res['groups']['A100 numpy (module, two processes side by side)']['verdict'] == 'leaks'
    assert res['a100_set_may_go'] is False


def test_the_summary_not_reproduced(tmp_path):
    root = c4d.root_of(tmp_path)
    _all(root, [0.03, 0.01, 0.06], [0.0] * 3, [0.0] * 3, [0.0] * 3)
    res = c4d.summary(tmp_path)
    assert res['findings'] == ["cpu numpy is clean: the rerun's +18% was not reproduced"]
    assert res['a100_set_may_go'] is True


def _spoil(d, what):
    if what == 'library':
        e = json.loads((d / 'environment.json').read_text())
        (d / 'environment.json').write_text(json.dumps(e | {'library': {'module': 'torch', 'torch': 'module'}[e['library']]}))
    elif what == 'policy':
        e = json.loads((d / 'environment.json').read_text())
        e['environment']['OMP_WAIT_POLICY'] = 'ACTIVE'
        (d / 'environment.json').write_text(json.dumps(e))
    elif what == 'c2':
        r = json.loads((d / 'result.json').read_text())
        (d / 'result.json').write_text(json.dumps(r | {'c2_passed': False}))
    else:
        (d / 'result.json').unlink()


@pytest.mark.parametrize('what', ['library', 'policy', 'missing'])
def test_the_a100_set_waits_on_an_a100_record_out_of_place(tmp_path, what):
    root = c4d.root_of(tmp_path)
    _all(root, [0.0] * 3, [0.0] * 3, [0.0] * 3, [0.0] * 3)
    _spoil(root / 'cuda_numpy_r1', what)
    res = c4d.summary(tmp_path)
    assert res['a100_set_may_go'] is False
    if what == 'missing':
        assert res['processes_without_result'] == ['cuda_numpy_r1']
    else:
        assert res['records_as_expected'] is False


@pytest.mark.parametrize('what', ['library', 'policy', 'missing', 'c2'])
def test_a_cpu_record_out_of_place_is_reported_and_does_not_gate_the_a100_set(tmp_path, what, capsys):
    root = c4d.root_of(tmp_path)
    _all(root, [0.0] * 3, [0.0] * 3, [0.0] * 3, [0.0] * 3)
    _spoil(root / 'cpu_torch_r2', what)
    res = c4d.summary(tmp_path)
    assert res['a100_set_may_go'] is True
    c4d.print_summary(res)
    printed = capsys.readouterr().out
    assert {'library': 'RECORDS NOT AS EXPECTED', 'policy': 'RECORDS NOT AS EXPECTED',
            'missing': 'PROCESSES WITHOUT A RESULT: cpu_torch_r2', 'c2': 'C2 FAILED IN A PROCESS'}[what] in printed


def test_c2_failing_on_the_a100_is_reported_and_does_not_gate(tmp_path):
    root = c4d.root_of(tmp_path)
    _all(root, [0.0] * 3, [0.0] * 3, [0.0] * 3, [0.0] * 3)
    _spoil(root / 'cuda_numpy_r0', 'c2')
    res = c4d.summary(tmp_path)
    assert res['a100_set_may_go'] is True and res['c2_passed'] is False


def test_the_summary_of_nothing_measured(tmp_path, capsys):
    res = c4d.summary(tmp_path)
    assert all(g['verdict'] == 'not measured' for g in res['groups'].values()) and res['findings'] == []
    c4d.print_summary(res)
    assert 'A100 SET MAY GO: NO' in capsys.readouterr().out


def test_the_summary_flags_a_group_faster_with_the_log(tmp_path):
    root = c4d.root_of(tmp_path)
    _all(root, [0.0] * 3, [-0.08] * 3, [0.0] * 3, [0.0] * 3)
    res = c4d.summary(tmp_path)
    assert res['groups']['cpu torch (wheel)']['verdict'] == 'faster with the log'
    assert 'a group is more than 5% faster with the log: not foreseen, to be explained' in res['findings']


def test_a_rerun_process_removes_an_earlier_attempts_result_first(ref, tmp_path, monkeypatch):
    """A resubmitted job whose process stops (here at the library check) leaves no earlier result
    beside its new records."""
    import experiments.p5_3_openmp as om
    monkeypatch.delenv('OMP_WAIT_POLICY', raising=False)
    d = c4d.root_of(tmp_path) / 'cpu_numpy_r0'
    d.mkdir(parents=True)
    (d / 'result.json').write_text('{}')

    def stop(*a, **k):
        raise SystemExit('library check failed')
    monkeypatch.setattr(om, 'start_process', stop)
    with pytest.raises(SystemExit, match='library check failed'):
        c4d.main(['run', '--kind', 'numpy', '--round', '0', '--device', 'cpu', '--out', str(tmp_path),
                  '--reference-dir', str(ref)])
    assert not (d / 'result.json').exists()
