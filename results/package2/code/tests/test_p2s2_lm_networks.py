"""Package 2, item 2 (P2-8): NiL-N trained by Levenberg-Marquardt. The
Taylor-mode residual and its Jacobian against autograd, the stopping rules,
and check C3 on every benchmark."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import experiments.p2_8_lm_networks as m

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _model(seed=0, width=6):
    from lilq.nn import MLP
    torch.manual_seed(seed)
    return MLP(hidden_dim=width, num_layers=2)


@pytest.mark.parametrize('bench', m.BENCHMARKS)
def test_taylor_residual_matches_the_package1_residual(bench):
    mod, config, _ = m.setup(bench, 25)
    model = _model()
    net = m.Net(model)
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    x = torch.rand(40, 1, requires_grad=True)
    y = torch.rand(40, 1, requires_grad=True)
    if bench == 'bratu':
        ref = mod._compute_pde_residual_nn(model, x, y, config.lambda_)
    elif bench == 'burgers':
        ref = mod._compute_pde_residual_nn(model, x, y, config.viscosity)
    else:
        ref = mod._compute_pde_residual_nn(model, x, y, mod.BLPhysics(config))
    u, du, ddu = net.fields(theta, torch.cat([x, y], 1).detach())
    np.testing.assert_allclose(m.pde_function(bench, config)(u, du, ddu).numpy(), ref.detach().numpy().ravel(),
                               rtol=1e-12, atol=1e-12)


def test_jacobian_against_autograd():
    _, config, _ = m.setup('burgers', 25)
    model = _model(1)
    net = m.Net(model)
    X = torch.rand(15, 2)
    blocks = [('initial', 'ic', torch.rand(4, 2), torch.rand(4)), ('left', 'bc', torch.rand(3, 2), None)]
    res = m.Residual(net, m.pde_function('burgers', config), X, blocks, (1.0, 10.0, 10.0))
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    J_ref = torch.autograd.functional.jacobian(res.vector, theta)
    np.testing.assert_allclose(res.jacobian(theta).numpy(), J_ref.numpy(), rtol=1e-10, atol=1e-12)


class _Flat:
    """A residual no step can lower: r = (1,) for every theta."""

    def vector(self, theta):
        return torch.ones(1) + 0 * theta.sum()

    def jacobian(self, theta):
        return torch.ones(1, theta.numel())

    def components(self, theta):
        return (1.0,) * 6


def test_lm_stall_rule_restarts_once(tmp_path):
    theta, s = m.lm(_Flat(), torch.zeros(3), 1e-9, lambda th: 0.0, tmp_path / 'log.csv')
    assert s['stopping_reason'] == 'optimizer_stall' and s['lm_restarts'] == 1 and s['total_iterations'] == 2
    with open(tmp_path / 'log.csv') as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(m.LOG_COLUMNS) and len(rows) == 3


def test_lm_caps(tmp_path):
    A = torch.randn(30, 4, generator=torch.Generator().manual_seed(0))
    b = torch.randn(30, generator=torch.Generator().manual_seed(1))

    class Lin(_Flat):
        def vector(self, theta):
            return A @ theta - b

        def jacobian(self, theta):
            return A

    _, s = m.lm(Lin(), torch.zeros(4), 0.0, lambda th: 0.0, tmp_path / 'a.csv', max_iterations=3)
    assert s['stopping_reason'] == 'iteration_cap' and s['total_iterations'] == 3
    _, s = m.lm(Lin(), torch.zeros(4), 1e9, lambda th: 0.0, tmp_path / 'b.csv')
    assert s['stopping_reason'] == 'target' and s['total_iterations'] == 1
    _, s = m.lm(Lin(), torch.zeros(4), 0.0, lambda th: 0.0, tmp_path / 'c.csv', wall_cap_s=0.0)
    assert s['stopping_reason'] == 'wall_time_cap'


def _reference_dir(tmp_path):
    """Stand-in references on small grids (the values only enter eps_ref)."""
    x = np.linspace(0, 1, 11)
    np.savez(tmp_path / 'bratu_ref_p48.npz', x=x, y=x, u=np.outer(np.sin(np.pi * x), np.sin(np.pi * x)))
    xb, t = np.linspace(-1, 1, 11), np.linspace(0, 1, 11)
    np.savez(tmp_path / 'burgers_cole_hopf.npz', x=xb, t=t, u=-np.outer(np.sin(np.pi * xb), np.exp(-t)))
    return tmp_path


@pytest.mark.parametrize('bench', ['bratu', 'burgers'])
def test_run_records_c3_and_the_log(tmp_path, bench):
    r = m.run(bench, 25, 1, _reference_dir(tmp_path), tmp_path, max_iterations=2)
    assert r['c3']['passed'] and r['c3']['rel_diff'] <= m.C3_TOL
    assert r['stopping_reason'] in ('iteration_cap', 'target') and r['method'] == m.METHOD
    d = tmp_path / 'P2_8_lm_networks' / f'{bench}_P25_seed1'
    assert json.loads((d / 'run.json').read_text())['seed'] == 1 and (d / 'network.pt').exists()
    with open(d / 'log.csv') as fh:
        rows = list(csv.DictReader(fh))
    assert float(rows[0]['loss']) == r['c3']['lm_r_dot_r_iteration_0']
    runs, med = m.summarize(tmp_path)
    assert len(runs) == 1 and med[0]['n_seeds'] == 1
    assert med[0]['eps_ref_min'] == med[0]['eps_ref_max'] == r['eps_ref_final']
    assert (tmp_path / 'P2_8_lm_networks' / 'four_method_lm_rows.csv').exists()
    table = (tmp_path / 'P2_8_lm_networks' / 'lm_vs_lbfgs.md').read_text()
    assert f'| {bench} | 25 | cpu |' in table and 'eps_ref' in table and 'time' not in table.split('\n')[4]


def test_markers():
    assert m._markers(['target'] * 3) == ''
    assert m._markers(['target', 'wall_time_cap', 'wall_time_cap']) == 'W 1/3'
    assert m._markers(['iteration_cap'] * 2 + ['optimizer_stall']) == '*'


def test_item_2_jobs():
    for name in ('p2s2_lm_networks_bratu', 'p2s2_lm_networks_burgers', 'p2s2_lm_networks_bl'):
        text = (REPO / 'scripts' / 'cluster' / 'package2' / f'{name}.slurm').read_text()
        assert '# lilq-resources: timed-cpu' in text and 'p2_8_lm_networks.py run' in text


def test_run_stage_skips_finished_runs(tmp_path, monkeypatch, capsys):
    done = tmp_path / 'P2_8_lm_networks' / 'bratu_P25_seed0'
    done.mkdir(parents=True)
    (done / 'run.json').write_text('{}')
    calls = []
    monkeypatch.setattr(m, 'run', lambda *a, **k: calls.append(a) or {
        'stopping_reason': 'target', 'total_iterations': 1, 'training_time': 0.0, 'final_loss': 0.0,
        'eps_ref_final': 0.0, 'c3': {'rel_diff': 0.0}})
    m.main(['run', '--benchmark', 'bratu', '--P', '25', '--seeds', '0', '1', '--out', str(tmp_path)])
    assert [c[2] for c in calls] == [1] and 'done, skipping' in capsys.readouterr().out


def test_gpu_rerun_job_and_list(tmp_path):
    text = (REPO / 'scripts' / 'cluster' / 'package2' / 'p2s2_lm_networks_gpu.slurm').read_text()
    assert '# lilq-resources: timed ' in text and 'gpu-list' in text and '--device cuda' in text
    base = tmp_path / 'P2_8_lm_networks'
    for name, reason, device in (('burgers_P625_seed0', 'wall_time_cap', 'cpu'), ('burgers_P625_seed1', 'target', 'cpu'),
                                 ('bratu_P25_seed0', 'target', 'cpu'), ('bl_P1024_seed0_cuda', 'wall_time_cap', 'cuda')):
        (base / name).mkdir(parents=True)
        bench, P = name.split('_')[0], int(name.split('_')[1][1:])
        (base / name / 'run.json').write_text(json.dumps({'benchmark': bench, 'P': P, 'device': device,
                                                          'stopping_reason': reason}))
    assert m.gpu_list(tmp_path) == [('burgers', 625)]


def test_lbfgs_reference_errors(tmp_path):
    a = tmp_path / 'scalar.csv'
    a.write_text('\n'.join(['benchmark,P,method,seed,device,eps_ref_final', 'bratu,25,NiL-N,0,cuda,0.1',
                            'bratu,25,NiL-N,0,cpu,0.2', 'bratu,25,NiL-Q,0,cuda,0.3']) + '\n')
    b = tmp_path / 'bl.csv'
    b.write_text('\n'.join(['case,P,method,seed,device,eps_ref_final', 'bl,64,NiL-N,1,cuda,0.4',
                            'bl,64,LiL-N,,cuda,0.5']) + '\n')
    assert m.lbfgs_reference_errors([a, b]) == {('bratu', 25, 0): 0.1, ('bl', 64, 1): 0.4}


def test_item_2_requests_stay_within_the_cap():
    """Section 1: item 2's cap is 600 SU. Requested = walltime x rate: the three CPU
    jobs at 48 SU/h (timed-cpu) and the contingent A100 job at 192 SU/h (timed)."""
    import re
    rate = {'timed-cpu': 48, 'timed': 192}
    total = 0.0
    for name in ('bratu', 'burgers', 'bl', 'gpu'):
        text = (REPO / 'scripts' / 'cluster' / 'package2' / f'p2s2_lm_networks_{name}.slurm').read_text()
        h, mnt, _ = map(int, re.search(r'--time=(\d+):(\d+):(\d+)', text).groups())
        total += (h + mnt / 60) * rate[re.search(r'# lilq-resources: ([a-z-]+)', text).group(1)]
    assert total <= 600


def test_gpu_list_puts_the_largest_burgers_last(tmp_path):
    """The A100 job runs gpu-list in order, so a walltime end drops the largest
    Burgers reruns first (Section 5)."""
    base = tmp_path / 'P2_8_lm_networks'
    for bench, P in (('burgers', 625), ('bl', 1024), ('burgers', 225), ('bratu', 225)):
        d = base / f'{bench}_P{P}_seed0'
        d.mkdir(parents=True)
        (d / 'run.json').write_text(json.dumps({'benchmark': bench, 'P': P, 'device': 'cpu',
                                                'stopping_reason': 'wall_time_cap'}))
    assert m.gpu_list(tmp_path) == [('bl', 1024), ('bratu', 225), ('burgers', 225), ('burgers', 625)]
