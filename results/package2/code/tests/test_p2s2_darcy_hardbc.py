"""Package 2, item 5 (P2-15): the Darcy network with hard Dirichlet conditions.
The lifting, the paper's initialization, the loss as r . r (check C3), the
Jacobian, a run, the rows, and the FASTER job and its submission."""

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import experiments.p2_15_darcy_hardbc as m
from problems.darcy import DarcyPhysics, DarcyPINN

REPO = Path(__file__).resolve().parents[1]
P2 = REPO / 'scripts' / 'cluster' / 'package2'


@pytest.fixture(scope='module')
def s1():
    config = m.config_for('S1')
    return config, DarcyPhysics(config, verbose=False)


def test_the_lifting_meets_the_dirichlet_data_exactly(s1):
    config, physics = s1
    pinn = m.HardBCDarcyPINN(config, physics, device='cpu', seed=0)
    x = torch.linspace(0, physics.LX, 9, dtype=torch.float64)[:, None]
    bottom = pinn._get_P(x, torch.zeros_like(x))
    top = pinn._get_P(x, torch.full_like(x, physics.LY))
    assert torch.all(bottom == config.P_BOTTOM) and torch.all(top == config.P_TOP)
    mid = pinn._get_P(x, torch.full_like(x, physics.LY / 2))
    assert not torch.allclose(mid, torch.full_like(mid, (config.P_TOP + config.P_BOTTOM) / 2))   # the network acts
    # the paper's output gain: at mid-height (omega = 1) one unit of NN moves P by P_HALF, as in DarcyPINN
    paper = DarcyPINN(config, physics, device='cpu', seed=0)
    nn_mid = pinn.net_P(torch.cat(pinn._norm_input(x, torch.full_like(x, physics.LY / 2)), dim=1))
    assert torch.allclose(mid - (config.P_TOP + config.P_BOTTOM) / 2, nn_mid * pinn.P_HALF, rtol=1e-13, atol=1e-9)
    assert torch.allclose(paper._get_P(x, torch.full_like(x, physics.LY / 2)) - paper.P_MID, nn_mid * paper.P_HALF)


def test_the_papers_networks_and_initialization(s1):
    config, physics = s1
    paper = DarcyPINN(config, physics, device='cpu', seed=1).network_state()
    hard = m.HardBCDarcyPINN(config, physics, device='cpu', seed=1).network_state()
    for net in ('net_P', 'net_U', 'net_V'):
        assert all(torch.equal(paper[net][k], hard[net][k]) for k in paper[net])
    assert isinstance(m.HardBCDarcyPINN(config, physics, device='cpu', seed=1).net_P.network[1], torch.nn.SiLU)
    assert sum(q.numel() for n in ('net_P', 'net_U', 'net_V') for q in paper[n].values()) == 3555


def test_c3_the_loss_is_the_papers_without_its_dirichlet_terms(s1):
    config, physics = s1
    pinn = m.HardBCDarcyPINN(config, physics, device='cpu', seed=2)
    res = m.DarcyResidual(pinn)
    r = res.vector(res.theta0())
    loss, parts = pinn._compute_loss()
    assert parts['bc_bot'] == 0 and parts['bc_top'] == 0
    assert float(r @ r) == pytest.approx(float(loss.detach()), rel=1e-12)
    logged = res.log_values(res.theta0())
    assert float(logged['darcy_y']) == pytest.approx(parts['darcy_y'], rel=1e-12)
    assert float(logged['bc_lr']) == pytest.approx(parts['bc_lr'], rel=1e-12)
    assert len(r) == 3 * 60 * 220 + 2 * 220


def test_the_jacobian_against_reverse_mode(s1):
    config, physics = s1
    pinn = m.HardBCDarcyPINN(config, physics, device='cpu', seed=0)
    res = m.DarcyResidual(pinn)
    idx = torch.arange(0, res.n_int, 700)                   # a few points of the grid, every row type
    res.X_int, res.sqrt_K = res.X_int[idx], res.sqrt_K[idx]
    res.X_lat = [X[:7] for X in res.X_lat]
    theta = res.theta0()
    J = res.jacobian(theta)
    J_ref = torch.func.jacrev(res.vector)(theta)
    assert J.shape == J_ref.shape and torch.allclose(J, J_ref, rtol=1e-10, atol=1e-12)


def test_a_run_and_the_rows(tmp_path):
    r = m.run('S1', 0, tmp_path, device='cpu', max_iterations=1)
    d = tmp_path / 'P2_15_darcy_hardbc' / 'S1_seed0'
    for f in ('run.json', 'log.csv', 'pressure_field.npz', 'network.pt', 'hardware.json'):
        assert (d / f).exists(), f
    assert r['c3']['passed'] and r['k_stop'] == 1 and r['reason'] == 'iteration_cap'
    with open(d / 'log.csv') as f:
        log = list(csv.DictReader(f))
    assert list(log[0]) == list(m.LOG_COLUMNS) and len(log) == 2
    assert float(log[1]['loss']) < float(log[0]['loss']) and float(log[-1]['eps_ref']) == pytest.approx(r['delta_FV'])
    z = np.load(d / 'pressure_field.npz')
    assert z['P'].shape == z['P_fv'].shape == (60, 220)
    # a package1 stand-in: the paper's soft-BC Adam NiL and LiL rows beside the run
    p1 = tmp_path / 'p1' / 'B_instrumentation' / 'darcy_fv' / 'S1_s0'
    p1.mkdir(parents=True)
    with open(p1 / 'darcy_fv_comparison.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['field', 'method', 'seed', 'delta_fv', 'dtype'])
        w.writeheader()
        w.writerows([{'field': 'S1', 'method': 'LiL', 'seed': '', 'delta_fv': 1.4e-4, 'dtype': 'float64'},
                     {'field': 'S1', 'method': 'NiL', 'seed': 0, 'delta_fv': 2.1e-2, 'dtype': 'float64'},
                     {'field': 'S1', 'method': 'NiL', 'seed': 0, 'delta_fv': 9.9e-2, 'dtype': 'float32'}])
    rows = m.summarize(tmp_path, tmp_path / 'p1')
    with open(tmp_path / 'P2_15_darcy_hardbc' / 'darcy_hardbc_rows.csv') as f:
        header = next(csv.reader(f))
    assert header[:8] == list(m.ROW_COLUMNS)
    assert rows[0]['delta_FV_paper_nil_soft_bc_adam'] == 2.1e-2 and rows[0]['delta_FV_lil'] == 1.4e-4
    assert json.loads((d / 'run.json').read_text())['differs_from_the_paper_nil'][1].startswith('Levenberg')


def test_the_faster_job():
    text = (P2 / 'p2s2_darcy_hardbc.slurm').read_text()
    assert '# lilq-resources: shared-gpu' in text and '#SBATCH --array=0-11' in text
    assert 'p2_15_darcy_hardbc.py run --array-task "$SLURM_ARRAY_TASK_ID"' in text and '--device cuda' in text
    h, mnt = (int(x) for x in text.split('#SBATCH --time=')[1][:5].split(':'))
    assert 35 <= 60 * h + mnt <= 60                               # the 30-min cap, set-up and the per-iteration delta_FV
    assert [m.FIELDS[t // 3] for t in range(12)].count('SPE10') == 3


def _fake_sbatch(tmp_path, dry):
    fake = tmp_path / 'sbatch'
    if dry:
        fake.write_text('#!/bin/bash\nfor a in "$@"; do [[ "$a" == --test-only ]] && { echo "$*" >> "$(dirname "$0")/log"; '
                        'echo "sbatch: Job 1 to start at soon" >&2; exit 0; }; done\nexit 1\n')
    else:
        fake.write_text('#!/bin/bash\nn=$(cat "$(dirname "$0")/n" 2>/dev/null || echo 100); echo $((n+1)) > "$(dirname "$0")/n"\n'
                        'echo "$*" >> "$(dirname "$0")/log"; echo $((n+1))\n')
    fake.chmod(0o755)
    return dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", LILQ_ACCOUNT='000000000000', YES='1',
                LILQ_RESULTS=str(tmp_path / 'results'), **({'DRY_RUN': '1'} if dry else {}))


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_faster_submission_chain(tmp_path):
    out = subprocess.run(['bash', str(P2 / 'submit_p2s2_faster.sh')], env=_fake_sbatch(tmp_path, False),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    log = (tmp_path / 'log').read_text().splitlines()
    assert len(log) == 3 and all('--partition=gpu' in l for l in log[:2])
    assert 'p2s2_preflight' in log[0] and 'p2s2_darcy_hardbc' in log[1] and '--dependency=afterok:101' in log[1]
    assert 'p2s2_report' in log[2] and '--dependency=afterany:102' in log[2]
    assert '--partition=cpu' in log[2]                            # FASTER's CPU partition (Grace's is medium)
    assert 'faster' in out.stdout.lower()


@pytest.mark.skipif(os.name == 'nt', reason='runs the bash submission script')
def test_faster_dry_run(tmp_path):
    out = subprocess.run(['bash', str(P2 / 'submit_p2s2_faster.sh')], env=_fake_sbatch(tmp_path, True),
                         capture_output=True, text=True)
    assert out.returncode == 0 and 'DRY RUN OK' in out.stdout, out.stdout + out.stderr
    assert len((tmp_path / 'log').read_text().splitlines()) == 3


# ---------------------------------------------------------------- item 5b, the control (soft rows + LM)

def test_5b_c3_the_loss_is_the_papers_with_its_dirichlet_terms(s1):
    config, physics = s1
    pinn = DarcyPINN(config, physics, device='cpu', seed=0)
    res = m.DarcyResidual(pinn, soft=True)
    r = res.vector(res.theta0())
    loss, parts = pinn._compute_loss()
    assert parts['bc_bot'] > 0 and parts['bc_top'] > 0                      # the paper's rows are in the loss
    assert float(r @ r) == pytest.approx(float(loss.detach()), rel=1e-12)
    logged = res.log_values(res.theta0())
    assert list(res.log_columns) == list(m.SOFT_LOG_COLUMNS)
    assert float(logged['bc_bot']) == pytest.approx(parts['bc_bot'], rel=1e-12)
    assert float(logged['bc_top']) == pytest.approx(parts['bc_top'], rel=1e-12)
    assert len(r) == 3 * 60 * 220 + 2 * 60 + 2 * 220
    # the paper's pressure, not the lifted one
    x = torch.linspace(0, physics.LX, 5, dtype=torch.float64)[:, None]
    X = torch.cat([x, torch.zeros_like(x)], 1)
    assert torch.allclose(res.pressure(res.theta0(), X), pinn._get_P(x, torch.zeros_like(x))[:, 0])


def test_5b_the_jacobian_against_reverse_mode(s1):
    config, physics = s1
    res = m.DarcyResidual(DarcyPINN(config, physics, device='cpu', seed=0), soft=True)
    idx = torch.arange(0, res.n_int, 700)
    res.X_int, res.sqrt_K = res.X_int[idx], res.sqrt_K[idx]
    res.X_lat = [X[:7] for X in res.X_lat]
    res.X_dir = [X[:9] for X in res.X_dir]
    theta = res.theta0()
    J, J_ref = res.jacobian(theta), torch.func.jacrev(res.vector)(theta)
    assert J.shape == J_ref.shape and torch.allclose(J, J_ref, rtol=1e-10, atol=1e-12)


def test_5b_a_run_and_both_summaries(tmp_path):
    r = m.run('S1', 0, tmp_path, device='cpu', max_iterations=1, variant='soft')
    d = tmp_path / 'P2_15b_darcy_softbc_lm' / 'S1_seed0'
    assert (d / 'run.json').exists() and (d / 'network.pt').exists()
    assert r['c3']['passed'] and r['item'] == '5b' and r['differs_from_the_paper_nil'] == ['Levenberg-Marquardt instead of Adam']
    with open(d / 'log.csv') as f:
        assert next(csv.reader(f)) == list(m.SOFT_LOG_COLUMNS)
    h = m.run('S1', 0, tmp_path, device='cpu', max_iterations=1)                  # item 5, the same field and seed
    soft_rows = m.summarize(tmp_path, variant='soft')
    hard_rows = m.summarize(tmp_path)
    assert soft_rows[0]['delta_FV_hard_bc_lm_5'] == h['delta_FV']
    assert hard_rows[0]['delta_FV_soft_bc_lm_5b'] == r['delta_FV']
    assert (tmp_path / 'P2_15b_darcy_softbc_lm' / 'darcy_softbc_lm_rows.csv').exists()


def test_the_5b_job_on_grace():
    text = (P2 / 'p2s2_darcy_softbc_lm.slurm').read_text()
    assert '# lilq-resources: shared-gpu' in text and '#SBATCH --array=0-3' in text
    assert '--variant soft --array-task "$SLURM_ARRAY_TASK_ID"' in text and '--device cuda' in text
    assert '/p2s2_darcy_softbc_lm.slurm' in (P2 / 'submit_p2s2.sh').read_text()
    assert 'p2s2_darcy_softbc_lm' not in (P2 / 'submit_p2s2_faster.sh').read_text()
    assert [m.FIELDS[t // len(m.VARIANTS['soft']['seeds'])] for t in range(4)] == list(m.FIELDS)
