"""Every production run saves its trained model, and the saved model reloads
to the same solution (``lilq.saved_models``)."""

import csv
import json

import numpy as np
import pytest
import torch

import experiments.b8_initial_guess as b8
import experiments.component_b as cb
import experiments.component_c as cc
import experiments.four_method_tables as fmt
from lilq.four_method_log import FourMethodLogger
from lilq.saved_models import (
    evaluate_field, load_f1, load_f2, load_network, load_solution, predict_f2, save_network, save_solution,
)


@pytest.fixture(autouse=True)
def float64_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _last_row(run_dir):
    with open(run_dir / 'iterations.csv', newline='') as f:
        return list(csv.DictReader(f))[-1]


def test_solution_round_trip(tmp_path):
    from lilq.basis import create_basis_2d
    basis = create_basis_2d('chebyshev', 5, 4, (0, 1), (0, 2))
    coef = np.random.default_rng(0).standard_normal(basis.n_basis)
    save_solution(tmp_path, {'u': (basis, coef)}, config={'a': 1}, extra={'note': 'x'})
    sol = load_solution(tmp_path)
    x, y = np.random.default_rng(1).uniform(0, 1, (2, 7, 3))
    np.testing.assert_array_equal(evaluate_field(sol, 'u', x, y),
                                  (basis.evaluate(x.ravel(), y.ravel()) @ coef).reshape(x.shape))
    assert sol['config'] == {'a': 1} and sol['extra'] == {'note': 'x'}
    assert not list(tmp_path.glob('*.tmp'))


def test_network_round_trip_leaves_the_original_in_place(tmp_path):
    net = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1))
    save_network(tmp_path, net, config={'P': 5})
    loaded = load_network(tmp_path)
    xy = torch.rand(4, 2)
    torch.testing.assert_close(loaded['model'](xy), net(xy), rtol=0, atol=0)
    assert loaded['config'] == {'P': 5} and next(net.parameters()).requires_grad


def test_component_b_solutions_reproduce_the_logged_test_errors(tmp_path):
    from problems.bratu import make_test_error_fn as bratu_test
    from problems.kovasznay import KovasznayPhysics, make_test_error_fn as kov_test
    root = tmp_path / 'B'
    runs = cb.build_runs(benchmarks=('bratu', 'kovasznay', 'elasticity', 'beltrami'),
                         passes=('paper',), devices=('cpu',), smoke=True)
    assert [cb.execute_run(r, root, verbose=False) for r in runs] == ['ok'] * len(runs)
    for r in runs:
        sol = load_solution(root / r.name)
        for entry in sol['fields'].values():
            assert entry['coefficients'].shape == (entry['basis'].n_basis,)
    bratu = next(r for r in runs if r.benchmark == 'bratu')
    sol = load_solution(root / bratu.name)
    eps = bratu_test(sol['fields']['u']['basis'], sol['config'])(sol['fields']['u']['coefficients'])
    assert eps['eps_u'] == pytest.approx(float(_last_row(root / bratu.name)['eps_u']), rel=1e-12)
    kov = next(r for r in runs if r.benchmark == 'kovasznay')
    sol = load_solution(root / kov.name)
    f = sol['fields']
    eps = kov_test(KovasznayPhysics(sol['config']), f['u']['basis'], f['v']['basis'], f['p']['basis'])(
        np.concatenate([f[k]['coefficients'] for k in 'uvp']))
    assert eps['eps_u'] == pytest.approx(float(_last_row(root / kov.name)['eps_u']), rel=1e-12)


def test_four_method_sweep_saves_every_model_before_its_row(tmp_path):
    from problems.bratu import BratuConfig, BratuOptConfig, run_lil_n, run_nil_n
    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = fmt._apply_quick_budgets(BratuOptConfig())
    logger, csv_path = FourMethodLogger(), tmp_path / 'four_method_tables.csv'
    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'NiL-N', run_nil_n, seeds=[0],
                     devices=[torch.device('cpu')], verbose=False, csv_path=csv_path)
    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'LiL-N', run_lil_n, seeds=None,
                     devices=[torch.device('cpu')], verbose=False, csv_path=csv_path)
    nil = load_network(tmp_path / 'models' / 'bratu_P9_NiL-N_s0_cpu')
    assert nil['config'].init_seed == 0
    assert torch.isfinite(nil['model'](torch.rand(3, 2))).all()
    lil = load_solution(tmp_path / 'models' / 'bratu_P9_LiL-N_sna_cpu')
    assert lil['fields']['u']['coefficients'].shape == (9,)


def test_b8_saves_every_model(tmp_path):
    b8.run_b8(tmp_path, quick=True, verbose=False, cases=('gravity',), guesses=('ic',), sizes=(8,),
              methods=('LiL-Q', 'LiL-N', 'NiL-N'), seeds=(0,))
    models = tmp_path / 'models'
    assert load_solution(models / 'gravity_ic_P64_LiL-Q_sna')['config'].initial_guess == 'ic'
    assert (models / 'gravity_ic_P64_LiL-N_sna' / 'solution.pt').exists()
    assert (models / 'gravity_ic_P64_NiL-N_s0' / 'network.pt').exists()


def test_component_c_runs_save_their_solutions(tmp_path):
    cc.run_sweep(tmp_path, benchmarks=(('bratu', 10),), ratios=(3,), distributions=(('random', 0),),
                 quick=True, verbose=False)
    (run_dir,) = [d for d in (tmp_path / 'runs').iterdir() if d.is_dir()]
    assert load_solution(run_dir)['fields']['u']['coefficients'].shape == (100,)


def test_basis_study_saves_each_basis(tmp_path):
    from experiments.run_burgers_basis_comparison import ComparisonConfig, run_table3_study
    config = ComparisonConfig(N_x=4, N_t=4, disable_stopping_rule=True, max_quasi_iters=2)
    run_table3_study(config, basis_keys=['cheb_cheb'], verbose=False, run_root=tmp_path)
    sol = load_solution(tmp_path / 'cheb_cheb')
    assert sol['extra']['basis_key'] == 'cheb_cheb'
    assert (tmp_path / 'cheb_cheb' / 'iterations.csv').exists()


def test_f1_and_f2_models_reload(tmp_path):
    import baselines.f1_pinn as f1
    import baselines.lm_kovasznay as lm
    from argparse import Namespace
    cfg = dict(id='tiny', family='F1', width=8, depth=2, trunk='shared', m=4, sigma_ff=1.0, bc='hard',
               lambda_bc=None, balancing=False, eta=1e-3, t_adam=5, n_int=32, resample=False)
    run = f1.f1_train(cfg, 0, 60, tmp_path / 'f1', max_adam_iters=5, max_lbfgs_calls=1)
    model, run_json = load_f1(tmp_path / 'f1')
    assert run_json['final_loss'] == run['final_loss']
    eu, _, _, _ = f1.test_errors(model, torch.device('cpu'), torch.float64)
    assert eu == pytest.approx(run['eps_u'], rel=1e-12)

    args = Namespace(width=8, depth=2, m=4, sigma_ff=1.0, n_int=40, w_int=1.0, w_pin=1.0, mu0=1e-3,
                     diag_floor=1e-12, budget_s=60, max_steps=3, max_params=20000, chunk=64, test_every=1,
                     seed=0, device='cpu', out=str(tmp_path / 'f2'))
    run = lm.lm_train(args)
    model, run_json = load_f2(tmp_path / 'f2')
    theta = torch.load(tmp_path / 'f2' / 'theta.pt')
    res = lm.Residual(model, torch.rand(5, 2), torch.tensor([lm.X0, lm.Y0]))
    eu, _, _, _ = lm.test_errors(model, res, theta, torch.device('cpu'))
    assert eu == pytest.approx(run['eps_u'], rel=1e-12)
    xy = torch.tensor([[0.2, 0.3], [lm.X0, 0.4]])
    uvp = predict_f2(model, xy)
    torch.testing.assert_close(uvp[1, :2], torch.stack(lm.exact(xy[1, 0], xy[1, 1])[:2]), rtol=0, atol=1e-12)


@pytest.fixture(scope='module')
def s1():
    from problems.darcy import DarcyConfig, DarcyPhysics
    return DarcyPhysics(DarcyConfig(perm_file='perm_field_S1.txt'), verbose=False)


def test_darcy_training_resumes_exactly_from_a_checkpoint(tmp_path, s1):
    from problems.darcy import DarcyPINN

    def pinn():
        return DarcyPINN(s1.config, s1, hidden_dim=8, num_layers=2, device='cpu', seed=1)

    straight = pinn()
    straight.train(max_epochs=8, log_every=2, verbose=False)

    crashed, ckpt, calls = pinn(), tmp_path / 'checkpoint.pt', {'n': 0}
    compute = crashed._compute_loss

    def crash_at_epoch_6():
        calls['n'] += 1
        if calls['n'] == 7:
            raise RuntimeError('simulated crash')
        return compute()
    crashed._compute_loss = crash_at_epoch_6
    with pytest.raises(RuntimeError):
        crashed.train(max_epochs=8, log_every=2, verbose=False, checkpoint_path=ckpt, checkpoint_every=3)

    resumed = pinn()
    out = resumed.train(max_epochs=8, log_every=2, verbose=False, checkpoint_path=ckpt, checkpoint_every=3)
    assert out['resumed_at'] == [3]
    for name in ('net_P', 'net_U', 'net_V'):
        for a, b in zip(getattr(straight, name).parameters(), getattr(resumed, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_darcy_network_is_saved_reloaded_and_not_retrained(tmp_path, s1, monkeypatch):
    import problems.darcy as darcy
    out = darcy.run_nil_n_darcy(s1.config, s1, max_epochs=3, hidden_dim=8, num_layers=2, device='cpu',
                                verbose=False, seed=0, model_dir=tmp_path)
    assert (tmp_path / 'network.pt').exists() and not (tmp_path / 'checkpoint.pt').exists()
    pinn, saved = darcy.load_darcy_pinn(tmp_path, physics=s1)
    np.testing.assert_array_equal(pinn.predict(np.array([10.0]), np.array([20.0]))['P'],
                                  out['pinn'].predict(np.array([10.0]), np.array([20.0]))['P'])
    monkeypatch.setattr(darcy.DarcyPINN, 'train', lambda *a, **k: pytest.fail('retrained'))
    again = darcy.run_nil_n_darcy(s1.config, s1, max_epochs=3, hidden_dim=8, num_layers=2, device='cpu',
                                  verbose=False, seed=0, model_dir=tmp_path)
    np.testing.assert_array_equal(again['fields']['P'], out['fields']['P'])
    assert again['final_loss'] == out['final_loss']
