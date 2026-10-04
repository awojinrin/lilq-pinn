"""experiments/figure_data.py (the advisor's reply to wave 4, Section 3): the
solution-field data are evaluations of the saved models, with their sources,
the GPU model used and listed where a CPU one is missing, and the
evaluation's round-off recorded. On the real final package the script's own
checks reproduce the logged errors: Darcy's delta_FV of every LiL and NiL
model, Kovasznay's and Buckley-Leverett's LiL-Q test errors (DECISIONS.md)."""

import json

import numpy as np
import torch

import experiments.figure_data as fd
from lilq.basis import create_basis_2d
from lilq.nn import MLP
from lilq.saved_models import save_network, save_solution
from problems.bratu import BratuConfig


def _package(tmp_path):
    B = tmp_path / 'B_instrumentation'
    cfg = BratuConfig()
    rng = np.random.default_rng(0)
    for P, n in ((25, 5), (225, 15)):
        basis = create_basis_2d(cfg.basis_type, n, n, cfg.x_domain, cfg.y_domain)
        save_solution(B / f'bratu_P{P}_cpu_paper', {'u': (basis, rng.standard_normal(basis.n_basis) * 1e-2)}, cfg)
        (B / f'bratu_P{P}_cpu_paper' / 'summary.json').write_text(json.dumps({'commit': 'abc1234'}))
        # P = 225's four-method models on the CPU; P = 25's only on the GPU (as in wave 1)
        dev, tag = ('cpu', 'cpu') if P == 225 else ('gpu', 'cuda')
        job = B / 'four_method_jobs' / f'bratu_{dev}'
        job.mkdir(parents=True, exist_ok=True)
        (job / 'hardware.json').write_text(json.dumps({'git': {'commit': 'def5678'}}))
        save_solution(job / 'models' / f'bratu_P{P}_LiL-N_sna_{tag}',
                      {'u': (basis, rng.standard_normal(basis.n_basis) * 1e-2)}, cfg)
        for m in ('NiL-N', 'NiL-Q'):
            torch.manual_seed(P)
            save_network(job / 'models' / f'bratu_P{P}_{m}_s0_{tag}', MLP(hidden_dim=8, num_layers=2), cfg)
    return tmp_path


def test_bratu_fields_sources_and_the_missing_cpu_models(tmp_path):
    ctx = fd.Context(_package(tmp_path))
    out = tmp_path / 'figure_data'
    out.mkdir()
    entry = fd.bratu(ctx, out)
    z = np.load(out / 'bratu.npz')
    assert z['x'].shape == z['y'].shape == (201,) and z['P25_NiL-Q'].shape == (201, 201)
    meta = json.loads(str(z['meta']))
    assert meta['sources']['P225_NiL-N']['device'] == 'cpu' and meta['sources']['P225_NiL-N']['commit'] == 'def5678'
    assert meta['sources']['P25_LiL-Q']['commit'] == 'abc1234'
    assert meta['sources']['P25_NiL-N']['device'] == 'cuda' and 'no CPU model' in meta['sources']['P25_NiL-N']['note']
    assert {m['method'] for m in ctx.missing} == {'LiL-N', 'NiL-N', 'NiL-Q'} and all(m['used_instead'] for m in ctx.missing)
    assert max(meta['evaluation_roundoff'].values()) < 1e-12
    # the saved network's own output, unchanged
    from lilq.saved_models import load_network
    model = load_network(tmp_path / 'B_instrumentation/four_method_jobs/bratu_cpu/models/bratu_P225_NiL-N_s0_cpu')['model']
    with torch.no_grad():
        u = model(torch.tensor([[0.5, 0.25]], dtype=torch.float64)).item()
    assert abs(z['P225_NiL-N'][100, 50] - u) <= 1e-15 * abs(u)   # one point vs a batch: round-off
    assert entry['file'] == 'bratu.npz'


def test_tensor_and_pointwise_evaluations_agree_and_the_difference_is_recorded(tmp_path):
    ctx = fd.Context(_package(tmp_path))
    from lilq.saved_models import load_solution
    sol = load_solution(tmp_path / 'B_instrumentation' / 'bratu_P225_cpu_paper')
    a = np.linspace(0, 1, 11)
    v = ctx.field('label', sol, 'u', [a, a])
    assert v.shape == (11, 11) and ctx.roundoff['label'] < 1e-13
