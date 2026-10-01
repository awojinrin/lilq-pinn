"""Component A's validation residual and the selection by it (the advisor's
reply to wave 2, 1 October 2026)."""

import json
import math

import pytest
import torch

import baselines.f1_pinn as f1
import baselines.lm_kovasznay as lm
import experiments.component_a as ca
from baselines import validation


@pytest.fixture(autouse=True)
def restore_default_dtype():
    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)


def _tiny_f1(bc):
    torch.manual_seed(0)
    return f1.F1Model(8, 2, 4, 1.0, 'shared', bc).double()


def test_protocol_points_are_fixed_and_device_independent():
    xy_int, xy_bc = validation.validation_points('cpu')
    assert xy_int.shape == (20000, 2) and xy_bc.shape == (1600, 2) and xy_int.dtype == torch.float64
    gen = torch.Generator().manual_seed(20261001)
    assert torch.equal(xy_int, f1.interior_points(20000, gen, torch.float64, torch.device('cpu')))
    assert validation.PROTOCOL['seed'] == 20261001 and validation.PROTOCOL['n_interior'] == 20000
    assert float(xy_int[:, 0].min()) >= f1.X0 and float(xy_int[:, 1].max()) <= f1.Y1


@pytest.mark.parametrize('bc', ['hard', 'soft'])
def test_residual_is_the_unweighted_loss_on_the_fresh_points(bc):
    """Chunked evaluation equals F1's own unweighted loss on the same points
    in one batch; the boundary term only for soft boundary conditions."""
    model = _tiny_f1(bc)
    terms = validation.residual_terms(model, bc == 'soft', 'cpu')
    xy_int, xy_bc = validation.validation_points('cpu')
    reference = f1.loss_terms(model, xy_int, xy_bc if bc == 'soft' else None)
    assert set(terms) == set(reference)
    assert sum(terms.values()) == pytest.approx(f1.unweighted_loss(reference), rel=1e-12)


def test_f2_fields_apply_the_hard_boundary_conditions():
    torch.set_default_dtype(torch.float64)
    model = lm.FourierMLP(8, 2, 4, 1.0, 0)
    xy = validation.validation_points('cpu')[1][:50]
    fields = validation.f2_fields(model)(xy).detach()
    from lilq.saved_models import predict_f2
    assert torch.allclose(fields, predict_f2(model, xy), rtol=0, atol=1e-14)
    ue, ve, _ = f1.exact(xy[:, 0], xy[:, 1])
    assert torch.allclose(fields[:, 0], ue, atol=1e-12) and torch.allclose(fields[:, 1], ve, atol=1e-12)


def _saved_f1_run(run_dir, bc='soft'):
    model = _tiny_f1(bc)
    run_dir.mkdir(parents=True)
    torch.save(model.state_dict(), run_dir / 'model.pt')
    config = dict(id='F1_00', family='F1', width=8, depth=2, m=4, sigma_ff=1.0, trunk='shared', bc=bc)
    (run_dir / 'run.json').write_text(json.dumps({'config': config, 'final_loss_unweighted': 1e-3}))
    return model


def test_ensure_records_once_where_the_run_was_written(tmp_path, monkeypatch):
    """A run made by this code gets the residual in run.json; an older run
    keeps its run.json and gets validation.json. Either is read back, never
    recomputed."""
    new, old = tmp_path / 'new' / 'F1_00_s0', tmp_path / 'old' / 'F1_00_s0'
    model = _saved_f1_run(new)
    _saved_f1_run(old)
    v_new = validation.ensure(new, record_in_run_json=True)
    assert json.loads((new / 'run.json').read_text())['val_residual'] == v_new
    assert not (new / validation.VALIDATION_FILE).exists()
    before = (old / 'run.json').read_text()
    v_old = validation.ensure(old)
    assert (old / 'run.json').read_text() == before and (old / validation.VALIDATION_FILE).exists()
    assert v_new == v_old == pytest.approx(sum(validation.residual_terms(model, True, 'cpu').values()), rel=1e-12)
    monkeypatch.setattr(validation, 'compute', lambda *a, **k: pytest.fail('recomputed'))
    assert validation.ensure(new) == v_new and validation.ensure(old) == v_old


def test_a_run_without_a_model_ranks_last(tmp_path):
    d = tmp_path / 'F2_00_s0'
    d.mkdir()
    (d / 'run.json').write_text(json.dumps({'config': {'family': 'F2'}, 'end_reason': 'failure'}))
    assert validation.ensure(d) == math.inf


def _screening(root, family, rows):
    """Screening runs with given (training loss, validation residual)."""
    configs = [dict(id=f'{family}_{i:02d}', family=family, width=8 + i) for i in range(len(rows))]
    (root / 'search').mkdir(parents=True, exist_ok=True)
    (root / 'search' / f'{family}_configs.json').write_text(json.dumps({'configs': configs}))
    for c, (train, val) in zip(configs, rows):
        d = root / 'screening' / f"{c['id']}_s0"
        d.mkdir(parents=True)
        key = 'final_loss_unweighted' if family == 'F1' else 'final_loss'
        (d / 'run.json').write_text(json.dumps({key: train, 'val_residual': val}))
    return [c['id'] for c in configs]


def test_selection_ranks_by_validation_and_keeps_the_training_ranking(tmp_path):
    """Wave 2's failure in miniature: the lowest training loss has the
    highest validation residual and is not selected."""
    ids = _screening(tmp_path, 'F2', [(1e-30, 3e-2), (1e-18, 1e-11), (1e-17, 4e-16), (1e-16, 3e-16), (1e-15, 5e-15)])
    sel = ca.select(tmp_path, 'F2')
    assert sel['top'] == sel['top_by_validation'] == [ids[3], ids[2], ids[4]]
    assert sel['top_by_training_loss'] == [ids[0], ids[1], ids[2]]
    assert [r['id'] for r in sel['ranking_by_training_loss']] == ids
    assert sel['validation_protocol']['seed'] == 20261001 and sel['selection_criterion'] == 'validation_residual'
    saved = json.loads((tmp_path / 'screening' / 'F2_selection.json').read_text())
    assert saved['top'] == sel['top']


def test_keep_top_keeps_the_given_finalists(tmp_path):
    """F1 in wave 4: wave 2's finalists stay; the validation top three is
    still recorded."""
    ids = _screening(tmp_path, 'F1', [(2e-7, 3.8), (3e-7, 2e-5), (6e-7, 3e-6), (1e-6, 1e-6)])
    sel = ca.select(tmp_path, 'F1', keep_top=ids[:3])
    assert sel['top'] == ids[:3] and sel['top_by_validation'] == [ids[3], ids[2], ids[1]] and 'top_kept' in sel
    with pytest.raises(ValueError, match='keep-top'):
        ca.select(tmp_path, 'F1', keep_top=['F1_99'])


def test_representative_is_the_lowest_median_validation_residual(tmp_path):
    """Existing full runs are not retrained; the representative is chosen by
    median validation residual, not by training loss."""
    ids = _screening(tmp_path, 'F1', [(2e-7, 3.8), (3e-7, 2e-5), (6e-7, 3e-6)])
    ca.select(tmp_path, 'F1', keep_top=ids)
    for cid, (train, val) in zip(ids, [(5e-8, 3.8), (2e-7, 2e-5), (2e-7, 3e-6)]):
        for s in (0, 1, 2):
            d = tmp_path / 'full' / f'{cid}_s{s}'
            d.mkdir(parents=True)
            (d / 'run.json').write_text(json.dumps({'final_loss_unweighted': train * (1 + s), 'val_residual': val,
                                                    'eps_u': val}))
    summary = ca.full(tmp_path, 'F1', 'cpu', budget_s=1, seeds=(0, 1, 2))
    assert summary['representative'] == ids[2]
    assert min(summary['median_final_loss'], key=summary['median_final_loss'].get) == ids[0]
    assert summary['representative_criterion'] == 'lowest median validation residual'
