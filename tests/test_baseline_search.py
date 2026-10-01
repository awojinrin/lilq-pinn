"""Component A random search (Package 1 v2.0 Section 4.3)."""

import json

import pytest

import baselines.lm_kovasznay as lm
from baselines import search


def test_draws_are_deterministic_and_inside_the_spaces():
    f1, f2 = search.draw_f1(), search.draw_f2()[0]
    assert f1 == search.draw_f1() and f2 == search.draw_f2()[0]
    assert len(f1) == len(f2) == 24
    for c in f1:
        for k, values in search.F1_SPACE.items():
            if k != 'lambda_bc':
                assert c[k] in values
        assert (c['lambda_bc'] is None) == (c['bc'] == 'hard')
    for c in f2:
        assert all(c[k] in v for k, v in search.F2_SPACE.items())
        assert c['n_theta'] <= 20000


@pytest.mark.parametrize("width, depth, m", [(32, 3, 32), (100, 4, 64), (24, 2, 16)])
def test_parameter_count_matches_the_network(width, depth, m):
    model = lm.FourierMLP(width, depth, m, 1.0, 0)
    assert search.fourier_mlp_params(width, depth, m) == sum(p.numel() for p in model.parameters())


def test_saved_search_is_never_silently_replaced(tmp_path):
    search.save_search(tmp_path)
    path = tmp_path / 'F1_configs.json'
    first = path.read_text()
    search.save_search(tmp_path)                       # same draws: left as is
    assert path.read_text() == first
    altered = json.loads(first)
    altered['configs'][0]['width'] = 999
    path.write_text(json.dumps(altered))
    with pytest.raises(FileExistsError):
        search.save_search(tmp_path)


def test_saved_lists_have_24_distinct_configurations(tmp_path):
    """The advisor's reply to wave 1, item 2.1: wave 1's F2 list had 17
    distinct of 24. Duplicates are redrawn from the same generator."""
    search.save_search(tmp_path)
    for family in ('F1', 'F2'):
        configs = json.loads((tmp_path / f'{family}_configs.json').read_text())['configs']
        keys = {json.dumps({k: v for k, v in c.items() if k != 'id'}, sort_keys=True) for c in configs}
        assert len(configs) == len(keys) == 24, family
    f2 = json.loads((tmp_path / 'F2_configs.json').read_text())
    assert f2['redrawn_duplicates'] > 0 and f2['seed'] == 12345


def test_f2_redraw_keeps_the_draws_before_the_first_duplicate():
    """Up to the first repeat, the list is wave 1's (same generator)."""
    wave1_first_11 = [(32, 3, 64, 1.0, 8000), (32, 4, 32, 2.0, 2000), (32, 3, 64, 2.0, 2000),
                      (64, 3, 32, 1.0, 2000), (64, 4, 32, 0.5, 2000), (32, 3, 32, 0.5, 8000),
                      (64, 3, 64, 1.0, 8000), (32, 4, 64, 1.0, 2000), (32, 3, 64, 0.5, 8000),
                      (64, 3, 32, 0.5, 8000), (64, 3, 64, 0.5, 8000)]
    configs = search.draw_f2()[0]
    assert [search.config_key(c) for c in configs[:11]] == wave1_first_11


def test_f2_draw_stops_when_the_cap_leaves_too_few_configurations():
    with pytest.raises(ValueError, match='satisfy'):
        search.draw_f2(max_params=5000)


def test_selection_takes_three_distinct_configurations(tmp_path):
    """Even from a list with repeats, `select` returns three different
    configurations, ranked by the screening runs' validation residual."""
    import experiments.component_a as ca
    same = {'family': 'F2', 'width': 32, 'depth': 3, 'm': 32, 'sigma_ff': 1.0, 'n_int': 2000, 'n_theta': 4291}
    configs = [{'id': 'F2_00', **same}, {'id': 'F2_01', **same},
               {'id': 'F2_02', **same, 'width': 64}, {'id': 'F2_03', **same, 'depth': 4},
               {'id': 'F2_04', **same, 'm': 64}]
    (tmp_path / 'search').mkdir()
    (tmp_path / 'search' / 'F2_configs.json').write_text(json.dumps({'configs': configs}))
    for c, loss in zip(configs, (0.1, 0.2, 0.3, 0.4, 0.05)):
        d = tmp_path / 'screening' / f"{c['id']}_s0"
        d.mkdir(parents=True)
        (d / 'run.json').write_text(json.dumps({'final_loss': loss, 'val_residual': loss}))
    assert ca.select(tmp_path, 'F2')['top'] == ['F2_04', 'F2_00', 'F2_02']
