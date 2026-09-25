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
