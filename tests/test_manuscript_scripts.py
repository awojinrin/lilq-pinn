"""The runner of the manuscript folder's scripts (Package 2, gate G1): every
path on the advisor's machine is mapped, package1 gets the corrected
Kovasznay comparison, the published waves keep the scripts' own inputs."""

from pathlib import Path

import pytest

import experiments.manuscript_scripts as ms


def _snapshot(tmp_path):
    fig = tmp_path / 'snap' / 'our_scripts' / 'figure_scripts'
    ana = tmp_path / 'snap' / 'our_scripts' / 'analysis_scripts'
    fig.mkdir(parents=True)
    ana.mkdir(parents=True)
    (fig / 'fig_compA.py').write_text("W4 = '/home/claude/w4/results/wave4/'\n"
                                      "cmp = W4 + 'fixes/option_b/section46/results/kovasznay_comparison.csv'\n"
                                      "OUT = '/home/claude/figs3/x.pdf'\n")
    (ana / 'heldout.py').write_text("B8 = '/home/claude/w3/results/wave3/B_instrumentation'\n"
                                    "OS = '/home/claude/wave1/results/wave1/C_oversampling/runs'\n")
    return tmp_path / 'snap'


def test_package_mode_maps_every_path_and_the_corrected_csv(tmp_path):
    snap, pkg, out = _snapshot(tmp_path), tmp_path / 'package1', tmp_path / 'out'
    made = ms.stage(snap, out, ms.path_map(pkg, tmp_path / 'fd', tmp_path / 'att', out))
    text = (out / 'fig_compA.py').read_text()
    assert '/home/claude' not in text and 'A_calibration/results/kovasznay_comparison.csv' in text
    assert pkg.resolve().as_posix() in (out / 'heldout.py').read_text()
    assert {m[0] for m in made} == {'fig_compA.py', 'heldout.py'}


def test_waves_mode_keeps_the_scripts_inputs(tmp_path):
    snap, out = _snapshot(tmp_path), tmp_path / 'out'
    ms.stage(snap, out, ms.path_map(tmp_path / 'package1', tmp_path / 'fd', tmp_path / 'att', out, tmp_path / 'w'))
    text = (out / 'fig_compA.py').read_text()
    assert 'results/wave4/' in text and 'fixes/option_b/section46' in text
    assert 'results/wave3/B_instrumentation' in (out / 'heldout.py').read_text()


def test_an_unmapped_path_stops_the_run(tmp_path):
    snap, out = _snapshot(tmp_path), tmp_path / 'out'
    (snap / 'our_scripts' / 'figure_scripts' / 'new.py').write_text("X = '/home/claude/elsewhere/a.csv'\n")
    with pytest.raises(AssertionError, match='unmapped'):
        ms.stage(snap, out, ms.path_map(tmp_path / 'p', tmp_path / 'fd', tmp_path / 'att', out))
