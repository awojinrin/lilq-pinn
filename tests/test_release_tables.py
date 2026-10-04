"""The release check (Package 2, gate G1): the manuscript's tables are read
out of LaTeX, numbers agree to the digits printed, markers and text agree,
and a wrong number or marker is caught."""

import csv

import pytest

import experiments.release_tables as rt

TEX = r"""
\begin{table}
\caption{A table.}
\label{tab:toy}
\begin{tabular}{@{}lrrr@{}}
\toprule
Name & $P$ & Iter. & Error \\
\midrule
\mbox{\textsc{NiL-N}} & 1{,}024 & 5{,}145$^{\dagger,\,1/3}$ & $2.5 \times 10^{-1}$ \\
[3pt] Bratu & 25 & 0.37--0.95 & $1.2 \times 10^{0\phantom{-}}$ (8) \\
\multirow{2}{*}{S1} & 7 & \multicolumn{2}{c}{$7.4{\times}10^{-9}$} \\
& 8 & -- & full (=P) \\
\bottomrule
\end{tabular}
\end{table}
"""


def test_tex_rows_and_cells():
    rows = rt.tex_rows(TEX, 'tab:toy')
    assert len(rows) == 4 and all(len(r) == 4 for r in rows)
    numbers, markers = rt.parse_cell(rows[0][2])
    assert [v for v, _ in numbers] == [5145] and markers == {'dagger', '1/3'}
    assert rt.parse_cell(rows[0][1])[0] == [(1024.0, 0.5)]
    (v, tol), = rt.parse_cell(rows[0][3])[0]
    assert v == pytest.approx(0.25) and tol == pytest.approx(0.005)
    assert [v for v, _ in rt.parse_cell(rows[1][2])[0]] == [0.37, 0.95]
    assert [v for v, _ in rt.parse_cell(rows[1][3])[0]] == [1.2, 8]
    assert rt.norm(rows[2][0]) == 'S1' and rt.parse_cell(rows[2][2])[0][0][0] == pytest.approx(7.4e-9)
    assert rt.parse_cell(rows[3][2]) == ([], set())


def test_compare_cell_catches_numbers_markers_and_text():
    assert rt.compare_cell(rt.num(0.2549), '$2.5 \\times 10^{-1}$') is None
    assert rt.compare_cell(rt.num(0.2551), '$2.5 \\times 10^{-1}$') is not None       # rounds to 2.6e-1
    assert rt.compare_cell(rt.num(9.0458e-4), '$9.1 \\times 10^{-4}$') is not None     # the BL slip
    assert rt.compare_cell(rt.num(5145, markers={'dagger', '1/3'}), '5{,}145$^{\\dagger,\\,1/3}$') is None
    assert rt.compare_cell(rt.num(5145, markers={'dagger'}), '5{,}145$^{\\dagger,\\,1/3}$') is not None
    assert rt.compare_cell(rt.num(5145), '5{,}145$^{*}$') is not None                  # an unexpected marker
    assert rt.compare_cell(rt.num(0.37, 0.95), '0.37--0.95') is None
    assert rt.compare_cell(rt.num(0.37), '0.37--0.95') is not None
    assert rt.compare_cell(rt.txt('NiL-N'), '\\mbox{\\textsc{NiL-N}}') is None
    assert rt.compare_cell(rt.num(), '--') is None


def test_four_method_markers():
    g = lambda *reasons: [{'stopping_reason': r} for r in reasons]
    assert rt._markers(g('target', 'target', 'target')) == set()
    assert rt._markers(g('target', 'target', 'optimizer_stall')) == {'2/3'}
    assert rt._markers(g('optimizer_stall', 'optimizer_stall', 'target')) == {'dagger', '1/3'}
    assert rt._markers(g('iteration_cap', 'iteration_cap', 'optimizer_stall')) == {'*'}
    assert rt._markers(g('optimizer_stall', 'optimizer_stall', 'optimizer_stall')) == {'dagger'}
    assert rt._markers(g('iteration_cap')) == {'*'}                                     # the single LiL-N run


def test_check_matches_rows_by_key_and_reports(tmp_path, monkeypatch):
    tex = TEX.replace('tab:toy', 'tab:keyed')
    rows = [[rt.txt('NiL-N'), rt.num(1024), rt.num(5145, markers={'dagger', '1/3'}), rt.num(0.251)],
            [rt.txt('Bratu'), rt.num(25), rt.num(0.37, 0.95), rt.num(1.2, 8)],
            [rt.txt('S1'), rt.num(7), rt.num(7.4e-9), rt.SKIP],
            [rt.txt('S1'), rt.num(8), rt.num(), rt.txt('full (=P)')]]
    monkeypatch.setitem(rt.TABLES, 'tab:keyed', (lambda pkg: (['a', 'b', 'c', 'd'], rows[::-1]), 2))
    built = rt.build(None, tmp_path, ['tab:keyed'])
    records = rt.check(tex, built)
    assert [r for r in records if r[5]] == []
    with open(tmp_path / 'tab_keyed.csv') as f:
        assert next(csv.reader(f)) == ['a', 'b', 'c', 'd']
    rows[0][3] = rt.num(0.26)
    bad = [r for r in rt.check(tex, rt.build(None, tmp_path, ['tab:keyed'])) if r[5]]
    assert len(bad) == 1 and '0.25' in bad[0][5]
