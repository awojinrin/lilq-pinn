"""
Run the manuscript folder's scripts on ``package1`` (Package 2, Part 1: the release, gate G1)
==============================================================================================

The manuscript's figures, its screening tables (Appendix B.3) and the
best-approximation numbers of Section 6.6 are produced by scripts kept in the
manuscript folder (``manuscript_snapshot_2026-10-04/our_scripts``; the
advisor's README there maps them). They read our result files through paths
on the advisor's machine (``/home/claude/...``). This copies the scripts,
points those paths at ``package1`` (and at the ``figure-data`` exports and
the package's attachments), runs each one unchanged otherwise, and checks
the output against the manuscript:

- **figures:** each regenerated PDF is rendered beside the manuscript's file
  of the same name (PyMuPDF, 100 dpi), and the share of pixels that differ by
  more than 10 grey levels is reported (``figures.csv``);
- **screening tables:** each regenerated row of ``tab:screening`` and
  ``tab:screening_f2`` is compared with the row of ``main.tex`` (``screening.csv``);
- **delta.py:** its printed best-approximation errors and ratios
  (``delta.txt``), against the numbers of Section 6.6.

The wave folders are copied into ``package1`` unchanged (``WAVES.json``), so
``wave<N>/results/wave<N>`` maps to ``package1``. One file differs on
purpose: ``fig_compA.py`` read wave 4's ``fixes/option_b/section46`` copy of
``kovasznay_comparison.csv``, which predates the correction of Package 2,
Section 2, item 4 (F1 medians of the final, not the minimum logged, errors);
it is pointed at ``package1``'s corrected file.

Usage::

    python experiments/manuscript_scripts.py --snapshot <manuscript_snapshot_2026-10-04> \\
        --package <package1> --figure-data <figure-data results/figure_data> \\
        --attachments <Package2v2/attachments> --out <G1_release/manuscript_scripts>

``--waves <folder>`` runs them on the wave folders as published on the
``wave<N>-results`` branches instead (``git archive wave<N>-results results``),
which are the scripts' own inputs.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FIGURE_SCRIPTS = ('fig_conv.py', 'fig_resid.py', 'fig_np.py', 'fig_basis.py', 'fig_compA.py', 'fig_fields.py')
ANALYSIS_SCRIPTS = ('gen_screening.py', 'delta.py')


def path_map(package, figure_data, attachments, out, waves=None):
    """``package``: every wave path goes to ``package1``. ``waves``: a folder
    holding ``results/wave1`` .. ``results/wave4`` as published on the
    ``wave<N>-results`` branches, the scripts' own inputs."""
    if waves is None:
        p = Path(package).resolve().as_posix()
        wave = {1: p, 2: p, 3: p, 4: p}
        corrected = [("'fixes/option_b/section46/results/kovasznay_comparison.csv'",
                      "'A_calibration/results/kovasznay_comparison.csv'")]
    else:
        w = Path(waves).resolve().as_posix()
        wave = {n: f'{w}/results/wave{n}' for n in (1, 2, 3, 4)}
        corrected = []
    return corrected + [
        ('/home/claude/wave1/results/wave1', wave[1]), ('/home/claude/w2/results/wave2', wave[2]),
        ('/home/claude/w3/results/wave3', wave[3]), ('/home/claude/w4/results/wave4', wave[4]),
        ('/home/claude/figdata/results/figure_data', Path(figure_data).resolve().as_posix()),
        ('/home/claude/c1c2', (Path(attachments) / 'S85_sampling_constants').resolve().as_posix()),
        ('/home/claude/pilot', (Path(attachments) / 'bratu_pilot').resolve().as_posix()),
        ('/home/claude/ms/edits', Path(out).resolve().as_posix()),
        ('/home/claude/figs3', Path(out).resolve().as_posix()), ('/home/claude/figs2', Path(out).resolve().as_posix()),
        ('/home/claude/figs', Path(out).resolve().as_posix()),
    ]


def stage(snapshot, out, mapping):
    """Copy the scripts into ``out`` with the paths rewritten; returns the rewrites made."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for sub in ('figure_scripts', 'analysis_scripts'):
        for src in sorted((Path(snapshot) / 'our_scripts' / sub).glob('*.py')):
            text = src.read_text(encoding='utf-8')
            for old, new in mapping:
                if old in text:
                    made.append((src.name, old, new))
                    text = text.replace(old, new)
            assert '/home/claude' not in text, f'{src.name}: an unmapped path remains'
            (out / src.name).write_text(text, encoding='utf-8')
    return made


def run(out, script):
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    p = subprocess.run([sys.executable, script], cwd=out, capture_output=True, text=True, env=env, timeout=3600)
    (Path(out) / f'{Path(script).stem}.log').write_text(p.stdout + p.stderr, encoding='utf-8')
    return p.returncode, p.stdout


def compare_pdf(a, b, dpi=100, level=10):
    """Share of pixels differing by more than ``level`` grey levels (first page; sizes must match)."""
    import fitz
    import numpy as np
    imgs = []
    for f in (a, b):
        page = fitz.open(f)[0]
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        imgs.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).astype(int))
    if imgs[0].shape != imgs[1].shape:
        return None, f'size {imgs[0].shape} vs {imgs[1].shape}'
    return float((abs(imgs[0] - imgs[1]) > level).mean()), ''


def screening_rows(tex_text, label):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.release_tables import tex_rows
    return [[re.sub(r'\s+', ' ', c).strip() for c in r] for r in tex_rows(tex_text, label)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the manuscript folder's scripts on package1 and compare.")
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--package', required=True)
    ap.add_argument('--figure-data', required=True)
    ap.add_argument('--attachments', required=True)
    ap.add_argument('--waves', default=None,
                    help='a folder with results/wave1..wave4 from the wave<N>-results branches (instead of package1)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    made = stage(args.snapshot, out, path_map(args.package, args.figure_data, args.attachments, out, args.waves))
    with open(out / 'path_rewrites.csv', 'w', newline='') as f:
        csv.writer(f).writerows([('script', 'old', 'new')] + made)
    status = {}
    for s in FIGURE_SCRIPTS + ANALYSIS_SCRIPTS:
        code, stdout = run(out, s)
        status[s] = code
        if s == 'delta.py':
            (out / 'delta.txt').write_text(stdout, encoding='utf-8')
        print(f'{s:18s} exit {code}')
    snapshot = Path(args.snapshot)
    rows = []
    for pdf in sorted((snapshot / 'figures').glob('*.pdf')):
        new = out / pdf.name
        if not new.exists():
            rows.append((pdf.name, '', 'not produced by these scripts'))
            continue
        share, note = compare_pdf(new, pdf)
        rows.append((pdf.name, '' if share is None else f'{share:.5f}', note))
    with open(out / 'figures.csv', 'w', newline='') as f:
        csv.writer(f).writerows([('figure', 'share_of_pixels_differing', 'note')] + rows)
    for r in rows:
        print(f'  {r[0]:36s} {r[1]:>9s} {r[2]}')
    tex = (snapshot / 'main.tex').read_text(encoding='utf-8')
    gen = (out / 'screening_tables.tex').read_text(encoding='utf-8') if (out / 'screening_tables.tex').exists() else ''
    srows = []
    for label in ('tab:screening', 'tab:screening_f2'):
        ms, mine = screening_rows(tex, label), screening_rows(gen, label) if gen else []
        for i, (a, b) in enumerate(zip(ms, mine)):
            srows.append((label, i, a == b, ' & '.join(a) if a != b else '', ' & '.join(b) if a != b else ''))
        if len(ms) != len(mine):
            srows.append((label, 'rows', False, len(ms), len(mine)))
    with open(out / 'screening.csv', 'w', newline='') as f:
        csv.writer(f).writerows([('table', 'row', 'identical', 'manuscript', 'regenerated')] + srows)
    print(f"screening rows identical: {sum(1 for r in srows if r[2] is True)} of {len(srows)}")
    return 0 if all(c == 0 for c in status.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
