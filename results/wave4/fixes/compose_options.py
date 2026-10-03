"""
Wave 4's clean-timing fix: the tables under option A and option B
==================================================================

Wave 4's clean timing of gravity BL at P = 256 (job 13a) is wrong: its timed
run took 3.17 s, against 0.72 s for its own warm-up and 0.74 s logged, all 12
iterations. Two fixes were run on Grace with wave 4's code (17b3539) and
protocol, each on an exclusive CPU node, so that the advisor can choose
between their results:

* **option A** (job 14a, ``option_a_gravity_rerun/``): the gravity BL clean
  timing again. P = 256's row replaces 13a's; the other three sizes are a
  consistency check, not used.
* **option B** (job 14b, ``option_b_replicate_1/``, ``option_b_replicate_2/``):
  two more independent clean timings of every quoted CPU time. Every CPU row
  is then the median of three: 13a's and the two replicates', each by the
  same protocol (warm-up, timed run, median of five under 1 s). The GPU rows
  (job 13b, all under 1 s and already medians of five) are unchanged.

This writes, beside the fix outputs in ``results/wave4/fixes/``:

* ``options_compared.csv``: every quoted time, logged, wave 4's (13a/13b),
  option A's and option B's, with the three timings behind option B and
  their spread;
* ``option_a/`` and ``option_b/``: under each option, ``clean_timing.csv``,
  ``four_method_lilq.csv`` (``four_method_tables.lilq_rows``) and the Section
  4.6 comparison (``kovasznay_comparison.build``: ``results/`` and
  ``figures/``; only LiL-Q's times differ between the options).

Every clean timing must take as many iterations as wave 4's of the same run
(the same solve); the script stops otherwise.

Usage, from a checkout of the code at 17b3539 (branch ``wave4-results``)::

    python results/wave4/fixes/compose_options.py --wave4 results/wave4 --package <package1>

``--package`` is package1 as assembled by ``90_finalize`` (on Grace,
``results/package1``); it supplies the logged paper passes' ``summary.json``
and the Component A runs for Section 4.6.
"""

import argparse
import csv
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CT = Path('B_instrumentation') / 'clean_timing' / 'clean_timing.csv'
QUESTION = 'bl_gravity_P256_cpu_paper'


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(c for r in rows for c in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, restval='')
        w.writeheader()
        w.writerows(rows)


def key(r):
    return r['run'], r['quantity']


def check_iterations(base, other, label):
    by = {key(r): r for r in base}
    for r in other:
        b = by.get(key(r))
        if b is None:
            raise SystemExit(f"{label}: {key(r)} is not in wave 4's clean timing")
        if r.get('error') or int(r['iterations']) != int(b['iterations']):
            raise SystemExit(f"{label}: {key(r)} took {r['iterations']} iterations (error: {r.get('error')!r}), "
                             f"wave 4's {b['iterations']}: not the same solve")


def option_a(base, rerun):
    """Wave 4's rows, with gravity BL P = 256's replaced by the rerun's."""
    new = {key(r): r for r in rerun if r['run'] == QUESTION}
    if not new:
        raise SystemExit(f"option A: no {QUESTION} in the rerun")
    return [dict(new[key(r)], source='14a (option A rerun)') if key(r) in new else dict(r, source='13a/13b (wave 4)')
            for r in base]


def option_b(base, rep1, rep2):
    """Every CPU row the median of three clean timings (13a's and the two
    replicates'); the GPU rows unchanged."""
    reps = [{key(r): r for r in rep} for rep in (rep1, rep2)]
    out = []
    for r in base:
        if r['device'] != 'cpu':
            out.append(dict(r, source='13b (wave 4; GPU, not replicated)'))
            continue
        times = [float(r['clean_time_s'])] + [float(rep[key(r)]['clean_time_s']) for rep in reps]
        out.append(dict(r, clean_time_s=statistics.median(times), clean_time_13a_s=times[0],
                        clean_time_rep1_s=times[1], clean_time_rep2_s=times[2],
                        spread_max_over_min=max(times) / min(times),
                        source='median of three: 13a and 14b replicates 1 and 2'))
    return out


def downstream(rows, package, out, kc, fmt):
    """four_method_lilq.csv and Section 4.6 under one option's clean times."""
    write(out / 'clean_timing.csv', rows)
    b_pkg = Path(package) / 'B_instrumentation'
    with tempfile.TemporaryDirectory() as tmp:
        shadow = Path(tmp)
        (shadow / 'clean_timing').mkdir()
        shutil.copy(out / 'clean_timing.csv', shadow / 'clean_timing' / 'clean_timing.csv')
        for r in rows:
            src = b_pkg / r['run'] / 'summary.json'
            if src.exists():
                (shadow / r['run']).mkdir(exist_ok=True)
                shutil.copy(src, shadow / r['run'] / 'summary.json')
        fmt.lilq_rows(shadow, out / 'four_method_lilq.csv')
    clean = {r['run']: float(r['clean_time_s']) for r in rows if r['quantity'] == 'solve_time_total'}
    original = kc.load_clean_times
    kc.load_clean_times = lambda b_root: clean
    try:
        kc.build(package, out / 'section46')
    finally:
        kc.load_clean_times = original


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--wave4', required=True, help='results/wave4 (with fixes/ from jobs 14a and 14b)')
    ap.add_argument('--package', required=True, help="package1 as assembled by 90_finalize")
    args = ap.parse_args(argv)
    import experiments.four_method_tables as fmt
    import experiments.kovasznay_comparison as kc

    w4 = Path(args.wave4)
    fixes = w4 / 'fixes'
    base = read(w4 / CT)
    rerun = read(fixes / 'option_a_gravity_rerun' / CT)
    rep1 = read(fixes / 'option_b_replicate_1' / CT)
    rep2 = read(fixes / 'option_b_replicate_2' / CT)
    for label, rows in (('14a', rerun), ('14b replicate 1', rep1), ('14b replicate 2', rep2)):
        check_iterations(base, rows, label)
    cpu = {key(r) for r in base if r['device'] == 'cpu'}
    for label, rows in (('14b replicate 1', rep1), ('14b replicate 2', rep2)):
        missing = cpu - {key(r) for r in rows}
        if missing:
            raise SystemExit(f"{label}: missing {sorted(missing)}")

    a, b = option_a(base, rerun), option_b(base, rep1, rep2)
    downstream(a, args.package, fixes / 'option_a', kc, fmt)
    downstream(b, args.package, fixes / 'option_b', kc, fmt)

    by_a, by_b = {key(r): r for r in a}, {key(r): r for r in b}
    by_rerun = {key(r): r for r in rerun}
    compared = []
    for r in base:
        k, ra, rb = key(r), by_a[key(r)], by_b[key(r)]
        w = float(r['clean_time_s'])
        compared.append({
            'run': r['run'], 'quantity': r['quantity'], 'device': r['device'], 'iterations': r['iterations'],
            'logged_time_s': r['logged_time_s'], 'wave4_clean_time_s': w,
            'option_a_time_s': ra['clean_time_s'], 'option_b_time_s': rb['clean_time_s'],
            'option_a_over_wave4': float(ra['clean_time_s']) / w, 'option_b_over_wave4': float(rb['clean_time_s']) / w,
            'option_b_timings_s': json.dumps([rb.get('clean_time_13a_s'), rb.get('clean_time_rep1_s'),
                                              rb.get('clean_time_rep2_s')]) if r['device'] == 'cpu' else '',
            'option_b_spread': rb.get('spread_max_over_min', ''),
            'gravity_rerun_time_s': by_rerun[k]['clean_time_s'] if k in by_rerun else '',
        })
    write(fixes / 'options_compared.csv', compared)
    changed = [c for c in compared if abs(c['option_b_over_wave4'] - 1) > 0.1 or c['option_a_over_wave4'] != 1]
    print(f"Wrote {fixes / 'options_compared.csv'} ({len(compared)} rows) and option_a/, option_b/.")
    for c in changed:
        print(f"  {c['run']} {c['quantity']}: wave 4 {c['wave4_clean_time_s']:.4g} s, A {float(c['option_a_time_s']):.4g}, "
              f"B {float(c['option_b_time_s']):.4g} (timings {c['option_b_timings_s']})")


if __name__ == '__main__':
    main()
