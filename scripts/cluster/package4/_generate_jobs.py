"""Writes Package 4's job scripts (LF line endings). Kept beside them, so the job
scripts can be regenerated and reviewed from one place.

Usage: python scripts/cluster/package4/_generate_jobs.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEAD = '#!/bin/bash\n'
TAIL = '#SBATCH --output=logs/%x.%j.out\n\nsource scripts/cluster/env.sh\nset -euo pipefail\n'
P1 = '"$RESULTS/package1_v2.0.0/package1"'
REF = '"$RESULTS/package2_stage1/reference"'
B1 = 'experiments/p4_1_burgers_gauss.py'
B2 = 'experiments/p4_2_pb_inclusions.py'
PTS = 'data/package4'
# until_end <command...>: the command under a cap that ends 5 min before the job's walltime
# (LILQ_WALL_S, set by the job), so a slow last run is stopped by us and not by Slurm. A run
# stopped this way keeps every iterate it wrote (the drivers write after each one).
UNTIL_END = ("# until_end <command...>: the command, stopped 5 min before the walltime if still running. It\n"
             "# keeps every iterate it wrote (the drivers write after each one); the job goes on.\n"
             'until_end() { timeout "$((LILQ_WALL_S - SECONDS - 300))s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }\n')

PREFLIGHT_BODY = f'''python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
# M0's package1 runs, item 1's Cole-Hopf reference (Package 2), item 2's shipped Halton points
inputs=({P1}/B_instrumentation/burgers_P625_cpu_paper/iterations.csv
        {P1}/B_instrumentation/bratu_P225_cpu_paper/iterations.csv
        {REF}/burgers_cole_hopf.npz
        {PTS}/halton_seed0.npz {PTS}/halton_seed1.npz {PTS}/halton_seed2.npz)
for f in "${{inputs[@]}}"; do
    [[ -f "$f" ]] || {{ echo "MISSING INPUT: $f"; exit 1; }}
done
echo "inputs OK (${{#inputs[@]}} files)"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
# The deliverables' points folder (the shipped draws and the evaluation quadrature), and
# Grace's own scipy against the shipped draws (a record: no set is built from Grace's draws)
python {B2} points --out "$PKG" --points {PTS}
python {B2} compare-points --out "$PKG" --points {PTS}
echo "PREFLIGHT OK"
'''

# Item 2f, side by side: one process per seed (8 BLAS threads each, 24 in all), each from
# p = 10 up; seed 0's also takes the two extreme 2d sets before its p = 30. Every p = 30 run
# is last in its process and runs until 5 min before the walltime (Section 6.2: the sizes up
# to p = 26 stand).
CONSTANTS2_BODY = f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END}seed_runs() {{   # seed_runs <seed> [--no-extremes]: one seed's 2f runs, p = 30 last
    local s=$1; shift
    python {B2} constants --out "$PKG" --points {PTS} --seeds "$s" --p 10 14 18 22 26 "$@"
    until_end python {B2} constants --out "$PKG" --points {PTS} --seeds "$s" --p 30 --no-extremes
}}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
pids=()
seed_runs 0 & pids+=($!)
seed_runs 1 --no-extremes & pids+=($!)
seed_runs 2 --no-extremes & pids+=($!)
failed=0
for pid in "${{pids[@]}}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || {{ echo "ITEM 2F: A SEED'S RUNS FAILED (see above)"; exit 1; }}
echo "ITEM 2F DONE"
'''

JOBS = {
    'p4_preflight.slurm': ('preflight', 'cpu', '00:30:00', '''# Package 4 (the advisor's instructions of 12 November 2026): the preflight.
# The first Package 4 job: env.sh locks results/package4 to this commit here. Versions, every
# input the jobs read (fail now, not later), the test suite, the points folder and Grace's
# scipy against the shipped Halton points. Every other Package 4 job waits for it.
# Submit: through scripts/cluster/package4/submit_p4.sh
''', PREFLIGHT_BODY),
    'p4_timed.slurm': ('timed', 'timed-cpu', '00:20:00', '''# Package 4 on one whole node (48 threads): check M0 (package1's Burgers P = 625 and Bratu
# P = 225 paper passes, to all digits), then the timings of items 1a, 1b and 2a (Section 2.3:
# warm-up, three timings, each the median of five under 1 s, stopped at the rule's iterate).
# The node's hardware.json and environment.txt go to provenance/timed/.
# Laptop: about 5 min in all.
# Submit: through scripts/cluster/package4/submit_p4.sh
''', f'''python experiments/p4_checks.py provenance --out "$PKG" --name timed
python experiments/p4_checks.py m0 --out "$PKG" --package1 {P1} || echo "M0 FAILED (exit $?)"
python {B1} time --out "$PKG"
python {B2} time --out "$PKG" --points {PTS}
'''),
    'p4_item1.slurm': ('item1', 'cpu', '00:40:00', '''# Package 4, item 1 (Section 3): 1a's seven logged runs (K_max = 60), 1b's four enrichment
# runs, 1c's constants at every iterate. Laptop: 3 min, 16 s and 17 min.
# Submit: through scripts/cluster/package4/submit_p4.sh
''', f'''python {B1} logged --out "$PKG" --reference-dir {REF}
python {B1} enrich --out "$PKG" --reference-dir {REF}
python {B1} constants --out "$PKG"
'''),
    'p4_item2.slurm': ('item2', 'cpu', '00:20:00', '''# Package 4, item 2 (Section 4): 2a's 18 logged runs, 2b, 2c, the 2d and 2e sweeps, M7 and the
# solution field. Laptop: 7 min, 10 s, 9 s, 1.5 min, 1.5 min, 30 s and 6 s.
# Submit: through scripts/cluster/package4/submit_p4.sh
''', f'''python {B2} lilq --out "$PKG" --points {PTS}
python {B2} square --out "$PKG" --points {PTS}
python {B2} under --out "$PKG" --points {PTS}
python {B2} rows --out "$PKG" --points {PTS}
python {B2} weight --out "$PKG" --points {PTS}
python {B2} m7 --out "$PKG"
python {B2} solution --out "$PKG"
'''),
    'p4_item2_constants.slurm': ('item2-constants', 'cpu', '01:15:00', '''# Package 4, item 2f (Section 4.7): c1, c2 and rho_r at every iterate of the 18 runs of 2a and
# of the two extreme sets of 2d at p = 22. After p4_item2, whose runs it reads. Laptop: 66 min
# one run after another at 24 threads; side by side about 1.8 times faster (p = 22: 321 s
# against 567 s), so about 40 min.
# Submit: through scripts/cluster/package4/submit_p4.sh
''', CONSTANTS2_BODY),
    'p4_report.slurm': ('report', 'cpu', '00:15:00', '''# Package 4: after the jobs (afterany). The summaries of items 1 and 2 (terminal.csv, the sweep
# tables, constants.csv, the checks), checks.json, then sacct, su_per_job.csv and
# results/package4.tar.gz to download (scripts/cluster/package2/stage_report.py). Checks M1, M6
# and M8 compare with the advisor's expected values, which are not on Grace: they are evaluated
# at the assembly on the laptop. A failed summary does not stop the tarball.
# Submit: through scripts/cluster/package4/submit_p4.sh
''', f'''python {B1} summarize --out "$PKG" --reference-dir {REF} || echo "ITEM 1 SUMMARY FAILED (exit $?)"
python {B2} summarize --out "$PKG" || echo "ITEM 2 SUMMARY FAILED (exit $?)"
python experiments/p4_checks.py collect --out "$PKG" || echo "CHECKS.JSON FAILED (exit $?)"
python scripts/cluster/package2/stage_report.py --stage p4 --results "$RESULTS"
'''),
}


def render(name):
    short, cls, wall, header, body = JOBS[name]
    h, m, s = map(int, wall.split(':'))
    body = body.replace('__WALL_S__', str(3600 * h + 60 * m + s))
    return (HEAD + header + f'# lilq-resources: {cls}   (applied by scripts/cluster/sbatch.sh from the cluster profile)\n'
            f'#SBATCH --job-name=lilq-p4-{short}\n#SBATCH --time={wall}\n' + TAIL + body)


def main():
    for name in JOBS:
        (HERE / name).write_bytes(render(name).encode())
    print(f'{len(JOBS)} job scripts written in {HERE}')


if __name__ == '__main__':
    main()
