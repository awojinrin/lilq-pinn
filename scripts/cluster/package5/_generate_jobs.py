"""Writes Package 5's job scripts for items 1, 2 and 4 (LF line endings). Kept beside them,
so the job scripts can be regenerated and reviewed from one place.

The plan: the advisor's reply to the cost check (14 November 2026) and DECISIONS.md
(Package 5, the cost). The untimed runs go on 24 cores (`cpu-mem`: 96 GB, billed as 24
cores, which the preflight checks), several processes side by side where there are many runs; the
timed runs on one whole node.

Usage: python scripts/cluster/package5/_generate_jobs.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEAD = '#!/bin/bash\n'
TAIL = '#SBATCH --output=logs/%x.%j.out\n\nsource scripts/cluster/env.sh\nset -euo pipefail\n'
REF2 = '"$RESULTS/package2_stage2/reference"'      # P2-16's and P3-2's references (item 1)
REF1 = '"$RESULTS/package2_stage1/reference"'      # Package 4's Cole-Hopf reference (item 2)
W = 'experiments/p5_1_weights.py'
BEL = 'experiments/p3_1_beltrami_certified.py'
NL = 'experiments/p5_2_nlls.py'
PB = 'experiments/p5_4_pb.py'
P_REF = 88                     # the advisor's answer 1; its neighbour p = 80
LIMIT_NEIGHBOUR = '40m'        # p = 80: about 15 to 18 min expected (laptop ladder, scaled as p^6)
LIMIT_REF = '60m'              # p = 88: about 25 to 32 min expected (the user's limit, 14 November 2026)
THREADS4 = 'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4'
UNTIL_END = ("# until_end <command...>: the command, stopped 5 min before the walltime if still running. It\n"
             "# keeps every iterate it wrote (the drivers write after each one); the job goes on.\n"
             "# With no time left it skips the command and says so (timeout 0 would mean no limit).\n"
             'until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))\n'
             '    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }\n'
             '    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }\n')
WAIT_ALL = '''failed=0
for pid in "${{pids[@]}}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || {{ echo "{what}: A PROCESS FAILED (see above)"; exit 1; }}
echo "{what} DONE"
'''

INPUTS = [f'{REF2}/{f}' for f in ('bratu_ref_p48.npz', 'burgers_cole_hopf.npz', 'bl_fd_ref_viscous_nu0.1.npz',
                                  'bl_fd_ref_gravity_nu0.1.npz')] + [f'{REF1}/burgers_cole_hopf.npz'] + \
    [f'data/package{k}/halton_seed{s}.npz' for k in (4, 5) for s in (0, 1, 2)]

PREFLIGHT_BODY = '''python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
# This job is cpu-mem (24 cores, 96 GB): the plan assumes Grace bills it as 24 cores. Stop
# everything (the other jobs wait for this one) if it is billed higher.
billing=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'billing=[0-9]*' | head -1 | cut -d= -f2 || true)
echo "cpu-mem billing: ${billing:-unknown} per hour (expected 24)"
[[ -n "$billing" ]] || echo "WARNING: the billing could not be read; check sacct's AllocTRES of this job"
[[ "${billing:-0}" -le 24 ]] || { echo "CPU-MEM IS BILLED AT $billing, NOT 24: STOPPING (nothing else runs)"; exit 1; }
# Every input the jobs read: item 1's references (Package 2, stage 2), item 2's Cole-Hopf
# reference (stage 1, as Package 4), the shipped Halton points of Packages 4 and 5
inputs=(__INPUTS__)
for f in "${inputs[@]}"; do
    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }
done
echo "inputs OK (${#inputs[@]} files)"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
# Grace's own scipy against the shipped 16,000 points (a record: every set is built from the shipped points)
python __PB__ compare-points --out "$PKG" --points data/package5
echo "PREFLIGHT OK"
'''.replace('__INPUTS__', '\n        '.join(INPUTS)).replace('__PB__', PB)


def reference_body(case):
    return f'''# The advisor's answer 1: p = {P_REF} with its neighbour p = {P_REF - 8}; if either does not finish
# within its limit or does not stop by the rule, this case falls back to p = 64 with 56
# (`choose`, which runs them and writes bl_reference/chosen_<case>.json with the reason).
ref() {{   # ref <limit> <p>
    timeout "$1" python {W} reference --case {case} --p-ref "$2" --out "$PKG" --reference-dir {REF2} \\
        || echo "REFERENCE {case} p = $2: STOPPED OR FAILED (exit $?)"
}}
ref {LIMIT_NEIGHBOUR} {P_REF - 8}
if python {W} ref-ok --case {case} --p-ref {P_REF - 8} --out "$PKG"; then
    ref {LIMIT_REF} {P_REF}
fi
python {W} choose --case {case} --p-ref {P_REF} --out "$PKG" --reference-dir {REF2}
'''


SWEEP_BODY = f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END}{THREADS4}
# Six processes side by side, 4 threads each (24 in all), one family each. A Buckley-Leverett
# family reads its case's fine reference from bl_reference/chosen_<case>.json and never runs one.
pids=()
for family in bratu burgers bl_cheb_viscous bl_cheb_gravity bl_sine_viscous bl_sine_gravity; do
    until_end python {W} runs --only "$family" --out "$PKG" --reference-dir {REF2} & pids+=($!)
done
''' + WAIT_ALL.format(what='ITEM 1')

BELTRAMI_BODY = f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END}# B1 and B2 at level 1, lambda_bc = lambda_ic = 1e3, 1e4, 1e6 (the delivered runs have 10),
# pin_scale 1, eight iterates; at 24 threads (the advisor's reply: untimed, no whole node).
for size in B1 B2; do
    for lam in 1e3 1e4 1e6; do
        until_end python {BEL} run --size "$size" --level 1 --lambda-bc "$lam" --lambda-ic "$lam" \\
            --tag "_lam$lam" --out "$PKG"
    done
done
'''

TIMED_BODY = f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END}python experiments/p4_checks.py provenance --out "$PKG" --name timed
# Item 4a's timings (Section 5: 2a at p = 34, 38, 42, seeds 0 to 2; the paper's protocol)
python {PB} time --out "$PKG" --points data/package5
# Item 2: the protocol for LiL-Q and the least-squares solvers; the two L-BFGS solvers by their
# one logged pass (the advisor's answer 2)
until_end python {NL} run --out "$PKG" --reference-dir {REF1} --points data/package4
'''

ITEM4_BODY = f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END}{THREADS4}
P4A=(--out "$PKG" --points data/package5)
# Six processes side by side, 4 threads each (24 in all). 4a, one seed each (2a logged, then
# 2c), each followed by part of the rest: 4b up to P = 800, delta_P of Package 4's and 4a's
# spaces, delta_P of 4b's spaces. 4b at P = 1,600, one sigma each.
pids=()
( until_end python {PB} lilq --seeds 0 "${{P4A[@]}}"; until_end python {PB} under --seeds 0 "${{P4A[@]}}"
  until_end python {PB} rf --P 100 200 400 800 "${{P4A[@]}}" ) & pids+=($!)
( until_end python {PB} lilq --seeds 1 "${{P4A[@]}}"; until_end python {PB} under --seeds 1 "${{P4A[@]}}"
  until_end python {PB} delta --which p4 4a --out "$PKG" ) & pids+=($!)
( until_end python {PB} lilq --seeds 2 "${{P4A[@]}}"; until_end python {PB} under --seeds 2 "${{P4A[@]}}"
  until_end python {PB} delta --which 4b --out "$PKG" ) & pids+=($!)
for sigma in 2 4 8; do
    until_end python {PB} rf --P 1600 --sigma "$sigma" "${{P4A[@]}}" & pids+=($!)
done
''' + WAIT_ALL.format(what='ITEM 4')

REPORT_BODY = f'''python {W} summarize --out "$PKG" --reference-dir {REF2} || echo "ITEM 1 SUMMARY FAILED (exit $?)"
python {BEL} summarize --out "$PKG" || echo "BELTRAMI SUMMARY FAILED (exit $?)"
python {NL} summarize --out "$PKG" || echo "ITEM 2 SUMMARY FAILED (exit $?)"
python {PB} summarize --out "$PKG" || echo "ITEM 4 SUMMARY FAILED (exit $?)"
python scripts/cluster/package2/stage_report.py --stage p5 --results "$RESULTS"
'''

JOBS = {
    'p5_preflight.slurm': ('preflight', 'cpu-mem', '00:30:00', '''# Package 5, items 1, 2 and 4 (the advisor's instructions of 13 November 2026 and his reply to
# the cost check of 14 November 2026): the preflight. The first Package 5 job: env.sh locks
# results/package5 to this commit here. The billing of the cpu-mem class, every input the
# jobs read (fail now, not later), the test suite, Grace's scipy against the shipped points.
# Every other Package 5 job waits for it.
# Submit: through scripts/cluster/package5/submit_p5.sh
''', PREFLIGHT_BODY),
    'p5_ref_viscous.slurm': ('ref-viscous', 'cpu-mem', '02:00:00', f'''# Package 5, item 1 (Section 3): the fine Buckley-Leverett reference, viscous case, at 24 threads.
# p = {P_REF - 8} (limit {LIMIT_NEIGHBOUR}), then p = {P_REF} (limit {LIMIT_REF}; about 38 GB), then the choice and,
# if needed, the fallback p = 64 with 56. Laptop, scaled: about 15 + 32 min.
# Submit: through scripts/cluster/package5/submit_p5.sh
''', reference_body('viscous')),
    'p5_ref_gravity.slurm': ('ref-gravity', 'cpu-mem', '02:00:00', f'''# Package 5, item 1 (Section 3): the fine Buckley-Leverett reference, gravity case, at 24 threads.
# p = {P_REF - 8} (limit {LIMIT_NEIGHBOUR}), then p = {P_REF} (limit {LIMIT_REF}; about 38 GB), then the choice and,
# if needed, the fallback p = 64 with 56. Laptop, scaled: about 14 + 26 min.
# Submit: through scripts/cluster/package5/submit_p5.sh
''', reference_body('gravity')),
    'p5_item1.slurm': ('item1', 'cpu-mem', '01:15:00', '''# Package 5, item 1 (Section 3): the 96 runs of Bratu, Burgers and Buckley-Leverett at the four
# weights, K_max = 60, every iterate stored. After both references (whatever their outcome).
# cpu-mem: six processes, each with its own quadrature of the norm of X (up to 176 x 176).
# Laptop: 49 min one after another at 16 threads; side by side about 30 min.
# Submit: through scripts/cluster/package5/submit_p5.sh
''', SWEEP_BODY),
    'p5_beltrami.slurm': ('beltrami', 'cpu-mem', '03:00:00', '''# Package 5, item 1 (Section 3): the six Beltrami runs at 24 threads (B2 about 18 GB). Package 3's
# whole-node times: 398 s (B1) and 1,301 s (B2) per run, about 1.4 h in all; the walltime
# allows 24 threads to be up to twice as slow.
# Submit: through scripts/cluster/package5/submit_p5.sh
''', BELTRAMI_BODY),
    'p5_timed.slurm': ('timed', 'timed-cpu', '01:00:00', '''# Package 5 on one whole node (48 threads): the timings of item 4a (9 runs), then item 2 (42 runs:
# LiL-Q and least_squares by the protocol, the L-BFGS solvers by one logged pass). The node's
# hardware.json and environment.txt go to provenance/timed/. Laptop: about 25 min.
# Submit: through scripts/cluster/package5/submit_p5.sh
''', TIMED_BODY),
    'p5_item4.slurm': ('item4', 'cpu-mem', '02:30:00', '''# Package 5, item 4 (Section 5): 4a logged (2a) and 2c at p = 34, 38, 42, seeds 0 to 2; 4b's 45
# runs; delta_P by gelsy and gelsd for every space. cpu-mem: six processes, those at P = 1,600
# each holding the H2 quadrature's basis (about 2.3 GB). Laptop: 1.9 h one after another at 16
# threads; side by side about 1.1 h.
# Submit: through scripts/cluster/package5/submit_p5.sh
''', ITEM4_BODY),
    'p5_report.slurm': ('report', 'cpu', '00:30:00', '''# Package 5, items 1, 2 and 4: after the jobs (afterany). The summaries (terminal.csv and the
# tables of each item), then sacct, su_per_job.csv and results/package5.tar.gz to download
# (scripts/cluster/package2/stage_report.py). The checks C1 to C3 and the comparisons with the
# advisor's expected values need files that are not on Grace: they run at the assembly on the
# laptop. A failed summary does not stop the tarball.
# Submit: through scripts/cluster/package5/submit_p5.sh
''', REPORT_BODY),
}


# The re-time of one item 2 run (the user's decision, 14 November 2026): stage p5t, its own results
# folder (results/package5_retime) under its own commit lock. In the timed job above, the job's first
# PyTorch L-BFGS pass (Burgers p = 10) carried a one-time cost of about 29 s inside its only timed
# pass; the driver now warms each one-pass solver up first (p5_2_nlls.prepare).
RETIME_BODY = f'''python experiments/p4_checks.py provenance --out "$PKG" --name timed
# Item 2, the one re-timed run: Burgers p = 10, the PyTorch L-BFGS, by one logged pass after an
# untimed warm-up of 5 iterations (warmup_s in its result.json)
python {NL} run --out "$PKG" --reference-dir {REF1} --points data/package4 --problem burgers --p 10 --solver lbfgs
python scripts/cluster/package2/stage_report.py --stage p5t --results "$RESULTS"
'''

RETIME_JOBS = {
    'p5_retime.slurm': ('retime', 'timed-cpu', '00:15:00', '''# Package 5, item 2: the PyTorch L-BFGS at Burgers p = 10 re-timed on one whole node (48 threads), as
# the timed job of 14 November 2026 ran it, with the warm-up. Then results/package5_retime.tar.gz (the
# job's own sacct row is added at the assembly). Laptop: a few seconds; the job, a few minutes.
# Submit: through scripts/cluster/package5/submit_p5_retime.sh
''', RETIME_BODY),
}


def render(name, jobs=JOBS, stage='p5'):
    short, cls, wall, header, body = jobs[name]
    h, m, s = map(int, wall.split(':'))
    body = body.replace('__WALL_S__', str(3600 * h + 60 * m + s))
    return (HEAD + header + f'# lilq-resources: {cls}   (applied by scripts/cluster/sbatch.sh from the cluster profile)\n'
            f'#SBATCH --job-name=lilq-{stage}-{short}\n#SBATCH --time={wall}\n' + TAIL + body)


def main():
    for name in JOBS:
        (HERE / name).write_bytes(render(name).encode())
    for name in RETIME_JOBS:
        (HERE / name).write_bytes(render(name, RETIME_JOBS, 'p5t').encode())
    print(f'{len(JOBS) + len(RETIME_JOBS)} job scripts written in {HERE}')


if __name__ == '__main__':
    main()
