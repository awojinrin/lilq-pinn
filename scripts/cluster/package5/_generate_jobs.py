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
# Item 3: a run writes its records when it ends, so a stop keeps only the finished runs
UNTIL_END3 = UNTIL_END.replace(
    "# keeps every iterate it wrote (the drivers write after each one); the job goes on.\n",
    "# keeps the runs that finished: a run writes its records when it ends, so the one in progress is\n"
    "# lost (a resubmitted job runs it again and skips the finished ones). The job goes on.\n")
assert UNTIL_END3 != UNTIL_END
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


# Addendum 2 (16 November 2026): stage p5a, its own results folder (results/package5_a2) under its own
# commit lock. One cpu-mem job: parts A to C and K1 at item 1's 4 threads (five processes and K1 side
# by side), then part D at the delivered P2-9 run's 24 threads, then the summary and the tarball
# (experiments/p5_a2_plateau.py; DECISIONS.md, Package 5, Addendum 2).
A2 = 'experiments/p5_a2_plateau.py'
DELIVERED = '"$RESULTS/package2_stage2/P2_9_bases/bl_gravity_paper_P1024_ic"'
ITEM1 = '"$RESULTS/package5/P5_1_weights"'
A2_INPUTS = ([f'{REF2}/bl_fd_ref_gravity_nu{nu}.npz' for nu in ('0.1', '0.05', '0.02')]
             + [f'{ITEM1}/bl_reference/chosen_gravity.json']
             + [f'{ITEM1}/bl_reference/gravity_p{p}/beta_reference.npy' for p in (P_REF, P_REF - 8)]
             + [f'{ITEM1}/bl_{v}_gravity/P1024_lam10000/{f}' for v in ('cheb', 'sine')
                for f in ('iterations.csv', 'errors_by_iterate.csv')]
             + [f'{DELIVERED}/iterations.csv'])
A2_BODY = f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END}# Every input: the finite-difference references (Package 2, stage 2), item 1's fine reference
# and its two gravity runs at P = 1,024, weight 10^4 (K1), the delivered P2-9 run (K2)
inputs=(__A2_INPUTS__)
for f in "${{inputs[@]}}"; do
    [[ -f "$f" ]] || {{ echo "MISSING INPUT: $f"; exit 1; }}
done
echo "inputs OK (${{#inputs[@]}} files)"
python experiments/p4_checks.py provenance --out "$PKG" --name a2
A2=(--out "$PKG" --reference-dir {REF2} --item1-root "$RESULTS/package5")
{THREADS4}
# Parts A to C (16 runs) in five processes and K1 beside them, 4 threads each (24 in all): item 1's
# thread count, so that K1 compares like with like
pids=()
until_end python {A2} run --part A --space cheb "${{A2[@]}}" & pids+=($!)
until_end python {A2} run --part A --space sine "${{A2[@]}}" & pids+=($!)
until_end python {A2} run --part B --space cheb "${{A2[@]}}" & pids+=($!)
until_end python {A2} run --part B --space sine "${{A2[@]}}" & pids+=($!)
until_end python {A2} run --part C "${{A2[@]}}" & pids+=($!)
until_end python {A2} k1 "${{A2[@]}}" & pids+=($!)
failed=0
for pid in "${{pids[@]}}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] && echo "PARTS A TO C AND K1 DONE" || echo "PARTS A TO C OR K1: A PROCESS FAILED (see above)"
# Part D at the delivered run's 24 threads (K2 compares the repeat at weight 10 with it)
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
until_end python {A2} run --part D "${{A2[@]}}"
python {A2} summarize "${{A2[@]}}" --delivered {DELIVERED} || echo "SUMMARY FAILED (exit $?)"
python scripts/cluster/package2/stage_report.py --stage p5a --results "$RESULTS"
'''.replace('__A2_INPUTS__', '\n        '.join(A2_INPUTS))

A2_JOBS = {
    'p5_a2.slurm': ('a2', 'cpu-mem', '01:00:00', '''# Package 5, Addendum 2 (Sections 2 and 3): the 18 runs of gravity Buckley-Leverett from the initial
# profile, K1 and K2, the summary, then results/package5_a2.tar.gz. cpu-mem: six processes, each
# about 8 GB while it builds item 1's norm of X. Laptop rehearsal at K_max = 4: 8 min; the 60
# iterations of every run about 11 min of solves at 16 threads.
# Submit: through scripts/cluster/package5/submit_p5_a2.sh
''', A2_BODY),
}


# Item 3 (Section 6: cost at a common accuracy). Two stages, each with its own results folder and
# commit lock: p5c, the CPU set (results/package5_item3_cpu), then p5g, the A100 set
# (results/package5_item3_gpu), which waits for the cost check (the advisor's rule 3). Every
# budget run is timed on a node that holds only our runs: one run at a time on a whole CPU node,
# two at a time on a whole two-A100 node, one per GPU (the user's decision, 16 November 2026).
# DECISIONS.md, Package 5, item 3: the job layout.
N3 = 'experiments/p5_3_networks.py'
L3 = 'experiments/p5_3_lil.py'
T3 = 'experiments/p5_3_table.py'
BENCHES = ('bratu', 'burgers', 'bl')
NET_SIZES = {'bratu': (25, 100, 225), 'burgers': (25, 100, 225, 400, 625), 'bl': (64, 256, 576, 1024)}
LIL_SIZES = {'bratu': (25, 100, 225), 'burgers': (36, 100, 196, 324, 484, 676, 900), 'bl': (64, 256, 576, 1024)}
NET_METHODS = ('nil_n', 'nil_q', 'lm')
NET_SEEDS = (0, 1, 2)
ITEM3_INPUTS = [f'{REF2}/{f}' for f in ('bratu_ref_p48.npz', 'burgers_cole_hopf.npz', 'bl_fd_ref_viscous_nu0.1.npz')]
# A budget run's wall-clock: the 600 s on its clock, plus its error log (off the clock) and a fixed
# part (start, pretraining, setup, saving). The error log's share of the budget, from the laptop
# probe (16 November 2026; 30 s budgets at the largest sizes, CPU and CUDA, the larger of the two,
# rounded up): LM logs at every iteration, the L-BFGS methods every 10. LiL-N L-BFGS is costed as
# a full budget run (the larger sizes reach it).
BUDGET_S = 600
LOG_SHARE = {'lm': {'bratu': 0.22, 'burgers': 0.13, 'bl': 0.08}, 'lbfgs': 0.02}
FIXED_S = 30
MARGIN = 1.10                      # walltime over the expected time
THREADS24 = 'export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24'


def net_runs(bench):
    """The network runs of a benchmark in the drivers' order (``p5_3_networks.main``)."""
    return [(P, m, s) for P in NET_SIZES[bench] for m in NET_METHODS for s in NET_SEEDS]


def part(runs, i, n):
    return runs[i::n]                                   # p5_3_networks.part_of


def run_wall_s(bench, method):
    """One budget run's expected wall-clock (method: a network method, or 'lbfgs' for LiL-N)."""
    share = LOG_SHARE['lm'][bench] if method == 'lm' else LOG_SHARE['lbfgs']
    return BUDGET_S * (1 + share) + FIXED_S


def process_s(bench, nets, lil):
    """A process's expected wall-clock: its network runs, then its LiL-N L-BFGS sizes."""
    return sum(run_wall_s(bench, m) for _, m, _ in nets) + len(lil) * run_wall_s(bench, 'lbfgs')


def walltime(expected_s):
    """hh:mm:00: the expected seconds times MARGIN, rounded up to 15 min."""
    s = expected_s * MARGIN
    m = -(-int(s) // 900) * 15
    return f'{m // 60:02d}:{m % 60:02d}:00'


def _inputs_check(inputs):
    return ('inputs=(' + '\n        '.join(inputs) + ')\nfor f in "${inputs[@]}"; do\n'
            '    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }\ndone\necho "inputs OK (${#inputs[@]} files)"\n')


# ---- p5c, the CPU set
# One whole CPU node per job (timed-cpu, 48 threads), one run at a time. The 122 budget runs (108
# network, 14 LiL-N L-BFGS) in five jobs of about the same length; LiL-Q, trf and the square
# collocation (the usual protocol) and checks C2 and C4 in a sixth.
CPU_SPLIT = {   # job: (bench, network part 'i/n', LiL-N L-BFGS sizes)
    'bratu': ('bratu', '0/1', LIL_SIZES['bratu']),
    'burgers-a': ('burgers', '0/2', (36, 196, 484)),
    'burgers-b': ('burgers', '1/2', (100, 324, 676, 900)),
    'bl-a': ('bl', '0/2', (64, 576)),
    'bl-b': ('bl', '1/2', (256, 1024)),
}


def cpu_runs(job):
    bench, p, lil = CPU_SPLIT[job]
    i, n = map(int, p.split('/'))
    return part(net_runs(bench), i, n), lil


def cpu_expected_s(job):
    return process_s(CPU_SPLIT[job][0], *cpu_runs(job))


C_PREFLIGHT_BODY = ('''python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__)"
# Every input the jobs read (fail now, not later), then the test suite
''' + _inputs_check(ITEM3_INPUTS) + '''python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
echo "PREFLIGHT OK"
''')

C_PROTOCOL_BODY = f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END3}python experiments/p4_checks.py provenance --out "$PKG" --name protocol
# LiL-Q, trf and (Bratu) the square collocation by the usual protocol: one logged pass, then three
# timed passes (the clock per iterate, the median of the timestamps)
for bench in {' '.join(BENCHES)}; do
    until_end python {L3} run --bench "$bench" --methods lilq trf square --device cpu --out "$PKG" \\
        --reference-dir {REF2} --tag cpu-protocol
done
# Checks C2 and C4: NiL-N and LM at the largest size, about 30 s each (sized by a pilot), without and
# with the error log
for bench in {' '.join(BENCHES)}; do
    until_end python {N3} checks --bench "$bench" --device cpu --out "$PKG" --reference-dir {REF2} --tag cpu-checks
done
'''


def cpu_budget_body(job):
    bench, p, lil = CPU_SPLIT[job]
    tag = f'cpu-{job}'
    return f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END3}python experiments/p4_checks.py provenance --out "$PKG" --name {tag}
# The network runs (part {p} of NiL-N, NiL-Q and LM over the sizes and seeds), each to the 600 s budget,
# one at a time, then LiL-N L-BFGS at P = {', '.join(map(str, lil))}
until_end python {N3} run --bench {bench} --part {p} --device cpu --out "$PKG" --reference-dir {REF2} --tag {tag}
until_end python {L3} run --bench {bench} --methods lbfgs --P {' '.join(map(str, lil))} --device cpu --out "$PKG" \\
    --reference-dir {REF2} --tag {tag}
'''


def report_body(stage, pairing=False):
    pre = f'timeout 5m python {N3} pairing --out "$PKG" || echo "PAIRING FAILED OR STOPPED (exit $?)"\n' if pairing else ''
    return (f'''# Each step under its own time limit, so that a stall cannot cost the tarball (the lesson of
# Addendum 2's first job)
{pre}timeout 10m python {T3} --out "$PKG" || echo "TABLE FAILED OR STOPPED (exit $?)"
timeout 15m python scripts/cluster/package2/stage_report.py --stage {stage} --results "$RESULTS" \\
    || echo "REPORT FAILED OR STOPPED (exit $?)"
''')


PROTOCOL_EXTRA_S = 1800            # the LiL protocol runs (a few minutes on the laptop) and C2/C4 at the largest sizes


def cpu_jobs():
    jobs = {
        'p5c_preflight.slurm': ('preflight', 'cpu', '00:30:00', '''# Package 5, item 3, the CPU set: the preflight. The first job of stage p5c: env.sh locks
# results/package5_item3_cpu to this commit here. Every input the jobs read, the test suite.
# Every other job of the stage waits for it.
# Submit: through scripts/cluster/package5/submit_p5_item3_cpu.sh
''', C_PREFLIGHT_BODY),
        'p5c_protocol.slurm': ('protocol', 'timed-cpu', walltime(PROTOCOL_EXTRA_S), '''# Package 5, item 3, the CPU set, on one whole node (48 threads): LiL-Q, trf and the square
# collocation by the usual protocol on the three problems, then checks C2 and C4.
# Submit: through scripts/cluster/package5/submit_p5_item3_cpu.sh
''', C_PROTOCOL_BODY),
    }
    for job in CPU_SPLIT:
        nets, lil = cpu_runs(job)
        n, e = len(nets) + len(lil), cpu_expected_s(job)
        jobs[f'p5c_{job.replace("-", "_")}.slurm'] = (job, 'timed-cpu', walltime(e), f'''# Package 5, item 3, the CPU set, on one whole node (48 threads): {n} budget runs of 600 s, one at a
# time ({CPU_SPLIT[job][0]}: network part {CPU_SPLIT[job][1]}, then LiL-N L-BFGS). Expected {e / 3600:.2f} h (each run with
# its error log, pretraining and setup); the walltime x {MARGIN}.
# Submit: through scripts/cluster/package5/submit_p5_item3_cpu.sh
''', cpu_budget_body(job))
    jobs['p5c_report.slurm'] = ('report', 'cpu', '00:30:00', '''# Package 5, item 3, the CPU set: after the jobs (afterany). The time-to-error table, then sacct,
# su_per_job.csv and results/package5_item3_cpu.tar.gz. C1 (against Package 1) and C3 (against
# Package 4) need files that are not on Grace: they run at the assembly on the laptop.
# Submit: through scripts/cluster/package5/submit_p5_item3_cpu.sh
''', report_body('p5c'))
    return jobs


# The OpenMP library (DECISIONS.md, 17 November 2026). Each driver is imported as its runs import it
# and must have loaded the GCC module's OpenMP library, as Package 1 and the LiL runs did; the job stops
# here otherwise (set -e), before any run. Each driver call also carries --require-openmp, so that a
# process that loaded another library stops before its first run.
OPENMP_REQUIRED = 'module'
OPENMP_CHECK = ('''# The OpenMP library each driver loads: the GCC module's, as Package 1's runs (DECISIONS.md, 17 November 2026)
for m in p5_3_networks p5_3_lil; do
    python -c "import sys; sys.path.insert(0, '.'); import experiments.$m; from lilq.provenance import require_openmp; require_openmp('__KIND__')"
done
'''.replace('__KIND__', OPENMP_REQUIRED))
REQ = f'--require-openmp {OPENMP_REQUIRED}'


# ---- p5r, the CPU set's network runs again (the user's decision, 17 November 2026; the advisor's
# approval in the item 3 cost check). The CPU set's network processes loaded the PyTorch wheel's
# OpenMP library (torch imported first); Package 1's and the LiL runs' loaded the module's. The
# Buckley-Leverett network runs were 60 to 75% slower per iteration with it, and C1 failed on 15 of
# 18 runs. Its own results folder (results/package5_item3_cpu_rerun) under its own commit lock: the
# CPU set stays as it ran. The same five jobs and parts as p5c, network runs only (its LiL-N L-BFGS
# runs loaded the module's library and are kept), and C2/C4 for the networks. The assembly takes the
# network runs from here and the LiL runs from p5c.
R_PREFLIGHT_BODY = ('''python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__)"
''' + OPENMP_CHECK + '''# Every input the jobs read (fail now, not later), then the test suite
''' + _inputs_check(ITEM3_INPUTS) + '''python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
echo "PREFLIGHT OK"
''')

R_CHECKS_BODY = f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END3}{OPENMP_CHECK}python experiments/p4_checks.py provenance --out "$PKG" --name checks
# Checks C2 and C4: NiL-N and LM at the largest size, about 30 s each (sized by a pilot), without and
# with the error log
for bench in {' '.join(b for b in BENCHES if b != 'bl')}; do
    until_end python {N3} checks --bench "$bench" --device cpu --out "$PKG" --reference-dir {REF2} --tag cpu-checks {REQ}
done
# Buckley-Leverett: the NiL-N pair timed three times (C4 was outside 5% in the CPU set; the advisor's reply of
# 17 November 2026 asks for the spread of three timings). C2's and C4's verdicts are the first pair's.
until_end python {N3} checks --bench bl --device cpu --out "$PKG" --reference-dir {REF2} --tag cpu-checks {REQ} --c4-repeats nil_n=3
'''
assert 'bl' in BENCHES


def rerun_body(job):
    bench, p, _ = CPU_SPLIT[job]
    tag = f'cpu-{job}'
    return f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END3}{OPENMP_CHECK}python experiments/p4_checks.py provenance --out "$PKG" --name {tag}
# The network runs (part {p} of NiL-N, NiL-Q and LM over the sizes and seeds), each to the 600 s budget,
# one at a time
until_end python {N3} run --bench {bench} --part {p} --device cpu --out "$PKG" --reference-dir {REF2} --tag {tag} {REQ}
'''


CHECKS_EXTRA_S = 1200              # C2/C4 at the largest sizes (the CPU set's protocol job, with the LiL runs: 15 min)


def rerun_jobs():
    jobs = {
        'p5r_preflight.slurm': ('preflight', 'cpu', '00:30:00', '''# Package 5, item 3, the CPU network runs again: the preflight. The first job of stage p5r: env.sh
# locks results/package5_item3_cpu_rerun to this commit here. The OpenMP library each driver loads,
# every input the jobs read, the test suite. Every other job of the stage waits for it.
# Submit: through scripts/cluster/package5/submit_p5_item3_rerun.sh
''', R_PREFLIGHT_BODY),
        'p5r_checks.slurm': ('checks', 'timed-cpu', walltime(CHECKS_EXTRA_S), '''# Package 5, item 3, the CPU network runs again, on one whole node (48 threads): checks C2 and C4 for
# NiL-N and LM on the three problems.
# Submit: through scripts/cluster/package5/submit_p5_item3_rerun.sh
''', R_CHECKS_BODY),
    }
    for job in CPU_SPLIT:
        nets, _ = cpu_runs(job)
        e = process_s(CPU_SPLIT[job][0], nets, ())
        jobs[f'p5r_{job.replace("-", "_")}.slurm'] = (job, 'timed-cpu', walltime(e), f'''# Package 5, item 3, the CPU network runs again, on one whole node (48 threads): {len(nets)} budget runs of
# 600 s, one at a time ({CPU_SPLIT[job][0]}: network part {CPU_SPLIT[job][1]}, as p5c). Expected {e / 3600:.2f} h (each run with its
# error log, pretraining and setup); the walltime x {MARGIN}.
# Submit: through scripts/cluster/package5/submit_p5_item3_rerun.sh
''', rerun_body(job))
    jobs['p5r_report.slurm'] = ('report', 'cpu', '00:30:00', '''# Package 5, item 3, the CPU network runs again: after the jobs (afterany). The time-to-error table of
# these runs, then sacct, su_per_job.csv and results/package5_item3_cpu_rerun.tar.gz. C1 runs at the
# assembly on the laptop (Package 1's files are not on Grace).
# Submit: through scripts/cluster/package5/submit_p5_item3_rerun.sh
''', report_body('p5r'))
    return jobs


# ---- p5g, the A100 set
# One whole two-A100 node per problem (gpu-pair: both A100s, 48 cores, 192 SU/h), two processes
# side by side, one per GPU, 24 threads each. First the solo check: one run alone on its GPU, the
# other idle, for 60 s; the report compares it with the same run in the pair (within 5%).
GPU_SPLIT = {   # bench: (LiL-N L-BFGS sizes of process 0, of process 1)
    'bratu': ((225,), (25, 100)),
    'burgers': ((36, 196, 484), (100, 324, 676, 900)),
    'bl': ((64, 576), (256, 1024)),
}
SOLO = {b: (NET_SIZES[b][-1], 'nil_n', 0) for b in BENCHES}   # the largest size, NiL-N, seed 0


def gpu_slots(bench, i):
    return len(part(net_runs(bench), i, 2)) + len(GPU_SPLIT[bench][i])


def gpu_expected_s(bench):
    """The pair job's expected wall-clock: the solo check, the pairing gate, then the longer of its two processes."""
    return SOLO_EXTRA_S + GATE_EXTRA_S + max(process_s(bench, part(net_runs(bench), i, 2), GPU_SPLIT[bench][i]) for i in (0, 1))


def solo_gpu(bench):
    """The GPU (process) that runs the solo check's run in the pair: the solo runs on the same one."""
    k = net_runs(bench).index(SOLO[bench])
    return k % 2


G_PREFLIGHT_BODY = ('''nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
''' + OPENMP_CHECK + _inputs_check(ITEM3_INPUTS) + '''# The item 3 tests with the A100 visible (the torch residuals and the off-clock timer on CUDA)
python -m pytest tests/test_p5_3_lil.py tests/test_p5_3_networks.py tests/test_run_probe.py tests/test_p5_3_table.py -q
echo "PREFLIGHT OK"
''')


def gpu_pair_body(bench):
    P, method, seed = SOLO[bench]
    g = solo_gpu(bench)
    procs = ''
    for i in (0, 1):
        tag = f'gpu-{bench}-{i}'
        lil = ' '.join(map(str, GPU_SPLIT[bench][i]))
        procs += (f'( export CUDA_VISIBLE_DEVICES={i}\n'
                  f'  until_end python {N3} run --bench {bench} --part {i}/2 --tag {tag} "${{NET[@]}}"\n'
                  f'  until_end python {L3} run --bench {bench} --methods lbfgs --P {lil} --tag {tag} "${{NET[@]}}" ) & pids+=($!)\n')
    return f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END3}gpus=$(nvidia-smi -L | grep -c '^GPU' || true)
[[ "$gpus" == 2 ]] || {{ echo "THIS JOB NEEDS BOTH A100S OF THE NODE, SEES $gpus: STOPPING"; exit 1; }}
{THREADS24}   # each process's share of the 48 cores (hardware.json records it)
{OPENMP_CHECK}python experiments/p4_checks.py provenance --out "$PKG" --name gpu-{bench}
NET=(--device cuda --out "$PKG" --reference-dir {REF2} {REQ})
# The solo check: NiL-N, P = {P}, seed {seed}, alone on GPU {g} (the one its paired run uses) for 60 s
( export CUDA_VISIBLE_DEVICES={g}
  until_end python {N3} solo --bench {bench} --P {P} --method {method} --seed {seed} --tag gpu-{bench}-solo "${{NET[@]}}" )
# The pairing gate: the same run on both GPUs at once for 60 s. The job stops here, before any budget
# run, unless the one on GPU {g} is within 5% of its solo clock (the report's pairing check, made first)
pids=()
( export CUDA_VISIBLE_DEVICES=0
  until_end python {N3} solo --bench {bench} --P {P} --method {method} --seed {seed} --gate 0 --tag gpu-{bench}-gate0 "${{NET[@]}}" ) & pids+=($!)
( export CUDA_VISIBLE_DEVICES=1
  until_end python {N3} solo --bench {bench} --P {P} --method {method} --seed {seed} --gate 1 --tag gpu-{bench}-gate1 "${{NET[@]}}" ) & pids+=($!)
for pid in "${{pids[@]}}"; do wait "$pid" || true; done
timeout 5m python {N3} gate --bench {bench} --P {P} --method {method} --seed {seed} --gpu {g} --device cuda --out "$PKG" \\
    || {{ echo "THE PAIRING GATE DID NOT PASS (see above): STOPPING BEFORE THE BUDGET RUNS"; exit 1; }}
# The pair: process i on GPU i, network part i/2, then its LiL-N L-BFGS sizes
pids=()
{procs}''' + WAIT_ALL.format(what=f'THE {bench.upper()} PAIR')


SOLO_EXTRA_S = 300                 # the solo check: warm-up, pretraining, 60 s
GATE_EXTRA_S = 300                 # the pairing gate: the same, on both GPUs at once, and the comparison


def gpu_jobs():
    jobs = {'p5g_preflight.slurm': ('preflight', 'shared-gpu', '00:20:00', '''# Package 5, item 3, the A100 set: the preflight, on one A100 (shared-gpu: 8 cores). The first job of
# stage p5g: env.sh locks results/package5_item3_gpu to this commit here. The inputs and the item 3
# tests with CUDA visible. Every other job of the stage waits for it.
# Submit: through scripts/cluster/package5/submit_p5_item3_gpu.sh
''', G_PREFLIGHT_BODY)}
    for bench in BENCHES:
        e = gpu_expected_s(bench)
        jobs[f'p5g_{bench}.slurm'] = (bench, 'gpu-pair', walltime(e), f'''# Package 5, item 3, the A100 set, {bench}: one whole two-A100 node, two processes side by side, one per
# GPU, 24 threads each ({gpu_slots(bench, 0)} and {gpu_slots(bench, 1)} budget runs of 600 s), after the solo check and the
# pairing gate. Expected {e / 3600:.2f} h (the longer process, each run with its error log, pretraining and
# setup); the walltime x {MARGIN}.
# Submit: through scripts/cluster/package5/submit_p5_item3_gpu.sh
''', gpu_pair_body(bench))
    jobs['p5g_report.slurm'] = ('report', 'cpu', '00:30:00', '''# Package 5, item 3, the A100 set: after the jobs (afterany). The paired-against-solo check, the
# time-to-error table, then sacct, su_per_job.csv and results/package5_item3_gpu.tar.gz.
# Submit: through scripts/cluster/package5/submit_p5_item3_gpu.sh
''', report_body('p5g', pairing=True))
    return jobs


def render(name, jobs=JOBS, stage='p5'):
    short, cls, wall, header, body = jobs[name]
    h, m, s = map(int, wall.split(':'))
    body = body.replace('__WALL_S__', str(3600 * h + 60 * m + s))
    return (HEAD + header + f'# lilq-resources: {cls}   (applied by scripts/cluster/sbatch.sh from the cluster profile)\n'
            f'#SBATCH --job-name=lilq-{stage}-{short}\n#SBATCH --time={wall}\n' + TAIL + body)


# The OpenMP library diagnostic (the user's decision, 17 November 2026): stage p5o, its own results folder
# (results/package5_item3_openmp) under its own commit lock. The CPU set's network processes loaded the
# PyTorch wheel's OpenMP library (torch imported first), Package 1's the GCC module's (numpy first), and
# C1 failed on 15 of 18 runs. Three rounds on one whole node, the order of the two libraries
# alternating; each process imports one package first (experiments/p5_3_openmp.py).
OMP = 'experiments/p5_3_openmp.py'
OPENMP_ROUNDS = (('numpy', 'torch'), ('torch', 'numpy'), ('numpy', 'torch'))
UNTIL_END_OMP = UNTIL_END.replace(
    "# keeps every iterate it wrote (the drivers write after each one); the job goes on.\n",
    "# keeps what the processes before it wrote; the job goes on.\n")
assert UNTIL_END_OMP != UNTIL_END


def openmp_body():
    runs = ''.join(f'until_end python {OMP} --first {first} --round {r} --require-library --out "$PKG" '
                   f'--reference-dir {REF2}\n' for r, pair in enumerate(OPENMP_ROUNDS) for first in pair)
    return f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END_OMP}python experiments/p4_checks.py provenance --out "$PKG" --name openmp
# Three rounds, the two orders alternating. Each process: its OpenMP library (it stops if that is not the
# one its first import should give), the warm-up, in round 0 the C1 runs (NiL-N seed 0 at each problem's
# largest size, 110 iterations), then NiL-N and LM at Bratu P = 225 and BL P = 1,024, 30 s each
{runs}timeout 15m python scripts/cluster/package2/stage_report.py --stage p5o --results "$RESULTS" \\
    || echo "REPORT FAILED OR STOPPED (exit $?)"
'''


OPENMP_JOBS = {
    'p5o_openmp.slurm': ('openmp', 'timed-cpu', '00:30:00', '''# Package 5, item 3: the OpenMP library diagnostic, on one whole CPU node (48 threads), as the CPU set
# ran. Six processes of about 3 min (three rounds, numpy first and torch first alternating), then
# results/package5_item3_openmp.tar.gz (the job's own sacct row is added at the assembly).
# Submit: through scripts/cluster/package5/submit_p5_item3_openmp.sh
''', openmp_body()),
}


# The OpenMP library of Package 1's Component A baselines on the CPU (the user's decision, 17 November
# 2026): stage p5k, its own results folder (results/package5_item3_openmp_kovasznay). Package 1's
# Component A processes (Kovasznay, F1 and F2) imported torch first and loaded the wheel's library;
# their CPU runs are timed against LiL-Q's, which loaded the module's. The same rounds as p5o
# (experiments/p5_3_openmp_kovasznay.py); the comparison with Package 1's runs is made at the assembly.
OMPK = 'experiments/p5_3_openmp_kovasznay.py'


def openmp_kovasznay_body():
    runs = ''.join(f'until_end python {OMPK} run --first {first} --round {r} --require-library --out "$PKG"\n'
                   for r, pair in enumerate(OPENMP_ROUNDS) for first in pair)
    return f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END_OMP}python experiments/p4_checks.py provenance --out "$PKG" --name openmp_kovasznay
# Three rounds, the two orders alternating. Each process: its OpenMP library (it stops if that is not the
# one its first import should give), then Package 1's representatives on the CPU, seed 0: F1_18 (201 Adam
# iterations, then L-BFGS; 140 s) and F2_05 (90 s)
{runs}timeout 15m python scripts/cluster/package2/stage_report.py --stage p5k --results "$RESULTS" \\
    || echo "REPORT FAILED OR STOPPED (exit $?)"
'''


OPENMP_KOVASZNAY_JOBS = {
    'p5k_openmp_kovasznay.slurm': ('openmp-kovasznay', 'timed-cpu', '00:40:00', '''# Package 5, item 3: the OpenMP library of Package 1's Component A baselines (Kovasznay, F1 and F2) on
# one whole CPU node (48 threads), as Package 1's CPU runs ran. Six processes of about 4 min (three
# rounds, numpy first and torch first alternating), then results/package5_item3_openmp_kovasznay.tar.gz.
# Submit: through scripts/cluster/package5/submit_p5_item3_openmp_kovasznay.sh
''', openmp_kovasznay_body()),
}


# The C4 diagnostic (the user's decision, 17 November 2026, after the rerun): stage p5e, its own results
# folder (results/package5_item3_c4_diagnostic) under its own commit lock. In the rerun, C4 failed for
# Buckley-Leverett LM (+18.1% with the error log, the module's OpenMP library); in the CPU set (the
# wheel's) it passed. The A100 set waits for this. Each process times the LM pair three times at P = 1,024
# (experiments/p5_3_c4_diagnostic.py). On one whole CPU node: three rounds of three processes, the
# module's library, the wheel's, and the module's with OMP_WAIT_POLICY=ACTIVE, each kind once in each
# position (a Latin square, against drift over the job). On one two-A100 node: two processes side by side,
# 24 threads each, the module's library, as the A100 set runs.
C4D = 'experiments/p5_3_c4_diagnostic.py'
C4D_ROUNDS = (('numpy', 'torch', 'active'), ('torch', 'active', 'numpy'), ('active', 'numpy', 'torch'))


def c4d_cpu_body():
    runs = ''.join(f'until_end {"env OMP_WAIT_POLICY=ACTIVE " if kind == "active" else ""}python {C4D} run '
                   f'--kind {kind} --round {r} --device cpu --require-library --out "$PKG" --reference-dir {REF2}\n'
                   for r, kinds in enumerate(C4D_ROUNDS) for kind in kinds)
    return f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END_OMP}python experiments/p4_checks.py provenance --out "$PKG" --name c4-cpu
# Three rounds of three processes, each kind once in each position. Each process: its OpenMP library (it
# stops if that is not the one its kind should give) and its wait policy (ACTIVE for the 'active' kind,
# unset for the others), the warm-up, then the Buckley-Leverett LM pair (without the error log, then with
# it) three times, 213 iterations each, as the rerun's C4
unset OMP_WAIT_POLICY
{runs}'''


def c4d_gpu_body():
    pair = ''.join(f'''( export CUDA_VISIBLE_DEVICES={g}
  until_end python {C4D} run --kind numpy --round {g} --device cuda --require-library --out "$PKG" --reference-dir {REF2} ) & pids+=($!)
''' for g in (0, 1))
    return f'''LILQ_WALL_S=__WALL_S__   # the walltime, in seconds
{UNTIL_END_OMP}gpus=$(nvidia-smi -L | grep -c '^GPU' || true)
[[ "$gpus" == 2 ]] || {{ echo "THIS JOB NEEDS BOTH A100S OF THE NODE, SEES $gpus: STOPPING"; exit 1; }}
{THREADS24}   # each process's share of the 48 cores, as the A100 set
unset OMP_WAIT_POLICY
python experiments/p4_checks.py provenance --out "$PKG" --name c4-gpu
# Two processes side by side, one per A100, as the A100 set: each its OpenMP library (the module's), the
# warm-up, then the Buckley-Leverett LM pair three times, sized by C4's pilot (about 30 s)
pids=()
{pair}failed=0
for pid in "${{pids[@]}}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || {{ echo "THE C4 PAIR ON THE A100S: A PROCESS FAILED (see above)"; exit 1; }}
'''


C4D_REPORT_BODY = f'''# Each step under its own time limit, so that a stall cannot cost the tarball
timeout 5m python {C4D} summary --out "$PKG" || echo "SUMMARY FAILED OR STOPPED (exit $?)"
timeout 15m python scripts/cluster/package2/stage_report.py --stage p5e --results "$RESULTS" \\
    || echo "REPORT FAILED OR STOPPED (exit $?)"
'''

C4D_JOBS = {
    'p5e_cpu.slurm': ('c4-cpu', 'timed-cpu', '01:00:00', """# Package 5, item 3: the C4 diagnostic on one whole CPU node (48 threads), as the rerun's checks ran.
# Nine processes of about 5 min (three rounds: the module's library, the wheel's, the module's with
# OMP_WAIT_POLICY=ACTIVE).
# Submit: through scripts/cluster/package5/submit_p5_item3_c4_diagnostic.sh
""", c4d_cpu_body()),
    'p5e_gpu.slurm': ('c4-gpu', 'gpu-pair', '00:20:00', """# Package 5, item 3: the C4 diagnostic on one whole two-A100 node (gpu-pair: both A100s, 48 cores, 192
# SU/h), two processes side by side, 24 threads each, as the A100 set runs. About 6 min.
# Submit: through scripts/cluster/package5/submit_p5_item3_c4_diagnostic.sh
""", c4d_gpu_body()),
    'p5e_report.slurm': ('report', 'cpu', '00:15:00', """# Package 5, item 3, the C4 diagnostic: after both jobs (afterany). The summary (its last line: A100 SET
# MAY GO: yes or NO), then sacct, su_per_job.csv and results/package5_item3_c4_diagnostic.tar.gz.
# Submit: through scripts/cluster/package5/submit_p5_item3_c4_diagnostic.sh
""", C4D_REPORT_BODY),
}


def main():
    for name in JOBS:
        (HERE / name).write_bytes(render(name).encode())
    for name in RETIME_JOBS:
        (HERE / name).write_bytes(render(name, RETIME_JOBS, 'p5t').encode())
    for name in A2_JOBS:
        (HERE / name).write_bytes(render(name, A2_JOBS, 'p5a').encode())
    item3 = [(cpu_jobs(), 'p5c'), (gpu_jobs(), 'p5g'), (rerun_jobs(), 'p5r')]
    for jobs, stage in item3:
        for name in jobs:
            (HERE / name).write_bytes(render(name, jobs, stage).encode())
    for name in OPENMP_JOBS:
        (HERE / name).write_bytes(render(name, OPENMP_JOBS, 'p5o').encode())
    for name in OPENMP_KOVASZNAY_JOBS:
        (HERE / name).write_bytes(render(name, OPENMP_KOVASZNAY_JOBS, 'p5k').encode())
    for name in C4D_JOBS:
        (HERE / name).write_bytes(render(name, C4D_JOBS, 'p5e').encode())
    print(f'{len(JOBS) + len(RETIME_JOBS) + len(A2_JOBS) + sum(len(j) for j, _ in item3) + len(OPENMP_JOBS) + len(OPENMP_KOVASZNAY_JOBS) + len(C4D_JOBS)} '
          f'job scripts written in {HERE}')


if __name__ == '__main__':
    main()
