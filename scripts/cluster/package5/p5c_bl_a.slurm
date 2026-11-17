#!/bin/bash
# Package 5, item 3, the CPU set, on one whole node (48 threads): 20 budget runs of 600 s, one at a
# time (bl: network part 0/2, then LiL-N L-BFGS). Expected 3.63 h (each run with
# its error log, pretraining and setup); the walltime x 1.1.
# Submit: through scripts/cluster/package5/submit_p5_item3_cpu.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5c-bl-a
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=14400   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps the runs that finished: a run writes its records when it ends, so the one in progress is
# lost (a resubmitted job runs it again and skips the finished ones). The job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
python experiments/p4_checks.py provenance --out "$PKG" --name cpu-bl-a
# The network runs (part 0/2 of NiL-N, NiL-Q and LM over the sizes and seeds), each to the 600 s budget,
# one at a time, then LiL-N L-BFGS at P = 64, 576
until_end python experiments/p5_3_networks.py run --bench bl --part 0/2 --device cpu --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" --tag cpu-bl-a
until_end python experiments/p5_3_lil.py run --bench bl --methods lbfgs --P 64 576 --device cpu --out "$PKG" \
    --reference-dir "$RESULTS/package2_stage2/reference" --tag cpu-bl-a
