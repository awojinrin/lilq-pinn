#!/bin/bash
# Package 5, item 3: the OpenMP library of Package 1's Component A baselines (Kovasznay, F1 and F2) on
# one whole CPU node (48 threads), as Package 1's CPU runs ran. Six processes of about 4 min (three
# rounds, numpy first and torch first alternating), then results/package5_item3_openmp_kovasznay.tar.gz.
# Submit: through scripts/cluster/package5/submit_p5_item3_openmp_kovasznay.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5k-openmp-kovasznay
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=2400   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps what the processes before it wrote; the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
python experiments/p4_checks.py provenance --out "$PKG" --name openmp_kovasznay
# Three rounds, the two orders alternating. Each process: its OpenMP library (it stops if that is not the
# one its first import should give), then Package 1's representatives on the CPU, seed 0: F1_18 (201 Adam
# iterations, then L-BFGS; 140 s) and F2_05 (90 s)
until_end python experiments/p5_3_openmp_kovasznay.py run --first numpy --round 0 --require-library --out "$PKG"
until_end python experiments/p5_3_openmp_kovasznay.py run --first torch --round 0 --require-library --out "$PKG"
until_end python experiments/p5_3_openmp_kovasznay.py run --first torch --round 1 --require-library --out "$PKG"
until_end python experiments/p5_3_openmp_kovasznay.py run --first numpy --round 1 --require-library --out "$PKG"
until_end python experiments/p5_3_openmp_kovasznay.py run --first numpy --round 2 --require-library --out "$PKG"
until_end python experiments/p5_3_openmp_kovasznay.py run --first torch --round 2 --require-library --out "$PKG"
timeout 15m python scripts/cluster/package2/stage_report.py --stage p5k --results "$RESULTS" \
    || echo "REPORT FAILED OR STOPPED (exit $?)"
