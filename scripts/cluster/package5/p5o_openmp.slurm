#!/bin/bash
# Package 5, item 3: the OpenMP library diagnostic, on one whole CPU node (48 threads), as the CPU set
# ran. Six processes of about 3 min (three rounds, numpy first and torch first alternating), then
# results/package5_item3_openmp.tar.gz (the job's own sacct row is added at the assembly).
# Submit: through scripts/cluster/package5/submit_p5_item3_openmp.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5o-openmp
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=1800   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps what the processes before it wrote; the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
python experiments/p4_checks.py provenance --out "$PKG" --name openmp
# Three rounds, the two orders alternating. Each process: its OpenMP library (it stops if that is not the
# one its first import should give), the warm-up, in round 0 the C1 runs (NiL-N seed 0 at each problem's
# largest size, 110 iterations), then NiL-N and LM at Bratu P = 225 and BL P = 1,024, 30 s each
until_end python experiments/p5_3_openmp.py --first numpy --round 0 --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_openmp.py --first torch --round 0 --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_openmp.py --first torch --round 1 --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_openmp.py --first numpy --round 1 --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_openmp.py --first numpy --round 2 --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_openmp.py --first torch --round 2 --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
timeout 15m python scripts/cluster/package2/stage_report.py --stage p5o --results "$RESULTS" \
    || echo "REPORT FAILED OR STOPPED (exit $?)"
