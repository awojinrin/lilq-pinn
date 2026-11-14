#!/bin/bash
# Package 5 on one whole node (48 threads): the timings of item 4a (9 runs), then item 2 (42 runs:
# LiL-Q and least_squares by the protocol, the L-BFGS solvers by one logged pass). The node's
# hardware.json and environment.txt go to provenance/timed/. Laptop: about 25 min.
# Submit: through scripts/cluster/package5/submit_p5.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5-timed
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=3600   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps every iterate it wrote (the drivers write after each one); the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
python experiments/p4_checks.py provenance --out "$PKG" --name timed
# Item 4a's timings (Section 5: 2a at p = 34, 38, 42, seeds 0 to 2; the paper's protocol)
python experiments/p5_4_pb.py time --out "$PKG" --points data/package5
# Item 2: the protocol for LiL-Q and the least-squares solvers; the two L-BFGS solvers by their
# one logged pass (the advisor's answer 2)
until_end python experiments/p5_2_nlls.py run --out "$PKG" --reference-dir "$RESULTS/package2_stage1/reference" --points data/package4
