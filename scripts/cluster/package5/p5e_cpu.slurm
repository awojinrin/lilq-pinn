#!/bin/bash
# Package 5, item 3: the C4 diagnostic on one whole CPU node (48 threads), as the rerun's checks ran.
# Nine processes of about 5 min (three rounds: the module's library, the wheel's, the module's with
# OMP_WAIT_POLICY=ACTIVE).
# Submit: through scripts/cluster/package5/submit_p5_item3_c4_diagnostic.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5e-c4-cpu
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=3600   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps what the processes before it wrote; the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
python experiments/p4_checks.py provenance --out "$PKG" --name c4-cpu
# Three rounds of three processes, each kind once in each position. Each process: its OpenMP library (it
# stops if that is not the one its kind should give) and its wait policy (ACTIVE for the 'active' kind,
# unset for the others), the warm-up, then the Buckley-Leverett LM pair (without the error log, then with
# it) three times, 213 iterations each, as the rerun's C4
unset OMP_WAIT_POLICY
until_end python experiments/p5_3_c4_diagnostic.py run --kind numpy --round 0 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_c4_diagnostic.py run --kind torch --round 0 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end env OMP_WAIT_POLICY=ACTIVE python experiments/p5_3_c4_diagnostic.py run --kind active --round 0 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_c4_diagnostic.py run --kind torch --round 1 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end env OMP_WAIT_POLICY=ACTIVE python experiments/p5_3_c4_diagnostic.py run --kind active --round 1 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_c4_diagnostic.py run --kind numpy --round 1 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end env OMP_WAIT_POLICY=ACTIVE python experiments/p5_3_c4_diagnostic.py run --kind active --round 2 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_c4_diagnostic.py run --kind numpy --round 2 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
until_end python experiments/p5_3_c4_diagnostic.py run --kind torch --round 2 --device cpu --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
