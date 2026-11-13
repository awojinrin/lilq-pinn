#!/bin/bash
# Package 4, item 2f (Section 4.7): c1, c2 and rho_r at every iterate of the 18 runs of 2a and
# of the two extreme sets of 2d at p = 22. After p4_item2, whose runs it reads. Laptop: 66 min
# one run after another at 24 threads; side by side about 1.8 times faster (p = 22: 321 s
# against 567 s), so about 40 min.
# Submit: through scripts/cluster/package4/submit_p4.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p4-item2-constants
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=4500   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps every iterate it wrote (the drivers write after each one); the job goes on.
until_end() { timeout "$((LILQ_WALL_S - SECONDS - 300))s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
seed_runs() {   # seed_runs <seed> [--no-extremes]: one seed's 2f runs, p = 30 last
    local s=$1; shift
    python experiments/p4_2_pb_inclusions.py constants --out "$PKG" --points data/package4 --seeds "$s" --p 10 14 18 22 26 "$@"
    until_end python experiments/p4_2_pb_inclusions.py constants --out "$PKG" --points data/package4 --seeds "$s" --p 30 --no-extremes
}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
pids=()
seed_runs 0 & pids+=($!)
seed_runs 1 --no-extremes & pids+=($!)
seed_runs 2 --no-extremes & pids+=($!)
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo "ITEM 2F: A SEED'S RUNS FAILED (see above)"; exit 1; }
echo "ITEM 2F DONE"
