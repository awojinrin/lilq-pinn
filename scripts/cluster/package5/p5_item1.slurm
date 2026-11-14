#!/bin/bash
# Package 5, item 1 (Section 3): the 96 runs of Bratu, Burgers and Buckley-Leverett at the four
# weights, K_max = 60, every iterate stored. After both references (whatever their outcome).
# cpu-mem: six processes, each with its own quadrature of the norm of X (up to 176 x 176).
# Laptop: 49 min one after another at 16 threads; side by side about 30 min.
# Submit: through scripts/cluster/package5/submit_p5.sh
# lilq-resources: cpu-mem   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5-item1
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=4500   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps every iterate it wrote (the drivers write after each one); the job goes on.
until_end() { timeout "$((LILQ_WALL_S - SECONDS - 300))s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# Six processes side by side, 4 threads each (24 in all), one family each. A Buckley-Leverett
# family reads its case's fine reference from bl_reference/chosen.json and never runs one.
pids=()
for family in bratu burgers bl_cheb_viscous bl_cheb_gravity bl_sine_viscous bl_sine_gravity; do
    until_end python experiments/p5_1_weights.py runs --only "$family" --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" & pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo "ITEM 1: A PROCESS FAILED (see above)"; exit 1; }
echo "ITEM 1 DONE"
