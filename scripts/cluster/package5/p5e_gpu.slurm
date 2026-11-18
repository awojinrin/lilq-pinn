#!/bin/bash
# Package 5, item 3: the C4 diagnostic on one whole two-A100 node (gpu-pair: both A100s, 48 cores, 192
# SU/h), two processes side by side, 24 threads each, as the A100 set runs. About 6 min.
# Submit: through scripts/cluster/package5/submit_p5_item3_c4_diagnostic.sh
# lilq-resources: gpu-pair   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5e-c4-gpu
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=1200   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps what the processes before it wrote; the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
gpus=$(nvidia-smi -L | grep -c '^GPU' || true)
[[ "$gpus" == 2 ]] || { echo "THIS JOB NEEDS BOTH A100S OF THE NODE, SEES $gpus: STOPPING"; exit 1; }
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24   # each process's share of the 48 cores, as the A100 set
unset OMP_WAIT_POLICY
python experiments/p4_checks.py provenance --out "$PKG" --name c4-gpu
# Two processes side by side, one per A100, as the A100 set: each its OpenMP library (the module's), the
# warm-up, then the Buckley-Leverett LM pair three times, sized by C4's pilot (about 30 s)
pids=()
( export CUDA_VISIBLE_DEVICES=0
  until_end python experiments/p5_3_c4_diagnostic.py run --kind numpy --round 0 --device cuda --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" ) & pids+=($!)
( export CUDA_VISIBLE_DEVICES=1
  until_end python experiments/p5_3_c4_diagnostic.py run --kind numpy --round 1 --device cuda --require-library --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" ) & pids+=($!)
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo "THE C4 PAIR ON THE A100S: A PROCESS FAILED (see above)"; exit 1; }
