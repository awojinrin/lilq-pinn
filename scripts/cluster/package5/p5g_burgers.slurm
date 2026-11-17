#!/bin/bash
# Package 5, item 3, the A100 set, burgers: one whole two-A100 node, two processes side by side, one per
# GPU, 24 threads each (26 and 26 budget runs of 600 s), after the solo check. Expected
# 4.87 h (the longer process, each run with its error log, pretraining and setup); the walltime x 1.1.
# Submit: through scripts/cluster/package5/submit_p5_item3_gpu.sh
# lilq-resources: gpu-pair   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5g-burgers
#SBATCH --time=05:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=19800   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps every iterate it wrote (the drivers write after each one); the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
gpus=$(nvidia-smi -L | grep -c '^GPU' || true)
[[ "$gpus" == 2 ]] || { echo "THIS JOB NEEDS BOTH A100S OF THE NODE, SEES $gpus: STOPPING"; exit 1; }
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24   # each process's share of the 48 cores (hardware.json records it)
python experiments/p4_checks.py provenance --out "$PKG" --name gpu-burgers
NET=(--device cuda --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference")
# The solo check: NiL-N, P = 625, seed 0, alone on GPU 0 (the one its paired run uses) for 60 s
( export CUDA_VISIBLE_DEVICES=0
  until_end python experiments/p5_3_networks.py solo --bench burgers --P 625 --method nil_n --seed 0 --tag gpu-burgers-solo "${NET[@]}" )
# The pair: process i on GPU i, network part i/2, then its LiL-N L-BFGS sizes
pids=()
( export CUDA_VISIBLE_DEVICES=0
  until_end python experiments/p5_3_networks.py run --bench burgers --part 0/2 --tag gpu-burgers-0 "${NET[@]}"
  until_end python experiments/p5_3_lil.py run --bench burgers --methods lbfgs --P 36 196 484 --tag gpu-burgers-0 "${NET[@]}" ) & pids+=($!)
( export CUDA_VISIBLE_DEVICES=1
  until_end python experiments/p5_3_networks.py run --bench burgers --part 1/2 --tag gpu-burgers-1 "${NET[@]}"
  until_end python experiments/p5_3_lil.py run --bench burgers --methods lbfgs --P 100 324 676 900 --tag gpu-burgers-1 "${NET[@]}" ) & pids+=($!)
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo "THE BURGERS PAIR: A PROCESS FAILED (see above)"; exit 1; }
echo "THE BURGERS PAIR DONE"
