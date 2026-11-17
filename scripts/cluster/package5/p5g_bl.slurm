#!/bin/bash
# Package 5, item 3, the A100 set, bl: one whole two-A100 node, two processes side by side, one per
# GPU, 24 threads each (20 and 20 budget runs of 600 s), after the solo check and the
# pairing gate. Expected 3.79 h (the longer process, each run with its error log, pretraining and
# setup); the walltime x 1.1.
# Submit: through scripts/cluster/package5/submit_p5_item3_gpu.sh
# lilq-resources: gpu-pair   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5g-bl
#SBATCH --time=04:15:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=15300   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps the runs that finished: a run writes its records when it ends, so the one in progress is
# lost (a resubmitted job runs it again and skips the finished ones). The job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
gpus=$(nvidia-smi -L | grep -c '^GPU' || true)
[[ "$gpus" == 2 ]] || { echo "THIS JOB NEEDS BOTH A100S OF THE NODE, SEES $gpus: STOPPING"; exit 1; }
export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24   # each process's share of the 48 cores (hardware.json records it)
# The OpenMP library each driver loads: the GCC module's, as Package 1's runs (DECISIONS.md, 17 November 2026)
for m in p5_3_networks p5_3_lil; do
    python -c "import sys; sys.path.insert(0, '.'); import experiments.$m; from lilq.provenance import require_openmp; require_openmp('module')"
done
python experiments/p4_checks.py provenance --out "$PKG" --name gpu-bl
NET=(--device cuda --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" --require-openmp module)
# The solo check: NiL-N, P = 1024, seed 0, alone on GPU 1 (the one its paired run uses) for 60 s
( export CUDA_VISIBLE_DEVICES=1
  until_end python experiments/p5_3_networks.py solo --bench bl --P 1024 --method nil_n --seed 0 --tag gpu-bl-solo "${NET[@]}" )
# The pairing gate: the same run on both GPUs at once for 60 s. The job stops here, before any budget
# run, unless the one on GPU 1 is within 5% of its solo clock (the report's pairing check, made first)
pids=()
( export CUDA_VISIBLE_DEVICES=0
  until_end python experiments/p5_3_networks.py solo --bench bl --P 1024 --method nil_n --seed 0 --gate 0 --tag gpu-bl-gate0 "${NET[@]}" ) & pids+=($!)
( export CUDA_VISIBLE_DEVICES=1
  until_end python experiments/p5_3_networks.py solo --bench bl --P 1024 --method nil_n --seed 0 --gate 1 --tag gpu-bl-gate1 "${NET[@]}" ) & pids+=($!)
for pid in "${pids[@]}"; do wait "$pid" || true; done
timeout 5m python experiments/p5_3_networks.py gate --bench bl --P 1024 --method nil_n --seed 0 --gpu 1 --device cuda --out "$PKG" \
    || { echo "THE PAIRING GATE DID NOT PASS (see above): STOPPING BEFORE THE BUDGET RUNS"; exit 1; }
# The pair: process i on GPU i, network part i/2, then its LiL-N L-BFGS sizes
pids=()
( export CUDA_VISIBLE_DEVICES=0
  until_end python experiments/p5_3_networks.py run --bench bl --part 0/2 --tag gpu-bl-0 "${NET[@]}"
  until_end python experiments/p5_3_lil.py run --bench bl --methods lbfgs --P 64 576 --tag gpu-bl-0 "${NET[@]}" ) & pids+=($!)
( export CUDA_VISIBLE_DEVICES=1
  until_end python experiments/p5_3_networks.py run --bench bl --part 1/2 --tag gpu-bl-1 "${NET[@]}"
  until_end python experiments/p5_3_lil.py run --bench bl --methods lbfgs --P 256 1024 --tag gpu-bl-1 "${NET[@]}" ) & pids+=($!)
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo "THE BL PAIR: A PROCESS FAILED (see above)"; exit 1; }
echo "THE BL PAIR DONE"
