#!/bin/bash
# Package 5, item 4 (Section 5): 4a logged (2a) and 2c at p = 34, 38, 42, seeds 0 to 2; 4b's 45
# runs; delta_P by gelsy and gelsd for every space. cpu-mem: six processes, those at P = 1,600
# each holding the H2 quadrature's basis (about 2.3 GB). Laptop: 1.9 h one after another at 16
# threads; side by side about 1.1 h.
# Submit: through scripts/cluster/package5/submit_p5.sh
# lilq-resources: cpu-mem   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5-item4
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=9000   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps every iterate it wrote (the drivers write after each one); the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P4A=(--out "$PKG" --points data/package5)
# Six processes side by side, 4 threads each (24 in all). 4a, one seed each (2a logged, then
# 2c), each followed by part of the rest: 4b up to P = 800, delta_P of Package 4's and 4a's
# spaces, delta_P of 4b's spaces. 4b at P = 1,600, one sigma each.
pids=()
( until_end python experiments/p5_4_pb.py lilq --seeds 0 "${P4A[@]}"; until_end python experiments/p5_4_pb.py under --seeds 0 "${P4A[@]}"
  until_end python experiments/p5_4_pb.py rf --P 100 200 400 800 "${P4A[@]}" ) & pids+=($!)
( until_end python experiments/p5_4_pb.py lilq --seeds 1 "${P4A[@]}"; until_end python experiments/p5_4_pb.py under --seeds 1 "${P4A[@]}"
  until_end python experiments/p5_4_pb.py delta --which p4 4a --out "$PKG" ) & pids+=($!)
( until_end python experiments/p5_4_pb.py lilq --seeds 2 "${P4A[@]}"; until_end python experiments/p5_4_pb.py under --seeds 2 "${P4A[@]}"
  until_end python experiments/p5_4_pb.py delta --which 4b --out "$PKG" ) & pids+=($!)
for sigma in 2 4 8; do
    until_end python experiments/p5_4_pb.py rf --P 1600 --sigma "$sigma" "${P4A[@]}" & pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo "ITEM 4: A PROCESS FAILED (see above)"; exit 1; }
echo "ITEM 4 DONE"
