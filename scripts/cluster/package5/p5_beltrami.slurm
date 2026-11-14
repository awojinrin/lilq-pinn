#!/bin/bash
# Package 5, item 1 (Section 3): the six Beltrami runs at 24 threads (B2 about 18 GB). Package 3's
# whole-node times: 398 s (B1) and 1,301 s (B2) per run, about 1.4 h in all; the walltime
# allows 24 threads to be up to twice as slow.
# Submit: through scripts/cluster/package5/submit_p5.sh
# lilq-resources: cpu-mem   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5-beltrami
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=10800   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps every iterate it wrote (the drivers write after each one); the job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
# B1 and B2 at level 1, lambda_bc = lambda_ic = 1e3, 1e4, 1e6 (the delivered runs have 10),
# pin_scale 1, eight iterates; at 24 threads (the advisor's reply: untimed, no whole node).
for size in B1 B2; do
    for lam in 1e3 1e4 1e6; do
        until_end python experiments/p3_1_beltrami_certified.py run --size "$size" --level 1 --lambda-bc "$lam" --lambda-ic "$lam" \
            --tag "_lam$lam" --out "$PKG"
    done
done
