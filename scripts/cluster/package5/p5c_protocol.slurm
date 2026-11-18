#!/bin/bash
# Package 5, item 3, the CPU set, on one whole node (48 threads): LiL-Q, trf and the square
# collocation by the usual protocol on the three problems, then checks C2 and C4.
# Submit: through scripts/cluster/package5/submit_p5_item3_cpu.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5c-protocol
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
LILQ_WALL_S=2700   # the walltime, in seconds
# until_end <command...>: the command, stopped 5 min before the walltime if still running. It
# keeps the runs that finished: a run writes its records when it ends, so the one in progress is
# lost (a resubmitted job runs it again and skips the finished ones). The job goes on.
# With no time left it skips the command and says so (timeout 0 would mean no limit).
until_end() { local left=$((LILQ_WALL_S - SECONDS - 300))
    (( left > 0 )) || { echo "NOT RUN, NO TIME LEFT: $*"; return 0; }
    timeout "${left}s" "$@" || echo "STOPPED OR FAILED (exit $?): $*"; }
python experiments/p4_checks.py provenance --out "$PKG" --name protocol
# LiL-Q, trf and (Bratu) the square collocation by the usual protocol: one logged pass, then three
# timed passes (the clock per iterate, the median of the timestamps)
for bench in bratu burgers bl; do
    until_end python experiments/p5_3_lil.py run --bench "$bench" --methods lilq trf square --device cpu --out "$PKG" \
        --reference-dir "$RESULTS/package2_stage2/reference" --tag cpu-protocol
done
# Checks C2 and C4: NiL-N and LM at the largest size, about 30 s each (sized by a pilot), without and
# with the error log
for bench in bratu burgers bl; do
    until_end python experiments/p5_3_networks.py checks --bench "$bench" --device cpu --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" --tag cpu-checks
done
