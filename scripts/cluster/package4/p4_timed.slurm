#!/bin/bash
# Package 4 on one whole node (48 threads): check M0 (package1's Burgers P = 625 and Bratu
# P = 225 paper passes, to all digits), then the timings of items 1a, 1b and 2a (Section 2.3:
# warm-up, three timings, each the median of five under 1 s, stopped at the rule's iterate).
# The node's hardware.json and environment.txt go to provenance/timed/.
# Laptop: about 5 min in all.
# Submit: through scripts/cluster/package4/submit_p4.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p4-timed
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p4_checks.py provenance --out "$PKG" --name timed
python experiments/p4_checks.py m0 --out "$PKG" --package1 "$RESULTS/package1_v2.0.0/package1" || echo "M0 FAILED (exit $?)"
python experiments/p4_1_burgers_gauss.py time --out "$PKG"
python experiments/p4_2_pb_inclusions.py time --out "$PKG" --points data/package4
