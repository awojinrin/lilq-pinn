#!/bin/bash
# Wave 4, Component A, F2 (option 3): the top three by validation residual
# of wave 2's 24 screening runs (copied into results/wave4), F2_10, F2_05 and
# F2_14, pinned with --keep-top (the advisor's reply on wave 3, item 2.4; the
# recomputed validation top three is recorded beside them) -- and run them with seeds 0-4 at 60 minutes each on
# the allocation of wave 2's job 31 (one A100, 48 cores), then the
# representative: the lowest median validation residual. About 15.5 h.
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-A-f2-full
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/component_a.py select --family F2 --device cuda --root "$A" --keep-top F2_10 F2_05 F2_14
python experiments/component_a.py full --family F2 --device cuda --root "$A"
