#!/bin/bash
# Timed LiL-Q work on the A100, one exclusive A100 node: the Section 3.3
# Kovasznay GPU runs (paper and K_max = 60 passes), then check B3 -- GPU
# against CPU gelsy at every size, with the gelsy / gels / GPU-QR timings on
# one system (the CPU timings from this node's CPU, the same model as the
# CPU nodes'; Addendum v2.2 Section 4.1).
# lilq-resources: timed   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-timed-lilq-gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python experiments/component_b.py --benchmarks kovasznay --devices cuda --out-root "$PKG"
python experiments/component_b.py --gpu-equivalence --out-root "$PKG"
