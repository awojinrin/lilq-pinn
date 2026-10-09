#!/bin/bash
# Package 3, item 1 (Section 3) and checks K3, K6, K8, on one whole node (48 threads):
# - the face and slab constants of all six grids (minutes);
# - B1 levels 1 and 2 (1 h cap each), with rho_r;
# - K6: B1 level 1 again (bit-identical); K3: B1 level 1 with the pin weight x 1e-3, x 1e3;
# - K8: the paper's pinned Beltrami run and viscous BL P = 576 against package1 (48 threads,
#   as package1).
# Submit: through scripts/cluster/package3/submit_p3.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-beltrami-small
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# capped <limit> <command...>: the run under its own wall cap (the advisor's, Section 3.4). A run
# stopped by its cap keeps what it logged (Section 8.2); the job goes on to the next run.
capped() { local cap=$1; shift; timeout "$cap" "$@" || echo "STOPPED OR FAILED (exit $?, cap $cap): $*"; }
B=experiments/p3_1_beltrami_certified.py
for s in B1 B2 B3; do for l in 1 2; do python $B constants --size $s --level $l --out "$PKG"; done; done
capped 1h python $B run --size B1 --level 1 --out "$PKG"
capped 1h python $B run --size B1 --level 1 --out "$PKG" --tag _rerun
capped 1h python $B run --size B1 --level 1 --out "$PKG" --pin-scale 1e-3 --tag _pin1e-3
capped 1h python $B run --size B1 --level 1 --out "$PKG" --pin-scale 1e3 --tag _pin1e3
capped 1h python $B run --size B1 --level 2 --out "$PKG"
python experiments/p3_k0_k8.py k8 --out "$PKG" --package1 "$RESULTS/package1_v2.0.0/package1"
