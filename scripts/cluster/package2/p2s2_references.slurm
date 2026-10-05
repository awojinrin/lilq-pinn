#!/bin/bash
# Package 2, Stage 2: every reference, once, before the compute jobs (which all wait for
# it; their own make-if-missing lines then do nothing). In results/package2_stage2/reference:
#   Bratu, SQ-CGL p = 48 and p = 64, and the C2 record (Section 6.2; item 1's port);
#   Burgers, Cole-Hopf, with the Basdevant and series checks (Section 6.2, check C4);
#   Buckley-Leverett, the finite-difference references refined to 1e-6 (Section 9.1,
#   check C7): viscous nu = 0.1, 0.05, 0.02, 0.01, gravity nu = 0.05, 0.02, and gravity
#   at the paper's nu = 0.1 (item 6b).
# The writers are deterministic and write atomically. Minutes.
# Submit: through scripts/cluster/package2/submit_p2s2.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p2s2-references
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p2_3_classical.py reference --out "$PKG"
python experiments/p2_references.py burgers --out "$PKG"
python experiments/p2_2_nu_refinement.py reference --out "$PKG"
python experiments/p2_9_bases.py --references-only --out "$PKG"
