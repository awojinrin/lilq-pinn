#!/bin/bash
# Component A's gate (Addendum v2.2 Section 2.6), on a shared A100: the
# Component A search (saved once, before any run), the Component A tests,
# check F2 (the LM Jacobian against finite differences on this node), check
# A2 (F2 reproducibility), and check A1 (the plain PINN must reach
# eps_u <= 1e-3 in the full GPU budget). Every check exits non-zero on
# failure; the Component A screening job starts only if this job succeeds.
# These checks do not gate Components B and C (00_preflight).
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-A1-gate
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python experiments/component_a.py search --root "$A"      # saved once, before any Component A run
python -m pytest -q $LILQ_COMPONENT_A_TESTS
python experiments/component_a.py f2-jacobian --device cuda --root "$A"
python experiments/component_a.py a2 --device cuda --root "$A"
python experiments/component_a.py a1 --device cuda --root "$A"
echo "A1 GATE OK"
