#!/bin/bash
# Pre-flight on an A100 (not timed, so not exclusive): versions,
# the test suite, smoke runs of every driver, and check F2 -- the LM
# Jacobian against finite differences on this node, which the addendum
# requires before any Component A run -- and check A2 (F2 reproducibility).
# Everything else waits for this.
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-preflight
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

python -c "from lilq.provenance import assert_torch_version; print('torch', assert_torch_version(), 'OK')"
python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests/ -q
python experiments/component_b.py --smoke --out-root "$SCRATCH/lilq-run/preflight"
python experiments/component_b.py --gpu-equivalence --smoke --out-root "$SCRATCH/lilq-run/preflight"
python experiments/four_method_tables.py --quick --out-dir "$SCRATCH/lilq-run/preflight/four_method"
python experiments/component_a.py f2-jacobian --device cuda --root "$A"
python experiments/component_a.py a2 --device cuda --root "$A"
python experiments/component_a.py search --root "$A"      # saved once, before any Component A run
echo "PREFLIGHT OK"
