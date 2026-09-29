#!/bin/bash
# Pre-flight on an A100 (not timed, so not exclusive): versions (torch
# must be 2.10.0), the test suite, and smoke runs of the Component B and C
# drivers. Components B and C wait for this. The Component A checks and
# tests are in 29_A1_gate, so they gate only Component A (Addendum v2.2 2.6).
# lilq-resources: shared-gpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-preflight
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail

# Smoke outputs outside the wave folder (which gets only COMMIT from this job),
# in a folder of their own: the drivers skip completed runs, so a rerun after a
# fix must not find the previous run's smoke results.
PRE="$SCRATCH/lilq-run/preflight/${SLURM_JOB_ID:-manual}"

python -c "from lilq.provenance import assert_torch_version; print('torch', assert_torch_version(), 'OK')"
python -c "import threadpoolctl; print('threadpoolctl', threadpoolctl.__version__, 'OK')"   # hardware.json's thread pools (2.12)
python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
python experiments/component_b.py --smoke --out-root "$PRE"
python experiments/component_b.py --gpu-equivalence --smoke --out-root "$PRE"
python experiments/four_method_tables.py --quick --out-dir "$PRE/four_method"
echo "PREFLIGHT OK"
