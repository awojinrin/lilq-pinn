#!/bin/bash
# Pre-flight on a Grace A100 node (not timed, so not exclusive): versions,
# the test suite, smoke runs of every driver, and check F2 -- the LM
# Jacobian against finite differences on this node, which the addendum
# requires before any Component A run -- and check A2 (F2 reproducibility).
# Everything else waits for this.
#SBATCH --job-name=lilq-preflight
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --account=132698954494
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%j.out

source scripts/grace/env.sh
set -euo pipefail

python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests/ -q
python experiments/component_b.py --smoke --out-root "$SCRATCH/lilq-run/preflight"
python experiments/component_b.py --gpu-equivalence --smoke --out-root "$SCRATCH/lilq-run/preflight"
python experiments/four_method_tables.py --quick --out-dir "$SCRATCH/lilq-run/preflight/four_method"
python experiments/component_a.py f2-jacobian --device cuda --root "$A"
python experiments/component_a.py a2 --device cuda --root "$A"
python experiments/component_a.py search --root "$A"      # saved once, before any Component A run
echo "PREFLIGHT OK"
