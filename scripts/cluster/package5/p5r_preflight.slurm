#!/bin/bash
# Package 5, item 3, the CPU network runs again: the preflight. The first job of stage p5r: env.sh
# locks results/package5_item3_cpu_rerun to this commit here. The OpenMP library each driver loads,
# every input the jobs read, the test suite. Every other job of the stage waits for it.
# Submit: through scripts/cluster/package5/submit_p5_item3_rerun.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5r-preflight
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__)"
# The OpenMP library each driver loads: the GCC module's, as Package 1's runs (DECISIONS.md, 17 November 2026)
for m in p5_3_networks p5_3_lil; do
    python -c "import sys; sys.path.insert(0, '.'); import experiments.$m; from lilq.provenance import require_openmp; require_openmp('module')"
done
# Every input the jobs read (fail now, not later), then the test suite
inputs=("$RESULTS/package2_stage2/reference"/bratu_ref_p48.npz
        "$RESULTS/package2_stage2/reference"/burgers_cole_hopf.npz
        "$RESULTS/package2_stage2/reference"/bl_fd_ref_viscous_nu0.1.npz)
for f in "${inputs[@]}"; do
    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }
done
echo "inputs OK (${#inputs[@]} files)"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
echo "PREFLIGHT OK"
