#!/bin/bash
# Package 4 (the advisor's instructions of 12 November 2026): the preflight.
# The first Package 4 job: env.sh locks results/package4 to this commit here. Versions, every
# input the jobs read (fail now, not later), the test suite, the points folder and Grace's
# scipy against the shipped Halton points. Every other Package 4 job waits for it.
# Submit: through scripts/cluster/package4/submit_p4.sh
# lilq-resources: cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p4-preflight
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
# M0's package1 runs, item 1's Cole-Hopf reference (Package 2), item 2's shipped Halton points
inputs=("$RESULTS/package1_v2.0.0/package1"/B_instrumentation/burgers_P625_cpu_paper/iterations.csv
        "$RESULTS/package1_v2.0.0/package1"/B_instrumentation/bratu_P225_cpu_paper/iterations.csv
        "$RESULTS/package2_stage1/reference"/burgers_cole_hopf.npz
        data/package4/halton_seed0.npz data/package4/halton_seed1.npz data/package4/halton_seed2.npz)
for f in "${inputs[@]}"; do
    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }
done
echo "inputs OK (${#inputs[@]} files)"
python -m pytest tests/ -q $(printf -- '--ignore=%s ' $LILQ_COMPONENT_A_TESTS)
# The deliverables' points folder (the shipped draws and the evaluation quadrature), and
# Grace's own scipy against the shipped draws (a record: no set is built from Grace's draws)
python experiments/p4_2_pb_inclusions.py points --out "$PKG" --points data/package4
python experiments/p4_2_pb_inclusions.py compare-points --out "$PKG" --points data/package4
echo "PREFLIGHT OK"
