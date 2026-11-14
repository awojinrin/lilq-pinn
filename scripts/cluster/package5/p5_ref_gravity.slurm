#!/bin/bash
# Package 5, item 1 (Section 3): the fine Buckley-Leverett reference, gravity case, at 24 threads.
# p = 80 (limit 40m), then p = 88 (limit 60m; about 38 GB), then the choice and,
# if needed, the fallback p = 64 with 56. Laptop, scaled: about 14 + 26 min.
# Submit: through scripts/cluster/package5/submit_p5.sh
# lilq-resources: cpu-mem   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5-ref-gravity
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
# The advisor's answer 1: p = 88 with its neighbour p = 80; if either does not finish
# within its limit or does not stop by the rule, this case falls back to p = 64 with 56
# (`choose`, which runs them and writes bl_reference/chosen.json with the reason).
ref() {   # ref <limit> <p>
    timeout "$1" python experiments/p5_1_weights.py reference --case gravity --p-ref "$2" --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference" \
        || echo "REFERENCE gravity p = $2: STOPPED OR FAILED (exit $?)"
}
ref 40m 80
if python experiments/p5_1_weights.py ref-ok --case gravity --p-ref 80 --out "$PKG"; then
    ref 60m 88
fi
python experiments/p5_1_weights.py choose --case gravity --p-ref 88 --out "$PKG" --reference-dir "$RESULTS/package2_stage2/reference"
