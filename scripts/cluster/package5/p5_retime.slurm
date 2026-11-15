#!/bin/bash
# Package 5, item 2: the PyTorch L-BFGS at Burgers p = 10 re-timed on one whole node (48 threads), as
# the timed job of 14 November 2026 ran it, with the warm-up. Then results/package5_retime.tar.gz (the
# job's own sacct row is added at the assembly). Laptop: a few seconds; the job, a few minutes.
# Submit: through scripts/cluster/package5/submit_p5_retime.sh
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p5t-retime
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python experiments/p4_checks.py provenance --out "$PKG" --name timed
# Item 2, the one re-timed run: Burgers p = 10, the PyTorch L-BFGS, by one logged pass after an
# untimed warm-up of 5 iterations (warmup_s in its result.json)
python experiments/p5_2_nlls.py run --out "$PKG" --reference-dir "$RESULTS/package2_stage1/reference" --points data/package4 --problem burgers --p 10 --solver lbfgs
python scripts/cluster/package2/stage_report.py --stage p5t --results "$RESULTS"
