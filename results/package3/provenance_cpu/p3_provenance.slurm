#!/bin/bash
# Package 3: the hardware and software record its jobs did not write (save_provenance), on a
# whole node of the runs' class (timed-cpu, partition medium), in the runs' environment and
# under their source lock. A minute or two; about 2 SU.
# Submit from $SCRATCH/lilq-run/lilq-pinn:
#   LILQ_WAVE=p3 bash scripts/cluster/sbatch.sh ../p3_provenance.slurm
# lilq-resources: timed-cpu   (applied by scripts/cluster/sbatch.sh from the cluster profile)
#SBATCH --job-name=lilq-p3-provenance
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x.%j.out

source scripts/cluster/env.sh
set -euo pipefail
python -c "from lilq.provenance import save_provenance; save_provenance('$PKG/provenance_cpu')"
ls -l "$PKG/provenance_cpu"
python -c "import json; h = json.load(open('$PKG/provenance_cpu/hardware.json')); print(h['cpu'])"
echo "PROVENANCE OK"
