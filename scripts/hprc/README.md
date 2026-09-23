# Running Component B on TAMU HPRC

All of Component B (Computational_Package_1_v2.md Section 3) as SLURM jobs.
Every job resumes where it stopped if resubmitted, and logs failures with
their tracebacks instead of dying.

## 1. Build and upload (local machine)

Commit first -- the bundle records the commit and any uncommitted diff.

```bash
python scripts/make_hprc_bundle.py          # -> ../lilq-pinn-<sha>.tar.gz
```

Upload it to the cluster (MobaXterm drag-and-drop), then on the login node:

```bash
mkdir -p $SCRATCH/lilq-run && tar -xzf lilq-pinn-<sha>.tar.gz -C $SCRATCH/lilq-run
```

## 2. Python environment (once)

```bash
cd $SCRATCH/lilq-run
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0
python -m venv --system-site-packages venv
source venv/bin/activate && pip install matplotlib pytest
```

## 3. Submit

```bash
cd $SCRATCH/lilq-run/lilq-pinn
bash scripts/hprc/submit_all.sh
squeue -u $USER
```

| Script | What | Resources | Expected time |
|---|---|---|---|
| `00_preflight` | test suite + smoke run of everything, on the cluster's libraries | 8 cores, 1 T4 | minutes |
| `10_logged_reruns_cpu` | Section 3.3, all CPU runs (52) | 24 cores | ~30 min |
| `11_logged_reruns_gpu` | Section 3.3 Kovasznay GPU runs + check B3 | 8 cores, 1 A100 | minutes |
| `20_four_method_gpu` | Section 3.4 GPU pass, array of 4 (one per benchmark) | 8 cores, 1 A100 each | hours each |
| `21_four_method_cpu` | Section 3.4 CPU pass at the largest sizes, array of 4 | 24 cores each | hours each |
| `30_basis_study_and_beltrami_pinned` | Sections 3.6 and 3.7 | 24 cores | ~1 h |
| `40_finalize` | merge four-method CSVs, check B1, Section 3.5 figures | 2 cores | seconds |

Everything waits for the pre-flight job to pass; the finalize job waits for
all others. Results land in `$SCRATCH/lilq-run/lilq-pinn/results/package1/`
(the spec's `package1_results/` layout). `$SCRATCH` is not backed up --
copy results home when done.

If a job hits its walltime, resubmit that script (for an array, only the
index that stopped: `sbatch --array=2 scripts/hprc/20_four_method_gpu.slurm`).

## Resource use

- **Cores and threads.** Every job requests one task and `--cpus-per-task`
  cores, and starts exactly that many BLAS/PyTorch threads. Inside a job the
  cores are yours alone (SLURM reserves them), so using all of them is
  correct, not greedy; starting more threads than cores would only slow the
  job down. Check a finished job's efficiency with `seff <jobid>`.
- **24 cores** matches the paper's thread count for the timed CPU work.
  These solves are moderate in size; if you want to trim, time Kovasznay at
  P = 1,875 with 8/16/24 cores once and keep the smallest count before the
  speed levels off.
- **GPU type.** Everything runs in float64. A100s run it at full rate; T4s at
  about 1/32. Only the pre-flight uses a T4.
- **`--exclusive`** (commented out in the timed CPU jobs): Section 2 wants
  nothing else on the machine during timed runs. Uncommenting it reserves
  the whole node -- charged for every core of it, for the job's duration.
  Cheap for the ~30-minute job; your call.
- **Never run the experiments or the tests on a login node** (limit: 8
  cores, 60 minutes). `salloc` first for anything interactive.
- **SUs.** Charged for time actually used, not the walltime requested. The
  four-method GPU array dominates. Grace's A100 surcharge is about half of
  FASTER's, so for the long GPU jobs change `--account` (and the partition,
  after checking `sinfo` on Grace) in `20_four_method_gpu.slurm`.

## Cluster vs. the paper's machine

Section 2 specifies the paper's machines (24-thread CPU, RTX 5080). Iteration
counts, errors, residuals and condition numbers carry over; wall-clock times
do not. `hardware.json` in every run folder records the node, the CPU/GPU and
the SLURM allocation, so the report can state exactly where each number came
from.
