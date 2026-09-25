# Running Package 1 on TAMU Grace

Addendum v2.1 Section 2: every timing in the paper comes from one platform,
a single Grace node with its CPU and its A100, held exclusively. These
scripts run all timed work that way; untimed work runs on a shared A100
(B9's NiL runs) or on the laptop (below). The FASTER scripts in
`scripts/hprc/` were the shakedown and are kept for reference.

## 1. Before the first submission (Grace login node)

```bash
module spider PyTorch        # the module names; FASTER's are in env.sh (LILQ_MODULES)
sinfo -p gpu -o "%N %c %m %G %l"   # A100 nodes: cores (scripts assume 48), memory, GPUs, time limit
sinfo -s                     # partition names: finalize needs one without GPUs (CPU_PARTITION)
myproject -l                 # account 132698954494 and its balance
```

If the module names differ, change `LILQ_MODULES` in `scripts/grace/env.sh`.
If A100 nodes have a core count other than 48, change `--cpus-per-task` in
the timed scripts (it must equal the node's cores). If the gpu partition's
time limit is under 22 hours, `31_A_full.slurm` needs splitting.

## 2. Build and upload (laptop), then the environment (Grace, once)

```bash
python scripts/make_hprc_bundle.py          # -> ../lilq-pinn-<sha>.tar.gz
```

```bash
mkdir -p $SCRATCH/lilq-run && tar -xzf lilq-pinn-<sha>.tar.gz -C $SCRATCH/lilq-run
cd $SCRATCH/lilq-run
module load <LILQ_MODULES>
python -m venv --system-site-packages venv
source venv/bin/activate && pip install matplotlib pytest
python -c "import numpy; print(numpy.__version__)"   # must be the module's numpy, not a pip upgrade
```

## 3. Submit

```bash
cd $SCRATCH/lilq-run/lilq-pinn
CPU_PARTITION=<from sinfo -s> bash scripts/grace/submit_all.sh
```

| Script | What | Node | Rough duration |
|---|---|---|---|
| `00_preflight` | tests, smoke runs, check F2 (LM Jacobian vs finite differences, on the A100) | shared A100, 8 cores | < 1 h |
| `10_timed_lilq` | Section 3.3 CPU runs; Kovasznay GPU runs; check B3 with its timings; Section 3.7 | exclusive | ~1.5 h |
| `20_four_method_gpu` | B4 GPU pass, one task per benchmark | exclusive x 4 | 1-4 h each |
| `21_four_method_cpu` | B4 CPU pass at the largest sizes | exclusive x 4 | < 1 h each |
| `30_A_screen` | Component A: search, then 24 configurations x 10 min per family | exclusive x 2 | ~4.5 h each |
| `31_A_full` | selection, top 3 x 5 seeds x 60 min per family, representative | exclusive x 2 | ~16 h each |
| `32_A_cpu` | each representative on the CPU, 5 seeds x 60 min | exclusive x 2 | ~5.5 h each |
| `33_A_float32_and_a1` | F1 float32-Adam run; check A1 | exclusive | ~2.5 h |
| `40_b9_nil_darcy` | B9: NiL Darcy, float64, 4 fields x 3 seeds | shared A100 x 12 | unmeasured on A100; 23 h each on the laptop GPU |
| `90_finalize` | reference/, code/, merged tables, B1, B5 figures, Section 4.6 comparison, B9 table | CPU | minutes |

Everything waits for the pre-flight; Component A's stages chain; finalize
waits for all. Every job resumes where it stopped if resubmitted (for an
array, only the index that stopped: `sbatch --array=2 ...`).

**SUs (estimate).** An exclusive A100 node costs 48 core-SU per hour plus
the GPU surcharge (72 per A100; whether an exclusive node is charged for
one A100 or both, sbatch's "Requested SUs" line on the first submission
shows): 120-192 SU per node-hour. The exclusive jobs total about 70
node-hours, 8,400-13,400 SU. B9's shared runs cost 80 SU per GPU-hour;
at an assumed 2-4 h per run, 1,900-3,800 SU. Total: roughly 10,000-17,000
of the 20,000.

## 4. Untimed work on the laptop

```bash
python experiments/b8_initial_guess.py                       # B8: 128 runs, ~8-9 h on the CPU
python experiments/component_c.py --root results/package1/C_oversampling   # C: 196 runs
python experiments/run_burgers_basis_comparison.py --table3 --output-dir results/package1/B_instrumentation/basis_study   # B6
python experiments/darcy_fv_comparison.py --out-dir results/package1/B_instrumentation/darcy_fv/lil   # B9, LiL part
```

B5 (the residual-band figures) is drawn by `90_finalize` from the Grace logs.

## Resource use

Timed jobs hold the whole node (`--exclusive`, `--mem=0`) and start one
thread per core for BLAS, OpenMP and PyTorch; `hardware.json` in every run
directory records the node's features and GPUs, the thread counts, and
whether the job held the node exclusively. Never run experiments or tests
on a login node.
