# Running Package 1 on TAMU HPRC (Grace or FASTER)

Every job of Package 1 v2.0 and Addendum v2.1 runs on the cluster. Timed work
(everything whose times go into the paper) holds a whole A100 node
(`--exclusive`), as Addendum Section 2 requires; untimed work runs on a
shared A100 or on CPU cores. The same job scripts serve both clusters: the
cluster-specific values live in `profiles/grace.sh` and `profiles/faster.sh`,
and `sbatch.sh` applies them according to each script's
`# lilq-resources:` class (`timed`, `shared-gpu`, `cpu`).

## 1. Before the first submission (login node)

```bash
module spider PyTorch                    # module names (profile's LILQ_MODULES)
sinfo -p gpu -o "%N %c %m %G %l"         # A100 nodes: cores (Grace profile: 48), memory, GPUs, time limit
sinfo -s                                 # a partition without GPUs, for CPU_PARTITION
myproject -l                             # the account and its balance
```

Edit the profile if the module names or the A100 nodes' core count differ.
The gpu partition's time limit must allow `31_A_full` (22 h).

## 2. Build and upload (laptop), then the environment (cluster, once)

```bash
python scripts/make_hprc_bundle.py          # -> ../lilq-pinn-<sha>.tar.gz
```

```bash
mkdir -p $SCRATCH/lilq-run && tar -xzf lilq-pinn-<sha>.tar.gz -C $SCRATCH/lilq-run
cd $SCRATCH/lilq-run
module load <LILQ_MODULES from the profile>
python -m venv --system-site-packages venv
source venv/bin/activate && pip install matplotlib pytest
pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126   # about 3 GB
export PYTHONPATH=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'):$PYTHONPATH
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"   # 2.10.0+cu126, and the module's numpy
```

The PyTorch module puts its own torch (2.9.1) on `PYTHONPATH`, ahead of the
venv. `env.sh` puts the venv first in every job, and the preflight fails
unless torch is 2.10.0 (Addendum v2.2 Section 2.4: 2.9.1's line search
ignores `max_eval`, which changes the L-BFGS evaluation counts).

## 3. Submit

```bash
cd $SCRATCH/lilq-run/lilq-pinn
CLUSTER=grace bash scripts/cluster/submit_all.sh
```

`SKIP_A=1` holds Component A back; submit it later with `bash scripts/cluster/submit_component_a.sh`
(it adds its own finalize). On Grace the CPU jobs use `medium` (1-day limit); FASTER's profile uses `cpu`.

To resubmit one script (after a walltime kill, say), use the wrapper so the
profile's resources apply: `bash scripts/cluster/sbatch.sh scripts/cluster/20_four_method_gpu.slurm --array=2`.
Every job resumes where it stopped: completed runs are skipped, and a B9
Darcy network resumes from its last 5,000-epoch checkpoint.

Every run saves its trained model next to its logs (`solution.pt`,
`network.pt`, F1's `model.pt`, F2's `theta.pt`); see "Reloading trained
models" in the top-level README. Keep `package1_results/` whole when copying
it off the cluster: the models are what later figures are made from.

| Script | What | Class | Rough duration |
|---|---|---|---|
| `00_preflight` | tests, smoke runs, checks F2 and A2 on the A100, saves the Component A search | shared-gpu | < 1 h |
| `10_timed_lilq` | Section 3.3 CPU runs; Kovasznay GPU runs; check B3 with its timings; Section 3.7 | timed | ~1.5 h |
| `20_four_method_gpu` | B4 GPU pass, one task per benchmark | timed x 4 | 1-4 h each |
| `21_four_method_cpu` | B4 CPU pass at the largest sizes | timed x 4 | < 1 h each |
| `30_A_screen` | Component A: 24 configurations x 10 min per family | timed x 2 | ~4.5 h each |
| `31_A_full` | selection, top 3 x 5 seeds x 60 min per family, representative | timed x 2 | ~16 h each |
| `32_A_cpu` | each representative on the CPU, 5 seeds x 60 min | timed x 2 | ~5.5 h each |
| `33_A_float32_and_a1` | F1 float32-Adam run; check A1 | timed | ~2.5 h |
| `40_b9_nil_darcy` | B9: NiL Darcy in float64, 4 fields x 3 seeds (LiL alongside) | shared-gpu x 12 | est. 2-4 h each (23 h on the laptop GPU) |
| `41_b8_initial_guess` | B8: 128 runs, one task per (case, guess); networks on the A100 | shared-gpu x 4 | ~2-3 h each |
| `42_component_c` | Component C: 196 LiL-Q runs | cpu | a few hours |
| `43_basis_study` | B6: the Burgers basis study | cpu | < 1 h |
| `90_finalize` | reference/, code/, merged tables (B4, B8, B9), B1, B5 figures, Section 4.6 comparison | cpu | minutes |

**SUs.** Charged for time used: cores x hours plus a GPU surcharge per GPU-hour
(Grace A100 72; FASTER A100 128). On Grace, an exclusive A100 node is 48 core-SU
plus 72 or 144 per hour depending on whether both of its A100s are charged
(the "Requested SUs" line sbatch prints shows it): 120-192 SU per node-hour,
~70 node-hours of timed jobs, 8,400-13,400 SU. Shared-A100 jobs cost 80 SU per
hour: B9 1,900-3,800, B8 ~700. CPU jobs ~150. Total roughly 11,000-18,000 of
20,000. On FASTER, A100s sit in 4-16-GPU nodes, so an exclusive node holds
several A100s at 128 SU each: the timed jobs there would cost several times more.

## Resource use

Timed jobs hold the whole node and start one thread per core for BLAS,
OpenMP and PyTorch; `hardware.json` in every run directory records the node's
features and GPUs, the thread counts, and whether the job held the node
exclusively. Never run experiments or tests on a login node.
