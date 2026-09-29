#!/bin/bash
# Wave 1 of 3 (Addendum v2.2 Section 4.2): validation and cheap results,
# requested ~1,410 SU (~1,770 if both GPUs of an exclusive node are charged). Into results/wave1:
#   00 preflight -> 29 A1 gate (check A1, A2, F2, the Component A search)
#                -> 10a / 10b (timed LiL-Q, CPU node / A100 node)
#                -> 42 Component C, 43 B6
#                -> 20 / 21 for Bratu only (four-method GPU and CPU passes)
#   -> 91 wave report: results/wave1_report.tar.gz, to send the advisor.
# Nothing else is submitted until the advisor has replied.
LILQ_WAVE=1
source "$(dirname "$0")/submit_lib.sh"
confirm_balance "~1,410 SU from the walltimes (~1,770 if both GPUs of an exclusive node are charged)"

submit pre  $S/00_preflight.slurm
submit gate $S/29_A1_gate.slurm          --dependency=afterok:$pre
submit lcpu $S/10a_timed_lilq_cpu.slurm  --dependency=afterok:$pre
submit lgpu $S/10b_timed_lilq_gpu.slurm  --dependency=afterok:$pre
submit cc   $S/42_component_c.slurm      --dependency=afterok:$pre
submit b6   $S/43_basis_study.slurm      --dependency=afterok:$pre
submit fmg  $S/20_four_method_gpu.slurm  --dependency=afterok:$pre --array=0 --time=03:00:00   # Bratu only (8 h is for the largest benchmark)
submit fmc  $S/21_four_method_cpu.slurm  --dependency=afterok:$pre --array=0
submit rep  $S/91_wave_report.slurm      --dependency=afterany:$gate:$lcpu:$lgpu:$cc:$b6:$fmg:$fmc
[[ "$DRY_RUN" == 1 ]] || echo "Wave 1 submitted. When job $rep finishes, send the advisor results/wave1_report.tar.gz,"
[[ "$DRY_RUN" == 1 ]] || echo "with the SUs charged per job (myproject -l, or the AMS report)."
