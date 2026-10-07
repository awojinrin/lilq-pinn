# NiL-N: Levenberg-Marquardt against L-BFGS

Medians over the seeds; `eps_ref` with its [min, max] over the seeds. L-BFGS: the tables' GPU runs (package1) and their Stage-1 reference errors. No time ratios: the devices differ.

| benchmark | P | LM device | LM iterations | LM final loss | LM eps_ref [min, max] | L-BFGS iterations | L-BFGS final loss | L-BFGS eps_ref [min, max] | target loss |
|---|---|---|---|---|---|---|---|---|---|
| bl | 64 | cpu | 19 | 7.38e-02 | 1.01e-01 [8.31e-02, 1.16e-01] | 185 | 8.30e-02 | 4.84e-02 [2.63e-02, 5.02e-02] | 8.50e-02 |
| bl | 256 | cpu | 26 | 1.17e-02 | 1.37e-02 [1.08e-02, 2.20e-02] | 350 | 1.49e-02 | 1.36e-02 [1.05e-02, 1.98e-02] | 1.50e-02 |
| bl | 576 | cpu | 44 | 7.05e-04 | 2.44e-03 [1.80e-03, 3.17e-03] | 2339 2/3 | 7.50e-04 | 4.83e-03 [4.58e-03, 8.09e-03] | 7.50e-04 |
| bl | 1024 | cpu | 105 | 7.43e-05 | 9.06e-04 [7.42e-04, 1.10e-03] | 7436 2/3 | 7.50e-05 | 1.03e-03 [7.81e-04, 1.23e-03] | 7.50e-05 |
| bratu | 25 | cpu | 28 2/3 | 2.06e-01 | 1.04e+00 [6.57e-02, 1.50e+00] | 232 | 2.49e-01 | 8.49e-02 [6.35e-02, 9.60e-02] | 2.50e-01 |
| bratu | 100 | cpu | 2000 * | 2.34e-04 | 2.28e-03 [1.03e-03, 1.99e-02] | 7422 dagger | 2.63e-03 | 7.13e-03 [5.21e-03, 9.48e-03] | 1.00e-04 |
| bratu | 225 | cpu | 2000 * | 5.27e-06 | 2.35e-04 [2.07e-04, 7.65e-04] | 5484 dagger | 7.58e-04 | 3.65e-03 [1.01e-03, 4.84e-03] | 2.50e-07 |
| burgers | 25 | cpu | 51 | 5.66e-02 | 4.48e-01 [2.33e-01, 1.35e+00] | 256 | 5.98e-02 | 1.30e-01 [1.28e-01, 1.89e-01] | 6.00e-02 |
| burgers | 100 | cpu | 34 | 9.34e-04 | 8.02e-03 [5.41e-03, 8.41e-03] | 1050 | 9.98e-04 | 7.63e-03 [5.48e-03, 8.84e-03] | 1.00e-03 |
| burgers | 225 | cpu | 65 | 1.97e-05 | 7.73e-04 [6.95e-04, 1.02e-03] | 4329 dagger | 3.14e-05 | 9.82e-04 [7.11e-04, 1.05e-03] | 2.00e-05 |
| burgers | 400 | cpu | 413 2/3 | 9.97e-08 | 4.33e-05 [3.68e-05, 1.62e-04] | 4531 dagger | 2.44e-05 | 7.01e-04 [6.17e-04, 7.71e-04] | 1.00e-07 |
| burgers | 625 | cpu | 846 | 5.00e-09 | 1.64e-05 [9.27e-06, 1.91e-05] | 4032 dagger | 2.16e-05 | 6.37e-04 [5.30e-04, 6.92e-04] | 5.00e-09 |
