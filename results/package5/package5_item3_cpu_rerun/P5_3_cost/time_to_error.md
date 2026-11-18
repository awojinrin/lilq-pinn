# Package 5, item 3: time to reach an error

Seconds on the run's clock (the solver from its initial coefficients; setup and pretraining beside it, in brackets); the smallest over the sizes of the median over the seeds, and the size (P) that gives it; a dash where no size reaches the level.

## Bratu, cpu

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.81 (5.85), P = 225 | - | - | - | - |
| NiL-Q (L-BFGS) | 8.54 (8.67), P = 100 | 185 (185), P = 225 | - | - | - |
| NiL-N (LM) | 1.75 (3.07), P = 225 | 9.73 (12.4), P = 225 | 300 (302), P = 225 | - | - |

## Burgers, cpu

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 2.82 (3.01), P = 225 | 44.1 (44.1), P = 225 | - | - | - |
| NiL-Q (L-BFGS) | 8 (8.27), P = 100 | 57.1 (57.3), P = 400 | - | - | - |
| NiL-N (LM) | 0.686 (1.23), P = 100 | 2.13 (2.19), P = 225 | 7.44 (7.57), P = 400 | 55.1 (55.4), P = 625 | - |

## Buckley-Leverett (viscous), cpu

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 3.48 (3.73), P = 64 | 245 (245), P = 576 | - | - | - |
| NiL-Q (L-BFGS) | 12.4 (12.8), P = 64 | 244 (245), P = 576 | - | - | - |
| NiL-N (LM) | 0.922 (1.24), P = 64 | 13.8 (14.4), P = 1024 | 39.6 (40.1), P = 576 | - | - |

