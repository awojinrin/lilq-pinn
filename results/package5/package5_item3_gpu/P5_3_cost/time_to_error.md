# Package 5, item 3: time to reach an error

Seconds on the run's clock (the solver from its initial coefficients; setup and pretraining beside it, in brackets); the smallest over the sizes of the median over the seeds, and the size (P) that gives it; a dash where no size reaches the level.

## Bratu, cuda

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 9.76 (9.93), P = 225 | - | - | - | - |
| NiL-Q (L-BFGS) | 15.7 (15.7), P = 225 | 238 (238), P = 225 | - | - | - |
| NiL-N (LM) | 0.643 (0.682), P = 225 | 4.38 (4.41), P = 225 | 142 (142), P = 225 | - | - |
| LiL-N (L-BFGS) | 0.364 (0.379), P = 100 | 0.456 (0.471), P = 100 | 3.87 (3.91), P = 225 | - | - |

## Burgers, cuda

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.44 (5.49), P = 225 | 42.7 (43.2), P = 625 | - | - | - |
| NiL-Q (L-BFGS) | 12.7 (13.1), P = 225 | 68 (68), P = 400 | - | - | - |
| NiL-N (LM) | 0.413 (0.536), P = 400 | 0.878 (0.919), P = 400 | 2.37 (2.42), P = 400 | 9.97 (12.4), P = 625 | - |
| LiL-N (L-BFGS) | 0.368 (0.373), P = 100 | 0.551 (0.572), P = 196 | 1.05 (1.09), P = 324 | 1.8 (1.87), P = 484 | 6.58 (6.66), P = 676 |

## Buckley-Leverett (viscous), cuda

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.68 (5.82), P = 64 | 164 (164), P = 1024 | - | - | - |
| NiL-Q (L-BFGS) | 14.6 (14.7), P = 256 | 249 (249), P = 576 | - | - | - |
| NiL-N (LM) | 0.487 (0.677), P = 576 | 1.67 (1.94), P = 1024 | 7.77 (8.04), P = 1024 | - | - |
| LiL-N (L-BFGS) | 1.86 (2.01), P = 256 | 2.24 (2.39), P = 256 | 14.8 (15.3), P = 1024 | - | - |

