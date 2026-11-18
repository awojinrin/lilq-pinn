# Package 5, item 3: time to reach an error

Seconds from the starting data to the iterate: the run's clock plus its setup and pretraining (the networks' pretraining; the LiL methods' system and initial coefficients, on Buckley-Leverett the fit of the initial profile), as the advisor's reply of 17 November 2026 asks; the smallest over the sizes of the median over the seeds, and the size (P) that gives it. Then the optimizer alone (the clock from the initial coefficients), with its size if another one gives its smallest. A dash where no size reaches the level.

## Bratu, cpu

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.85 (P = 225); optimizer only 5.81 | - | - | - | - |
| NiL-Q (L-BFGS) | 8.67 (P = 100); optimizer only 8.54 | 185 (P = 225); optimizer only 185 | - | - | - |
| NiL-N (LM) | 3.07 (P = 100); optimizer only 1.75 (P = 225) | 12.4 (P = 225); optimizer only 9.73 | 302 (P = 225); optimizer only 300 | - | - |
| LiL-N (L-BFGS) | 0.147 (P = 100); optimizer only 0.133 | 0.176 (P = 100); optimizer only 0.162 | 1.35 (P = 225); optimizer only 1.31 | - | - |
| LiL-N (trf) | 0.084 (P = 100); optimizer only 0.0706 | 0.084 (P = 100); optimizer only 0.0706 | 0.34 (P = 225); optimizer only 0.306 | - | - |
| LiL-Q | 0.0435 (P = 100); optimizer only 0.0299 | 0.0435 (P = 100); optimizer only 0.0299 | 0.23 (P = 225); optimizer only 0.194 | - | - |
| square collocation (SQ-CGL) | 0.0024 (P = 25); optimizer only 0.000233 | 0.0024 (P = 25); optimizer only 0.000233 | 0.00919 (P = 100); optimizer only 0.00355 | 0.0198 (P = 225); optimizer only 0.00865 | - |

## Bratu, cuda

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 9.93 (P = 225); optimizer only 9.76 | - | - | - | - |
| NiL-Q (L-BFGS) | 15.7 (P = 225); optimizer only 15.7 | 238 (P = 225); optimizer only 238 | - | - | - |
| NiL-N (LM) | 0.682 (P = 225); optimizer only 0.643 | 4.41 (P = 225); optimizer only 4.38 | 142 (P = 225); optimizer only 142 | - | - |
| LiL-N (L-BFGS) | 0.379 (P = 100); optimizer only 0.364 | 0.471 (P = 100); optimizer only 0.456 | 3.91 (P = 225); optimizer only 3.87 | - | - |

## Burgers, cpu

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 3.01 (P = 225); optimizer only 2.82 | 44.1 (P = 225); optimizer only 44.1 | - | - | - |
| NiL-Q (L-BFGS) | 8.27 (P = 100); optimizer only 8 | 57.3 (P = 400); optimizer only 57.1 | - | - | - |
| NiL-N (LM) | 1.23 (P = 225); optimizer only 0.686 (P = 100) | 2.19 (P = 225); optimizer only 2.13 | 7.57 (P = 400); optimizer only 7.44 | 55.4 (P = 625); optimizer only 55.1 | - |
| LiL-N (L-BFGS) | 0.0873 (P = 100); optimizer only 0.0832 | 0.191 (P = 196); optimizer only 0.178 | 0.329 (P = 324); optimizer only 0.282 | 0.809 (P = 484); optimizer only 0.756 (P = 676) | 1.49 (P = 676); optimizer only 1.38 |
| LiL-N (trf) | 0.0512 (P = 100); optimizer only 0.0472 | 0.187 (P = 196); optimizer only 0.179 | 0.415 (P = 324); optimizer only 0.398 | 1.12 (P = 484); optimizer only 1.09 | 2.08 (P = 676); optimizer only 2.01 |
| LiL-Q | 0.0221 (P = 100); optimizer only 0.0181 | 0.132 (P = 196); optimizer only 0.113 | 0.231 (P = 324); optimizer only 0.211 | 0.55 (P = 484); optimizer only 0.506 | 0.974 (P = 676); optimizer only 0.891 |

## Burgers, cuda

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.49 (P = 225); optimizer only 5.44 | 43.2 (P = 625); optimizer only 42.7 | - | - | - |
| NiL-Q (L-BFGS) | 13.1 (P = 225); optimizer only 12.7 | 68 (P = 400); optimizer only 68 | - | - | - |
| NiL-N (LM) | 0.536 (P = 225); optimizer only 0.413 (P = 400) | 0.919 (P = 400); optimizer only 0.878 | 2.42 (P = 400); optimizer only 2.37 | 12.4 (P = 625); optimizer only 9.97 | - |
| LiL-N (L-BFGS) | 0.373 (P = 100); optimizer only 0.368 | 0.572 (P = 196); optimizer only 0.551 | 1.09 (P = 324); optimizer only 1.05 | 1.87 (P = 484); optimizer only 1.8 | 6.66 (P = 676); optimizer only 6.58 |

## Buckley-Leverett (viscous), cpu

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 3.73 (P = 64); optimizer only 3.48 | 245 (P = 576); optimizer only 245 | - | - | - |
| NiL-Q (L-BFGS) | 12.8 (P = 256); optimizer only 12.4 (P = 64) | 245 (P = 576); optimizer only 244 | - | - | - |
| NiL-N (LM) | 1.24 (P = 64); optimizer only 0.922 | 14.4 (P = 1024); optimizer only 13.8 | 40.1 (P = 576); optimizer only 39.6 | - | - |
| LiL-N (L-BFGS) | 0.881 (P = 256); optimizer only 0.766 | 1.03 (P = 256); optimizer only 0.911 | 20.2 (P = 1024); optimizer only 19.8 | - | - |
| LiL-N (trf) | 1.07 (P = 256); optimizer only 0.99 | 1.2 (P = 256); optimizer only 1.11 | 14 (P = 1024); optimizer only 13.6 | - | - |
| LiL-Q | 0.327 (P = 256); optimizer only 0.244 | 0.388 (P = 256); optimizer only 0.305 | 3.44 (P = 1024); optimizer only 3.05 | - | - |

## Buckley-Leverett (viscous), cuda

| method | 0.01 | 0.001 | 0.0001 | 1e-05 | 1e-06 |
|---|---|---|---|---|---|
| NiL-N (L-BFGS) | 5.82 (P = 64); optimizer only 5.68 | 164 (P = 1024); optimizer only 164 | - | - | - |
| NiL-Q (L-BFGS) | 14.7 (P = 256); optimizer only 14.6 | 249 (P = 576); optimizer only 249 | - | - | - |
| NiL-N (LM) | 0.677 (P = 576); optimizer only 0.487 | 1.94 (P = 1024); optimizer only 1.67 | 8.04 (P = 1024); optimizer only 7.77 | - | - |
| LiL-N (L-BFGS) | 2.01 (P = 256); optimizer only 1.86 | 2.39 (P = 256); optimizer only 2.24 | 15.3 (P = 1024); optimizer only 14.8 | - | - |

