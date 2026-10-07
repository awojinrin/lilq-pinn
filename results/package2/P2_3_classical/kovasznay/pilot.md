# Kovasznay square-system pilot (Package 2, Section 4.2, step 1)

Square spectral collocation, P_N - P_{N-2}: velocity in the tensor Chebyshev space of p_d modes per direction, pressure in p_d - 2; momentum and continuity at the (p_d - 2)^2 interior CGL points, the velocity Dirichlet data at the 4 p_d - 4 boundary CGL points; one continuity row (at the interior point nearest (-0.5, -0.5)) replaced by the pressure pin. Re = 40; Newton from zero; stop at relative coefficient change < 1e-9, K_max = 60; errors on the 301 x 401 test grid. Rank and kappa_2 by SVD of the Jacobian at the final iterate (rank threshold n eps sigma_max).

| p_d | unknowns = equations | pin | rank | kappa_2 | rank, kappa_2 at zero | Newton iterations | eps_u | eps_v | eps_p (pin gauge) | eps_p (mean-free) |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 264 | corner (-0.5, -0.5) | 264 | 1.83e+04 | 264, 4.76e+03 | 10 | 2.32e-02 | 8.82e-02 | 9.17e-01 | 3.81e-02 |
| 10 | 264 | interior (-0.4548, -0.4397) | 264 | 4.39e+03 | 264, 1.89e+03 | 10 | 2.32e-02 | 8.82e-02 | 1.81e-01 | 3.81e-02 |
| 15 | 619 | corner (-0.5, -0.5) | 619 | 3.36e+05 | 619, 1.71e+05 | 6 | 1.85e-05 | 1.16e-04 | 7.18e-03 | 9.06e-05 |
| 15 | 619 | interior (-0.4812, -0.4749) | 619 | 1.00e+05 | 619, 6.40e+04 | 6 | 1.85e-05 | 1.16e-04 | 2.07e-03 | 9.06e-05 |

**Result:** full rank with the corner pin at every size. The interior pin (the fallback) is shown for comparison; it changes only the pressure gauge (the velocity and mean-free pressure errors are identical).

Commit `edaf02ea70bcb6e9657783385ae63c3002193d83`; `python experiments/p2_3_classical.py kovasznay-pilot`.
