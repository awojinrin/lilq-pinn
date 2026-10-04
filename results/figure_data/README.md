# Field data for the solution-field figures (3 October 2026)

The advisor's reply to wave 4, Section 3. Each file evaluates the **saved
models of the regenerated runs** on the figure's grid. Nothing was
retrained or re-solved.

**How they were made:** `experiments/figure_data.py` at commit `6945d50`
(this branch's base), run on the final package. That package is the final
assembly of waves 1-4, whose derived tables are on branch `package1-final`.
`manifest.json`'s `package` is that package's local path. The run took
about a minute on a laptop CPU.

## Loading

```python
import json, numpy as np
z = np.load('kovasznay.npz')
meta = json.loads(str(z['meta']))       # sources, checks, round-off, grid
u, x, y = z['LiL-Q_u'], z['x'], z['y']  # u[i, j] at (x[i], y[j])
```

**Array layout:** every 2D array is `[i, j]` over the first and second
coordinate (`indexing='ij'`).

**`meta['sources']`:** for each method, the run folder (relative to the
package), the device it ran on, and the commit that produced it.

## Files

| File | Arrays | Grid |
|---|---|---|
| `bratu.npz` | `x`, `y`; `P25_*` and `P225_*` for LiL-Q, LiL-N, NiL-N, NiL-Q | 201 x 201 on [0,1]^2. No exact solution. |
| `burgers.npz` | `x`, `t`; `P625_*` for the four formulations | 201 x 201 in (x, t) |
| `buckley_leverett.npz` | `x`, `t`; per case, `bl_*` (viscous) and `bl_gravity_*`: `reference`, `LiL-Q`, `NiL-N` | 201 x 201 in (x, t), P = 1,024 (N = 32) |
| `kovasznay.npz` | `x`, `y`; `LiL-Q_u/v/p` and `exact_u/v/p` | 301 x 401 on [-0.5,1] x [-0.5,1.5], P = 1,875 |
| `beltrami.npz` | `s`; per face `xm1`, `xp1`, `ym1`, `yp1`, `zm1`, `zp1`: `LiL-Q_u/v/w/p` and `exact_u/v/w/p` | 41 x 41 per face of [-1,1]^3, t = 1 |
| `darcy.npz` | per field `S1`, `S2`, `S3`, `SPE10`: `x`, `y` (cell centres), `FV`, `LiL`, `NiL_float64`, `NiL_float32` | the finite-volume cell centres |
| `elasticity.npz` | `x`, `y`; `LiL_ux/uy` and `exact_ux/uy` | 200 x 200 on [0,1]^2, P = 50 |

**Beltrami's faces:**
- `xm1` and `xp1`: x = -1 and +1, over (y, z) = (s, s);
- `y*`: over (x, z);
- `z*`: over (x, y).

**Runs and pressures:**
- **LiL-Q** is the Section 3.3 CPU paper pass.
- **LiL-N and NiL (seed 0)** are the CPU four-method runs.
- **Darcy's NiL** is B9's (seed 0, trained on the shared A100), and its FV
  pressure is the one B9 stored.
- **Pressures** (Kovasznay, Beltrami) are in each run's own pin gauge. The
  mean-free Kovasznay error is in `meta['checks']`.

## Missing models

**Bratu at P = 25 has no CPU four-method models.** Wave 1's CPU four-method
job ran P = 225 only.
- **What is used instead:** LiL-N, NiL-N and NiL-Q at P = 25 come from the
  **GPU** four-method job, the same method and seed.
- **How they are marked:** `meta['sources']['P25_*']['device']` = `cuda`,
  with a note.
- **The list:** `manifest.json`'s `missing_models`.
- **LiL-Q at P = 25** is the CPU paper pass, as everywhere.

Nothing else is missing.

## Evaluation and checks

**How LiL fields are evaluated:** as tensor products
(`lilq.test_errors.tensor_grid_values`), the code path the runs' logged
test errors used. `meta['evaluation_roundoff']` gives each field's largest
difference from the pointwise evaluation.
- **Everywhere but one case,** it is at most 1.3e-13.
- **Viscous Buckley-Leverett LiL-Q at P = 1,024:** 1.7e-5. Its coefficients
  reach 2.5e9, so its field carries round-off of that order.

**Checks:** each error a run logged is recomputed from these fields
(`meta['checks']`).
- **Darcy:** delta_FV of every LiL and NiL model equals B9's table:
  exactly for LiL and NiL float64, to 1e-7 for float32. The re-evaluated
  LiL pressures equal those the runs stored to 6e-12.
- **Kovasznay:** eps_u, eps_v and the mean-free eps_p agree to 3e-7
  relative.
- **Buckley-Leverett:**
  - gravity eps_u agrees to 2e-9;
  - viscous eps_u is 1.95806e-4 against 1.95802e-4 logged, within the
    round-off above.
- **Elasticity:** both errors are of order 1e-16. The basis reproduces the
  exact solution, so an error panel shows round-off only.
