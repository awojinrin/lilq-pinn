# Halton points for the Poisson–Boltzmann problem

`halton_seed{0,1,2}.npz` hold the interior collocation points of the
Poisson–Boltzmann problem (`problems/poisson_boltzmann.py`), one draw per seed.
Each file has the first 8,000 points of a scrambled Halton sequence that lie
inside the domain:

| Key | Contents |
|---|---|
| `x`, `y` | the points, in order (8,000 each) |
| `seed` | the scrambling seed |
| `n` | the number of points (8,000) |
| `scipy_version` | the SciPy version that drew them (1.17.0) |

The draws come from `pb.halton(8000, seed)` and are written by
`python experiments/p4_2_pb_inclusions.py points --out <root>`.

**Why the points are shipped.** Scrambled Halton points depend on the SciPy
version. SciPy 1.17.0 reproduces the reference points, which were drawn with
SciPy 1.18.1. Older versions, such as the 1.13.1 on the cluster, may not. Every
point set of the experiments is the first n points of one of these draws,
because the first n points of a draw are the same for any longer draw. The
cluster runs therefore read these files. They do not draw again.

`p4_2_pb_inclusions.py compare-points` records how the local SciPy's draw
compares with these files (`points/halton_compare.json`). That comparison is a
record only and is never used to build a point set.
