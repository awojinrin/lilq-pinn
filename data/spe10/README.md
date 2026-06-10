# SPE10 Permeability Grid Dataset

This directory contains the permeability fields used in the Heterogeneous Darcy Flow benchmark (Section 4.8 of the manuscript).

## 1. SPE10 Dataset (`perm_field_SPE10.txt`)
* **Origin**: Extracted from the **SPE10 Model 2 benchmark** (Christie & Blunt, 2001), which is a standard reference problem for flow simulation in highly heterogeneous reservoirs.
* **Extraction details**: Represents a 2D horizontal slice of size $60 \times 220$ grid blocks (extracted from Layer 1, channelized Tarbert formation).
* **Units**: Permeability values are reported in milliDarcy (mD).
* **Heterogeneity**: The field exhibits extreme channelization and high contrast in permeability, presenting a challenging non-smooth coefficient field for spectral collocation.

## 2. Spatially Correlated Log-Normal Fields (S1–S3)
The fields `perm_field_S1.txt`, `perm_field_S2.txt`, and `perm_field_S3.txt` are synthetic log-normal fields generated as spatial correlation test cases. They can be re-generated using `scripts/generate_permeability.py` with the following parameters:

* **Grid Dimensions**: $60 \times 220$
* **Log-Mean ($\mu_{\ln K}$)**: $2.0$
* **Log-Standard Deviation ($\sigma_{\ln K}$)**: $3.0$
* **Generation Process**: White noise is generated and smoothed with a Gaussian kernel in Fourier space:
  $$H(k_x, k_y) = \exp\left(-2 \pi^2 \ell_c^2 (k_x^2 + k_y^2)\right)$$
  where $\ell_c$ is the correlation length. The smoothed field is normalized and exponentiated:
  $$K(x, y) = \exp\left(\mu_{\ln K} + \sigma_{\ln K} \cdot \bar{G}(x, y)\right)$$
* **Specific Parameters**:
  * **S1**: Correlation length $\ell_c = 3.0$, Seed = $100$
  * **S2**: Correlation length $\ell_c = 5.0$, Seed = $200$
  * **S3**: Correlation length $\ell_c = 8.0$, Seed = $300$
