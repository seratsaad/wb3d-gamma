# Hierarchical Bayesian Reanalysis of Wide Binaries with Reported Gravitational Anomaly

**Saad & Ting (2026)** — Paper II in the WB3D series

## Overview

We reanalyze the 36 wide-binary systems from [Chae (2026)](https://arxiv.org/abs/XXXX.XXXXX) using a hierarchical Bayesian model that infers a single global gravity boost factor γ across all systems while fitting three-dimensional orbital elements.

We compare two model variants:
- **Baseline model**: semi-major axis is a free parameter; the true 3D separation is computed from Kepler's equation.
- **Geometric deprojection model**: no independent semi-major axis; the true separation is derived from the observed projected separation and viewing geometry, analogous to the approach of Chae (2026).

### Key result

| Model | γ (median) | 68% CI | P(γ > 1) |
|---|---|---|---|
| Baseline | 1.15 | [0.94, 1.44] | 0.75 |
| Geometric deprojection | 1.56 | [1.38, 1.77] | ~1.00 |

The two models differ only in the treatment of the orbital separation, yet the inferred γ shifts from consistency with Newtonian gravity to consistency with the reported gravitational anomaly.

## Repository contents

```
├── run_gamma_analysis.py      # Main analysis script (PyMC model + sampling)
├── posterior_samples.npz      # Pre-computed posterior samples
├── data/                      # Input data (see below)
│   ├── chae_2026_data.csv
│   └── chae_2026_gaia.csv
└── README.md
```

## Data

The input data are taken from [Chae (2026)](https://arxiv.org/abs/XXXX.XXXXX). Place the two CSV files in a `data/` directory:
- `chae_2026_clean_sample.csv` — the 36-system clean sample with RVs and masses
- `chae_2025_complete.csv` — full astrometric data (positions, proper motions, parallaxes)

## Requirements

- Python ≥ 3.10
- numpy
- pandas
- pymc ≥ 5.0
- pytensor
- matplotlib
- arviz

Install with:
```bash
pip install numpy pandas pymc matplotlib arviz
```

## Usage

### Run the full analysis

```bash
python run_gamma_analysis.py
```

This will:
1. Build and sample the baseline model (~292 parameters)
2. Build and sample the geometric deprojection model
3. Print summary statistics
4. Save posterior samples to `posterior_samples.npz`
5. Save a comparison plot to `gamma_posterior.png`

**Note:** Sampling takes several hours on a modern CPU (4 chains × 2000 tuning + 3000 sampling steps each).

### Load pre-computed posteriors

```python
import numpy as np

d = np.load("posterior_samples.npz")
gamma_baseline = d["baseline"]           # shape (12000,)
gamma_deproj = d["geometric_deprojection"]  # shape (12000,)
a_over_rproj = d["a_over_rproj_medians"]    # shape (36,) — per-system median a/r_proj
```

## Citation

If you use this code or results, please cite:

```
Saad & Ting (2026), "Hierarchical Bayesian Reanalysis of Wide Binaries
with Reported Gravitational Anomaly", Paper II
```

## License

MIT
