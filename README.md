# Hierarchical Bayesian Reanalysis of 36 Wide Binaries with Reported Gravitational Anomaly

**Saad & Ting (2026)**: https://arxiv.org/abs/XXXX.XXXXX

## Repository contents

```
├── run_gamma_analysis.py      # Main analysis script (PyMC model + sampling)
├── posterior_samples.npz      # Pre-computed posterior samples
├── data/                      # Input data (see below)
│   ├── chae_2026_data.csv
│   └── chae_2026_gaia.csv
│   └── README.md         # Instructions on how to curate data files
└── README.md
```

## Data

The input data are taken from [Chae (2026)](https://arxiv.org/abs/2601.21728v2) and Gaia. Place the two CSV files in the `data/` directory:
- `chae_2026_data.csv` — the 36-system clean sample with RVs and masses
- `chae_2025_gaia.csv` — full astrometric data (positions, proper motions, parallaxes)

Detailed instruction has been added in the `data/README.md` file.

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
1. Build and sample the baseline model
2. Build and sample the geometric deprojection model
3. Print summary statistics
4. Save posterior samples to `posterior_samples.npz`
5. Save a comparison plot to `gamma_posterior.png`

**Note:** Sampling takes several hours on a modern CPU.

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
with Reported Gravitational Anomaly"
```

## License

MIT
