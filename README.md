# No Gravitational Anomaly in Wide Binaries from Forward Modelling of 3D Orbits

**Saad & Ting (2026)**: [https://arxiv.org/abs/2603.11015](https://arxiv.org/abs/2603.11015)

We reanalyze the 36 wide-binary systems from Chae (2026) using a hierarchical Bayesian model that infers a global gravity boost factor γ while forward modelling three-dimensional Keplerian orbits. With a broken power-law prior on the semi-major axis (Opik's law below 5 kAU; slope −1.6 above), we obtain **γ = 1.00 ± 0.24**, consistent with Newtonian gravity. The anomaly reported by Chae (2026) is reproduced only when the semi-major axis is replaced by a geometric de-projection of the observed projected separation.

## Repository contents

```
├── run_gamma_analysis.py          # Main analysis script (PyMC model + sampling)
├── posterior_samples.npz          # Pre-computed posterior samples (baseline + geometric)
├── plots/
│   ├── plot_gamma_overlay.py      # Figure 2: gamma and Gamma posterior overlay
│   └── plot_true_anomaly.py       # Appendix B: true anomaly diagnostic
├── data/
│   ├── chae_2026_data.csv
│   ├── chae_2026_gaia.csv
│   └── README.md
└── README.md
```

## Results

| Model | γ (median) | 68% CI | P(γ < 1) |
|---|---|---|---|
| Baseline (broken power-law prior on a) | 1.00 | [0.81, 1.24] | 0.50 |
| Geometric de-projection (no independent a) | 1.56 | [1.38, 1.77] | 0.00 |

## Data

Input data are taken from [Chae (2026)](https://arxiv.org/abs/2601.21728). Place the two CSV files in the `data/` directory:
- `chae_2026_data.csv` — 36-system sample with RVs and masses
- `chae_2026_gaia.csv` — astrometric data (positions, proper motions, parallaxes)

See `data/README.md` for details.

## Requirements

- Python >= 3.10
- numpy, pandas, matplotlib, scipy
- pymc >= 5.0
- arviz

Install with:
```bash
pip install numpy pandas pymc matplotlib arviz scipy
```

## Usage

### Run the full analysis

```bash
python run_gamma_analysis.py
```

This samples both model variants and saves results to `posterior_samples.npz`. Sampling takes several hours on a modern CPU.

### Load pre-computed posteriors

```python
import numpy as np

d = np.load("posterior_samples.npz")
gamma_baseline  = d["baseline"]                # shape (12000,)
gamma_geometric = d["geometric_deprojection"]  # shape (12000,)
```

### Reproduce the figures

```bash
python plots/plot_gamma_overlay.py    # Figure 2: gamma/Gamma posteriors
python plots/plot_true_anomaly.py     # Appendix B: true anomaly diagnostic
```

Both scripts read from the `.npz` posterior files in the parent directory.

## Citation

```
Saad & Ting (2026), "No Gravitational Anomaly in Wide Binaries from
Forward Modelling of 3D Orbits", MNRAS (submitted, revised)
```

## License

MIT
