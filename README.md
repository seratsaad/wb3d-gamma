# No Gravitational Anomaly in Wide Binaries from Forward Modelling of 3D Orbits

**Saad & Ting (2026)**: [https://arxiv.org/abs/2603.11015](https://arxiv.org/abs/2603.11015)

We reanalyze the 36 wide-binary systems from Chae (2026) using a hierarchical Bayesian model that infers a global gravity boost factor γ while forward modelling three-dimensional Keplerian orbits. With a broken power-law prior on the semi-major axis (Öpik's law below 5 kAU; slope −1.6 above), we obtain **γ = 1.00 ± 0.24**, consistent with Newtonian gravity. The anomaly reported by Chae (2026) is reproduced only when the semi-major axis is replaced by a geometric de-projection of the observed projected separation. An injection–recovery test confirms the method recovers the injected γ without systematic bias.

## Repository contents

```
├── run_gamma_analysis.py          # Main analysis (PyMC model + sampling)
├── run_injection.py               # Injection–recovery validation
├── posterior_samples.npz          # gamma posteriors (baseline / geometric / geometric r_perp-exact)
├── true_anomaly_samples.npz       # pooled true-anomaly and eccentricity posteriors
├── injection_recovery.json        # injection–recovery results
├── plots/
│   ├── plot_gamma_overlay.py      # Figure: gamma / Gamma posterior overlay
│   ├── plot_rproj_test.py         # Appendix: r_perp exact vs uncertain (geometric)
│   ├── plot_true_anomaly.py       # Appendix: true-anomaly diagnostic
│   └── plot_injection.py          # Appendix: injection–recovery
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
| Geometric, r_perp treated as exact | 1.59 | [1.40, 1.82] | 0.00 |

**Method validation.** Injection–recovery on synthetic catalogs matched to the
real sample: injecting γ = 1.0 recovers a median γ ≈ 0.95, and injecting γ = 1.6
recovers a median γ ≈ 1.68 — unbiased in the median and cleanly separated (see
`run_injection.py` and `plots/plot_injection.py`).

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

This samples both model variants (the baseline uses the broken power-law prior by
default) and saves results to `posterior_samples.npz`. The model builder is
`build_gamma_model(data, include_sma=True/False, sma_prior="bpl"|"lognormal")`.

### Run the injection–recovery validation

```bash
python run_injection.py --gammas 1.0,1.6 --n-datasets 5 \
    --draws 3000 --tune 2000 --chains 4 --sma-prior bpl
```

### Load pre-computed posteriors

```python
import numpy as np

d = np.load("posterior_samples.npz")
gamma_baseline   = d["baseline"]                # broken power-law prior on a
gamma_geometric  = d["geometric_deprojection"]  # geometric de-projection
gamma_geom_exact = d["geometric_rperp_exact"]   # geometric, r_perp exact
```

### Reproduce the figures

```bash
python plots/plot_gamma_overlay.py    # gamma / Gamma posterior overlay
python plots/plot_rproj_test.py       # r_perp exact vs uncertain (geometric)
python plots/plot_true_anomaly.py     # true-anomaly diagnostic
python plots/plot_injection.py        # injection–recovery
```

All plotting scripts read the `.npz` / `.json` files in the repository root.

## Citation

```
Saad & Ting (2026), "No Gravitational Anomaly in Wide Binaries from
Forward Modelling of 3D Orbits", MNRAS (submitted, revised)
```

## License

MIT
