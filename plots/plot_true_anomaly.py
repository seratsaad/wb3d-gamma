#!/usr/bin/env python3
"""Appendix figure: population-level posterior of the true anomaly nu, compared
to the Kepler time-weighted prior for the inferred eccentricities. Reads
../true_anomaly_samples.npz; writes true_anomaly_chaestyle.pdf."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "true_anomaly_samples.npz"))
nu = np.mod(d["nu_deg"].ravel(), 360.0)
e = d["e"].ravel()
ORANGE = "#ff7f0e"


def kepler_prior(grid_deg, e):
    """Time-weighted true-anomaly pdf averaged over eccentricities (per degree)."""
    nu = np.deg2rad(grid_deg)
    e = e[np.isfinite(e)]
    e = e[(e >= 0) & (e < 0.999)]
    if e.size > 4000:
        e = np.random.default_rng(0).choice(e, 4000, replace=False)
    pdf = np.zeros_like(nu)
    for ei in e:
        w = (1 - ei ** 2) ** 1.5 / (1 + ei * np.cos(nu)) ** 2
        pdf += w / np.trapezoid(w, nu)
    return pdf / e.size * (np.pi / 180.0)


fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(nu, bins=np.arange(0, 360 + 20, 20), density=True, alpha=0.6,
        color="#6baed6", edgecolor="white", label="Posterior (combined)")
grid = np.linspace(0, 360, 361)
ax.plot(grid, kepler_prior(grid, e), color=ORANGE, lw=2.5, label="Kepler-weighted prior")
ax.set_xlabel(r"True anomaly $\nu$ ($^\circ$)", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_xlim(0, 360)
ax.set_xticks(np.arange(0, 361, 60))
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "true_anomaly_chaestyle.pdf"), dpi=300, bbox_inches="tight")
print("wrote true_anomaly_chaestyle.pdf")
