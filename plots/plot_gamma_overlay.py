#!/usr/bin/env python3
"""Figure: posterior overlay of gamma (and Gamma = log10 sqrt(gamma)) for the
baseline (broken power-law prior) and geometric de-projection models.
Reads ../posterior_samples.npz; writes gamma_overlay.pdf and Gamma_overlay.pdf."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "posterior_samples.npz"))
base = d["baseline"].ravel()                 # baseline, broken power-law prior on a
geo = d["geometric_deprojection"].ravel()    # geometric de-projection
BLUE, ORANGE, GREEN, RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"


def overlay(values, transform, xlim, xlabel, vlines, extra_hlines, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    grid = np.linspace(*xlim, 400)
    for g, color, ls, lab in values:
        x = transform(g)
        ax.hist(x, bins=60, density=True, range=xlim, alpha=0.18, color=color)
        ax.plot(grid, gaussian_kde(x)(grid), color=color, lw=2.2, ls=ls, label=lab)
    for xpos, color, lab in vlines:
        ax.axvline(xpos, color=color, ls="--", lw=2, label=lab)
    for ypos, lab in extra_hlines:
        ax.axhline(ypos, color="gray", ls=":", lw=1.5, label=lab)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Posterior density", fontsize=14)
    ax.set_xlim(*xlim)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, out), dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote", out)


series = [(base, BLUE, "-", "Baseline model"),
          (geo, ORANGE, "-.", r"Geometric $r_{\rm true}$")]

overlay(series, lambda g: g, (0.5, 2.5), r"$\gamma$ (gravity boost factor)",
        [(1.0, RED, r"Newtonian ($\gamma=1$)"), (1.6, GREEN, r"Chae+26 ($\gamma\approx1.6$)")],
        [], "gamma_overlay.pdf")

overlay(series, lambda g: 0.5 * np.log10(g), (-1, 1), r"$\Gamma \equiv \log_{10}\sqrt{\gamma}$",
        [(0.0, RED, r"Newtonian ($\Gamma=0$)"),
         (0.5 * np.log10(1.6), GREEN, r"Chae+26 ($\gamma\approx1.6$)")],
        [(0.5, r"Prior $\Gamma\sim\mathcal{U}(-1,1)$")], "Gamma_overlay.pdf")
