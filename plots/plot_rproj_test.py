#!/usr/bin/env python3
"""Appendix figure: effect of treating the projected separation as exact vs
uncertain, both using the geometric de-projection. Reads
../posterior_samples.npz; writes gamma_overlay_2.pdf and Gamma_overlay_2.pdf."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "posterior_samples.npz"))
geo = d["geometric_deprojection"].ravel()      # r_perp uncertain
geox = d["geometric_rperp_exact"].ravel()       # r_perp exact
ORANGE, GREEN, RED = "#ff7f0e", "#2ca02c", "#d62728"


def panel(transform, xlim, xlabel, vlines, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    grid = np.linspace(*xlim, 400)
    for g, color, ls, lab in [(geo, ORANGE, "-", r"Geometric $r_{\rm true}$"),
                              (geox, GREEN, "-.", r"Geo. $r_{\rm true}$, exact $r_\perp$")]:
        x = transform(g)
        ax.hist(x, bins=60, density=True, range=xlim, alpha=0.18, color=color)
        ax.plot(grid, gaussian_kde(x)(grid), color=color, lw=2.2, ls=ls, label=lab)
    for xpos, color, lab in vlines:
        ax.axvline(xpos, color=color, ls="--", lw=2, label=lab)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Posterior density", fontsize=14)
    ax.set_xlim(*xlim)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, out), dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote", out)


panel(lambda g: g, (0.75, 2.5), r"$\gamma$ (gravity boost factor)",
      [(1.0, RED, r"Newtonian ($\gamma=1$)"), (1.6, GREEN, r"Chae+26 ($\gamma\approx1.6$)")],
      "gamma_overlay_2.pdf")
panel(lambda g: 0.5 * np.log10(g), (-0.1, 0.3), r"$\Gamma \equiv \log_{10}\sqrt{\gamma}$",
      [(0.0, RED, r"Newtonian ($\Gamma=0$)"),
       (0.5 * np.log10(1.6), GREEN, r"Chae+26 ($\gamma\approx1.6$)")],
      "Gamma_overlay_2.pdf")
