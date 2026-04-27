#!/usr/bin/env python3
"""
True anomaly population plot in Chae (2026) Figure 16 style.
Shows composite of posterior PDFs across all 36 systems, compared with
the Kepler-weighted prior curve.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Style ---------------------------------------------------------------
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif"],
    "font.size":         13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
})

def apply_ticks(ax):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in',
                   length=8, width=1.5, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=4, width=1.0, top=True, right=True)

# -------------------------------------------------------------------------
d = np.load('/home/saad.104/Downloads/PEPSI/true_anomaly_posterior_20260427_071021.npz')
nu_flat = d['nu_flat']   # (144000,) degrees in [0, 360]
e_flat  = d['e_flat']

print(f"Overall nu: mean={nu_flat.mean():.1f}, std={nu_flat.std():.1f}")

# -------------------------------------------------------------------------
# Kepler prior: sample M uniformly, e from posterior eccentricities
rng    = np.random.default_rng(42)
n_prior = 500_000
M_samp  = rng.uniform(0, 2*np.pi, n_prior)
e_samp  = rng.choice(e_flat, n_prior, replace=True)

def kepler_E(M, e, n_iter=10):
    E = M.copy()
    for _ in range(n_iter):
        E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
    return E

E_samp       = kepler_E(M_samp, e_samp)
ratio        = np.sqrt(np.clip((1 + e_samp)/(1 - e_samp), 0, None))
nu_prior     = 2.0 * np.arctan2(ratio * np.sin(E_samp/2), np.cos(E_samp/2))
nu_prior_deg = np.rad2deg(nu_prior) % 360

# -------------------------------------------------------------------------
BINS  = np.linspace(0, 360, 19)   # 18 bins × 20 degrees
MIDS  = 0.5 * (BINS[:-1] + BINS[1:])
BIN_W = BINS[1] - BINS[0]

N_sys  = 36
N_samp = len(nu_flat)
hist_counts, _ = np.histogram(nu_flat,      bins=BINS)
prior_counts, _ = np.histogram(nu_prior_deg, bins=BINS)

freq       = hist_counts  * N_sys / N_samp
prior_freq = prior_counts * N_sys / len(nu_prior_deg)

# Prior smooth curve scaled to match histogram total
nu_grid  = np.linspace(0, 360, 500)
nu_rad   = np.deg2rad(nu_grid)
e_med    = float(np.median(e_flat))
p_kepler = (1 - e_med**2)**1.5 / (2*np.pi*(1 + e_med*np.cos(nu_rad))**2)
p_norm   = p_kepler / (p_kepler.sum() * (360/len(nu_grid)))
p_scaled = p_norm * freq.sum() * BIN_W

# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.5))

ax.bar(MIDS, freq, width=BIN_W*0.90, color='#1f77b4', edgecolor='#1f77b4',
       linewidth=0.8, alpha=0.40, label='Posterior (combined)', zorder=2)

ax.plot(nu_grid, p_scaled, color='#ff7f0e', lw=2.2,
        label='Kepler-weighted prior', zorder=3)

ax.set_xlabel(r'True anomaly $\nu\;(^{\circ})$')
ax.set_ylabel('Frequency')
ax.set_xlim(0, 360)
ax.set_ylim(bottom=0)
ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
apply_ticks(ax)

plt.tight_layout()
outpdf = '/home/saad.104/Downloads/PEPSI/true_anomaly_chaestyle.pdf'
outpng = '/home/saad.104/Downloads/PEPSI/true_anomaly_chaestyle.png'
fig.savefig(outpdf, bbox_inches='tight', dpi=200)
fig.savefig(outpng, bbox_inches='tight', dpi=200)
print(f"Saved {outpdf}")
print("DONE.")
