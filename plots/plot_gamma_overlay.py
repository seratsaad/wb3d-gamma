#!/usr/bin/env python3
"""
Replot Figure 2: gamma posterior overlay (revised).
Matches original figure style — serif font, histogram approach, inward ticks.
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

def hist_plot(ax, samples, color, ls, label, bins=60):
    ax.hist(samples, bins=bins, density=True, alpha=0.20, color=color)
    counts, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, color=color, lw=2, ls=ls, label=label)

# ---- Load data -----------------------------------------------------------
d_broken  = np.load('/home/saad.104/Downloads/PEPSI/powerlaw_sma_broken_20260427_053909.npz')
g_broken  = d_broken['broken_gamma']       # median ≈ 1.00

d_old     = np.load('/home/saad.104/Downloads/PEPSI/gamma_tests_20260310_060118.npz')
g_nosma   = d_old['no_sma']               # median ≈ 1.56

Gam_broken = 0.5 * np.log10(g_broken)
Gam_nosma  = 0.5 * np.log10(g_nosma)

CHAE_GAM   = 0.5 * np.log10(1.6)          # ≈ 0.102

# ---- PANEL 1: gamma ------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6.5, 5.0))

hist_plot(ax1, g_broken, color='#1f77b4', ls='-',  label='Baseline model')
hist_plot(ax1, g_nosma,  color='#ff7f0e', ls='-.', label=r'Geometric $r_{\rm true}$')

ax1.axvline(1.0, color='red',         lw=1.8, ls='--', label=r'Newtonian ($\gamma=1$)')
ax1.axvline(1.6, color='forestgreen', lw=1.8, ls='--', label=r'Chae+26 ($\gamma\approx1.6$)')

ax1.set_xlabel(r'$\gamma$ (boost factor)')
ax1.set_ylabel('Posterior density')
ax1.set_xlim(0.4, 2.6)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=12, loc='upper right', framealpha=0.9)
apply_ticks(ax1)

plt.tight_layout()
fig1.savefig('/home/saad.104/Downloads/PEPSI/gamma_overlay.pdf',         bbox_inches='tight')
fig1.savefig('/home/saad.104/Downloads/PEPSI/gamma_overlay_revised.png', bbox_inches='tight', dpi=200)
print("Saved gamma_overlay.pdf")

# ---- PANEL 2: Gamma ------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6.5, 5.0))

hist_plot(ax2, Gam_broken, color='#1f77b4', ls='-',  label='Baseline model')
hist_plot(ax2, Gam_nosma,  color='#ff7f0e', ls='-.', label=r'Geometric $r_{\rm true}$')

ax2.axvline(0.0,      color='red',         lw=1.8, ls='--', label=r'Newtonian ($\Gamma=0$)')
ax2.axvline(CHAE_GAM, color='forestgreen', lw=1.8, ls='--',
            label=fr'Chae+26 ($\Gamma\approx{CHAE_GAM:.3f}$)')

flat_level = 1.0 / 2.0   # Uniform(-1,1) density
ax2.axhline(flat_level, color='black', lw=1.4, ls='--',
            label=r'Prior: $\Gamma\sim\mathcal{U}(-1,1)$')

ax2.set_xlabel(r'$\Gamma \equiv \log_{10}\!\sqrt{\gamma}$')
ax2.set_ylabel('Posterior density')
ax2.set_xlim(-0.28, 0.26)
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=12, loc='upper left', framealpha=0.9)
apply_ticks(ax2)

plt.tight_layout()
fig2.savefig('/home/saad.104/Downloads/PEPSI/Gamma_overlay.pdf',         bbox_inches='tight')
fig2.savefig('/home/saad.104/Downloads/PEPSI/Gamma_overlay_revised.png', bbox_inches='tight', dpi=200)
print("Saved Gamma_overlay.pdf")
print("DONE.")
