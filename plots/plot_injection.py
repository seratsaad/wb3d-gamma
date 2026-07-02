#!/usr/bin/env python3
"""Appendix figure: injection-recovery test. Recovered gamma vs injected gamma.
Reads ../injection_recovery.json (output of run_injection.py); writes
injection_recovery.pdf."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "injection_recovery.json")))
fits, agg = d["fits"], d["aggregate"]
BLUE, ORANGE, RED = "#1f77b4", "#ff7f0e", "#d62728"

fig, ax = plt.subplots(figsize=(8, 6))
rng = np.random.default_rng(0)
injected = sorted({f["injected"] for f in fits})
for gt in injected:
    grp = [f for f in fits if f["injected"] == gt]
    xs = gt + rng.uniform(-0.035, 0.035, len(grp))
    ys = np.array([f["median"] for f in grp])
    lo = np.array([f["ci68"][0] for f in grp])
    hi = np.array([f["ci68"][1] for f in grp])
    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o", color=BLUE, ms=6,
                elinewidth=1.3, capsize=2.5, alpha=0.85,
                label="Individual realizations" if gt == injected[0] else None)
    med = agg[str(gt)]["median_recovered"]
    ax.plot([gt - 0.07, gt + 0.07], [med, med], color=ORANGE, lw=3,
            solid_capstyle="round",
            label="Recovered median" if gt == injected[0] else None)

lim = [0.5, 2.0]
ax.plot(lim, lim, ls="--", color="0.5", lw=1.8, label=r"$\gamma_{\rm rec}=\gamma_{\rm true}$")
ax.axhline(1.0, color=RED, ls=":", lw=1.8, alpha=0.8, label=r"Real sample ($\gamma=1.00$)")
ax.set_xlim(0.8, 1.8)
ax.set_ylim(0.5, 2.05)
ax.set_xlabel(r"Injected $\gamma_{\rm true}$", fontsize=14)
ax.set_ylabel(r"Recovered $\gamma$", fontsize=14)
ax.set_xticks(injected)
ax.legend(loc="upper left", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "injection_recovery.pdf"), dpi=300, bbox_inches="tight")
print("wrote injection_recovery.pdf")
