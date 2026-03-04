#!/usr/bin/env python3
"""
Hierarchical Bayesian Reanalysis of Wide Binaries with Reported Gravitational Anomaly

This script builds and samples two hierarchical Bayesian models to infer the
gravity boost factor gamma from the 36 wide-binary systems of Chae (2026):

  - Baseline model: semi-major axis is a free parameter; r_true from Kepler orbit.
  - Geometric deprojection model: no semi-major axis; r_true derived from r_obs
    and viewing geometry, analogous to the approach of Chae (2026).

Requirements: numpy, pandas, pymc, pytensor, matplotlib, arviz

Author: Serat Mahmud Saad & Yuan-Sen Ting
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import arviz as az
import warnings
import time
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================
AU = 1.496e11            # metres
PC_TO_M = 3.085677581e16 # metres per parsec
G = 6.67430e-11          # gravitational constant (SI)
M_SUN = 1.98847e30       # solar mass (kg)
K_PM = 4.74047           # km/s per mas/yr per pc

# =============================================================================
# Configuration
# =============================================================================
MASS_LN_SIGMA = 0.05     # log-normal width for mass prior
RV_SYS_MS = 40.0         # additional RV systematic (m/s)
FRAC_FLOOR = 0.05        # minimum fractional uncertainty on r_obs


# =============================================================================
# Data preparation
# =============================================================================
def prepare_data(clean_csv_path, full_csv_path):
    """
    Load and merge the Chae (2026) clean sample with full astrometric data.

    Parameters
    ----------
    clean_csv_path : str
        Path to the cleaned sample CSV (with RVs, masses, selection flags).
    full_csv_path : str
        Path to the complete astrometric CSV (positions, proper motions, parallaxes).

    Returns
    -------
    pd.DataFrame
        Merged data with computed observables (projected separation, PM differences, etc.).
    """
    clean = pd.read_csv(clean_csv_path)
    clean["gaia_a"] = clean["gaia_a"].astype("int64")
    clean["gaia_b"] = clean["gaia_b"].astype("int64")

    full = pd.read_csv(full_csv_path)
    full["gaia_a"] = full["gaia_a"].astype("int64")
    full["gaia_b"] = full["gaia_b"].astype("int64")

    key = clean[["gaia_a", "gaia_b"]].copy()
    key["__row__"] = np.arange(len(key))
    df = (
        key.merge(full, on=["gaia_a", "gaia_b"], how="left", validate="one_to_one")
           .sort_values("__row__")
           .reset_index(drop=True)
    )
    clean = clean.reset_index(drop=True)
    N = len(df)

    # Distances from parallax
    d_a_pc = 1000.0 / df["parallax_a"].values
    d_b_pc = 1000.0 / df["parallax_b"].values
    d_mean_pc = 0.5 * (d_a_pc + d_b_pc)

    # Sky positions (radians)
    ra_a = np.deg2rad(df["ra_a"].values)
    dec_a = np.deg2rad(df["dec_a"].values)
    ra_b = np.deg2rad(df["ra_b"].values)
    dec_b = np.deg2rad(df["dec_b"].values)

    # Projected separation (metres)
    delta_ra = (ra_b - ra_a) * np.cos(0.5 * (dec_a + dec_b))
    delta_dec = dec_b - dec_a
    sep_rad = np.sqrt(delta_ra**2 + delta_dec**2)
    r_obs = d_mean_pc * sep_rad * PC_TO_M

    # Separation uncertainty from parallax errors
    par_err_a = df["parallax_error_a"].values
    par_err_b = df["parallax_error_b"].values
    d_a_err_pc = 1000.0 * par_err_a / (df["parallax_a"].values ** 2)
    d_b_err_pc = 1000.0 * par_err_b / (df["parallax_b"].values ** 2)
    d_mean_err_m = 0.5 * np.sqrt((d_a_err_pc * PC_TO_M) ** 2 + (d_b_err_pc * PC_TO_M) ** 2)
    r_err = np.maximum(FRAC_FLOOR * r_obs, d_mean_err_m)

    # Differential proper motions
    dpmra = df["pmra_b"].values - df["pmra_a"].values
    dpmdec = df["pmdec_b"].values - df["pmdec_a"].values
    pm_err_a = np.sqrt(df["pmra_error_a"].values ** 2 + df["pmdec_error_a"].values ** 2)
    pm_err_b = np.sqrt(df["pmra_error_b"].values ** 2 + df["pmdec_error_b"].values ** 2)
    pm_err = 0.5 * np.sqrt(pm_err_a**2 + pm_err_b**2)

    # Radial velocities (convert km/s -> m/s)
    rv_diff_ms = clean["vr_kms"].values * 1000.0
    rv_sigma_ms = clean["vr_sigma_kms"].values * 1000.0

    # Masses
    mass_a = np.where(np.isnan(clean["mass_a"].values.astype(float)), 1.0,
                      clean["mass_a"].values.astype(float))
    mass_b = np.where(np.isnan(clean["mass_b"].values.astype(float)), 1.0,
                      clean["mass_b"].values.astype(float))

    print(f"Loaded {N} systems")

    return pd.DataFrame({
        "r_obs": r_obs, "r_err": r_err,
        "rv_diff": rv_diff_ms, "rv_sigma": rv_sigma_ms,
        "distance_pc": d_mean_pc,
        "distance_a_pc": d_a_pc, "distance_b_pc": d_b_pc,
        "separation": sep_rad,
        "ra_a": ra_a, "dec_a": dec_a, "ra_b": ra_b, "dec_b": dec_b,
        "pmra_diff": dpmra, "pmdec_diff": dpmdec, "pm_err": pm_err,
        "mass_a_lnmu": np.log(mass_a), "mass_b_lnmu": np.log(mass_b),
    })


# =============================================================================
# Model builder
# =============================================================================
def build_gamma_model(data, include_sma=True):
    """
    Build the hierarchical Bayesian model for gamma inference.

    Parameters
    ----------
    data : pd.DataFrame
        Output of prepare_data().
    include_sma : bool
        If True, include semi-major axis as a free parameter (baseline model).
        If False, derive r_true from geometric deprojection of r_obs.

    Returns
    -------
    pm.Model
    """
    N = len(data)

    with pm.Model() as model:

        # -- Observed data --
        r_obs = pm.Data("r_obs", data["r_obs"].values)
        r_err = pm.Data("r_err", data["r_err"].values)
        RV_diff = pm.Data("RV_diff", data["rv_diff"].values)
        RV_sigma = pm.Data("RV_sigma", data["rv_sigma"].values)
        ra_a = pm.Data("ra_a_rad", data["ra_a"].values)
        dec_a = pm.Data("dec_a_rad", data["dec_a"].values)
        ra_b = pm.Data("ra_b_rad", data["ra_b"].values)
        dec_b = pm.Data("dec_b_rad", data["dec_b"].values)
        distance_a_pc = pm.Data("distance_a_pc", data["distance_a_pc"].values)
        distance_b_pc = pm.Data("distance_b_pc", data["distance_b_pc"].values)
        distance_pc = pm.Data("distance_pc", data["distance_pc"].values)

        # -- Global gravity boost factor --
        Gam = pm.Uniform("Gam", lower=-1.0, upper=1.0)
        gamma = pm.Deterministic("gamma", 10.0 ** (2.0 * Gam))

        # -- Stellar masses --
        M1_sol = pm.LogNormal("M1_sol", mu=data["mass_a_lnmu"].values,
                              sigma=MASS_LN_SIGMA, shape=N)
        M2_sol = pm.LogNormal("M2_sol", mu=data["mass_b_lnmu"].values,
                              sigma=MASS_LN_SIGMA, shape=N)
        M1 = M1_sol * M_SUN
        M2 = M2_sol * M_SUN

        # -- Semi-major axis (baseline only) --
        if include_sma:
            a_over_robs = pm.LogNormal("a_over_robs", mu=0.5, sigma=0.8, shape=N)
            a = pm.Deterministic("a", a_over_robs * r_obs)

        # -- Eccentricity (separation-dependent thermal prior) --
        sep_au = pm.Data("sep_au", data["r_obs"].values / AU)
        bin_edges = np.array([0, 100, 300, 1000, 3000, 1e6], dtype=float)
        bin_idx = np.clip(
            np.digitize(np.asarray(sep_au.get_value()), bin_edges) - 1,
            0, len(bin_edges) - 2,
        )
        bin_idx_data = pm.Data("bin_idx", bin_idx)

        mu_alpha = np.array([-0.2, 0.4, 0.8, 1.0, 1.2])
        sigma_alpha = np.array([0.3, 0.2, 0.2, 0.2, 0.2])
        alpha_bins = pm.Normal("alpha_bins", mu=mu_alpha, sigma=sigma_alpha,
                               shape=len(mu_alpha))
        alpha = pm.Deterministic("alpha", alpha_bins[bin_idx_data])
        e_raw = pm.Beta("e_raw", alpha=pt.maximum(alpha + 1.0, 0.1),
                        beta=1.0, shape=N)
        e = pm.Deterministic("e", pt.minimum(e_raw, 0.98))

        # -- Orbital angles --
        M_anom = pm.Uniform("M_anom", 0.0, 2 * np.pi, shape=N)
        cos_i = pm.Uniform("cos_i", lower=-1.0, upper=1.0, shape=N)
        i = pm.Deterministic("i", pt.arccos(cos_i))
        Omega = pm.Uniform("Omega", lower=0.0, upper=2 * np.pi, shape=N)
        omega = pm.Uniform("omega", lower=0.0, upper=2 * np.pi, shape=N)

        # -- Kepler solver --
        def kepler_E(M, e, n_iter=7):
            E = M + e * pt.sin(M) + 0.5 * e**2 * pt.sin(2 * M)
            for _ in range(n_iter):
                f = E - e * pt.sin(E) - M
                fp = 1.0 - e * pt.cos(E)
                E = E - f / fp
            return E

        def true_anomaly(E, e):
            s = pt.sqrt((1.0 + e) / (1.0 - e))
            return 2.0 * pt.arctan2(s * pt.sin(E / 2.0), pt.cos(E / 2.0))

        E_val = kepler_E(M_anom, e)
        nu = pm.Deterministic("nu", true_anomaly(E_val, e))

        # -- True 3D separation --
        if include_sma:
            denom = pt.clip(1.0 + e * pt.cos(nu), 1e-9, np.inf)
            r_true = pm.Deterministic("r_true", a * (1.0 - e**2) / denom)
        else:
            phi_orbital = omega + nu
            deproj = pt.sqrt(pt.cos(phi_orbital)**2 + pt.cos(i)**2 * pt.sin(phi_orbital)**2)
            deproj = pt.clip(deproj, 1e-6, np.inf)
            r_true = pm.Deterministic("r_true", r_obs / deproj)

        # -- Rotation matrix (orbital -> ICRS) --
        def rot_z(angle):
            c, s = pt.cos(angle), pt.sin(angle)
            z, o = pt.zeros_like(c), pt.ones_like(c)
            return pt.stack([
                pt.stack([c, -s, z], axis=-1),
                pt.stack([s, c, z], axis=-1),
                pt.stack([z, z, o], axis=-1),
            ], axis=-2)

        def rot_x(angle):
            c, s = pt.cos(angle), pt.sin(angle)
            z, o = pt.zeros_like(c), pt.ones_like(c)
            return pt.stack([
                pt.stack([o, z, z], axis=-1),
                pt.stack([z, c, -s], axis=-1),
                pt.stack([z, s, c], axis=-1),
            ], axis=-2)

        R = rot_z(Omega) @ rot_x(i) @ rot_z(omega)

        # -- Orbital velocity --
        F = (1.0 + e**2 + 2.0 * e * pt.cos(nu)) / (1.0 + e * pt.cos(nu))
        F = pt.clip(F, 1e-9, np.inf)
        v_rel_mag = pt.sqrt(pt.maximum(gamma * G * (M1 + M2) / r_true * F, 1e-30))

        vr_fac = -e * pt.sin(nu) / pt.sqrt(1.0 + e**2 + 2.0 * e * pt.cos(nu))
        vt_fac = (1.0 + e * pt.cos(nu)) / pt.sqrt(1.0 + e**2 + 2.0 * e * pt.cos(nu))
        vr_rel = v_rel_mag * vr_fac
        vt_rel = v_rel_mag * vt_fac

        vx_rel = vr_rel * pt.cos(nu) - vt_rel * pt.sin(nu)
        vy_rel = vr_rel * pt.sin(nu) + vt_rel * pt.cos(nu)

        coef1 = M2 / (M1 + M2)
        coef2 = -M1 / (M1 + M2)
        v1_orb = pt.stack([coef1 * vx_rel, coef1 * vy_rel, pt.zeros_like(vx_rel)], axis=1)
        v2_orb = pt.stack([coef2 * vx_rel, coef2 * vy_rel, pt.zeros_like(vx_rel)], axis=1)

        v1_icrs = (R @ v1_orb[..., None]).squeeze(-1)
        v2_icrs = (R @ v2_orb[..., None]).squeeze(-1)

        # -- Local coordinate triads --
        def triad(ra, dec):
            cdec, sdec = pt.cos(dec), pt.sin(dec)
            cra, sra = pt.cos(ra), pt.sin(ra)
            r_hat = pt.stack([cdec * cra, cdec * sra, sdec], axis=1)
            e_hat = pt.stack([-sra, cra, pt.zeros_like(ra)], axis=1)
            n_hat = pt.stack([-sdec * cra, -sdec * sra, cdec], axis=1)
            return r_hat, e_hat, n_hat

        rhat_a, ehat_a, nhat_a = triad(ra_a, dec_a)
        rhat_b, ehat_b, nhat_b = triad(ra_b, dec_b)

        # -- Systemic velocity --
        v_sys_x = pm.Normal("v_sys_x", mu=0.0, sigma=30e3, shape=N)
        v_sys_y = pm.Normal("v_sys_y", mu=0.0, sigma=30e3, shape=N)
        v_sys_z = pm.Normal("v_sys_z", mu=0.0, sigma=30e3, shape=N)
        v_sys = pt.stack([v_sys_x, v_sys_y, v_sys_z], axis=-1)

        # -- Model-predicted RV difference --
        rv1 = pt.sum(v1_icrs * rhat_a, axis=1) + pt.sum(v_sys * rhat_a, axis=1)
        rv2 = pt.sum(v2_icrs * rhat_b, axis=1) + pt.sum(v_sys * rhat_b, axis=1)
        RV_diff_model = pm.Deterministic("RV_diff_model", rv2 - rv1)

        # -- Model-predicted proper motion differences --
        v1_t_a = v1_icrs - (pt.sum(v1_icrs * rhat_a, axis=1)[:, None]) * rhat_a
        v2_t_b = v2_icrs - (pt.sum(v2_icrs * rhat_b, axis=1)[:, None]) * rhat_b
        v_sys_t_a = v_sys - (pt.sum(v_sys * rhat_a, axis=1)[:, None]) * rhat_a
        v_sys_t_b = v_sys - (pt.sum(v_sys * rhat_b, axis=1)[:, None]) * rhat_b

        v1_E_a = pt.sum((v1_t_a + v_sys_t_a) * ehat_a, axis=1)
        v1_N_a = pt.sum((v1_t_a + v_sys_t_a) * nhat_a, axis=1)
        v2_E_b = pt.sum((v2_t_b + v_sys_t_b) * ehat_b, axis=1)
        v2_N_b = pt.sum((v2_t_b + v_sys_t_b) * nhat_b, axis=1)

        pmra_diff_model = pm.Deterministic(
            "pmra_diff_model",
            -(v2_E_b / (K_PM * distance_b_pc) - v1_E_a / (K_PM * distance_a_pc))
        )
        pmdec_diff_model = pm.Deterministic(
            "pmdec_diff_model",
            (v2_N_b / (K_PM * distance_b_pc) - v1_N_a / (K_PM * distance_a_pc))
        )

        # -- Model-predicted projected separation --
        rx_orb = r_true * pt.cos(nu)
        ry_orb = r_true * pt.sin(nu)
        r_orb_vec = pt.stack([rx_orb, ry_orb, pt.zeros_like(rx_orb)], axis=1)
        r_icrs = (R @ r_orb_vec[..., None]).squeeze(-1)
        r_E = pt.sum(r_icrs * ehat_a, axis=1)
        r_N = pt.sum(r_icrs * nhat_a, axis=1)
        r_proj_model = pm.Deterministic("r_proj_model", pt.sqrt(r_E**2 + r_N**2))

        # -- Likelihoods --
        rv_jitter = pm.HalfNormal("rv_jitter", sigma=10.0)
        pm_jitter = pm.HalfNormal("pm_jitter", sigma=0.05)

        pm.Normal("lik_rproj", mu=r_proj_model, sigma=r_err, observed=r_obs)

        rv_sig_eff = pt.sqrt(RV_sigma**2 + RV_SYS_MS**2 + rv_jitter**2)
        pm.StudentT("lik_RVdiff", nu=5, mu=RV_diff_model,
                    sigma=rv_sig_eff, observed=RV_diff)

        dpmra_obs = pm.Data("dpmra_obs", data["pmra_diff"].values)
        dpmdec_obs = pm.Data("dpmdec_obs", data["pmdec_diff"].values)
        pm_err = pm.Data("pm_err", data["pm_err"].values)
        pm.Normal("lik_pmra", mu=pmra_diff_model,
                  sigma=pt.sqrt(pm_err**2 + pm_jitter**2), observed=dpmra_obs)
        pm.Normal("lik_pmdec", mu=pmdec_diff_model,
                  sigma=pt.sqrt(pm_err**2 + pm_jitter**2), observed=dpmdec_obs)

    return model


# =============================================================================
# Sampling and summary
# =============================================================================
def sample_model(model, n_tune=1500, n_samples=2000, n_chains=4):
    """Sample the model using NUTS."""
    with model:
        trace = pm.sample(n_samples, tune=n_tune, chains=n_chains,
                          target_accept=0.98, return_inferencedata=True,
                          random_seed=123)
    return trace


def print_gamma_stats(trace, label=""):
    """Print summary statistics for the gamma posterior."""
    g = trace.posterior["gamma"].values.flatten()
    med = np.median(g)
    ci68 = np.percentile(g, [16, 84])
    ci95 = np.percentile(g, [2.5, 97.5])
    print(f"  {label}")
    print(f"  gamma  = {med:.4f}  68% CI = [{ci68[0]:.4f}, {ci68[1]:.4f}]")
    print(f"  95% CI = [{ci95[0]:.4f}, {ci95[1]:.4f}]")
    print(f"  P(gamma > 1) = {np.mean(g > 1):.4f}")
    return g


# =============================================================================
# Main
# =============================================================================
def run_analysis(clean_csv, full_csv, n_tune=2000, n_samples=3000, n_chains=4):
    """
    Run the full analysis: baseline + geometric deprojection models.

    Parameters
    ----------
    clean_csv : str
        Path to Chae (2026) clean sample CSV.
    full_csv : str
        Path to complete astrometric CSV.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = prepare_data(clean_csv, full_csv)

    # --- Baseline model ---
    print("\n" + "=" * 60)
    print("Baseline model (with semi-major axis)")
    print("=" * 60)
    t0 = time.time()
    model_baseline = build_gamma_model(data, include_sma=True)
    trace_baseline = sample_model(model_baseline, n_tune, n_samples, n_chains)
    print(f"  Elapsed: {time.time() - t0:.0f}s")
    g_baseline = print_gamma_stats(trace_baseline, "Baseline")

    # --- Geometric deprojection model ---
    print("\n" + "=" * 60)
    print("Geometric deprojection model (no semi-major axis)")
    print("=" * 60)
    t0 = time.time()
    model_deproj = build_gamma_model(data, include_sma=False)
    trace_deproj = sample_model(model_deproj, n_tune, n_samples, n_chains)
    print(f"  Elapsed: {time.time() - t0:.0f}s")
    g_deproj = print_gamma_stats(trace_deproj, "Geometric deprojection")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for label, g in [("Baseline", g_baseline), ("Geometric deprojection", g_deproj)]:
        med = np.median(g)
        ci = np.percentile(g, [16, 84])
        print(f"  {label:35s}  gamma={med:.4f}  68%=[{ci[0]:.4f}, {ci[1]:.4f}]  "
              f"P(gamma>1)={np.mean(g > 1):.4f}")

    # --- Save posterior samples ---
    a_over = trace_baseline.posterior["a_over_robs"].values
    a_over_medians = np.median(a_over.reshape(-1, len(data)), axis=0)

    outfile = f"posterior_samples.npz"
    np.savez(outfile,
             baseline=g_baseline,
             geometric_deprojection=g_deproj,
             a_over_rproj_medians=a_over_medians)
    print(f"\nSaved posterior samples to {outfile}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, g, color, ls in [
        ("Baseline (with SMA)", g_baseline, "#1f77b4", "-"),
        ("Geometric deprojection", g_deproj, "#2ca02c", "-."),
    ]:
        ax.hist(g, bins=60, density=True, alpha=0.2, color=color)
        counts, edges = np.histogram(g, bins=60, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, color=color, lw=2, ls=ls, label=label)
    ax.axvline(1.0, color="red", ls="--", lw=2, alpha=0.7, label=r"Newtonian ($\gamma=1$)")
    ax.set_xlabel(r"$\gamma$", fontsize=14)
    ax.set_ylabel("Posterior density", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("gamma_posterior.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved gamma_posterior.png")


if __name__ == "__main__":
    clean_csv = "data/chae_2026_data.csv"
    full_csv = "data/chae_2026_gaia.csv"
    run_analysis(clean_csv, full_csv, n_tune=2000, n_samples=3000, n_chains=4)
