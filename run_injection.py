#!/usr/bin/env python3
"""B1 — Newtonian injection-recovery validation (editor's requested test).

Simulate synthetic wide-binary catalogs matched to the 36 real systems with a
known injected gravity boost gamma_true, run each through the SAME baseline
model (BPL semi-major-axis prior), and check the recovered gamma. gamma_true=1.0
is the Newtonian validation; gamma_true=1.6 confirms the method also recovers a
real anomaly. Writes results incrementally so partial output is always available.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import run_gamma_analysis as rga
import pymc as pm
import arviz as az

AU = 1.496e11
PC_TO_M = 3.085677581e16
G = 6.67430e-11
M_SUN = 1.98847e30
K_PM = 4.74047
RV_SYS_MS = 40.0


def _rot_z(a):
    c, s = np.cos(a), np.sin(a); R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0] = c; R[..., 0, 1] = -s; R[..., 1, 0] = s; R[..., 1, 1] = c; R[..., 2, 2] = 1.0
    return R


def _rot_x(a):
    c, s = np.cos(a), np.sin(a); R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0] = 1.0; R[..., 1, 1] = c; R[..., 1, 2] = -s; R[..., 2, 1] = s; R[..., 2, 2] = c
    return R


def _triad(ra, dec):
    cdec, sdec = np.cos(dec), np.sin(dec); cra, sra = np.cos(ra), np.sin(ra)
    r = np.stack([cdec * cra, cdec * sra, sdec], axis=1)
    e = np.stack([-sra, cra, np.zeros_like(ra)], axis=1)
    n = np.stack([-sdec * cra, -sdec * sra, cdec], axis=1)
    return r, e, n


def _forward(a, e, inc, Om, om, M, M1s, M2s, vsys, tpl, gamma):
    M1, M2 = M1s * M_SUN, M2s * M_SUN
    E = M + e * np.sin(M) + 0.5 * e ** 2 * np.sin(2 * M)
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r_true = a * (1.0 - e ** 2) / (1.0 + e * np.cos(nu))
    R = _rot_z(Om) @ _rot_x(inc) @ _rot_z(om)
    F = (1.0 + e ** 2 + 2.0 * e * np.cos(nu)) / (1.0 + e * np.cos(nu))
    vmag = np.sqrt(gamma * G * (M1 + M2) / r_true * F)
    dn = np.sqrt(1.0 + e ** 2 + 2.0 * e * np.cos(nu))
    vr = vmag * (-e * np.sin(nu) / dn); vt = vmag * ((1.0 + e * np.cos(nu)) / dn)
    vx = vr * np.cos(nu) - vt * np.sin(nu); vy = vr * np.sin(nu) + vt * np.cos(nu)
    c1 = M2 / (M1 + M2); c2 = -M1 / (M1 + M2)
    v1 = (R @ np.stack([c1 * vx, c1 * vy, np.zeros_like(vx)], axis=1)[..., None]).squeeze(-1)
    v2 = (R @ np.stack([c2 * vx, c2 * vy, np.zeros_like(vx)], axis=1)[..., None]).squeeze(-1)
    ra_a, dec_a = tpl["ra_a"].values, tpl["dec_a"].values
    ra_b, dec_b = tpl["ra_b"].values, tpl["dec_b"].values
    da, db = tpl["distance_a_pc"].values, tpl["distance_b_pc"].values
    rha, eha, nha = _triad(ra_a, dec_a); rhb, ehb, nhb = _triad(ra_b, dec_b)
    dot = lambda u, w: np.sum(u * w, axis=1)
    rv_diff = (dot(v2, rhb) + dot(vsys, rhb)) - (dot(v1, rha) + dot(vsys, rha))
    v1t = v1 - dot(v1, rha)[:, None] * rha; v2t = v2 - dot(v2, rhb)[:, None] * rhb
    vsa = vsys - dot(vsys, rha)[:, None] * rha; vsb = vsys - dot(vsys, rhb)[:, None] * rhb
    pmra = -(dot(v2t + vsb, ehb) / (K_PM * db) - dot(v1t + vsa, eha) / (K_PM * da))
    pmdec = (dot(v2t + vsb, nhb) / (K_PM * db) - dot(v1t + vsa, nha) / (K_PM * da))
    ricrs = (R @ np.stack([r_true * np.cos(nu), r_true * np.sin(nu), np.zeros_like(nu)], axis=1)[..., None]).squeeze(-1)
    r_proj = np.sqrt(dot(ricrs, eha) ** 2 + dot(ricrs, nha) ** 2)
    return r_proj, rv_diff, pmra, pmdec


def simulate(tpl, gamma_true, seed):
    rng = np.random.default_rng(seed); N = len(tpl); r_t = tpl["r_obs"].values
    a = rng.lognormal(0.5, 0.8, N) * r_t
    e = np.clip(rng.beta(2.0, 1.0, N), 0, 0.98)
    inc = np.arccos(rng.uniform(-1, 1, N))
    Om = rng.uniform(0, 2 * np.pi, N); om = rng.uniform(0, 2 * np.pi, N); M = rng.uniform(0, 2 * np.pi, N)
    M1 = np.exp(tpl["mass_a_lnmu"].values); M2 = np.exp(tpl["mass_b_lnmu"].values)
    vsys = rng.normal(0, 15e3, (N, 3))
    # r_proj scales linearly with a (fixed angles/e/phase); rescale a so the
    # synthetic projected separation matches the real template -> the synthetic
    # systems resemble the real binaries and r_obs stays safely positive.
    r_proj0, _, _, _ = _forward(a, e, inc, Om, om, M, M1, M2, vsys, tpl, gamma_true)
    a = a * (r_t / np.maximum(r_proj0, 1e-3 * r_t))
    r_proj, rv, pmra, pmdec = _forward(a, e, inc, Om, om, M, M1, M2, vsys, tpl, gamma_true)
    # observational noise consistent with the (real-like) separation
    r_err = np.maximum(0.05 * r_proj, tpl["r_err"].values * 0.0 + 1.0)
    rv_sig = np.sqrt(tpl["rv_sigma"].values ** 2 + RV_SYS_MS ** 2); pm_err = tpl["pm_err"].values
    sim = tpl.copy()
    sim["r_obs"] = np.abs(r_proj + rng.normal(0, r_err))
    sim["r_err"] = r_err
    sim["rv_diff"] = rv + rng.normal(0, rv_sig)
    sim["pmra_diff"] = pmra + rng.normal(0, pm_err)
    sim["pmdec_diff"] = pmdec + rng.normal(0, pm_err)
    return sim


def gstats(g):
    g = np.asarray(g).ravel(); lo, hi = np.percentile(g, [16, 84]); med = float(np.median(g))
    return dict(median=med, ci68=[float(lo), float(hi)], minus=float(med - lo),
                plus=float(hi - med), P_gt_1=float(np.mean(g > 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", default="1.0,1.6")
    ap.add_argument("--n-datasets", type=int, default=5)
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--max-treedepth", type=int, default=10)
    ap.add_argument("--sma-prior", default="bpl")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--clean", default="data/chae_2026_data.csv")
    ap.add_argument("--full", default="data/chae_2026_gaia.csv")
    ap.add_argument("--out", default="outputs/injection_recovery.json")
    a = ap.parse_args()

    tpl = rga.prepare_data(a.clean, a.full)
    gammas = [float(x) for x in a.gammas.split(",")]
    report = {"settings": vars(a), "host": os.uname().nodename,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "fits": []}

    for gt in gammas:
        for k in range(a.n_datasets):
            t0 = time.time()
            sim = simulate(tpl, gt, a.seed + int(gt * 100) + k)
            model = rga.build_gamma_model(sim, include_sma=True, sma_prior=a.sma_prior)
            with model:
                step = pm.NUTS(target_accept=a.target_accept,
                               max_treedepth=a.max_treedepth)
                tr = pm.sample(a.draws, tune=a.tune, chains=a.chains, cores=a.cores,
                               step=step, random_seed=a.seed + k,
                               return_inferencedata=True, progressbar=False)
            st = gstats(tr.posterior["gamma"].values)
            summ = az.summary(tr, var_names=["gamma"])
            sig = 0.5 * (st["ci68"][1] - st["ci68"][0])
            st.update(injected=gt, dataset=k, rhat=float(summ.loc["gamma", "r_hat"]),
                      ess=float(summ.loc["gamma", "ess_bulk"]),
                      bias_sigma=float((st["median"] - gt) / sig) if sig else None,
                      elapsed_s=round(time.time() - t0, 1))
            report["fits"].append(st)
            json.dump(report, open(a.out, "w"), indent=2)  # incremental
            print(f"gamma_true={gt} k={k}: recovered={st['median']:.3f} "
                  f"68={[round(x,3) for x in st['ci68']]} bias={st['bias_sigma']:.2f}sig "
                  f"rhat={st['rhat']:.3f} ({st['elapsed_s']:.0f}s)", flush=True)

    # aggregate per injected gamma
    agg = {}
    for gt in gammas:
        meds = [f["median"] for f in report["fits"] if f["injected"] == gt]
        agg[str(gt)] = {"n": len(meds), "mean_recovered": float(np.mean(meds)),
                        "std_recovered": float(np.std(meds)),
                        "median_recovered": float(np.median(meds))}
    report["aggregate"] = agg
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(report, open(a.out, "w"), indent=2)
    print("AGG", json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
