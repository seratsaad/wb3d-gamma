# Data

The input data are from [Chae (2026)]([https://arxiv.org/abs/XXXX.XXXXX](https://arxiv.org/abs/2601.21728v2)). The data files are not included in this repository. They are available upon request (saad.104@osu.edu).

Place the two CSV files in this directory with the following names:

---

## `chae_2026_data.csv`

The 36-system clean sample with radial velocities and stellar masses. One row per binary system.

| Column | Description | Units |
|---|---|---|
| `gaia_a` | Gaia DR3 source ID of star A | — |
| `gaia_b` | Gaia DR3 source ID of star B | — |
| `vr_kms` | Radial velocity difference (B − A) | km/s |
| `vr_sigma_kms` | Uncertainty on RV difference | km/s |
| `mass_a` | Stellar mass of star A | M☉ |
| `mass_b` | Stellar mass of star B | M☉ |


---

## `chae_2026_gaia.csv`

Full Gaia DR3 astrometric data for the same 36 systems. One row per binary system.

| Column | Description | Units |
|---|---|---|
| `gaia_a` | Gaia DR3 source ID of star A | — |
| `gaia_b` | Gaia DR3 source ID of star B | — |
| `ra_a` | Right ascension of star A | degrees |
| `dec_a` | Declination of star A | degrees |
| `ra_b` | Right ascension of star B | degrees |
| `dec_b` | Declination of star B | degrees |
| `parallax_a` | Parallax of star A | mas |
| `parallax_error_a` | Parallax uncertainty of star A | mas |
| `parallax_b` | Parallax of star B | mas |
| `parallax_error_b` | Parallax uncertainty of star B | mas |
| `pmra_a` | Proper motion in RA of star A (μ_α*) | mas/yr |
| `pmra_error_a` | Uncertainty on μ_α* of star A | mas/yr |
| `pmdec_a` | Proper motion in Dec of star A | mas/yr |
| `pmdec_error_a` | Uncertainty on μ_δ of star A | mas/yr |
| `pmra_b` | Proper motion in RA of star B (μ_α*) | mas/yr |
| `pmra_error_b` | Uncertainty on μ_α* of star B | mas/yr |
| `pmdec_b` | Proper motion in Dec of star B | mas/yr |
| `pmdec_error_b` | Uncertainty on μ_δ of star B | mas/yr |

---

## Merging

The two files are joined on `gaia_a` and `gaia_b` (one-to-one match). The code handles this automatically in `prepare_data()`.
