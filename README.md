# Topological Data Analysis of Zebrafish Patterns — Topic 1.3

Code and data for our seminar contribution on **Topic 1.3 — Inferring Cellular
Interactions from Pattern Statistics**, in *Synergies of Mathematical Biology*,
TU Dresden, SoSe 26.

**Paper studied:** M. R. McGuirl, A. Volkening & B. Sandstede,
"Topological Data Analysis of Zebrafish Patterns," *PNAS* 117(10):5113–5124, 2020.
doi:[10.1073/pnas.1917763117](https://doi.org/10.1073/pnas.1917763117) (open access, CC BY 4.0).

**Authors:** Ahmed Alaa Elsaadani (1.3a) · Mahmoud Labib Elsamadnoy (1.3b)

---

## What this repository does

Rather than only summarising the paper, we work directly with the authors'
public simulation data to **reproduce their two key results** and to **add an
out-of-sample validation** that the original code release does not include.

- **1.3b — Figure 7 (parameter sweep + linear regression).** We refit the linear
  models relating the long-range interaction radius *R* to pattern statistics
  (stripe / interstripe width, spot size), recovering R² values close to the
  paper's, and then test them out-of-sample.
- **1.3a — Figure 6 (stochasticity).** We recompute the topological summaries
  (β₀ = spots, β₁ = stripes / interstripes) directly from raw pigment-cell
  coordinates using persistent homology (Ripser) on the periodic domain.

## Repository contents

| File | Purpose |
| --- | --- |
| `fig7_regression_recreation.py` | Recreates the Figure 7 regression / statistics panels from the simulation summaries. |
| `fig7_regression_validation.py` | Re-evaluates the same linear models with a random train/test holdout and grouped leave-one-*R*-out validation. |
| `fig6_reproduce_simulation.py` | Illustrative subsample of the Figure 6 stochasticity experiment (β₀ / β₁ via persistent homology). |
| `data/*.csv` | Per-simulation summary tables (wild-type, *pfeffer*, *shady*). |
| `data/mat_wt/*.mat` | Wild-type MATLAB summaries used to recompute stripe / interstripe widths. |
| `requirements.txt` | Python dependencies. |

These scripts are **not** a pixel-perfect Figure 7 reproduction: they target the
regression / statistics relationships, use scatter + per-*R* medians instead of
the paper's KDE density panels, and omit the example-pattern images (panel G).

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Running

**Figure 7 — recreation**

```bash
venv/bin/python fig7_regression_recreation.py      # → fig7_regression_recreation.png
```

**Figure 7 — validation**

```bash
venv/bin/python fig7_regression_validation.py      # → fig7_regression_validation.png + console table
```

The console table reports, for each statistic:

- the paper's in-sample R²,
- our reproduced in-sample R²,
- random simulation-level holdout R²,
- grouped leave-one-*R*-out R² (prediction at *unseen* values of *R*).

**Figure 6 — stochasticity demonstration**

```bash
venv/bin/python fig6_reproduce_simulation.py       # → fig6_demo.png
```

This computes (β₀, β₁) per condition via persistent homology. It needs the
wild-type and *pfeffer* coordinate files, which are **not committed** (they are
large raw simulation dumps). Restore them from the Figshare deposit under:

```text
data/samples/Out_WT_default_1.mat
data/samples/Out_pfef_default_1.mat
data/sigma/Out_WT_pcpdTest_sigma_20_*.mat
data/sigma/Out_pfef_pcpdTest_sigma_20_*.mat
```

## Data

The committed data (the CSV summaries and `data/mat_wt/`) is enough to run the
Figure 7 scripts. To rebuild the data directory from scratch, download the
simulation data from the authors' Figshare project
([Zebrafish_simulation_data](https://figshare.com/projects/Zebrafish_simulation_data/72689))
and restore:

```text
data/WT_LSTest_annulusIn_MB.csv
data/pfeffer_LSTest_annulusIn_MB.csv
data/shady_LSTest_annulusIn_MB.csv
data/mat_wt/*.mat
```

## A note on the *pfeffer* fit

Our *pfeffer* spot-size R² (≈ 0.84) is higher than the paper's (0.72). We traced
this to the handling of zero-spot patterns at small *R*: those degenerate low-*R*
simulations are dropped by the positive-size filter, which removes the low-*R*
scatter and steepens the fit. Keeping them in returns a slope (≈ 0.38) that
matches the paper's (0.37). The scripts surface this in the console output rather
than hiding the mismatch.

## Acknowledgements

The simulation data and the original analysis approach are due to McGuirl,
Volkening & Sandstede (PNAS 2020), with their code at
[sandstede-lab/Quantifying_Zebrafish_Patterns](https://github.com/sandstede-lab/Quantifying_Zebrafish_Patterns)
and data on Figshare — both released openly under permissive terms.
