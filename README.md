# Figure 7 Regression Analysis

This repository contains a small, review-oriented Python analysis for the
Figure 7 regression case study from:

McGuirl, Volkening, and Sandstede, "Topological Data Analysis of Zebrafish
Patterns", PNAS 2020. DOI: `10.1073/pnas.1917763117`.

The code uses the authors' public Figshare simulation data to recreate the
main Figure 7 regression relationships and to add an out-of-sample validation
layer that is not included in the original GitHub code release.

## What Is Included

- `fig7_regression_recreation.py` recreates the Figure 7 regression/statistics
  panels from the available simulation summaries.
- `fig7_regression_validation.py` evaluates the same linear models with random
  train/test holdout and grouped leave-one-R-out validation.
- `fig6_reproduce_simulation.py` is an illustrative subsample of the Figure 6
  stochasticity experiment. It shows the core 1.3a story (wild-type stripes stay
  characteristic under noise σ while mutant patterns degrade) using raw
  pigment-cell coordinates plus persistent homology (Ripser) on a periodic
  domain to quantify β₀ (spots/components) and β₁ (stripe/interstripe loops). It
  is a small demonstration, not the full 24,000-simulation study.
- `data/*.csv` contains the summary tables used for pfeffer, shady, and
  wild-type pattern statistics.
- `data/mat_wt/*.mat` contains the small wild-type MATLAB summary files needed
  to recompute stripe and interstripe widths.

The scripts intentionally do not claim to be a pixel-perfect Figure 7
reproduction. They focus on the regression/statistics targets and use
scatter/median summaries instead of the paper's KDE panel style.

## What Is Not Included

The repository ignores local or generated files that are not useful for code
review:

- local virtual environments and Python caches
- generated PNG/PDF/PPTX presentation outputs
- raw zip archives and larger local simulation dumps

- scratch investigation notebooks

## Setup

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

The committed data files are enough for the current Fig. 7 scripts. If the data
directory is rebuilt from scratch, download the simulation data from the authors'
Figshare project and restore the same paths:

```text
data/WT_LSTest_annulusIn_MB.csv
data/pfeffer_LSTest_annulusIn_MB.csv
data/shady_LSTest_annulusIn_MB.csv
data/mat_wt/*.mat
```

## Running

Recreate the Fig. 7 regression/statistics panels:

```bash
venv/bin/python fig7_regression_recreation.py
```

This writes `fig7_regression_recreation.png`.

Run the validation experiment:

```bash
venv/bin/python fig7_regression_validation.py
```

This writes `fig7_regression_validation.png` and prints a console table with:

- paper in-sample R-squared values
- reproduced in-sample R-squared values
- random simulation-level holdout R-squared values
- grouped leave-one-R-out R-squared values for prediction at unseen `R`

Run the Figure 6 stochasticity demonstration:

```bash
venv/bin/python fig6_reproduce_simulation.py
```

This computes (β₀, β₁) per condition via persistent homology and writes
`fig6_demo.png`. It requires the wild-type and pfeffer sample/sigma simulation
files under `data/samples/` and `data/sigma/`, which are not committed to this
repository (they are large local simulation dumps). Restore them from the
authors' Figshare project before running:

```text
data/samples/Out_WT_default_1.mat
data/samples/Out_pfef_default_1.mat
data/sigma/Out_WT_pcpdTest_sigma_20_*.mat
data/sigma/Out_pfef_pcpdTest_sigma_20_*.mat
```

## Notes for Review

The pfeffer spot-size fit is the main value to inspect carefully. With the
available CSV column and positive-size filtering, the recreated fit does not
exactly match the paper's reported coefficient and R-squared. The scripts keep
this visible in the console output rather than hiding the mismatch.

For human review, Python scripts are preferable to notebooks because they are
easier to diff, rerun, and test. A notebook version can be useful later for a
presentation walkthrough, but it should be generated from or kept consistent
with these scripts rather than replacing them as the source of truth.
