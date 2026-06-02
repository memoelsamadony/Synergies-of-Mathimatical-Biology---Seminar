#!/usr/bin/env python3
"""
Train/test validation of the Figure 7 linear regressions from:
  "Topological Data Analysis of Zebrafish Patterns"
  McGuirl, Volkening & Sandstede — PNAS 2020

The paper reports IN-SAMPLE R² (fit on all data, scored on the same data).
This script tests whether the predictive claim of the abstract holds OUT-OF-SAMPLE in
two distinct senses:
  (1) random 75/25 holdout  — held-out SIMULATIONS at already-seen R values;
                              tests stability, NOT prediction at new scales.
  (2) grouped leave-one-R-out — train on all-other-R, predict each held-out R, pool
                              predictions → one R²; tests genuine prediction at an
                              UNSEEN interaction scale (within the tested 10–400 µm).

If the grouped R² ≈ in-sample R², the linear "predictive instrument" genuinely predicts
at unseen R — backing up the abstract's "allow for predictive analyses" with evidence
the paper itself did not include. (Scope is the tested range, not in vivo.)

Run:    venv/bin/python fig7_regression_validation.py
Output: console table + fig7_regression_validation.png

Reuses the same Figshare data loaded by fig7_regression_recreation.py.
"""

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_DIR  = 'data'
DEFAULT_R = 210
SEED      = 42          # fixed seed → reproducible split (no Date/random surprises)
TEST_FRAC = 0.25        # 75% train / 25% random holdout


# ── Data loading (mirrors reproduce_fig7.py) ───────────────────────────────────

def _mat_r(fpath):
    return float(os.path.basename(fpath).split('_')[-3])


def load_wt_widths():
    """Per-simulation max stripe & interstripe widths (mm) from WT mat files."""
    mat_dir = os.path.join(DATA_DIR, 'mat_wt')
    R_vals, stripe, inter = [], [], []
    for fp in sorted(glob.glob(os.path.join(mat_dir, '*.mat'))):
        r = _mat_r(fp)
        struct = loadmat(fp)['pattern_info'][0, 0]
        xC, xS = struct['xanC_widths'], struct['xanS_widths']
        for i in range(xC.shape[0]):
            vc = np.asarray(xC[i, 0], dtype=float).flatten()
            vs = np.asarray(xS[i, 0], dtype=float).flatten()
            vc = vc[(vc > 0) & np.isfinite(vc)]
            vs = vs[(vs > 0) & np.isfinite(vs)]
            R_vals.append(r)
            stripe.append(2.0 * np.max(vc) / 1000.0 if len(vc) else np.nan)
            inter.append(2.0 * np.max(vs) / 1000.0 if len(vs) else np.nan)
    return np.array(R_vals), np.array(stripe), np.array(inter)


def load_csv(path):
    with open(path) as f:
        header = f.readline().strip().split(',')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)
    return {col: data[:, i] for i, col in enumerate(header)}


def clean(x, y):
    """Drop NaN / non-positive y, return column-shaped X and y."""
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    return x[m].reshape(-1, 1), y[m]


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(name, x, y):
    """
    Returns dict with three R² flavours:
      (1) in-sample      — fit and score on all data (what the paper reports)
      (2) random holdout — 75/25 split; tests held-out SIMULATIONS at already-seen R
      (3) grouped (LORO) — leave-one-R-out: train on all-other-R, predict each held-out
                           R, POOL the predictions → one R². This is the honest
                           "predict at an UNSEEN interaction scale" number, i.e. the
                           one that actually matches the abstract's predictive claim.
    """
    X, yv = clean(x, y)
    n = len(yv)
    R = X.flatten()                      # interaction scale = the group label

    # (1) In-sample (what the paper reports)
    full = LinearRegression().fit(X, yv)
    r2_in = r2_score(yv, full.predict(X))

    # (2) Single random hold-out split (simulation-level)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, yv, test_size=TEST_FRAC, random_state=SEED)
    split = LinearRegression().fit(X_tr, y_tr)
    r2_train = r2_score(y_tr, split.predict(X_tr))
    r2_test  = r2_score(y_te, split.predict(X_te))

    # (3) Grouped leave-one-R-out — POOLED predictions across held-out R values.
    #     NB: never score R² within a single held-out R: x is constant there, so there
    #     is no x-variance to explain and per-group R² is meaningless (goes negative).
    #     Pooling all held-out predictions into one R² is the correct grouped metric.
    preds = np.empty_like(yv)
    for rv in set(R):
        te = R == rv
        m = LinearRegression().fit(X[~te], yv[~te])
        preds[te] = m.predict(X[te])
    r2_grouped = r2_score(yv, preds)

    return {
        'name': name, 'n': n,
        'slope': full.coef_[0], 'intercept': full.intercept_,
        'r2_in': r2_in, 'r2_train': r2_train, 'r2_test': r2_test,
        'r2_grouped': r2_grouped,
        'X': X, 'y': yv, 'split': split,
        'X_tr': X_tr, 'y_tr': y_tr, 'X_te': X_te, 'y_te': y_te,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('Loading Figshare simulation data …\n')
    R_wt, stripe_mm, inter_mm = load_wt_widths()
    pf = load_csv(os.path.join(DATA_DIR, 'pfeffer_LSTest_annulusIn_MB.csv'))
    sh = load_csv(os.path.join(DATA_DIR, 'shady_LSTest_annulusIn_MB.csv'))

    runs = [
        validate('WT stripe width (mm)',      R_wt,                stripe_mm),
        validate('WT interstripe width (mm)', R_wt,                inter_mm),
        validate('Pfeffer spot size (cells)', pf['annulus_In_MB'], pf['med_cluster_size_all']),
        validate('Shady spot size (cells)',   sh['annulus_In_MB'], sh['mel_med_cluster_size_all']),
    ]

    # Paper's reported in-sample R² for reference
    paper_r2 = {
        'WT stripe width (mm)':      0.91,
        'WT interstripe width (mm)': 0.81,
        'Pfeffer spot size (cells)': 0.72,
        'Shady spot size (cells)':   0.90,
    }

    # ── Console table ──────────────────────────────────────────────────────────
    print(f"{'Metric':<28} {'n':>5} {'paper':>6} {'in-samp':>8} "
          f"{'rand-TEST':>10} {'grouped-LORO':>13}")
    print('─' * 78)
    for r in runs:
        print(f"{r['name']:<28} {r['n']:>5} {paper_r2[r['name']]:>6.2f} "
              f"{r['r2_in']:>8.3f} {r['r2_test']:>10.3f} {r['r2_grouped']:>13.3f}")

    print('\nInterpretation (precise wording for the talk):')
    print('  rand-TEST ≈ in-sample    → model is STABLE under simulation-level holdout')
    print('                             (held-out sims at ALREADY-SEEN R values).')
    print('  grouped-LORO ≈ in-sample → model PREDICTS at UNSEEN R within the tested')
    print('                             range [10–400 µm]; this is the result that')
    print('                             supports the abstract\'s predictive claim.')
    print('  Scope: "predicts at unseen R within the tested range" — NOT a blanket')
    print('         claim of prediction in untested parameter regimes or in vivo.')
    print('  No overfitting: a 1-D linear model (2 params) vs ~1000–1600 points cannot')
    print('         overfit, so train ≈ test is expected, not surprising.')

    # ── Figure: held-out predictions vs truth ──────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    fig.patch.set_facecolor('white')
    x_line = np.linspace(10, 400, 200).reshape(-1, 1)

    for ax, r in zip(axes, runs):
        # training points (faint) + held-out test points (highlighted) — disjoint sets
        ax.scatter(r['X_tr'], r['y_tr'], s=6, alpha=0.08, color='#999999',
                   zorder=1, label='train (75%)')
        ax.scatter(r['X_te'], r['y_te'], s=14, alpha=0.55, color='#e76f51',
                   zorder=3, label='held-out test (25%)')
        ax.plot(x_line, r['split'].predict(x_line), color='navy', lw=2,
                zorder=4, label='fit (train only)')
        ax.axvline(DEFAULT_R, color='gray', ls='--', lw=1, alpha=0.6, zorder=2)
        ax.set_title(f"{r['name']}\nrand-test $R^2$ = {r['r2_test']:.3f}   "
                     f"grouped $R^2$ = {r['r2_grouped']:.3f}",
                     fontsize=9, fontweight='bold', color='#1a3a5c')
        ax.set_xlabel('Interaction scale in M birth R (μm)', fontsize=8)
        ax.set_ylabel(r['name'], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 420)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(
        'Out-of-sample validation of the Fig. 7 linear models  '
        f'({int((1-TEST_FRAC)*100)}% train / {int(TEST_FRAC*100)}% random holdout, seed={SEED})  '
        '— title shows random-split AND grouped leave-one-R-out $R^2$',
        fontsize=10, fontweight='bold', color='#1a3a5c')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = 'fig7_regression_validation.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
