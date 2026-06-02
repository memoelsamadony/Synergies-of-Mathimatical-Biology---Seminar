#!/usr/bin/env python3
"""
Recreate the Figure 7 regression panels from:
  "Topological Data Analysis of Zebrafish Patterns"
  McGuirl, Volkening & Sandstede — PNAS 2020
  DOI: 10.1073/pnas.1917763117

Uses simulation summaries downloaded from Figshare article 11568675:
  https://figshare.com/articles/dataset/11568675

Required files (already present in ./data/):
  data/mat_wt/          WT mat files per R value (from WT_annulus_MB_experiment.zip)
  data/pfeffer_LSTest_annulusIn_MB.csv
  data/shady_LSTest_annulusIn_MB.csv

Run:    venv/bin/python fig7_regression_recreation.py
Output: fig7_regression_recreation.png

This is not a pixel-for-pixel reproduction of the PNAS figure. It focuses on
the Fig. 7 regression targets and summarizes the same simulation sweeps.
"""

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATA_DIR   = 'data'
DEFAULT_R  = 210      # default Ω_long inner radius (μm)
XLIM       = (0, 420)


# ── Data loading ──────────────────────────────────────────────────────────────

def _mat_r(fpath):
    """Extract R value from filename like *_MB_210_outputs_all.mat."""
    return float(os.path.basename(fpath).split('_')[-3])


def load_wt_widths():
    """
    Extract per-simulation max stripe and interstripe widths from WT mat files.

    From paper (p. 5118):
      stripe width      ≈ 2 × max β₁ persistence of X^d (xanC_widths field)
      interstripe width ≈ 2 × max β₁ persistence of X^l (xanS_widths field)
    """
    mat_dir = os.path.join(DATA_DIR, 'mat_wt')
    R_vals, stripe, inter = [], [], []

    for fp in sorted(glob.glob(os.path.join(mat_dir, '*.mat'))):
        r = _mat_r(fp)
        struct = loadmat(fp)['pattern_info'][0, 0]
        xC = struct['xanC_widths']   # (100, 1) object array of β₁ persistence arrays
        xS = struct['xanS_widths']

        for i in range(xC.shape[0]):
            vc = np.asarray(xC[i, 0], dtype=float).flatten()
            vs = np.asarray(xS[i, 0], dtype=float).flatten()
            vc = vc[(vc > 0) & np.isfinite(vc)]
            vs = vs[(vs > 0) & np.isfinite(vs)]

            R_vals.append(r)
            stripe.append(2.0 * np.max(vc) if len(vc) > 0 else np.nan)
            inter.append(2.0 * np.max(vs) if len(vs) > 0 else np.nan)

    return np.array(R_vals), np.array(stripe), np.array(inter)


def load_csv(path):
    with open(path) as f:
        header = f.readline().strip().split(',')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)
    return {col: data[:, i] for i, col in enumerate(header)}


# ── Regression ────────────────────────────────────────────────────────────────

def fit_regression(x, y, pos_only=True):
    """Return (slope, intercept, R²) or None if insufficient data."""
    mask = np.isfinite(y)
    if pos_only:
        mask &= (y > 0)
    xv, yv = x[mask], y[mask]
    if len(xv) < 20:
        return None
    m = LinearRegression()
    m.fit(xv.reshape(-1, 1), yv)
    r2 = r2_score(yv, m.predict(xv.reshape(-1, 1)))
    return m.coef_[0], m.intercept_, r2


def reg_label(slope, intercept, r2):
    sign = '+' if intercept >= 0 else '−'
    return f'y = {slope:.2f}x {sign} {abs(intercept):.1f}\n$R^2$ = {r2:.2f}'


# ── Scatter with per-R medians ────────────────────────────────────────────────

def scatter_with_medians(ax, R, y, color, alpha_pts=0.07, s_pts=3):
    valid = np.isfinite(y)
    ax.scatter(R[valid], y[valid], s=s_pts, alpha=alpha_pts, color=color, zorder=1)
    for rv in sorted(set(R)):
        pts = y[R == rv]
        pts = pts[np.isfinite(pts)]
        if len(pts) > 0:
            ax.scatter(rv, np.median(pts), s=22, color=color, zorder=3,
                       edgecolors='k', linewidths=0.4)


def style_ax(ax, ylabel='', title=''):
    ax.set_xlabel('Interaction scale in M birth (μm)', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold', color='#1a3a5c')
    ax.tick_params(labelsize=7)
    ax.set_xlim(XLIM)
    ax.axvline(DEFAULT_R, color='gray', ls='--', lw=1.0, alpha=0.7, zorder=2)
    ax.spines[['top', 'right']].set_visible(False)


def add_reg_box(ax, slope, intercept, r2, color='#1a3a5c', y_pos=0.93):
    ax.text(0.05, y_pos, reg_label(slope, intercept, r2),
            transform=ax.transAxes, fontsize=8, va='top', color=color,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#cccccc'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading WT mat files …')
    R_wt, stripe_um, inter_um = load_wt_widths()

    print('Loading pfeffer / shady CSVs …')
    pf = load_csv(os.path.join(DATA_DIR, 'pfeffer_LSTest_annulusIn_MB.csv'))
    sh = load_csv(os.path.join(DATA_DIR, 'shady_LSTest_annulusIn_MB.csv'))
    wt_csv = load_csv(os.path.join(DATA_DIR, 'WT_LSTest_annulusIn_MB.csv'))

    R_pf = pf['annulus_In_MB']
    R_sh = sh['annulus_In_MB']
    R_wt_csv = wt_csv['annulus_In_MB']

    # WT widths: paper y-axis is in mm; regression equation is in μm
    stripe_mm = stripe_um / 1000.0
    inter_mm  = inter_um  / 1000.0

    x_line = np.linspace(10, 400, 300)

    # Regressions
    res_A = fit_regression(R_wt, stripe_mm)
    res_B = fit_regression(R_wt, inter_mm)
    res_pf_E = fit_regression(R_pf, pf['med_cluster_size_all'])
    res_sh_E = fit_regression(R_sh, sh['mel_med_cluster_size_all'])

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.34,
                           left=0.07, right=0.97, top=0.87, bottom=0.09)

    # ── Panel A: WT Max Stripe Width ──────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])
    scatter_with_medians(ax_A, R_wt, stripe_mm, '#2e6da4')
    if res_A:
        s, i, r2 = res_A
        ax_A.plot(x_line, s * x_line + i, '-', color='navy', lw=2, zorder=4)
        add_reg_box(ax_A, s * 1000, i * 1000, r2)   # display equation in μm
    style_ax(ax_A, ylabel='Max stripe width (mm)', title='A — Wild-type  |  Max stripe width')

    # ── Panel B: WT Max Interstripe Width ─────────────────────────────────────
    ax_B = fig.add_subplot(gs[0, 1])
    scatter_with_medians(ax_B, R_wt, inter_mm, '#2e6da4')
    if res_B:
        s, i, r2 = res_B
        ax_B.plot(x_line, s * x_line + i, '-', color='navy', lw=2, zorder=4)
        add_reg_box(ax_B, s * 1000, i * 1000, r2)
    style_ax(ax_B, ylabel='Max interstripe width (mm)',
             title='B — Wild-type  |  Max interstripe width')

    # ── Panel C: WT Stripe Curviness ──────────────────────────────────────────
    ax_C = fig.add_subplot(gs[0, 2])
    curv = wt_csv['avg_straightness_all']
    scatter_with_medians(ax_C, R_wt_csv, curv, '#2e6da4')
    style_ax(ax_C, ylabel='Stripe curviness (%)',
             title='C — Wild-type  |  Stripe curviness')

    # ── Panel D: Spot Count (pfeffer + shady) ─────────────────────────────────
    # Paper Fig. 7D shows BOTH pfeffer and shady spot counts (β₀ of M cells).
    ax_D = fig.add_subplot(gs[1, 0])
    count_pf = pf['b0_mel0_all']
    count_sh = sh['b0_mel0_all']
    scatter_with_medians(ax_D, R_pf, count_pf, '#e76f51')
    scatter_with_medians(ax_D, R_sh, count_sh, '#2d6a4f')
    # Per-R median curves to show the non-monotone trend for each mutant
    for R_arr, cnt, col in ((R_pf, count_pf, '#b03020'), (R_sh, count_sh, '#1a5c3a')):
        r_unique = np.array(sorted(set(R_arr)))
        med = np.array([np.median(cnt[R_arr == rv][np.isfinite(cnt[R_arr == rv])])
                        for rv in r_unique])
        ax_D.plot(r_unique, med, '-', color=col, lw=1.5, zorder=4, alpha=0.85)
    ax_D.text(0.05, 0.93, 'Non-linear / no regression\n(non-monotone)',
              transform=ax_D.transAxes, fontsize=8, va='top',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#cccccc'))
    ax_D.legend(handles=[
        Line2D([0], [0], color='#e76f51', lw=2, label='Pfeffer'),
        Line2D([0], [0], color='#2d6a4f', lw=2, label='Shady'),
    ], fontsize=8, loc='upper right')
    style_ax(ax_D, ylabel='Number of spots',
             title='D — Spot count  |  Pfeffer + Shady (non-linear)')

    # ── Panel E: Spot Size (pfeffer + shady) ─────────────────────────────────
    ax_E = fig.add_subplot(gs[1, 1])
    sz_pf = pf['med_cluster_size_all']
    sz_sh = sh['mel_med_cluster_size_all']
    scatter_with_medians(ax_E, R_pf, sz_pf, '#e76f51')
    scatter_with_medians(ax_E, R_sh, sz_sh, '#2d6a4f')

    if res_pf_E:
        s, i, r2 = res_pf_E
        ax_E.plot(x_line, s * x_line + i, '-', color='#e76f51', lw=2, zorder=4)
        add_reg_box(ax_E, s, i, r2, color='#c04010', y_pos=0.93)
    if res_sh_E:
        s, i, r2 = res_sh_E
        ax_E.plot(x_line, s * x_line + i, '-', color='#2d6a4f', lw=2, zorder=4)
        add_reg_box(ax_E, s, i, r2, color='#1a5c3a', y_pos=0.65)

    # Legend
    ax_E.legend(handles=[
        Line2D([0], [0], color='#e76f51', lw=2, label='Pfeffer'),
        Line2D([0], [0], color='#2d6a4f', lw=2, label='Shady'),
    ], fontsize=8, loc='lower right')
    style_ax(ax_E, ylabel='Median spot size (# cells)',
             title='E — Spot size  |  Pfeffer + Shady')

    # ── Panel F: Spot Roundness (pfeffer + shady) ─────────────────────────────
    ax_F = fig.add_subplot(gs[1, 2])
    rd_pf = pf['med_pca_ratio_iriL_all']
    rd_sh = sh['med_pca_ratio_mel_all']
    scatter_with_medians(ax_F, R_pf, rd_pf, '#e76f51')
    scatter_with_medians(ax_F, R_sh, rd_sh, '#2d6a4f')
    ax_F.text(0.05, 0.93, 'No linear trend\n(scattered)',
              transform=ax_F.transAxes, fontsize=8, va='top',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#cccccc'))
    ax_F.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e76f51', ms=7, label='Pfeffer'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2d6a4f', ms=7, label='Shady'),
    ], fontsize=8, loc='upper right')
    style_ax(ax_F, ylabel='Spot roundness (PCA ratio)',
             title='F — Spot roundness  |  Pfeffer + Shady')

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        'Recreation of Figure 7 statistics & regressions — McGuirl, Volkening & Sandstede (PNAS 2020)\n'
        'Effect of Ω_long inner radius R on in silico pattern statistics'
        '  [Figshare simulation data, 100 sims × 16 R values]\n'
        'Note: regression/statistics recreation, not a pixel-match — omits paper 7G example images; '
        'uses scatter+medians rather than KDE density panels.',
        fontsize=9, fontweight='bold', color='#1a3a5c', y=0.985,
    )

    out = 'fig7_regression_recreation.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    # ── Comparison table ──────────────────────────────────────────────────────
    paper = {
        'WT stripe width (μm)':      (0.91, 1.97,  487.95),
        'WT interstripe width (μm)': (0.81, 1.71,  471.76),
        'Pfeffer spot size (cells)': (0.72, 0.37,  -46.36),
        'Shady spot size (cells)':   (0.90, 0.24,  -29.72),
    }

    results = {}
    if res_A:
        s, i, r2 = res_A
        results['WT stripe width (μm)'] = (r2, s * 1000, i * 1000)
    if res_B:
        s, i, r2 = res_B
        results['WT interstripe width (μm)'] = (r2, s * 1000, i * 1000)
    if res_pf_E:
        s, i, r2 = res_pf_E
        results['Pfeffer spot size (cells)'] = (r2, s, i)
    if res_sh_E:
        s, i, r2 = res_sh_E
        results['Shady spot size (cells)'] = (r2, s, i)

    print()
    print(f"{'Metric':<30} {'R² (real)':>10}  {'R² (paper)':>11}  "
          f"{'slope':>7}  {'intercept':>11}  match")
    print('─' * 82)
    for name, (r2_p, sl_p, in_p) in paper.items():
        if name in results:
            r2_r, sl_r, in_r = results[name]
            ok = '✓' if abs(r2_r - r2_p) < 0.12 else '✗'
            print(f'{name:<30} {r2_r:>10.3f}  {r2_p:>11.2f}  '
                  f'{sl_r:>7.2f}  {in_r:>11.1f}  {ok}')

    print()
    print('Data source: Figshare article 11568675 (in silico simulations)')
    print('  Stripe/interstripe: 2 × max β₁ persistence (X^d and X^l cells)')
    print('  Spot size:          median cluster size (single-linkage hierarchical clustering)')
    print('  Spot roundness:     median PCA eigenvalue ratio (no linear trend)')


if __name__ == '__main__':
    main()
