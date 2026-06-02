#!/usr/bin/env python3
"""
Scaled-down demonstration of Figure 6 (stochasticity experiment) from:
  "Topological Data Analysis of Zebrafish Patterns"
  McGuirl, Volkening & Sandstede — PNAS 2020
  DOI: 10.1073/pnas.1917763117

This is an ILLUSTRATIVE SUBSAMPLE, not the full 24,000-simulation study.
It shows the core 1.3a story — wild-type patterns stay characteristic under
noise (σ) while mutant patterns degrade — using:

  Tier 0 (visual):  raw pigment-cell coordinates, σ = 0 (default) vs σ = 0.2
  Tier 1 (TDA):     persistent homology (Ripser) on the SAME patterns,
                    quantifying β₀ (spots / components) and β₁ (stripe loops),
                    on a PERIODIC domain (periodic in x, as in the model).

Data sources:
  σ = 0   (default regime):  GitHub sample_inputs  (sandstede-lab repo)
  σ = 0.2 (20% noise):       Figshare article 11328419, fetched per-member
                             with remotezip (a few sims, not the 3 GB archive)

Run:    venv/bin/python reproduce_fig6_demo.py
Output: fig6_demo.png
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ripser import ripser

PAD          = -1000.0   # rows below this are padding (actual pad value = -40000)
FINAL_T      = -1        # final simulation day
PERS_CUT     = 200.0     # µm: persistence cutoff for significant features
                         #     (matches paper's pers_cutoff = 200 in quantify_stripes.m)
RIPSER_THRESH = 700.0    # µm: max filtration radius (must exceed interstripe scale
                         #     so stripe loops are born and die within the diagram)
MAX_PTS      = 1100      # subsample cap per cell set (keeps Ripser fast)

rng = np.random.default_rng(0)


# ── Data extraction ───────────────────────────────────────────────────────────

def get_cells(mat, var, t=FINAL_T):
    """Return valid (x, y) coords of one cell type at timepoint t, padding removed."""
    arr = mat[var][:, :, t]
    valid = arr[:, 0] > PAD
    return arr[valid]


def domain_width(mat, t=FINAL_T):
    """Periodic-x domain width Lx at timepoint t."""
    return float(mat['boundaryX'][0, t])


# ── Periodic distance + persistent homology (Tier 1) ───────────────────────────

def periodic_distance_matrix(pts, Lx):
    """Pairwise distances; periodic in x (period Lx), Euclidean in y."""
    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    dx = np.abs(x - x.T)
    dx = np.minimum(dx, Lx - dx)        # wrap-around in x
    dy = np.abs(y - y.T)
    return np.sqrt(dx**2 + dy**2)


def betti_numbers(pts, Lx):
    """
    Compute (β₀, β₁) for a cell point cloud on the periodic domain, counting
    only topological features whose persistence exceeds PERS_CUT (= 200 µm,
    the paper's significance cutoff).
      β₀ = # connected components that persist past the cutoff  (spots / bands)
      β₁ = # loops that persist past the cutoff                 (stripe loops)
    Returns (b0, b1, n_points) or (np.nan, np.nan, n) if too few points.
    """
    n = len(pts)
    if n < 5:
        return np.nan, np.nan, n
    if n > MAX_PTS:                      # subsample for speed
        idx = rng.choice(n, MAX_PTS, replace=False)
        pts = pts[idx]
    D = periodic_distance_matrix(pts, Lx)
    dgms = ripser(D, distance_matrix=True,
                  maxdim=1, thresh=RIPSER_THRESH)['dgms']
    h0, h1 = dgms[0], dgms[1]
    # H0 birth = 0, so persistence = death; the infinite bar is the global component.
    b0 = int(np.sum((h0[:, 1] > PERS_CUT) | ~np.isfinite(h0[:, 1])))
    b1 = int(np.sum((h1[:, 1] - h1[:, 0]) > PERS_CUT))
    return b0, b1, n


# ── Cell sets to load per condition ─────────────────────────────────────────────
# WT: melanophores (M, black stripes) + dense xanthophores (X^d = Xc, interstripes)
# pfeffer: no X^d; melanophores form spots
CONDITIONS = {
    'Wild-type': {'color_M': 'k',       'color_X': '#e8a33d'},   # M black + X^d gold
    'Pfeffer':   {'color_M': '#1a1a1a', 'color_X': None},        # no X^d in pfeffer
}


def find_files():
    """Map (condition, sigma_label) -> list of mat file paths actually present."""
    files = {
        ('Wild-type', 'σ = 0 (default)'):  ['data/samples/Out_WT_default_1.mat'],
        ('Wild-type', 'σ = 0.2'):          sorted(glob.glob('data/sigma/Out_WT_pcpdTest_sigma_20_*.mat')),
        ('Pfeffer',   'σ = 0 (default)'):  ['data/samples/Out_pfef_default_1.mat'],
        ('Pfeffer',   'σ = 0.2'):          sorted(glob.glob('data/sigma/Out_pfef_pcpdTest_sigma_20_*.mat')),
    }
    return {k: [f for f in v if os.path.exists(f)] for k, v in files.items()}


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    files = find_files()

    # ── Tier 1: quantify β₀/β₁ averaged over the available sims ──────────────────
    # β₀(M) = melanophore components  → many for spotty mutants, few for WT stripes
    # β₁(M) = melanophore loops       → stripe count for WT (stripes wrap the domain)
    # β₁(X^d) = dense-xanthophore loops → interstripe count for WT (absent in pfeffer)
    print('Computing persistent homology (Ripser, periodic domain) …\n')
    quant = {}   # (cond, sigma) -> dict of mean betti numbers
    for (cond, sigma), paths in files.items():
        if not paths:
            continue
        b0M, b1M, b1X = [], [], []
        for p in paths:
            mat = loadmat(p)
            Lx = domain_width(mat)
            M = get_cells(mat, 'cellsM')
            X = get_cells(mat, 'cellsXc')          # X^d (dense xanthophores)
            b0, b1, _ = betti_numbers(M, Lx)
            b0M.append(b0)
            b1M.append(b1)
            if len(X) >= 5:
                _, b1x, _ = betti_numbers(X, Lx)
                b1X.append(b1x)
        quant[(cond, sigma)] = {
            'b0M': np.nanmean(b0M), 'b1M': np.nanmean(b1M),
            'b1X': (np.nanmean(b1X) if b1X else np.nan), 'n': len(paths),
        }
        x_str = f"{quant[(cond, sigma)]['b1X']:.1f}" if b1X else '—'
        print(f'  {cond:<10} {sigma:<16}  n={len(paths)}  '
              f"β₀(M)={np.nanmean(b0M):4.1f}  β₁(M)={np.nanmean(b1M):.1f}  β₁(X^d)={x_str}")

    # ── Tier 0: visual panels (2×2) with β annotations ───────────────────────────
    sigmas = ['σ = 0 (default)', 'σ = 0.2']
    conds  = ['Wild-type', 'Pfeffer']

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor('white')

    for r, cond in enumerate(conds):
        cfg = CONDITIONS[cond]
        for c, sigma in enumerate(sigmas):
            ax = axes[r, c]
            paths = files.get((cond, sigma), [])
            if not paths:
                ax.text(0.5, 0.5, 'data missing', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_axis_off()
                continue

            mat = loadmat(paths[0])        # representative simulation
            M = get_cells(mat, 'cellsM')
            ax.scatter(M[:, 0], M[:, 1], s=3, color=cfg['color_M'], zorder=2)
            if cfg['color_X']:
                X = get_cells(mat, 'cellsXc')
                ax.scatter(X[:, 0], X[:, 1], s=2, color=cfg['color_X'],
                           alpha=0.55, zorder=1)

            # β annotation (averaged over available sims)
            q = quant.get((cond, sigma))
            if q:
                lines = [f"β₀(M) = {q['b0M']:.1f}   (spots/bands)",
                         f"β₁(M) = {q['b1M']:.1f}   (stripe loops)"]
                if np.isfinite(q['b1X']):
                    lines.append(f"β₁(X$^d$) = {q['b1X']:.1f}   (interstripes)")
                lines.append(f"(n = {q['n']})")
                ax.text(0.015, 0.985, '\n'.join(lines), transform=ax.transAxes,
                        fontsize=8.5, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  alpha=0.88, ec='#bbbbbb'))

            ax.set_title(f'{cond}  |  {sigma}', fontsize=10, fontweight='bold',
                         color='#1a3a5c')
            ax.set_xlabel('x (μm)', fontsize=8)
            ax.set_ylabel('y (μm)', fontsize=8)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)

    fig.suptitle(
        'Figure 6 demonstration — effect of cell-interaction noise σ on pattern robustness\n'
        'Wild-type stripes persist under σ = 0.2 noise; pfeffer spots degrade  '
        '[illustrative subsample · TDA: Ripser on periodic domain]',
        fontsize=11, fontweight='bold', color='#1a3a5c', y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = 'fig6_demo.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')

    # ── Interpretation note ──────────────────────────────────────────────────────
    print()
    print('Interpretation (1.3a):')
    print('  β₁ = number of loops = interstripe bands (X^d) for WT — should stay robust.')
    print('  β₀ = number of components = spots (M) — mutants scatter into more/smaller spots.')
    print('  Method matches paper: persistent homology on a periodic-x domain (Ripser).')
    print('  Caveat: subsample (a few sims), not the full 1,000-per-σ study.')


if __name__ == '__main__':
    main()
