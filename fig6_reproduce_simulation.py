#!/usr/bin/env python3
"""
Scaled-down demonstration of Figure 6 (stochasticity experiment) from:
  "Topological Data Analysis of Zebrafish Patterns"
  McGuirl, Volkening & Sandstede — PNAS 2020
  DOI: 10.1073/pnas.1917763117

This is an ILLUSTRATIVE SUBSAMPLE, not the full 24,000-simulation study, but the
quantification now follows the paper's published pipeline (Methods + SI Appendix)
so the Betti counts match the paper instead of over-counting boundary bands:

  Tier 0 (visual):  raw pigment-cell coordinates, σ = 0 (default) vs σ = 0.2
  Tier 1 (TDA):     persistent homology (Ripser) on the SAME patterns, periodic
                    in x (as in the model), quantifying β₀ (spots / components)
                    and β₁ (stripe / interstripe loops).

Paper quantification rules implemented here (this is what fixes the 3/4 → 2/3
mismatch — previously every band that wrapped the domain, including the
half-formed dorsal/ventral edge bands, was counted):

  1. TRIM. Remove cells in the top and bottom 10% of the domain before counting,
     "to help avoid quantifying partially formed stripes or spots" (paper Methods).
  2. β₀ persistence threshold T0p, per cell type:
        melanophores  T0p = 90 µm
        iridophores   T0p = 100 µm
     A component counts toward β₀ if it persists past T0p (the essential/global
     component always counts).
  3. β₁ rule: a loop counts toward β₁ only if persistence ≥ T1p AND birth radius
     ≤ T1b, where T1p = 200 µm (universal) and the per-type birth cap T1b is:
        melanophores (stripes)           T1b = 90 µm
        dense xanthophores X^d (interstr.) T1b = 80 µm
        loose xanthophores X^l            T1b = 100 µm
     The birth cap is what removes the sparse, late-born edge/gap loops.
  4. Pfeffer spots are counted from IRIDOPHORES, not melanophores: in pfeffer "M
     appear randomly on the domain, so using these cells to count spots would
     introduce spurious connected components" (paper Methods).

Data sources (raw coordinates — NOT committed to this repo, restore from Figshare):
  σ = 0   (default regime):  data/samples/Out_*_default_1.mat
  σ = 0.2 (20% noise):       data/sigma/Out_*_pcpdTest_sigma_20_*.mat
Each .mat must contain the per-cell coordinate arrays (cellsM, cellsXc, …) and
the time-indexed domain width boundaryX.

Run:    venv/bin/python fig6_reproduce_simulation.py
Output: fig6_demo.png
"""

import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ripser import ripser

PAD           = -1000.0   # rows below this are padding (actual pad value = -40000)
FINAL_T       = -1        # final simulation day
RIPSER_THRESH = 700.0     # µm: max filtration radius (must exceed interstripe scale
                          #     so stripe loops are born and die within the diagram)
MAX_PTS       = 3000      # subsample cap per cell set (for Ripser speed). Subsampling
                          #     can distort topology, so this is set high; a warning is
                          #     printed whenever it actually triggers.

# ── Paper quantification parameters (Methods + SI Appendix) ─────────────────────
TRIM_FRAC = 0.10          # remove top and bottom 10% of the domain before counting
T1P       = 200.0         # µm: universal β₁ persistence threshold

# β₀ persistence thresholds T0p (µm), per cell type. Mel/iri are given explicitly in
# the paper; xanthophore β₀ is not used for the WT/pfeffer story, so it defaults to 90.
T0P = {'M': 90.0, 'Xd': 90.0, 'Xl': 90.0, 'Il': 100.0, 'Id': 100.0}

# β₁ birth-radius caps T1b (µm), per cell type (paper SI Appendix).
T1B = {'M': 90.0, 'Xd': 80.0, 'Xl': 100.0}

# Cell-type code → coordinate variable name in the Out_*.mat files.
# These are the Volkening-model variable names; adjust if your .mat files differ.
CELLVAR = {'M': 'cellsM', 'Xd': 'cellsXc', 'Xl': 'cellsXl',
           'Il': 'cellsIl', 'Id': 'cellsId'}

rng = np.random.default_rng(0)


# ── Data extraction ─────────────────────────────────────────────────────────────

def get_cells(mat, code, t=FINAL_T):
    """Return valid (x, y) coords of one cell type at timepoint t, padding removed."""
    var = CELLVAR.get(code, code)
    if var not in mat:
        return np.empty((0, 2))
    arr = mat[var][:, :, t]
    valid = arr[:, 0] > PAD
    return arr[valid]


def domain_width(mat, t=FINAL_T):
    """Periodic-x domain width Lx at timepoint t."""
    return float(mat['boundaryX'][0, t])


def domain_yrange(mat, t=FINAL_T):
    """
    (y_lo, y_hi) of the domain at timepoint t. Uses boundaryY if present (analogous
    to boundaryX); otherwise falls back to the y-extent of all pigment cells, which
    for a developed pattern fills the domain and is a close proxy.
    """
    if 'boundaryY' in mat:
        by = np.asarray(mat['boundaryY'])
        col = by[:, t] if by.ndim == 2 else by.ravel()
        if col.size >= 2:
            return float(np.min(col)), float(np.max(col))
        return 0.0, float(col.ravel()[0])
    ys = []
    for code in CELLVAR:
        c = get_cells(mat, code, t)
        if len(c):
            ys.append(c[:, 1])
    if not ys:
        return 0.0, 0.0
    ys = np.concatenate(ys)
    return float(ys.min()), float(ys.max())


def trim_y(pts, y_lo, y_hi, frac=TRIM_FRAC):
    """Drop cells in the top/bottom `frac` of the domain (paper's partial-band guard)."""
    if len(pts) == 0:
        return pts
    h = y_hi - y_lo
    lo, hi = y_lo + frac * h, y_hi - frac * h
    keep = (pts[:, 1] >= lo) & (pts[:, 1] <= hi)
    return pts[keep]


# ── Periodic distance + persistent homology (Tier 1) ───────────────────────────

def periodic_distance_matrix(pts, Lx):
    """Pairwise distances; periodic in x (period Lx), Euclidean in y."""
    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    dx = np.abs(x - x.T)
    dx = np.minimum(dx, Lx - dx)        # wrap-around in x
    dy = np.abs(y - y.T)
    return np.sqrt(dx**2 + dy**2)


def persistence(pts, Lx, label=''):
    """Return (h0, h1, n_used) persistence diagrams on the periodic domain."""
    n = len(pts)
    if n < 5:
        return None, None, n
    if MAX_PTS and n > MAX_PTS:
        print(f'    [warn] subsampling {label} from {n} to {MAX_PTS} points '
              f'— may perturb topology', file=sys.stderr)
        pts = pts[rng.choice(n, MAX_PTS, replace=False)]
    D = periodic_distance_matrix(pts, Lx)
    dgms = ripser(D, distance_matrix=True, maxdim=1, thresh=RIPSER_THRESH)['dgms']
    return dgms[0], dgms[1], len(pts)


def count_b0(h0, t0p):
    """β₀ = components persisting past T0p, including the essential global component."""
    if h0 is None:
        return np.nan
    return int(np.sum((h0[:, 1] >= t0p) | ~np.isfinite(h0[:, 1])))


def count_b1(h1, t1b, t1p=T1P):
    """β₁ = loops with persistence ≥ T1p AND birth radius ≤ T1b (paper rule)."""
    if h1 is None:
        return np.nan
    if len(h1) == 0:
        return 0
    pers = h1[:, 1] - h1[:, 0]
    return int(np.sum((pers >= t1p) & (h1[:, 0] <= t1b)))


def betti_for_type(mat, code, Lx, y_lo, y_hi):
    """Trim to the analysis region, then return (β₀, β₁, n_used) for one cell type."""
    pts = trim_y(get_cells(mat, code), y_lo, y_hi)
    h0, h1, n = persistence(pts, Lx, label=code)
    b0 = count_b0(h0, T0P.get(code, 90.0))
    b1 = count_b1(h1, T1B.get(code, 90.0)) if code in T1B else np.nan
    return b0, b1, n


# ── Conditions ──────────────────────────────────────────────────────────────────
# WT: melanophores (M, black stripes) + dense xanthophores (X^d = Xc, interstripes).
#     stripes      = β₁(M),  T1b = 90 µm
#     interstripes = β₁(X^d), T1b = 80 µm
# pfeffer: spots counted from iridophores (β₀, T0p = 100 µm), NOT melanophores.
CONDITIONS = {
    'Wild-type': {'color_M': 'k',       'color_X': '#e8a33d', 'spot_type': None},
    'Pfeffer':   {'color_M': '#1a1a1a', 'color_X': None,      'spot_type': 'Il'},
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    files = find_files()
    if not any(files.values()):
        sys.exit(
            'ERROR: no raw-coordinate simulation files found.\n'
            'This script needs the Figshare dumps (cellsM/cellsXc/… coordinates),\n'
            'which are NOT committed to this repo. Restore them to:\n'
            '  data/samples/Out_WT_default_1.mat\n'
            '  data/samples/Out_pfef_default_1.mat\n'
            '  data/sigma/Out_WT_pcpdTest_sigma_20_*.mat\n'
            '  data/sigma/Out_pfef_pcpdTest_sigma_20_*.mat\n'
            '(The committed data/mat_wt/*.mat files hold only precomputed summaries,\n'
            ' not coordinates, so the recompute cannot run on them.)'
        )

    # ── Tier 1: quantify β₀/β₁ averaged over the available sims ──────────────────
    print('Computing persistent homology (Ripser, periodic domain, paper thresholds) …\n')
    quant = {}   # (cond, sigma) -> dict of mean betti numbers
    for (cond, sigma), paths in files.items():
        if not paths:
            continue
        spot_type = CONDITIONS[cond]['spot_type']
        stripes, interstr, spots = [], [], []
        for p in paths:
            mat = loadmat(p)
            Lx = domain_width(mat)
            y_lo, y_hi = domain_yrange(mat)

            # WT stripes = β₁(M); WT interstripes = β₁(X^d)
            _, b1M, _ = betti_for_type(mat, 'M', Lx, y_lo, y_hi)
            stripes.append(b1M)
            b0Xd, b1Xd, nX = betti_for_type(mat, 'Xd', Lx, y_lo, y_hi)
            if nX >= 5:
                interstr.append(b1Xd)

            # mutant spots = β₀ of the designated spot cell type (iridophores for pfeffer)
            if spot_type:
                b0s, _, ns = betti_for_type(mat, spot_type, Lx, y_lo, y_hi)
                if ns >= 5:
                    spots.append(b0s)

        quant[(cond, sigma)] = {
            'stripes':  np.nanmean(stripes)  if stripes  else np.nan,
            'interstr': np.nanmean(interstr) if interstr else np.nan,
            'spots':    np.nanmean(spots)    if spots    else np.nan,
            'spot_type': spot_type, 'n': len(paths),
        }
        q = quant[(cond, sigma)]
        msg = f'  {cond:<10} {sigma:<16}  n={len(paths)}  '
        msg += f"β₁(M)={q['stripes']:.1f} stripes  "
        if np.isfinite(q['interstr']):
            msg += f"β₁(X^d)={q['interstr']:.1f} interstripes  "
        if np.isfinite(q['spots']):
            msg += f"β₀({spot_type})={q['spots']:.1f} spots"
        print(msg)

    # ── Tier 0: visual panels (2×2) with β annotations + trim lines ──────────────
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
            y_lo, y_hi = domain_yrange(mat)
            M = get_cells(mat, 'M')
            ax.scatter(M[:, 0], M[:, 1], s=3, color=cfg['color_M'], zorder=2)
            if cfg['color_X']:
                X = get_cells(mat, 'Xd')
                ax.scatter(X[:, 0], X[:, 1], s=2, color=cfg['color_X'],
                           alpha=0.55, zorder=1)

            # show the excluded top/bottom 10% bands (counted region is between lines)
            h = y_hi - y_lo
            for yy in (y_lo + TRIM_FRAC * h, y_hi - TRIM_FRAC * h):
                ax.axhline(yy, color='#cc3333', lw=0.8, ls='--', alpha=0.7, zorder=3)

            q = quant.get((cond, sigma))
            if q:
                lines = [f"β₁(M) = {q['stripes']:.1f}   (stripes)"]
                if np.isfinite(q['interstr']):
                    lines.append(f"β₁(X$^d$) = {q['interstr']:.1f}   (interstripes)")
                if np.isfinite(q['spots']):
                    lines.append(f"β₀({q['spot_type']}) = {q['spots']:.1f}   (spots)")
                lines.append(f"(n = {q['n']}, top/bottom 10% trimmed)")
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
        '[subsample · TDA: Ripser, paper thresholds, top/bottom 10% trimmed (dashed)]',
        fontsize=11, fontweight='bold', color='#1a3a5c', y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = 'fig6_demo.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')

    # ── Interpretation note ──────────────────────────────────────────────────────
    print()
    print('Interpretation (1.3a):')
    print('  β₁(M)   = dark-stripe loops      → robust for WT, degrades for mutants.')
    print('  β₁(X^d) = interstripe loops      → robust for WT under noise.')
    print('  β₀      = spot/component count   → mutants scatter into more spots.')
    print('  Method follows the paper: periodic-x persistent homology with the top/')
    print('  bottom 10% trimmed and the published T0p / T1p / T1b thresholds, so the')
    print('  counts no longer include partial dorsal/ventral edge bands.')


if __name__ == '__main__':
    main()
