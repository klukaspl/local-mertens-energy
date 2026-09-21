#!/usr/bin/env python3
"""Generate all numerical figures of the paper from the CSV files."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L = math.log(1.5)
TAIL_C = 6 * (L + 1 - math.exp(-L)) / math.pi ** 3   # conjectural Gonek tail, ~0.143
REF_MEAN = 0.02222


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--trajectory', type=Path, default=Path('mertens_energy_trajectory_comparison.csv'))
    ap.add_argument('--energy', type=Path, default=Path('mertens_local_energy_1e8.csv'))
    ap.add_argument('--meanconv', type=Path, default=Path('mean_convergence_zeros.csv'))
    ap.add_argument('--weights', type=Path, default=Path('brown_modal_weights.csv'))
    ap.add_argument('--model', type=Path, default=Path('random_phase_samples.csv'))
    ap.add_argument('--outdir', type=Path, default=Path('figures'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1. trajectory (last five log-units)
    df = pd.read_csv(args.trajectory)
    mask = df['Y'] >= df['Y'].max() - 5.0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df.loc[mask, 'Y'], df.loc[mask, 'empirical'], label='empirical', lw=1.6, color='C0')
    for N, col, lw in ((1, 'C1', 0.9), (10, 'C2', 0.9), (80, 'C3', 1.0), (1000, 'k', 0.8)):
        name = f'pred_{N}zero'
        if name in df:
            ax.plot(df.loc[mask, 'Y'], df.loc[mask, name], lw=lw, color=col,
                    label=f'{N} zero' + ('s' if N > 1 else ''))
    ax.set_xlabel('Y = log X'); ax.set_ylabel('local Mertens energy')
    ax.set_title('Local Mertens energy: empirical trajectory vs explicit formula')
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / 'mertens_energy_trajectory_1_10_80zeros.png', dpi=180)
    plt.close(fig)

    # 2. convergence of the zero-sum prediction for the mean
    c = pd.read_csv(args.meanconv)
    idx = [n - 1 for n in (10, 20, 40, 80, 120, 160, 200, 300, 500, 1000) if n <= len(c)]
    n = c['n'].to_numpy()[idx]; part = c['energy_mean_partial'].to_numpy()[idx]
    gam = c['gamma'].to_numpy()[idx]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(n, part, 'o-', label='partial zero sum')
    ax.semilogx(n, part + TAIL_C / gam, 'o-', label=r'partial sum + $0.143/\gamma_N$')
    ax.axhline(REF_MEAN, ls='--', lw=1, label=f'reference {REF_MEAN}')
    ax.set_xlabel('number of positive zeros'); ax.set_ylabel('predicted mean local energy')
    ax.set_title('Convergence of the zero-sum prediction for the mean')
    ax.grid(alpha=.25, which='both'); ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / 'mertens_mean_convergence_zeros.png', dpi=180)
    plt.close(fig)

    # 3. histogram: arithmetic data vs random-phase model
    e = pd.read_csv(args.energy)['energy'].to_numpy()
    bins = np.linspace(0, 0.07, 57)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.hist(e, bins=bins, density=True, alpha=.65, label='empirical local Mertens energy')
    if args.model.exists():
        m = pd.read_csv(args.model)['energy'].to_numpy()
        ax.hist(m, bins=bins, density=True, histtype='step', color='k', lw=1.3,
                label='random-phase model (300 zeros)')
    for x in (0.00693, 0.01843):
        ax.axvline(x, ls='--', lw=1, color='C0')
    ax.axvline(x, ls='--', lw=1, color='C0', label='first-zero arcsine support')
    ax.axvline(REF_MEAN, ls=':', lw=1.2, color='C0', label=f'predicted limiting mean $\\approx$ {REF_MEAN}')
    ax.set_xlabel('local Mertens energy'); ax.set_ylabel('density')
    ax.set_title('Empirical distribution, random-phase model, and the first-zero scale')
    ax.grid(alpha=.2); ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / 'mertens_energy_histogram_clean.png', dpi=180)
    plt.close(fig)

    # 4. modal weights
    w = pd.read_csv(args.weights)
    sub = w[(w['j'] <= 5) & (w['k'] <= 5)]
    mat = sub.pivot(index='j', columns='k', values='p_j_gamma_k').to_numpy()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    im = ax.imshow(mat, aspect='auto', vmin=0, vmax=np.nanmax(mat))
    ax.set_xticks(range(5), [f'$\\gamma_{k}$' for k in range(1, 6)])
    ax.set_yticks(range(5), [f'$j={j}$' for j in range(1, 6)])
    ax.set_xlabel('Riemann zero'); ax.set_ylabel('Brownian mode')
    ax.set_title(r'Modal weights $p_j(\gamma_k)$ for the first five zeros and modes')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{100*mat[i, j]:.1f}%', ha='center', va='center',
                    color='k' if mat[i, j] > 0.3 else 'w', fontsize=9)
    fig.colorbar(im, ax=ax, label=r'$p_j(\gamma_k)$')
    fig.tight_layout()
    fig.savefig(args.outdir / 'brownian_modal_weights_first5.png', dpi=180)
    plt.close(fig)
    print(f'Wrote figures to {args.outdir}')


if __name__ == '__main__':
    main()
