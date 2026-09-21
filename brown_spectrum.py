#!/usr/bin/env python3
"""Brownian-kernel spectrum and zeta-zero modal weights p_j(gamma_k)."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq


def brown_roots(L: float, J: int) -> np.ndarray:
    roots = []
    for j in range(1, J + 1):
        lo = ((j - 0.5) * math.pi + 1e-10) / L
        hi = (j * math.pi - 1e-10) / L
        roots.append(brentq(lambda nu: math.tan(nu * L) + 2 * nu, lo, hi))
    return np.asarray(roots)


def h_j(nu: float, Nnorm: float, gamma: np.ndarray, L: float) -> np.ndarray:
    z = 1j * gamma
    # Integral_0^L exp(-i gamma u) sin(nu u) du / N_j.
    # Closed form using exponentials.
    def I(alpha):
        return np.expm1(alpha * L) / alpha
    return (I(1j * (nu - gamma)) - I(-1j * (nu + gamma))) / (2j * Nnorm)


def A_L(gamma: np.ndarray, L: float) -> np.ndarray:
    z = -0.5 + 1j * gamma
    return L + 1 - math.exp(-L) - 2 * np.real((np.exp(z * L) - 1) / z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--zeros', type=Path, default=Path('zeros_80.csv'))
    ap.add_argument('--modes', type=int, default=30)
    ap.add_argument('--zero-count', type=int, default=5)
    ap.add_argument('--L', type=float, default=math.log(1.5))
    ap.add_argument('--spectrum-out', type=Path, default=Path('brown_spectrum.csv'))
    ap.add_argument('--weights-out', type=Path, default=Path('brown_modal_weights.csv'))
    args = ap.parse_args()

    zdf = pd.read_csv(args.zeros)
    gamma = zdf['gamma'].to_numpy(float)[:args.zero_count]
    rho_abs2 = 0.25 + gamma * gamma

    nu = brown_roots(args.L, args.modes)
    lam = 1.0 / (nu * nu + 0.25)
    N2 = args.L / 2.0 + lam / 4.0
    Nnorm = np.sqrt(N2)
    pinf = lam * lam * nu * nu / (N2 * (args.L + 1 - math.exp(-args.L)))

    sdf = pd.DataFrame({
        'j': np.arange(1, args.modes + 1),
        'nu_j': nu,
        'lambda_j': lam,
        'N_j_sq': N2,
        'p_j_infty': pinf,
        'lambda_partial_sum': np.cumsum(lam),
        'p_infty_partial_sum': np.cumsum(pinf),
    })
    sdf.to_csv(args.spectrum_out, index=False)

    Avec = A_L(gamma, args.L)
    rows = []
    for j in range(args.modes):
        h = h_j(nu[j], Nnorm[j], gamma, args.L)
        p = lam[j] * np.abs(h) ** 2 * rho_abs2 / Avec
        for k, val in enumerate(p, 1):
            rows.append((j + 1, k, gamma[k - 1], float(val)))
    wdf = pd.DataFrame(rows, columns=['j', 'k', 'gamma_k', 'p_j_gamma_k'])
    wdf.to_csv(args.weights_out, index=False)

    print('Trace check:')
    print('  finite sum lambda_j =', lam.sum())
    print('  exact trace          =', args.L - 1 + math.exp(-args.L))
    print('Fixed-mode high-zero mass check:')
    print('  finite sum p_j(infty)=', pinf.sum())
    print('  exact total          =', (1 - math.exp(-args.L)) / (args.L + 1 - math.exp(-args.L)))
    print(f'Wrote {args.spectrum_out} and {args.weights_out}')


if __name__ == '__main__':
    main()
