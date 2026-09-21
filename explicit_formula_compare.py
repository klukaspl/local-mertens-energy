#!/usr/bin/env python3
"""Compare empirical local Mertens energy with truncated zeta-zero formulas.

With a_g = c_g * exp(i g (Y-L)), c_g = 1/(rho zeta'(rho)), and
g_g(v) = exp(i g v) - exp(-v/2), the truncated process is
    F(v) = 2 Re sum_g a_g g_g(v),
and the truncated energy is
    ||F||^2 = 2 * sum_{j,k} a_j conj(a_k) K_jk + 2 Re sum_{j,k} a_j a_k B_jk,
    K_jk = int_0^L g_j conj(g_k) dv,   B_jk = int_0^L g_j g_k dv.
NOTE: the Hermitian term must be A^T K conj(A).  An earlier version used
conj(A)^T K A, which conjugates the off-diagonal kernel entries and is wrong
as soon as more than one zero is used.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd


def int_exp(alpha: np.ndarray, L: float) -> np.ndarray:
    """int_0^L exp(alpha v) dv, vectorized, safe at alpha = 0."""
    alpha = np.asarray(alpha, dtype=np.complex128)
    out = np.full(alpha.shape, L, dtype=np.complex128)
    nz = np.abs(alpha) > 1e-14
    out[nz] = np.expm1(alpha[nz] * L) / alpha[nz]
    return out


def build_kernels(gamma: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    gj = gamma[:, None]
    gk = gamma[None, :]
    one = np.ones_like(gj * gk)
    e1 = float(-np.expm1(-L))  # int_0^L exp(-v) dv
    K = (int_exp(1j * (gj - gk), L) - int_exp((1j * gj - 0.5) * one, L)
         - int_exp((-1j * gk - 0.5) * one, L) + e1)
    B = (int_exp(1j * (gj + gk), L) - int_exp((1j * gj - 0.5) * one, L)
         - int_exp((1j * gk - 0.5) * one, L) + e1)
    return K, B


def trajectory(Y: np.ndarray, gamma: np.ndarray, coeff: np.ndarray,
               K: np.ndarray, B: np.ndarray, L: float) -> np.ndarray:
    A = coeff[:, None] * np.exp(1j * gamma[:, None] * (Y[None, :] - L))
    herm = np.real(np.sum(A * (K @ np.conj(A)), axis=0))   # A^T K conj(A)
    bil = np.real(np.sum(A * (B @ A), axis=0))             # A^T B A
    return 2 * herm + 2 * bil


def trajectory_quadrature(y: float, gamma: np.ndarray, coeff: np.ndarray,
                          L: float, n: int = 20001) -> float:
    """Slow direct quadrature of ||F||^2; used only as a self-test."""
    v = np.linspace(0.0, L, n)
    a = coeff * np.exp(1j * gamma * (y - L))
    F = 2 * np.real((a[:, None] * (np.exp(1j * gamma[:, None] * v[None, :])
                                   - np.exp(-v / 2)[None, :])).sum(0))
    return float(np.trapezoid(F * F, v))


def A_L(gamma: np.ndarray, L: float) -> np.ndarray:
    z = -0.5 + 1j * gamma
    return L + 1 - math.exp(-L) - 2 * np.real((np.exp(z * L) - 1) / z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--energy', type=Path, default=Path('mertens_local_energy_1e8.csv'))
    ap.add_argument('--zeros', type=Path, default=Path('zeros_1000.csv'))
    ap.add_argument('--counts', default='1,10,80,300,1000')
    ap.add_argument('--L', type=float, default=math.log(1.5))
    ap.add_argument('--max-points', type=int, default=6000)
    ap.add_argument('--output', type=Path, default=Path('mertens_energy_trajectory_comparison.csv'))
    ap.add_argument('--summary', type=Path, default=Path('explicit_formula_summary.csv'))
    args = ap.parse_args()

    emp = pd.read_csv(args.energy)
    zdf = pd.read_csv(args.zeros)
    counts = [int(x) for x in args.counts.split(',') if int(x) <= len(zdf)]
    nmax = max(counts)

    step = max(1, len(emp) // args.max_points)
    sub = emp.iloc[::step].copy()
    Y = sub['Ylog'].to_numpy(float)
    empirical = sub['energy'].to_numpy(float)

    gamma = zdf['gamma'].to_numpy(float)[:nmax]
    zp = (zdf['zeta_prime_real'].to_numpy(float)
          + 1j * zdf['zeta_prime_imag'].to_numpy(float))[:nmax]
    rho = 0.5 + 1j * gamma
    coeff = 1.0 / (rho * zp)
    K, B = build_kernels(gamma, args.L)

    # Self-test of the closed-form energy against direct quadrature.
    ntest = min(nmax, 80)
    for y in (Y[len(Y) // 3], Y[-1]):
        cf = trajectory(np.array([y]), gamma[:ntest], coeff[:ntest],
                        K[:ntest, :ntest], B[:ntest, :ntest], args.L)[0]
        qd = trajectory_quadrature(float(y), gamma[:ntest], coeff[:ntest], args.L)
        assert abs(cf - qd) < 1e-7 * max(1.0, abs(qd)), (cf, qd)

    out = pd.DataFrame({'Y': Y, 'X': sub['X'].to_numpy(int), 'empirical': empirical})
    metrics = []
    for N in counts:
        pred = trajectory(Y, gamma[:N], coeff[:N], K[:N, :N], B[:N, :N], args.L)
        out[f'pred_{N}zero'] = pred
        metrics.append({
            'zeros': N,
            'gamma_N': float(gamma[N - 1]),
            'corr': float(np.corrcoef(empirical, pred)[0, 1]),
            'RMS': float(np.sqrt(np.mean((empirical - pred) ** 2))),
            'MAE': float(np.mean(np.abs(empirical - pred))),
            'bias_emp_minus_pred': float(np.mean(empirical - pred)),
            'pred_mean': float(np.mean(pred)),
            'emp_mean': float(np.mean(empirical)),
        })

    Avec = A_L(gamma, args.L)
    den = np.abs(rho * zp) ** 2
    conv = pd.DataFrame({
        'n': np.arange(1, nmax + 1),
        'gamma': gamma,
        'beta_partial': np.cumsum(2 / den),
        'energy_mean_partial': np.cumsum(2 * Avec / den),
        'Jminus1_over_gamma': np.cumsum(1 / np.abs(zp) ** 2) / gamma,
    })

    out.to_csv(args.output, index=False)
    pd.DataFrame(metrics).to_csv(args.summary, index=False)
    conv.to_csv(args.summary.with_name('mean_convergence_zeros.csv'), index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f'Wrote {args.output}, {args.summary}, mean_convergence_zeros.csv')


if __name__ == '__main__':
    main()
