#!/usr/bin/env python3
"""Compute the local Mertens energy used in the paper.

For integer X and a=exp(-L),
    M_L(X) = int_{aX}^X |M(y)-M(aX)|^2 / y^2 dy.
Because M(y) is piecewise constant, this is evaluated exactly (no numerical
quadrature) as a finite weighted sum over integer intervals.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd


def mobius_sieve(n: int) -> np.ndarray:
    """Return mu[0..n] as int8 using a prime sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
    mu[0] = 0
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, n + 1):
        if is_prime[p]:
            mu[p::p] *= -1
            pp = p * p
            if pp <= n:
                mu[pp::pp] = 0
                is_prime[pp::p] = False
    return mu


def local_energy_exact(X: int, M: np.ndarray, a_num: int, a_den: int) -> tuple[float, int]:
    """Exact local energy for integer X and rational a = a_num/a_den.

    floor(a*X) is computed in exact integer arithmetic: with a float `a`
    the product a*X can land just below an integer when a*X is itself an
    integer (e.g. X divisible by 3 for a = 2/3), shifting the anchor M(aX).
    """
    lo = (a_num * X) // a_den
    m0 = int(M[lo])
    # The first nonzero interval is [lo+1, lo+2); the upper endpoint X is exact.
    n = np.arange(lo + 1, X, dtype=np.int64)
    if n.size == 0:
        return 0.0, m0
    d = M[n].astype(np.float64) - m0
    weights = 1.0 / n - 1.0 / (n + 1.0)
    return float(np.dot(d * d, weights)), m0


def log_uniform_integer_grid(xmin: int, xmax: int, points: int) -> np.ndarray:
    ys = np.linspace(math.log(xmin), math.log(xmax), points)
    xs = np.rint(np.exp(ys)).astype(np.int64)
    xs = np.clip(xs, xmin, xmax)
    # Preserve order and uniqueness without losing the log-grid character.
    return np.unique(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--xmin', type=int, default=10_000)
    ap.add_argument('--xmax', type=int, default=100_000_000)
    ap.add_argument('--points', type=int, default=30_000)
    ap.add_argument('--a-num', type=int, default=2, help='a = a_num/a_den = exp(-L)')
    ap.add_argument('--a-den', type=int, default=3)
    ap.add_argument('--output', type=Path, default=Path('mertens_local_energy.csv'))
    args = ap.parse_args()

    if args.xmin < 2 or args.xmax < args.xmin:
        raise SystemExit('Require 2 <= xmin <= xmax.')

    print(f'Computing mu(n) up to {args.xmax:,} ...')
    mu = mobius_sieve(args.xmax)
    M = np.cumsum(mu, dtype=np.int64)
    xs = log_uniform_integer_grid(args.xmin, args.xmax, args.points)

    rows = []
    for i, X in enumerate(xs, 1):
        e, m0 = local_energy_exact(int(X), M, args.a_num, args.a_den)
        rows.append((int(X), math.log(int(X)), e, m0))
        if i % 2000 == 0:
            print(f'  {i}/{len(xs)}')

    df = pd.DataFrame(rows, columns=['X', 'Ylog', 'energy', 'Mlo'])
    df.to_csv(args.output, index=False)
    print(f'Wrote {args.output} ({len(df)} rows)')


if __name__ == '__main__':
    main()
