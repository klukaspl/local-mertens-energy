#!/usr/bin/env python3
"""Sample the LI random-phase model E_{L,T} = ||2 Re sum c_g g_g U_g||^2."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from explicit_formula_compare import build_kernels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--zeros', type=Path, default=Path('zeros_1000.csv'))
    ap.add_argument('--count', type=int, default=300)
    ap.add_argument('--samples', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--L', type=float, default=math.log(1.5))
    ap.add_argument('--output', type=Path, default=Path('random_phase_samples.csv'))
    args = ap.parse_args()
    z = pd.read_csv(args.zeros).iloc[:args.count]
    g = z['gamma'].to_numpy(float)
    zp = z['zeta_prime_real'].to_numpy(float) + 1j * z['zeta_prime_imag'].to_numpy(float)
    c = 1.0 / ((0.5 + 1j * g) * zp)
    K, B = build_kernels(g, args.L)
    rng = np.random.default_rng(args.seed)
    out = []
    for _ in range(0, args.samples, 5000):
        U = np.exp(2j * math.pi * rng.random((args.count, 5000)))
        A = c[:, None] * U
        out.append(2 * np.real(np.sum(A * (K @ np.conj(A)), axis=0))
                   + 2 * np.real(np.sum(A * (B @ A), axis=0)))
    e = np.concatenate(out)[:args.samples]
    pd.DataFrame({'energy': e}).to_csv(args.output, index=False)
    print(f'mean={e.mean():.6f} median={np.median(e):.6f} std={e.std():.6f} '
          f'min={e.min():.6f} max={e.max():.6f}')


if __name__ == '__main__':
    main()
