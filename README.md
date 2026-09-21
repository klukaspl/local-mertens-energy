# Local Mertens energy — numerical code

Python code and CSV data for the numerical section of the paper.

## Files

- `compute_mertens_energy.py` — Möbius sieve and exact local-energy evaluation (no quadrature;
  `floor(aX)` in exact integer arithmetic for rational `a = 2/3`).
- `explicit_formula_compare.py` — trajectory from truncated zeta-zero explicit formulas, correlation/RMS
  metrics, partial sums for the mean. Contains a quadrature self-test of the closed-form energy.
- `random_phase_model.py` — samples the LI random-phase model for the limiting law.
- `brown_spectrum.py` — Brownian-kernel eigenvalues and modal weights `p_j(gamma_k)`.
- `make_figures.py` — regenerates all four figures of the paper (PNG, file names as used in the TeX source).
- `zeros_80.csv`, `zeros_1000.csv` — positive zeta zeros with `zeta'(rho)` (mpmath, 25 digits).
- `mertens_local_energy_1e8.csv` — 30,000 log-spaced samples up to `X = 1e8`.
- derived tables: `mertens_energy_trajectory_comparison.csv`, `explicit_formula_summary.csv`,
  `mean_convergence_zeros.csv`, `brown_spectrum.csv`, `brown_modal_weights.csv`, `random_phase_samples.csv`.

## Reproduce

```bash
python compute_mertens_energy.py --xmax 100000000 --points 30000 --output mertens_local_energy_1e8.csv
python explicit_formula_compare.py          # uses zeros_1000.csv, counts 1,10,80,300,1000
python random_phase_model.py                # 300 zeros, 100k samples, fixed seed
python brown_spectrum.py
python make_figures.py
```

## Changelog

- **Bug fix (explicit_formula_compare.py):** the Hermitian part of the truncated energy was evaluated as
  `conj(A)^T K A` instead of `A^T K conj(A)`, which conjugates the off-diagonal kernel entries. The one-zero
  curve was unaffected; multi-zero curves were wrong. After the fix the 80-zero correlation rises from
  0.853 to 0.962 (RMS 0.00684 -> 0.00352). A direct-quadrature self-test now guards this formula.
- Exact integer `floor(aX)` (the shipped data were verified unaffected).
- Added `zeros_1000.csv`, the random-phase sampler, and the two missing figures to `make_figures.py`.
