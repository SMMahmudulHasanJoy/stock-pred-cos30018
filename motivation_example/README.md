# Motivation example: KAN fits of two piecewise-linear functions

Section 4.4 of the paper. Two linear splines on [0, 10], sampled every 0.01 (1001 points):

S(t) = Σ a_j · max(0, t − θ_j), with slopes a = (2, −1, 4, −3, −1)

| Function | Knots θ | |
|---|---|---|
| S1 | 0, 2, 4, 6, 8 | evenly spaced |
| S2 | 0, 3, 5, 6, 7 | not evenly spaced |

Both functions have the same slopes; only the positions of the knots differ. A KAN edge is a spline on an evenly spaced grid of G sub-intervals, so the example shows how each function is approximated as G changes.

| Experiment | Network | Table |
|---|---|---|
| A | No hidden layer, linear splines (k = 1), G = 5, 10, 20, 50, 100 | Table 5 |
| B | No hidden layer, cubic splines (k = 3), same G | Table 6 |
| C | One hidden layer with 2 or 3 nodes, linear splines, G = 5 and 10 | Table 7 |

All fits use PyKAN with 200 L-BFGS steps, a fixed grid on [0, 10] and seed 0.

## Code

| Script | Purpose |
|---|---|
| `run_motivation_example.py` | Defines S1 and S2 and runs the 28 fits; resumable |
| `make_tables_figures.py` | Tables 5 to 7 and all figures |

```bash
python run_motivation_example.py --status
python make_tables_figures.py
```

## Results

| Location | Content |
|---|---|
| `results/raw/` | One JSON file per fit with errors, parameter count and the fitted curve |
| `results/tables/` | `table05_…`, `table06_…`, `table07_…` as CSV; `all_fits.csv`; `S1_S2_values.csv`; `motivation_example_tables.xlsx` |
| `results/figures/target_functions/` | S1 and S2 (Figure 2) |
| `results/figures/error_vs_grid/` | Worst error against G for experiments A and B |
| `results/figures/fitted_curves/experiment_A/`, `…_B/`, `…_C/` | Target, fit and error of every network |

"Worst" in the tables is the largest absolute deviation over the 1001 points. The number of parameters of a network without a hidden layer is G + k + 6; with one hidden layer of h nodes it is 2h · (G + k + 6).

## Reproducibility

The fits without a hidden layer (A and B) are least-squares problems with one solution and re-run to the same four decimals. The hidden-layer fits (C) are not convex. Most re-run identically, but the fit of S2 with 2 hidden nodes and G = 10 ends in a different local minimum from one Python process to the next (worst error between about 0.007 and 0.04), even with the seed fixed. The stored files are the fits reported in the paper.
