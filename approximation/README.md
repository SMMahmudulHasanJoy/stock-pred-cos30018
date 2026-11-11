# Approximation: KAN, MLP and Fourier MLP at equal numbers of parameters

Sections 4.1 to 4.3 and 5.2 of the paper. Each model receives a whole series of scaled closing prices, indexed by the rescaled time t in [0, 1], and is fitted to it by least squares. There is no train/test split: the question is how small the fitting error can be made with a given number of trainable parameters N.

## Design

Three assets (CBA.AX, TSLA, LLY), five target sizes N* = 15, 30, 60, 120, 240, four models and three seeds (0, 1, 2): 180 fits.

| Key | Model | Setting per budget | Parameters N | Training |
|---|---|---|---|---|
| `kan11` | KAN without a hidden layer, width [1,1] | grid size G = 6, 21, 51, 111, 231 | G + 9 | 80 L-BFGS steps |
| `kan1h1` | KAN with one hidden layer, width [1,h,1], G = 3 | h = 1, 1, 2, 5, 10 | 24 h | 80 L-BFGS steps |
| `mlp` | MLP of t, two tanh hidden layers | width w = 2, 4, 6, 9, 14 | w² + 4w + 1 | 20,000 Adam steps |
| `fmlp` | Fourier MLP: the same network fed sin and cos of t at m fixed random frequencies | (m, w) = (2,2), (2,3), (4,4), (6,6), (8,9) | w² + (2m+3)w + 1 | 20,000 Adam steps |

N is the number of trainable parameters reported by the software. A KAN edge has G + 3 cubic-spline coefficients plus 6 further trainable numbers that PyKAN keeps on every edge. The smallest one-hidden-layer KAN has 24 parameters, so the budgets 15 and 30 share the same fit. The actual N of every fit is recorded and used in all tables and figures.

## Code

| Script | Purpose |
|---|---|
| `run_approximation.py` | The four models and the 180 fits; resumable |
| `make_tables.py` | Summary tables, fitted curves as CSV, Excel workbook of actual values |
| `differences_ttests.py` | Pairwise differences between models and paired t-tests, with Excel workbooks |
| `make_figures.py` | All figures |

```bash
python run_approximation.py --status
python run_approximation.py --out my_fits     # re-run all 180 fits into a new folder (about 20 minutes of fitting time on CPU)
python make_tables.py
python differences_ttests.py
python make_figures.py
```

The fits reproduce exactly on CPU: re-running a fit gives the stored curve to the last digit.

## Results

### `results/raw/`

| Folder | Content |
|---|---|
| `fits_json/` | One file per fit, `<ticker>__<model>__b<budget>__s<seed>.json`: parameter count, RMSE, maximum error, seconds, setting and the fitted curve |
| `fitted_curves_csv/` | The same fitted curves as CSV: `t, close_scaled, fit, error` |

### `results/tables/`

| File | Content |
|---|---|
| `all_fits.csv` | One row per fit (180 rows) |
| `summary_by_model.csv` | Mean and standard deviation over the three seeds (60 rows) |
| `table04_settings.csv` | Setting and parameter count per budget (Table 4) |
| `table14_mlp_fourier_mlp.csv` | MLP and Fourier MLP (Table 14) |
| `table15_kan.csv` | KAN without and with one hidden layer (Table 15) |

### `results/differences/` and `results/ttests/`

Differences are always A − B, so a negative difference in error means model A is better.

| File | Content |
|---|---|
| `differences/differences_per_seed.csv` | Seed-matched differences for every comparison, asset and budget |
| `differences/differences_summary.csv` | Mean difference per comparison, asset and budget, with the values of both models (60 rows) |
| `differences/table16_kan_minus_mlp.csv` | KAN [1,1] against MLP and Fourier MLP (Table 16) |
| `ttests/paired_ttests.csv` | Paired t-test, 95 % interval, effect size, Wilcoxon signed-rank test and sign test, for all 15 cells and per asset |

The paired t-test is a one-sample t-test on the 15 differences (three assets × five budgets, each the mean over three seeds). The 15 cases are not independent, because the five budgets of an asset use the same series, so the p-values describe how consistent a difference is and are not exact significance levels.

| Comparison (RMSE) | Mean difference | t (14 d.f.) | p | Wilcoxon p |
|---|---|---|---|---|
| KAN [1,1] − MLP of t | −0.0197 | −6.43 | 0.00002 | 0.0002 |
| KAN [1,1] − Fourier MLP | −0.0050 | −2.79 | 0.015 | 0.030 |
| KAN [1,h,1] − KAN [1,1] | +0.0188 | 6.04 | 0.00003 | 0.0004 |
| Fourier MLP − MLP of t | −0.0146 | −3.16 | 0.007 | 0.010 |

### `results/excel/`

| File | Content |
|---|---|
| `approximation_actual_values.xlsx` | Actual values: one row per asset and budget with all four models side by side; summary by model; all 180 fits; settings |
| `differences_and_ttests.xlsx` | Differences per cell, paired t-tests, differences per seed |
| `TableDiff.xlsx` | Differences in the 60-row layout used for the paper |
| `TableActual.xlsx` | Actual values of both models in the same 60-row layout |

### `results/figures/`

| Sub-folder | Content |
|---|---|
| `error_vs_parameters/` | RMSE and maximum error against N, one panel per asset |
| `fitted_curves/<ticker>/` | For each budget: target, fitted curve and error of the four models (best of three seeds) |
| `training_time/` | Fitting time against N |
| `differences/` | The 15 differences of each comparison with mean and 95 % interval |

## Notes

**Fitting times** compare 80 L-BFGS steps with 20,000 Adam steps, so they reflect these settings and not an intrinsic cost of either architecture. They were recorded on the machine that produced the stored fits (CPU only).

**Training length.** The KAN without a hidden layer has converged well before 80 steps. The KAN with a hidden layer and the two MLPs still improve with longer training (for example, KAN [1,h,1] on CBA.AX at N = 120 goes from an RMSE of 0.045 to about 0.037 with 320 to 640 steps, and the MLP at N = 253 from 0.039 to 0.036 with 40,000 steps). These changes do not alter the ordering of the models.
