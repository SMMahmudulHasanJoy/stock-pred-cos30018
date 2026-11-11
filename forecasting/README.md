# Forecasting: KAN, LSTM and Naive persistence

Sections 3 and 5.1 of the paper. Given the last 20 trading days of Open, High, Low, Close and Volume, each model predicts the next H closing prices in one shot (H = 1, 2, 5, 10, 100, 200).

## Models

| Model | Description |
|---|---|
| Naive persistence | Repeats the last observed close for all H days. No training. |
| LSTM | Stacked LSTM layers with dropout 0.2 and a Dense(H) output; RMSprop, 25 epochs, batch size 30. Nine architectures are screened at the 1-day horizon; the best (2 layers, 20 units, tanh output) is used at every horizon. |
| KAN | PyKAN on the flattened 20 x 5 window, width [100, N_h, H] with N_h = 3 x floor(N_train / 50), grid 3, cubic splines, 10 full-batch L-BFGS steps. Six configurations are screened at the 1-day horizon. |

## Evaluation protocol

- **Walk-forward validation.** Five expanding folds; fold i trains on all windows before its test block. The pooled test blocks cover the last 60 % of the period.
- **Leak-free scaling.** Min-Max scaling is fitted on the training rows of each fold only.
- **Seeds.** LSTM and KAN are trained from seeds 0, 100 and 200. Tables give the mean and sample standard deviation of the pooled test RMSE over the three seeds.
- **Market regimes.** Each test window is labelled Normal, Volatile or Trending from its 20 input days, using thresholds from the training data.
- **Two protocols.** `standard` is the protocol of the paper's tables. `purged` additionally removes the last H-1 training windows of every fold, whose targets lie inside the test period. The two give the same results at 1 and 2 days and very different results at 100 and 200 days (see below).

## Code

| Script | Purpose |
|---|---|
| `forecast_lib.py` | Data loading, windows, folds, scaling, purging, regimes, Naive forecast, RMSE |
| `run_forecasts.py` | Trains the models and stores test predictions; resumable |
| `make_tables.py` | Builds every table from the stored predictions |
| `make_figures.py` | Builds every figure from the tables and predictions |

```bash
python run_forecasts.py --status                    # CBA.AX, standard protocol, main experiment
python run_forecasts.py --experiment lstm_grid      # Table 9
python run_forecasts.py --experiment kan_grid       # Table 11
python run_forecasts.py --ticker TSLA --purge       # another asset, purged protocol
python run_forecasts.py --out my_predictions        # re-run from scratch into a new folder
python make_tables.py
python make_figures.py
```

## Results

### `results/predictions/<protocol>/<ticker>/<model>/`

One `.npz` file per model, horizon, seed and fold, named `<model>__<configuration>__h<horizon>__s<seed>__f<fold>.npz`. Each file holds `prediction` and `truth` (test windows x H, in reporting scale), `labels` (regime of each window), `dates` (last observed day of each window) and `meta` (training size, number of parameters, training seconds, training RMSE, scaling limits).

| Folder | Content |
|---|---|
| `naive/` | Naive persistence |
| `lstm/` | LSTM; for CBA.AX under the standard protocol this includes the nine screened architectures |
| `kan/` | KAN in the configuration of the paper |
| `kan_grid/` | The six screened KAN configurations (CBA.AX, standard protocol) |

### `results/tables/`

| File | Table in the paper |
|---|---|
| `table09_lstm_configurations.csv` | Table 9 |
| `table10_lstm_by_horizon.csv` | Table 10 |
| `table11_kan_configurations.csv` | Table 11 |
| `table12_regimes_by_horizon.csv` | Table 12 |
| `table13_training_time.csv` | Table 13 |
| `kan_long_horizon_stability.csv` | KAN at 100 and 200 days with and without the first fold (Section 5.1.3) |
| `standard_vs_purged.csv` | All three assets under both protocols |
| `all_runs.csv` | One row per stored run |
| `forecasting_tables.xlsx` | All of the above, one sheet each |

Tables 10, 12 and 13 also contain the 5- and 10-day horizons, which the paper does not show; the column `in_paper` marks the rows that it does.

### `results/figures/`

| Sub-folder | Content |
|---|---|
| `configuration_screening/` | Train and test RMSE of the LSTM and KAN configurations (Figures 3 and 4) |
| `one_day_forecast/` | One-day-ahead forecasts over the last fold, with the plotted series as CSV (Figure 5) |
| `market_regimes/` | Table 12 as bar charts |
| `horizons/` | RMSE against horizon for each asset under both protocols, and the ratio to the Naive forecast |

## Notes on the results

**Long horizons and purging.** Under the standard protocol the last H-1 training windows have targets inside the test period. At H = 100 and 200 this gives the trained models information about the test period and makes them look close to the Naive forecast. With those windows removed, the test RMSE on CBA.AX is:

| Horizon | Naive | LSTM standard | LSTM purged | KAN standard | KAN purged |
|---|---|---|---|---|---|
| 1 | 0.0201 | 0.0586 | 0.0586 | 0.0338 | 0.0331 |
| 2 | 0.0252 | 0.0579 | 0.0598 | 0.0475 | 0.0444 |
| 100 | 0.1109 | 0.1079 | 0.2490 | 0.2418 | 0.4080 |
| 200 | 0.1323 | 0.1375 | 0.3628 | 0.2764 | 0.5935 |

The 1- and 2-day results are the same under both protocols. The 100- and 200-day results should be read from the purged columns.

**Reproducibility.** The Naive and LSTM results reproduce exactly from the code. The KAN results do not reproduce file by file: PyKAN's fit is not deterministic from one Python process to the next, even with every seed fixed, so a re-run gives slightly different predictions per fold. The pooled RMSE values agree within the reported standard deviations.

**Training times** were recorded on the machine that produced the stored predictions (CPU only) and are meaningful relative to one another, not as absolute values.
