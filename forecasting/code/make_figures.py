"""
make_figures.py
===============
Draws every forecasting figure from the tables and stored predictions.
Run make_tables.py first. Nothing is trained here.

Output (forecasting/results/figures/)
-------------------------------------
    configuration_screening/
        lstm_configurations.png      train/test RMSE of nine LSTMs   (Fig. 3)
        kan_configurations.png       train/test RMSE of six KANs     (Fig. 4)
    one_day_forecast/
        forecast_last_fold.png       actual vs LSTM, KAN, Naive      (Fig. 5)
        forecast_last_fold.csv       the plotted series
    market_regimes/
        rmse_by_regime.png           Table 12 as bars, one panel per horizon
    horizons/
        rmse_vs_horizon_<ticker>.png RMSE against horizon, standard and purged
        ratio_to_naive.png           model RMSE / Naive RMSE, purged protocol

    python make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecast_lib as fl
from make_tables import PAPER_HORIZONS, load_pooled
from run_forecasts import BEST_LSTM

COLOURS = {"lstm": "#2f6f9f", "kan": "#c46a3f", "naive": "#444444"}


def read_table(name):
    return pd.read_csv(os.path.join(fl.TABLES_DIR, name + ".csv"))


def save(figure, subfolder, name):
    """Save a figure as PNG in its sub-folder of results/figures/."""
    folder = os.path.join(fl.FIGURES_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)
    figure.tight_layout()
    figure.savefig(os.path.join(folder, name), dpi=200)
    plt.close(figure)
    print("wrote", os.path.join(subfolder, name))


# --------------------------------------------------------------------------
def screening_bars(table_name, title, file_name):
    """Train and test RMSE (mean, with +/- one sd whiskers) per configuration,
    with the Naive test RMSE as a dashed reference line."""
    table = read_table(table_name)
    models = table[table.configuration != "Naive persistence"]
    naive = float(table[table.configuration == "Naive persistence"].test_rmse_mean.iloc[0])
    x = np.arange(len(models))
    figure, axis = plt.subplots(figsize=(8, 3.6))
    axis.bar(x - 0.2, models.train_rmse_mean, 0.4, yerr=models.train_rmse_sd, capsize=3,
             color="#7f8ccf", label="Train RMSE")
    axis.bar(x + 0.2, models.test_rmse_mean, 0.4, yerr=models.test_rmse_sd, capsize=3,
             color="#f0a050", label="Test RMSE")
    axis.axhline(naive, color=COLOURS["naive"], linestyle="--", linewidth=1, label="Naive persistence (test)")
    axis.set_xticks(x)
    axis.set_xticklabels(models.configuration, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("RMSE (scaled)")
    axis.set_title(title, fontsize=10)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8, frameon=False)
    save(figure, "configuration_screening", file_name)


def one_day_forecast():
    """One-day-ahead forecasts over the last walk-forward fold, seed 0."""
    series = {}
    for model, folder, stem in (("lstm", "lstm", "lstmcfg__" + BEST_LSTM), ("kan", "kan", "kan__-"),
                                ("naive", "naive", "naive__-")):
        prediction, truth, _, _ = load_pooled("standard", "CBA.AX", folder, stem, 1, 0, folds=[4])
        series[model] = prediction[:, 0]
    with np.load(os.path.join(fl.PREDICTIONS_DIR, "standard", "CBA.AX", "naive", "naive__-__h1__s0__f4.npz")) as data:
        origin = pd.to_datetime(data["dates"])        # last observed day of each window
    actual = truth[:, 0]

    frame = pd.DataFrame({"forecast_origin": origin, "actual": actual, **series})
    folder = os.path.join(fl.FIGURES_DIR, "one_day_forecast")
    os.makedirs(folder, exist_ok=True)
    frame.to_csv(os.path.join(folder, "forecast_last_fold.csv"), index=False, float_format="%.6f")

    figure, axis = plt.subplots(figsize=(9, 3.6))
    axis.plot(origin, actual, color="black", linewidth=1.3, label="Actual")
    for model, label in (("lstm", "LSTM"), ("kan", "KAN"), ("naive", "Naive")):
        axis.plot(origin, series[model], color=COLOURS[model], linewidth=1,
                  linestyle="--" if model == "naive" else "-",
                  label="%s (RMSE %.3f)" % (label, fl.rmse(series[model], actual)))
    axis.set_xlabel("forecast origin (last observed day)")
    axis.set_ylabel("next-day close (scaled)")
    axis.set_title("One-day-ahead forecasts, last walk-forward fold, seed 0 (CBA.AX)", fontsize=10)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, frameon=False, ncol=4)
    save(figure, "one_day_forecast", "forecast_last_fold.png")


def regime_bars():
    """Table 12 as grouped bars: one panel per horizon shown in the paper."""
    table = read_table("table12_regimes_by_horizon")
    figure, axes = plt.subplots(1, len(PAPER_HORIZONS), figsize=(13, 3.4))
    for axis, horizon in zip(axes, PAPER_HORIZONS):
        rows = table[table.horizon_days == horizon]
        x = np.arange(len(rows))
        axis.bar(x - 0.27, rows.lstm_rmse_mean, 0.27, yerr=rows.lstm_rmse_sd, capsize=2, color=COLOURS["lstm"], label="LSTM")
        axis.bar(x, rows.kan_rmse_mean, 0.27, yerr=rows.kan_rmse_sd, capsize=2, color=COLOURS["kan"], label="KAN")
        axis.bar(x + 0.27, rows.naive_rmse, 0.27, color=COLOURS["naive"], label="Naive")
        axis.set_xticks(x)
        axis.set_xticklabels(["%s\n(n=%d)" % (r, n) for r, n in zip(rows.regime, rows.windows)], fontsize=7)
        axis.set_title("%d-day horizon" % horizon, fontsize=10)
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("RMSE (scaled)")
    axes[0].legend(fontsize=8, frameon=False)
    save(figure, "market_regimes", "rmse_by_regime.png")


def horizon_curves():
    """RMSE against forecast horizon for each asset, standard and purged."""
    table = read_table("standard_vs_purged")
    for ticker in fl.TICKERS:
        rows = table[table.ticker == ticker]
        figure, axis = plt.subplots(figsize=(6, 3.8))
        axis.plot(rows.horizon_days, rows.naive_rmse, "o-", color=COLOURS["naive"], label="Naive persistence")
        for model, label in (("lstm", "LSTM"), ("kan", "KAN")):
            axis.errorbar(rows.horizon_days, rows[model + "_standard_mean"], yerr=rows[model + "_standard_sd"],
                          marker="s", linestyle=":", capsize=2, color=COLOURS[model], label=label + ", standard")
            axis.errorbar(rows.horizon_days, rows[model + "_purged_mean"], yerr=rows[model + "_purged_sd"],
                          marker="o", linestyle="-", capsize=2, color=COLOURS[model], label=label + ", purged")
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_xlabel("forecast horizon (days)"); axis.set_ylabel("test RMSE (scaled)")
        axis.set_title(ticker, fontsize=10)
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7, frameon=False)
        save(figure, "horizons", "rmse_vs_horizon_%s.png" % ticker)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for axis, (model, label) in zip(axes, (("lstm", "LSTM"), ("kan", "KAN"))):
        for ticker, marker in zip(fl.TICKERS, "osD"):
            rows = table[table.ticker == ticker]
            axis.plot(rows.horizon_days, rows[model + "_purged_over_naive"], marker=marker, label=ticker)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("forecast horizon (days)")
        axis.set_title("%s RMSE / Naive RMSE (purged); above 1 = Naive is better" % label, fontsize=9)
        axis.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("ratio")
    axes[0].legend(fontsize=8, frameon=False)
    save(figure, "horizons", "ratio_to_naive.png")


if __name__ == "__main__":
    screening_bars("table09_lstm_configurations",
                   "LSTM configurations, 1-day horizon (mean +/- sd over three seeds)", "lstm_configurations.png")
    screening_bars("table11_kan_configurations",
                   "KAN configurations, 1-day horizon (mean +/- sd over three seeds)", "kan_configurations.png")
    one_day_forecast()
    regime_bars()
    horizon_curves()
