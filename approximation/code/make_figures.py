"""
make_figures.py
===============
Draws the figures of the approximation study from the stored fits and tables.
Run make_tables.py and differences_ttests.py first. Nothing is fitted here.

Output (approximation/results/figures/)
---------------------------------------
    error_vs_parameters/
        rmse_vs_parameters.png        RMSE against N, log-log, one panel per asset
        max_error_vs_parameters.png   the same for the maximum absolute error
    fitted_curves/<ticker>/
        budget_<N*>.png               target and fitted curve of the four models
                                      (best of the three seeds), with the error
    training_time/
        seconds_vs_parameters.png     fitting time against N
    differences/
        differences_by_cell.png       the 15 cell differences of every
                                      comparison with mean and 95 % interval

    python make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_approximation as study
from make_tables import load_fits

FIGURES_DIR = os.path.join(study.RESULTS_DIR, "figures")
COLOURS = {"kan11": "#c46a3f", "kan1h1": "#e0a070", "mlp": "#2f6f9f", "fmlp": "#5fb0e0"}
TICKER_COLOURS = {"CBA.AX": "#2f6f9f", "TSLA": "#c46a3f", "LLY": "#4a9a4a"}


def save(figure, subfolder, name):
    folder = os.path.join(FIGURES_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)
    figure.tight_layout()
    figure.savefig(os.path.join(folder, name), dpi=180)
    plt.close(figure)


# --------------------------------------------------------------------------
def metric_vs_parameters(summary, column, sd_column, ylabel, subfolder, name, log_y=True):
    """One panel per asset: mean (+/- sd over seeds) against the actual N."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, ticker in zip(axes, study.TICKERS):
        for model in study.MODELS:
            rows = summary[(summary.ticker == ticker) & (summary.model == model)]
            axis.errorbar(rows.parameters, rows[column], yerr=rows[sd_column], marker="o", markersize=4,
                          capsize=2, color=COLOURS[model], label=study.MODEL_LABELS[model])
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("number of parameters N")
        axis.set_ylabel(ylabel)
        axis.set_title(ticker)
        axis.grid(alpha=0.3, which="both", linewidth=0.4)
    axes[0].legend(fontsize=7, frameon=False)
    save(figure, subfolder, name)


def fitted_curves(fits, records):
    """For each asset and budget: the four fitted curves over the target
    (left) and their errors (right). The best of the three seeds is shown."""
    by_fit = {(r["ticker"], r["budget"], r["model"], r["seed"]): r for r in records}
    for ticker in study.TICKERS:
        t, y = study.load_series(ticker)
        for budget in study.BUDGETS:
            figure, axes = plt.subplots(len(study.MODELS), 2, figsize=(12, 9), sharex=True)
            for row, model in enumerate(study.MODELS):
                cell = fits[(fits.ticker == ticker) & (fits.budget == budget) & (fits.model == model)]
                best = cell.loc[cell.rmse.idxmin()]
                fit = np.array(by_fit[(ticker, budget, model, int(best.seed))]["pred"])
                left, right = axes[row]
                left.plot(t, y, color="#999999", linewidth=0.8, label="scaled close")
                left.plot(t, fit, color=COLOURS[model], linewidth=1.2,
                          label="%s, N = %d, RMSE %.4f" % (study.MODEL_LABELS[model], best.parameters, best.rmse))
                left.legend(fontsize=7, frameon=False, loc="upper left")
                left.grid(alpha=0.3, linewidth=0.4)
                right.fill_between(t, fit - y, 0, color=COLOURS[model], alpha=0.6, linewidth=0)
                right.axhline(0, color="black", linewidth=0.6)
                right.set_ylabel("fit - close", fontsize=8)
                right.set_title("max |error| %.4f" % best.max_error, fontsize=8)
                right.grid(alpha=0.3, linewidth=0.4)
            axes[-1, 0].set_xlabel("t  (0 = first day, 1 = last day)")
            axes[-1, 1].set_xlabel("t")
            figure.suptitle("%s, budget N* = %d" % (ticker, budget), fontsize=11)
            save(figure, os.path.join("fitted_curves", ticker), "budget_%03d.png" % budget)


def differences_by_cell():
    """Every comparison: the 15 cell differences (dots, coloured by asset),
    their mean and the 95 % confidence interval of the mean."""
    summary = pd.read_csv(os.path.join(study.RESULTS_DIR, "differences", "differences_summary.csv"))
    tests = pd.read_csv(os.path.join(study.RESULTS_DIR, "ttests", "paired_ttests.csv"))
    tests = tests[tests.scope == "all 15 cells"]
    comparisons = list(dict.fromkeys(summary.comparison))
    panels = [("d_rmse_mean", "RMSE"), ("d_max_error_mean", "max |error|")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for axis, (column, metric) in zip(axes, panels):
        for i, comparison in enumerate(comparisons):
            cells = summary[summary.comparison == comparison]
            for ticker in study.TICKERS:
                values = cells[cells.ticker == ticker][column]
                axis.scatter(values, [i + 0.22] * len(values), s=14, color=TICKER_COLOURS[ticker],
                             label=ticker if i == 0 else None)
            test = tests[(tests.comparison == comparison) & (tests.metric == metric)].iloc[0]
            axis.errorbar(test.mean_difference, i, xerr=test.mean_difference - test.ci95_low,
                          fmt="D", color="black", capsize=4)
            axis.text(0.99, i - 0.3, "t = %.2f, p = %.2g" % (test.t_statistic, test.p_value),
                      transform=axis.get_yaxis_transform(), ha="right", fontsize=7)
        axis.axvline(0, color="black", linestyle=":")
        axis.set_yticks(range(len(comparisons)))
        axis.set_yticklabels(comparisons, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("difference A - B in %s  (below 0: A is better)" % metric, fontsize=8)
        axis.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    save(figure, "differences", "differences_by_cell.png")


if __name__ == "__main__":
    fits, records = load_fits()
    summary = pd.read_csv(os.path.join(study.RESULTS_DIR, "tables", "summary_by_model.csv"))
    metric_vs_parameters(summary, "rmse_mean", "rmse_sd", "RMSE (scaled units)",
                         "error_vs_parameters", "rmse_vs_parameters.png")
    metric_vs_parameters(summary, "max_error_mean", "max_error_sd", "max |error| (scaled units)",
                         "error_vs_parameters", "max_error_vs_parameters.png")
    metric_vs_parameters(summary, "seconds_mean", "seconds_sd", "fitting time (s)",
                         "training_time", "seconds_vs_parameters.png")
    fitted_curves(fits, records)
    differences_by_cell()
    print("figures written to", FIGURES_DIR)
