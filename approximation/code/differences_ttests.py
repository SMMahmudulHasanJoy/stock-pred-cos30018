"""
differences_ttests.py
=====================
Pairwise comparison of the approximation models: differences in RMSE, maximum
error and fitting time, and paired t-tests on those differences.

Comparisons (always A - B, so a NEGATIVE error difference means A is better)
    KAN [1,1]    - MLP of t
    KAN [1,1]    - Fourier MLP
    KAN [1,h,1]  - KAN [1,1]
    Fourier MLP  - MLP of t

How the test is built
---------------------
A *cell* is one asset at one budget (3 x 5 = 15 cells). Inside a cell, seed s
of model A is paired with seed s of model B and the three differences are
averaged; by linearity this equals (mean of A) - (mean of B). The paired
t-test is then the one-sample t-test of "mean difference = 0" on the 15 cell
differences (14 degrees of freedom), and on the 5 cells of each asset
(4 degrees of freedom).

The 15 cells are not independent (the five budgets of an asset use the same
series), so the p-values describe how consistent a difference is across
cells; they are not exact significance levels. A Wilcoxon signed-rank test
and a sign test, which need no normality assumption, are reported alongside.

Output
------
results/differences/
    differences_per_seed.csv     one row per comparison, asset, budget, seed
    differences_summary.csv      one row per comparison, asset, budget (60 rows)
    table16_kan_minus_mlp.csv    the two KAN [1,1] comparisons, as in the paper
results/ttests/
    paired_ttests.csv            t, p, 95 % interval, effect size, Wilcoxon, sign test
results/excel/
    differences_and_ttests.xlsx  the three tables above in one workbook
    TableDiff.xlsx               differences, in the layout used for the paper
    TableActual.xlsx             the matching actual values of both models

    python differences_ttests.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

import run_approximation as study
from make_tables import load_fits, tidy_workbook

DIFFERENCES_DIR = os.path.join(study.RESULTS_DIR, "differences")
TTESTS_DIR = os.path.join(study.RESULTS_DIR, "ttests")
EXCEL_DIR = os.path.join(study.RESULTS_DIR, "excel")

# (model A, model B, label)
COMPARISONS = [
    ("kan11", "mlp", "KAN [1,1] - MLP of t"),
    ("kan11", "fmlp", "KAN [1,1] - Fourier MLP"),
    ("kan1h1", "kan11", "KAN [1,h,1] - KAN [1,1]"),
    ("fmlp", "mlp", "Fourier MLP - MLP of t"),
]
METRICS = [("rmse", "RMSE"), ("max_error", "max |error|"), ("seconds", "seconds")]


# --------------------------------------------------------------------------
def per_seed_differences(fits):
    """Seed-matched differences A - B for every comparison, asset and budget."""
    keyed = fits.set_index(["ticker", "budget", "model", "seed"])
    rows = []
    for model_a, model_b, label in COMPARISONS:
        for ticker in study.TICKERS:
            for budget in study.BUDGETS:
                for seed in study.SEEDS:
                    a = keyed.loc[(ticker, budget, model_a, seed)]
                    b = keyed.loc[(ticker, budget, model_b, seed)]
                    row = {"comparison": label, "ticker": ticker, "budget": budget, "seed": seed,
                           "parameters_A": a.parameters, "parameters_B": b.parameters}
                    for metric, _ in METRICS:
                        row[metric + "_A"] = a[metric]
                        row[metric + "_B"] = b[metric]
                        row["d_" + metric] = a[metric] - b[metric]
                    rows.append(row)
    return pd.DataFrame(rows)


def cell_summary(per_seed):
    """Average the three seeds of every cell (mean and sd of the differences,
    and the mean value of each model)."""
    aggregations = {"parameters_A": ("parameters_A", "first"), "parameters_B": ("parameters_B", "first")}
    for metric, _ in METRICS:
        aggregations["d_%s_mean" % metric] = ("d_" + metric, "mean")
        aggregations["d_%s_sd" % metric] = ("d_" + metric, "std")
        aggregations[metric + "_A"] = (metric + "_A", "mean")
        aggregations[metric + "_B"] = (metric + "_B", "mean")
    return per_seed.groupby(["comparison", "ticker", "budget"], sort=False).agg(**aggregations).reset_index()


def paired_test(differences):
    """One-sample tests of 'mean difference = 0' on a vector of cell differences."""
    d = np.asarray(differences, dtype=float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    t_statistic, p_value = stats.ttest_1samp(d, 0.0)
    half_width = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)        # 95 % confidence interval
    negative = int((d < 0).sum())
    try:
        wilcoxon_p = stats.wilcoxon(d).pvalue
    except ValueError:                                              # all differences zero
        wilcoxon_p = np.nan
    return {"n_cells": n, "mean_difference": mean, "sd_difference": sd,
            "ci95_low": mean - half_width, "ci95_high": mean + half_width,
            "t_statistic": t_statistic, "degrees_of_freedom": n - 1, "p_value": p_value,
            "effect_size_dz": mean / sd if sd > 0 else np.nan,
            "wilcoxon_p_value": wilcoxon_p,
            "cells_with_negative_difference": negative,
            "sign_test_p_value": stats.binomtest(negative, n, 0.5).pvalue}


def all_tests(summary):
    """Tests for every comparison and metric: all 15 cells, then per asset."""
    rows = []
    for _, _, label in COMPARISONS:
        cells = summary[summary.comparison == label]
        for metric, metric_label in METRICS:
            column = "d_%s_mean" % metric
            rows.append({"comparison": label, "metric": metric_label, "scope": "all 15 cells",
                         **paired_test(cells[column])})
            for ticker in study.TICKERS:
                rows.append({"comparison": label, "metric": metric_label, "scope": ticker,
                             **paired_test(cells[cells.ticker == ticker][column])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    for folder in (DIFFERENCES_DIR, TTESTS_DIR, EXCEL_DIR):
        os.makedirs(folder, exist_ok=True)
    fits, _ = load_fits()

    per_seed = per_seed_differences(fits)
    summary = cell_summary(per_seed)
    tests = all_tests(summary)

    # Table 16 of the paper: KAN without hidden layer against the two MLPs
    table16 = summary[summary.comparison.isin([COMPARISONS[0][2], COMPARISONS[1][2]])][
        ["comparison", "ticker", "parameters_A", "parameters_B",
         "d_rmse_mean", "d_max_error_mean", "d_seconds_mean"]]

    per_seed.to_csv(os.path.join(DIFFERENCES_DIR, "differences_per_seed.csv"), index=False, float_format="%.6f")
    summary.to_csv(os.path.join(DIFFERENCES_DIR, "differences_summary.csv"), index=False, float_format="%.6f")
    table16.to_csv(os.path.join(DIFFERENCES_DIR, "table16_kan_minus_mlp.csv"), index=False, float_format="%.6f")
    tests.to_csv(os.path.join(TTESTS_DIR, "paired_ttests.csv"), index=False, float_format="%.6g")

    with pd.ExcelWriter(os.path.join(EXCEL_DIR, "differences_and_ttests.xlsx")) as workbook:
        summary.to_excel(workbook, sheet_name="Differences (A - B)", index=False)
        tests.to_excel(workbook, sheet_name="Paired t-tests", index=False)
        per_seed.to_excel(workbook, sheet_name="Per seed", index=False)

    # The two workbooks in the layout used while writing the paper:
    # 60 rows = 4 comparisons x 3 assets x 5 budgets, same order in both files.
    layout = summary[["comparison", "ticker", "parameters_A", "parameters_B"]]
    with pd.ExcelWriter(os.path.join(EXCEL_DIR, "TableDiff.xlsx")) as workbook:
        pd.concat([layout, summary[["d_rmse_mean", "d_max_error_mean", "d_seconds_mean"]]], axis=1).to_excel(
            workbook, sheet_name="Difference", index=False)
    with pd.ExcelWriter(os.path.join(EXCEL_DIR, "TableActual.xlsx")) as workbook:
        pd.concat([layout, summary[["rmse_A", "rmse_B", "max_error_A", "max_error_B",
                                    "seconds_A", "seconds_B"]]], axis=1).to_excel(
            workbook, sheet_name="Actual", index=False)

    for name in ("differences_and_ttests.xlsx", "TableDiff.xlsx", "TableActual.xlsx"):
        tidy_workbook(os.path.join(EXCEL_DIR, name))
    print("wrote differences/ (3 files), ttests/paired_ttests.csv, excel/ (3 workbooks)\n")
    headline = tests[(tests.scope == "all 15 cells") & (tests.metric != "seconds")]
    print(headline[["comparison", "metric", "mean_difference", "t_statistic", "p_value",
                    "wilcoxon_p_value", "cells_with_negative_difference"]].to_string(index=False))


if __name__ == "__main__":
    main()
