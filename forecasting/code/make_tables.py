"""
make_tables.py
==============
Turns the stored predictions into every forecasting table of the paper.
Nothing is trained here: the script only reads the .npz files written by
run_forecasts.py, so it runs in a few seconds.

How a number is obtained
------------------------
For one model, horizon and seed, the predictions of the five walk-forward
folds are pooled and one RMSE is computed over all of them. The tables report
the mean and the sample standard deviation of that RMSE over the three seeds
(0, 100, 200). The Naive forecast has no randomness, hence no deviation.

Protocol of the paper's tables
------------------------------
The paper reports PURGED results (see forecast_lib.make_fold): Tables 10, 12
and 13 are built from the purged runs for every horizon above one day. At
the 1-day horizon purging removes nothing, so the standard 1-day runs are
used there; they also contain the two configuration screenings (Tables 9
and 11). standard_vs_purged.csv shows both protocols for all three assets.

Output (forecasting/results/tables/)
------------------------------------
    table09_lstm_configurations.csv      screening of nine LSTM architectures
    table10_lstm_by_horizon.csv          selected LSTM at every horizon
    table11_kan_configurations.csv       screening of six KAN configurations
    table12_regimes_by_horizon.csv       LSTM / KAN / Naive per market regime
    table13_training_time.csv            seconds to train one model on one fold
    kan_long_horizon_stability.csv       KAN at 100/200 days, with and without fold 0
    standard_vs_purged.csv               all three assets, both protocols
    all_runs.csv                         one row per (protocol, asset, model,
                                         configuration, horizon, seed)
    forecasting_tables.xlsx              the same tables, one sheet each

Table numbers follow the manuscript of 9 November 2026.

    python make_tables.py
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

import forecast_lib as fl
from run_forecasts import BEST_LSTM, HORIZONS, KAN_GRID, LSTM_GRID, SEEDS

PAPER_HORIZONS = [1, 2, 100, 200]          # the horizons shown in the paper


def paper_protocol(horizon):
    """Protocol behind the tables of the paper.

    The paper reports purged results. Purging removes the last H-1 training
    windows of a fold, so at H = 1 it removes nothing: the training sets are
    identical and the 'standard' 1-day runs (which also hold the two
    configuration screenings) are used. For H > 1 the purged runs are used.
    """
    return "standard" if horizon == 1 else "purged"
REGIME_ROWS = fl.REGIMES + ["All"]


# --------------------------------------------------------------------------
# Reading prediction files
# --------------------------------------------------------------------------
def prediction_dir(protocol, ticker, folder):
    return os.path.join(fl.PREDICTIONS_DIR, protocol, ticker, folder)


def load_pooled(protocol, ticker, folder, stem, horizon, seed, folds=range(fl.N_FOLDS)):
    """Pool the folds of one run.

    stem is the start of the file name, e.g. "lstmcfg__L2-20-tanh".
    Returns prediction, truth, regime labels and the per-fold metadata.
    """
    prediction, truth, labels, meta = [], [], [], []
    for fold in folds:
        name = "%s__h%d__s%d__f%d.npz" % (stem, horizon, seed, fold)
        with np.load(os.path.join(prediction_dir(protocol, ticker, folder), name)) as data:
            prediction.append(data["prediction"].astype(float))
            truth.append(data["truth"].astype(float))
            labels.append(data["labels"])
            meta.append(json.loads(str(data["meta"])))
    return np.concatenate(prediction), np.concatenate(truth), np.concatenate(labels), meta


def mean_sd(values):
    """Mean and sample standard deviation (0 when there is a single value)."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def seed_statistics(protocol, ticker, folder, stem, horizon, seeds=SEEDS):
    """Test (and, when recorded, train) RMSE over seeds for one model."""
    test, train = [], []
    for seed in seeds:
        prediction, truth, _, meta = load_pooled(protocol, ticker, folder, stem, horizon, seed)
        test.append(fl.rmse(prediction, truth))
        if all(m.get("train_rmse") is not None for m in meta):
            # training RMSE of the run: average of the folds, weighted by fold size
            sizes = np.array([m["n_train"] for m in meta], dtype=float)
            train.append(float((sizes * np.array([m["train_rmse"] for m in meta])).sum() / sizes.sum()))
    out = {"test_mean": mean_sd(test)[0], "test_sd": mean_sd(test)[1], "test_best": float(min(test))}
    if train:
        out["train_mean"], out["train_sd"] = mean_sd(train)
    return out


def naive_rmse(protocol, ticker, horizon):
    prediction, truth, _, _ = load_pooled(protocol, ticker, "naive", "naive__-", horizon, 0)
    return fl.rmse(prediction, truth)


# --------------------------------------------------------------------------
# The tables (CBA.AX, standard protocol, as printed in the paper)
# --------------------------------------------------------------------------
def naive_train_rmse(horizon=1):
    """Naive forecast evaluated on the TRAINING windows of each fold, pooled
    with the same fold-size weighting as the models' training RMSE."""
    frame = fl.load_prices("CBA.AX")
    sizes, errors = [], []
    for fold_index in range(fl.N_FOLDS):
        fold = fl.make_fold(frame, horizon, fold_index)
        to_report = fold["to_report"]
        errors.append(fl.rmse(to_report(fl.naive_persistence(fold["x_train"], horizon)),
                              to_report(fold["y_train"])))
        sizes.append(len(fold["x_train"]))
    sizes = np.array(sizes, dtype=float)
    return float((sizes * np.array(errors)).sum() / sizes.sum())


def table_lstm_configurations():
    rows = []
    for name, spec in LSTM_GRID.items():
        s = seed_statistics("standard", "CBA.AX", "lstm", "lstmcfg__" + name, 1)
        rows.append({"configuration": name, "layers": spec["layers"], "units": spec["units"],
                     "output_activation": spec["activation"],
                     "train_rmse_mean": s["train_mean"], "train_rmse_sd": s["train_sd"],
                     "test_rmse_mean": s["test_mean"], "test_rmse_sd": s["test_sd"]})
    rows.append({"configuration": "Naive persistence", "train_rmse_mean": naive_train_rmse(1),
                 "test_rmse_mean": naive_rmse("standard", "CBA.AX", 1)})
    table = pd.DataFrame(rows)
    table[["layers", "units"]] = table[["layers", "units"]].astype("Int64")   # keep integers despite the empty Naive cells
    return table


def table_lstm_by_horizon():
    rows = []
    for horizon in HORIZONS:
        s = seed_statistics(paper_protocol(horizon), "CBA.AX", "lstm", "lstmcfg__" + BEST_LSTM, horizon)
        rows.append({"horizon_days": horizon, "configuration": BEST_LSTM, "seeds": len(SEEDS),
                     "test_rmse_mean": s["test_mean"], "test_rmse_sd": s["test_sd"],
                     "best_seed_test_rmse": s["test_best"],
                     "naive_rmse": naive_rmse("standard", "CBA.AX", horizon),
                     "in_paper": horizon in PAPER_HORIZONS})
    return pd.DataFrame(rows)


def table_kan_configurations():
    rows = []
    for name, spec in KAN_GRID.items():
        s = seed_statistics("standard", "CBA.AX", "kan_grid", "kancfg__" + name, 1)
        rows.append({"configuration": name, "grid": spec["grid"], "k": spec["k"],
                     "hidden_nodes": "floor(N_train/%d)" % spec["divisor"],
                     "train_rmse_mean": s["train_mean"], "train_rmse_sd": s["train_sd"],
                     "test_rmse_mean": s["test_mean"], "test_rmse_sd": s["test_sd"]})
    rows.append({"configuration": "Naive persistence", "train_rmse_mean": naive_train_rmse(1),
                 "test_rmse_mean": naive_rmse("standard", "CBA.AX", 1)})
    table = pd.DataFrame(rows)
    table[["grid", "k"]] = table[["grid", "k"]].astype("Int64")
    return table


def regime_statistics(protocol, ticker, folder, stem, horizon, seeds):
    """RMSE per regime (and overall) for each seed -> mean, sd, window count."""
    per_regime = {regime: [] for regime in REGIME_ROWS}
    windows = {}
    for seed in seeds:
        prediction, truth, labels, _ = load_pooled(protocol, ticker, folder, stem, horizon, seed)
        for regime in REGIME_ROWS:
            mask = np.ones(len(labels), bool) if regime == "All" else labels == regime
            per_regime[regime].append(fl.rmse(prediction[mask], truth[mask]))
            windows[regime] = int(mask.sum())
    return {regime: mean_sd(v) + (windows[regime],) for regime, v in per_regime.items()}


def table_regimes(ticker="CBA.AX"):
    rows = []
    for horizon in HORIZONS:
        protocol = paper_protocol(horizon)
        lstm = regime_statistics(protocol, ticker, "lstm", "lstmcfg__" + BEST_LSTM, horizon, SEEDS)
        kan = regime_statistics(protocol, ticker, "kan", "kan__-", horizon, SEEDS)
        naive = regime_statistics(protocol, ticker, "naive", "naive__-", horizon, [0])
        for regime in REGIME_ROWS:
            rows.append({"horizon_days": horizon, "regime": regime, "windows": lstm[regime][2],
                         "lstm_rmse_mean": lstm[regime][0], "lstm_rmse_sd": lstm[regime][1],
                         "kan_rmse_mean": kan[regime][0], "kan_rmse_sd": kan[regime][1],
                         "naive_rmse": naive[regime][0],
                         "in_paper": horizon in PAPER_HORIZONS})
    return pd.DataFrame(rows)


def table_training_time():
    """Wall-clock seconds to train one model on one fold (3 seeds x 5 folds).

    The times were recorded on the machine that produced the stored
    predictions and are only meaningful relative to one another."""
    rows = []
    for horizon in HORIZONS:
        seconds = {}
        for model, folder, stem in (("lstm", "lstm", "lstmcfg__" + BEST_LSTM), ("kan", "kan", "kan__-")):
            values = []
            for seed in SEEDS:
                values += [m["train_seconds"] for m in load_pooled(paper_protocol(horizon), "CBA.AX", folder, stem, horizon, seed)[3]]
            seconds[model] = np.array(values)
        rows.append({"horizon_days": horizon,
                     "lstm_seconds_mean": seconds["lstm"].mean(), "lstm_seconds_min": seconds["lstm"].min(),
                     "lstm_seconds_max": seconds["lstm"].max(),
                     "kan_seconds_mean": seconds["kan"].mean(), "kan_seconds_min": seconds["kan"].min(),
                     "kan_seconds_max": seconds["kan"].max(),
                     "kan_over_lstm": seconds["kan"].mean() / seconds["lstm"].mean(),
                     "in_paper": horizon in PAPER_HORIZONS})
    return pd.DataFrame(rows)


def table_kan_stability():
    """KAN at the long horizons, fold by fold (purged protocol).

    The 10-step L-BFGS fit is unstable on small training sets. For each
    horizon the table gives the pooled RMSE, the RMSE without fold 0 (the
    smallest training set), the number of training windows in fold 0 and the
    range of the fold-0 predictions (the data lie in [0, 1])."""
    rows = []
    for horizon in (100, 200):
        protocol = paper_protocol(horizon)
        without_fold0, low, high = [], [], []
        for seed in SEEDS:
            prediction, truth, _, _ = load_pooled(protocol, "CBA.AX", "kan", "kan__-", horizon, seed, folds=range(1, 5))
            without_fold0.append(fl.rmse(prediction, truth))
            fold0, _, _, meta0 = load_pooled(protocol, "CBA.AX", "kan", "kan__-", horizon, seed, folds=[0])
            low.append(fold0.min()); high.append(fold0.max())
        all_folds = seed_statistics(protocol, "CBA.AX", "kan", "kan__-", horizon)
        rows.append({"horizon_days": horizon, "fold0_training_windows": meta0[0]["n_train"],
                     "kan_rmse_all_folds_mean": all_folds["test_mean"], "kan_rmse_all_folds_sd": all_folds["test_sd"],
                     "kan_rmse_without_fold0_mean": mean_sd(without_fold0)[0],
                     "kan_rmse_without_fold0_sd": mean_sd(without_fold0)[1],
                     "fold0_prediction_min": min(low), "fold0_prediction_max": max(high)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# All assets, both protocols
# --------------------------------------------------------------------------
def table_standard_vs_purged():
    """Naive, LSTM and KAN on the three assets, with and without purging."""
    rows = []
    for ticker in fl.TICKERS:
        for horizon in HORIZONS:
            row = {"ticker": ticker, "horizon_days": horizon,
                   "naive_rmse": naive_rmse("standard", ticker, horizon)}
            for protocol in ("standard", "purged"):
                for model, folder, stem in (("lstm", "lstm", "lstmcfg__" + BEST_LSTM), ("kan", "kan", "kan__-")):
                    s = seed_statistics(protocol, ticker, folder, stem, horizon)
                    row["%s_%s_mean" % (model, protocol)] = s["test_mean"]
                    row["%s_%s_sd" % (model, protocol)] = s["test_sd"]
            row["lstm_purged_over_naive"] = row["lstm_purged_mean"] / row["naive_rmse"]
            row["kan_purged_over_naive"] = row["kan_purged_mean"] / row["naive_rmse"]
            rows.append(row)
    return pd.DataFrame(rows)


def table_all_runs():
    """One row per stored run: the most detailed tabular view of the results."""
    pattern = re.compile(r"^(?P<stem>.+)__h(?P<h>\d+)__s(?P<s>\d+)__f0\.npz$")
    rows = []
    for protocol in ("standard", "purged"):
        for ticker in fl.TICKERS:
            for folder in ("naive", "lstm", "kan", "kan_grid"):
                for path in sorted(glob.glob(os.path.join(prediction_dir(protocol, ticker, folder), "*__f0.npz"))):
                    m = pattern.match(os.path.basename(path))
                    stem, horizon, seed = m["stem"], int(m["h"]), int(m["s"])
                    prediction, truth, _, meta = load_pooled(protocol, ticker, folder, stem, horizon, seed)
                    rows.append({"protocol": protocol, "ticker": ticker, "model": folder,
                                 "configuration": stem.split("__")[1], "horizon_days": horizon, "seed": seed,
                                 "test_windows": len(truth), "test_rmse": fl.rmse(prediction, truth),
                                 "parameters_last_fold": meta[-1]["parameters"],
                                 "train_seconds_total": sum(m_["train_seconds"] for m_ in meta)})
    return pd.DataFrame(rows).sort_values(
        ["protocol", "ticker", "model", "configuration", "horizon_days", "seed"]).reset_index(drop=True)


def tidy_workbook(path):
    """Freeze the header row and widen the columns of every sheet."""
    from openpyxl import load_workbook
    workbook = load_workbook(path)
    for sheet in workbook:
        sheet.freeze_panes = "A2"
        for column in sheet.columns:
            width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column[:50])
            sheet.column_dimensions[column[0].column_letter].width = min(max(10, width + 2), 45)
            for cell in column[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"
    workbook.save(path)


# --------------------------------------------------------------------------
def main():
    os.makedirs(fl.TABLES_DIR, exist_ok=True)
    tables = {
        "table09_lstm_configurations": table_lstm_configurations(),
        "table10_lstm_by_horizon": table_lstm_by_horizon(),
        "table11_kan_configurations": table_kan_configurations(),
        "table12_regimes_by_horizon": table_regimes(),
        "table13_training_time": table_training_time(),
        "kan_long_horizon_stability": table_kan_stability(),
        "standard_vs_purged": table_standard_vs_purged(),
        "all_runs": table_all_runs(),
    }
    with pd.ExcelWriter(os.path.join(fl.TABLES_DIR, "forecasting_tables.xlsx")) as workbook:
        for name, table in tables.items():
            table.to_csv(os.path.join(fl.TABLES_DIR, name + ".csv"), index=False, float_format="%.6f")
            table.to_excel(workbook, sheet_name=name[:31], index=False)
            print("wrote %-32s %4d rows" % (name + ".csv", len(table)))
    tidy_workbook(os.path.join(fl.TABLES_DIR, "forecasting_tables.xlsx"))
    print("\nTable 12, 'All' rows (CBA.AX, protocol of the paper):")
    t = tables["table12_regimes_by_horizon"]
    print(t[t.regime == "All"].drop(columns=["regime", "in_paper"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
