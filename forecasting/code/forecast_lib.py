"""
forecast_lib.py
===============
Shared building blocks of the forecasting study (Sections 3 and 5.1 of the
paper): data loading, sequence construction, walk-forward folds, leak-free
scaling, market-regime labels, the Naive persistence forecast and the RMSE
metric. Every other script in this folder imports from here, so the protocol
is defined in exactly one place.

Protocol in brief
-----------------
* Input      : the last L = 20 trading days of Open, High, Low, Close, Volume.
* Target     : the next H closing prices, predicted in one shot
               (direct multi-output forecasting), H in {1, 2, 5, 10, 100, 200}.
* Validation : five expanding walk-forward folds; fold i trains on every
               sequence before its test block, so nothing is shuffled.
* Scaling    : Min-Max, fitted on the training rows of each fold only.
* Purging    : optional (see make_fold). Removes the training sequences whose
               targets reach into the test period.
* Regimes    : each test window is labelled Normal / Volatile / Trending from
               its 20 input days only, with thresholds taken from training data.
* Reporting  : predictions are mapped back to the scale of the data file
               (whole-series Min-Max of each asset) before RMSE is computed.
"""

import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants of the protocol
# --------------------------------------------------------------------------
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
LOOKBACK = 20                    # L: length of the input window, in trading days
N_FOLDS = 5                      # number of walk-forward folds
INITIAL_TRAIN_FRACTION = 0.40    # share of the sequences used to train fold 0
REGIMES = ["Normal", "Volatile", "Trending"]

# --------------------------------------------------------------------------
# Locations (everything is relative to the repository, nothing is hard-coded)
# --------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(CODE_DIR))
DATA_DIR = os.path.join(REPO_DIR, "data")
RESULTS_DIR = os.path.join(os.path.dirname(CODE_DIR), "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TICKERS = ["CBA.AX", "TSLA", "LLY"]
DATA_FILES = {ticker: os.path.join(DATA_DIR, "%s_2020-01-01_2023-08-01.csv" % ticker)
              for ticker in TICKERS}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_prices(ticker="CBA.AX"):
    """Return the daily data of one asset as a DataFrame: Date + five features.

    The files in data/ are already Min-Max scaled to [0, 1] over the whole
    period; that scale is only the *reporting* unit. The models never see it,
    because make_fold() re-scales every fold from its training rows.
    """
    frame = pd.read_csv(DATA_FILES[ticker])
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").dropna().reset_index(drop=True)
    return frame[["Date"] + FEATURES]


def build_sequences(frame, horizon, lookback=LOOKBACK):
    """Cut the series into (input window, target vector) pairs.

    Returns
    -------
    x     : (n, lookback, 5)  the input windows
    y     : (n, horizon)      Close[t+1 .. t+horizon] for each window
    dates : (n,)              date of the last observed day of each window
    """
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    close = frame["Close"].to_numpy(dtype=np.float32)
    dates = frame["Date"].to_numpy()

    x, y, last_day = [], [], []
    for end in range(lookback - 1, len(frame) - horizon):
        x.append(values[end - lookback + 1: end + 1])
        y.append(close[end + 1: end + 1 + horizon])
        last_day.append(dates[end])
    return (np.asarray(x, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            np.asarray(last_day))


# --------------------------------------------------------------------------
# Walk-forward folds
# --------------------------------------------------------------------------
def walk_forward_folds(n_sequences, n_folds=N_FOLDS, initial=INITIAL_TRAIN_FRACTION):
    """Expanding-window folds as a list of (train_slice, test_slice).

    The first 40 % of the sequences train fold 0; the remaining 60 % are cut
    into five consecutive test blocks. Fold i trains on everything before its
    own block, so the pooled test blocks cover the last 60 % of the period.
    """
    start = int(initial * n_sequences)
    block = (n_sequences - start) // n_folds
    folds = []
    for i in range(n_folds):
        test_start = start + i * block
        test_end = n_sequences if i == n_folds - 1 else start + (i + 1) * block
        folds.append((slice(0, test_start), slice(test_start, test_end)))
    return folds


def fold_scaling(frame, horizon, train_slice, lookback=LOOKBACK):
    """Min-Max limits (lo, hi) from the rows a fold's training set can see.

    The last row used is the final target row of the last training sequence,
    so no test-period value influences the scaling.
    """
    last_row = (lookback - 1) + (train_slice.stop - 1) + horizon
    train_rows = frame[FEATURES].iloc[: last_row + 1]
    lo, hi = train_rows.min(), train_rows.max()
    hi = hi.where(hi > lo, lo + 1.0)        # guard against a constant column
    return lo, hi


# --------------------------------------------------------------------------
# Market regimes
# --------------------------------------------------------------------------
def window_statistics(x):
    """Volatility and Kaufman efficiency ratio of every input window.

    volatility : standard deviation of the day-to-day changes of Close
    efficiency : |last - first| / sum of |day-to-day changes|
                 (1 = a straight move, 0 = pure back-and-forth)
    Both are computed from the 20 observed days only.
    """
    closes = x[:, :, FEATURES.index("Close")].astype(np.float64)
    changes = np.diff(closes, axis=1)
    volatility = changes.std(axis=1, ddof=1)
    net_move = np.abs(closes[:, -1] - closes[:, 0])
    path_length = np.abs(changes).sum(axis=1)
    efficiency = np.where(path_length > 0, net_move / np.maximum(path_length, 1e-12), 0.0)
    return volatility, efficiency


def regime_thresholds(volatility, efficiency, quantile=0.75):
    """75th percentiles of both statistics, measured on training windows."""
    return {"volatility_q": float(np.quantile(volatility, quantile)),
            "efficiency_q": float(np.quantile(efficiency, quantile)),
            "quantile": quantile}


def assign_regimes(volatility, efficiency, thresholds):
    """Volatile if volatility is high; otherwise Trending if efficiency is
    high; otherwise Normal."""
    labels = np.full(len(volatility), "Normal", dtype=object)
    is_volatile = volatility > thresholds["volatility_q"]
    is_trending = (~is_volatile) & (efficiency > thresholds["efficiency_q"])
    labels[is_volatile] = "Volatile"
    labels[is_trending] = "Trending"
    return labels


# --------------------------------------------------------------------------
# One fold, ready for training
# --------------------------------------------------------------------------
def make_fold(frame, horizon, fold, purge=False, lookback=LOOKBACK, n_folds=N_FOLDS):
    """Everything one training job needs for one fold.

    Returns a dictionary with
        x_train, y_train, x_test, y_test : arrays in FOLD scale (what the
                                           model trains and predicts on)
        truth_report  : y_test in the reporting scale (the units of the file)
        to_report     : function mapping fold-scale predictions to that scale
        labels        : regime label of every test window
        dates_test    : last observed day of every test window
        thresholds, scale, purged : bookkeeping written into the result file

    purge
        With direct multi-step targets, training sequence i has targets on
        days end_i+1 .. end_i+H. The last H-1 training sequences therefore
        carry targets that lie AFTER the first test origin. For H = 1 this is
        harmless; for H = 100 or 200 every test window shares target days
        with the training labels. purge=True removes those H-1 sequences
        (and fits the scaler on the same reduced range), as in purged
        cross-validation (Lopez de Prado, 2018, ch. 7). The test windows are
        identical in both settings, so the two can be compared directly.
    """
    x_all, y_all, dates = build_sequences(frame, horizon, lookback)
    train_slice, test_slice = walk_forward_folds(len(x_all), n_folds)[fold]

    # Regime labels: thresholds from the windows before the test block. These
    # depend on input windows only, so purging does not change them.
    volatility, efficiency = window_statistics(x_all)
    thresholds = regime_thresholds(volatility[train_slice], efficiency[train_slice])
    labels = assign_regimes(volatility[test_slice], efficiency[test_slice], thresholds)

    if purge:
        train_slice = slice(0, max(1, test_slice.start - horizon + 1))

    # Leak-free scaling: fit on training rows, apply to the whole series.
    lo, hi = fold_scaling(frame, horizon, train_slice, lookback)
    scaled = frame.copy()
    scaled[FEATURES] = (frame[FEATURES] - lo) / (hi - lo)
    x_scaled, y_scaled, _ = build_sequences(scaled, horizon, lookback)

    close_lo, close_hi = float(lo["Close"]), float(hi["Close"])

    def to_report(prediction):
        """Undo the fold scaling of Close."""
        return np.asarray(prediction, dtype=np.float64) * (close_hi - close_lo) + close_lo

    return {
        "x_train": x_scaled[train_slice], "y_train": y_scaled[train_slice],
        "x_test": x_scaled[test_slice], "y_test": y_scaled[test_slice],
        "truth_report": y_all[test_slice],
        "to_report": to_report,
        "labels": labels,
        "dates_test": dates[test_slice],
        "thresholds": thresholds,
        "scale": {"close_lo": close_lo, "close_hi": close_hi,
                  "last_train_row": int((lookback - 1) + (train_slice.stop - 1) + horizon)},
        "purged": bool(purge),
    }


# --------------------------------------------------------------------------
# Baseline and metric
# --------------------------------------------------------------------------
def naive_persistence(x, horizon):
    """Naive persistence forecast: repeat the last observed Close H times."""
    last_close = x[:, -1, FEATURES.index("Close")].astype(np.float64)
    return np.repeat(last_close[:, None], horizon, axis=1)


def rmse(prediction, truth):
    """Root mean squared error over every element of the two arrays."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))
