"""
regime_lib.py
Shared data preparation, market-regime labelling and evaluation helpers for the
regime-conditional KAN vs LSTM comparison.

Design notes (fixed before any model was evaluated):

* Data source is the same CSV the published experiments read:
  v0.2/MLP/data/CBA.AX_2020-01-01_2023-08-01.csv (905 daily rows,
  2020-01-02 to 2023-07-31). Each feature column in that file is already
  Min-Max scaled to [0, 1] over the full period, so all errors reported here
  are in the same scaled space as the published paper. Raw prices are not
  recoverable from the stored file, which is noted as a limitation.

* Sequences use a look-back of L = 20 trading days and direct multi-output
  targets y = Close[t+1 .. t+h], matching the published protocol.

* Splitting is chronological blocked walk-forward, NOT a shuffled split. The
  published pipeline called train_test_split without a date split, which lets
  overlapping windows appear in both train and test. Walk-forward removes that
  and lets the pooled test set span the whole 2020-2023 period, which is what
  makes a regime breakdown meaningful.

* Regime labels are computed from the INPUT window only (the 20 days the model
  actually sees), so no future information is used to assign a label.
  - volatility   v  = std of first differences of scaled Close over the window
  - efficiency   ER = |close_end - close_start| / sum |first differences|
                      (Kaufman efficiency ratio; 1 = perfectly directional,
                       0 = pure chop). Unitless, so invariant to the affine
                      Min-Max scaling.
  Assignment, evaluated in this order:
      Volatile  if v  >  Q75(v)  measured on the fold's training window
      Trending  else if ER > Q75(ER) measured on the fold's training window
      Normal    otherwise
  Thresholds come from training data only, never from the test block.
"""

import json
import os

import numpy as np
import pandas as pd

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
LOOKBACK = 20
N_FOLDS = 5
INITIAL_TRAIN_FRACTION = 0.40
REGIMES = ["Normal", "Volatile", "Trending"]

DATA_CSV = os.environ.get(
    "REGIME_DATA_CSV",
    "/sessions/adoring-bold-cannon/mnt/stock-pred-cos30018/"
    "v0.2/MLP/data/CBA.AX_2020-01-01_2023-08-01.csv",
)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_prices(csv_path=None):
    """Return a tidy DataFrame with a Date column and the five scaled features."""
    csv_path = csv_path or DATA_CSV
    raw = pd.read_csv(csv_path)
    date_col = None
    for col in raw.columns:
        if col.startswith("Date"):
            sample = str(raw[col].iloc[0])
            if "-" in sample and len(sample) >= 8:
                date_col = col
    if date_col is None:
        raise ValueError("no date column found in %s" % csv_path)
    out = raw[[date_col] + FEATURES].copy()
    out = out.rename(columns={date_col: "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    for col in FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out


def build_sequences(df, horizon, lookback=LOOKBACK):
    """
    Direct multi-output sequences.

    Returns
    -------
    X     : (n, lookback, 5) float32 model input
    Y     : (n, horizon)     float32 targets, Close[t+1 .. t+horizon]
    dates : (n,) datetime64  date of the LAST observed day in each input window
    """
    values = df[FEATURES].to_numpy(dtype=np.float32)
    close = df["Close"].to_numpy(dtype=np.float32)
    dates = df["Date"].to_numpy()

    n_rows = len(df)
    xs, ys, ds = [], [], []
    for end in range(lookback - 1, n_rows - horizon):
        window = values[end - lookback + 1 : end + 1]
        target = close[end + 1 : end + 1 + horizon]
        if len(target) < horizon:
            break
        xs.append(window)
        ys.append(target)
        ds.append(dates[end])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ds),
    )


# --------------------------------------------------------------------------
# regime features and labelling
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# per-fold scaling (leak-free)
# --------------------------------------------------------------------------
def fold_scaling(df, horizon, train_slice, lookback=LOOKBACK):
    """
    Min-Max parameters fitted on the rows a fold's training sequences can see.

    The CSV on disk is already Min-Max scaled over the whole series, which is
    the data-leakage issue both referees raised: the scaler had seen the test
    period. Re-fitting per fold on training rows only removes that. Because a
    Min-Max fit is invariant to a prior affine transform, fitting it on the
    pre-scaled file is identical to fitting it on raw prices, so the model sees
    exactly what it would see with raw data and training-only scaling.

    Returns (lo, hi): Series indexed by FEATURES. The last training row is the
    final target row of the last training sequence, so no test-period row
    contributes.
    """
    last_row = (lookback - 1) + (train_slice.stop - 1) + horizon
    train_rows = df[FEATURES].iloc[: last_row + 1]
    lo, hi = train_rows.min(), train_rows.max()
    hi = hi.where(hi > lo, lo + 1.0)  # guard against a constant column
    return lo, hi


def make_fold(df, horizon, fold, lookback=LOOKBACK, n_folds=N_FOLDS):
    """
    Everything one training job needs for one fold, with leak-free scaling.

    Returns a dict with x_train, y_train, x_test, y_test in FOLD scale (what the
    model trains and predicts on), truth_report (y_test in the common reporting
    scale, i.e. the file's units), labels and thresholds, and to_report(pred), a
    function that maps fold-scale Close predictions back to the reporting scale.

    The reporting scale is the whole-series Min-Max of the CSV. It is only a
    unit convention applied after prediction; nothing about it reaches the model.
    """
    x_all, y_all, dates = build_sequences(df, horizon, lookback)
    folds = walk_forward_folds(len(x_all), n_folds)
    train_slice, test_slice = folds[fold]

    # regimes: ratio statistics, invariant to the scale choice; thresholds train-only
    volatility, efficiency = window_statistics(x_all)
    thresholds = regime_thresholds(volatility[train_slice], efficiency[train_slice])
    labels = assign_regimes(volatility[test_slice], efficiency[test_slice], thresholds)

    lo, hi = fold_scaling(df, horizon, train_slice, lookback)
    scaled = df.copy()
    scaled[FEATURES] = (df[FEATURES] - lo) / (hi - lo)
    xs, ys, _ = build_sequences(scaled, horizon, lookback)

    c_lo, c_hi = float(lo["Close"]), float(hi["Close"])

    def to_report(pred):
        return np.asarray(pred, dtype=np.float64) * (c_hi - c_lo) + c_lo

    return {
        "x_train": xs[train_slice], "y_train": ys[train_slice],
        "x_test": xs[test_slice], "y_test": ys[test_slice],
        "truth_report": y_all[test_slice],
        "dates_test": dates[test_slice],
        "labels": labels, "thresholds": thresholds,
        "to_report": to_report,
        "scale": {"close_lo": c_lo, "close_hi": c_hi, "last_train_row": int((lookback - 1) + (train_slice.stop - 1) + horizon)},
        "train_slice": train_slice, "test_slice": test_slice,
    }


def window_statistics(X):
    """
    Volatility and Kaufman efficiency ratio for every input window.

    X is (n, lookback, 5); the Close column is index 3 in FEATURES.
    """
    close_idx = FEATURES.index("Close")
    closes = X[:, :, close_idx].astype(np.float64)
    diffs = np.diff(closes, axis=1)

    volatility = diffs.std(axis=1, ddof=1)

    net_move = np.abs(closes[:, -1] - closes[:, 0])
    path_length = np.abs(diffs).sum(axis=1)
    efficiency = np.where(path_length > 0, net_move / np.maximum(path_length, 1e-12), 0.0)

    return volatility, efficiency


def regime_thresholds(volatility, efficiency, quantile=0.75):
    """Thresholds derived from a training window only."""
    return {
        "volatility_q": float(np.quantile(volatility, quantile)),
        "efficiency_q": float(np.quantile(efficiency, quantile)),
        "quantile": quantile,
    }


def assign_regimes(volatility, efficiency, thresholds):
    """Apply the fixed decision rule. Returns an array of string labels."""
    labels = np.full(len(volatility), "Normal", dtype=object)
    is_volatile = volatility > thresholds["volatility_q"]
    is_trending = (~is_volatile) & (efficiency > thresholds["efficiency_q"])
    labels[is_volatile] = "Volatile"
    labels[is_trending] = "Trending"
    return labels


# --------------------------------------------------------------------------
# blocked walk-forward folds
# --------------------------------------------------------------------------
def walk_forward_folds(n_samples, n_folds=N_FOLDS, initial=INITIAL_TRAIN_FRACTION):
    """
    Expanding-window folds. Each fold trains on everything before its test
    block and tests on a contiguous block, so pooled test predictions cover
    roughly the last 60 percent of the sample period without any shuffling.
    """
    start = int(initial * n_samples)
    block = (n_samples - start) // n_folds
    folds = []
    for i in range(n_folds):
        test_start = start + i * block
        test_end = n_samples if i == n_folds - 1 else start + (i + 1) * block
        folds.append((slice(0, test_start), slice(test_start, test_end)))
    return folds


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def rmse(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def rmse_by_regime(pred, true, labels):
    """RMSE overall and per regime, plus the sample count behind each figure."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    out = {"Overall": {"rmse": rmse(pred, true), "n": int(len(true))}}
    for regime in REGIMES:
        mask = labels == regime
        count = int(mask.sum())
        out[regime] = {
            "rmse": rmse(pred[mask], true[mask]) if count else float("nan"),
            "n": count,
        }
    return out


def naive_last_value(X, horizon):
    """Persistence baseline: repeat the last observed Close for every step."""
    close_idx = FEATURES.index("Close")
    last = X[:, -1, close_idx].astype(np.float64)
    return np.repeat(last[:, None], horizon, axis=1)


# --------------------------------------------------------------------------
# result persistence
# --------------------------------------------------------------------------
def append_result(record, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    existing = []
    if os.path.exists(path):
        try:
            with open(path) as handle:
                existing = json.load(handle)
            if not isinstance(existing, list):
                existing = []
        except (ValueError, OSError):
            existing = []
    existing.append(record)
    with open(path, "w") as handle:
        json.dump(existing, handle, indent=2)
