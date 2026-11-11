"""
run_forecasts.py
================
Trains the forecasting models and stores their test predictions.

One *job* = one model, one horizon, one seed, one walk-forward fold. Each job
writes a single .npz file to

    forecasting/results/predictions/<protocol>/<ticker>/<model folder>/

and is skipped when its file already exists, so the script can be stopped and
restarted at any time and never repeats finished work.

Models
------
naive      Naive persistence: repeat the last observed close (no training).
lstm       Stacked LSTM (Keras), dropout 0.2, RMSprop, 25 epochs, batch 30.
           The architecture is picked by name from LSTM_GRID.
kan        KAN (PyKAN) on the flattened window, width [100, N_h, H],
           grid 3, cubic splines, 10 L-BFGS steps (the paper's Table 2).
kan_grid   The six KAN configurations screened at the 1-day horizon.

Experiments (--experiment)
--------------------------
main       naive, kan and the selected LSTM (2 layers, 20 units, tanh) at
           horizons 1, 2, 5, 10, 100, 200.            -> Tables 10, 12, 13
lstm_grid  nine LSTM architectures at the 1-day horizon -> Table 9
kan_grid   six KAN configurations at the 1-day horizon  -> Table 11

Examples
--------
    python run_forecasts.py --status                       # what is done
    python run_forecasts.py --experiment lstm_grid         # CBA.AX, standard
    python run_forecasts.py --ticker TSLA --purge          # purged protocol

The shipped predictions make every job count as done. To re-run from scratch,
write somewhere else:   python run_forecasts.py --out my_predictions
"""

import argparse
import json
import os
import sys
import time

# PyKAN iterates over a hash-ordered collection during fit(); without a fixed
# hash seed the same seeded run gives slightly different numbers in every new
# Python process. The variable must be set before Python starts, so the script
# restarts itself once with it set.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # silence TensorFlow start-up

import numpy as np

import forecast_lib as fl

# --------------------------------------------------------------------------
# Experimental settings
# --------------------------------------------------------------------------
HORIZONS = [1, 2, 5, 10, 100, 200]
SEEDS = [0, 100, 200]

# LSTM training (identical for every architecture)
EPOCHS = 25
BATCH_SIZE = 30
DROPOUT = 0.2

# The nine LSTM architectures of the screening (Table 9).
LSTM_GRID = {
    "L4-100-linear": {"layers": 4, "units": 100, "activation": "linear"},
    "L5-100-linear": {"layers": 5, "units": 100, "activation": "linear"},
    "L6-100-linear": {"layers": 6, "units": 100, "activation": "linear"},
    "L6-50-linear":  {"layers": 6, "units": 50,  "activation": "linear"},
    "L6-20-linear":  {"layers": 6, "units": 20,  "activation": "linear"},
    "L6-20-tanh":    {"layers": 6, "units": 20,  "activation": "tanh"},
    "L3-20-tanh":    {"layers": 3, "units": 20,  "activation": "tanh"},
    "L2-20-tanh":    {"layers": 2, "units": 20,  "activation": "tanh"},
    "L2-10-tanh":    {"layers": 2, "units": 10,  "activation": "tanh"},
}
BEST_LSTM = "L2-20-tanh"          # lowest test RMSE in the screening

# The six KAN configurations of the screening (Table 11). The hidden layer
# has floor(N_train / divisor) nodes. C2 and C5 are identical on purpose:
# their agreement is a check that the evaluation is stable.
KAN_GRID = {
    "C1": {"grid": 3, "k": 6, "divisor": 10},
    "C2": {"grid": 3, "k": 2, "divisor": 10},
    "C3": {"grid": 7, "k": 2, "divisor": 10},
    "C4": {"grid": 3, "k": 2, "divisor": 4},
    "C5": {"grid": 3, "k": 2, "divisor": 10},
    "C6": {"grid": 3, "k": 2, "divisor": 5},
}
KAN_STEPS = 10                    # L-BFGS steps for every forecasting KAN

# Folder and file-name prefix used for each model.
MODEL_FOLDER = {"naive": "naive", "lstm": "lstm", "kan": "kan", "kan_grid": "kan_grid"}
FILE_PREFIX = {"naive": "naive", "lstm": "lstmcfg", "kan": "kan", "kan_grid": "kancfg"}


# --------------------------------------------------------------------------
# Job lists
# --------------------------------------------------------------------------
def job_list(experiment):
    """Return the jobs of an experiment as (model, config, horizon, seed, fold)."""
    folds = range(fl.N_FOLDS)
    jobs = []
    if experiment == "main":
        for horizon in HORIZONS:
            jobs += [("naive", "-", horizon, 0, fold) for fold in folds]
            jobs += [("lstm", BEST_LSTM, horizon, seed, fold) for seed in SEEDS for fold in folds]
            jobs += [("kan", "-", horizon, seed, fold) for seed in SEEDS for fold in folds]
    elif experiment == "lstm_grid":
        for name in LSTM_GRID:
            jobs += [("lstm", name, 1, seed, fold) for seed in SEEDS for fold in folds]
    elif experiment == "kan_grid":
        for name in KAN_GRID:
            jobs += [("kan_grid", name, 1, seed, fold) for seed in SEEDS for fold in folds]
    else:
        raise ValueError(experiment)
    return jobs


def job_name(job):
    model, config, horizon, seed, fold = job
    return "%s__%s__h%d__s%d__f%d" % (FILE_PREFIX[model], config, horizon, seed, fold)


def job_path(job, out_dir):
    return os.path.join(out_dir, MODEL_FOLDER[job[0]], job_name(job) + ".npz")


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_lstm(spec, lookback, n_features, horizon, seed):
    """Stacked LSTM -> Dropout blocks followed by a Dense(H) output layer."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=(lookback, n_features))]
    for index in range(spec["layers"]):
        is_last = index == spec["layers"] - 1
        layers.append(tf.keras.layers.LSTM(spec["units"], return_sequences=not is_last))
        layers.append(tf.keras.layers.Dropout(DROPOUT))
    layers.append(tf.keras.layers.Dense(horizon, activation=spec["activation"]))
    model = tf.keras.Sequential(layers)
    model.compile(loss="mean_squared_error", optimizer="RMSprop")
    return model


def train_lstm(spec, fold, horizon, seed):
    """Train one LSTM; return (test prediction, train prediction, #parameters)."""
    x_train, y_train = fold["x_train"], fold["y_train"]
    net = build_lstm(spec, x_train.shape[1], x_train.shape[2], horizon, seed)
    net.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return (net.predict(fold["x_test"], verbose=0),
            net.predict(x_train, verbose=0),
            net.count_params())


def train_kan(fold, horizon, seed, grid, k, hidden):
    """Train one PyKAN network on the flattened window.

    The 20 x 5 input window is flattened to a vector of 100 numbers, because
    a standard KAN takes a flat input. Network width: [100, hidden, H].
    """
    import torch
    from kan import KAN

    # PyKAN shuffles with numpy's global generator, so numpy must be seeded too.
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_train = torch.tensor(fold["x_train"].reshape(len(fold["x_train"]), -1), dtype=torch.float32)
    x_test = torch.tensor(fold["x_test"].reshape(len(fold["x_test"]), -1), dtype=torch.float32)
    dataset = {
        "train_input": x_train,
        "train_label": torch.tensor(fold["y_train"], dtype=torch.float32),   # shape (n, H)
        "test_input": x_test,
        "test_label": torch.tensor(fold["y_test"], dtype=torch.float32),
    }
    net = KAN(width=[x_train.shape[1], hidden, horizon], grid=grid, k=k,
              seed=seed, device="cpu", auto_save=False)
    net.fit(dataset, opt="LBFGS", steps=KAN_STEPS)
    with torch.no_grad():
        test_prediction = net(x_test).cpu().numpy()
        train_prediction = net(x_train).cpu().numpy()
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return test_prediction, train_prediction, n_parameters


# --------------------------------------------------------------------------
# Running one job
# --------------------------------------------------------------------------
def run_job(job, frame, fold_cache, purge, out_dir):
    """Train (if needed), predict the test block and save one result file."""
    model, config, horizon, seed, fold_index = job

    key = (horizon, fold_index)
    if key not in fold_cache:
        fold_cache[key] = fl.make_fold(frame, horizon, fold_index, purge=purge)
    fold = fold_cache[key]
    to_report = fold["to_report"]
    n_train = len(fold["x_train"])
    run_seed = seed + fold_index            # a different initialisation per fold

    started = time.time()
    train_prediction = None
    if model == "naive":
        prediction, n_parameters = fl.naive_persistence(fold["x_test"], horizon), 0
    elif model == "lstm":
        prediction, train_prediction, n_parameters = train_lstm(
            LSTM_GRID[config], fold, horizon, run_seed)
    elif model == "kan":
        hidden = max(3, 3 * (n_train // 50))            # heuristic of Table 2
        prediction, train_prediction, n_parameters = train_kan(
            fold, horizon, run_seed, grid=3, k=3, hidden=hidden)
    elif model == "kan_grid":
        spec = KAN_GRID[config]
        hidden = max(3, n_train // spec["divisor"])
        prediction, train_prediction, n_parameters = train_kan(
            fold, horizon, run_seed, grid=spec["grid"], k=spec["k"], hidden=hidden)
    else:
        raise ValueError(model)
    seconds = time.time() - started

    train_rmse = None
    if train_prediction is not None:
        train_rmse = fl.rmse(to_report(train_prediction), to_report(fold["y_train"]))

    path = job_path(job, out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        prediction=np.asarray(to_report(prediction), dtype=np.float32),   # reporting scale
        truth=np.asarray(fold["truth_report"], dtype=np.float32),
        labels=np.asarray(fold["labels"], dtype="U8"),                    # regime per window
        dates=np.asarray([np.datetime_as_string(d, unit="D") for d in fold["dates_test"]]),
        meta=json.dumps({
            "model": FILE_PREFIX[model], "config": config, "horizon": horizon,
            "seed": seed, "fold": fold_index,
            "n_train": int(n_train), "n_test": int(len(fold["x_test"])),
            "parameters": int(n_parameters),
            "train_seconds": round(seconds, 3), "train_rmse": train_rmse,
            "thresholds": fold["thresholds"], "scale": fold["scale"],
            "scaling": "per-fold, training rows only", "purged": fold["purged"],
        }),
    )
    return seconds


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiment", choices=["main", "lstm_grid", "kan_grid"], default="main")
    parser.add_argument("--ticker", choices=fl.TICKERS, default="CBA.AX")
    parser.add_argument("--purge", action="store_true",
                        help="drop training sequences whose targets reach into the test period")
    parser.add_argument("--out", default=None,
                        help="write predictions here instead of forecasting/results/predictions")
    parser.add_argument("--budget", type=float, default=float("inf"),
                        help="stop starting new jobs after this many seconds")
    parser.add_argument("--status", action="store_true", help="only report progress")
    args = parser.parse_args()

    protocol = "purged" if args.purge else "standard"
    out_dir = os.path.join(args.out or fl.PREDICTIONS_DIR, protocol, args.ticker)

    jobs = job_list(args.experiment)
    pending = [job for job in jobs if not os.path.exists(job_path(job, out_dir))]
    print("%s / %s / %s: %d jobs, %d done, %d pending"
          % (args.ticker, protocol, args.experiment, len(jobs), len(jobs) - len(pending), len(pending)))
    if args.status or not pending:
        return

    frame = fl.load_prices(args.ticker)
    fold_cache = {}
    started = time.time()
    for job in pending:
        if time.time() - started > args.budget:
            print("time budget used; run again to continue")
            break
        seconds = run_job(job, frame, fold_cache, args.purge, out_dir)
        print("%-40s %6.1f s" % (job_name(job), seconds), flush=True)


if __name__ == "__main__":
    main()
