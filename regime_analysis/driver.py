"""
driver.py
Resumable job runner for the regime-conditional experiments.

The sandbox these runs were executed in caps each shell invocation at well under
a minute, so the work is split into one job per (model, config, horizon, seed,
fold). Each job trains a single model on a single fold, writes its test-block
predictions to results/preds/, and records a manifest entry. Re-running the
driver skips anything already on disk, so the full grid can be completed across
as many short invocations as needed and can be resumed after an interruption.

    python driver.py --budget 38        # run jobs until ~38 seconds are used
    python driver.py --status           # show progress
"""

import argparse
import json
import os
import sys
import time

# pykan's fit() is sensitive to Python's per-process hash randomisation: it
# iterates over a hash-ordered collection, so the floating-point summation
# order, and after a few L-BFGS steps the trained model itself, differs from
# one interpreter launch to the next even with numpy and torch fully seeded.
# (Diagnosed by running the same seeded fit in three fresh processes: three
# different results; with PYTHONHASHSEED pinned: bit-identical.) The variable
# only takes effect at interpreter startup, so re-exec once if it is unset.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np

import regime_lib as rl

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = os.path.dirname(os.path.abspath(__file__))

# Where predictions are written and, just as importantly, where the driver
# looks to decide what has already been done. The repository ships with a
# populated results/ directory holding the runs behind the published tables,
# so anyone re-running this on a fresh clone would find every job already
# present and produce nothing of their own.
#
# To reproduce the experiments independently, point this somewhere empty:
#
#     REGIME_RESULTS_DIR=results_mine python driver.py --status
#
# and the same for aggregate.py, which reads the variable too. The two
# directories can then be compared directly.
RESULTS_DIR = os.environ.get("REGIME_RESULTS_DIR") or os.path.join(HERE, "results")
if not os.path.isabs(RESULTS_DIR):
    RESULTS_DIR = os.path.join(HERE, RESULTS_DIR)
PRED_DIR = os.path.join(RESULTS_DIR, "preds")
MANIFEST = os.path.join(RESULTS_DIR, "manifest.json")

# Which implementation backs the KAN jobs.
#   custom - kan_layer.py (TensorFlow reimplementation; see FINDINGS.md)
#   pykan  - the pykan package itself (requires torch), as used in the paper.
# pykan results are written to results/preds_pykan/ so the two engines can be
# compared file-for-file.
KAN_ENGINE = "custom"

HORIZONS = [1, 2, 100, 200]
SEEDS = [0, 100, 200]
LSTM_CONFIGS = {
    "100u-linear": {"units": 100, "activation": "linear", "layers": 4},
    "10u-tanh": {"units": 10, "activation": "tanh", "layers": 4},
}
EPOCHS = 25
BATCH_SIZE = 30
DROPOUT = 0.2


# --------------------------------------------------------------------------
# LSTM configuration used per horizon. These are the "best config" choices
# reported in the published paper; they are inherited rather than re-tuned,
# because re-selecting them on the walk-forward protocol would require a
# separate validation split and would bias the comparison.
PRIMARY_LSTM_CONFIG = {
    1: "100u-linear",
    2: "10u-tanh",
    100: "10u-tanh",
    200: "10u-tanh",
}

# Single-seed cross-check that the inherited choice is not obviously wrong.
CONFIG_CHECK = {1: "10u-tanh", 2: "100u-linear"}

# The six KAN configurations of Table 6 (all at the 1-day horizon, as in the
# paper). "divisor" gives the hidden width as floor(n_train / divisor).
# Note configs 2 and 5 share an identical specification in the published
# table; both rows are re-run so the table can keep its shape.
# The nine LSTM architectures of Table 4, re-run under the walk-forward
# protocol at the 1-day horizon. (4, 100, linear) coincides with the
# "100u-linear" configuration of the main grid, whose runs are reused.
LSTM_GRID_HORIZON = 1
LSTM_GRID = {
    "L4-100-linear": {"layers": 4, "units": 100, "activation": "linear"},
    "L5-100-linear": {"layers": 5, "units": 100, "activation": "linear"},
    "L6-100-linear": {"layers": 6, "units": 100, "activation": "linear"},
    "L6-50-linear": {"layers": 6, "units": 50, "activation": "linear"},
    "L6-20-linear": {"layers": 6, "units": 20, "activation": "linear"},
    "L6-20-tanh": {"layers": 6, "units": 20, "activation": "tanh"},
    "L3-20-tanh": {"layers": 3, "units": 20, "activation": "tanh"},
    "L2-20-tanh": {"layers": 2, "units": 20, "activation": "tanh"},
    "L2-10-tanh": {"layers": 2, "units": 10, "activation": "tanh"},
}

KAN_GRID_HORIZON = 1
KAN_GRID = {
    "C1": {"grid": 3, "k": 6, "divisor": 10},
    "C2": {"grid": 3, "k": 2, "divisor": 10},
    "C3": {"grid": 7, "k": 2, "divisor": 10},
    "C4": {"grid": 3, "k": 2, "divisor": 4},
    "C5": {"grid": 3, "k": 2, "divisor": 10},
    "C6": {"grid": 3, "k": 2, "divisor": 5},
}


def job_list():
    jobs = []
    for horizon in HORIZONS:
        for fold in range(rl.N_FOLDS):
            jobs.append(("naive", "-", horizon, 0, fold))
    for horizon in HORIZONS:
        for seed in SEEDS:
            for fold in range(rl.N_FOLDS):
                jobs.append(("kan", "-", horizon, seed, fold))
        for seed in SEEDS:
            for fold in range(rl.N_FOLDS):
                jobs.append(("lstm", PRIMARY_LSTM_CONFIG[horizon], horizon, seed, fold))
    for horizon, config in CONFIG_CHECK.items():
        for fold in range(rl.N_FOLDS):
            jobs.append(("lstm", config, horizon, SEEDS[0], fold))
    return jobs


def kan_grid_job_list():
    """Table 6 re-run: six configurations, pykan engine only."""
    jobs = []
    for name in KAN_GRID:
        for seed in SEEDS:
            for fold in range(rl.N_FOLDS):
                jobs.append(("kancfg", name, KAN_GRID_HORIZON, seed, fold))
    return jobs


def lstm_grid_job_list():
    """Table 4 re-run.

    Every configuration is run through this list, including L4-100-linear.
    That configuration is architecturally identical to the main grid's
    "100u-linear" (4 layers, 100 units, linear), but the main grid's job
    type does not record training error, so running it here is what
    supplies the train RMSE column for that row of Table 4.
    """
    jobs = []
    for name in LSTM_GRID:
        for seed in SEEDS:
            for fold in range(rl.N_FOLDS):
                jobs.append(("lstmcfg", name, LSTM_GRID_HORIZON, seed, fold))
    return jobs


def new_model_job_list():
    """Referee 1, major 2: MLP on the flat vector and a recurrent KAN, at
    every horizon, three seeds, five folds, under the leak-free protocol."""
    jobs = []
    for model in ("mlp", "rkan"):
        for horizon in HORIZONS:
            for seed in SEEDS:
                for fold in range(rl.N_FOLDS):
                    jobs.append((model, "-", horizon, seed, fold))
    return jobs


def best_check_job_list():
    """The best config from the new screening, at the remaining horizons."""
    jobs = []
    for horizon in (2, 100, 200):
        for seed in SEEDS:
            for fold in range(rl.N_FOLDS):
                jobs.append(("lstmcfg", "L2-20-tanh", horizon, seed, fold))
    return jobs


def estimate_seconds(job):
    """Rough cost model so a run can stop before it overshoots its time budget."""
    model, config, horizon, _, _ = job
    if model == "naive":
        return 0.5
    if model == "kancfg":
        return 6.0
    if model == "lstmcfg":
        spec = LSTM_GRID[config]
        return 4.0 + 0.09 * spec["units"] * spec["layers"]
    if model == "kan":
        return {1: 4.0, 2: 4.0, 100: 8.0, 200: 14.0}[horizon]
    if model == "mlp":
        return 8.0
    if model == "rkan":
        return 30.0
    if config == "100u-linear":
        return 34.0
    return {1: 14.0, 2: 14.0, 100: 16.0, 200: 18.0}[horizon]


def job_key(job):
    model, config, horizon, seed, fold = job
    return "%s__%s__h%d__s%d__f%d" % (model, config, horizon, seed, fold)


def job_path(job):
    directory = PRED_DIR
    if KAN_ENGINE == "pykan" and job[0] == "kan":
        directory = PRED_DIR + "_pykan"
    if job[0] == "kancfg":
        directory = PRED_DIR + "_kangrid"
    if job[0] == "lstmcfg":
        directory = PRED_DIR + "_lstmgrid"
    if job[0] in ("mlp", "rkan"):
        directory = PRED_DIR + "_newmodels"
    return os.path.join(directory, job_key(job) + ".npz")


# --------------------------------------------------------------------------
def build_lstm(config, lookback, n_features, horizon, seed):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.layers.Input(shape=(lookback, n_features))]
    for index in range(config["layers"]):
        is_last = index == config["layers"] - 1
        layers.append(tf.keras.layers.LSTM(config["units"], return_sequences=not is_last))
        layers.append(tf.keras.layers.Dropout(DROPOUT))
    layers.append(tf.keras.layers.Dense(horizon, activation=config["activation"]))
    model = tf.keras.Sequential(layers)
    model.compile(loss="mean_squared_error", optimizer="RMSprop")
    return model


def run_job(job, dataframe, cache):
    model_name, config_name, horizon, seed, fold = job

    # Leak-free fold: Min-Max fitted on this fold's training rows only. The
    # model trains and predicts in fold scale; predictions are mapped back to
    # the common reporting scale before anything is saved or scored. (Earlier
    # runs trained on a file scaled over the whole series; see FINDINGS.md.)
    key = (horizon, fold)
    if key not in cache:
        cache[key] = rl.make_fold(dataframe, horizon, fold)
    F = cache[key]
    x_train, y_train = F["x_train"], F["y_train"]
    x_test, y_test = F["x_test"], F["y_test"]
    thresholds, labels, to_report = F["thresholds"], F["labels"], F["to_report"]
    dates, test_slice = F["dates_test"], F["test_slice"]

    started = time.time()
    train_rmse = None
    if model_name == "lstmcfg":
        import tensorflow as tf

        spec = LSTM_GRID[config_name]
        net = build_lstm(spec, x_train.shape[1], x_train.shape[2], horizon,
                         seed + fold)
        net.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        prediction = net.predict(x_test, verbose=0)
        train_rmse = rl.rmse(to_report(net.predict(x_train, verbose=0)), to_report(y_train))
        parameters = net.count_params()
    elif model_name == "kancfg":
        import torch

        from kan import KAN

        spec = KAN_GRID[config_name]
        # pykan's fit() permutes the training rows each step with
        # numpy's GLOBAL generator (np.random.choice), which torch.manual_seed
        # does not touch. Left unseeded, that permutation changes on every
        # run; the summation order changes with it, and ten L-BFGS steps
        # amplify the floating-point drift into visibly different predictions
        # (found when an independent re-run failed to reproduce per-fold
        # outputs). Seeding numpy pins the permutation.
        np.random.seed(seed + fold)
        torch.manual_seed(seed + fold)
        flat_train = torch.tensor(x_train.reshape(len(x_train), -1), dtype=torch.float32)
        flat_test = torch.tensor(x_test.reshape(len(x_test), -1), dtype=torch.float32)
        hidden = max(3, len(flat_train) // spec["divisor"])
        net = KAN(width=[flat_train.shape[1], hidden, horizon], grid=spec["grid"],
                  k=spec["k"], seed=seed + fold, device="cpu", auto_save=False)
        dataset = {
            "train_input": flat_train,
            "train_label": torch.tensor(y_train, dtype=torch.float32),
            "test_input": flat_test,
            "test_label": torch.tensor(y_test, dtype=torch.float32),
        }
        net.fit(dataset, opt="LBFGS", steps=10)
        with torch.no_grad():
            prediction = net(flat_test).cpu().numpy()
            train_prediction = net(flat_train).cpu().numpy()
        train_rmse = rl.rmse(to_report(train_prediction), to_report(y_train))
        parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    elif model_name == "mlp":
        import tensorflow as tf

        from kan_layer import build_mlp

        flat_train = x_train.reshape(len(x_train), -1)
        flat_test = x_test.reshape(len(x_test), -1)
        net = build_mlp(flat_train.shape[1], horizon, width=32, seed=seed + fold)
        stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20,
                                                restore_best_weights=True)
        net.fit(flat_train, y_train, epochs=200, batch_size=BATCH_SIZE,
                verbose=0, callbacks=[stop])
        prediction = net.predict(flat_test, verbose=0)
        train_rmse = rl.rmse(to_report(net.predict(flat_train, verbose=0)), to_report(y_train))
        parameters = net.count_params()
    elif model_name == "rkan":
        from kan_layer import build_recurrent_kan

        net = build_recurrent_kan(x_train.shape[1], x_train.shape[2], horizon,
                                  units=20, seed=seed + fold)
        net.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        prediction = net.predict(x_test, verbose=0)
        train_rmse = rl.rmse(to_report(net.predict(x_train, verbose=0)), to_report(y_train))
        parameters = net.count_params()
    elif model_name == "naive":
        prediction = rl.naive_last_value(x_test, horizon)
        parameters = 0
    elif model_name == "lstm":
        net = build_lstm(LSTM_CONFIGS[config_name], x_train.shape[1], x_train.shape[2],
                         horizon, seed + fold)
        net.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        prediction = net.predict(x_test, verbose=0)
        parameters = net.count_params()
    elif model_name == "kan" and KAN_ENGINE == "pykan":
        import torch

        from kan import KAN

        # pykan's fit() permutes the training rows each step with
        # numpy's GLOBAL generator (np.random.choice), which torch.manual_seed
        # does not touch. Left unseeded, that permutation changes on every
        # run; the summation order changes with it, and ten L-BFGS steps
        # amplify the floating-point drift into visibly different predictions
        # (found when an independent re-run failed to reproduce per-fold
        # outputs). Seeding numpy pins the permutation.
        np.random.seed(seed + fold)
        torch.manual_seed(seed + fold)
        flat_train = torch.tensor(x_train.reshape(len(x_train), -1), dtype=torch.float32)
        flat_test = torch.tensor(x_test.reshape(len(x_test), -1), dtype=torch.float32)
        hidden = max(3, 3 * (len(flat_train) // 50))
        net = KAN(width=[flat_train.shape[1], hidden, horizon], grid=3, k=3,
                  seed=seed + fold, device="cpu", auto_save=False)
        dataset = {
            "train_input": flat_train,
            # labels are (n, h): NO unsqueeze(1) - that is the bug documented
            # in FINDINGS.md section 2.1
            "train_label": torch.tensor(y_train, dtype=torch.float32),
            "test_input": flat_test,
            "test_label": torch.tensor(y_test, dtype=torch.float32),
        }
        net.fit(dataset, opt="LBFGS", steps=10)
        with torch.no_grad():
            prediction = net(flat_test).cpu().numpy()
        parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    elif model_name == "kan":
        import tensorflow as tf

        from kan_layer import build_kan, fit_lbfgs

        tf.keras.utils.set_random_seed(seed + fold)
        flat_train = x_train.reshape(len(x_train), -1)
        flat_test = x_test.reshape(len(x_test), -1)
        hidden = max(3, 3 * (len(flat_train) // 50))
        net = build_kan(flat_train.shape[1], hidden, horizon, grid_size=3, spline_order=3,
                        seed=seed + fold)
        fit_lbfgs(net, flat_train, y_train, max_iterations=10)
        prediction = net.predict(flat_test, verbose=0)
        parameters = net.count_params()
    else:
        raise ValueError(model_name)
    elapsed = time.time() - started

    os.makedirs(os.path.dirname(job_path(job)), exist_ok=True)
    np.savez_compressed(
        job_path(job),
        prediction=np.asarray(to_report(prediction), dtype=np.float32),
        truth=np.asarray(F["truth_report"], dtype=np.float32),
        labels=np.asarray(labels, dtype="U8"),
        dates=np.asarray([np.datetime_as_string(d, unit="D") for d in dates]),
        meta=json.dumps(
            {
                "model": model_name,
                "config": config_name,
                "horizon": horizon,
                "seed": seed,
                "fold": fold,
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "parameters": int(parameters),
                "thresholds": thresholds,
                "train_seconds": round(elapsed, 3),
                "train_rmse": train_rmse,
                "scaling": "per-fold, training rows only",
                "scale": F["scale"],
            }
        ),
    )
    return elapsed


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=38.0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cost-scale", type=float, default=1.0)
    parser.add_argument("--kan-engine", choices=["custom", "pykan"], default="custom")
    parser.add_argument("--kan-grid", action="store_true",
                        help="run the Table 6 configuration grid (pykan)")
    parser.add_argument("--lstm-grid", action="store_true",
                        help="run the Table 4 configuration grid")
    parser.add_argument("--best-check", action="store_true",
                        help="run the best screened LSTM config at h=2/100/200")
    parser.add_argument("--new-models", action="store_true",
                        help="run the MLP-on-flat-vector and recurrent-KAN comparison")
    args = parser.parse_args()

    global KAN_ENGINE
    KAN_ENGINE = args.kan_engine

    if args.threads:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    if args.new_models:
        jobs = new_model_job_list()
    elif args.best_check:
        jobs = best_check_job_list()
    elif args.lstm_grid:
        jobs = lstm_grid_job_list()
    elif args.kan_grid:
        jobs = kan_grid_job_list()
    else:
        jobs = job_list()
        if KAN_ENGINE == "pykan":
            # only KAN jobs differ between engines; naive and LSTM are unchanged
            jobs = [job for job in jobs if job[0] == "kan"]
    pending = [job for job in jobs if not os.path.exists(job_path(job))]
    if args.workers > 1:
        pending = [job for index, job in enumerate(pending) if index % args.workers == args.worker]

    if args.status:
        print("total %d  done %d  pending %d" % (len(jobs), len(jobs) - len(pending), len(pending)))
        if pending:
            print("next:", job_key(pending[0]))
        return

    dataframe = rl.load_prices()
    cache = {}
    started = time.time()
    completed = 0
    for job in pending:
        used = time.time() - started
        if used + estimate_seconds(job) * args.cost_scale > args.budget:
            continue
        elapsed = run_job(job, dataframe, cache)
        completed += 1
        print("%s  %.1fs" % (job_key(job), elapsed), flush=True)

    remaining = len([job for job in jobs if not os.path.exists(job_path(job))])
    print("ran %d jobs, %d remaining" % (completed, remaining))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MANIFEST, "w") as handle:
        json.dump({"total": len(jobs), "remaining": remaining}, handle, indent=2)


if __name__ == "__main__":
    main()
