"""
run_approximation.py
====================
The approximation study (Sections 4 and 5.2 of the paper).

Question: with a fixed number of trainable parameters N, how closely can a
model reproduce a whole price series? Each model receives the scaled closing
prices p_1..p_T indexed by the rescaled time t_i = (i-1)/(T-1) in [0, 1] and
is fitted to all of them by least squares. There is no train/test split:
the quantity of interest is the fitting error as a function of model size.

Four models, five sizes ("budgets" N* = 15, 30, 60, 120, 240), three assets,
three seeds: 4 x 5 x 3 x 3 = 180 fits.

    key      model                               setting per budget     N
    -------  ----------------------------------  ---------------------  ----------------
    kan11    KAN, no hidden layer, width [1,1]   grid size G            G + 9
    kan1h1   KAN, one hidden layer, [1,h,1]      h hidden nodes, G = 3  24 h
    mlp      MLP of t, two tanh hidden layers    width w                w^2 + 4w + 1
    fmlp     Fourier MLP (same network fed       m frequencies,         w^2 + (2m+3)w + 1
             sin/cos features of t)              width w

Where the counts come from
    KAN edge : (G + k) spline coefficients, k = 3, plus 6 further trainable
               numbers PyKAN keeps on every edge (2 scale weights and 4
               parameters of its unused symbolic branch). [1,1] has one edge,
               [1,h,1] has 2h edges with G = 3, i.e. 12 per edge.
    MLP      : w + w (layer 1) + w^2 + w (layer 2) + w + 1 (output).
    Fourier  : layer 1 has 2m inputs instead of one; the m random
               frequencies are fixed, not trained and not counted.

Each fit is saved as one JSON file (errors, parameter count, seconds and the
fitted curve) in approximation/results/raw/fits_json/ and is skipped if the
file already exists, so the script can be interrupted and resumed.

    python run_approximation.py --status
    python run_approximation.py                 # run whatever is missing
    python run_approximation.py --out my_fits   # re-run everything elsewhere
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(CODE_DIR))
DATA_DIR = os.path.join(REPO_DIR, "data")
RESULTS_DIR = os.path.join(os.path.dirname(CODE_DIR), "results")
FITS_DIR = os.path.join(RESULTS_DIR, "raw", "fits_json")

# --------------------------------------------------------------------------
# Design of the study
# --------------------------------------------------------------------------
TICKERS = ["CBA.AX", "TSLA", "LLY"]
BUDGETS = [15, 30, 60, 120, 240]
SEEDS = [0, 1, 2]
MODELS = ["kan11", "kan1h1", "mlp", "fmlp"]
MODEL_LABELS = {
    "kan11": "KAN [1,1] (no hidden layer)",
    "kan1h1": "KAN [1,h,1] (one hidden layer)",
    "mlp": "MLP of t",
    "fmlp": "Fourier MLP",
}

# Setting used at each budget, chosen so that N is as close to N* as possible.
KAN11_GRID = {15: 6, 30: 21, 60: 51, 120: 111, 240: 231}       # G        -> N = G + 9
KAN1H1_HIDDEN = {15: 1, 30: 1, 60: 2, 120: 5, 240: 10}         # h        -> N = 24 h
MLP_WIDTH = {15: 2, 30: 4, 60: 6, 120: 9, 240: 14}             # w        -> N = w^2 + 4w + 1
FMLP_SETTING = {15: (2, 2), 30: (2, 3), 60: (4, 4),            # (m, w)   -> N = w^2 + (2m+3)w + 1
                120: (6, 6), 240: (8, 9)}
# Budgets 15 and 30 cannot be met by [1,h,1] (its smallest size is 24), so
# both use h = 1 and give identical fits.

KAN_SPLINE_DEGREE = 3          # k: cubic B-splines
KAN_HIDDEN_GRID = 3            # G of the hidden-layer KAN
KAN_STEPS = 80                 # full-batch L-BFGS steps
MLP_STEPS = 20000              # full-batch Adam steps
MLP_LEARNING_RATE = 1e-2       # cosine-annealed to zero
FOURIER_SIGMA = 6.0            # frequencies b_j ~ N(0, sigma^2), in cycles over [0, 1]


# --------------------------------------------------------------------------
# Data and error measures
# --------------------------------------------------------------------------
def load_series(ticker):
    """Return (t, y): rescaled time in [0, 1] and the scaled closing price."""
    frame = pd.read_csv(os.path.join(DATA_DIR, "%s_2020-01-01_2023-08-01.csv" % ticker))
    y = frame["Close"].to_numpy(dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(y))
    return t, y


def errors(fit, y):
    """RMSE and maximum absolute error of a fitted curve."""
    residual = np.asarray(fit, dtype=float) - y
    return float(np.sqrt(np.mean(residual ** 2))), float(np.abs(residual).max())


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def fit_kan(t, y, width, grid, seed, steps=KAN_STEPS):
    """Fit a PyKAN network to y(t).

    width : [1, 1] for a single spline edge, [1, h, 1] for one hidden layer.
    grid  : number of equal sub-intervals of [0, 1] used by every edge.
    The grid is fixed (no adaptive update), so the number of parameters and
    the knot positions do not change during the fit.
    Returns (fitted curve, number of trainable parameters).
    """
    import torch
    from kan import KAN

    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.tensor(t[:, None], dtype=torch.float32)
    target = torch.tensor(y[:, None], dtype=torch.float32)
    dataset = {"train_input": x, "train_label": target, "test_input": x, "test_label": target}

    net = KAN(width=list(width), grid=grid, k=KAN_SPLINE_DEGREE, grid_range=[0, 1],
              seed=seed, device="cpu", auto_save=False)
    net.fit(dataset, opt="LBFGS", steps=steps, update_grid=False)
    with torch.no_grad():
        fit = net(x).numpy().ravel()
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return fit, int(n_parameters)


def fit_mlp(t, y, width, seed, fourier=None, steps=MLP_STEPS, learning_rate=MLP_LEARNING_RATE):
    """Fit an MLP with two tanh hidden layers of the given width to y(t).

    fourier=None        : the input is the single number t ("MLP of t").
    fourier=(m, sigma)  : the input is the vector
                              [sin(2 pi b_j t), cos(2 pi b_j t)],  j = 1..m,
                          with b_j drawn once from N(0, sigma^2) and kept
                          fixed ("Fourier MLP").
    Trained with full-batch Adam; the weights with the lowest training loss
    seen during the run are kept.
    Returns (fitted curve, number of trainable parameters).
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.tensor(t[:, None], dtype=torch.float32)
    target = torch.tensor(y[:, None], dtype=torch.float32)

    if fourier is None:
        features = x
    else:
        n_frequencies, sigma = fourier
        frequencies = torch.tensor(np.random.default_rng(seed).normal(0, sigma, (1, n_frequencies)),
                                   dtype=torch.float32)
        angle = 2 * np.pi * x @ frequencies
        features = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)

    net = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 1))
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)

    best_loss, best_weights = np.inf, None
    for _ in range(steps):
        optimiser.zero_grad()
        loss = torch.mean((net(features) - target) ** 2)
        loss.backward()
        optimiser.step()
        schedule.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = [p.detach().clone() for p in net.parameters()]

    with torch.no_grad():
        for parameter, best in zip(net.parameters(), best_weights):
            parameter.copy_(best)
        fit = net(features).numpy().ravel()
    n_parameters = sum(p.numel() for p in net.parameters())
    return fit, int(n_parameters)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
def job_list():
    """All 180 fits as (ticker, budget, model, seed)."""
    return [(ticker, budget, model, seed)
            for ticker in TICKERS for budget in BUDGETS for model in MODELS for seed in SEEDS]


def job_path(job, fits_dir=FITS_DIR):
    ticker, budget, model, seed = job
    return os.path.join(fits_dir, "%s__%s__b%d__s%d.json" % (ticker, model, budget, seed))


def run_job(job, fits_dir=FITS_DIR):
    """Run one fit and save it as JSON."""
    ticker, budget, model, seed = job
    t, y = load_series(ticker)

    started = time.time()
    if model == "kan11":
        setting = {"grid": KAN11_GRID[budget]}
        fit, n_parameters = fit_kan(t, y, [1, 1], KAN11_GRID[budget], seed)
    elif model == "kan1h1":
        setting = {"hidden": KAN1H1_HIDDEN[budget], "grid": KAN_HIDDEN_GRID}
        fit, n_parameters = fit_kan(t, y, [1, KAN1H1_HIDDEN[budget], 1], KAN_HIDDEN_GRID, seed)
    elif model == "mlp":
        setting = {"width": MLP_WIDTH[budget]}
        fit, n_parameters = fit_mlp(t, y, MLP_WIDTH[budget], seed)
    elif model == "fmlp":
        n_frequencies, width = FMLP_SETTING[budget]
        setting = {"frequencies": n_frequencies, "width": width, "sigma": FOURIER_SIGMA}
        fit, n_parameters = fit_mlp(t, y, width, seed, fourier=(n_frequencies, FOURIER_SIGMA))
    else:
        raise ValueError(model)
    seconds = time.time() - started

    rmse, max_error = errors(fit, y)
    record = {"ticker": ticker, "budget": budget, "model": model, "seed": seed,
              "params": n_parameters, "rmse": rmse, "max_err": max_error,
              "seconds": round(seconds, 2), "n_points": int(len(y)),
              "setting": setting, "pred": np.round(fit, 5).tolist()}
    os.makedirs(fits_dir, exist_ok=True)
    with open(job_path(job, fits_dir), "w") as handle:
        json.dump(record, handle)
    return record


def main():
    parser = argparse.ArgumentParser(description="Run the approximation fits that are not on disk yet.")
    parser.add_argument("--status", action="store_true", help="only report progress")
    parser.add_argument("--out", default=FITS_DIR, help="folder for the fit files")
    parser.add_argument("--budget", type=float, default=float("inf"),
                        help="stop starting new fits after this many seconds")
    args = parser.parse_args()

    jobs = job_list()
    pending = [job for job in jobs if not os.path.exists(job_path(job, args.out))]
    print("%d fits, %d done, %d pending" % (len(jobs), len(jobs) - len(pending), len(pending)))
    if args.status:
        return

    started = time.time()
    for job in pending:
        if time.time() - started > args.budget:
            print("time budget used; run again to continue")
            break
        r = run_job(job, args.out)
        print("%-7s N*=%-3d %-6s seed %d  N=%3d  RMSE %.4f  max %.4f  %5.1f s"
              % (r["ticker"], r["budget"], r["model"], r["seed"], r["params"],
                 r["rmse"], r["max_err"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
