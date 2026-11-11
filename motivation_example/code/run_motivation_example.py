"""
run_motivation_example.py
=========================
The motivation example of Section 4.4: how well does a KAN reproduce a
piecewise-linear function, and does it matter whether the function's knots
are evenly spaced?

Two linear splines on [0, 10], sampled every 0.01 (1001 points):

    S(t) = sum_j  a_j * max(0, t - theta_j),   slopes a = (2, -1, 4, -3, -1)

    S1 : knots theta = (0, 2, 4, 6, 8)   evenly spaced
    S2 : knots theta = (0, 3, 5, 6, 7)   not evenly spaced

Both have the same slopes and five linear pieces; only the knots differ.
A KAN edge is a spline on an evenly spaced grid of G sub-intervals of
[0, 10], so the example asks how each function fares as G is varied.

Experiments
    A  single edge, width [1,1], linear splines (k = 1), G = 5..100   Table 5
    B  single edge, width [1,1], cubic splines  (k = 3), G = 5..100   Table 6
    C  one hidden layer, [1,2,1] and [1,3,1], linear splines,
       G = 5 and 10                                                   Table 7

Every fit uses PyKAN, 200 full-batch L-BFGS steps, a fixed grid on [0, 10]
and seed 0. The target is divided by its maximum before fitting and the
result is multiplied back, so errors are in the units of S1 and S2.

Each fit is stored as one JSON file (with the fitted curve) in
motivation_example/results/raw/ and skipped if already present. The files
shipped in that folder are the fits behind Tables 5-7.

Reproducibility: the single-edge fits (A and B) are least-squares problems
with a unique solution and re-run to the same four decimals. The
hidden-layer fits (C) are non-convex; most re-run identically, but the
[1,2,1], G = 10 fit of S2 lands in different local minima from one Python
process to the next (max error between about 0.007 and 0.04), even with
the seed fixed. To re-run everything, delete results/raw/ first.

    python run_motivation_example.py --status
    python run_motivation_example.py
"""

import argparse
import contextlib
import io
import json
import os
import time

import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(CODE_DIR), "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")

# --------------------------------------------------------------------------
# The two target functions
# --------------------------------------------------------------------------
T = np.round(np.arange(0, 10.0001, 0.01), 2)            # 1001 points
SLOPES = [2, -1, 4, -3, -1]
KNOTS = {"S1": [0, 2, 4, 6, 8], "S2": [0, 3, 5, 6, 7]}


def linear_spline(knots, slopes=SLOPES, t=T):
    """S(t) = sum_j slopes[j] * max(0, t - knots[j])."""
    return sum(a * np.maximum(0.0, t - theta) for a, theta in zip(slopes, knots))


TARGETS = {name: linear_spline(knots) for name, knots in KNOTS.items()}

# --------------------------------------------------------------------------
# The experiments: (label, width, spline degree k, list of grid sizes G)
# --------------------------------------------------------------------------
STEPS = 200
SEED = 0
EXPERIMENTS = [
    ("A", [1, 1], 1, [5, 10, 20, 50, 100]),
    ("B", [1, 1], 3, [5, 10, 20, 50, 100]),
    ("C", [1, 2, 1], 1, [5, 10]),
    ("C", [1, 3, 1], 1, [5, 10]),
]


def job_list():
    """All fits as (experiment, width, k, grid, function name)."""
    return [(label, width, k, grid, name)
            for label, width, k, grids in EXPERIMENTS for grid in grids for name in TARGETS]


def job_path(job):
    label, width, k, grid, name = job
    return os.path.join(RAW_DIR, "%s__w%s__k%d__g%03d__%s.json"
                        % (label, "-".join(map(str, width)), k, grid, name))


def fit_kan(y, width, grid, k):
    """Fit one KAN to y(T). Returns (fitted curve, number of parameters)."""
    import torch
    from kan import KAN

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    scale = float(np.abs(y).max())                       # bring the target to O(1)
    x = torch.tensor(T, dtype=torch.float32)[:, None]
    target = torch.tensor(y / scale, dtype=torch.float32)[:, None]
    dataset = {"train_input": x, "train_label": target, "test_input": x, "test_label": target}

    net = KAN(width=list(width), grid=grid, k=k, grid_range=[0, 10], seed=SEED,   # PyKAN edits the list it is given
              device="cpu", auto_save=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        net.fit(dataset, opt="LBFGS", steps=STEPS, update_grid=False)
    with torch.no_grad():
        fit = net(x).numpy()[:, 0] * scale
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return fit, int(n_parameters)


def run_job(job):
    label, width, k, grid, name = job
    y = TARGETS[name]
    fit, n_parameters = fit_kan(y, width, grid, k)
    error = fit - y
    record = {"experiment": label, "width": width, "k": k, "grid": grid, "function": name,
              "hidden_nodes": width[1] if len(width) == 3 else 0,
              "params": n_parameters,
              "max_err": float(np.abs(error).max()),
              "rmse": float(np.sqrt(np.mean(error ** 2))),
              "pred": np.round(fit, 6).tolist()}
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(job_path(job), "w") as handle:
        json.dump(record, handle)
    return record


def main():
    parser = argparse.ArgumentParser(description="Fit KANs to the two linear splines S1 and S2.")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--budget", type=float, default=float("inf"),
                        help="stop starting new fits after this many seconds")
    args = parser.parse_args()

    jobs = job_list()
    pending = [job for job in jobs if not os.path.exists(job_path(job))]
    print("%d fits, %d done, %d pending" % (len(jobs), len(jobs) - len(pending), len(pending)))
    if args.status:
        return
    started = time.time()
    for job in pending:
        if time.time() - started > args.budget:
            print("time budget used; run again to continue")
            break
        r = run_job(job)
        print("%s  width %-9s k=%d  G=%3d  %s  N=%3d  max %.4f  RMSE %.4f"
              % (r["experiment"], r["width"], r["k"], r["grid"], r["function"],
                 r["params"], r["max_err"], r["rmse"]), flush=True)


if __name__ == "__main__":
    main()
