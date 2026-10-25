"""
approx_study.py
In-sample APPROXIMATION of a price series p(t), t in [0,1], by six function
approximators at matched parameter budgets. This is not forecasting: every
model sees the whole series and is scored on how closely, and how compactly,
it reproduces it. See APPROX_REPORT.md for the rationale.

Models (one thing changes between neighbours):
  kan11   single-edge KAN [1,1], cubic B-spline (k=3) on G uniform intervals + silu base
  kan1h1  KAN [1,h,1], k=3, G=3: the same edge functions, but with a hidden layer
  mlp     MLP of t: 1 -> w -> w -> 1, tanh hidden units, linear output
  fmlp    the same MLP preceded by a fixed random Fourier feature map of t
          (Tancik et al. 2020): [sin(2*pi*B t), cos(2*pi*B t)], B ~ N(0, sigma^2)
  lsq     least-squares cubic spline with G uniform knots (classical, deterministic)
  fks     free-knot cubic spline: knot positions optimised (multi-start L-BFGS-B)

Parameter budgets are approximately 15, 30, 60, 120, 240; the ACTUAL count of
every fit is recorded and used on the x-axis. Neural models use 3 seeds.
Errors are RMSE and max |error| in the file's Min-Max scale.

Resumable: one JSON per fit in approx_study/fits/. Run repeatedly with a time
budget:  python approx_study.py --budget 150
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FITS = os.path.join(HERE, "fits")
TICKERS = {
    "CBA.AX": os.path.join(ROOT, "..", "v0.2", "MLP", "data", "CBA.AX_2020-01-01_2023-08-01.csv"),
    "TSLA": os.path.join(ROOT, "..", "data", "TSLA_2020-01-01_2023-08-01.csv"),
    "LLY": os.path.join(ROOT, "..", "data", "LLY_2020-01-01_2023-08-01.csv"),
}
BUDGETS = [15, 30, 60, 120, 240]
SEEDS = [0, 1, 2]
# per-budget settings chosen so the actual parameter counts land near the budget
KAN11_G = {15: 6, 30: 21, 60: 51, 120: 111, 240: 231}        # params = G + k + 6 = G + 9
KAN1H1_H = {15: 1, 30: 1, 60: 2, 120: 5, 240: 10}            # pykan count ~ 24h for G=3,k=3 (budgets 15/30 both get h=1)
MLP_W = {15: 2, 30: 4, 60: 6, 120: 9, 240: 14}               # params = w^2 + 4w + 1
FMLP = {15: (2, 2), 30: (2, 3), 60: (4, 4), 120: (6, 6), 240: (8, 9)}  # (n_freq, width); params = w^2 + (2nf+3)w + 1
LSQ_G = {15: 12, 30: 27, 60: 57, 120: 117, 240: 237}         # params = G + 3 coefficients
FKS_G = {15: 7, 30: 15, 60: 30, 120: 60, 240: 120}           # params = (G+3) coef + (G-1) knots


def load(ticker):
    df = pd.read_csv(TICKERS[ticker])
    y = df["Close"].to_numpy(dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(y))
    return t, y


def errors(pred, y):
    e = np.asarray(pred, float) - y
    return float(np.sqrt(np.mean(e ** 2))), float(np.abs(e).max())


# ---------------------------------------------------------------- splines
def fit_lsq(t, y, G):
    from scipy.interpolate import LSQUnivariateSpline
    knots = np.linspace(0, 1, G + 1)[1:-1]
    sp = LSQUnivariateSpline(t, y, knots, k=3)
    return sp(t), len(sp.get_coeffs()), {"knots": knots.tolist()}


def fit_fks(t, y, G, seed=0, starts=12):
    """Free-knot cubic spline: optimise G-1 interior knots, LSQ coefficients inside."""
    from scipy.interpolate import LSQUnivariateSpline
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    n_int = G - 1
    gap = 4.0 / len(t)  # at least ~4 data points between knots (Schoenberg-Whitney)

    def to_knots(z):
        # z unconstrained -> increasing knots in (gap, 1-gap) via softmax spacing
        w = np.exp(z - z.max()); w = w / w.sum()
        cum = np.cumsum(w)[:-1]
        return gap + cum * (1 - 2 * gap)

    def obj(z):
        k = to_knots(z)
        if np.any(np.diff(np.r_[0, k, 1]) < gap):
            return 1e3
        try:
            sp = LSQUnivariateSpline(t, y, k, k=3)
        except Exception:
            return 1e3
        return float(np.mean((sp(t) - y) ** 2))

    best = (np.inf, None)
    for s in range(starts):
        z0 = np.zeros(n_int + 1) if s == 0 else rng.normal(0, 1.0, n_int + 1)
        r = minimize(obj, z0, method="L-BFGS-B", options={"maxiter": 300})
        r2 = minimize(obj, r.x, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-12})
        if r2.fun < best[0]:
            best = (r2.fun, r2.x)
    k = to_knots(best[1])
    sp = LSQUnivariateSpline(t, y, k, k=3)
    return sp(t), len(sp.get_coeffs()) + n_int, {"knots": k.tolist()}


# ---------------------------------------------------------------- KAN (pykan)
def fit_kan(t, y, width, G, seed, steps=80):
    import torch
    from kan import KAN
    torch.manual_seed(seed); np.random.seed(seed)
    net = KAN(width=width, grid=G, k=3, grid_range=[0, 1], seed=seed, device="cpu", auto_save=False)
    X = torch.tensor(t[:, None], dtype=torch.float32); Y = torch.tensor(y[:, None], dtype=torch.float32)
    ds = {"train_input": X, "train_label": Y, "test_input": X, "test_label": Y}
    net.fit(ds, opt="LBFGS", steps=steps, update_grid=False)
    with torch.no_grad():
        p = net(X).numpy().ravel()
    npar = sum(q.numel() for q in net.parameters() if q.requires_grad)
    return p, int(npar), {}


# ---------------------------------------------------------------- MLPs (torch)
def fit_mlp(t, y, width, seed, fourier=None, epochs=20000, lr=1e-2):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.tensor(t[:, None], dtype=torch.float32); Y = torch.tensor(y[:, None], dtype=torch.float32)
    if fourier:
        n_freq, sigma = fourier
        B = torch.tensor(np.random.default_rng(seed).normal(0, sigma, (1, n_freq)), dtype=torch.float32)
        def feat(x):
            z = 2 * np.pi * x @ B
            return torch.cat([torch.sin(z), torch.cos(z)], 1)
        d_in = 2 * n_freq
    else:
        feat = lambda x: x
        d_in = 1
    net = torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    F = feat(X)
    best = (np.inf, None)
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.mean((net(F) - Y) ** 2)
        loss.backward(); opt.step(); sched.step()
        if loss.item() < best[0]:
            best = (loss.item(), [q.detach().clone() for q in net.parameters()])
    with torch.no_grad():
        for q, b in zip(net.parameters(), best[1]):
            q.copy_(b)
        p = net(F).numpy().ravel()
    npar = sum(q.numel() for q in net.parameters())
    return p, int(npar), {"fourier": fourier}


# ---------------------------------------------------------------- job list
def jobs():
    out = []
    for tk in TICKERS:
        for b in BUDGETS:
            out.append((tk, b, "lsq", 0))
            out.append((tk, b, "fks", 0))
            for s in SEEDS:
                out.append((tk, b, "kan11", s))
                out.append((tk, b, "kan1h1", s))
                out.append((tk, b, "mlp", s))
                out.append((tk, b, "fmlp", s))
    return out


def path(job):
    tk, b, m, s = job
    return os.path.join(FITS, "%s__%s__b%d__s%d.json" % (tk, m, b, s))


def run(job):
    tk, b, m, s = job
    t, y = load(tk)
    started = time.time()
    if m == "lsq":
        p, n, extra = fit_lsq(t, y, LSQ_G[b])
    elif m == "fks":
        p, n, extra = fit_fks(t, y, FKS_G[b], seed=s)
    elif m == "kan11":
        p, n, extra = fit_kan(t, y, [1, 1], KAN11_G[b], s)
    elif m == "kan1h1":
        p, n, extra = fit_kan(t, y, [1, KAN1H1_H[b], 1], 3, s)
    elif m == "mlp":
        p, n, extra = fit_mlp(t, y, MLP_W[b], s)
    elif m == "fmlp":
        nf, w = FMLP[b]
        p, n, extra = fit_mlp(t, y, w, s, fourier=(nf, 6.0))
    rm, mx = errors(p, y)
    rec = {"ticker": tk, "budget": b, "model": m, "seed": s, "params": n, "rmse": rm, "max_err": mx,
           "seconds": round(time.time() - started, 2), "n_points": int(len(y)), "extra": extra,
           "pred": np.round(p, 5).tolist()}
    os.makedirs(FITS, exist_ok=True)
    with open(path(job), "w") as f:
        json.dump(rec, f)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    pending = [j for j in jobs() if not os.path.exists(path(j))]
    print("total %d pending %d" % (len(jobs()), len(pending)))
    if a.status:
        return
    t0 = time.time()
    for j in pending:
        if time.time() - t0 > a.budget:
            break
        r = run(j)
        print("%-7s b%-3d %-6s s%d params=%4d rmse=%.4f max=%.4f %.1fs" % (
            r["ticker"], r["budget"], r["model"], r["seed"], r["params"], r["rmse"], r["max_err"], r["seconds"]), flush=True)


if __name__ == "__main__":
    main()
