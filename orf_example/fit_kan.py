"""
fit_kan.py  --  approximate S1 and S2 (ExampleForORF.m) with small KANs.

S1 and S2 are piecewise linear with the same five slopes; S1's knots are
evenly spaced (0,2,4,6,8), S2's are not (0,3,5,6,7). A KAN's edge functions
are B-splines on a FIXED, evenly spaced grid, so the experiment asks: how
does each function fare as that grid is varied?

Usage:  python fit_kan.py            (about 2-3 minutes on CPU)
Writes: results.json, results.md, figure_fits.png, figure_error_vs_grid.png
"""
import contextlib, io, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from kan import KAN

HERE = os.path.dirname(os.path.abspath(__file__))
T = np.round(np.arange(0, 10.0001, 0.01), 2)
S1 = np.loadtxt(os.path.join(HERE, "S1.txt"), delimiter=",")
S2 = np.loadtxt(os.path.join(HERE, "S2.txt"), delimiter=",")
STEPS = 200          # L-BFGS steps for the KAN fits
SEED = 0

def fit(y, width, grid, k, steps=STEPS):
    """Fit one KAN to y(T). width=[1,1] is a single spline edge."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    x = torch.tensor(T, dtype=torch.float32)[:, None]
    yy = torch.tensor(y, dtype=torch.float32)[:, None]
    # scale the target to O(1) for the optimiser; undo afterwards
    s = float(np.abs(y).max())
    ds = {"train_input": x, "train_label": yy / s, "test_input": x, "test_label": yy / s}
    net = KAN(width=width, grid=grid, k=k, grid_range=[0, 10], seed=SEED,
              device="cpu", auto_save=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        net.fit(ds, opt="LBFGS", steps=steps, update_grid=False)
    with torch.no_grad():
        pred = net(x).numpy()[:, 0] * s
    err = pred - y
    n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return pred, float(np.abs(err).max()), float(np.sqrt(np.mean(err ** 2))), n_par

results = []
# --- Experiment A: a single spline edge, grid size swept, linear and cubic ---
for k in (1, 3):
    for grid in (5, 10, 20, 50, 100):
        for name, y in (("S1", S1), ("S2", S2)):
            _, mx, rm, npar = fit(y, [1, 1], grid, k)
            results.append(dict(exp="A: single edge [1,1]", fn=name, k=k, grid=grid, params=npar, max_err=mx, rmse=rm))
            print(f"[1,1] k={k} grid={grid:3d} {name}: max err {mx:.4f}  rmse {rm:.4f}  ({npar} params)", flush=True)

# --- Experiment B: one hidden layer, 2 or 3 units, coarse grid (what Nadia suggested) ---
for hidden in (2, 3):
    for grid in (5, 10):
        for name, y in (("S1", S1), ("S2", S2)):
            _, mx, rm, npar = fit(y, [1, hidden, 1], grid, 1)
            results.append(dict(exp=f"B: hidden layer [1,{hidden},1]", fn=name, k=1, grid=grid, params=npar, max_err=mx, rmse=rm))
            print(f"[1,{hidden},1] k=1 grid={grid:3d} {name}: max err {mx:.4f}  rmse {rm:.4f}  ({npar} params)", flush=True)

# --- Reference: free-knot piecewise-linear fit (knots AND slopes learned) ---
# For fixed knots the slopes are a linear least-squares problem, so only the
# knots need a non-convex search; several random starts, keep the best.
def free_knot(y, n_knots=5, starts=14, steps=300):
    t = torch.tensor(T, dtype=torch.float64); yy = torch.tensor(y, dtype=torch.float64)
    best = None
    for st in range(starts):
        g = torch.Generator().manual_seed(SEED + st)
        theta = torch.nn.Parameter(torch.sort(torch.rand(n_knots, generator=g, dtype=torch.float64) * 9)[0])
        opt = torch.optim.Adam([theta], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            B = torch.cat([torch.ones_like(t)[:, None], torch.relu(t[:, None] - theta[None, :])], 1)
            coef = torch.linalg.lstsq(B, yy[:, None]).solution
            loss = ((B @ coef)[:, 0] - yy).pow(2).mean(); loss.backward(); opt.step()
            with torch.no_grad(): theta.clamp_(0, 10)
        with torch.no_grad():
            B = torch.cat([torch.ones_like(t)[:, None], torch.relu(t[:, None] - theta[None, :])], 1)
            pred = (B @ torch.linalg.lstsq(B, yy[:, None]).solution)[:, 0].numpy()
        err = pred - y; mx = float(np.abs(err).max())
        if best is None or mx < best[1]:
            best = (pred, mx, float(np.sqrt(np.mean(err ** 2))), sorted(theta.detach().numpy().round(3).tolist()))
    return best
for name, y in (("S1", S1), ("S2", S2)):
    _, mx, rm, knots = free_knot(y)
    results.append(dict(exp="C: free-knot linear spline (reference)", fn=name, k=1, grid=5, params=11, max_err=mx, rmse=rm, knots=knots))
    print(f"free-knot {name}: max err {mx:.4f} rmse {rm:.4f} knots found {knots}", flush=True)

json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)

# --- Figures ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for i, (name, y) in enumerate((("S1", S1), ("S2", S2))):
    ax[i].plot(T, y, "k", lw=2, label=name + " (target)")
    for grid, c in ((5, "#c46a3f"), (20, "#2f6f9f")):
        p, mx, _, _ = fit(y, [1, 1], grid, 1)
        ax[i].plot(T, p, c, lw=1, ls="--", label=f"KAN, linear spline, grid={grid} (max err {mx:.2f})")
    knots = [0, 2, 4, 6, 8] if name == "S1" else [0, 3, 5, 6, 7]
    for kx in knots: ax[i].axvline(kx, color="grey", lw=0.5, alpha=0.5)
    ax[i].set_title(f"{name}: true knots at {knots}"); ax[i].legend(fontsize=7); ax[i].set_xlabel("t")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure_fits.png"), dpi=150)

fig, ax = plt.subplots(figsize=(6, 3.6))
for name, m, c in (("S1", "o", "#2f6f9f"), ("S2", "s", "#c46a3f")):
    for k, ls in ((1, "-"), (3, "--")):
        rows = sorted([r for r in results if r["exp"].startswith("A") and r["fn"] == name and r["k"] == k], key=lambda r: r["grid"])
        ax.plot([r["grid"] for r in rows], [r["max_err"] for r in rows], marker=m, ls=ls, color=c, label=f"{name}, k={k}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("grid intervals (evenly spaced)"); ax.set_ylabel("max |error|")
ax.set_title("Single-edge KAN: error vs grid size"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figure_error_vs_grid.png"), dpi=150)
print("done")
