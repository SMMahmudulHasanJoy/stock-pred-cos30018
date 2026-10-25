"""aggregate_approx.py: tables, checks and figures for the approximation study."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import approx_study as A

MODELS = ["lsq", "fks", "kan11", "kan1h1", "mlp", "fmlp"]
LABEL = {"lsq": "LSQ cubic spline (fixed knots)", "fks": "free-knot cubic spline",
         "kan11": "KAN [1,1] (one spline edge)", "kan1h1": "KAN [1,h,1] (hidden layer)",
         "mlp": "MLP of t (tanh)", "fmlp": "Fourier-feature MLP of t"}
COL = {"lsq": "#555555", "fks": "#000000", "kan11": "#c46a3f", "kan1h1": "#e0a070",
       "mlp": "#2f6f9f", "fmlp": "#5fb0e0"}

recs = [json.load(open(p)) for p in glob.glob(os.path.join(HERE, "fits", "*.json"))]
assert len(recs) == 210, len(recs)
by = defaultdict(list)
for r in recs:
    by[(r["ticker"], r["model"], r["budget"])].append(r)

# ---- checks
# 1. kan11 vs lsq at equal knot count: pykan's grid G equals G intervals; compare where counts match
print("CHECK: KAN[1,1] vs LSQ spline at the same number of intervals (G) and seed-to-seed spread")
for tk in A.TICKERS:
    for b in A.BUDGETS:
        k = sorted(r["rmse"] for r in by[(tk, "kan11", b)])
        l = by[(tk, "lsq", b)][0]["rmse"]
        print("  %-7s b%-3d KAN11 G=%-3d rmse seeds=%s | LSQ G=%d rmse=%.4f"
              % (tk, b, A.KAN11_G[b], ["%.4f" % v for v in k], A.LSQ_G[b], l))
# 2. parameter counts actually used
print("\nCHECK: actual parameter counts per model and budget (CBA; identical for other tickers)")
for m in MODELS:
    print("  %-7s" % m, [by[("CBA.AX", m, b)][0]["params"] for b in A.BUDGETS])

# ---- table
lines = ["| Ticker | Budget | " + " | ".join(LABEL[m] for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
summary = {}
for tk in A.TICKERS:
    for b in A.BUDGETS:
        cells = []
        for m in MODELS:
            rs = by[(tk, m, b)]
            v = [r["rmse"] for r in rs]
            mean, sd = float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            summary["%s|%s|%d" % (tk, m, b)] = {"params": rs[0]["params"], "rmse_mean": mean, "rmse_sd": sd,
                                                "max_err_mean": float(np.mean([r["max_err"] for r in rs])), "n": len(v)}
            cells.append(("%.4f" % mean) if len(v) == 1 else ("%.4f ± %.4f" % (mean, sd)))
            cells[-1] += " (%d)" % rs[0]["params"]
        lines.append("| %s | %d | " % (tk, b) + " | ".join(cells) + " |")
table = "\n".join(lines)
open(os.path.join(HERE, "approx_table.md"), "w").write(table + "\n\nRMSE (mean ± sd over 3 seeds for neural models) in the ticker's Min-Max scale; actual parameter count in brackets.\n")
json.dump(summary, open(os.path.join(HERE, "approx_summary.json"), "w"), indent=1)
print("\n" + table)

# ---- figure: RMSE vs parameters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, tk in zip(axes, A.TICKERS):
    for m in MODELS:
        xs = [summary["%s|%s|%d" % (tk, m, b)]["params"] for b in A.BUDGETS]
        ys = [summary["%s|%s|%d" % (tk, m, b)]["rmse_mean"] for b in A.BUDGETS]
        es = [summary["%s|%s|%d" % (tk, m, b)]["rmse_sd"] for b in A.BUDGETS]
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=2, label=LABEL[m], color=COL[m],
                    linestyle="--" if m in ("lsq", "fks") else "-")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of parameters"); ax.set_ylabel("in-sample RMSE (Min-Max scale)")
    ax.set_title(tk); ax.grid(alpha=0.3, which="both", linewidth=0.4)
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "approx_rmse_vs_params.png"), dpi=200); plt.close(fig)

# ---- figure: fits at budget 60 on CBA
t, y = A.load("CBA.AX")
fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, sharey=True)
for ax, m in zip(axes.ravel(), MODELS):
    r = sorted(by[("CBA.AX", m, 60)], key=lambda r: r["rmse"])[0]
    ax.plot(t, y, color="#999999", linewidth=0.8, label="CBA close (scaled)")
    ax.plot(t, r["pred"], color=COL[m], linewidth=1.2, label="%s, %d params, RMSE %.4f" % (LABEL[m], r["params"], r["rmse"]))
    if m in ("lsq", "fks"):
        for k in r["extra"]["knots"]:
            ax.axvline(k, color=COL[m], alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(alpha=0.3, linewidth=0.4)
axes[-1, 0].set_xlabel("t (0 = 2 Jan 2020, 1 = 31 Jul 2023)"); axes[-1, 1].set_xlabel("t")
fig.suptitle("Approximating the CBA close series with ~60 parameters (best of 3 seeds); vertical lines = spline knots")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "approx_fits_cba_b60.png"), dpi=170); plt.close(fig)
print("\nfigures written")
