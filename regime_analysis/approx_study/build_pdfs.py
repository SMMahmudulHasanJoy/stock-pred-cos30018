"""
build_pdfs.py: three PDFs, one per model pair, with every number and every
fit from the approximation study (fits/*.json).

  approx_splines.pdf   least-squares cubic spline (fixed knots)  ->  free-knot cubic spline
  approx_kans.pdf      KAN [1,1] (one spline edge)               ->  KAN [1,h,1] (hidden layer)
  approx_mlps.pdf      MLP of t (tanh)                           ->  Fourier-feature MLP of t
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether)

import approx_study as A

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "pdf_figs"); os.makedirs(FIG, exist_ok=True)
recs = [json.load(open(p)) for p in glob.glob(os.path.join(HERE, "fits", "*.json"))]
assert len(recs) == 210
by = defaultdict(list)
for r in recs:
    by[(r["ticker"], r["model"], r["budget"])].append(r)
for k in by:
    by[k].sort(key=lambda r: r["seed"])
TICK = {"CBA.AX": "CBA.AX (Commonwealth Bank, benchmark series)", "TSLA": "TSLA (Tesla, volatile)", "LLY": "LLY (Eli Lilly, stable trend)"}
NAIVE1 = {"CBA.AX": 0.0201, "TSLA": 0.0255, "LLY": 0.0148}

PAIRS = {
    "splines": {
        "file": "approx_splines.pdf",
        "title": "Approximating price series, part 1: least-squares cubic spline (fixed knots) and free-knot cubic spline",
        "models": [("lsq", "Least-squares cubic spline, fixed uniform knots"), ("fks", "Free-knot cubic spline, knot positions optimised")],
        "colors": {"lsq": "#555555", "fks": "#000000"},
        "what": [
            "Both models are classical cubic B-splines fitted by least squares to the whole series p(t), t in [0, 1]. "
            "They are deterministic (no seeds).",
            "Fixed-knot: G equal intervals on [0, 1]; the only free parameters are the G + 3 B-spline coefficients.",
            "Free-knot: G intervals whose G - 1 interior knot positions are also optimised (12 random starts, L-BFGS-B "
            "followed by Nelder-Mead, coefficients re-solved by least squares inside every evaluation; knots kept at "
            "least 4 data points apart). Parameters = (G + 3) coefficients + (G - 1) knot positions.",
            "The one thing that changes between the two models is whether the knots may move. Because a knot position "
            "costs a parameter, the free-knot spline has FEWER intervals than the fixed-knot spline at the same budget "
            "(e.g. 30 vs 57 intervals at ~60 parameters).",
        ],
        "params": lambda b: (A.LSQ_G[b], A.FKS_G[b]),
        "pnote": "G (intervals): fixed-knot {0}, free-knot {1}",
    },
    "kans": {
        "file": "approx_kans.pdf",
        "title": "Approximating price series, part 2: KAN [1,1] (one spline edge) and KAN [1,h,1] (hidden layer)",
        "models": [("kan11", "KAN [1,1]: one edge, cubic B-spline on G intervals + silu base term"), ("kan1h1", "KAN [1,h,1]: hidden layer of h nodes, every edge a cubic B-spline on 3 intervals + silu")],
        "colors": {"kan11": "#c46a3f", "kan1h1": "#e0a070"},
        "what": [
            "Both are pykan 0.2.8 networks (Liu et al. 2024) fitted with L-BFGS, 80 steps, grid fixed on [0, 1], k = 3.",
            "Every edge computes w_b * silu(x) + w_s * spline(x); nodes only add. There are no fixed activation functions.",
            "KAN [1,1]: a single edge from t to the output, so the whole model is one cubic B-spline on G intervals plus "
            "the silu residual. Parameters = G + 9 (G + 3 spline coefficients, scales, base and bias terms).",
            "KAN [1,h,1]: the same edge function, but with a hidden layer of h nodes; 2h edges, each a cubic B-spline on "
            "only 3 intervals. Parameters about 24h. The two smallest budgets cannot be met (h = 1 gives 24 parameters).",
            "The one thing that changes is depth: resolution on one edge versus composition of coarse edges.",
            "Three seeds (0, 1, 2) were run; the [1,1] model is deterministic to 4 decimals, the [1,h,1] model is not.",
        ],
        "params": lambda b: (A.KAN11_G[b], A.KAN1H1_H[b]),
        "pnote": "KAN [1,1] grid G = {0}; KAN [1,h,1] hidden width h = {1} (G = 3)",
    },
    "mlps": {
        "file": "approx_mlps.pdf",
        "title": "Approximating price series, part 3: MLP of t (tanh) and Fourier-feature MLP of t",
        "models": [("mlp", "MLP of t: 1 -> w -> w -> 1, tanh hidden units, linear output"), ("fmlp", "Fourier-feature MLP: [sin(2 pi B t), cos(2 pi B t)] -> w -> w -> 1, same network")],
        "colors": {"mlp": "#2f6f9f", "fmlp": "#5fb0e0"},
        "what": [
            "Both are PyTorch MLPs with two hidden layers of width w and tanh units, trained full-batch with Adam "
            "(lr 0.01, cosine schedule) for 20,000 steps, keeping the best-loss weights. Three seeds (0, 1, 2).",
            "MLP of t: input is the scalar t. Parameters = w^2 + 4w + 1.",
            "Fourier-feature MLP: the identical network, but its input is a fixed random map of t into 2 n_freq "
            "sinusoids, frequencies B ~ N(0, 6^2) drawn once per seed and NOT trained (Tancik et al. 2020). "
            "Parameters = w^2 + (2 n_freq + 3) w + 1; the width w is reduced so the total matches the budget.",
            "The one thing that changes is the input basis. Tancik et al. prove a plain MLP of a coordinate learns "
            "low frequencies first and high frequencies extremely slowly (spectral bias); the Fourier map removes that.",
            "Training length matters for the plain MLP: 4,000 steps gave RMSE 0.074 on CBA at 61 parameters, "
            "20,000 steps 0.048; all results below use 20,000 steps.",
        ],
        "params": lambda b: (A.MLP_W[b], A.FMLP[b]),
        "pnote": "MLP width w = {0}; Fourier MLP (n_freq, w) = {1}",
    },
}


def fig_rmse_vs_params(models, colors, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, tk in zip(axes, A.TICKERS):
        for m, lab in models:
            xs, ys, es = [], [], []
            for b in A.BUDGETS:
                rs = by[(tk, m, b)]; v = [r["rmse"] for r in rs]
                xs.append(rs[0]["params"]); ys.append(np.mean(v)); es.append(np.std(v, ddof=1) if len(v) > 1 else 0)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=3, label=lab.split(":")[0], color=colors[m])
        # reference: LSQ spline (grey) on every plot except the splines pdf itself
        if "lsq" not in dict(models):
            ax.plot([by[(tk, "lsq", b)][0]["params"] for b in A.BUDGETS], [by[(tk, "lsq", b)][0]["rmse"] for b in A.BUDGETS],
                    "--", color="#aaaaaa", label="LSQ spline (reference)")
        ax.axhline(NAIVE1[tk], color="k", linewidth=0.6, linestyle=":", label="1-day naive forecast RMSE")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(tk)
        ax.set_xlabel("number of parameters"); ax.set_ylabel("in-sample RMSE (Min-Max scale)")
        ax.grid(alpha=0.3, which="both", linewidth=0.4)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def fig_fits(tk, m, color, out, budgets):
    """budgets x (fit, error) for one ticker and model; best seed per budget."""
    t, y = A.load(tk)
    fig, axes = plt.subplots(len(budgets), 2, figsize=(13, 2.75 * len(budgets)), gridspec_kw={"width_ratios": [2.4, 1]})
    for row, b in enumerate(budgets):
        rs = by[(tk, m, b)]; r = min(rs, key=lambda r: r["rmse"]); p = np.array(r["pred"])
        ax = axes[row, 0]
        ax.plot(t, y, color="#999999", linewidth=0.7, label="close (scaled)")
        ax.plot(t, p, color=color, linewidth=1.1, label="fit, %d params, seed %d" % (r["params"], r["seed"]))
        if m in ("lsq", "fks"):
            for k in r["extra"]["knots"]:
                ax.axvline(k, color=color, alpha=0.25, linewidth=0.5)
        ax.set_title("budget %d: RMSE %.4f, max |err| %.4f%s" % (
            b, r["rmse"], r["max_err"], "" if len(rs) == 1 else " (best of %d seeds; seed RMSEs %s)" % (
                len(rs), ", ".join("%.4f" % q["rmse"] for q in rs))), fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(alpha=0.3, linewidth=0.4); ax.set_ylim(-0.05, 1.08)
        axe = axes[row, 1]
        axe.plot(t, p - y, color=color, linewidth=0.7); axe.axhline(0, color="k", linewidth=0.5)
        axe.set_ylim(-0.3, 0.3); axe.set_title("error = fit - close (fixed scale +-0.3)", fontsize=9); axe.grid(alpha=0.3, linewidth=0.4)
    axes[-1, 0].set_xlabel("t (0 = 2 Jan 2020, 1 = 31 Jul 2023)"); axes[-1, 1].set_xlabel("t")
    fig.suptitle("%s: %s" % (tk, m), fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_knots(tk, out):
    t, y = A.load(tk)
    fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True)
    for ax, b in zip(axes, A.BUDGETS):
        r = by[(tk, "fks", b)][0]
        ax.plot(t, y, color="#999999", linewidth=0.7)
        for k in r["extra"]["knots"]:
            ax.axvline(k, color="k", alpha=0.6, linewidth=0.8)
        ax.set_title("free-knot spline, budget %d: %d interior knots (RMSE %.4f)" % (b, len(r["extra"]["knots"]), r["rmse"]), fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.4)
    axes[-1].set_xlabel("t (0 = 2 Jan 2020, 1 = 31 Jul 2023)")
    fig.suptitle("%s: where the optimised knots land" % tk, fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def build(key):
    P = PAIRS[key]
    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["Normal"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=styles["Normal"], fontSize=7.5, leading=9.5)
    h1 = styles["Title"]; h2 = styles["Heading2"]; h3 = styles["Heading3"]
    doc = SimpleDocTemplate(os.path.join(HERE, P["file"]), pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm, topMargin=1.0 * cm, bottomMargin=1.0 * cm)
    S = []
    S.append(Paragraph(P["title"], h1))
    S.append(Paragraph("Approximation study, 2020-01-02 to 2023-07-31 daily closes, three tickers. All numbers recomputed from the saved fits (approx_study/fits/*.json).", body))
    S.append(Spacer(1, 6))
    S.append(Paragraph("1. What is being measured", h2))
    S.append(Paragraph(
        "This is an in-sample APPROXIMATION task, not forecasting. Each model receives the entire close-price series of a stock, "
        "p(t), with t rescaled to [0, 1] (905 points for CBA, 901 for TSLA and LLY; prices in the file's whole-series Min-Max scale, "
        "0 = lowest close of the period, 1 = highest), and must reproduce it as a function of t with a fixed number of parameters. "
        "The score is the RMSE and the maximum absolute error between the fitted curve and the actual series, and how both fall as "
        "the parameter budget grows (about 15, 30, 60, 120, 240). There is no train/test split and no prediction; in-sample fit "
        "says nothing about forecasting. For scale, the dotted line in the error plots is the one-day-ahead naive forecasting "
        "error of the same series (0.0201 CBA, 0.0255 TSLA, 0.0148 LLY).", body))
    S.append(Paragraph("2. The two models in this document", h2))
    for m, lab in P["models"]:
        S.append(Paragraph("<b>%s</b>" % lab, body))
    for w in P["what"]:
        S.append(Paragraph("&bull; " + w, body))
    S.append(Paragraph("3. Settings per budget", h2))
    rows = [["target budget"] + [str(b) for b in A.BUDGETS]]
    for m, lab in P["models"]:
        rows.append([lab.split(":")[0] + " actual params"] + [str(by[("CBA.AX", m, b)][0]["params"]) for b in A.BUDGETS])
    rows.append(["setting"] + [P["pnote"].format(*P["params"](b)) for b in A.BUDGETS])
    tb = Table(rows, hAlign="LEFT")
    tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    S.append(tb)
    S.append(PageBreak())

    S.append(Paragraph("4. Error versus number of parameters (all tickers)", h2))
    f = os.path.join(FIG, "%s_rmse.png" % key); fig_rmse_vs_params(P["models"], P["colors"], f)
    S.append(Image(f, width=26 * cm, height=8 * cm))
    S.append(Paragraph("Points are the mean over seeds, bars the sample standard deviation across seeds (absent for deterministic models). "
                       "Log-log axes: a straight line means error falls as a power of the parameter count.", small))
    # slopes
    S.append(Paragraph("Slope of log RMSE against log parameters, budgets 30-240 (how fast each model improves with size):", body))
    srows = [["model"] + list(A.TICKERS) + ["mean"]]
    for m, lab in P["models"]:
        sl = []
        for tk in A.TICKERS:
            x = [np.log(by[(tk, m, b)][0]["params"]) for b in A.BUDGETS[1:]]
            yv = [np.log(np.mean([r["rmse"] for r in by[(tk, m, b)]])) for b in A.BUDGETS[1:]]
            sl.append(np.polyfit(x, yv, 1)[0])
        srows.append([lab.split(":")[0]] + ["%+.2f" % s for s in sl] + ["%+.2f" % np.mean(sl)])
    tb = Table(srows, hAlign="LEFT"); tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey)]))
    S.append(tb)
    S.append(PageBreak())

    S.append(Paragraph("5. Every fit: full table", h2))
    S.append(Paragraph("One row per fit (ticker, model, budget, seed). RMSE and max |error| in the ticker's Min-Max scale; seconds = wall-clock fit time.", small))
    rows = [["ticker", "model", "budget", "seed", "params", "RMSE", "max |err|", "seconds", "setting"]]
    for tk in A.TICKERS:
        for m, lab in P["models"]:
            for b in A.BUDGETS:
                for r in by[(tk, m, b)]:
                    ex = r["extra"]
                    if m in ("lsq", "fks"):
                        setting = "%d interior knots" % len(ex["knots"])
                    elif m == "kan11":
                        setting = "G=%d, k=3" % A.KAN11_G[b]
                    elif m == "kan1h1":
                        setting = "h=%d, G=3, k=3" % A.KAN1H1_H[b]
                    elif m == "mlp":
                        setting = "w=%d" % A.MLP_W[b]
                    else:
                        setting = "n_freq=%d, w=%d, sigma=6" % A.FMLP[b]
                    rows.append([tk, m, str(b), str(r["seed"]), str(r["params"]), "%.4f" % r["rmse"], "%.4f" % r["max_err"], "%.1f" % r["seconds"], setting])
    tb = Table(rows, hAlign="LEFT", repeatRows=1)
    tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    S.append(tb)
    S.append(PageBreak())

    S.append(Paragraph("6. Summary per ticker and budget (mean +- sd over seeds)", h2))
    rows = [["ticker", "budget"] + sum(([lab.split(":")[0] + " RMSE", "max |err|", "params"] for m, lab in P["models"]), [])]
    for tk in A.TICKERS:
        for b in A.BUDGETS:
            row = [tk, str(b)]
            for m, lab in P["models"]:
                rs = by[(tk, m, b)]; v = [r["rmse"] for r in rs]; mx = [r["max_err"] for r in rs]
                row += [("%.4f" % v[0]) if len(v) == 1 else ("%.4f +- %.4f" % (np.mean(v), np.std(v, ddof=1))), "%.4f" % np.mean(mx), str(rs[0]["params"])]
            rows.append(row)
    tb = Table(rows, hAlign="LEFT", repeatRows=1)
    tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    S.append(tb)

    S.append(Spacer(1, 8))
    S.append(Paragraph("7. The fits themselves (following pages)", h2))
    S.append(Paragraph("For every ticker, model and budget: left, the fitted curve over the actual series (best seed where there are seeds; "
                       "all seed RMSEs in the title); right, the error fit - close on a fixed +-0.3 scale so panels are comparable. "
                       "Vertical lines in spline panels are knot positions.", small))
    first = True
    for tk in A.TICKERS:
        for m, lab in P["models"]:
            for part, buds in enumerate((A.BUDGETS[:3], A.BUDGETS[3:])):
                f = os.path.join(FIG, "%s_%s_%s_fits%d.png" % (key, tk, m, part)); fig_fits(tk, m, P["colors"][m], f, buds)
                S.append(PageBreak())
                S.append(KeepTogether([Paragraph("%s, %s (budgets %s)" % (TICK[tk], lab, ", ".join(map(str, buds))), h3),
                                       Image(f, width=25 * cm, height=25 * cm * 2.75 * len(buds) / 13)]))
    if key == "splines":
        for tk in A.TICKERS:
            f = os.path.join(FIG, "knots_%s.png" % tk); fig_knots(tk, f)
            S.append(PageBreak()); S.append(KeepTogether([Paragraph("%s: optimised knot positions of the free-knot spline" % TICK[tk], h3),
                                       Image(f, width=21 * cm, height=16.2 * cm)]))
    doc.build(S)
    print("wrote", P["file"])


if __name__ == "__main__":
    for k in PAIRS:
        build(k)
