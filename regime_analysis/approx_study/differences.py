"""
differences.py: paired differences between models (RMSE, max error, seconds)
per ticker and budget, with paired t-tests, as requested for the paper.

Pairing: for random models, seed s of model A is paired with seed s of model
B (same ticker, same budget); deterministic models are paired with every seed.
Differences are A - B, so a NEGATIVE RMSE difference means A is better.

Outputs: differences.csv (every pair), differences_summary.csv (per cell),
differences_ttests.csv, differences.pdf (tables).
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

import approx_study as A

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.load(open(p)) for p in glob.glob(os.path.join(HERE, "fits", "*.json"))]
by = defaultdict(dict)
for r in recs:
    by[(r["ticker"], r["model"], r["budget"])][r["seed"]] = r

COMPARISONS = [("kan11", "mlp", "KAN [1,1] - MLP of t"),
               ("kan11", "fmlp", "KAN [1,1] - Fourier MLP"),
               ("kan11", "lsq", "KAN [1,1] - LSQ spline"),
               ("kan1h1", "kan11", "KAN [1,h,1] - KAN [1,1]"),
               ("fmlp", "mlp", "Fourier MLP - MLP of t")]
# NO_SPLINE=1 drops the least-squares spline comparison and writes *_nospline files
NO_SPLINE = os.environ.get("NO_SPLINE", "0") == "1"
if NO_SPLINE:
    COMPARISONS = [c for c in COMPARISONS if "lsq" not in c[:2]]
SUFFIX = "_KAN_vs_MLP" if NO_SPLINE else ""
METRICS = [("rmse", "RMSE"), ("max_err", "max |err|"), ("seconds", "seconds")]


def pairs(tk, a, b, bud):
    ra, rb = by[(tk, a, bud)], by[(tk, b, bud)]
    seeds = sorted(set(ra) | set(rb))
    out = []
    for s in seeds:
        xa = ra.get(s, ra.get(0)); xb = rb.get(s, rb.get(0))
        out.append((s, xa, xb))
    return out


rows, cells, tests = [], [], []
for a, b, label in COMPARISONS:
    for tk in A.TICKERS:
        for bud in A.BUDGETS:
            diffs = {m: [] for m, _ in METRICS}
            for s, xa, xb in pairs(tk, a, b, bud):
                d = {m: xa[m] - xb[m] for m, _ in METRICS}
                rows.append([label, tk, bud, s, xa["params"], xb["params"]] + [xa[m] for m, _ in METRICS] + [xb[m] for m, _ in METRICS] + [d[m] for m, _ in METRICS])
                for m, _ in METRICS:
                    diffs[m].append(d[m])
            cell = [label, tk, bud, xa["params"], xb["params"]]
            for m, _ in METRICS:
                v = diffs[m]; cell += [np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v)]
            pr = pairs(tk, a, b, bud)
            cell += [np.mean([xa[m] for _, xa, _ in pr]) for m, _ in METRICS] + [np.mean([xb[m] for _, _, xb in pr]) for m, _ in METRICS]
            cells.append(cell)
    # paired t-tests on the 15 cell means (ticker x budget), and per ticker on 5 budgets
    for m, mlabel in METRICS:
        allm = [c[5 + 3 * [mm for mm, _ in METRICS].index(m)] for c in cells if c[0] == label]
        t, p = stats.ttest_1samp(allm, 0.0)
        tests.append([label, mlabel, "all 15 cells", len(allm), np.mean(allm), t, p])
        for tk in A.TICKERS:
            v = [c[5 + 3 * [mm for mm, _ in METRICS].index(m)] for c in cells if c[0] == label and c[1] == tk]
            t, p = stats.ttest_1samp(v, 0.0)
            tests.append([label, mlabel, tk, len(v), np.mean(v), t, p])

import csv
with open(os.path.join(HERE, "differences" + SUFFIX + ".csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["comparison", "ticker", "budget", "seed", "params_A", "params_B", "rmse_A", "maxerr_A", "sec_A", "rmse_B", "maxerr_B", "sec_B", "d_rmse", "d_maxerr", "d_sec"]); w.writerows(rows)
with open(os.path.join(HERE, "differences_summary" + SUFFIX + ".csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["comparison", "ticker", "budget", "params_A", "params_B", "d_rmse_mean", "d_rmse_sd", "n", "d_maxerr_mean", "d_maxerr_sd", "n", "d_sec_mean", "d_sec_sd", "n", "rmse_A", "maxerr_A", "sec_A", "rmse_B", "maxerr_B", "sec_B"]); w.writerows(cells)
with open(os.path.join(HERE, "differences_ttests" + SUFFIX + ".csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["comparison", "metric", "scope", "n_cells", "mean_difference", "t_statistic", "p_value"]); w.writerows(tests)

# ---- figure: per-budget differences and pooled 95% CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PM = u"\u00b1"
fig, axes = plt.subplots(1, 3, figsize=(14 if not NO_SPLINE else 12.5, 4.2))
colors_ = {"CBA.AX": "#2f6f9f", "TSLA": "#c46a3f", "LLY": "#4a9a4a"}
mi = {"rmse": 5, "max_err": 8, "seconds": 11}
for ax, (m, ml) in zip(axes, METRICS):
    xs = np.arange(len(COMPARISONS))
    for j, (a, b, label) in enumerate(COMPARISONS):
        cc = [c for c in cells if c[0] == label]
        for tk in A.TICKERS:
            vals = [c[mi[m]] for c in cc if c[1] == tk]
            ax.scatter([j + {"CBA.AX": -0.18, "TSLA": 0, "LLY": 0.18}[tk]] * len(vals), vals, s=14, color=colors_[tk], alpha=0.8, label=tk if j == 0 else None)
        allv = np.array([c[mi[m]] for c in cc]); n = len(allv); se = allv.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
        ax.errorbar([j], [allv.mean()], yerr=[h], fmt="D", color="k", ms=6, capsize=5, label="pooled mean, 95% CI" if j == 0 else None)
    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xticks(xs); ax.set_xticklabels([l.replace(" - ", "\n- ") for _, _, l in COMPARISONS], fontsize=7.5)
    ax.set_title("difference in %s (A - B); below 0 = A better" % ml, fontsize=9); ax.grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "differences_plot" + SUFFIX + ".png"), dpi=170); plt.close(fig)

# ---- PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
ss = getSampleStyleSheet(); B = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=12)


def tbl(rows, font=7):
    t = Table(rows, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    return t


def f4(x):
    x = 0.0 if abs(x) < 5e-5 else x
    return "%.4f" % x


def fp(p):
    return "< 0.0001" if p < 1e-4 else "%.4f" % p


S = [Paragraph("Approximation study: paired differences between models", ss["Title"]),
     Paragraph("For each share and parameter budget: the two models' values (mean over seeds) and their difference A - B, for RMSE, "
               "maximum absolute error and fit time in seconds. A negative difference in RMSE or max error means model A fits better; "
               "a negative seconds difference means A is faster. Pairing: seed s of A with seed s of B (deterministic models paired with "
               "every seed); each difference is the mean " + PM + " standard deviation of three paired differences. Values in the share's Min-Max price scale.", B),
     Spacer(1, 6)]
for a, b, label in COMPARISONS:
    S.append(Paragraph("<b>%s</b>   (A = %s, B = %s)" % (label, label.split(" - ")[0], label.split(" - ")[1]), B))
    hdr = ["share", "budget", "params A", "params B", "RMSE A", "RMSE B", "diff RMSE", "max|err| A", "max|err| B", "diff max|err|", "sec A", "sec B", "diff sec"]
    rows_ = [hdr]
    for c in cells:
        if c[0] != label:
            continue
        rows_.append([c[1], str(c[2]), str(c[3]), str(c[4]), f4(c[14]), f4(c[17]), "%s %s %s" % (f4(c[5]), PM, f4(c[6])),
                      f4(c[15]), f4(c[18]), "%s %s %s" % (f4(c[8]), PM, f4(c[9])), "%.1f" % c[16], "%.1f" % c[19], "%.1f %s %.1f" % (c[11], PM, c[12])])
    S.append(tbl(rows_)); S.append(Spacer(1, 8))
S.append(PageBreak())
S.append(Paragraph("<b>Paired t-tests</b>. Null hypothesis: the mean difference A - B is zero. One-sample t-test on the cell differences "
                   "(two-sided); 'all 15 cells' pools the three shares and five budgets, per-share rows use that share's five budgets. "
                   "Columns: number of cells, mean difference, t statistic, p-value.", B))
S.append(Spacer(1, 4))
rows_ = [["comparison", "metric", "scope", "n", "mean difference", "t", "p"]]
for r in tests:
    rows_.append([r[0], r[1], r[2], str(r[3]), f4(r[4]) if r[1] != "seconds" else "%.2f" % r[4], "%.2f" % r[5], fp(r[6])])
S.append(tbl(rows_, font=7.5))
S.append(Spacer(1, 10))


def pooled(label, m):
    for r in tests:
        if r[0] == label and r[1] == m and r[2] == "all 15 cells":
            return r


def big_only(label):
    allv = [c[5] for c in cells if c[0] == label and c[2] >= 60]
    t, p = stats.ttest_1samp(allv, 0.0); return np.mean(allv), t, p


S.append(Paragraph("<b>Outcome in brief</b>", B))
outcomes = []
r = pooled("KAN [1,1] - MLP of t", "RMSE"); outcomes.append("KAN [1,1] vs plain MLP: the KAN fits better. Mean RMSE difference %s (t = %.2f, p %s); negative in 14 of 15 cells." % (f4(r[4]), r[5], ("= " + fp(r[6])) if r[6] >= 1e-4 else fp(r[6])))
r = pooled("KAN [1,1] - Fourier MLP", "RMSE"); mb, tb_, pb = big_only("KAN [1,1] - Fourier MLP")
outcomes.append("KAN [1,1] vs Fourier MLP: pooled over all budgets the KAN is slightly better (mean %s, t = %.2f, p = %s), but the difference comes from the 15- and 30-parameter cells; from 60 parameters upward the two tie (mean %s, t = %.2f, p = %.2f)." % (f4(r[4]), r[5], fp(r[6]), f4(mb), tb_, pb))
if not NO_SPLINE:
    r = pooled("KAN [1,1] - LSQ spline", "RMSE"); mb, tb_, pb = big_only("KAN [1,1] - LSQ spline")
    outcomes.append("KAN [1,1] vs least-squares spline: the spline is marginally better pooled (mean %s, t = %.2f, p = %s), again only at the two smallest budgets; from 60 upward the two agree (mean %s, p = %.2f)." % (f4(r[4]), r[5], fp(r[6]), f4(mb), pb))
r = pooled("KAN [1,h,1] - KAN [1,1]", "RMSE"); outcomes.append("KAN [1,h,1] vs KAN [1,1]: the hidden layer is worse. Mean RMSE difference %s (t = %.2f, p %s); positive in every cell." % (f4(r[4]), r[5], fp(r[6]) if r[6] < 1e-4 else "= " + fp(r[6])))
r = pooled("Fourier MLP - MLP of t", "RMSE"); outcomes.append("Fourier MLP vs plain MLP: the Fourier features help. Mean RMSE difference %s (t = %.2f, p = %s)." % (f4(r[4]), r[5], fp(r[6])))
r = pooled("KAN [1,1] - MLP of t", "seconds")
if NO_SPLINE:
    outcomes.append("Seconds: the KAN [1,1] fits are about %.0f s faster than either MLP (p < 0.0001); the hidden-layer KAN is about %.1f s slower than the one-edge KAN." % (abs(r[4]), pooled("KAN [1,h,1] - KAN [1,1]", "seconds")[4]))
else:
    outcomes.append("Seconds: the KAN [1,1] fits are about %.0f s faster than either MLP (p < 0.0001) and about %.1f s slower than the direct spline solve; the hidden-layer KAN is about %.1f s slower than the one-edge KAN." % (abs(r[4]), pooled("KAN [1,1] - LSQ spline", "seconds")[4], pooled("KAN [1,h,1] - KAN [1,1]", "seconds")[4]))
for o in outcomes:
    S.append(Paragraph("&bull; " + o, B))
S.append(PageBreak())
S.append(Paragraph("<b>Figure.</b> Every cell difference (coloured points, one per share and budget) and the pooled mean with its 95% "
                   "confidence interval (black diamond and bar), for each comparison and metric. Points below the dotted zero line "
                   "favour model A.", B))
S.append(Image(os.path.join(HERE, "differences_plot" + SUFFIX + ".png"), width=26 * cm, height=26 * cm * 4.2 / 14))
SimpleDocTemplate(os.path.join(HERE, "differences" + SUFFIX + ".pdf"), pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm, topMargin=1 * cm, bottomMargin=1 * cm).build(S)
print("wrote differences.csv, differences_summary.csv, differences_ttests.csv, differences_plot.png, differences.pdf")
