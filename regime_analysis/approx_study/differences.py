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
with open(os.path.join(HERE, "differences.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["comparison", "ticker", "budget", "seed", "params_A", "params_B", "rmse_A", "maxerr_A", "sec_A", "rmse_B", "maxerr_B", "sec_B", "d_rmse", "d_maxerr", "d_sec"]); w.writerows(rows)
with open(os.path.join(HERE, "differences_summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["comparison", "ticker", "budget", "params_A", "params_B", "d_rmse_mean", "d_rmse_sd", "n", "d_maxerr_mean", "d_maxerr_sd", "n", "d_sec_mean", "d_sec_sd", "n"]); w.writerows(cells)
with open(os.path.join(HERE, "differences_ttests.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["comparison", "metric", "scope", "n_cells", "mean_difference", "t_statistic", "p_value"]); w.writerows(tests)

# ---- PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
ss = getSampleStyleSheet(); B = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=12)
SM = ParagraphStyle("s", parent=B, fontSize=7.5, leading=10)


def tbl(rows, font=7.5):
    t = Table(rows, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    return t


S = [Paragraph("Approximation study: paired differences between models", ss["Title"]),
     Paragraph("Differences are A - B for RMSE, maximum absolute error and fit time in seconds, per share and parameter budget. "
               "A negative RMSE or max-error difference means model A fits better; a negative seconds difference means A is faster. "
               "Pairing: seed s of A with seed s of B (deterministic models paired with every seed), so each cell is the mean and "
               "standard deviation of three paired differences. All values in the share's Min-Max price scale.", B),
     Spacer(1, 4),
     Paragraph("<b>Caution on the seconds column.</b> Fit times are not architecture-intrinsic: the KANs were fitted with 80 L-BFGS "
               "steps, the MLPs with 20,000 Adam steps (chosen because 4,000 was insufficient), splines by a direct linear solve. "
               "The seconds therefore mostly reflect those training-length choices and the single-CPU machine used, not a fair "
               "cost comparison. RMSE and max-error differences are the ones that bear on the paper.", B),
     Spacer(1, 6)]
for a, b, label in COMPARISONS:
    S.append(Paragraph("<b>%s</b>" % label, B))
    hdr = ["share", "budget", "params A", "params B", "d RMSE (mean +- sd)", "d max|err| (mean +- sd)", "d seconds (mean +- sd)"]
    rows_ = [hdr]
    for c in cells:
        if c[0] != label:
            continue
        rows_.append([c[1], str(c[2]), str(c[3]), str(c[4]), "%+.4f +- %.4f" % (c[5], c[6]), "%+.4f +- %.4f" % (c[8], c[9]), "%+.1f +- %.1f" % (c[11], c[12])])
    S.append(tbl(rows_)); S.append(Spacer(1, 8))
S.append(PageBreak())
S.append(Paragraph("<b>Paired t-tests</b> (one-sample t-test of the cell-mean differences against zero; 'all 15 cells' pools the three "
                   "shares and five budgets, per-share tests use the five budgets). With n = 5 or 15 these are indicative only; cells "
                   "at different budgets are not independent draws from one population, so treat p-values as descriptive.", B))
S.append(Spacer(1, 4))
rows_ = [["comparison", "metric", "scope", "n", "mean difference", "t", "p"]]
for r in tests:
    rows_.append([r[0], r[1], r[2], str(r[3]), "%+.4f" % r[4], "%+.2f" % r[5], "%.4f" % r[6]])
S.append(tbl(rows_))
SimpleDocTemplate(os.path.join(HERE, "differences.pdf"), pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm, topMargin=1 * cm, bottomMargin=1 * cm).build(S)
print("wrote differences.csv, differences_summary.csv, differences_ttests.csv, differences.pdf")
for r in tests:
    if r[2] == "all 15 cells":
        print("%-28s %-10s mean %+.4f  t=%+.2f  p=%.4f" % (r[0], r[1], r[4], r[5], r[6]))
