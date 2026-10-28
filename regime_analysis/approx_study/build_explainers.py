"""
build_explainers.py: three plain-language but rigorous explainer PDFs for the
approximation study. One per model pair:
  explain_splines.pdf   least-squares cubic spline -> free-knot cubic spline
  explain_kans.pdf      KAN [1,1] -> KAN [1,h,1]
  explain_mlps.pdf      MLP of t -> Fourier-feature MLP of t
Every number quoted is read from approx_summary.json / fits at build time.
"""
import glob
import json
import os

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether)

import approx_study as A

HERE = os.path.dirname(os.path.abspath(__file__))
EF = os.path.join(HERE, "explainer_figs")
PF = os.path.join(HERE, "pdf_figs")
S = json.load(open(os.path.join(HERE, "approx_summary.json")))
NAIVE1 = {"CBA.AX": 0.0201, "TSLA": 0.0255, "LLY": 0.0148}


def v(tk, m, b):
    return S["%s|%s|%d" % (tk, m, b)]


def f4(x):
    return "%.4f" % x


# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=18, leading=22, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=4, keepWithNext=1)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=2, keepWithNext=1)
B = ParagraphStyle("B", parent=ss["Normal"], fontSize=10, leading=14, spaceAfter=5)
BOX = ParagraphStyle("BOX", parent=B, backColor=colors.whitesmoke, borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=8)
EQ = ParagraphStyle("EQ", parent=B, fontName="Courier", fontSize=9.5, leading=13, leftIndent=18, spaceAfter=6)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=10)
SM = ParagraphStyle("SM", parent=B, fontSize=8.5, leading=11)


def P(txt, st=B):
    return Paragraph(txt, st)


def img(path, w=16.5):
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    return Image(path, width=w * cm, height=w * cm * ih / iw)


def fig(name, caption, w=16.5, folder=EF):
    return KeepTogether([img(os.path.join(folder, name), w), P(caption, CAP)])


def table(rows, col_widths=None, font=8.5):
    tb = Table(rows, hAlign="LEFT", colWidths=col_widths, repeatRows=1)
    tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return tb


# ---------------------------------------------------------------- shared sections
def shared_intro(story, title, pair_sentence):
    story += [P(title, H1),
              P("A plain-language but rigorous account of one experiment: %s. Written for a reader with no background in "
                "computer science, statistics or higher mathematics. Every number is taken from the saved experimental "
                "output; every formula is stated in full." % pair_sentence, B)]
    story += [P("1. The question", H2),
              P("Take the daily closing price of a share over three and a half years. It is a jagged line with about 900 points. "
                "Suppose you may describe that line using only a small list of numbers, say 15, or 60, or 240, and someone else "
                "must redraw the line from your list alone. How close can the redrawn line get? And how much closer does it get "
                "each time you are allowed more numbers? That is the whole experiment. In mathematics it is called "
                "<b>function approximation</b>: finding a simple formula whose graph stays close to a complicated one.", B),
              P("This is deliberately <b>not</b> forecasting. Nobody is asked what the price will be tomorrow. The entire series, "
                "past and future, is on the table from the start. The earlier part of this project showed that no model we tried "
                "could forecast prices better than simply repeating yesterday's price, because day-to-day price changes are close to "
                "unpredictable. Approximation sidesteps that: it asks only how well a given kind of formula can <i>describe</i> what "
                "has already happened, which is a question about the formula, not about the market.", B),
              P("<b>Why it matters.</b> If one kind of formula describes price paths far more compactly than another, that tells us "
                "something about the formula's expressive power that forecasting experiments cannot, because forecasting results are "
                "dominated by the unpredictability of the target rather than by the model. It also tells us how 'rough' price data is, "
                "which turns out to be the deepest finding here.", B)]
    story += [P("2. The data", H2),
              P("Three shares, chosen to be as different as possible: Commonwealth Bank (CBA.AX, an Australian bank, the series the "
                "original paper used), Tesla (TSLA, very volatile) and Eli Lilly (LLY, a steady upward trend). Daily closing prices "
                "from 2 January 2020 to 31 July 2023: 905 trading days for CBA, 901 for the two US shares.", B),
              P("Two rescalings make the three series comparable. Time is rescaled so that the first day is t = 0 and the last day is "
                "t = 1. Price is rescaled so that the lowest close in the period is 0 and the highest is 1 (this is called Min-Max "
                "scaling: p_scaled = (p - p_min) / (p_max - p_min)). After this, every series is a set of about 900 points (t_i, p_i) "
                "with both coordinates between 0 and 1. An error of 0.01 therefore means one hundredth of the share's whole price "
                "range over the period.", B),
              fig("01_series_and_curve.png", "Figure 1. The CBA series after rescaling (grey), and one smooth curve that tries to follow "
                  "it using only 15 numbers (red). The curve catches the broad movements and misses the fine ones. Everything in "
                  "this document is about making curves like the red one, with different kinds of formula and different numbers of parameters.")]


def shared_math(story):
    story += [P("3. The mathematics we need", H2),
              P("3.1 Functions and parameters", H3),
              P("A <b>function</b> is a rule that turns an input into an output. Here the input is a time t between 0 and 1 and "
                "the output is a price. We write f(t). A <b>family</b> of functions is a formula with blanks in it; the numbers that "
                "fill the blanks are the <b>parameters</b>. For example f(t) = a + b t + c t^2 is a family with three parameters "
                "a, b, c; choosing values for them picks one parabola out of the family. 'Describing the price line with 60 numbers' "
                "means picking the member of a 60-parameter family that sits closest to the data.", B),
              P("3.2 Measuring closeness: the error and the RMSE", H3),
              P("For each day i we have the real price p_i and the curve's value f(t_i). Their difference e_i = f(t_i) - p_i is that "
                "day's <b>error</b>: positive if the curve is above the price, negative if below. One number summarises all the "
                "errors, the <b>root-mean-square error</b>:", B),
              P("RMSE = sqrt( (1/n) * sum over i of e_i^2 )        (n = number of days)", EQ),
              P("In words: square every error (so positives and negatives do not cancel and big errors count more), average the "
                "squares, take the square root to get back to price units. A second number, the <b>maximum absolute error</b> "
                "max |e_i|, records the single worst day. Both are reported for every fit.", B),
              fig("02_error_and_rmse.png", "Figure 2. Top: a stretch of the price (grey) and a curve (red); each black bar is one "
                  "day's error. Bottom: the same errors drawn as a wave, red where the curve is above the price, blue where below. "
                  "The RMSE is the typical height of this wave. The result PDFs draw this wave for every fit, with a black envelope "
                  "showing the 20-day rolling RMSE, i.e. the local typical error with the day-to-day jitter smoothed out."),
              P("3.3 'Least squares'", H3),
              P("Among all members of a family, the one we want is the one with the smallest RMSE. Because the RMSE is built from "
                "squared errors, this is called the <b>least-squares fit</b>. For some families (splines) the least-squares fit can "
                "be computed exactly by solving a system of linear equations; for others (neural networks) it must be searched for "
                "by trial and improvement, which is what 'training' means, and the search may stop at a good-but-not-best answer.", B),
              P("3.4 Comparing fairly: parameter budgets", H3),
              P("A formula with more parameters can always fit at least as well, so comparing two formulas is only meaningful at the "
                "<b>same number of parameters</b>. We therefore fixed five budgets, about 15, 30, 60, 120 and 240 parameters, set "
                "each model up to use as close to the budget as its structure allows, and recorded the exact count actually used. "
                "The plots put the true count on the horizontal axis.", B),
              P("3.5 Rate of improvement, and why we use logarithmic axes", H3),
              P("The most informative quantity is not any single error but how fast the error shrinks as the parameter count N grows. "
                "For many approximation methods the error follows a <b>power law</b>, error = C * N^(-a), where a is the <b>rate</b>. "
                "Taking logarithms turns this into log(error) = log C - a * log N, a straight line whose slope is -a. So on axes where "
                "both the parameter count and the error are drawn logarithmically, a straight line means a power law and its "
                "steepness is the rate: slope -1 means doubling the parameters halves the error; slope -0.5 means doubling the "
                "parameters only cuts the error by a factor of 1.41 (the square root of 2). We fit this slope to each model's results.", B),
              fig("10_power_law.png", "Figure 3. The CBA results on logarithmic axes. Each model's points lie roughly on a straight "
                  "line; the dashed reference line has slope -0.5 (error proportional to 1 / sqrt(N)). The slope of each model is "
                  "listed in the results section.", w=11),
              P("3.6 How rough is a price line? (the key background fact)", H3),
              P("Zoom into a smooth curve and it eventually looks like a straight line. Zoom into a price line and it never does: at "
                "every magnification it is still jagged (Figure 4). Mathematicians measure this with the <b>Holder exponent</b>: a "
                "function is Holder-continuous of order alpha if |f(x) - f(y)| is at most a constant times |x - y|^alpha for all "
                "nearby x, y. A smooth curve has alpha = 1 (it is 'Lipschitz'); the path of a random walk, which is the standard "
                "model of daily prices, has alpha just below 1/2 and no higher (this is a theorem about Brownian motion; see Morters "
                "and Peres, <i>Brownian Motion</i>, Chapter 1). Classical approximation theory (de Boor, <i>A Practical Guide to "
                "Splines</i>) then says that any piecewise-smooth approximation with N pieces of a Holder-alpha function has error of "
                "order N^(-alpha). For a price path that predicts a rate of about 0.5: <b>doubling the parameters can only reduce the "
                "error by about 30 %, whatever the architecture</b>. Keep this prediction in mind; the results test it.", B),
              fig("11_roughness.png", "Figure 4. Top row: the CBA price line at three magnifications; it stays jagged. Bottom row: a "
                  "smooth curve at the same magnifications; it straightens out. This difference is what limits every approximator "
                  "in this study.")]


def shared_setup(story, extra_rows):
    story += [P("5. The experimental setup", H2),
              P("For each of the three shares, each model and each of the five parameter budgets, the model was fitted to the whole "
                "series by least squares (exactly for splines, by numerical optimisation for the networks). Models whose fitting "
                "involves randomness (random starting values, random frequencies) were run three times with different random seeds "
                "(0, 1, 2) and the mean and spread are reported. In total 210 fits. For every fit we saved the fitted value on every "
                "day, the RMSE, the maximum error, the parameter count, the wall-clock time and the settings.", B),
              P("Settings per budget for the two models in this document:", B),
              table(extra_rows),
              Spacer(1, 6),
              P("Software: Python 3.10, NumPy/SciPy for splines, pykan 0.2.8 for KANs, PyTorch 2.2 for MLPs; single CPU. Code: "
                "approx_study.py (fits), aggregate_approx.py (tables and figures), build_pdfs.py (result PDFs).", SM)]


def results_table(story, models, labels):
    rows = [["share", "budget"] + sum(([labels[m] + " RMSE", "max err", "params"] for m in models), [])]
    for tk in A.TICKERS:
        for b in A.BUDGETS:
            row = [tk, str(b)]
            for m in models:
                r = v(tk, m, b)
                row += [f4(r["rmse_mean"]) + (" +- " + f4(r["rmse_sd"]) if r["n"] > 1 else ""), f4(r["max_err_mean"]), str(r["params"])]
            rows.append(row)
    story += [table(rows, font=7.5), P("Table 1. RMSE (mean +- standard deviation over three seeds where the model is random) and mean "
                                       "maximum error, in the share's scaled price units, with the exact parameter count. For scale, the "
                                       "one-day-ahead naive forecast error of the same series is 0.0201 (CBA), 0.0255 (TSLA), 0.0148 (LLY).", CAP)]


def slopes(models):
    out = {}
    for m in models:
        sl = []
        for tk in A.TICKERS:
            x = [np.log(v(tk, m, b)["params"]) for b in A.BUDGETS[1:]]
            yv = [np.log(v(tk, m, b)["rmse_mean"]) for b in A.BUDGETS[1:]]
            sl.append(np.polyfit(x, yv, 1)[0])
        out[m] = sl
    return out


def shared_limits(story, extra):
    story += [P("9. What this experiment cannot tell you (limitations)", H2)]
    for s in ["Everything is in-sample: the curve is scored on the very data it was fitted to. This is the right thing for an "
              "approximation study, but it says nothing about predicting prices that have not happened yet.",
              "Three shares and one period. The conclusions about the <i>models</i> are consistent across all three, but the "
              "numbers themselves would differ on other shares or periods.",
              "Parameter counts are matched approximately (within about 10 %), not exactly.",
              "Three random seeds; no formal significance test was applied. Where the spread across seeds is reported, treat "
              "differences smaller than the spread as not established."] + extra:
        story.append(P("&bull; " + s, B))


def glossary(story, extra):
    story += [P("10. Glossary", H2)]
    base = [("Function", "a rule turning an input (here, time t) into an output (here, a price)."),
            ("Parameter", "a number inside a formula that is chosen to make the formula fit the data."),
            ("Least squares", "choosing the parameters so that the sum of squared errors is as small as possible."),
            ("RMSE", "root-mean-square error: square the daily errors, average, take the square root; the typical error size."),
            ("Min-Max scaling", "rescaling so the smallest value becomes 0 and the largest becomes 1."),
            ("Power law / rate", "error proportional to N^(-a); a is the rate; on log-log axes a straight line of slope -a."),
            ("Holder exponent", "a measure of roughness: how fast |f(x)-f(y)| shrinks as x and y approach; 1 = smooth, 1/2 = random-walk rough."),
            ("Seed", "the starting value of the random number generator; different seeds give different random starts.")]
    rows = [["term", "meaning"]] + [[Paragraph(a, SM), Paragraph(b, SM)] for a, b in base + extra]
    story.append(table(rows, col_widths=[4 * cm, 12.5 * cm]))


# ---------------------------------------------------------------- SPLINES document
def doc_splines():
    st = []
    shared_intro(st, "Describing a price line with splines: fixed knots and free knots",
                 "fitting cubic splines to three share-price series, first with equally spaced knots and then with knots that are allowed to move")
    shared_math(st)
    st += [P("4. The two models", H2),
           P("4.1 What a spline is", H3),
           P("A <b>spline</b> is a curve made of polynomial pieces joined end to end. Split the time axis [0, 1] into G equal "
             "<b>intervals</b>; the G - 1 interior boundary points are the <b>knots</b>. On each interval the curve is a cubic "
             "polynomial (a + b t + c t^2 + d t^3), and at every knot the pieces are joined so that the curve, its slope and its "
             "curvature all match (in symbols: f, f' and f'' are continuous). That smoothness at the joins is what makes a spline look "
             "like one curve rather than a chain of segments.", B),
           P("Rather than writing the polynomial of each piece separately, a cubic spline is written as a weighted sum of fixed "
             "'bump' functions called <b>B-splines</b>: f(t) = sum over j of c_j B_j(t). There are G + 3 bumps for G intervals, each "
             "bump is non-zero on only four neighbouring intervals, and the bumps are the same for every data set; only the weights "
             "c_j (the <b>coefficients</b>) are fitted. The coefficients are the model's parameters: a fixed-knot cubic spline on G "
             "intervals has exactly G + 3 of them.", B),
           fig("03_bspline_basis.png", "Figure 5. Top: the nine B-spline bumps for six intervals (knots dotted). Bottom: scaling each "
               "bump by its fitted coefficient and adding them up gives the spline (red) that best follows the CBA series with nine numbers."),
           P("Because f(t) is a <i>linear</i> combination of known bumps, the least-squares coefficients are found exactly by solving "
             "G + 3 linear equations; there is no searching and no randomness. This is why the spline rows in the results have no "
             "'+-' spread: the same input always gives the same answer.", B),
           P("4.2 Model A: least-squares cubic spline with fixed, equally spaced knots", H3),
           P("Knots at t = 1/G, 2/G, ..., (G-1)/G. Parameters = G + 3 coefficients. Budgets 15, 30, 60, 120, 240 use G = 12, 27, 57, "
             "117, 237 intervals. Implementation: scipy.interpolate.LSQUnivariateSpline with k = 3.", B),
           P("4.3 Model B: free-knot cubic spline", H3),
           P("Identical, except that the G - 1 interior knot positions are themselves chosen to minimise the RMSE. Each knot position "
             "is one more parameter, so the total is (G + 3) + (G - 1) = 2G + 2. To hit the same budgets the free-knot spline must use "
             "fewer intervals: G = 7, 15, 30, 60, 120 (giving 16, 32, 62, 122, 242 parameters).", B),
           P("Finding the best knot positions is a genuinely hard search: the RMSE as a function of the knot positions has many "
             "local minima, and the usual linear-equation trick no longer applies. We used 12 random starting arrangements, each "
             "improved first by a gradient method (L-BFGS-B) and then by a derivative-free method (Nelder-Mead), with the coefficients "
             "re-solved exactly by least squares every time the knots move, and kept the best of the 12. Knots were kept at least four "
             "data points apart so that the linear equations stay solvable.", B),
           P("4.4 The one thing that differs", H3),
           P("Whether the knots may move. The hope behind free knots is that a price line has a few 'important moments' (a crash, a "
             "turning point) where a knot is worth far more than elsewhere, so that cleverly placed knots beat many evenly spaced ones. "
             "Figure 6 shows the idea on a toy function with exactly that structure.", B),
           fig("04_fixed_vs_free_knots.png", "Figure 6. A piecewise-linear toy function with kinks at 3, 5, 6, 7 (grey). Left: a linear "
               "spline with knots at 2, 4, 6, 8 cannot reproduce it (max error 1.76). Right: with knots moved to the kinks the "
               "reproduction is exact. Free knots are powerful when the function has a few special places. The question is whether "
               "a price line is like that.")]
    rows = [["budget"] + [str(b) for b in A.BUDGETS],
            ["fixed-knot: intervals G / parameters"] + ["%d / %d" % (A.LSQ_G[b], A.LSQ_G[b] + 3) for b in A.BUDGETS],
            ["free-knot: intervals G / parameters"] + ["%d / %d" % (A.FKS_G[b], 2 * A.FKS_G[b] + 2) for b in A.BUDGETS]]
    shared_setup(st, rows)
    st += [PageBreak(), P("6. Results", H2)]
    results_table(st, ["lsq", "fks"], {"lsq": "fixed-knot", "fks": "free-knot"})
    st += [fig("splines_rmse.png", "Figure 7. Error against parameter count for the two splines on the three shares (log-log). The "
               "dotted horizontal line is the one-day naive forecast error, for scale only.", folder=PF)]
    sl = slopes(["lsq", "fks"])
    st += [P("Rates of improvement (slope of log RMSE against log parameters, budgets 30 to 240): fixed-knot %s (mean %+.2f); "
             "free-knot %s (mean %+.2f), for CBA, TSLA, LLY respectively." % (
                 ", ".join("%+.2f" % s for s in sl["lsq"]), np.mean(sl["lsq"]), ", ".join("%+.2f" % s for s in sl["fks"]), np.mean(sl["fks"])), B)]
    lsq60, fks60 = v("CBA.AX", "lsq", 60)["rmse_mean"], v("CBA.AX", "fks", 60)["rmse_mean"]
    lsq240, fks240 = v("CBA.AX", "lsq", 240)["rmse_mean"], v("CBA.AX", "fks", 240)["rmse_mean"]
    st += [P("7. Reading the results", H2),
           P("<b>Both splines work, and improve steadily.</b> On CBA the fixed-knot spline's error falls from %s with 15 numbers to %s with "
             "240; the worst single day falls from %s to %s. By 240 parameters the typical error is below the day-to-day noise "
             "level of the series (0.020), i.e. the curve is tracking the price to within the size of an ordinary daily move." % (
                 f4(v("CBA.AX", "lsq", 15)["rmse_mean"]), f4(lsq240), f4(v("CBA.AX", "lsq", 15)["max_err_mean"]), f4(v("CBA.AX", "lsq", 240)["max_err_mean"])), B),
           P("<b>Free knots do not pay for themselves on this data.</b> At the same parameter count the free-knot spline is slightly worse "
             "on every share at every budget from 30 upward (CBA at 60: %s vs %s; at 240: %s vs %s), and its rate of improvement is "
             "lower (%+.2f vs %+.2f). This is not because the knot search failed: given the <i>same number of intervals</i>, moving the "
             "knots does help (CBA, 30 intervals: RMSE 0.0353 fixed vs 0.0275 free). It is because each moved knot costs a parameter, "
             "and on a price line that parameter buys more when it is spent on an extra interval than on better placement. In the "
             "toy example of Figure 6 the function had four special places and nothing else; a price line is rough everywhere, so "
             "there are no special places worth paying for. The knot-position maps in the result PDF show the optimiser clustering "
             "knots around the 2020 crash and the 2022 swings and leaving the rest coarse, which is the best it can do." % (
                 f4(fks60), f4(lsq60), f4(fks240), f4(lsq240), np.mean(sl["fks"]), np.mean(sl["lsq"])), B),
           P("<b>The rate is the one predicted by roughness.</b> The fixed-knot spline improves with slope about %+.2f, i.e. error "
             "proportional to 1 / sqrt(N). That is exactly the N^(-1/2) rate that Section 3.6 predicted for a function with Holder "
             "exponent 1/2. The spline is not being held back by anything about splines; it is being held back by the data." % np.mean(sl["lsq"]), B),
           fig("splines_CBA.AX_lsq_fits0.png", "Figure 8. CBA, fixed-knot spline at 15, 30 and 60 parameters: the fit (left) and the "
               "error wave with its rolling-RMSE envelope (right). As the budget rises the envelope narrows and what remains is "
               "fine, fast jitter with no slow structure: the model has captured everything it can.", folder=PF),
           P("8. What it means", H2),
           P("For someone wanting to describe a price history compactly, a plain cubic spline with evenly spaced knots is as good "
             "as the cleverer free-knot version, is instantaneous to compute, and has no randomness. The deeper lesson is that the "
             "error curve of the spline is a measurement of the data's roughness, and it comes out at the random-walk value. Any "
             "other method, however sophisticated, is competing against this benchmark; the other two documents in this series "
             "show that the best neural methods reach it and none beat it. Free-knot methods shine on functions with a few sharp "
             "features (the S1/S2 example in the earlier report, where they recovered the knots exactly); a price path is the "
             "opposite kind of function.", B)]
    shared_limits(st, ["The free-knot search is non-convex; 12 starts were used, so 'did not win' means 'did not win with this optimiser'."])
    glossary(st, [("Spline", "a curve made of polynomial pieces joined smoothly at knots."),
                  ("Knot", "a point where two polynomial pieces of a spline meet."),
                  ("B-spline", "one of the fixed 'bump' functions whose weighted sum gives a spline."),
                  ("Coefficient", "the weight of one bump; the spline's parameters."),
                  ("Free-knot", "a spline whose knot positions are optimised as well as its coefficients."),
                  ("Non-convex", "an optimisation landscape with many valleys, where a search can get stuck in the wrong one.")])
    return st


# ---------------------------------------------------------------- KAN document
def doc_kans():
    st = []
    shared_intro(st, "Describing a price line with Kolmogorov-Arnold Networks: one edge versus a hidden layer",
                 "fitting KANs to three share-price series, first with a single learnable edge and then with a hidden layer")
    shared_math(st)
    st += [P("4. The two models", H2),
           P("4.1 What a KAN is", H3),
           P("A neural network is a diagram of nodes and edges through which a number flows from input to output. In the familiar "
             "kind (an MLP, the subject of the third document) each edge multiplies the number by a weight and each node applies a "
             "fixed squashing function. A <b>Kolmogorov-Arnold Network</b> (KAN; Liu et al., 2024, arXiv:2404.19756) turns this "
             "inside out: <b>each edge carries its own learnable function</b> phi, and nodes do nothing but add up what arrives. "
             "There are no weights and no fixed squashing functions. The name comes from a theorem of Kolmogorov and Arnold (1957) "
             "stating that any continuous function of several variables can be written exactly as sums of functions of one variable, "
             "which is what such a network computes.", B),
           P("In the implementation used here (pykan 0.2.8) every edge function has the form", B),
           P("phi(x) = w_b * silu(x) + w_s * s(x),      silu(x) = x / (1 + e^(-x)),      s(x) = sum_j c_j B_j(x)", EQ),
           P("that is, a small fixed 'base' term (silu, a smooth ramp) plus a <b>cubic B-spline</b> s(x) on G intervals, exactly the "
             "object described in the splines document, with G + 3 learnable coefficients c_j and two learnable scales w_b, w_s. "
             "So a KAN edge is a spline with a ramp added. The parameters are fitted by a numerical optimiser (L-BFGS, 80 steps), "
             "not by solving linear equations, because the scales multiply the spline and make the problem non-linear.", B),
           fig("05_kan_edge.png", "Figure 5. The three ingredients of one KAN edge: the fixed ramp silu(x) (left), a cubic B-spline "
               "on 3 intervals with its 6 coefficients (middle), and their weighted sum, the edge function phi(x) (right)."),
           P("4.2 Model A: KAN [1,1], one edge", H3),
           P("One input node (t), one output node (the price), one edge between them. The whole model is therefore a single edge "
             "function: f(t) = w_b silu(t) + w_s s(t). With the spline on G intervals this has G + 9 parameters (G + 3 coefficients, "
             "the two scales, and four fixed-structure terms pykan always includes). Budgets 15 to 240 use G = 6, 21, 51, 111, 231.", B),
           P("4.3 Model B: KAN [1,h,1], a hidden layer", H3),
           P("Between input and output sits a layer of h nodes. There are h edges from t to the hidden nodes and h edges from them "
             "to the output: 2h learnable functions in all. The output is", B),
           P("f(t) = sum over k of psi_k( phi_k(t) ),      k = 1 .. h", EQ),
           P("a sum of h <i>compositions</i>: first phi_k reshapes t, then psi_k reshapes the result. Each edge's spline has only "
             "G = 3 intervals (pykan's default), so each is coarse; the hope is that composing coarse functions creates fine detail. "
             "Parameters are about 24 per hidden node; h = 1, 1, 2, 5, 10 give 24, 24, 48, 120, 240 (the two smallest budgets cannot "
             "be met, since even h = 1 costs 24). This model has random initial values, so three seeds were run.", B),
           fig("06_architectures.png", "Figure 6. Left: KAN [1,1], one learnable function on one edge. Middle: KAN [1,3,1], six "
               "learnable functions and three adding nodes. Right, for contrast: an MLP, where edges are numbers and nodes apply the "
               "fixed function tanh (see the third document)."),
           P("4.4 The one thing that differs", H3),
           P("Depth. Model A puts all its parameters into the resolution of one spline (many intervals on one edge). Model B puts "
             "them into more edges with coarse splines and relies on composition. Same edge function, same optimiser, same data; only "
             "the wiring changes.", B)]
    rows = [["budget"] + [str(b) for b in A.BUDGETS],
            ["KAN [1,1]: grid G / parameters"] + ["%d / %d" % (A.KAN11_G[b], A.KAN11_G[b] + 9) for b in A.BUDGETS],
            ["KAN [1,h,1]: hidden h / parameters"] + ["%d / %d" % (A.KAN1H1_H[b], v("CBA.AX", "kan1h1", b)["params"]) for b in A.BUDGETS]]
    shared_setup(st, rows)
    st += [PageBreak(), P("6. Results", H2)]
    results_table(st, ["kan11", "kan1h1"], {"kan11": "KAN [1,1]", "kan1h1": "KAN [1,h,1]"})
    st += [fig("kans_rmse.png", "Figure 7. Error against parameter count for the two KANs on the three shares (log-log). The grey dashed "
               "line is the classical fixed-knot spline from the first document, for reference; the dotted horizontal line is the "
               "one-day naive forecast error.", folder=PF)]
    sl = slopes(["kan11", "kan1h1"])
    st += [P("Rates of improvement (slope of log RMSE against log parameters, budgets 30 to 240): KAN [1,1] %s (mean %+.2f); "
             "KAN [1,h,1] %s (mean %+.2f), for CBA, TSLA, LLY." % (
                 ", ".join("%+.2f" % s for s in sl["kan11"]), np.mean(sl["kan11"]), ", ".join("%+.2f" % s for s in sl["kan1h1"]), np.mean(sl["kan1h1"])), B)]
    k60, h60 = v("CBA.AX", "kan11", 60)["rmse_mean"], v("CBA.AX", "kan1h1", 60)["rmse_mean"]
    k240, h240 = v("CBA.AX", "kan11", 240)["rmse_mean"], v("CBA.AX", "kan1h1", 240)["rmse_mean"]
    st += [P("7. Reading the results", H2),
           P("<b>The one-edge KAN is, in effect, a spline, and behaves like one.</b> Its error on CBA falls from %s (15 parameters) "
             "to %s (240), and at every budget from 60 upward it agrees with the classical least-squares spline to three or four "
             "decimals (CBA at 240: %s vs 0.0131; LLY at 240: %s vs 0.0081). The small gap at 15 and 30 parameters is bookkeeping: "
             "pykan spends nine of the budget on scales and fixed terms, so its spline has fewer intervals (6 against 12 at budget "
             "15). It is deterministic to four decimals across seeds. Its rate, %+.2f, is the roughness-limited N^(-1/2) rate." % (
                 f4(v("CBA.AX", "kan11", 15)["rmse_mean"]), f4(k240), f4(k240), f4(v("LLY", "kan11", 240)["rmse_mean"]), np.mean(sl["kan11"])), B),
           P("<b>Adding a hidden layer makes the KAN worse, at every budget, on every share.</b> At 60 parameters on CBA the hidden-layer "
             "model's error is %s against %s for the one-edge model (%.1f times larger); at 240 it is %s against %s (%.1f times). Its "
             "rate of improvement is only %+.2f: extra parameters buy very little. The fits (Figure 8) show why: the hidden-layer "
             "model draws the broad shape and misses the medium-scale swings that the one-edge model captures." % (
                 f4(h60), f4(k60), h60 / k60, f4(h240), f4(k240), h240 / k240, np.mean(sl["kan1h1"])), B),
           P("<b>Why depth hurts here.</b> Each edge in the deep model is a spline on only 3 intervals: a curve with at most a couple "
             "of bends. Composing two such curves gives a curve with a few more bends, but still only a handful; it cannot manufacture "
             "the dozens of independent wiggles a price line has. The parameters go into more edges, not into resolution, and "
             "resolution is what a rough function needs. (This is a derivation from the architecture, consistent with the data; it "
             "is not a theorem.) The KAN paper's own successes with depth are on functions that have a genuine compositional "
             "structure, such as physics formulas, where composing simple pieces is exactly right.", B),
           fig("kans_CBA.AX_kan1h1_fits0.png", "Figure 8. CBA, KAN [1,h,1] at 24, 24 and 48 parameters (h = 1, 1, 2): the fit (left) and "
               "the error wave with rolling-RMSE envelope (right). The envelope stays wide and the error has slow red/blue bands, "
               "meaning the model is missing whole swings, not just daily jitter. Compare with the one-edge panels in the result PDF.", folder=PF),
           P("8. What it means", H2),
           P("A KAN with one edge is a good approximator of a price line because it <i>is</i> a spline; it inherits the spline's "
             "exact rate and has no advantage over it. The 'learnable activation function' that distinguishes KANs from ordinary "
             "networks is, on a single edge, precisely the classical object numerical analysts have used for sixty years, and the "
             "experiment confirms the identity numerically. Depth, which is where KANs are claimed to differ from splines, reduces "
             "accuracy for this task. If a case is to be made for KANs in describing price data, it cannot rest on accuracy per "
             "parameter; it would have to rest on properties not measured here, such as the readability of the fitted edge "
             "functions or the possibility of pruning them to a symbolic formula. And the error floor the one-edge KAN reaches is "
             "the N^(-1/2) roughness limit of the data, which no architecture can lower.", B)]
    shared_limits(st, ["Only one grid size (G = 3) was used for the hidden-layer model; a deeper KAN with finer edge splines was not tried "
                       "and might close part of the gap.",
                       "pykan's optimiser is not bit-reproducible; differences of about 0.001 in RMSE between seeds are noise."])
    glossary(st, [("KAN", "Kolmogorov-Arnold Network: a network whose edges carry learnable functions and whose nodes add."),
                  ("Edge function phi", "the learnable one-variable function on a KAN edge: a spline plus a silu ramp."),
                  ("silu", "the smooth ramp x / (1 + e^-x); pykan's fixed base term on every edge."),
                  ("Grid G", "the number of intervals in an edge's spline."),
                  ("Hidden layer", "a row of nodes between input and output; its size h sets the number of edges."),
                  ("Composition", "feeding the output of one function into another: psi(phi(t))."),
                  ("L-BFGS", "a numerical optimiser used to fit the KAN's parameters.")])
    return st


# ---------------------------------------------------------------- MLP document
def doc_mlps():
    st = []
    shared_intro(st, "Describing a price line with multilayer perceptrons: plain input versus Fourier features",
                 "fitting small neural networks to three share-price series, first with time as the raw input and then with sines and cosines of time as the input")
    shared_math(st)
    st += [P("4. The two models", H2),
           P("4.1 What an MLP is", H3),
           P("A <b>multilayer perceptron</b> (MLP) is the standard neural network. Numbers flow left to right through layers of nodes. "
             "Each edge multiplies the number it carries by a <b>weight</b>; each node adds up its incoming numbers, adds a <b>bias</b>, "
             "and applies a fixed squashing function, here tanh(x) = (e^x - e^-x)/(e^x + e^-x), a smooth step from -1 to +1. The "
             "weights and biases are the parameters. With input t, two hidden layers of w nodes each, and one output, the formula is", B),
           P("f(t) = W3 . tanh( W2 . tanh( W1 t + b1 ) + b2 ) + b3", EQ),
           P("where W1 is a column of w weights, W2 a w-by-w block, W3 a row of w weights, and b1, b2, b3 the biases. Counting: "
             "w + w^2 + w weights and w + w + 1 biases, total w^2 + 4w + 1. Widths w = 2, 4, 6, 9, 14 give 13, 33, 61, 118, 253 "
             "parameters for the five budgets. The universal approximation theorem (Cybenko 1989; Hornik 1991) guarantees that some "
             "MLP can get arbitrarily close to any continuous function; it says nothing about how large it must be or whether "
             "training will find it.", B),
           fig("12_mlp_params.png", "Figure 5. The MLP with w = 2, with every weight (edge) and bias (node) counted: 13 parameters.", w=13),
           fig("07_tanh.png", "Figure 6. Left: the tanh squashing function. Right: a sum of a few scaled, shifted tanh steps. Such sums "
               "make smooth bends easily but fine wiggles only with difficulty: each wiggle needs its own steep, precisely placed step."),
           P("Training: the parameters start at random values and are adjusted step by step to reduce the RMSE (the Adam optimiser, "
             "20,000 steps, with a decreasing step size). Because the start is random, three seeds were run. An important practical "
             "point found during this work: 4,000 steps were not enough (RMSE 0.074 on CBA at 61 parameters) and 20,000 gave 0.048; "
             "all results below use 20,000.", B),
           P("4.2 Model A: MLP of t", H3),
           P("The network above with the raw time t as its single input.", B),
           P("4.3 Model B: Fourier-feature MLP of t", H3),
           P("The <i>identical</i> network, but before t enters it is replaced by a list of sines and cosines of t:", B),
           P("gamma(t) = [ sin(2 pi b_1 t), ..., sin(2 pi b_m t), cos(2 pi b_1 t), ..., cos(2 pi b_m t) ]", EQ),
           P("with m frequencies b_1 .. b_m drawn once at random from a normal distribution with standard deviation 6 (so typical "
             "frequencies are a few cycles over the period) and then <b>frozen</b>: they are not parameters. The first layer now has "
             "2m inputs instead of one, so the count becomes w^2 + (2m + 3) w + 1, and w is reduced to keep the budget: (m, w) = "
             "(2, 2), (2, 3), (4, 4), (6, 6), (8, 9) give 19, 31, 61, 127, 253 parameters.", B),
           fig("08_fourier_features.png", "Figure 7. Top: three of the sine features at different frequencies. Bottom: a random weighted "
               "sum of eight such features already wiggles on the scale of a price line; the network only has to combine and shape them."),
           P("4.4 The one thing that differs, and why it was expected to matter", H3),
           P("Only the input basis changes. The reason to expect a large effect is a theorem about how MLPs learn, known as "
             "<b>spectral bias</b> (Tancik et al., NeurIPS 2020, arXiv:2006.10739, building on Rahaman et al. 2019): when an MLP is "
             "trained by gradient descent on a low-dimensional input such as t, it fits the slowly varying components of the target "
             "quickly and the rapidly varying ones extremely slowly, in proportion to a rapidly decreasing function of frequency. "
             "A price line is mostly rapidly varying components. The Fourier map hands the network those components ready-made, so "
             "it no longer has to construct them from tanh steps. Tancik et al. show this turns the network's effective behaviour "
             "into that of a tunable-bandwidth interpolator.", B)]
    rows = [["budget"] + [str(b) for b in A.BUDGETS],
            ["MLP: width w / parameters"] + ["%d / %d" % (A.MLP_W[b], v("CBA.AX", "mlp", b)["params"]) for b in A.BUDGETS],
            ["Fourier MLP: (m, w) / parameters"] + ["(%d, %d) / %d" % (A.FMLP[b][0], A.FMLP[b][1], v("CBA.AX", "fmlp", b)["params"]) for b in A.BUDGETS]]
    shared_setup(st, rows)
    st += [PageBreak(), P("6. Results", H2)]
    results_table(st, ["mlp", "fmlp"], {"mlp": "MLP of t", "fmlp": "Fourier MLP"})
    st += [fig("mlps_rmse.png", "Figure 8. Error against parameter count for the two MLPs on the three shares (log-log). Grey dashed: "
               "the classical fixed-knot spline, for reference; dotted horizontal: one-day naive forecast error.", folder=PF)]
    sl = slopes(["mlp", "fmlp"])
    st += [P("Rates of improvement (slope of log RMSE against log parameters, budgets 30 to 240): MLP %s (mean %+.2f); Fourier MLP "
             "%s (mean %+.2f), for CBA, TSLA, LLY." % (
                 ", ".join("%+.2f" % s for s in sl["mlp"]), np.mean(sl["mlp"]), ", ".join("%+.2f" % s for s in sl["fmlp"]), np.mean(sl["fmlp"])), B)]
    m60, f60 = v("CBA.AX", "mlp", 60)["rmse_mean"], v("CBA.AX", "fmlp", 60)["rmse_mean"]
    m240, f240 = v("CBA.AX", "mlp", 240)["rmse_mean"], v("CBA.AX", "fmlp", 240)["rmse_mean"]
    st += [P("7. Reading the results", H2),
           P("<b>The plain MLP is the weakest approximator in the whole study.</b> On CBA it reaches only %s with 61 parameters and "
             "%s with 253, against 0.0260 and 0.0131 for a spline of the same size; its rate of improvement is %+.2f, so quadrupling "
             "its size barely moves it. Figure 9 shows the characteristic failure: the curve follows the slow trend and simply does "
             "not attempt the swings. The error wave has long red and blue bands, meaning whole months are missed." % (
                 f4(m60), f4(m240), np.mean(sl["mlp"])), B),
           P("<b>Fourier features fix it completely.</b> With the identical network and the sine/cosine input, the error on CBA is %s at "
             "61 parameters and %s at 253: indistinguishable from the spline (0.0260, 0.0131) and the one-edge KAN. The same holds on "
             "TSLA and LLY. Its rate, %+.2f, is the steepest of any model in the study. At the two smallest budgets it is erratic "
             "(spread across seeds up to 0.04) because with only two or three random frequencies the result depends on which ones "
             "were drawn; from four frequencies upward it is stable." % (f4(f60), f4(f240), np.mean(sl["fmlp"])), B),
           fig("09_spectral_bias.png", "Figure 9. CBA at about 60 parameters: the plain MLP (left) draws the trend and ignores the swings; "
               "the same network with Fourier features (right) follows them. Nothing changed except the form of the input."),
           P("<b>What the comparison establishes.</b> The plain MLP's failure is not a shortage of capacity (the universal approximation "
             "theorem says the capacity is there) nor a shortage of training (we quintupled the training and it was still far "
             "behind). It is the spectral bias predicted by theory: tanh networks build fine detail very slowly from a raw "
             "coordinate. Supplying an oscillatory basis removes the obstacle, after which the MLP reaches the same N^(-1/2) "
             "roughness floor as every other good method, and no further. The two curves in Figure 8 are the theory's prediction "
             "drawn by the data.", B),
           fig("mlps_CBA.AX_mlp_fits0.png", "Figure 10. CBA, plain MLP at 13, 33 and 61 parameters. The rolling-RMSE envelope barely "
               "narrows as the budget rises and the error keeps its slow bands.", folder=PF),
           P("8. What it means", H2),
           P("Whether 'an MLP' approximates a price line well depends entirely on what you feed it. Fed raw time, it is the worst "
             "model tested; fed sines and cosines of time, it ties the best. Any claim that a new architecture (such as a KAN) beats "
             "an MLP at approximating signals must say which MLP: against the plain one the claim is easy and uninformative, against "
             "the Fourier-feature one it is, on this data, not true. Beyond that, the Fourier MLP's convergence to the same "
             "N^(-1/2) rate as splines and KANs is further evidence that the rate belongs to the roughness of price data and not to "
             "any method: once a model can represent fine detail at all, the data decide how fast the error can fall.", B)]
    shared_limits(st, ["One network shape (two hidden layers, tanh) and one Fourier bandwidth (standard deviation 6) were used; neither was "
                       "tuned. A spot check with bandwidth 20 gave 0.0295 against 0.0285 at 61 parameters on CBA, so the result is not "
                       "sensitive to it, but it was not optimised.",
                       "The plain MLP is training-limited; still longer training would lower its error somewhat, but by the spectral-bias "
                       "result not to the spline's level at fine scales."])
    glossary(st, [("MLP", "multilayer perceptron: the standard neural network of weighted edges and fixed node functions."),
                  ("Weight / bias", "the multiplier on an edge / the constant added at a node; the MLP's parameters."),
                  ("tanh", "the smooth step function from -1 to +1 applied at every hidden node."),
                  ("Training / gradient descent", "adjusting the parameters step by step to reduce the error."),
                  ("Adam", "a widely used variant of gradient descent."),
                  ("Fourier features", "replacing the input t by sines and cosines of t at several frequencies."),
                  ("Spectral bias", "the proven tendency of MLPs to learn slow components first and fast components very slowly."),
                  ("Universal approximation", "the theorem that some MLP can approximate any continuous function to any accuracy.")])
    return st


def build(name, story):
    out = os.path.join(HERE, name)
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title=name, author="")
    doc.build(story)
    print("wrote", name)


if __name__ == "__main__":
    build("explain_splines.pdf", doc_splines())
    build("explain_kans.pdf", doc_kans())
    build("explain_mlps.pdf", doc_mlps())
