"""explainer_figs.py: concept figures for the three explainer PDFs."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.interpolate import BSpline, LSQUnivariateSpline

import approx_study as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "explainer_figs"); os.makedirs(OUT, exist_ok=True)
t, y = A.load("CBA.AX")
GREY = "#999999"


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=160); plt.close(fig)


# 1. the price series as a function of t, and what "approximating" means
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(t, y, color=GREY, linewidth=0.9, label="CBA closing price, scaled (0 = lowest, 1 = highest), 905 trading days")
sp = LSQUnivariateSpline(t, y, np.linspace(0, 1, 13)[1:-1], k=3)
ax.plot(t, sp(t), color="#c0392b", linewidth=2, label="one smooth curve described by 15 numbers")
ax.set_xlabel("t   (0 = 2 January 2020,  1 = 31 July 2023)"); ax.set_ylabel("scaled price")
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
save(fig, "01_series_and_curve.png")

# 2. error and RMSE illustrated
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
sub = slice(300, 420)
axes[0].plot(t[sub], y[sub], color=GREY, linewidth=1.2, label="actual price")
axes[0].plot(t[sub], sp(t)[sub], color="#c0392b", linewidth=2, label="curve")
for i in range(300, 420, 6):
    axes[0].plot([t[i], t[i]], [y[i], sp(t[i])], color="k", linewidth=0.8)
axes[0].legend(frameon=False, fontsize=9); axes[0].set_ylabel("scaled price"); axes[0].set_title("Each black bar is one day's error: curve minus actual", fontsize=10)
e = sp(t) - y
axes[1].fill_between(t[sub], 0, e[sub], where=e[sub] >= 0, color="#c0392b", alpha=0.5)
axes[1].fill_between(t[sub], 0, e[sub], where=e[sub] < 0, color="#2e86c1", alpha=0.5)
axes[1].axhline(0, color="k", linewidth=0.6); axes[1].set_ylabel("error"); axes[1].set_xlabel("t")
axes[1].set_title("The same errors drawn as a wave: red = curve above price, blue = below.  RMSE = sqrt(mean of error^2) = %.4f here" % np.sqrt(np.mean(e[sub] ** 2)), fontsize=10)
save(fig, "02_error_and_rmse.png")

# 3. B-spline basis functions and a spline as their weighted sum
knots_int = np.linspace(0, 1, 7)[1:-1]
kk = np.r_[[0] * 4, knots_int, [1] * 4]
n_basis = len(kk) - 4
xx = np.linspace(0, 1, 600)
fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
for i in range(n_basis):
    c = np.zeros(n_basis); c[i] = 1
    axes[0].plot(xx, BSpline(kk, c, 3)(xx), linewidth=1.5)
for k in knots_int:
    axes[0].axvline(k, color="k", linestyle=":", linewidth=0.8)
axes[0].set_title("The %d cubic B-spline 'bumps' for 6 intervals (dotted lines = knots). Each bump is a smooth piece of cubic polynomial that lives on 4 neighbouring intervals." % n_basis, fontsize=9)
axes[0].set_ylabel("bump height")
sp6 = LSQUnivariateSpline(t, y, knots_int, k=3)
coef = sp6.get_coeffs()
axes[1].plot(t, y, color=GREY, linewidth=0.8, label="price")
for i in range(n_basis):
    c = np.zeros(n_basis); c[i] = coef[i]
    axes[1].plot(xx, BSpline(kk, c, 3)(xx), linewidth=1, alpha=0.6)
axes[1].plot(xx, sp6(xx), color="#c0392b", linewidth=2.2, label="spline = sum of the scaled bumps (%d coefficients)" % n_basis)
for k in knots_int:
    axes[1].axvline(k, color="k", linestyle=":", linewidth=0.8)
axes[1].legend(frameon=False, fontsize=9); axes[1].set_xlabel("t"); axes[1].set_ylabel("scaled price")
save(fig, "03_bspline_basis.png")

# 4. fixed vs free knots on a toy function with a kink (S2-like)
xs = np.linspace(0, 10, 1001)
breaks = [0, 3, 5, 6, 7]; slopes = [2, -1, 4, -3, -1]
f = np.zeros_like(xs); val = 0.0
for i, b in enumerate(breaks):
    nxt = breaks[i + 1] if i + 1 < len(breaks) else 10.01
    m = (xs >= b) & (xs < nxt)
    f[m] = val + slopes[i] * (xs[m] - b); val = val + slopes[i] * (nxt - b)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
from scipy.interpolate import LSQUnivariateSpline as L
fixed = L(xs, f, np.linspace(0, 10, 6)[1:-1], k=1); free = L(xs, f, [3, 5, 6, 7], k=1)
for ax, s, title in ((axes[0], fixed, "knots at 2, 4, 6, 8 (fixed, equally spaced)"), (axes[1], free, "knots at 3, 5, 6, 7 (placed where the kinks are)")):
    ax.plot(xs, f, color=GREY, linewidth=3, label="target: piecewise-linear S2"); ax.plot(xs, s(xs), color="#c0392b", linewidth=1.5, label="linear spline, max error %.2f" % np.abs(s(xs) - f).max())
    for k in s.get_knots()[1:-1]:
        ax.axvline(k, color="k", linestyle=":", linewidth=0.8)
    ax.set_title(title, fontsize=9); ax.legend(frameon=False, fontsize=8); ax.set_xlabel("x")
save(fig, "04_fixed_vs_free_knots.png")

# 5. KAN edge: spline + silu
xx = np.linspace(-1.5, 2.5, 400)
silu = xx / (1 + np.exp(-xx))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
axes[0].plot(xx, silu, color="#2e86c1"); axes[0].set_title("base term  b(x) = silu(x) = x / (1 + e^-x)", fontsize=9)
kk2 = np.r_[[0] * 4, np.linspace(0, 1, 4)[1:-1], [1] * 4]; c2 = np.array([0.1, 0.6, -0.2, 0.8, 0.3, 0.5])
xs2 = np.linspace(0, 1, 300); s2 = BSpline(kk2, c2, 3)(xs2)
axes[1].plot(xs2, s2, color="#c46a3f"); axes[1].set_title("spline term  s(x): cubic B-spline on 3 intervals (6 coefficients)", fontsize=9)
for k in np.linspace(0, 1, 4)[1:-1]:
    axes[1].axvline(k, color="k", linestyle=":", linewidth=0.8)
axes[2].plot(xs2, 0.3 * (xs2 / (1 + np.exp(-xs2))) + 1.0 * s2, color="k"); axes[2].set_title("edge function  phi(x) = w_b b(x) + w_s s(x)", fontsize=9)
for ax in axes:
    ax.grid(alpha=0.3); ax.set_xlabel("x")
save(fig, "05_kan_edge.png")


def node(ax, x, y, label, r=0.22, fc="white"):
    ax.add_patch(Circle((x, y), r, fc=fc, ec="k", linewidth=1.2, zorder=3)); ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=4)


def edge(ax, x0, y0, x1, y1, label=None, color="#c46a3f"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.5, shrinkA=14, shrinkB=14, zorder=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, label, ha="center", fontsize=8, color=color)


# 6. KAN [1,1] and [1,3,1] diagrams, MLP diagram
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
ax = axes[0]; node(ax, 0, 0, "t"); node(ax, 2.5, 0, "y"); edge(ax, 0, 0, 2.5, 0, "phi(t) = w_b silu(t)\n + w_s spline(t)")
ax.set_title("KAN [1,1]: one input, one output,\nONE learnable function on the single edge", fontsize=9)
ax = axes[1]; node(ax, 0, 0, "t")
for i, yy in enumerate((1, 0, -1)):
    node(ax, 1.6, yy, "+"); edge(ax, 0, 0, 1.6, yy, "phi%d" % (i + 1) if i == 0 else None); edge(ax, 1.6, yy, 3.2, 0, "psi%d" % (i + 1) if i == 0 else None)
node(ax, 3.2, 0, "y"); ax.set_title("KAN [1,3,1]: hidden layer of 3 nodes; 6 edges, each its own spline;\nnodes just ADD what comes in", fontsize=9)
ax = axes[2]; node(ax, 0, 0, "t")
for yy in (1, 0, -1):
    node(ax, 1.3, yy, "tanh", r=0.26, fc="#eef"); edge(ax, 0, 0, 1.3, yy, None, color="#2f6f9f")
for yy in (1, 0, -1):
    node(ax, 2.6, yy, "tanh", r=0.26, fc="#eef")
    for y0 in (1, 0, -1):
        edge(ax, 1.3, y0, 2.6, yy, None, color="#2f6f9f")
node(ax, 3.9, 0, "y")
for y0 in (1, 0, -1):
    edge(ax, 2.6, y0, 3.9, 0, None, color="#2f6f9f")
ax.set_title("MLP 1-3-3-1: every edge is a NUMBER (a weight);\nevery node applies the FIXED function tanh", fontsize=9)
for ax in axes:
    ax.set_xlim(-0.5, 4.4); ax.set_ylim(-1.6, 1.7); ax.axis("off")
save(fig, "06_architectures.png")

# 7. tanh and what a sum of tanh's can draw
xx = np.linspace(-4, 4, 400)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].plot(xx, np.tanh(xx), color="#2f6f9f"); axes[0].set_title("tanh(x): a smooth step from -1 to +1", fontsize=9); axes[0].grid(alpha=0.3)
xs2 = np.linspace(0, 1, 400)
comp = 0.5 + 0.3 * np.tanh(12 * (xs2 - 0.3)) - 0.25 * np.tanh(20 * (xs2 - 0.6)) + 0.15 * np.tanh(8 * (xs2 - 0.85))
axes[1].plot(xs2, comp, color="#2f6f9f"); axes[1].set_title("a few scaled and shifted tanh steps added together: smooth bends, no fine wiggles", fontsize=9); axes[1].grid(alpha=0.3)
save(fig, "07_tanh.png")

# 8. Fourier features
fig, axes = plt.subplots(2, 1, figsize=(10, 5))
for b, c in zip((1, 3, 7), ("#5fb0e0", "#2e86c1", "#1b4f72")):
    axes[0].plot(xs2, np.sin(2 * np.pi * b * xs2), color=c, label="sin(2 pi %d t)" % b)
axes[0].legend(frameon=False, fontsize=8, ncol=3); axes[0].set_title("Fourier features: the input t is replaced by sines and cosines of t at several frequencies", fontsize=9)
rng = np.random.default_rng(0); B = rng.normal(0, 6, 4)
feats = np.concatenate([np.sin(2 * np.pi * np.outer(xs2, B)), np.cos(2 * np.pi * np.outer(xs2, B))], 1)
w = rng.normal(0, 0.3, 8)
axes[1].plot(xs2, 0.5 + feats @ w, color="#1b4f72"); axes[1].set_title("a weighted sum of 8 such features already has fine wiggles 'for free'; the MLP then only has to combine them", fontsize=9)
for ax in axes:
    ax.grid(alpha=0.3)
axes[1].set_xlabel("t")
save(fig, "08_fourier_features.png")

# 9. spectral bias: actual fits at budget 60 on CBA: mlp vs fmlp (from saved fits)
def best(ticker, model, budget):
    import glob
    rs = [json.load(open(p)) for p in glob.glob(os.path.join(HERE, "fits", "%s__%s__b%d__s*.json" % (ticker, model, budget)))]
    return min(rs, key=lambda r: r["rmse"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, m, lab, c in ((axes[0], "mlp", "plain MLP of t", "#2f6f9f"), (axes[1], "fmlp", "the same MLP with Fourier features", "#5fb0e0")):
    r = best("CBA.AX", m, 60)
    ax.plot(t, y, color=GREY, linewidth=0.8); ax.plot(t, r["pred"], color=c, linewidth=1.4)
    ax.set_title("%s, %d parameters, RMSE %.4f" % (lab, r["params"], r["rmse"]), fontsize=9); ax.set_xlabel("t"); ax.grid(alpha=0.3)
save(fig, "09_spectral_bias.png")

# 10. power law decay, log-log, with reference slopes
S = json.load(open(os.path.join(HERE, "approx_summary.json")))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for m, lab, c in (("lsq", "LSQ spline", "#555"), ("kan11", "KAN [1,1]", "#c46a3f"), ("kan1h1", "KAN [1,h,1]", "#e0a070"), ("mlp", "MLP of t", "#2f6f9f"), ("fmlp", "Fourier MLP", "#5fb0e0")):
    xs_ = [S["CBA.AX|%s|%d" % (m, b)]["params"] for b in A.BUDGETS]; ys_ = [S["CBA.AX|%s|%d" % (m, b)]["rmse_mean"] for b in A.BUDGETS]
    ax.plot(xs_, ys_, "o-", color=c, label=lab, ms=4)
N = np.array([15, 240]); ax.plot(N, 0.26 * N ** -0.5, "k--", linewidth=0.8, label="reference: error ~ 1/sqrt(N)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("number of parameters N"); ax.set_ylabel("RMSE"); ax.set_title("CBA: on log-log axes a power law is a straight line", fontsize=9)
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, which="both", linewidth=0.4)
save(fig, "10_power_law.png")

# 11. roughness: zooming into a price path never gets smooth; compare with a smooth curve
fig, axes = plt.subplots(2, 3, figsize=(11, 5))
smooth = sp(t)
for j, (a, b) in enumerate(((0, 905), (300, 420), (350, 375))):
    axes[0, j].plot(t[a:b], y[a:b], color=GREY, linewidth=1); axes[0, j].set_title("price, zoom level %d" % (j + 1), fontsize=9)
    axes[1, j].plot(t[a:b], smooth[a:b], color="#c0392b", linewidth=1); axes[1, j].set_title("smooth curve, zoom level %d" % (j + 1), fontsize=9)
for ax in axes.ravel():
    ax.grid(alpha=0.3); ax.set_xticks([])
save(fig, "11_roughness.png")

# 12. parameter counting diagram for MLP (w=2)
fig, ax = plt.subplots(figsize=(7, 2.8))
node(ax, 0, 0, "t")
for yy in (0.6, -0.6):
    node(ax, 1.5, yy, "tanh", r=0.28, fc="#eef"); edge(ax, 0, 0, 1.5, yy, None, "#2f6f9f")
for yy in (0.6, -0.6):
    node(ax, 3.0, yy, "tanh", r=0.28, fc="#eef")
    for y0 in (0.6, -0.6):
        edge(ax, 1.5, y0, 3.0, yy, None, "#2f6f9f")
node(ax, 4.5, 0, "y")
for y0 in (0.6, -0.6):
    edge(ax, 3.0, y0, 4.5, 0, None, "#2f6f9f")
ax.text(2.25, -1.35, "weights: 2 + 4 + 2 = 8;  biases: 2 + 2 + 1 = 5;  total 13 parameters (= w^2 + 4w + 1 with w = 2)", ha="center", fontsize=9)
ax.set_xlim(-0.5, 5); ax.set_ylim(-1.7, 1.2); ax.axis("off")
save(fig, "12_mlp_params.png")

print("figures:", len(os.listdir(OUT)))
