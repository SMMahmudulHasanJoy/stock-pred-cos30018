# Approximating price series: KAN vs MLP vs Fourier-MLP vs splines

Date: 2026-10-26. Code: `approx_study.py` (fits), `aggregate_approx.py`
(tables, checks, figures). 210 fits, all saved in `fits/` with predictions.

## 1. What question this answers, and what it does not

This is an APPROXIMATION study, not a forecasting study. Each model is given
the entire price series p(t) of a stock, t rescaled to [0, 1] over 2 Jan 2020
to 31 Jul 2023 (905 trading days for CBA, 901 for TSLA and LLY), and asked to
reproduce it as a function of t with a fixed number of parameters. The score
is in-sample error (RMSE, and max error in the saved JSON) against the number
of parameters. The question is: how compactly can each architecture represent
a realised price path, and how fast does its error fall as it is given more
parameters? There is no prediction, no train/test split, and therefore no
random-walk or efficient-market objection. Equally, nothing here says anything
about forecasting; an in-sample fit is not a forecast (Welch & Goyal 2008).

## 2. The six models

Each neighbouring pair differs in one thing.

| Model | What it is | Parameters |
|---|---|---|
| LSQ cubic spline | classical least-squares cubic spline on G equal intervals (deterministic) | G + 3 coefficients |
| free-knot cubic spline | same, but the G-1 interior knot positions are optimised (12 random starts, L-BFGS-B then Nelder-Mead) | G + 3 coefficients + G-1 knots |
| KAN [1,1] | one pykan edge: cubic B-spline on G intervals plus a silu base term, L-BFGS 80 steps | G + 9 |
| KAN [1,h,1] | pykan with a hidden layer of h nodes, each edge a cubic B-spline on G = 3 intervals plus silu | about 24h |
| MLP of t | 1 -> w -> w -> 1, tanh hidden units, linear output, Adam 20,000 full-batch steps, cosine LR | w^2 + 4w + 1 |
| Fourier-feature MLP of t | the same MLP fed [sin(2 pi B t), cos(2 pi B t)], B ~ N(0, 6^2), n_freq fixed random frequencies (Tancik et al. 2020) | w^2 + (2 n_freq + 3) w + 1 |

Parameter budgets targeted 15, 30, 60, 120, 240. The ACTUAL counts are used
everywhere (x-axis, brackets in the table): LSQ 15/30/60/120/240; free-knot
16/32/62/122/242; KAN [1,1] 15/30/60/120/240; KAN [1,h,1] 24/24/48/120/240 (h
= 1, 1, 2, 5, 10; the two smallest budgets cannot be met with a hidden layer);
MLP 13/33/61/118/253; Fourier-MLP 19/31/61/127/253. Neural models were run
with 3 seeds; the KAN [1,1] is deterministic to 4 decimals across seeds.

## 3. Results

In-sample RMSE in each ticker's Min-Max scale (mean +- sd over 3 seeds where
applicable; actual parameter count in brackets). For scale, the one-day naive
forecasting error on these series is 0.020 (CBA), 0.026 (TSLA), 0.015 (LLY).

| Ticker | Budget | LSQ spline | free-knot spline | KAN [1,1] | KAN [1,h,1] | MLP (tanh) | Fourier-MLP |
|---|---|---|---|---|---|---|---|
| CBA.AX | 15 | 0.0607 (15) | 0.0581 (16) | 0.0707 (15) | 0.0720 +- 0.0064 (24) | 0.0753 +- 0.0005 (13) | 0.0842 +- 0.0373 (19) |
| CBA.AX | 30 | 0.0413 (30) | 0.0425 (32) | 0.0464 (30) | 0.0720 +- 0.0064 (24) | 0.0634 +- 0.0009 (33) | 0.0579 +- 0.0189 (31) |
| CBA.AX | 60 | 0.0260 (60) | 0.0275 (62) | 0.0272 (60) | 0.0523 +- 0.0025 (48) | 0.0566 +- 0.0074 (61) | 0.0258 +- 0.0012 (61) |
| CBA.AX | 120 | 0.0173 (120) | 0.0196 (122) | 0.0172 (120) | 0.0446 +- 0.0041 (120) | 0.0531 +- 0.0078 (118) | 0.0177 +- 0.0016 (127) |
| CBA.AX | 240 | 0.0131 (240) | 0.0161 (242) | 0.0131 (240) | 0.0388 +- 0.0016 (240) | 0.0419 +- 0.0024 (253) | 0.0130 +- 0.0014 (253) |
| TSLA | 15 | 0.0661 (15) | 0.0651 (16) | 0.0905 (15) | 0.0877 +- 0.0061 (24) | 0.0868 +- 0.0075 (13) | 0.0985 +- 0.0398 (19) |
| TSLA | 30 | 0.0405 (30) | 0.0448 (32) | 0.0495 (30) | 0.0877 +- 0.0061 (24) | 0.0658 +- 0.0010 (33) | 0.0562 +- 0.0114 (31) |
| TSLA | 60 | 0.0276 (60) | 0.0295 (62) | 0.0295 (60) | 0.0592 +- 0.0057 (48) | 0.0614 +- 0.0106 (61) | 0.0285 +- 0.0022 (61) |
| TSLA | 120 | 0.0189 (120) | 0.0225 (122) | 0.0190 (120) | 0.0420 +- 0.0037 (120) | 0.0489 +- 0.0028 (118) | 0.0182 +- 0.0012 (127) |
| TSLA | 240 | 0.0134 (240) | 0.0168 (242) | 0.0138 (240) | 0.0407 +- 0.0044 (240) | 0.0404 +- 0.0041 (253) | 0.0143 +- 0.0013 (253) |
| LLY | 15 | 0.0369 (15) | 0.0372 (16) | 0.0491 (15) | 0.0481 +- 0.0036 (24) | 0.0495 +- 0.0084 (13) | 0.0676 +- 0.0309 (19) |
| LLY | 30 | 0.0216 (30) | 0.0225 (32) | 0.0261 (30) | 0.0481 +- 0.0036 (24) | 0.0400 +- 0.0008 (33) | 0.0431 +- 0.0175 (31) |
| LLY | 60 | 0.0167 (60) | 0.0183 (62) | 0.0173 (60) | 0.0315 +- 0.0043 (48) | 0.0349 +- 0.0051 (61) | 0.0184 +- 0.0005 (61) |
| LLY | 120 | 0.0114 (120) | 0.0136 (122) | 0.0120 (120) | 0.0233 +- 0.0018 (120) | 0.0352 +- 0.0027 (118) | 0.0124 +- 0.0015 (127) |
| LLY | 240 | 0.0081 (240) | 0.0106 (242) | 0.0083 (240) | 0.0243 +- 0.0017 (240) | 0.0314 +- 0.0028 (253) | 0.0093 +- 0.0007 (253) |

How fast error falls with parameters (slope of log RMSE against log parameter
count over budgets 30 to 240, averaged over the three tickers): LSQ spline
-0.52, free-knot -0.45, KAN [1,1] -0.59, KAN [1,h,1] -0.30, MLP -0.18,
Fourier-MLP -0.68.

Figures: `approx_rmse_vs_params.png` (error vs parameters, log-log, all
models, all tickers) and `approx_fits_cba_b60.png` (what each ~60-parameter
fit looks like against the CBA series, knots marked for the splines).

## 4. What the outcomes mean, in plain terms

**(a) A one-edge KAN is spline regression.** At every budget from 60 upwards,
KAN [1,1] and the least-squares spline agree to the third or fourth decimal on
all three stocks (CBA 0.0272 vs 0.0260, 0.0172 vs 0.0173, 0.0131 vs 0.0131).
The small gap at 15 and 30 parameters is bookkeeping: pykan spends 9 of the
budget on scale and base-function terms, so its spline has fewer intervals (6
vs 12 at budget 15). The KAN's famous "learnable activation" is, on one edge,
exactly the classical tool. This was the hypothesis going in and it held.

**(b) Adding a hidden layer to the KAN makes it worse, not better.** KAN
[1,h,1] is 1.6 to 3 times worse than KAN [1,1] at the same parameter count,
and its error improves only slowly with more parameters (slope -0.30 vs
-0.59). Reason (derivation from the architecture): with G = 3 each edge is a
very coarse spline, and composing coarse smooth functions cannot create the
fine detail a rough price path contains; the parameters go into more edges,
not finer resolution. For this task, resolution on one edge beats depth. The
original hypothesis that "the interesting differences will come from deeper
KANs" is rejected by the data; depth hurt.

**(c) A plain MLP of t cannot fit the detail, exactly as spectral-bias theory
predicts.** The tanh MLP is the worst model at every budget above 15 and its
error barely improves with size (slope -0.18; 0.042 at 253 parameters on CBA
against 0.013 for the spline). The fit plots show why: it draws the slow trend
and ignores the wiggles. Tancik et al. (2020) prove that an MLP of a coordinate
learns low frequencies first and high frequencies extremely slowly; we also
confirmed it is training-limited (0.074 at 4,000 steps, 0.048 at 20,000, same
model), which is the same phenomenon seen from the optimiser's side.

**(d) Putting Fourier features in front of the same MLP removes the problem
completely.** With the identical network and a fixed random sine/cosine map of
t, the MLP matches the spline from 60 parameters upward on all three stocks
(CBA 0.0258 vs 0.0260 at 60; 0.0130 vs 0.0131 at 240) and has the steepest
improvement with size of any model (-0.68). At 15 to 30 parameters it is
erratic (seed sd up to 0.04) because with only 2 to 3 random frequencies the
result depends on which frequencies were drawn. So the MLP's failure in (c)
was not a capacity problem but a basis problem; give it an oscillatory basis
and it is as good a function approximator as the spline. This is the single
clearest architectural lesson in the study.

**(e) Free-knot splines do not win at matched parameter count on this
data.** Given the same number of intervals, optimising knot positions does
help (CBA, 30 intervals: 0.0353 fixed vs 0.0275 free, checked separately). But
each free knot costs a parameter, and at equal total parameters the fixed-knot
spline with more intervals is slightly better everywhere (0.0131 vs 0.0161 at
240 on CBA). On a path that is rough everywhere, there is no small set of
"important" change-points for the knots to find; the fit plot shows the free
knots clustering around the 2020 crash and the 2022 swings and leaving the
rest coarse. This is the opposite of the S1/S2 example, where free knots
recovered the function exactly, because S2 really did have five kinks and
nothing else. (Caveat: knot optimisation is non-convex; 12 starts may not find
the global optimum, so "does not win" is "did not win with this optimiser".)

**(f) Every good method improves at about the same, slow rate, and that rate
is a property of the data.** The spline, KAN [1,1] and Fourier-MLP all show
error falling roughly like (parameters)^(-0.5 to -0.7). Derivation: a random
walk's path is Holder-continuous of order just under 1/2, and spline theory
gives error ~ h^alpha for an alpha-Holder function with mesh h, i.e.
~ N^(-1/2) in the number of intervals N. The measured -0.52 for the spline is
that bound. No smooth approximator can beat it on this kind of series,
regardless of architecture; the only way to a faster rate is a target that is
smoother than a price path.

## 5. Limitations (stated before any use of these numbers)

* In-sample only, by design. Nothing here transfers to forecasting.
* One configuration family per model. The MLP uses 2 hidden layers and tanh;
  the Fourier bandwidth sigma = 6 was fixed, not tuned (a spot check with
  sigma = 20 gave 0.0295 vs 0.0285 at 61 parameters on CBA, so the result is
  not sensitive, but it was not optimised). KAN [1,h,1] used G = 3 only; a
  deeper KAN with finer grids was not tried and might close some of the gap.
* Parameter counts are matched approximately (within about 10 %), and the
  two smallest KAN [1,h,1] budgets could not be met.
* The MLP is optimiser-limited: 20,000 Adam steps were used after finding
  4,000 insufficient; longer training would lower its error further but, by
  Tancik et al.'s result, not close the gap at fine scales.
* Free-knot optimisation is non-convex; 12 starts were used.
* Three seeds for neural models; no significance testing.
* Three tickers, one period (2020-2023).

## 6. What this signifies for the KAN work

1. The claim "a KAN approximates a price path better than an MLP" is true
   only for an MLP without a frequency basis; against a Fourier-feature MLP at
   equal parameters it is a tie, and against classical spline regression it is
   an identity. A paper making the approximation claim must include both
   controls or a referee will supply them.
2. The KAN's value on this task is therefore not accuracy. If there is a case
   for it, it is interpretability (the fitted edge function is a readable
   spline) and the option of pruning or symbolic conversion, which this study
   did not test.
3. Depth is not where the KAN's advantage lies for 1-D signal approximation;
   it is where the advantage disappears.
4. The slow, architecture-independent N^(-1/2) decay is the real finding
   about the data: realised price paths are too rough for any compact smooth
   representation, which is the approximation-theoretic face of the same
   random-walk property that makes them hard to forecast.

Sources: Liu et al. 2024 (arXiv 2404.19756); Yu, Yu & Wang 2024 (arXiv
2407.16674); Tancik et al. 2020 (arXiv 2006.10739); Welch & Goyal 2008 (RFS
21(4)); de Boor, A Practical Guide to Splines (spline approximation rates);
Morters & Peres, Brownian Motion, ch. 1 (Holder regularity of paths).
