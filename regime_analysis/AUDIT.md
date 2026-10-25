# Code and results audit: naive, LSTM and KAN forecasting pipeline

Date: 2026-10-25. Scope: `regime_lib.py`, `driver.py` (naive, lstmcfg, kan/pykan
paths), the saved prediction files in `results/` (submitted), `results_rev1/`
(revision, per-fold scaling) and `results_mt/` (TSLA, LLY), and a purged
re-run written to `results_purged/`. Every number below was recomputed from
the `.npz` prediction files or the CSVs during this audit, not copied from
earlier notes.

## 1. What was checked and passed

Independent recomputation, for all 3 tickers x 6 horizons x 5 folds (90
fold-sets, 630 prediction files per protocol):

* `truth` arrays equal `Close[end+1 .. end+h]` read directly from the CSV.
* Naive predictions equal `Close[end]` repeated h times (and therefore equal
  the previous row's truth, checked separately).
* Stored window dates equal the CSV date of the last input row.
* The per-fold scaler parameters stored in `meta.scale` equal the min and max
  of Close over rows `0 .. last_train_row`, and `last_train_row` equals
  `(L-1) + (train_stop-1) + h` as the code intends.
* Truth, dates, regime labels and scaler metadata are identical across the
  naive, LSTM and KAN files of the same (horizon, fold).
* Sequence counts (e.g. 885 windows and 531 test windows for CBA at h=1) and
  fold boundaries match the arithmetic in `walk_forward_folds`.
* Regime thresholds come from training-window input statistics only.
* `results_rev1` reproduces `revision_numbers.json` (the paper's Table 7) to
  machine precision.
* Seeding: LSTM `set_random_seed(seed+fold)`; pykan seeded in numpy and torch.
  Re-running LSTM jobs with identical inputs reproduces identical RMSE (seen
  at h=1 below). pykan runs are not bit-reproducible (documented in
  FINDINGS.md s4c); the h=1 purged vs unpurged difference (0.0338 vs 0.0331 on
  CBA, with identical training data) is a direct measurement of that noise,
  about 0.001 RMSE.

No coding errors were found in the window construction, fold logic, scaling
back to the reporting scale, naive baseline, or file writing.

## 2. Finding 1 (protocol flaw, material): no purge/embargo for multi-step targets

With direct h-step targets, training sequence i has labels at rows
`end_i+1 .. end_i+h`. The last h-1 training sequences therefore carry labels
that lie AFTER the forecast origin of the first test windows, and the per-fold
Min-Max scaler (fitted through `last_train_row`) sees those rows too. At test
time the model has been trained on the true values of rows it is now asked to
forecast (for other windows). This is the look-ahead that purged
cross-validation removes (Lopez de Prado, *Advances in Financial Machine
Learning*, 2018, ch. 7; Bergmeir & Benitez, *Information Sciences* 191, 2012).

Measured extent (test windows whose target rows overlap training label rows,
summed over the five folds; identical pattern on all three tickers):

| h | 1 | 2 | 5 | 10 | 100 | 200 |
|---|---|---|---|---|---|---|
| affected test windows (CBA) | 0 of 531 | 5 of 531 | 20 of 529 | 45 of 526 | 472 of 472 | 412 of 412 |

At h=1 the protocol is clean. At h=100 and 200 every test window in every
fold is affected, because the fold blocks (94 and 82 windows) are shorter
than the horizon. This flaw is present in the submitted runs (`results/`), the
revision runs (`results_rev1/`) and the multi-ticker runs (`results_mt/`).

### Fix

`regime_lib.make_fold(..., purge=True)` (or `REGIME_PURGE=1`) drops the last
h-1 training sequences so that every training label row is <= the first test
window's origin row, and fits the scaler on the same reduced range. Test sets,
dates and regime labels are unchanged (verified: identical arrays), so purged
and unpurged runs are scored on the same targets. Regime thresholds are still
computed from all pre-test input windows (inputs end before the origin, so this
is not leakage) to keep labels identical.

Cost of the fix in training data (CBA, sequences per fold, folds 0..4):

| h | unpurged | purged |
|---|---|---|
| 100 | 314, 408, 502, 596, 690 | 215, 309, 403, 497, 591 |
| 200 | 274, 356, 438, 520, 602 | 75, 157, 239, 321, 403 |

At h=200, fold 0 trains on 75 sequences. This is not an artefact of the fix;
it is the honest amount of data available for a 200-day-ahead model on a
3.5-year series. It is a limitation of the horizon, and should be stated as
such.

### Effect on results (full re-run, 630 jobs, `results_purged/`)

Test RMSE in each ticker's reporting scale, mean +- sd over 3 seeds, 5 pooled
folds. "Unpurged" is `results_rev1` / `results_mt`; "purged" is the new run.

CBA.AX

| h | Naive | LSTM unpurged | LSTM purged | KAN unpurged | KAN purged |
|---|---|---|---|---|---|
| 1 | 0.0201 | 0.0586 +- 0.0009 | 0.0586 +- 0.0009 | 0.0338 +- 0.0006 | 0.0331 +- 0.0016 |
| 2 | 0.0252 | 0.0579 +- 0.0014 | 0.0598 +- 0.0031 | 0.0475 +- 0.0031 | 0.0444 +- 0.0013 |
| 5 | 0.0364 | 0.0673 +- 0.0051 | 0.0715 +- 0.0060 | 0.0621 +- 0.0061 | 0.0552 +- 0.0046 |
| 10 | 0.0510 | 0.0737 +- 0.0050 | 0.0738 +- 0.0024 | 0.0913 +- 0.0119 | 0.0893 +- 0.0035 |
| 100 | 0.1109 | 0.1079 +- 0.0126 | **0.2490 +- 0.0348** | 0.2418 +- 0.0863 | **0.4080 +- 0.0997** |
| 200 | 0.1323 | 0.1375 +- 0.0082 | **0.3628 +- 0.0033** | 0.2764 +- 0.0584 | **0.5935 +- 0.1811** |

TSLA

| h | Naive | LSTM unpurged | LSTM purged | KAN unpurged | KAN purged |
|---|---|---|---|---|---|
| 1 | 0.0255 | 0.0743 +- 0.0083 | 0.0743 +- 0.0083 | 0.0446 +- 0.0032 | 0.0451 +- 0.0039 |
| 2 | 0.0311 | 0.0746 +- 0.0032 | 0.0770 +- 0.0030 | 0.0524 +- 0.0028 | 0.0499 +- 0.0027 |
| 5 | 0.0441 | 0.0798 +- 0.0066 | 0.0832 +- 0.0122 | 0.0780 +- 0.0062 | 0.0878 +- 0.0437 |
| 10 | 0.0612 | 0.0934 +- 0.0061 | 0.1024 +- 0.0036 | 0.1011 +- 0.0077 | 0.1028 +- 0.0142 |
| 100 | 0.1701 | 0.1757 +- 0.0047 | 0.1840 +- 0.0024 | 0.1837 +- 0.0132 | 0.2389 +- 0.0127 |
| 200 | 0.2173 | 0.1867 +- 0.0084 | **0.3013 +- 0.0031** | 0.2148 +- 0.0043 | **0.4400 +- 0.0391** |

LLY

| h | Naive | LSTM unpurged | LSTM purged | KAN unpurged | KAN purged |
|---|---|---|---|---|---|
| 1 | 0.0148 | 0.0967 +- 0.0005 | 0.0967 +- 0.0005 | 0.0451 +- 0.0107 | 0.0413 +- 0.0039 |
| 2 | 0.0180 | 0.0979 +- 0.0013 | 0.1009 +- 0.0010 | 0.0585 +- 0.0083 | 0.0526 +- 0.0038 |
| 5 | 0.0255 | 0.0992 +- 0.0037 | 0.0999 +- 0.0023 | 0.0795 +- 0.0044 | 0.0797 +- 0.0028 |
| 10 | 0.0353 | 0.1009 +- 0.0038 | 0.1020 +- 0.0015 | 0.0956 +- 0.0141 | 0.1258 +- 0.0107 |
| 100 | 0.1053 | 0.1014 +- 0.0053 | **0.1875 +- 0.0045** | 0.1557 +- 0.0404 | **0.2846 +- 0.0135** |
| 200 | 0.1395 | 0.1012 +- 0.0089 | **0.2621 +- 0.0021** | 0.2000 +- 0.0353 | **0.5354 +- 0.1240** |

Reading:

* h = 1: unchanged by construction (0 sequences dropped). LSTM numbers are
  bit-identical; KAN differences are pykan run-to-run noise.
* h = 2 to 10: changes are within one to two seed standard deviations in
  either direction. The short-horizon conclusions of the paper (naive best,
  KAN closer to naive than LSTM at 1-5 days, crossover near 10 days) stand.
* h = 100 and 200: the picture changes materially. Without label look-ahead
  the LSTM is 1.1x to 2.7x worse than naive (previously "at or slightly below
  naive"), and the KAN is 1.4x to 4.5x worse. The statement in the revised
  paper that the LSTM matches persistence at long horizons, and the TSLA/LLY
  observation that the LSTM beats persistence at 200 days, were artefacts of
  training on labels from the forecast period. The corrected result is the
  textbook one: at long horizons neither learned model approaches persistence
  on raw price levels.
* The long-horizon KAN variance (sd up to 0.18) and its very large errors
  indicate the pykan model is unstable when training sets are small and test
  inputs leave the training range; see Finding 3.

Figures: `results_purged/multi_ticker_rmse.png`, `multi_ticker_ratio.png`
(produced by `MT_PURGED=1 python multi_ticker.py`).

## 3. Finding 2 (design property, must be reported): the LSTM cannot forecast above its training maximum

The L2-20-tanh configuration (and every "tanh" configuration in Tables 4/5)
applies `tanh` as the activation of the final `Dense(horizon)` layer. In fold
scale the training Close lies in [0, 1], so tanh's upper bound of 1 maps to the
training maximum. Checked on the saved files (h=1, seed 0, every fold, every
ticker): the LSTM's maximum prediction never exceeds `close_hi`. On LLY,
between 34% and 81% of test truth lies above `close_hi` in folds 0-4, which is
why the LSTM's error on LLY is flat at about 0.10 across all horizons. The
"linear" configurations do not have this bound.

This is the inherited configuration from the published paper, not a coding
error, but it is a structural handicap for the recurrent model on any series
that makes new highs during the test period, and it is one reason the
KAN-vs-LSTM gap is largest on the trending stock. It should be stated in the
methodology and discussed with the LLY result. A re-screen with a linear output
layer would be the natural control; it has not been run.

## 4. Finding 3 (documentation / consistency items)

* `aggregate.py` builds its Table 7 from `results*/preds/kan__*` which are the
  custom TensorFlow KAN engine; the paper's Table 7 and `revision_numbers.py`
  use the pykan engine (`preds_pykan/`). The two engines differ substantially
  (CBA, unpurged, h=1: custom 0.0640 vs pykan 0.0338; h=200: custom 0.1287 vs
  pykan 0.2764). Nothing in the paper comes from `aggregate.py`, but the script
  should either be pointed at `preds_pykan` or labelled as the custom-engine
  diagnostic to avoid confusion for anyone reproducing the work.
* The pykan hidden width is `3 * (n_train // 50)`, so the KAN architecture
  changes with fold (and shrinks under purging at long horizons). This is in
  the code and should be stated explicitly in the methodology.
* `regime_lib.DATA_CSV` default was an absolute path to this machine until
  the multi-ticker commit; it is now relative to the repository.
* The LSTM used at every horizon in Table 7 is the configuration selected at
  h=1 (L2-20-tanh). This is a stated protocol choice, not an error.

## 5. Finding 4 (comparison with the submitted results, CBA)

Three protocols side by side (test RMSE, mean +- sd over 3 seeds):

| h | model | A: submitted (whole-series scaler, no purge) | B: revision (per-fold scaler, no purge) | C: audit (per-fold scaler + purge) |
|---|---|---|---|---|
| 1 | LSTM | 0.0537 +- 0.0003 | 0.0586 +- 0.0009 | 0.0586 +- 0.0009 |
| 1 | KAN | 0.0409 +- 0.0059 | 0.0338 +- 0.0006 | 0.0331 +- 0.0016 |
| 2 | LSTM | 0.0528 +- 0.0032 | 0.0579 +- 0.0014 | 0.0598 +- 0.0031 |
| 2 | KAN | 0.0473 +- 0.0053 | 0.0475 +- 0.0031 | 0.0444 +- 0.0013 |
| 100 | LSTM | 0.1144 +- 0.0118 | 0.1079 +- 0.0126 | 0.2490 +- 0.0348 |
| 100 | KAN | 0.4462 +- 0.2252 | 0.2418 +- 0.0863 | 0.4080 +- 0.0997 |
| 200 | LSTM | 0.1406 +- 0.0077 | 0.1375 +- 0.0082 | 0.3628 +- 0.0033 |
| 200 | KAN | 0.2654 +- 0.0868 | 0.2764 +- 0.0584 | 0.5935 +- 0.1811 |

Naive is 0.0201 / 0.0252 / 0.1109 / 0.1323 under all three (it uses only the
last observed close). A -> B removed the whole-series scaler leak: small
changes at short horizons, as reported in the revision. B -> C removes the
label look-ahead: no change at h=1, noise-level changes to h=10, large changes
at h=100/200.

## 6. What this means for the paper

1. Short-horizon results and the main KAN-vs-LSTM comparison (1, 2, 5 days) do
   not change. Naive remains the benchmark nobody beats.
2. The 100- and 200-day rows of Tables 5 and 7, Figures 2-4 where they show
   long horizons, and the sentences saying the LSTM "matches" or "approaches"
   persistence at long horizons need to be replaced with the purged numbers.
   The corrected statement is that both learned models are substantially
   worse than persistence at 100 and 200 days, with the KAN worse than the
   LSTM.
3. The methodology must describe the purge (one sentence plus the reference
   to purged cross-validation) and state the per-fold training sizes at long
   horizons, since 75 training sequences at h=200 is a real limitation.
4. The tanh output bound should be stated as a property of the inherited LSTM
   configuration, with the LLY result as its illustration.
5. The multi-ticker report (`results_mt/MULTI_TICKER_REPORT.md`) should be
   read with its long-horizon rows replaced by the purged ones; its
   short-horizon conclusions are unchanged.

## 7. Limitations of this audit

* One LSTM configuration and one KAN configuration were re-run; the Table 4
  and Table 6 configuration grids (h=1 only) are unaffected by the purge and
  were not re-run.
* The MLP-on-flat-vector and recurrent-KAN comparison runs (`preds_newmodels`)
  are unpurged and their 100/200-day rows carry the same look-ahead; they were
  not re-run (short-horizon rows unaffected).
* Three seeds; no significance tests applied here.
* pykan is not bit-reproducible; differences of about 0.001 RMSE at h=1 are
  noise, not protocol effects.
* The linear-output LSTM control for Finding 2 has not been run.

## 8. Files

* Code: `regime_lib.py` (purge option, `purged` flag in fold dict),
  `driver.py` (`purged` written to metadata), `run_purged.sh`,
  `multi_ticker.py` (`MT_PURGED=1`).
* Data: `results_purged/{CBA,TSLA,LLY}/` (210 files each),
  `results_purged/purged_vs_unpurged_summary.json`,
  `results_purged/cba_three_way.json`, `results_purged/multi_ticker_*.{md,tex,png,json}`.
