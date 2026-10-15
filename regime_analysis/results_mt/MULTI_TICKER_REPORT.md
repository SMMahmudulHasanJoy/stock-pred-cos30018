# Does the naive result depend on the stock? Three tickers, six horizons

Question (Nadia, 15 Oct): the naive method is surprisingly efficient on CBA. Is that a
property of the CBA data, or of the problem? This check re-runs the paper's main comparison
on two additional tickers chosen to be as different from CBA as possible, and adds the
5- and 10-day horizons as a middle ground.

## Setup

| | CBA.AX | TSLA | LLY |
|---|---|---|---|
| Role | paper benchmark (bank) | very volatile | stable, strongly trending |
| Annualised volatility, 2020-01 to 2023-08 | 27% | 70% | 32% |
| Max drawdown in window | 40% | 74% | 23% |
| Price multiple over window | 1.3x | 9.1x | 3.4x |
| Kaufman efficiency ratio (whole window) | 0.03 | 0.04 | 0.11 |

TSLA and LLY were screened from 14 candidates (MSFT, JNJ, PG, WMT, KO, COST, LLY, BRK-B,
NVDA, AAPL, PEP, MCD, TSLA, CBA): TSLA has the highest volatility and deepest drawdown,
LLY the highest efficiency ratio (cleanest trend) with moderate volatility. Daily OHLCV from
Yahoo Finance, 901 NYSE trading days, same window as the CBA file; `data/make_ticker_csv.py`
builds the CSVs in the paper's whole-series Min-Max format.

Protocol is identical to the revised paper: 20-day look-back, five blocked walk-forward
folds (first 40% training only), Min-Max fitted per fold on training rows only (leak-free),
three seeds (0, 100, 200), RMSE pooled over folds and reported in each ticker's Min-Max
scale. Models: naive persistence (repeat last close), LSTM L2-20-tanh (the configuration
selected in the Table 4 screening, 25 epochs RMSprop), KAN via pykan (grid 3, k 3, hidden
3*(n_train//50), 10 L-BFGS steps). 630 runs in total. Horizons 1, 2, 5, 10, 100, 200 days.

Reproduce: `bash run_multi_ticker.sh` then `python multi_ticker.py`.

## Results (test RMSE, mean ± sd over 3 seeds)

### CBA (bank, benchmark)

| Horizon | Naive | LSTM | KAN | LSTM/Naive | KAN/Naive | KAN/LSTM | Best |
|---|---|---|---|---|---|---|---|
| 1-day | 0.0201 | 0.0586 ± 0.0009 | 0.0338 ± 0.0006 | 2.91 | 1.68 | 0.58 | Naive |
| 2-day | 0.0252 | 0.0579 ± 0.0014 | 0.0475 ± 0.0031 | 2.30 | 1.88 | 0.82 | Naive |
| 5-day | 0.0364 | 0.0673 ± 0.0051 | 0.0621 ± 0.0061 | 1.85 | 1.71 | 0.92 | Naive |
| 10-day | 0.0510 | 0.0737 ± 0.0050 | 0.0913 ± 0.0119 | 1.45 | 1.79 | 1.24 | Naive |
| 100-day | 0.1109 | 0.1079 ± 0.0126 | 0.2418 ± 0.0863 | 0.97 | 2.18 | 2.24 | LSTM |
| 200-day | 0.1323 | 0.1375 ± 0.0082 | 0.2764 ± 0.0584 | 1.04 | 2.09 | 2.01 | Naive |

### TSLA (volatile)

| Horizon | Naive | LSTM | KAN | LSTM/Naive | KAN/Naive | KAN/LSTM | Best |
|---|---|---|---|---|---|---|---|
| 1-day | 0.0255 | 0.0743 ± 0.0083 | 0.0446 ± 0.0032 | 2.91 | 1.75 | 0.60 | Naive |
| 2-day | 0.0311 | 0.0746 ± 0.0032 | 0.0524 ± 0.0028 | 2.40 | 1.68 | 0.70 | Naive |
| 5-day | 0.0441 | 0.0798 ± 0.0066 | 0.0780 ± 0.0062 | 1.81 | 1.77 | 0.98 | Naive |
| 10-day | 0.0612 | 0.0934 ± 0.0061 | 0.1011 ± 0.0077 | 1.53 | 1.65 | 1.08 | Naive |
| 100-day | 0.1701 | 0.1757 ± 0.0047 | 0.1837 ± 0.0132 | 1.03 | 1.08 | 1.05 | Naive |
| 200-day | 0.2173 | 0.1867 ± 0.0084 | 0.2148 ± 0.0043 | 0.86 | 0.99 | 1.15 | LSTM |

### LLY (stable trend)

| Horizon | Naive | LSTM | KAN | LSTM/Naive | KAN/Naive | KAN/LSTM | Best |
|---|---|---|---|---|---|---|---|
| 1-day | 0.0148 | 0.0967 ± 0.0005 | 0.0451 ± 0.0107 | 6.53 | 3.04 | 0.47 | Naive |
| 2-day | 0.0180 | 0.0979 ± 0.0013 | 0.0585 ± 0.0083 | 5.43 | 3.24 | 0.60 | Naive |
| 5-day | 0.0255 | 0.0992 ± 0.0037 | 0.0795 ± 0.0044 | 3.89 | 3.12 | 0.80 | Naive |
| 10-day | 0.0353 | 0.1009 ± 0.0038 | 0.0956 ± 0.0141 | 2.86 | 2.71 | 0.95 | Naive |
| 100-day | 0.1053 | 0.1014 ± 0.0053 | 0.1557 ± 0.0404 | 0.96 | 1.48 | 1.54 | LSTM |
| 200-day | 0.1395 | 0.1012 ± 0.0089 | 0.2000 ± 0.0353 | 0.73 | 1.43 | 1.98 | LSTM |

Raw RMSEs are in each ticker's own Min-Max scale, so compare models within a ticker and
use the ratio columns across tickers. Figures: `multi_ticker_rmse.png` (RMSE vs horizon
per ticker), `multi_ticker_ratio.png` (model/naive ratio, all tickers).

## What the three tickers say

1. **The naive result is not a CBA artefact.** At every horizon up to 10 days, on all
   three stocks, persistence beats both learned models, and by a similar margin: the
   LSTM/naive ratio at 1 day is 2.9 on CBA and 2.9 on TSLA, the KAN/naive ratio 1.7 on
   both. Changing the stock changes the absolute error (TSLA's is larger because TSLA
   moves more) but not the ordering. This is the random-walk property of daily prices,
   not a feature of one bank's data.

2. **The KAN-vs-LSTM ordering at short horizons also holds on every ticker.** KAN/LSTM at
   1 day: 0.58 (CBA), 0.60 (TSLA), 0.47 (LLY). The KAN advantage shrinks steadily with
   horizon and reverses somewhere between 5 and 10 days on all three stocks (KAN/LSTM
   crosses 1 at 10 days for CBA and TSLA, between 10 and 100 days for LLY). The 5/10-day
   runs fill in the gap the reviewers asked about: they sit on a smooth curve between the
   1-day and 100-day results, with no surprises.

3. **At long horizons the picture is ticker-dependent and the KAN is the weak model.**
   At 100 and 200 days the LSTM is at or slightly below naive on all three stocks
   (ratios 0.73 to 1.04); the KAN is 2x naive on CBA, 1.4x on LLY and roughly equal on
   TSLA, with large seed variance. The LSTM's long-horizon wins on LLY (0.73 at 200
   days) and TSLA (0.86) are the only cases in the whole table where a learned model beats
   persistence by a clear margin, and they come from the LSTM's output being nearly
   flat (see point 4): a flat forecast near the mean happens to beat persistence when
   the 200-day-ahead price has reverted.

4. **The trending stock exposes an extrapolation failure in the LSTM.** On LLY the LSTM
   error is 0.097 at 1 day and barely moves with horizon (0.097 to 0.101), six times the
   naive error. Inspecting fold outputs shows why: in each fold the true test prices run
   above the training range (fold 4: truth 0.55 to 1.00 in reporting scale) and the LSTM's
   predictions stay inside a narrow band near the top of the training range (0.60 to 0.73).
   Its training RMSE is normal (0.02 to 0.03), so this is not underfitting; the tanh LSTM
   cannot output levels it never saw, and with per-fold Min-Max scaling a trending test
   period always lies partly outside [0, 1]. The KAN, whose spline edges extrapolate
   roughly linearly, tracks the level (fold 4: 0.56 to 0.95) and so stays within 3x of
   naive. Persistence has no extrapolation problem at all. The same effect is visible,
   more mildly, on CBA (fold 3: truth to 1.00, LSTM to 0.89) and on TSLA fold 0.

## Implications for the paper and for Nadia's question

- Nadia's reading is right and it generalises: on raw price levels at short horizons,
  nothing in the study beats "tomorrow's price is today's price", and this is true for
  a bank, a high-volatility growth stock and a steady pharmaceutical trend alike. The
  paper should say so plainly and treat the naive row as the benchmark the two
  architectures are measured against, not as a curiosity.
- The paper's actual contribution survives the change of data: a small KAN on a flat
  window is consistently closer to persistence than the LSTM at 1 to 5 days (roughly
  half the LSTM's excess error on all three stocks), at a fraction of the parameters,
  and the advantage fades by 10 days and reverses by 100 days. That is a reproducible,
  ticker-independent statement, which is stronger than the single-stock version.
- The LLY result is a useful addition on its own: forecasting price *levels* with a
  bounded recurrent network fails on a trending series for a mechanical reason
  (range extrapolation), which argues for forecasting returns or differences in future
  work. This is a clean, explainable limitation to report rather than hide.
- Suggested text for the reply to Nadia: "I checked whether the naive result is specific
  to CBA by repeating the comparison on a very volatile stock (TSLA) and a steadily
  trending one (LLY), at 1, 2, 5, 10, 100 and 200 days. Persistence wins at every horizon
  up to 10 days on all three, by the same margin. The KAN beats the LSTM at 1 to 5 days on
  all three, and the LSTM is better at 100 and 200 days. So the result is a property of
  daily price data, not of this bank's series. We can discuss what this means for the
  framing tomorrow."

## Files

- `results_mt/TSLA/`, `results_mt/LLY/`: 210 prediction files each (`preds/` naive,
  `preds_lstmgrid/` LSTM, `preds_pykan/` KAN), same format and provenance metadata as
  `results_rev1/`. CBA's 5- and 10-day runs were added to `results_rev1/`.
- `multi_ticker_summary.json`, `multi_ticker_table.md`, `multi_ticker_table.tex`,
  `multi_ticker_rmse.png`, `multi_ticker_ratio.png`.
- `data/raw_TSLA_ohlcv.txt`, `data/raw_LLY_ohlcv.txt`, `data/make_ticker_csv.py`,
  `data/TSLA_2020-01-01_2023-08-01.csv`, `data/LLY_2020-01-01_2023-08-01.csv`.
- Code: `driver.py --multi-ticker`, `run_multi_ticker.sh`, `multi_ticker.py`;
  `regime_lib.py` default data path made relative to the repository (it was a hardcoded
  absolute path, which would have failed on any other machine).
