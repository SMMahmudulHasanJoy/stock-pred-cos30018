# Data

Daily data of the three assets used in the paper, 2 January 2020 to the end of July 2023.

| File | Asset | Days | Notes |
|---|---|---|---|
| `CBA.AX_2020-01-01_2023-08-01.csv` | Commonwealth Bank of Australia | 905 | ASX calendar, last day 31 July 2023 |
| `TSLA_2020-01-01_2023-08-01.csv` | Tesla | 901 | NYSE calendar, last day 1 August 2023 |
| `LLY_2020-01-01_2023-08-01.csv` | Eli Lilly | 901 | NYSE calendar, last day 1 August 2023 |

Each file has the columns `Date, Open, High, Low, Close, Volume`. Every column is Min-Max scaled to [0, 1] over the whole period, so all errors reported in this repository are in these scaled units (1.0 = the full range of the asset over the period). This scale is only a reporting unit: in the forecasting study the models are trained on data re-scaled from the training rows of each fold.

For reference, the closing-price ranges are 24.08 to 409.97 USD for TSLA and 119.05 to 468.98 USD for LLY. The CBA.AX file was stored in scaled form only.

`raw_TSLA_ohlcv.txt` and `raw_LLY_ohlcv.txt` are the unscaled values downloaded from Yahoo Finance; `make_ticker_csv.py` turns them into the two CSV files.
