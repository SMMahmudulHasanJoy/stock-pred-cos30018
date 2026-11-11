"""
make_ticker_csv.py
==================
Builds the TSLA and LLY data files from the raw daily values, in the same
format as the CBA.AX file: Date, Open, High, Low, Close, Volume, with every
column Min-Max scaled to [0, 1] over the whole period.

Source: Yahoo Finance daily data, 2 January 2020 to 1 August 2023
(901 NYSE trading days). raw_<TICKER>_ohlcv.txt holds Open, High, Low, Close
and Volume (in thousands), one row per trading day, without dates; the NYSE
calendar is rebuilt here and checked to have 901 days. Because every column
is rescaled, the original units do not matter.

    python make_ticker_csv.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
START, END = "2020-01-01", "2023-08-01"
NYSE_HOLIDAYS = """
2020-01-01 2020-01-20 2020-02-17 2020-04-10 2020-05-25 2020-07-03 2020-09-07 2020-11-26 2020-12-25
2021-01-01 2021-01-18 2021-02-15 2021-04-02 2021-05-31 2021-07-05 2021-09-06 2021-11-25 2021-12-24
2022-01-17 2022-02-21 2022-04-15 2022-05-30 2022-06-20 2022-07-04 2022-09-05 2022-11-24 2022-12-26
2023-01-02 2023-01-16 2023-02-20 2023-04-07 2023-05-29 2023-06-19 2023-07-04
""".split()
COLS = ["Open", "High", "Low", "Close", "Volume"]


def trading_days():
    days = pd.bdate_range("2020-01-02", "2023-08-01")
    return [d for d in days if d.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS]


def build(ticker):
    raw = np.loadtxt(os.path.join(HERE, "raw_%s_ohlcv.txt" % ticker), delimiter=",")
    days = trading_days()
    assert len(days) == len(raw) == 901, (len(days), len(raw))
    df = pd.DataFrame(raw, columns=COLS)
    for c in COLS:
        lo, hi = df[c].min(), df[c].max()
        df[c] = (df[c] - lo) / (hi - lo)
    df.insert(0, "Date", [d.strftime("%Y-%m-%d") for d in days])
    out = os.path.join(HERE, "%s_%s_%s.csv" % (ticker, START, END))
    df.to_csv(out, index=False)
    print("wrote", out, df.shape)


if __name__ == "__main__":
    for t in ["TSLA", "LLY"]:
        build(t)
