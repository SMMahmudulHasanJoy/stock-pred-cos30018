#!/usr/bin/env bash
# Multi-ticker robustness check: naive / LSTM (L2-20-tanh) / KAN (pykan)
# at horizons 1,2,5,10,100,200, three seeds, five folds, per ticker.
# Resumable: driver.py skips any job whose output file already exists, so
# the script can be re-run until every ticker reports "pending 0".
#
#   bash run_multi_ticker.sh                 # run everything (CBA, TSLA, LLY)
#   bash run_multi_ticker.sh TSLA            # one ticker
#   BUDGET=500 bash run_multi_ticker.sh LLY  # stop starting new jobs after ~500 s
#
# CBA reuses the 1/2/100/200-day runs already in results_rev1 and only adds
# the 5- and 10-day horizons; TSLA and LLY go to results_mt/<TICKER>/.
set -u
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=3
BUDGET=${BUDGET:-100000}
TICKERS=${@:-CBA TSLA LLY}
for T in $TICKERS; do
  if [ "$T" = CBA ]; then
    CSV=../v0.2/MLP/data/CBA.AX_2020-01-01_2023-08-01.csv; DIR=results_rev1
  else
    CSV=../data/${T}_2020-01-01_2023-08-01.csv; DIR=results_mt/$T
  fi
  mkdir -p "$DIR"
  echo "== $T -> $DIR"
  REGIME_DATA_CSV=$CSV REGIME_RESULTS_DIR=$DIR \
    python driver.py --multi-ticker --threads 1 --budget "$BUDGET" 2>/dev/null
  REGIME_DATA_CSV=$CSV REGIME_RESULTS_DIR=$DIR python driver.py --multi-ticker --status 2>/dev/null
done
