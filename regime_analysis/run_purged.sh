#!/usr/bin/env bash
# Purged re-run of the multi-ticker grid (naive / LSTM L2-20-tanh / KAN pykan,
# horizons 1,2,5,10,100,200, 3 seeds, 5 folds). Identical to run_multi_ticker.sh
# except REGIME_PURGE=1: training sequences whose labels fall after the first
# test origin are dropped and the scaler is fitted on the same reduced range
# (see regime_lib.make_fold). Outputs go to results_purged/<TICKER>/.
#
#   BUDGET=110 bash run_purged.sh CBA      # resumable; repeat until "pending 0"
set -u
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=3 REGIME_PURGE=1
BUDGET=${BUDGET:-100000}
WORKERS=${WORKERS:-2}
for T in ${@:-CBA TSLA LLY}; do
  if [ "$T" = CBA ]; then CSV=../v0.2/MLP/data/CBA.AX_2020-01-01_2023-08-01.csv
  else CSV=../data/${T}_2020-01-01_2023-08-01.csv; fi
  DIR=results_purged/$T; mkdir -p "$DIR"
  for ((w=0; w<WORKERS; w++)); do
    REGIME_DATA_CSV=$CSV REGIME_RESULTS_DIR=$DIR python driver.py --multi-ticker --threads 1 \
      --budget "$BUDGET" --workers "$WORKERS" --worker "$w" > "$DIR/log_w$w.txt" 2>/dev/null &
  done
  wait
  REGIME_DATA_CSV=$CSV REGIME_RESULTS_DIR=$DIR python driver.py --multi-ticker --status 2>/dev/null
done
