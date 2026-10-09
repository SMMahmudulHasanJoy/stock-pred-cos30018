#!/bin/bash
# Re-run every experiment under the leak-free protocol into results_rev1/.
# Resumable: each call does up to $1 seconds of work (default 150) and exits.
cd "$(dirname "$0")"
export REGIME_RESULTS_DIR=results_rev1 PYTHONHASHSEED=0 TF_CPP_MIN_LOG_LEVEL=3
BUDGET=${1:-150}; START=$(date +%s)
left() { echo $(( BUDGET - ($(date +%s) - START) )); }
run2() {  # two workers in parallel, one thread each
  L=$(left); [ "$L" -le 8 ] && return
  OMP_NUM_THREADS=1 python3 driver.py $1 --budget $L --workers 2 --worker 0 2>&1 | grep -E "^ran" | sed "s/^/[$1 w0] /" &
  OMP_NUM_THREADS=1 python3 driver.py $1 --budget $L --workers 2 --worker 1 2>&1 | grep -E "^ran" | sed "s/^/[$1 w1] /" &
  wait
}
for FLAGS in "" "--kan-engine pykan" "--lstm-grid" "--best-check"; do
  L=$(left); [ "$L" -le 8 ] && break
  echo "[$FLAGS] $(python3 driver.py $FLAGS --budget $L 2>&1 | grep -E '^ran')"
done
run2 "--kan-grid --cost-scale 6"
run2 "--new-models"
