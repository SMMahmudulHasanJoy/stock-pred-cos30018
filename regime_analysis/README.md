# regime_analysis

Regime-conditional re-evaluation of the KAN vs LSTM comparison, built to replace
Table 7 of the Operations Research Forum submission with measured results.

Read `FINDINGS.md` first — it explains what the analysis found and why some of
the published numbers do not reproduce.

## Revision (October 2026)

`results_rev1/` holds the re-run with the scaler fitted per fold on
training rows only (the data-leakage point both referees raised) plus two
models added for referee 1's major comment 2: an MLP on the same flat
vector KAN receives, and a recurrent KAN. `bash run_revision.sh` rebuilds
it; `python compare_revision.py` prints the before/after tables. Details
and numbers: FINDINGS.md section 6. `results/` is the submitted run, kept
for comparison.

## Reproducing

**Read this before running anything.** The repository ships with a populated
`results/` directory holding the runs behind the paper's tables, and
`driver.py` skips any job whose output file already exists. On a fresh clone
that means every job counts as done: the driver writes nothing, and
`aggregate.py` simply re-derives the shipped tables from the shipped files.
To produce results of your own, point the pipeline at an empty directory:

```bash
REGIME_RESULTS_DIR=results_mine python driver.py --status   # should say: done 0
REGIME_RESULTS_DIR=results_mine python driver.py --budget 600
REGIME_RESULTS_DIR=results_mine python aggregate.py
python provenance.py --dir results_mine --compare results   # whose files are whose
```

(On Windows, set the variable first: `set REGIME_RESULTS_DIR=results_mine`.)
Every script that reads or writes results honours the same variable. Without
it, the scripts operate on the shipped `results/`, which is the right mode for
regenerating tables and figures from the recorded runs, and the wrong mode for
independent verification.

`provenance.py` reports whether the files in a results directory came with the
clone or were written locally, and compares two directories job by job. When
comparing, expect the LSTM and custom-engine files to match exactly; the pykan
files will differ per fold even on the same machine, because pykan's grid
update is not deterministic across process launches (`FINDINGS.md` 4c). Judge
pykan reproductions on pooled RMSE, not per-file identity.

```bash
python replicate_published.py      # rerun the ORIGINAL protocol for comparison
```

`driver.py` is resumable: it writes one `.npz` per (model, config, horizon, seed,
fold) and skips anything already there, so it can be run in short bursts or
restarted after an interruption. `--budget` is a wall-clock allowance in
seconds and `--workers N --worker i` splits the grid across processes.

The grid is 150 runs: naive baseline (4 horizons x 5 folds), KAN
(4 x 3 seeds x 5 folds), LSTM at its per-horizon configuration
(4 x 3 x 5) and a single-seed configuration cross-check at 1 and 2 days.

## Protocol summary

| | Published pipeline | Here |
|---|---|---|
| Split | `train_test_split`, shuffled, no seed | blocked walk-forward, 5 folds |
| Test set | LSTM: all of it. KAN: last `h` rows | identical windows for both |
| Regimes | undefined | volatility + efficiency ratio, train-only thresholds |
| Baseline | none | persistence (repeat last close) |
| 200-day | LSTM not run | both models run |
| Seeds | none fixed | 0, 100, 200 |

Regime rule, applied to the 20-day input window only (no look-ahead):

* `Volatile` if window volatility exceeds the training-set 75th percentile
* `Trending` else if Kaufman efficiency ratio exceeds the training-set 75th percentile
* `Normal` otherwise

## KAN engines

The KAN grid exists twice, and the two engines corroborate each other (see
`FINDINGS.md` section 4):

* `results/preds/` — `kan_layer.py`, a TensorFlow reimplementation of pykan's
  construction (run with the default `--kan-engine custom`)
* `results/preds_pykan/` — **the real pykan package** (pykan 0.2.8,
  torch 2.2.2), run with `--kan-engine pykan`; summary in
  `results/summary_pykan.json`

The paper should quote the pykan numbers, with the fold-0 L-BFGS divergence at
long horizons disclosed (documented in `FINDINGS.md`). The LSTM half uses Keras
exactly as the original did in both cases.

## Requirements

Python 3.10, `numpy`, `pandas`, `scikit-learn`, `matplotlib`,
`tensorflow-cpu==2.16.2`, plus `pykan==0.2.8` and `torch==2.2.2` for the pykan
engine.

Data is read from `../v0.2/MLP/data/CBA.AX_2020-01-01_2023-08-01.csv`; override
with the `REGIME_DATA_CSV` environment variable.
