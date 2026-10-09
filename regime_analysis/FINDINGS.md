# Regime analysis for the KAN vs LSTM paper — findings

Prepared 28 July 2026, for the Operations Research Forum submission.

This started as the market-regime analysis needed to put Table 7 on a proper
footing. Doing it surfaced problems in the published pipeline that are more
serious than the table formatting, so those are reported first.

---

## 1. What the regime analysis found

Both architectures were retrained and rescored under a corrected protocol
(blocked walk-forward validation, identical test windows for both models,
regimes defined from the observed input window with thresholds taken from
training data only, three seeds, five folds, plus a persistence baseline).
Full numbers are in `results/table_regime_comparison.tex`; the headline figures
are pooled RMSE in Min-Max scaled space:

| Horizon | LSTM | KAN | Naive (persistence) |
|---|---|---|---|
| 1-day | 0.068 ± 0.009 | **0.055 ± 0.006** | **0.020** |
| 2-day | 0.072 ± 0.003 | **0.063 ± 0.001** | **0.025** |
| 100-day | **0.110 ± 0.009** | 0.124 ± 0.006 | 0.111 |
| 200-day | 0.153 ± 0.010 | **0.128 ± 0.002** | 0.132 |

Three conclusions:

1. **The 6.5–10× LSTM advantage does not survive matched evaluation.** The two
   models are within about one and a half seed standard deviations of each other
   at every horizon, and the ordering flips depending on horizon.
2. **Regime structure is real but modest.** Volatile windows hurt both models at
   short horizons, the LSTM more so (2-day RMSE rises 80% from Normal to
   Volatile, against 20% for the KAN). At 200 days, Trending windows are the
   hardest for both.
3. **Neither model beats persistence at 1 or 2 days,** by a factor of roughly
   three. At 100 and 200 days all three methods are comparable. This is the
   result with the most consequence for the paper's framing.

The 200-day "not tested" gap is closed: the LSTM was run at 200 days here
without difficulty, so nothing in Table 7 needs an empty cell any more.

---

## 2. Why the published numbers differ — three concrete causes

### 2.1 A tensor-shape bug inflates every KAN error

`v0.2/KANs/kan_main.py` line ~90 builds the dataset as:

```python
'train_label': torch.tensor(y_train).float().to(device).unsqueeze(1),
'test_label':  torch.tensor(y_test_subset).float().to(device).unsqueeze(1)
```

`unsqueeze(1)` turns the labels from shape `(n, h)` into `(n, 1, h)`. The KAN
outputs `(n, h)`. Under broadcasting, `pred - label` becomes `(n, n, h)`: every
prediction is compared against every label, not against its own. The reported
loss is then dominated by the cross-sectional spread of the price series rather
than by forecast error.

Demonstration with the actual CBA close series: a forecaster with a true RMSE of
**0.029** reports **0.354** through that code path — a 12× inflation, and
squarely inside the published KAN range of 0.152–0.331. The paper's claimed
"6.5 to 10 times lower error" for the LSTM is approximately the size of this
artifact.

Removing `.unsqueeze(1)` from both label tensors is the fix.

### 2.2 The KAN was scored on a handful of test rows

The same script keeps only

```python
x_test_subset = x_test[-STEPS_TO_PREDICT:]
```

so the KAN is evaluated on the last `h` rows of the test set: **one sample** at
h=1, two at h=2, one hundred at h=100. The LSTM, meanwhile, is evaluated on the
entire test set. The two models in Tables 4–6 are therefore not scored on the
same data, and at short horizons the KAN figure is a single-sample statistic.
Reruns of the original protocol give KAN "test RMSE" values of 0.014, 0.024 and
0.037 on the same configuration purely because the one or two sampled rows
changed.

### 2.3 The split shuffles overlapping windows

Both pipelines call `get_data(..., split_by_date=False)`, which routes to
`sklearn.model_selection.train_test_split`. Adjacent sequences share 19 of their
20 input days, so shuffling puts near-duplicate windows on both sides of the
split. There is also no fixed seed, which is why Table 5's standard deviations
mix initialisation variance with split variance.

Under the original protocol this reimplementation reproduces the published LSTM
figures well (1-day test RMSE 0.051 against the published 0.039–0.046 range;
100-day 0.103 against the published 0.103–0.108), which is the check that the
code here is behaving like the original. The KAN figures are the ones that do
not reproduce, for the reasons above.

---

## 3. Table 7 specifically

Separate from the pipeline issues, the published Table 7 has problems of its own:

* **Its numbers do not appear in Tables 4–6.** Table 4/5 give LSTM test RMSE
  between 0.074 and 0.118; Table 7 reports 0.039–0.085. Table 6 gives KAN test
  RMSE between 0.152 and 0.331; Table 7 reports 0.385–0.670.
* **It contradicts the paper's own text.** Section 5.5 states the best KAN test
  RMSE was 0.152 at the 1-day horizon. Table 7 reports 0.385–0.390 at that
  horizon.
* **The market conditions were never defined.** "Normal", "Volatile" and
  "Trending" appear nowhere in the paper except inside Table 7 and the sentences
  quoting it — no thresholds, no date ranges, no segmentation method.
* **The 1-day LSTM values look like repeat runs relabelled as conditions.** The
  logged runs in `v0.2/MLP/model_errors.json` contain 0.0385, 0.0400, 0.0409,
  0.0426, 0.0430, 0.0466 for the 100u-linear configuration; Table 5 summarises
  them as mean 0.0459, min 0.0385. Table 7's "Normal 0.039 / Volatile 0.045 /
  Trending 0.042" tracks those run statistics rather than any market
  segmentation.

The replacement table in `results/table_regime_comparison.tex` is generated
directly from logged per-fold predictions, so it cannot drift out of step with
the runs behind it.

---

## 4. Cross-check with real pykan

The KAN grid was run twice: once with `kan_layer.py` (a TensorFlow
reimplementation of pykan's construction, written when the PyTorch index was
briefly unreachable) and then again with **the real pykan package
(pykan 0.2.8 + torch 2.2.2)** — all 60 KAN jobs, same folds, same seeds, same
label handling (no `unsqueeze(1)`). Run `driver.py --kan-engine pykan`;
per-fold outputs are in `results/preds_pykan/`.

Pooled test RMSE, mean ± sd over three seeds:

| Horizon | pykan (all folds) | pykan (folds 1–4) | reimpl. (all folds) |
|---|---|---|---|
| 1-day | 0.041 ± 0.006 | 0.030 ± 0.002 | 0.055 ± 0.006 |
| 2-day | 0.047 ± 0.005 | 0.041 ± 0.004 | 0.063 ± 0.001 |
| 100-day | 0.446 ± 0.225 | 0.125 ± 0.006 | 0.124 ± 0.006 |
| 200-day | 0.265 ± 0.087 | 0.152 ± 0.009 | 0.128 ± 0.002 |

Two things to note:

1. **The conclusions are corroborated, and slightly strengthened.** With real
   pykan the KAN *beats* the LSTM at short horizons (0.041 vs 0.068 at 1 day)
   and matches it at long horizons on the stable folds. The published claim of
   a 6.5–10× LSTM advantage is reversed, not merely reduced. The naive
   baseline (0.020 at 1 day) still beats both.
2. **pykan's L-BFGS diverges on the smallest training set.** On fold 0 only
   (~300 training samples) at the 100- and 200-day horizons, pykan's 10-step
   full-batch L-BFGS produced predictions outside the [0, 1] data range (as
   extreme as −2.5 and 3.3), giving fold RMSEs of 0.35–1.5 and inflating the
   pooled figures at those horizons. Folds 1–4 are stable and agree with the
   reimplementation to within seed noise. This is a genuine property of the
   published training configuration (10 L-BFGS steps, no regularisation or
   early stopping) and belongs in the paper's limitations; the "folds 1–4"
   column shows the behaviour when training data is adequate.

The reimplementation and pykan tell the same story, so either can back the
table; the paper should use the pykan numbers (`results/summary_pykan.json`)
since pykan is what the paper's methods section describes, with the fold-0
divergence disclosed.

---

## 4b. Table 6 re-run (KAN configuration grid)

The six Table 6 configurations were re-run with pykan under the corrected
walk-forward protocol (1-day horizon, 3 seeds x 5 folds; `driver.py --kan-grid`,
outputs in `results/preds_kangrid/`, summary in `results/summary_kangrid.json`):

| Config | grid | k | hidden | Train RMSE | Test RMSE | published test |
|---|---|---|---|---|---|---|
| C1 | 3 | 6 | ⌊N/10⌋ | 0.0161 ± 0.0003 | **0.0750 ± 0.0142** | 0.188 |
| C2 | 3 | 2 | ⌊N/10⌋ | 0.0130 ± 0.0001 | 0.0368 ± 0.0025 | 0.188 |
| C3 | 7 | 2 | ⌊N/10⌋ | 0.0053 ± 0.0002 | 0.0385 ± 0.0006 | 0.331 |
| C4 | 3 | 2 | ⌊N/4⌋ | 0.0119 ± 0.0002 | 0.0385 ± 0.0017 | 0.238 |
| C5 | 3 | 2 | ⌊N/10⌋ | 0.0130 ± 0.0001 | 0.0358 ± 0.0040 | 0.152 |
| C6 | 3 | 2 | ⌊N/5⌋ | 0.0121 ± 0.0001 | 0.0418 ± 0.0055 | 0.152 |

The published hyperparameter conclusions do not survive: k=6 is now clearly the
worst configuration (not tied with k=2), grid 7 is unremarkable (not
catastrophic), and width barely matters. C2 and C5 — identical specifications
that the published table reported as 0.188 vs 0.152 — now agree to within seed
noise, which is a direct consistency check on the corrected pipeline. Table 6
and Figure 3 in the Overleaf paper have been regenerated from these numbers
(the figure is now drawn with pgfplots directly from the data).

## 4c. Reproducibility of the pykan runs (added after independent verification)

Tabish re-ran the 60 pykan jobs on his own machine. His pooled results
match ours (1-day 0.0400 vs 0.0409, 2-day 0.0470 vs 0.0473; with the
divergent first fold excluded, 100-day 0.1294 vs 0.1250 and 200-day
0.1511 vs 0.1516), and he reproduces the fold-0 L-BFGS divergence at both
long horizons. His per-fold predictions, however, match none of ours
exactly, with per-window differences up to 1.3 in scaled units on the
divergent folds.

Chasing that down led somewhere worth recording. In order of discovery:

1. Our scripts seeded torch but never numpy, and pykan's `fit()` permutes
   the training rows each step through numpy's global generator
   (`np.random.choice`). Fixed: both RNGs are now seeded per job.
2. pykan's fit is also sensitive to Python's per-process hash
   randomisation. The drivers now pin `PYTHONHASHSEED=0`, re-executing
   themselves once if needed, since the variable only takes effect at
   interpreter startup.
3. Even with both fixed, every seed pinned, and every thread pool
   (torch, OMP, OpenBLAS, MKL) forced to a single thread, the same seeded
   fit still produces different results in different process launches.
   The divergence enters in `update_grid`, whose `curve2coef` solves a
   least-squares system with `torch.linalg.lstsq` on nearly
   rank-deficient spline collocation matrices. Many coefficient vectors
   fit almost equally well; which one the solver returns varies from
   launch to launch, and ten steps of full-batch L-BFGS amplify the
   difference. This is inside pykan and LAPACK, not our code.

The practical consequences:

- Exact per-file reproduction is the wrong acceptance test for the pykan
  runs. Statistical agreement is the right one, and the independent run
  passes it: pooled short-horizon RMSE agrees to about 0.003, and the
  qualitative findings (KAN competitive at short horizons, fold-0
  divergence at long horizons) reproduce exactly.
- The seed-to-seed spreads quoted for KAN understate total run-to-run
  variability at the 100- and 200-day horizons, where launch-to-launch
  differences of roughly 0.03 to 0.04 in pooled RMSE sit on top of seed
  variance. The 1- and 2-day figures are stable to well within the
  precision quoted in the paper.
- The LSTM and custom-engine results are unaffected: they reproduce
  byte-identically under the same protocol.

## 6. Revision runs: leak-free scaling, and the flattening question

Both referees asked for the Min-Max scaler to be fitted on training data
only. Referee 1 also asked that the flattened-input question be addressed,
either with a like-for-like baseline (an MLP on the same flat vector) or a
KAN with a recurrent wrapper. Both are now done; everything below is from
`results_rev1/` (600 runs, 2.4 h of compute).

**The fix.** `regime_lib.make_fold` fits Min-Max on the rows a fold's
training sequences can see, trains and predicts in that fold's scale, and
maps predictions back to the file's whole-series scale before anything is
saved, so numbers stay comparable with the submitted tables. The reporting
scale is a unit convention applied after prediction; nothing about it
reaches the model. Checks: naive is unchanged to float32 precision, and
fold 4 (whose training rows already span the whole range) reproduces the
submitted LSTM bit for bit.

**Effect on the submitted results (pooled RMSE, mean over 3 seeds):**

| model | h=1 | h=2 | h=100 | h=200 |
|---|---|---|---|---|
| LSTM (2-layer, 20-unit) | 0.0537 -> 0.0586 (+9%) | 0.0528 -> 0.0579 (+10%) | 0.1144 -> 0.1079 (-6%) | 0.1406 -> 0.1375 (-2%) |
| KAN (PyKAN) | 0.0409 -> 0.0338 (-17%) | 0.0473 -> 0.0475 (0%) | 0.4462 -> 0.2418 (-46%) | 0.2654 -> 0.2764 (+4%) |
| Naive | unchanged | unchanged | unchanged | unchanged |

The conclusions survive. KAN's short-horizon advantage widens (0.034 vs
0.059 at 1 day; DM = -11.3, p < 0.001). Naive still beats every learned
model at 1 and 2 days, with DM statistics between +5 and +13. At 100 and
200 days no learned model is distinguishable from naive (all p > 0.2). KAN
still diverges on the earliest training fold at the long horizons, though
less violently than before.

**The two new models (same protocol, 3 seeds x 5 folds):**

| model | input | params (h=1) | h=1 | h=2 | h=100 | h=200 |
|---|---|---|---|---|---|---|
| MLP (2x32 tanh, Adam, early stop) | flat, as KAN | 4,321 | 0.0337 | 0.0366 | 0.1237 | 0.1154 |
| Recurrent KAN (KANLayer as RNN cell, RMSprop 25 ep) | sequence, as LSTM | 4,181 | 0.0448 | 0.0412 | 0.1236 | 0.1626 |
| KAN (PyKAN), for reference | flat | 54,540 | 0.0338 | 0.0475 | 0.2418 | 0.2764 |
| LSTM (2-layer, 20-unit), for reference | sequence | 5,381 | 0.0586 | 0.0579 | 0.1079 | 0.1375 |

What this says about referee 1's comment:

1. *Flattening is not what separates KAN from LSTM.* A plain MLP on the
   identical flat vector matches KAN at 1 day (0.0337 vs 0.0338) and beats
   it at 2 days (DM p = 0.012), and both flat models beat the order-aware
   LSTM at short horizons by a wide margin. If losing temporal order were
   the handicap, the flat MLP could not do this.
2. *Giving KAN recurrence does not rescue it.* The recurrent KAN is worse
   than flat KAN at 1 day (0.045 vs 0.034, p < 0.001), better at 2 days,
   level with the MLP at 100 days and worst of all at 200 days. It does
   beat the LSTM at 1 and 2 days (p < 0.001), so a recurrent cell built
   from KAN layers is competitive with an LSTM cell, which is a result in
   its own right, but recurrence is not the missing ingredient.
3. The honest reframing the referee offered is the right one: at short
   horizons on this data, static approximation on a flat window is at
   least as good as recurrence, and nothing beats persistence.

Caveat on training budgets: the MLP is trained with Adam and early
stopping (up to 200 epochs); LSTM and recurrent KAN use the paper's
RMSprop 25-epoch budget; KAN its 10 L-BFGS steps. A pilot of the MLP on
the LSTM's budget scored 0.098 at 1 day, three times worse, so training
budget matters at least as much as architecture here. The paper should
say so rather than hide it.

Re-run: `bash run_revision.sh` (resumable; call until every list reports
0 remaining). Compare: `python compare_revision.py`.

## 5. What I would suggest to the team

The cosmetic Table 7 fix can go in immediately. The rest is a bigger
conversation, because it affects the paper's central claim and the paper is
already public as arXiv:2511.18613v2.

Options, roughly in order of how much they cost:

1. **Fix the KAN pipeline and re-run everything**, then rewrite the comparison
   around the corrected findings — the two architectures are comparable, and
   neither beats persistence at short horizons. This is a more modest claim but
   a defensible one, and the regime analysis and naive baseline both strengthen
   it. It requires an arXiv v3 with a clear change note.
2. **Fix the pipeline but narrow the paper's scope** to the methodological
   lesson: how easily a shape bug and an unmatched test set can manufacture an
   order-of-magnitude result, with the corrected numbers as evidence. Journals
   in this space do publish that kind of paper.
3. **Submit with Table 7 repaired but the pipeline untouched.** I would not
   recommend this. A referee who reads `kan_main.py` will find the same things,
   and the internal contradiction between Table 7 and Section 5.5 is visible
   without running anything.

Whatever is decided, the reproduction script (`replicate_published.py`) and the
corrected harness are both here, so the difference between the two protocols can
be shown to reviewers rather than described.

---

## Files

| File | What it is |
|---|---|
| `regime_lib.py` | Data loading, sequence construction, regime rule, metrics |
| `kan_layer.py` | KAN implementation (spline edges + L-BFGS) |
| `driver.py` | Resumable per-fold job runner; writes `results/preds/*.npz` |
| `aggregate.py` | Pools folds, computes statistics, emits the LaTeX table and figure |
| `replicate_published.py` | Reruns the original protocol for comparison |
| `paper_sections.tex` | Drop-in methodology, results and limitations text |
| `results/table_regime_comparison.tex` | Generated replacement for Table 7 |
| `results/regime_comparison_grid.png` | Generated replacement comparison figure |
| `results/summary.json` | Every reported statistic, machine readable |
| `results/preds/` | 150 per-fold prediction files, one per run |
