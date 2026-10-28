# Approximation study: raw data and figures

- fits/*.json            one file per fit (210): ticker, model, budget, seed, params, rmse, max_err,
                         seconds, extra (knot positions / Fourier settings) and `pred`, the fitted
                         value at every day (same order as the price file).
- fits_csv/*.csv         the same fits as CSV: t, close_scaled, fit, error (= fit - close).
- all_fits_summary.csv   one row per fit with the summary numbers.
- approx_summary.json    mean/sd over seeds per (ticker, model, budget); approx_table.md the same as a table.
- pdf_figs/*.png         every figure used in the three PDFs (error-vs-parameters, all fit/error panels, knot maps).
- approx_rmse_vs_params.png, approx_fits_cba_b60.png   the two overview figures from APPROX_REPORT.md.
- approx_study.py (fits), aggregate_approx.py (tables/figures), build_pdfs.py (PDFs) reproduce everything.

Naming: <ticker>__<model>__b<budget>__s<seed>; models lsq, fks, kan11, kan1h1, mlp, fmlp.
Prices are in the file's whole-series Min-Max scale, t in [0,1] over 2020-01-02 .. 2023-07-31.
