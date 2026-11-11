"""
make_tables.py
==============
Summarises the 180 approximation fits as tables (CSV) and Excel workbooks.
Nothing is fitted here; the script reads approximation/results/raw/fits_json/.

Output
------
results/raw/fitted_curves_csv/<ticker>__<model>__b<budget>__s<seed>.csv
    the fitted curve of every fit: t, scaled close, fit, error

results/tables/
    all_fits.csv                    one row per fit (180 rows)
    summary_by_model.csv            mean and sd over the three seeds (60 rows)
    table04_settings.csv            setting and parameter count per budget
    table14_mlp_fourier_mlp.csv     MLP and Fourier MLP side by side
    table15_kan.csv                 KAN without / with one hidden layer

results/excel/
    approximation_actual_values.xlsx  the workbook of actual values:
                                      one sheet in wide layout (one row per
                                      asset and budget) and one with all fits

Table numbers follow the manuscript of 9 November 2026. All errors are in the
Min-Max scaled [0, 1] price units of each asset.

    python make_tables.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

import run_approximation as study

TABLES_DIR = os.path.join(study.RESULTS_DIR, "tables")
EXCEL_DIR = os.path.join(study.RESULTS_DIR, "excel")
CURVES_DIR = os.path.join(study.RESULTS_DIR, "raw", "fitted_curves_csv")


# --------------------------------------------------------------------------
def load_fits():
    """Read every fit file. Returns (table without curves, list of raw records)."""
    records = [json.load(open(path)) for path in sorted(glob.glob(os.path.join(study.FITS_DIR, "*.json")))]
    assert len(records) == len(study.job_list()), "expected 180 fits, found %d" % len(records)
    table = pd.DataFrame([{
        "ticker": r["ticker"], "budget": r["budget"], "model": r["model"],
        "model_label": study.MODEL_LABELS[r["model"]], "seed": r["seed"],
        "parameters": r["params"], "rmse": r["rmse"], "max_error": r["max_err"],
        "seconds": r["seconds"], "setting": json.dumps(r["setting"]),
    } for r in records])
    order = {name: i for i, name in enumerate(study.TICKERS)}
    model_order = {name: i for i, name in enumerate(study.MODELS)}
    table = table.sort_values(
        by=["ticker", "budget", "model", "seed"],
        key=lambda column: column.map(order) if column.name == "ticker"
        else column.map(model_order) if column.name == "model" else column).reset_index(drop=True)
    return table, records


def summarise(fits):
    """Mean and sample standard deviation over the three seeds of each cell."""
    grouped = fits.groupby(["ticker", "budget", "model"], sort=False)
    summary = grouped.agg(
        model_label=("model_label", "first"), parameters=("parameters", "first"),
        seeds=("seed", "count"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        max_error_mean=("max_error", "mean"), max_error_sd=("max_error", "std"),
        seconds_mean=("seconds", "mean"), seconds_sd=("seconds", "std")).reset_index()
    return summary


def side_by_side(summary, left, right):
    """Two models next to each other, one row per asset and budget."""
    def block(model, tag):
        part = summary[summary.model == model][
            ["ticker", "budget", "parameters", "rmse_mean", "max_error_mean", "seconds_mean"]]
        return part.rename(columns={c: "%s_%s" % (tag, c) for c in part.columns if c not in ("ticker", "budget")})
    return block(left, left).merge(block(right, right), on=["ticker", "budget"])


def settings_table():
    """Setting used at each budget and the resulting parameter count N."""
    rows = []
    for budget in study.BUDGETS:
        g, h, w = study.KAN11_GRID[budget], study.KAN1H1_HIDDEN[budget], study.MLP_WIDTH[budget]
        m, wf = study.FMLP_SETTING[budget]
        rows.append({"budget": budget,
                     "kan11_grid_G": g, "kan11_N": g + 9,
                     "kan1h1_hidden_h": h, "kan1h1_N": 24 * h,
                     "mlp_width_w": w, "mlp_N": w * w + 4 * w + 1,
                     "fmlp_frequencies_m": m, "fmlp_width_w": wf, "fmlp_N": wf * wf + (2 * m + 3) * wf + 1})
    return pd.DataFrame(rows)


def write_curves(records):
    """One CSV per fit with the fitted curve, for plotting elsewhere."""
    os.makedirs(CURVES_DIR, exist_ok=True)
    series = {ticker: study.load_series(ticker) for ticker in study.TICKERS}
    for r in records:
        t, y = series[r["ticker"]]
        fit = np.array(r["pred"])
        name = "%s__%s__b%d__s%d.csv" % (r["ticker"], r["model"], r["budget"], r["seed"])
        pd.DataFrame({"t": t, "close_scaled": y, "fit": fit, "error": fit - y}).to_csv(
            os.path.join(CURVES_DIR, name), index=False, float_format="%.6f")


def wide_layout(summary):
    """One row per asset and budget; six columns per model (as in the paper)."""
    columns = ["parameters", "rmse_mean", "rmse_sd", "max_error_mean", "max_error_sd", "seconds_mean"]
    wide = None
    for model in study.MODELS:
        part = summary[summary.model == model][["ticker", "budget"] + columns]
        part = part.rename(columns={c: "%s | %s" % (study.MODEL_LABELS[model], c) for c in columns})
        wide = part if wide is None else wide.merge(part, on=["ticker", "budget"])
    return wide


def tidy_workbook(path):
    """Freeze the header row and widen the columns of every sheet."""
    from openpyxl import load_workbook
    workbook = load_workbook(path)
    for sheet in workbook:
        sheet.freeze_panes = "A2"
        for column in sheet.columns:
            width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column[:50])
            sheet.column_dimensions[column[0].column_letter].width = min(max(10, width + 2), 45)
            for cell in column[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"
    workbook.save(path)


# --------------------------------------------------------------------------
def main():
    for folder in (TABLES_DIR, EXCEL_DIR):
        os.makedirs(folder, exist_ok=True)
    fits, records = load_fits()
    summary = summarise(fits)

    # the parameter counts in the files must equal the formulas of the paper
    check = settings_table().set_index("budget")
    for _, row in summary.iterrows():
        assert row.parameters == check.loc[row.budget, row.model + "_N"], row

    tables = {
        "all_fits": fits,
        "summary_by_model": summary,
        "table04_settings": settings_table(),
        "table14_mlp_fourier_mlp": side_by_side(summary, "mlp", "fmlp"),
        "table15_kan": side_by_side(summary, "kan11", "kan1h1"),
    }
    for name, table in tables.items():
        table.to_csv(os.path.join(TABLES_DIR, name + ".csv"), index=False, float_format="%.6f")
        print("wrote tables/%-28s %4d rows" % (name + ".csv", len(table)))

    write_curves(records)
    print("wrote raw/fitted_curves_csv/         %4d files" % len(records))

    with pd.ExcelWriter(os.path.join(EXCEL_DIR, "approximation_actual_values.xlsx")) as workbook:
        wide_layout(summary).to_excel(workbook, sheet_name="Actual values", index=False)
        summary.to_excel(workbook, sheet_name="Summary by model", index=False)
        fits.drop(columns=["setting"]).to_excel(workbook, sheet_name="Per seed (180 fits)", index=False)
        settings_table().to_excel(workbook, sheet_name="Settings", index=False)
    tidy_workbook(os.path.join(EXCEL_DIR, "approximation_actual_values.xlsx"))
    print("wrote excel/approximation_actual_values.xlsx")


if __name__ == "__main__":
    main()
