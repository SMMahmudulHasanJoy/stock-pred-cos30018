"""
make_tables_figures.py
======================
Tables and figures of the motivation example (Section 4.4), built from the
fits stored by run_motivation_example.py. Nothing is fitted here.

Output (motivation_example/results/)
------------------------------------
tables/
    table05_experiment_A_linear.csv        single edge, k = 1
    table06_experiment_B_cubic.csv         single edge, k = 3
    table07_experiment_C_hidden_layer.csv  one hidden layer, k = 1
    all_fits.csv                           one row per fit (28 rows)
    S1_S2_values.csv                       t, S1(t), S2(t)
    motivation_example_tables.xlsx         the tables above, one sheet each
figures/
    target_functions/S1_and_S2.png         the two splines (Fig. 2)
    error_vs_grid/error_vs_grid.png        max error against G, experiments A and B
    fitted_curves/experiment_<A|B|C>/...   target, fit and error of every fit

"worst" in the tables is the largest absolute deviation over the 1001 points.

    python make_tables_figures.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_motivation_example as example

TABLES_DIR = os.path.join(example.RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(example.RESULTS_DIR, "figures")
COLOURS = {"S1": "#2f6f9f", "S2": "#c46a3f"}


def load_fits():
    records = [json.load(open(path)) for path in sorted(glob.glob(os.path.join(example.RAW_DIR, "*.json")))]
    assert len(records) == len(example.job_list()), "expected 28 fits, found %d" % len(records)
    return records


def experiment_table(records, label):
    """One row per network: errors for S1 and S2 side by side."""
    rows = {}
    for r in records:
        if r["experiment"] != label:
            continue
        key = (r["hidden_nodes"], r["grid"])
        row = rows.setdefault(key, {"hidden_nodes": r["hidden_nodes"], "subintervals_G": r["grid"],
                                    "parameters": r["params"]})
        row[r["function"] + "_worst"] = r["max_err"]
        row[r["function"] + "_rmse"] = r["rmse"]
    table = pd.DataFrame([rows[key] for key in sorted(rows)])
    if label != "C":
        table = table.drop(columns=["hidden_nodes"])
    return table


def save(figure, subfolder, name):
    folder = os.path.join(FIGURES_DIR, subfolder)
    os.makedirs(folder, exist_ok=True)
    figure.tight_layout()
    figure.savefig(os.path.join(folder, name), dpi=170)
    plt.close(figure)


def figure_targets():
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for axis, name in zip(axes, example.TARGETS):
        axis.plot(example.T, example.TARGETS[name], color=COLOURS[name])
        for knot in example.KNOTS[name]:
            axis.axvline(knot, color="grey", linewidth=0.5, alpha=0.6)
        axis.set_title("%s: knots at %s" % (name, example.KNOTS[name]), fontsize=10)
        axis.set_xlabel("t")
        axis.grid(alpha=0.3)
    save(figure, "target_functions", "S1_and_S2.png")


def figure_error_vs_grid(records):
    figure, axis = plt.subplots(figsize=(6, 3.8))
    for name in example.TARGETS:
        for label, k, style in (("A", 1, "-"), ("B", 3, "--")):
            rows = sorted((r for r in records if r["experiment"] == label and r["function"] == name),
                          key=lambda r: r["grid"])
            axis.plot([r["grid"] for r in rows], [max(r["max_err"], 1e-6) for r in rows], marker="o",
                      linestyle=style, color=COLOURS[name], label="%s, k = %d" % (name, k))
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("number of sub-intervals G"); axis.set_ylabel("worst absolute error")
    axis.set_title("Single-edge KAN: error against grid size (exact fits drawn at 1e-6)", fontsize=9)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8, frameon=False)
    save(figure, "error_vs_grid", "error_vs_grid.png")


def figure_fits(records):
    """One figure per network: S1 and S2 with their fits (top) and errors (bottom)."""
    networks = sorted({(r["experiment"], tuple(r["width"]), r["k"], r["grid"]) for r in records})
    for label, width, k, grid in networks:
        figure, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
        for column, name in enumerate(example.TARGETS):
            r = next(x for x in records if (x["experiment"], tuple(x["width"]), x["k"], x["grid"], x["function"])
                     == (label, width, k, grid, name))
            fit = np.array(r["pred"]); target = example.TARGETS[name]
            axes[0, column].plot(example.T, target, color="black", linewidth=1.6, label=name)
            axes[0, column].plot(example.T, fit, color=COLOURS[name], linestyle="--", linewidth=1.2, label="KAN fit")
            for knot in np.linspace(0, 10, grid + 1):
                axes[0, column].axvline(knot, color="grey", linewidth=0.3, alpha=0.5)
            axes[0, column].set_title("%s: worst %.4f, RMSE %.4f" % (name, r["max_err"], r["rmse"]), fontsize=9)
            axes[0, column].legend(fontsize=7, frameon=False)
            axes[1, column].plot(example.T, fit - target, color=COLOURS[name], linewidth=0.9)
            axes[1, column].axhline(0, color="black", linewidth=0.5)
            axes[1, column].set_xlabel("t"); axes[1, column].set_ylabel("fit - target", fontsize=8)
        figure.suptitle("Experiment %s: width %s, k = %d, G = %d, %d parameters (grey lines: KAN grid)"
                        % (label, list(width), k, grid, r["params"]), fontsize=10)
        name = "width_%s_G%03d.png" % ("-".join(map(str, width)), grid)
        save(figure, os.path.join("fitted_curves", "experiment_" + label), name)


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


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    records = load_fits()
    tables = {
        "table05_experiment_A_linear": experiment_table(records, "A"),
        "table06_experiment_B_cubic": experiment_table(records, "B"),
        "table07_experiment_C_hidden_layer": experiment_table(records, "C"),
        "all_fits": pd.DataFrame([{key: value for key, value in r.items() if key != "pred"} for r in records]),
        "S1_S2_values": pd.DataFrame({"t": example.T, **example.TARGETS}),
    }
    with pd.ExcelWriter(os.path.join(TABLES_DIR, "motivation_example_tables.xlsx")) as workbook:
        for name, table in tables.items():
            table.to_csv(os.path.join(TABLES_DIR, name + ".csv"), index=False, float_format="%.6f")
            table.to_excel(workbook, sheet_name=name[:31], index=False)
    tidy_workbook(os.path.join(TABLES_DIR, "motivation_example_tables.xlsx"))
    figure_targets()
    figure_error_vs_grid(records)
    figure_fits(records)
    for name in list(tables)[:3]:
        print("\n" + name + "\n" + tables[name].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
