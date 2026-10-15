"""
multi_ticker.py
Compare naive, LSTM (L2-20-tanh) and KAN (pykan) errors across tickers and
horizons from the runs produced by run_multi_ticker.sh.

For every (ticker, model, horizon, seed) the five fold predictions are pooled
and RMSE is computed in the ticker's reporting scale (whole-series Min-Max of
that ticker's CSV, the same convention as the paper). Mean and sample sd over
the three seeds are reported. Because each ticker has its own scale, the
numbers are comparable *within* a ticker (model vs model) and, for the ratio
columns, across tickers; raw RMSEs should not be compared across tickers.

Outputs (in results_mt/):
    multi_ticker_summary.json   every number
    multi_ticker_table.md       markdown table, one block per ticker
    multi_ticker_table.tex      LaTeX version
    multi_ticker_rmse.png       RMSE vs horizon, one panel per ticker
    multi_ticker_ratio.png      model RMSE / naive RMSE, all tickers

    python multi_ticker.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_mt")
HORIZONS = [1, 2, 5, 10, 100, 200]
SEEDS = [0, 100, 200]
TICKERS = {
    "CBA.AX": {"dir": os.path.join(HERE, "results_rev1"), "label": "CBA (bank, benchmark)"},
    "TSLA": {"dir": os.path.join(OUT, "TSLA"), "label": "TSLA (volatile)"},
    "LLY": {"dir": os.path.join(OUT, "LLY"), "label": "LLY (stable trend)"},
}
MODELS = {
    "Naive": ("preds", "naive__-__h%d__s0__f%d.npz"),
    "LSTM": ("preds_lstmgrid", "lstmcfg__L2-20-tanh__h%d__s%d__f%d.npz"),
    "KAN": ("preds_pykan", "kan__-__h%d__s%d__f%d.npz"),
}


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def run_rmse(directory, model, horizon, seed):
    sub, pattern = MODELS[model]
    preds, truths = [], []
    for fold in range(5):
        name = pattern % ((horizon, fold) if model == "Naive" else (horizon, seed, fold))
        path = os.path.join(directory, sub, name)
        if not os.path.exists(path):
            return None
        with np.load(path, allow_pickle=False) as d:
            preds.append(d["prediction"])
            truths.append(d["truth"])
    return rmse(np.concatenate(preds).ravel(), np.concatenate(truths).ravel())


def collect():
    summary = {}
    for ticker, spec in TICKERS.items():
        summary[ticker] = {}
        for horizon in HORIZONS:
            block = {}
            for model in MODELS:
                seeds = [0] if model == "Naive" else SEEDS
                vals = [run_rmse(spec["dir"], model, horizon, s) for s in seeds]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                block[model] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                    "per_seed": vals,
                }
            if "Naive" in block:
                for model in ("LSTM", "KAN"):
                    if model in block:
                        block[model]["ratio_to_naive"] = block[model]["mean"] / block["Naive"]["mean"]
            summary[ticker][horizon] = block
    return summary


def fmt(block, model):
    if model not in block:
        return "--"
    b = block[model]
    if model == "Naive":
        return "%.4f" % b["mean"]
    return "%.4f ± %.4f" % (b["mean"], b["std"])


def markdown(summary):
    lines = []
    for ticker, spec in TICKERS.items():
        lines.append("### %s" % spec["label"])
        lines.append("")
        lines.append("| Horizon | Naive | LSTM | KAN | LSTM/Naive | KAN/Naive | KAN/LSTM | Best |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            b = summary[ticker].get(h, {})
            if not b:
                continue
            means = {m: b[m]["mean"] for m in b}
            best = min(means, key=means.get)
            r_l = "%.2f" % b["LSTM"]["ratio_to_naive"] if "LSTM" in b and "Naive" in b else "--"
            r_k = "%.2f" % b["KAN"]["ratio_to_naive"] if "KAN" in b and "Naive" in b else "--"
            r_kl = "%.2f" % (b["KAN"]["mean"] / b["LSTM"]["mean"]) if "KAN" in b and "LSTM" in b else "--"
            lines.append("| %d-day | %s | %s | %s | %s | %s | %s | %s |"
                         % (h, fmt(b, "Naive"), fmt(b, "LSTM"), fmt(b, "KAN"), r_l, r_k, r_kl, best))
        lines.append("")
    lines.append("RMSE in each ticker's Min-Max reporting scale (mean ± sd over 3 seeds, "
                 "5 pooled walk-forward folds). Ratios are comparable across tickers; raw RMSEs are not.")
    return "\n".join(lines)


def latex(summary):
    lines = [r"\begin{table}[!ht]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{|l|l|c|c|c|c|c|}", r"\hline",
             r"\textbf{Ticker} & \textbf{Horizon} & \textbf{Naive} & \textbf{LSTM} & "
             r"\textbf{KAN} & \textbf{LSTM/Naive} & \textbf{KAN/Naive} \\", r"\hline"]
    for ticker, spec in TICKERS.items():
        first = True
        for h in HORIZONS:
            b = summary[ticker].get(h, {})
            if not b:
                continue
            label = r"\multirow{%d}{*}{%s}" % (len(summary[ticker]), ticker) if first else ""
            first = False
            cells = [fmt(b, "Naive")]
            for m in ("LSTM", "KAN"):
                cells.append("$%.3f \\pm %.3f$" % (b[m]["mean"], b[m]["std"]) if m in b else "--")
            for m in ("LSTM", "KAN"):
                cells.append("%.2f" % b[m]["ratio_to_naive"] if m in b else "--")
            lines.append("  %s & %d-day & %s \\\\" % (label, h, " & ".join(cells)))
        lines.append(r"\hline")
    lines += [r"\end{tabular}",
              r"\caption{Naive, LSTM and KAN test RMSE (Min-Max scale of each ticker, mean $\pm$ sd "
              r"over three seeds, five blocked walk-forward folds) at six horizons for three tickers. "
              r"Ratios below 1 mean the model beats persistence.}",
              r"\label{table_multi_ticker}", r"\end{table}"]
    return "\n".join(lines)


def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"Naive": "#9a9a9a", "LSTM": "#2f6f9f", "KAN": "#c46a3f"}
    fig, axes = plt.subplots(1, len(TICKERS), figsize=(4.2 * len(TICKERS), 3.6))
    for ax, (ticker, spec) in zip(axes, TICKERS.items()):
        for model in MODELS:
            hs = [h for h in HORIZONS if model in summary[ticker].get(h, {})]
            if not hs:
                continue
            m = [summary[ticker][h][model]["mean"] for h in hs]
            s = [summary[ticker][h][model]["std"] for h in hs]
            ax.errorbar(hs, m, yerr=s, marker="o", ms=4, capsize=3, label=model,
                        color=colours[model], linestyle="--" if model == "Naive" else "-")
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.set_xlabel("forecast horizon (days)")
        ax.set_ylabel("RMSE (ticker's Min-Max scale)")
        ax.set_title(spec["label"])
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multi_ticker_rmse.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    styles = {"CBA.AX": "o-", "TSLA": "s-", "LLY": "^-"}
    for ax, model in zip(axes, ("LSTM", "KAN")):
        for ticker in TICKERS:
            hs = [h for h in HORIZONS if model in summary[ticker].get(h, {})]
            if not hs:
                continue
            ax.plot(hs, [summary[ticker][h][model]["ratio_to_naive"] for h in hs],
                    styles[ticker], ms=5, label=ticker, color=colours[model],
                    alpha={"CBA.AX": 1.0, "TSLA": 0.65, "LLY": 0.4}[ticker])
        ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.set_xlabel("forecast horizon (days)")
        ax.set_title("%s RMSE / naive RMSE" % model)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("ratio (1 = equal to persistence)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multi_ticker_ratio.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = collect()
    with open(os.path.join(OUT, "multi_ticker_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md = markdown(summary)
    with open(os.path.join(OUT, "multi_ticker_table.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(OUT, "multi_ticker_table.tex"), "w") as f:
        f.write(latex(summary) + "\n")
    figures(summary)
    print(md)


if __name__ == "__main__":
    main()
