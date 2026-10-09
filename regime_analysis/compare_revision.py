"""
compare_revision.py
Side-by-side of the submitted runs (results/, scaler fitted on the whole
series) and the revision runs (results_rev1/, scaler fitted per fold on
training rows only), plus the two models added for referee 1's major
comment 2: an MLP on the same flat vector KAN receives, and a recurrent KAN.

    python compare_revision.py            # prints the tables
    python compare_revision.py --json out.json
"""
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [1, 2, 100, 200]; SEEDS = [0, 100, 200]; FOLDS = range(5)
REGIMES = ["Normal", "Volatile", "Trending"]

# (label, subdir, filename pattern with %d for horizon, %d seed, %d fold)
MODELS = [
    ("Naive",         "preds",           "naive__-__h%d__s0__f%d"),
    ("LSTM best cfg", "preds_lstmgrid",  "lstmcfg__L2-20-tanh__h%d__s%d__f%d"),
    ("KAN (pykan)",   "preds_pykan",     "kan__-__h%d__s%d__f%d"),
    ("MLP flat",      "preds_newmodels", "mlp__-__h%d__s%d__f%d"),
    ("Recurrent KAN", "preds_newmodels", "rkan__-__h%d__s%d__f%d"),
]

def load(root, sub, pat, h, s):
    P, T, L = [], [], []
    for f in FOLDS:
        name = pat % ((h, f) if pat.count("%d") == 2 else (h, s, f))
        p = os.path.join(root, sub, name + ".npz")
        if not os.path.exists(p): return None
        with np.load(p, allow_pickle=False) as d:
            P.append(d["prediction"].astype(float)); T.append(d["truth"].astype(float)); L.append(d["labels"])
    return np.concatenate(P), np.concatenate(T), np.concatenate(L)

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def summarise(root):
    out = {}
    for label, sub, pat in MODELS:
        for h in HORIZONS:
            per_seed = {r: [] for r in REGIMES + ["All"]}
            for s in SEEDS:
                r = load(root, sub, pat, h, s)
                if r is None: continue
                P, T, L = r
                for reg in REGIMES + ["All"]:
                    m = np.ones(len(L), bool) if reg == "All" else (L == reg)
                    per_seed[reg].append(rmse(P[m], T[m]))
                if pat.count("%d") == 2: break   # naive: deterministic
            if not per_seed["All"]: continue
            out[(label, h)] = {reg: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                               for reg, v in per_seed.items()}
    return out

def dm_test(root, sub_a, pat_a, sub_b, pat_b, h, s):
    """Diebold-Mariano, squared-error loss, HLN small-sample correction.
    Positive statistic => model A has the larger loss (B is better)."""
    A = load(root, sub_a, pat_a, h, s); B = load(root, sub_b, pat_b, h, s)
    if A is None or B is None: return None
    d = ((A[0] - A[1]) ** 2).mean(1) - ((B[0] - B[1]) ** 2).mean(1)   # per window
    n = len(d); k = h
    dbar = d.mean(); dc = d - dbar
    gamma = [np.dot(dc[:n - j], dc[j:]) / n for j in range(k)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0: return None
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * k + k * (k - 1) / n) / n)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(hln) / sqrt(2))))
    return float(hln), float(p), int(n)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    old = summarise(os.path.join(HERE, "results")); new = summarise(os.path.join(HERE, "results_rev1"))
    print("A. POOLED RMSE BY HORIZON  (mean +- sd over 3 seeds; reporting scale = whole-series Min-Max units)")
    print("%-15s %-6s %-18s %-18s %s" % ("model", "h", "submitted (leaky)", "revision (fixed)", "change"))
    for label, _, _ in MODELS:
        for h in HORIZONS:
            o = old.get((label, h)); n = new.get((label, h))
            if n is None: continue
            os_ = "%.4f +- %.4f" % o["All"] if o else "   (new model)   "
            ch = "%+.0f%%" % (100 * (n["All"][0] - o["All"][0]) / o["All"][0]) if o else ""
            print("%-15s %-6d %-18s %.4f +- %.4f    %s" % (label, h, os_, n["All"][0], n["All"][1], ch))
        print()
    print("B. REVISION, RMSE BY REGIME (mean over seeds)")
    print("%-15s %-6s %-8s %-8s %-8s %-8s" % ("model", "h", "Normal", "Volatile", "Trending", "All"))
    for label, _, _ in MODELS:
        for h in HORIZONS:
            n = new.get((label, h))
            if n: print("%-15s %-6d %-8.4f %-8.4f %-8.4f %-8.4f" % (label, h, *[n[r][0] for r in REGIMES + ["All"]]))
        print()
    print("C. DIEBOLD-MARIANO vs NAIVE (revision, seed 0; positive = naive better; p two-sided)")
    for label, sub, pat in MODELS[1:]:
        row = []
        for h in HORIZONS:
            r = dm_test(os.path.join(HERE, "results_rev1"), sub, pat, "preds", "naive__-__h%d__s0__f%d", h, 0)
            row.append("h%d: DM=%+.2f p=%.3f" % (h, r[0], r[1]) if r else "h%d: n/a" % h)
        print("%-15s %s" % (label, "   ".join(row)))
    print("\nD. DIEBOLD-MARIANO, model vs model at each horizon (revision, seed 0; positive = second model better)")
    pairs = [("KAN (pykan)", "LSTM best cfg"), ("MLP flat", "KAN (pykan)"), ("Recurrent KAN", "KAN (pykan)"), ("Recurrent KAN", "LSTM best cfg")]
    look = {m[0]: m for m in MODELS}
    for a_, b_ in pairs:
        row = []
        for h in HORIZONS:
            r = dm_test(os.path.join(HERE, "results_rev1"), look[a_][1], look[a_][2], look[b_][1], look[b_][2], h, 0)
            row.append("h%d: %+.2f (p=%.3f)" % (h, r[0], r[1]) if r else "h%d: n/a" % h)
        print("%-30s %s" % (a_ + " vs " + b_, "   ".join(row)))
    if a.json:
        json.dump({"submitted": {"%s|%d" % k: v for k, v in old.items()}, "revision": {"%s|%d" % k: v for k, v in new.items()}}, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
