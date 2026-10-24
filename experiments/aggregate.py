"""Aggregate the Monte Carlo results over seeds into the CSV files used for the figures.

``monte_carlo.py`` runs this at the end, so it is only needed to re-aggregate
existing results (e.g. after changing the summary statistics).

Usage::

    python experiments/aggregate.py experiments/results/mc --out experiments/data

Writes ``mae_beta.csv``, ``mae_W.csv`` and ``mae_y.csv`` with the columns
``P, type, T, method, n_seed, mean, std, median, q25, q75``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .monte_carlo import load_results
    from .run_one import METHODS
    from .simulation import W_TYPE_LABELS
except ImportError:
    from monte_carlo import load_results
    from run_one import METHODS
    from simulation import W_TYPE_LABELS

# CSV file -> (metric prefix, methods)
METRICS = {
    "mae_beta.csv": ("beta_MAE", METHODS),
    "mae_W.csv": ("W_MAE", ("CASSPAR",)),
    "mae_y.csv": ("y_MAE", METHODS),
}


def summarize(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return dict(
        n_seed=len(v),
        mean=float(np.mean(v)),
        std=float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
        median=float(np.median(v)),
        q25=float(np.percentile(v, 25)),
        q75=float(np.percentile(v, 75)),
    )


def aggregate(df, prefix, methods):
    """Collapse the seeds of a results DataFrame into summary statistics (long format)."""
    records = []
    for (P, W_type, T), g in df.groupby(["P", "W_type", "T"]):
        for m in methods:
            col = f"{prefix}_{m}"
            if col in g and g[col].notna().any():
                records.append(dict(P=int(P), type=W_TYPE_LABELS[W_type], T=int(T), method=m,
                                    **summarize(g[col])))
    return pd.DataFrame(records).sort_values(["P", "type", "method", "T"]).reset_index(drop=True)


def aggregate_results(results_dir, out_dir=None):
    """Read ``<results_dir>/results.jsonl`` and write the summary CSV files to ``out_dir``."""
    out = Path(out_dir or results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)
    for fname, (prefix, methods) in METRICS.items():
        aggregate(df, prefix, methods).to_csv(out / fname, index=False)
        print(f"saved -> {out / fname}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="directory containing results.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: results_dir)")
    args = parser.parse_args()
    aggregate_results(args.results_dir, args.out)


if __name__ == "__main__":
    main()
