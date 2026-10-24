"""Make the figures of the numerical example from the CSV files in ``experiments/data``.

Usage::

    python experiments/make_figures.py                     # figures from the bundled CSV files
    python experiments/make_figures.py --data experiments/results/mc_quick   # after monte_carlo.py

The PDF files are written to ``experiments/figures``. Fig. 4 and 5 are computed
by three single runs; Fig. 6 is drawn from the CSV files of the Monte Carlo results.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from . import plotting  # noqa: E402
    from .run_one import run_one  # noqa: E402
    from .simulation import W_TYPES  # noqa: E402
except ImportError:
    import plotting  # noqa: E402
    from run_one import run_one  # noqa: E402
    from simulation import W_TYPES  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# CSV -> (output name, keyword arguments of plotting.plot_mae)
MAE_FIGURES = {
    "mae_beta.csv": ("beta_MAE", dict(methods=["LR", "CASSPAR", "LiNGAM"], sharey="col", ymax={"Type 1": 0.2})),
    "mae_W.csv": ("W_MAE", dict(methods=["CASSPAR"], sharey=True)),
    "mae_y.csv": ("y_MAE", dict(methods=["LR", "CASSPAR", "LiNGAM"], sharey="col")),
}


def single_runs(T=1000, P=49):
    """Single runs of the paper (seed=None) -> true / estimated W for each type."""
    runs = {w: run_one(W_type=w, P=P, T=T, seed=None, return_arrays=True) for w in W_TYPES}
    return {w: r["W_true"] for w, r in runs.items()}, {w: r["W_hat"] for w, r in runs.items()}


def save(fig, name, out_dir):
    path = out_dir / f"{name}.pdf"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="directory of the CSV files")
    parser.add_argument("--out", type=Path, default=FIG_DIR, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plotting.use_paper_style()

    # Fig. 4 and 5: true and estimated W (three single runs, about 30 seconds)
    W_true, W_hat = single_runs()
    save(plotting.plot_W_types(W_true, kind="arrows", titles=True), "W_true_arrows", args.out)
    save(plotting.plot_W_types(W_true), "W_true_heatmap", args.out)
    save(plotting.plot_W_types(W_hat), "W_estimated_heatmap", args.out)

    # Fig. 6: MAE of beta, W and Y
    for fname, (name, kwargs) in MAE_FIGURES.items():
        if (args.data / fname).exists():
            save(plotting.plot_mae(pd.read_csv(args.data / fname), **kwargs), name, args.out)


if __name__ == "__main__":
    main()
