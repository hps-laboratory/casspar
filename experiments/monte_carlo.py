"""Monte Carlo simulation of the numerical example.

Runs ``run_one`` for every (W_type, P, T, seed) and appends one JSON line per
run to ``<out>/results.jsonl``. Finished runs are skipped when re-running, so
the simulation can be interrupted and resumed. At the end, the results are
aggregated over seeds into ``mae_beta.csv``, ``mae_W.csv`` and ``mae_y.csv``
in ``<out>`` (see ``aggregate.py``).

Usage::

    python experiments/monte_carlo.py --quick              # a few minutes
    python experiments/monte_carlo.py --n-jobs -1          # full grid of the paper (very long)
"""

import argparse
import json
import time
import traceback
from itertools import product
from pathlib import Path

from joblib import Parallel, delayed

try:
    from .run_one import run_one
    from .simulation import W_TYPES
except ImportError:
    from run_one import run_one
    from simulation import W_TYPES

# Grid of the paper
PAPER_GRID = dict(
    W_type=list(W_TYPES),
    P=[9, 25, 49, 81],
    T=[200, 400, 600, 800, 1000, 1500, 2000],
    seed=list(range(200)),
)
# Small grid to check that everything works
QUICK_GRID = dict(W_type=list(W_TYPES), P=[9, 25], T=[200, 400], seed=list(range(3)))

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "mc"


def _run_id(W_type, P, T, seed):
    return f"{W_type}_P{P}_T{T}_seed{seed}"


def _job(W_type, P, T, seed):
    t0 = time.time()
    row = {"run_id": _run_id(W_type, P, T, seed), "W_type": W_type, "P": P, "T": T, "seed": seed}
    try:
        row.update(run_one(W_type=W_type, P=P, T=T, seed=seed))
        row["status"] = "ok"
    except Exception as exc:  # keep the other runs going
        row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc())
    row["elapsed_s"] = time.time() - t0
    return row


def load_results(out_dir, only_ok=True):
    """Read ``results.jsonl`` as a DataFrame."""
    import pandas as pd

    path = Path(out_dir) / "results.jsonl"
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows).drop_duplicates("run_id", keep="last")
    return df[df["status"] == "ok"].reset_index(drop=True) if only_ok else df


def run_monte_carlo(grid, out_dir=DEFAULT_OUT, n_jobs=1):
    """Run all combinations of ``grid`` that are not yet in ``<out_dir>/results.jsonl``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.jsonl"

    done = set()
    if path.exists():
        with open(path) as f:
            done = {json.loads(line)["run_id"] for line in f if line.strip()}

    jobs = [c for c in product(grid["W_type"], grid["P"], grid["T"], grid["seed"])
            if _run_id(*c) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already done) -> {path}")

    with open(path, "a") as f:
        for i, row in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_job)(*c) for c in jobs), start=1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row["status"] != "ok":
                print(f"  [error] {row['run_id']}: {row['error']}")
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="run a small grid (P=9,25, T=200,400, 3 seeds)")
    parser.add_argument("--n-seeds", type=int, default=None, help="number of seeds (default: 200)")
    parser.add_argument("--n-jobs", type=int, default=1, help="number of parallel jobs (-1: all cores)")
    parser.add_argument("--out", type=Path, default=None, help=f"output directory (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    grid = dict(QUICK_GRID if args.quick else PAPER_GRID)
    if args.n_seeds is not None:
        grid["seed"] = list(range(args.n_seeds))
    out = args.out or (DEFAULT_OUT.parent / "mc_quick" if args.quick else DEFAULT_OUT)
    run_monte_carlo(grid, out_dir=out, n_jobs=args.n_jobs)

    try:  # imported here because aggregate.py imports load_results from this module
        from .aggregate import aggregate_results
    except ImportError:
        from aggregate import aggregate_results
    aggregate_results(out)


if __name__ == "__main__":
    main()
