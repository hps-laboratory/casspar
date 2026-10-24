# CASSPAR

This repository contains the implementation and the numerical example of the paper:

> "CASSPAR: Causal Structure Search for the Spatial Autoregressive Model and Its Application to the Injection Molding Process"

**CASSPAR** (Causal Structure Search for the Spatial Autoregressive Model) estimates the causal relationships among locations using the spatial autoregressive (SAR) model:

$$
y^{[n]} = \rho W y^{[n]} + X^{[n]} \beta + e^{[n]}, \qquad n = 1, \dots, N.
$$

The spatial weight matrix $W$ is regarded as the adjacency matrix of a causal graph, and it is estimated directly from data. The estimation consists of two steps:

1. **Step 1**: [DirectLiNGAM](https://github.com/cdt15/lingam) is applied to $Y$ (locations as variables) to estimate $\widehat{W}$.
2. **Step 2**: $\rho$ and $\beta$ are estimated by two-stage least squares (2SLS) with the instruments $[X, \widehat{W}X, \dots, \widehat{W}^L X]$.

CASSPAR requires four assumptions:

1. $W$ represents a directed acyclic graph (DAG).
2. The components of $u^{[n]} = X^{[n]}\beta + e^{[n]}$ are non-Gaussian and mutually independent across locations.
3. The instruments are exogenous (uncorrelated with the structural errors).
4. The instruments are relevant (correlated with the spatial lag $Wy^{[n]}$).

## Installation

With [uv](https://docs.astral.sh/uv/) (the versions in `uv.lock` are those used for the paper):

```bash
git clone https://github.com/hps-laboratory/casspar.git
cd casspar
uv sync --extra experiments
```

or with pip:

```bash
pip install -e ".[experiments]"
```

The library itself (`casspar/`) depends only on `numpy`, `scipy`, `scikit-learn` and `lingam`. The `experiments` extra adds the packages for the numerical example and the notebooks.

## Usage

See [examples/CASSPAR.ipynb](examples/CASSPAR.ipynb) for a tutorial.

```python
from casspar import CASSPAR

model = CASSPAR(random_state=0)
model.fit(Y, X)                     # Y: (n_samples, n_locations), X: (n_samples, n_locations, n_covariates)

model.spatial_weight_matrix_        # estimated W
model.rho_, model.coef_             # estimated rho and beta
model.predict(X_new)                # (I - rho W)^{-1} X beta
```

Center (or standardize) Y and X before fitting.

| Major parameter | Default | Description |
|---|---|---|
| `n_lags` | 10 | Maximum power $L$ of $\widehat{W}$ in the instruments |
| `n_bootstrap` | None | If > 1, $\widehat{W}$ is the median of DirectLiNGAM estimates on bootstrap samples |
| `random_state` | | Passed to DirectLiNGAM |

| Attribute | Description |
|---|---|
| `spatial_weight_matrix_` | $\widehat{W}$ after normalization (used in Step 2) |
| `rho_`, `coef_`, `intercept_` | Estimated $\rho$, $\beta$ and intercept |

## Numerical example

[experiments/numerical_example.ipynb](experiments/numerical_example.ipynb) reproduces the figures of the numerical example in the paper. The figures can also be regenerated with the following commands:

```bash
cd experiments
python make_figures.py                      # PDF files in experiments/figures/
```

`make_figures.py` does not run the Monte Carlo simulation. It reads the results of the paper (200 seeds, aggregated) bundled in `experiments/data/`, because the full simulation takes many hours. To re-run the simulation:

```bash
python monte_carlo.py --quick --n-jobs -1   # small grid, about a minute
python monte_carlo.py --n-jobs -1           # full grid of the paper (16,800 runs, many hours)
python make_figures.py --data results/mc    # monte_carlo.py also writes the aggregated CSV files to results/mc
```

| File | Content |
|---|---|
| `simulation.py` | Data generating process (three spatial structures, covariates from a random DAG) |
| `run_one.py` | One run: data generation, LR / CASSPAR / LiNGAM, evaluation |
| `monte_carlo.py` | All conditions and seeds (resumable) |
| `aggregate.py` | Median, IQR, mean and standard deviation over seeds (run at the end of `monte_carlo.py`) |
| `plotting.py`, `make_figures.py` | Figures |

## Repository structure

```
casspar/        the library (a single module casspar/casspar.py)
examples/       tutorial notebook
experiments/    numerical example of the paper
```

## Relation to lingam

CASSPAR builds on the [lingam](https://github.com/cdt15/lingam) package and follows its conventions: fitted attributes are properties with a trailing underscore, and `random_state` is used as in DirectLiNGAM.

## License

[MIT](LICENSE)
