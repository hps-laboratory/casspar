"""One run of the numerical example: generate data, fit LR / CASSPAR / LiNGAM, evaluate.

Procedure (identical to the paper):

1. Generate W, beta, Y and X (``simulation.py``).
2. Standardize X and Y (pooled over samples and locations).
3. Split the samples into train (80%) and test (20%).
4. Fit the three methods to the training data.
5. Evaluate beta (on the original scale), W and the prediction of Y (test data).
"""

import numpy as np
from lingam import DirectLiNGAM
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from casspar import CASSPAR

try:
    from . import simulation as sim
except ImportError:
    import simulation as sim

METHODS = ("LR", "CASSPAR", "LiNGAM")


def run_one(W_type="point", P=49, T=1000, seed=0, K=3, rho=0.7, beta_scale=10.0,
            n_lags=10, return_arrays=False):
    """Run one condition with one seed.

    Parameters
    ----------
    W_type : {"point", "column", "queen"}
        Spatial structure (Type 1, 2, 3).
    P : int
        Number of locations (a square number).
    T : int
        Number of samples.
    seed : int or None
        Seed of the data generation and of the estimation. ``None`` reproduces
        the single run of the paper (data: seed 0, estimation: seed 42,
        beta: ``RandomState(42)``).
    K, rho, beta_scale : DGP parameters.
    n_lags : int
        ``n_lags`` of CASSPAR.
    return_arrays : bool
        If True, also return the true / estimated W and beta.

    Returns
    -------
    result : dict
        MAE of beta, W and Y (test) for each method, and the estimated rho.
    """
    grid_size = int(round(np.sqrt(P)))
    if grid_size ** 2 != P:
        raise ValueError("P must be a square number.")
    sim_seed = 0 if seed is None else seed
    est_seed = 42 if seed is None else seed

    # 1. data
    cfg = sim.SimulationConfig(grid_size=grid_size, T=T, K=K, W_type=W_type, rho=rho)
    beta = sim.make_beta(seed, scale=beta_scale)[:K]
    W_true = sim.make_W(cfg)
    Y, X, _ = sim.simulate(cfg, W_true, beta, seed=sim_seed)

    # 2. standardize
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X.reshape(-1, K)).reshape(X.shape)
    scaler_Y = StandardScaler()
    Y = scaler_Y.fit_transform(Y.reshape(-1, 1)).reshape(Y.shape)

    # 3. split
    train_idx, test_idx = train_test_split(np.arange(T), test_size=0.2, random_state=est_seed)
    X_train, X_test, Y_train, Y_test = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]
    X_train_flat, X_test_flat = X_train.reshape(-1, K), X_test.reshape(-1, K)

    def to_original_scale(coef):
        return np.asarray(coef, dtype=float) * (scaler_Y.scale_[0] / np.maximum(scaler_X.scale_, 1e-12))

    # 4. fit
    lr = LinearRegression().fit(X_train_flat, Y_train.ravel())

    casspar = CASSPAR(n_lags=n_lags, random_state=est_seed).fit(Y_train, X_train)

    lingam = DirectLiNGAM(measure="pwling").fit(np.column_stack([X_train_flat, Y_train.ravel()]))
    B = lingam.adjacency_matrix_  # the last variable is Y

    beta_hat = {
        "LR": to_original_scale(lr.coef_),
        "CASSPAR": to_original_scale(casspar.coef_),
        "LiNGAM": to_original_scale(B[K, :K]),
    }
    Y_pred = {
        "LR": lr.predict(X_test_flat).reshape(Y_test.shape),
        "CASSPAR": casspar.predict(X_test),
        "LiNGAM": (X_test_flat @ B[K, :K]).reshape(Y_test.shape),
    }

    # 5. evaluate
    off_diag = ~np.eye(P, dtype=bool)
    W_hat = casspar.spatial_weight_matrix_
    result = {"W_type": W_type, "P": P, "T": T, "seed": seed,
              "rho_hat_CASSPAR": casspar.rho_,
              "W_MAE_CASSPAR": float(np.mean(np.abs(W_true[off_diag] - W_hat[off_diag])))}
    for m in METHODS:
        result[f"beta_MAE_{m}"] = float(mean_absolute_error(beta, beta_hat[m]))
        result[f"y_MAE_{m}"] = float(mean_absolute_error(Y_test.ravel(), Y_pred[m].ravel()))

    if return_arrays:
        result.update(W_true=W_true, W_hat=W_hat, beta_true=beta,
                      **{f"beta_{m}": beta_hat[m] for m in METHODS})
    return result
