"""Data generating process (DGP) of the numerical example.

A P = n x n grid of locations. For t = 1, ..., T:

    y_t = (I - rho W)^{-1} (X_t beta + eps_t),

where

* W is one of three spatial structures (``W_TYPES``), row-standardized and
  divided by its largest singular value,
* each row X_{t,i} (K covariates) follows a linear non-Gaussian acyclic
  model X = B_X X + e with a random DAG B_X and uniform e,
* eps_t ~ Uniform(-1, 1).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigvals, inv
from scipy.spatial.distance import cdist

# Type 1 / Type 2 / Type 3 in the paper
W_TYPES = ("point", "column", "queen")
W_TYPE_LABELS = {"point": "Type 1", "column": "Type 2", "queen": "Type 3"}
W_TYPE_TITLES = {
    "point": "Type 1: Spread from center",
    "column": "Type 2: Flow from left to right",
    "queen": "Type 3: Queen adjacency(8-directional adjacency)",
}


@dataclass
class SimulationConfig:
    """Parameters of the DGP (defaults are those used in the paper)."""

    grid_size: int = 7          # sqrt(P)
    T: int = 1000               # number of samples (after burn-in)
    K: int = 3                  # number of covariates
    W_type: str = "point"       # "point", "column" or "queen"
    rho: float = 0.7            # spatial autoregressive coefficient
    edge_prob: float = 0.2      # edge probability of the DAG of X
    weight_min: float = -1.0    # range of the edge weights of the DAG of X
    weight_max: float = 1.0
    x_noise_scale: float = 1.0  # e ~ Uniform(-x_noise_scale, x_noise_scale)
    noise_eps: float = 1.0      # eps ~ Uniform(-noise_eps, noise_eps)
    range_dist: float = 100     # radius of the edges from the peak (Type 1, 2)
    peak_idx: int = 0           # source column of Type 2
    burn_in: int = 50


# --------------------------------------------------------------------------
# Spatial weight matrices
# --------------------------------------------------------------------------
def _W_point(n, range_dist, weight):
    """Type 1: edges flow outward from the center of the grid (4-neighbors)."""
    coords = np.array([(i, j) for i in range(n) for j in range(n)])
    N = n * n
    peak = np.array((n // 2, n // 2)).reshape(1, 2)
    dist_to_peak = cdist(coords, peak).flatten()
    height = 1 / (1 + dist_to_peak)

    W = np.zeros((N, N))
    index = lambda r, c: r * n + c
    for r in range(n):
        for c in range(n):
            if dist_to_peak[index(r, c)] > range_dist:
                continue
            i = index(r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    j = index(rr, cc)
                    if height[i] > height[j]:  # from high to low
                        W[j, i] = weight
    return _stabilize(W)


def _W_column(n, range_dist, peak_idx, weight):
    """Type 2: edges flow left/right away from column ``peak_idx``."""
    N = n * n
    idx = lambda r, c: r * n + c
    height = np.tile(1 / (1 + np.abs(np.arange(n) - peak_idx)), (n, 1))
    within_range = lambda r, c: abs(c - peak_idx) <= range_dist

    W = np.zeros((N, N))
    for r in range(n):
        for c in range(n):
            if not within_range(r, c):
                continue
            i = idx(r, c)
            for dr, dc in [(0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and within_range(rr, cc):
                    j = idx(rr, cc)
                    if height[r, c] > height[rr, cc]:  # from high to low
                        W[j, i] = weight
    return _stabilize(W)


def _W_queen(n):
    """Type 3: queen adjacency (8 neighbors), row-standardized."""
    N = n * n
    W = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        W[i * n + j, ni * n + nj] = 1
    return row_standardize(W)


def _stabilize(W):
    """Scale W so that its spectral radius is below 1."""
    rho_spec = max(abs(eigvals(W)))
    if rho_spec >= 1:
        W *= 0.95 / rho_spec
    return W


def row_standardize(W):
    W = np.array(W, dtype=float)
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return W / row_sums[:, np.newaxis]


def spectral_norm_normalize(W, eps=1e-12):
    smax = np.linalg.svd(W, compute_uv=False)[0]
    return W / (smax + eps)


def make_W(cfg: SimulationConfig):
    """True spatial weight matrix W (row-standardized, largest singular value 1)."""
    n = cfg.grid_size
    # The edge weight cancels out in the row-standardization, so its value does not matter.
    if cfg.W_type == "point":
        W = _W_point(n, cfg.range_dist, cfg.rho)
    elif cfg.W_type == "column":
        W = _W_column(n, cfg.range_dist, cfg.peak_idx, cfg.rho)
    elif cfg.W_type == "queen":
        W = _W_queen(n)
    else:
        raise ValueError(f"Unknown W_type: {cfg.W_type}")
    return spectral_norm_normalize(row_standardize(W))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _build_x_dag(k, edge_prob, w_min, w_max, rng):
    """Random DAG B_X of the covariates (random causal order, Uniform weights)."""
    order = rng.permutation(k)
    Bx = np.zeros((k, k), dtype=float)
    for idx_child in range(1, k):
        child = order[idx_child]
        parents = order[:idx_child]
        mask = rng.random(size=idx_child) < edge_prob
        if np.any(mask):
            Bx[child, parents[mask]] = rng.uniform(w_min, w_max, size=int(mask.sum()))
    return Bx


def _sample_X_from_dag(Bx, T_tot, N, x_noise_scale, rng):
    """X_{t,i} = (I - B_X)^{-1} e_{t,i}, e ~ Uniform(-x_noise_scale, x_noise_scale)."""
    K = Bx.shape[0]
    Sx = inv(np.eye(K) - Bx)
    E = rng.uniform(-x_noise_scale, x_noise_scale, size=(T_tot, N, K))
    return E @ Sx.T


def simulate(cfg: SimulationConfig, W, beta, seed=0):
    """Generate Y and X from the SAR model.

    Parameters
    ----------
    cfg : SimulationConfig
    W : array-like, shape (P, P)
        True spatial weight matrix.
    beta : array-like, shape (K,)
        True regression coefficients.
    seed : int
        Seed of the random number generator.

    Returns
    -------
    Y : array-like, shape (T, P)
    X : array-like, shape (T, P, K)
    Bx : array-like, shape (K, K)
        DAG of the covariates.
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    beta = np.asarray(beta, dtype=float)
    S = inv(np.eye(N) - cfg.rho * W)
    T_tot = cfg.burn_in + cfg.T

    Bx = _build_x_dag(len(beta), cfg.edge_prob, cfg.weight_min, cfg.weight_max, rng)
    X = _sample_X_from_dag(Bx, T_tot, N, cfg.x_noise_scale, rng)
    eps = rng.uniform(-cfg.noise_eps, cfg.noise_eps, size=(T_tot, N))

    Y = np.zeros((T_tot, N), dtype=float)
    for t in range(T_tot):
        Y[t] = S @ (X[t] @ beta + eps[t])
    return Y[cfg.burn_in:], X[cfg.burn_in:], Bx


def make_beta(seed, scale=10.0, size=200):
    """True beta (the first K elements are used). Shared by all conditions of a seed.

    ``seed=None`` reproduces the single run in the paper (``RandomState(42)``).
    """
    if seed is None:
        return np.random.RandomState(42).normal(loc=0, scale=scale, size=size)
    return np.random.default_rng(seed).normal(loc=0, scale=scale, size=size)
