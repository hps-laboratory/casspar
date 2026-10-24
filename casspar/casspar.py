"""
Python implementation of CASSPAR (Causal Structure Search for the
Spatial Autoregressive Model).
"""

import numbers

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_array, resample

from lingam import DirectLiNGAM


class CASSPAR:
    """Implementation of CASSPAR Algorithm

    CASSPAR estimates the spatial autoregressive (SAR) model

    .. math::

        y_t = \\rho W y_t + X_t \\beta + \\alpha + \\varepsilon_t,
        \\quad t = 1, \\dots, T,

    where the spatial weight matrix :math:`W` is unknown. It is estimated
    from the data in two steps:

    1. **Step 1**: DirectLiNGAM is applied to :math:`Y` (locations as
       variables) to estimate :math:`\\tilde{W}`, which is then optionally
       row-standardized and divided by its largest singular value.
    2. **Step 2**: :math:`\\rho`, :math:`\\beta` and :math:`\\alpha` are
       estimated by two-stage least squares (2SLS) with the instruments
       :math:`[X, \\tilde{W}X, \\dots, \\tilde{W}^L X]`.

    Notes
    -----
    Center (or standardize) Y and X before fitting. The instruments
    are centered in the first stage of 2SLS, so with non-centered data the
    intercept absorbs the mean of the spatial lag and ``intercept_`` is biased.
    """

    def __init__(
        self,
        n_lags=10,
        row_standardize=True,
        spectral_norm_normalize=True,
        n_bootstrap=None,
        n_jobs=None,
        random_state=None,
    ):
        """Construct a CASSPAR model.

        Parameters
        ----------
        n_lags : int, optional (default=10)
            Maximum power ``L`` of the spatial lags of X used as instruments
            in Step 2, i.e. ``[X, WX, ..., W^L X]``.
        row_standardize : bool, optional (default=True)
            Whether to row-standardize the W estimated in Step 1.
        spectral_norm_normalize : bool, optional (default=True)
            Whether to divide the W estimated in Step 1 by its largest singular value.
        n_bootstrap : int, optional (default=None)
            Number of bootstrap samples used to estimate W in Step 1.
            If ``None`` or ``<= 1``, DirectLiNGAM is fitted once to all samples.
            Otherwise, W is the element-wise median of the bootstrap estimates.
        n_jobs : int, optional (default=None)
            Number of jobs for the bootstrap estimation of W.
            ``None`` means 1 and ``-1`` means using all processors.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        if not isinstance(n_lags, (numbers.Integral, np.integer)) or n_lags < 0:
            raise ValueError("n_lags must be a non-negative integer.")
        if n_bootstrap is not None and not isinstance(n_bootstrap, (numbers.Integral, np.integer)):
            raise ValueError("n_bootstrap must be None or an integer.")

        self._n_lags = n_lags
        self._row_standardize = row_standardize
        self._spectral_norm_normalize = spectral_norm_normalize
        self._n_bootstrap = n_bootstrap
        self._n_jobs = n_jobs
        self._random_state = random_state

        self._causal_order = None
        self._adjacency_matrix = None
        self._spatial_weight_matrix = None
        self._bootstrap_adjacency_matrices = None
        self._rho = None
        self._coef = None
        self._intercept = None

    def fit(self, Y, X):
        """Fit the model to Y and X.

        Parameters
        ----------
        Y : array-like, shape (n_samples, n_locations)
            Target variable, where ``n_samples`` is the number of samples
            (e.g. time points) and ``n_locations`` is the number of locations.
        X : array-like, shape (n_samples, n_locations, n_covariates)
            Covariates, where ``n_covariates`` is the number of covariates.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        Y, X = self._check_data(Y, X)

        # Step 1: estimate W by DirectLiNGAM
        if self._n_bootstrap is None or self._n_bootstrap <= 1:
            model = DirectLiNGAM(random_state=self._random_state, measure="pwling")
            model.fit(Y)
            causal_order = model.causal_order_
            adjacency_matrix = model.adjacency_matrix_
            bootstrap_adjacency_matrices = None
        else:
            W_samples = self._bootstrap_W(Y)
            causal_order = None
            adjacency_matrix = np.median(W_samples, axis=0)
            bootstrap_adjacency_matrices = W_samples

        W = adjacency_matrix
        if self._row_standardize:
            W = _row_standardize(W)
        if self._spectral_norm_normalize:
            W = _spectral_norm_normalize(W)

        # Step 2: estimate rho, beta and alpha by 2SLS
        coef, rho, intercept = _two_stage_least_squares(
            Y, X, W, n_lags=self._n_lags
        )

        # Commit the fitted state only after all estimation steps succeed.
        self._causal_order = causal_order
        self._adjacency_matrix = adjacency_matrix
        self._bootstrap_adjacency_matrices = bootstrap_adjacency_matrices
        self._spatial_weight_matrix = W
        self._coef, self._rho, self._intercept = coef, rho, intercept
        return self

    def predict(self, X):
        """Predict Y from X with the reduced form
        :math:`y_t = (I - \\rho W)^{-1} (X_t \\beta + \\alpha)`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_locations, n_covariates)
            Covariates.

        Returns
        -------
        Y_pred : array-like, shape (n_samples, n_locations)
            Predicted values of Y.
        """
        if self._coef is None:
            raise ValueError("This CASSPAR instance is not fitted yet. Call 'fit' first.")
        X = check_array(X, allow_nd=True)
        if X.ndim != 3 or X.shape[1:] != (self._spatial_weight_matrix.shape[0], self._coef.shape[0]):
            raise ValueError("X must have shape (n_samples, n_locations, n_covariates) used in fit.")

        n_locations = self._spatial_weight_matrix.shape[0]
        A = np.eye(n_locations) - self._rho * self._spatial_weight_matrix
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        return (X @ self._coef + self._intercept) @ A_inv.T

    def _check_data(self, Y, X):
        Y = check_array(Y)
        X = check_array(X, allow_nd=True)
        if X.ndim != 3:
            raise ValueError("X must be a 3D array of shape (n_samples, n_locations, n_covariates).")
        if X.shape[:2] != Y.shape:
            raise ValueError("The first two dimensions of X must match the shape of Y.")
        return Y, X

    def _bootstrap_W(self, Y):
        """Fit DirectLiNGAM to bootstrap resamples of Y and return the estimates of W."""
        seeds = np.random.RandomState(self._random_state).randint(
            0, np.iinfo(np.int32).max, size=self._n_bootstrap
        )
        index = np.arange(Y.shape[0])

        def _fit_one(seed):
            Y_resampled, _ = resample(Y, index, random_state=seed)
            model = DirectLiNGAM(random_state=seed, measure="pwling")
            model.fit(Y_resampled)
            return model.adjacency_matrix_

        W_samples = Parallel(n_jobs=self._n_jobs)(delayed(_fit_one)(seed) for seed in seeds)
        return np.asarray(W_samples)

    @property
    def causal_order_(self):
        """Causal order of the locations estimated in Step 1.

        Returns
        -------
        causal_order_ : array-like, shape (n_locations)
            The causal order of the locations. ``None`` when W is estimated by bootstrapping.
        """
        return self._causal_order

    @property
    def adjacency_matrix_(self):
        """W estimated by DirectLiNGAM in Step 1 (before normalization).

        Returns
        -------
        adjacency_matrix_ : array-like, shape (n_locations, n_locations)
            The element ``[i, j]`` is the effect of location ``j`` on location ``i``.
            When ``n_bootstrap > 1``, the element-wise median of the bootstrap estimates.
        """
        return self._adjacency_matrix

    @property
    def spatial_weight_matrix_(self):
        """Spatial weight matrix used in Step 2 (after normalization).

        Returns
        -------
        spatial_weight_matrix_ : array-like, shape (n_locations, n_locations)
            ``adjacency_matrix_`` after row-standardization and normalization
            by its largest singular value (if enabled).
        """
        return self._spatial_weight_matrix

    @property
    def bootstrap_adjacency_matrices_(self):
        """Bootstrap estimates of W in Step 1.

        Returns
        -------
        bootstrap_adjacency_matrices_ : array-like, shape (n_bootstrap, n_locations, n_locations)
            ``None`` when ``n_bootstrap`` is ``None`` or ``<= 1``.
        """
        return self._bootstrap_adjacency_matrices

    @property
    def rho_(self):
        """Estimated spatial autoregressive coefficient.

        Returns
        -------
        rho_ : float
        """
        return self._rho

    @property
    def coef_(self):
        """Estimated regression coefficients of the covariates.

        Returns
        -------
        coef_ : array-like, shape (n_covariates)
        """
        return self._coef

    @property
    def intercept_(self):
        """Estimated intercept.

        Returns
        -------
        intercept_ : float
        """
        return self._intercept


def _row_standardize(W):
    """Divide each row of W by its sum (rows summing to zero are left unchanged)."""
    W = np.array(W, dtype=float)
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return W / row_sums[:, np.newaxis]


def _spectral_norm_normalize(W, eps=1e-12):
    """Divide W by its largest singular value."""
    smax = np.linalg.svd(W, compute_uv=False)[0]
    return W / (smax + eps)


def _two_stage_least_squares(Y, X, W, n_lags=10, standardize=True):
    """Estimate beta, rho and alpha of the SAR model by 2SLS.

    The spatial lag ``S = W y`` is projected onto the instruments
    ``[X, WX, ..., W^L X]`` (first stage). beta is obtained by regressing
    y on X after partialling out the projected lag (Frisch-Waugh-Lovell),
    and rho is the coefficient of the projected lag in the regression of
    y on ``[S_hat, X]`` (second stage).

    Parameters
    ----------
    Y : array-like, shape (n_samples, n_locations)
    X : array-like, shape (n_samples, n_locations, n_covariates)
    W : array-like, shape (n_locations, n_locations)
    n_lags : int
        Maximum power of W in the instruments.
    standardize : bool
        Whether to standardize the columns of the instruments in the first stage.

    Returns
    -------
    coef : array-like, shape (n_covariates)
    rho : float
    intercept : float
    """
    T, N, K = X.shape
    S = (W @ Y.T).T.reshape(-1, 1)
    y = Y.reshape(-1, 1)
    X_stack = X.reshape(T * N, K)
    X_stack = np.column_stack([np.ones((T * N, 1)), X_stack])

    # instruments Z = [X, WX, W^2 X, ...]
    Z_list = [X_stack]
    WX = X.copy()
    for _ in range(n_lags):
        WX = np.einsum("ij,tjk->tik", W, WX)
        Z_list.append(WX.reshape(T * N, -1))
    Z = np.column_stack(Z_list)

    if standardize:
        Zn = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-12)
        Xn = (X_stack - X_stack.mean(0, keepdims=True)) / (X_stack.std(0, keepdims=True) + 1e-12)
    else:
        Zn, Xn = Z, X_stack

    # first stage: project S onto the column space of [Z, X]
    Q, _ = np.linalg.qr(np.column_stack([Zn, Xn]), mode="reduced")
    S_hat = Q @ (Q.T @ S)

    # partial out S_hat from y and X
    ss = float((S_hat.T @ S_hat).item())
    g_y = float((S_hat.T @ y).item() / ss) if ss > 0 else 0.0
    y_perp = y - S_hat * g_y
    X_perp = []
    for j in range(X_stack.shape[1]):
        xj = X_stack[:, [j]]
        g_x = float((S_hat.T @ xj).item() / ss) if ss > 0 else 0.0
        X_perp.append(xj - S_hat * g_x)
    X_perp = np.hstack(X_perp)

    # second stage: beta (and alpha) from the partialled regression
    Q_perp, R_perp = np.linalg.qr(X_perp, mode="reduced")
    coef = np.linalg.solve(R_perp, Q_perp.T @ y_perp).ravel()

    # second stage: rho from the regression of y on [S_hat, X]
    Q_sx, R_sx = np.linalg.qr(np.column_stack([S_hat, X_stack]), mode="reduced")
    theta = np.linalg.solve(R_sx, Q_sx.T @ y)
    rho = float(theta[0].item())

    intercept = float(coef[0])
    return coef[-K:], rho, intercept
