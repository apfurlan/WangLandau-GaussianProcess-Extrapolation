"""
gp_model.py
===========
Gaussian Process model for s(e, 1/L) = ln g(E,L) / L^2.

Why a GP and not a neural net
------------------------------
Even with dL=2 you have, at most, a few hundred distinct L *curves* --
not independent samples in the statistical-learning sense (a deep net
wants thousands to millions of i.i.d. examples to avoid overfitting).
A GP, in contrast:

  - is well-behaved with scarce/structured training data,
  - gives a calibrated uncertainty for every prediction (essential here:
    you want to know exactly where the L=1024 guess is trustworthy and
    where it isn't, typically near the spectrum edges),
  - lets you bake the physics directly into the kernel.

Kernel design
-------------
The kernel is anisotropic over X = [e, 1/L], with *separate* length
scales for each direction:
  - Matern(nu=1.5) along e:  s(e) is smooth but not perfectly so
    (curvature changes near e=0, where the DOS peaks); Matern is a
    safer default than RBF for that.
  - the same kernel handles 1/L: finite-size corrections are expected
    to be smooth (close to analytic) in 1/L, so the optimizer typically
    finds a much longer length scale in that direction -- the data
    itself tells you how "exponential-in-L"-like the convergence is,
    rather than you having to guess a parametric form up front.
  - WhiteKernel absorbs the residual Monte-Carlo noise from WL.
"""

from __future__ import annotations
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C, WhiteKernel, Matern
)
from scipy.interpolate import CubicSpline


def build_kernel(length_scale_e: float = 0.2, length_scale_invL: float = 0.05,
                  noise_level: float = 1e-6):
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=[length_scale_e, length_scale_invL],
                 length_scale_bounds=(1e-3, 10.0), nu=1.5)
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-1))
    )
    return kernel


class DOSExtrapolator:
    """Thin wrapper around sklearn's GaussianProcessRegressor."""

    def __init__(self, kernel=None, n_restarts_optimizer: int = 3,
                 normalize_y: bool = True, random_state: int = 0):
        self.kernel = kernel if kernel is not None else build_kernel()
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DOSExtrapolator":
        self.gp.fit(X, y)
        self._fitted = True
        return self

    def predict(self, e: np.ndarray, L_target: int):
        """Predict s(e, L_target) and its 1-sigma uncertainty."""
        if not self._fitted:
            raise RuntimeError("call .fit() before .predict()")
        Xq = np.column_stack([e, np.full_like(e, 1.0 / L_target)])
        s_mean, s_std = self.gp.predict(Xq, return_std=True)
        return s_mean, s_std

    @property
    def fitted_kernel_(self):
        return self.gp.kernel_


def predict_dense(model: DOSExtrapolator, e_dense: np.ndarray, L_target: int,
                   n_coarse: int = 400):
    """
    Predict s(e, L_target) at every point of a (potentially huge,
    ~L_target^2) dense energy grid, WITHOUT ever forming a
    (n_train x n_dense) kernel matrix -- for L_target=1024 that grid
    has ~10^6 points, which would mean tens of GB of memory if queried
    directly against the GP.

    Instead: GP-predict on a coarse grid of n_coarse points spanning
    the same e-range (cheap), then cubic-spline-interpolate to the full
    dense grid. This is safe because s(e, L) is, by construction, a
    smooth entropy density -- the GP already resolved the physics at
    coarse resolution; the spline is only filling in a smooth
    interpolation. This mirrors -- and replaces with something
    better-grounded -- the second manual fitting step in the original
    workflow ("ajusto outra função, agora DOS em função de E, para
    estimar os valores faltantes").
    """
    e_min, e_max = e_dense.min(), e_dense.max()
    e_coarse = np.linspace(e_min, e_max, n_coarse)
    s_coarse, std_coarse = model.predict(e_coarse, L_target)

    s_spline = CubicSpline(e_coarse, s_coarse)
    std_spline = CubicSpline(e_coarse, std_coarse)

    s_dense = s_spline(e_dense)
    std_dense = np.clip(std_spline(e_dense), 0.0, None)
    return s_dense, std_dense


def leave_one_L_out(data: dict, L_holdout: int, kernel=None,
                     max_points_per_L: int = 25, n_eval_points: int = 200,
                     max_L_for_training: int = 30):
    """
    Train on every L except L_holdout, predict s at L_holdout's own
    energies, and compare against the real (already-simulated) curve.

    Run this BEFORE trusting an extrapolation to a brand-new L you
    don't have: e.g. hold out L=256, train on L=4..254, and see how
    well the model -- which never saw L=256 -- reconstructs it.
    """
    # local imports to avoid a circular import at module load time
    from data_utils import (
        to_reduced, build_training_table, subsample_by_e, select_training_L
    )

    candidate_L = [L for L in data if L != L_holdout]
    train_L = select_training_L(candidate_L, max_L=max_L_for_training)
    train_data = {L: data[L] for L in train_L}
    X, y, _, _ = build_training_table(train_data, max_points_per_L)
    model = DOSExtrapolator(kernel=kernel).fit(X, y)

    E_true_full, lng_true_full = data[L_holdout]
    E_true, lng_true = subsample_by_e(E_true_full, lng_true_full,
                                       L_holdout, n_eval_points)
    e_true, s_true = to_reduced(E_true, lng_true, L_holdout)
    s_pred, s_std = model.predict(e_true, L_holdout)

    rmse = float(np.sqrt(np.mean((s_pred - s_true) ** 2)))
    max_err = float(np.max(np.abs(s_pred - s_true)))
    return {
        "L": L_holdout, "e": e_true, "s_true": s_true,
        "s_pred": s_pred, "s_std": s_std,
        "rmse": rmse, "max_abs_err": max_err, "model": model,
    }