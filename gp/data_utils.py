"""
data_utils.py
=============
Utilities for loading Wang-Landau density-of-states (DOS) results for
several lattice sizes L, converting them to the reduced "data-collapse"
variables used by the GP extrapolation model, and generating synthetic
test data so the pipeline can be exercised end-to-end before real WL
output files are plugged in.

Reduced variables
------------------
    e = E / L**2          (energy per site)
    s = ln g(E,L) / L**2  (entropy per site)

These are the natural finite-size-scaling (FSS) variables: for the 2D
Ising model on a square lattice, s(e, L) -> s_inf(e) as L -> inf, with
corrections that are smooth in 1/L. Working in (e, s) instead of raw
(E, ln g) is what makes curves from very different L's comparable and
lets a single regression model be trained on all of them at once.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

from wl_postprocess import enforce_sum_rule

Array = np.ndarray


# ----------------------------------------------------------------------
# Loading real data
# ----------------------------------------------------------------------

def load_dos_csv(path: str | Path) -> Tuple[Array, Array]:
    """
    Load a single (E, ln_g) table from a CSV/whitespace file.

    Expected columns (header optional, case-insensitive): E, lng (or ln_g).
    IMPORTANT: lines for energetically *inaccessible* values (g = 0)
    should simply be absent from the file -- do not encode them as
    -inf, just omit that E.
    """
    try:
        data = np.genfromtxt(path, delimiter=",", names=True)
    except Exception:
        data = None

    if data is not None and data.dtype.names is not None:
        names = [n.lower() for n in data.dtype.names]
        E = data[data.dtype.names[names.index("e")]]
        lng_key = "lng" if "lng" in names else "ln_g"
        lng = data[data.dtype.names[names.index(lng_key)]]
    else:
        raw = np.loadtxt(path)
        E, lng = raw[:, 0], raw[:, 1]

    order = np.argsort(E)
    return np.asarray(E[order], dtype=float), np.asarray(lng[order], dtype=float)


def load_all_L(data_dir: str | Path, L_list: List[int],
               pattern: str = "L{L}.csv") -> Dict[int, Tuple[Array, Array]]:
    """Load every L in L_list from data_dir/pattern.format(L=L)."""
    data_dir = Path(data_dir)
    out = {}
    for L in L_list:
        f = data_dir / pattern.format(L=L)
        if not f.exists():
            raise FileNotFoundError(f"missing DOS file for L={L}: {f}")
        out[L] = load_dos_csv(f)
    return out


# ----------------------------------------------------------------------
# Reduced variables
# ----------------------------------------------------------------------

def to_reduced(E: Array, lng: Array, L: int) -> Tuple[Array, Array]:
    """(E, ln g) -> (e, s) = (E/L^2, ln g / L^2)."""
    N = L * L
    return E / N, lng / N


def from_reduced(e: Array, s: Array, L: int) -> Tuple[Array, Array]:
    """(e, s) -> (E, ln g), inverse of to_reduced."""
    N = L * L
    return e * N, s * N


# ----------------------------------------------------------------------
# Subsampling (keeps the GP training set tractable)
# ----------------------------------------------------------------------

def subsample_by_e(E: Array, lng: Array, L: int, n_points: int = 30
                    ) -> Tuple[Array, Array]:
    """
    Select ~n_points points roughly evenly spaced in e = E/L^2.

    This matters a lot in practice: for L=256 a WL run already gives you
    ~L^2 ~ 65000 accessible energies. Training a GP on every single
    point from every single L (you may have ~127 L's with dL=2) would
    mean millions of training rows -- infeasible for an exact GP, whose
    cost scales as O(n_train^3). Since s(e, L) is, physically, a smooth
    function of e (an entropy density), a few dozen well-spread points
    per L is enough to pin down its shape; the fine WL resolution is
    only needed in the *final* dense reconstruction, not for fitting
    the regression itself (see gp_model.predict_dense).
    """
    if len(E) <= n_points:
        return E, lng
    e = E / (L * L)
    e_grid = np.linspace(e.min(), e.max(), n_points)
    idx = np.searchsorted(e, e_grid)
    idx = np.clip(idx, 0, len(e) - 1)
    idx = np.unique(idx)
    return E[idx], lng[idx]


def select_training_L(L_list: List[int], max_L: int = 30) -> List[int]:
    """
    Pick an evenly-spaced subset of at most `max_L` values from L_list.

    Why this matters: the GP already treats the 1/L direction as smooth
    (see gp_model.build_kernel) -- so once enough L's are present to
    resolve that smooth dependence, adding more barely changes the fit
    but multiplies training cost, since GP fitting scales as
    O(n_train^3). With dL=2 you may have 100+ available L's; a few
    dozen, evenly spread across the range, is normally enough to get
    the same accuracy at a fraction of the cost. Use the leave-one-out
    check in gp_model.leave_one_L_out to confirm this holds for your
    own data (try increasing max_L and see if RMSE actually improves).
    """
    L_sorted = sorted(L_list)
    if len(L_sorted) <= max_L:
        return L_sorted
    idx = np.linspace(0, len(L_sorted) - 1, max_L).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [L_sorted[i] for i in idx]


def build_training_table(data: Dict[int, Tuple[Array, Array]],
                          max_points_per_L: int = 30
                          ) -> Tuple[Array, Array, Array, Array]:
    """
    Stack every L into one training set for the GP, subsampling each
    L's curve down to `max_points_per_L` points first (see
    subsample_by_e).

    Returns
    -------
    X      : (n, 2) array of [e, 1/L]        -- GP input features
    y      : (n,)   array of s = ln g / L^2  -- GP target
    E_all  : (n,) raw energies (bookkeeping / plotting)
    L_all  : (n,) the L each row came from (bookkeeping / plotting)
    """
    e_list, s_list, E_list, L_list = [], [], [], []
    for L, (E, lng) in data.items():
        E_sub, lng_sub = subsample_by_e(E, lng, L, max_points_per_L)
        e, s = to_reduced(E_sub, lng_sub, L)
        e_list.append(e); s_list.append(s)
        E_list.append(E_sub); L_list.append(np.full_like(E_sub, L))

    e_all = np.concatenate(e_list)
    s_all = np.concatenate(s_list)
    E_all = np.concatenate(E_list)
    L_all = np.concatenate(L_list)
    X = np.column_stack([e_all, 1.0 / L_all])
    return X, s_all, E_all, L_all


# ----------------------------------------------------------------------
# Synthetic data (ONLY for demonstrating / testing the pipeline)
# ----------------------------------------------------------------------

def generate_synthetic_dos(L: int, rng: np.random.Generator,
                            noise: float = 2e-4) -> Tuple[Array, Array]:
    """
    Produce a plausible-looking, but NOT physically exact, ln g(E) curve
    for an LxL square-lattice Ising model: correct qualitative
    finite-size shape, symmetric in E, ground-state degeneracy = 2, and
    properly normalised so sum_E g(E) = 2^(L^2).

    This exists ONLY so run_pipeline.py can be executed before you plug
    in your real Wang-Landau output files. Replace load_all_L(...) with
    real data and skip this function entirely once you have it.
    """
    N = L * L
    Emin, Emax = -2 * N, 2 * N
    E = np.arange(Emin, Emax + 1, 4, dtype=float)  # step 4: typical
                                                    # single-flip energy
                                                    # spacing; replace
                                                    # with your own
                                                    # accessible-energy
                                                    # set when you swap
                                                    # in real data
    e = E / N
    # crude finite-size entropy-per-site ansatz with a 1/L curvature
    # correction -- only meant to *look* like real Ising DOS for demo
    # purposes, not to be physically accurate.
    s_inf = np.log(2.0) * (1 - 0.5 * e**2) - 0.05 * e**4
    correction = (0.15 * np.cos(2 * np.pi * e)) / L
    s = np.clip(s_inf + correction, 0, None)
    s = s + rng.normal(scale=noise, size=s.shape)  # mimic WL stat. noise
    lng = s * N
    lng[0] = lng[-1] = np.log(2.0)                 # pin ground state
    lng = enforce_sum_rule(E, lng, L)               # fix normalisation
    return E, lng
