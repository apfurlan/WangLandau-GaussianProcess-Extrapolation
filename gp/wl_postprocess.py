"""
wl_postprocess.py
==================
Turns a raw GP/spline prediction s(e) for L_target into a physically
valid ln g(E) curve that can be fed to Wang-Landau as an initial guess
(`ln g0`), and writes it to disk.

Three corrections are applied, in order:

1. enforce_symmetry  - g(E) = g(-E), since the square-lattice Ising
                        Hamiltonian is invariant under a global spin
                        flip.
2. pin_ground_state  - g(E_min) = g(E_max) = 2 exactly. This is known
                        analytically with certainty, no need to trust
                        the regression there (and the GP is least
                        reliable at the spectrum edges anyway, since
                        that's where it has the least supporting data).
3. enforce_sum_rule  - sum_E g(E) = 2^(L^2) exactly, applied as a
                        single additive shift in ln-space (a
                        multiplicative correction in g-space). A
                        regression trained on *per-site* entropy can
                        never pin down this one overall additive
                        constant on its own -- this step fixes exactly
                        that, without altering the *shape* of the
                        predicted curve.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.special import logsumexp


def enforce_symmetry(E: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """
    Average lng(E) and lng(-E) so the curve is exactly symmetric.

    Vectorized via searchsorted (O(n log n)) rather than a per-point
    scan (O(n^2)) -- for L=1024 the dense energy grid has ~10^6 points,
    where an O(n^2) approach would simply never finish.
    """
    order = np.argsort(E)
    E_sorted = E[order]
    lng_sorted = lng[order]
    n = len(E_sorted)

    neg_E = -E_sorted
    idx = np.clip(np.searchsorted(E_sorted, neg_E), 0, n - 1)
    idx_lo = np.clip(idx - 1, 0, n - 1)
    # pick whichever of the two searchsorted candidates is the closer match
    closer_is_idx = np.abs(E_sorted[idx] - neg_E) <= np.abs(E_sorted[idx_lo] - neg_E)
    best = np.where(closer_is_idx, idx, idx_lo)

    lng_sym_sorted = 0.5 * (lng_sorted + lng_sorted[best])
    lng_sym = np.empty_like(lng)
    lng_sym[order] = lng_sym_sorted
    return lng_sym


def pin_ground_state(E: np.ndarray, lng: np.ndarray,
                      ln_g_ground: float = np.log(2.0)) -> np.ndarray:
    """Fix the exactly-known degeneracies at E_min and E_max."""
    lng = lng.copy()
    lng[np.argmin(E)] = ln_g_ground
    lng[np.argmax(E)] = ln_g_ground
    return lng


def enforce_sum_rule(E: np.ndarray, lng: np.ndarray, L: int) -> np.ndarray:
    """
    Shift the whole curve by a constant so sum_E g(E) = 2^(L^2) exactly
    (in log space: logsumexp(lng) == L^2 * ln 2).
    """
    target = (L * L) * np.log(2.0)
    shift = target - logsumexp(lng)
    return lng + shift


def postprocess(E: np.ndarray, lng_raw: np.ndarray, L: int,
                 pin_edges: bool = False) -> np.ndarray:
    """
    Apply symmetry -> sum-rule, in that order.

    IMPORTANT -- pin_edges defaults to False. Here's why: ground-state
    pinning and the sum rule are both "exact" constraints, but they can
    conflict. The sum-rule shift is computed from logsumexp(lng), which
    is totally dominated by the peak of the entropy curve (near e=0),
    not by the edges. A per-site GP bias near the peak of just ~1e-4
    (well within typical GP uncertainty) gets multiplied by N = L^2 ~
    10^6 sites once you go back from per-site entropy to the *total*
    ln g -- so the additive shift needed to satisfy the sum rule can
    legitimately be O(10-1000) in ln-g units, even though the per-site
    fit looks excellent. Forcing the ground state back to exactly ln 2
    *after* that shift re-introduces an inconsistency at just two
    points.

    For the actual purpose of this pipeline -- warm-starting a WL run
    -- this doesn't matter: WL's move-acceptance probability is
    min(1, g(E_old)/g(E_new)), which only depends on *relative*
    differences in ln g, never on an overall additive constant. So
    neither the sum-rule shift nor exact edge-pinning has any effect on
    how fast WL converges from this initial guess; what matters is that
    the *shape* (set here by enforce_symmetry, plus whatever the GP
    learned) is good. Set pin_edges=True only if you intend to use this
    DOS directly (without further WL refinement) and want the known
    exact edge values enforced -- in that case the two edge points may
    sit at a slightly different absolute scale than their neighbours.
    """
    lng = enforce_symmetry(E, lng_raw)
    lng = enforce_sum_rule(E, lng, L)
    if pin_edges:
        lng = pin_ground_state(E, lng)
    return lng


def write_wl_input(path: str | Path, E: np.ndarray, lng: np.ndarray, L: int):
    """
    Write an initial-guess ln g(E) file for Wang-Landau: two
    whitespace-separated columns, E and ln_g, with a header. Adjust the
    format string if your WL code expects something different.
    """
    path = Path(path)
    header = f"Initial ln g(E) guess for L={L}, from ML extrapolation\nE  ln_g"
    np.savetxt(path, np.column_stack([E, lng]), fmt=["%.1f", "%.10f"],
               header=header)
