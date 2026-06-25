"""
fss_model.py
============
Extrapolação via Finite-Size Scaling (FSS) da densidade de estados do
modelo de Ising 2D.

Teoria
------
A entropia por sítio s(e, L) = ln g(E, L) / L² converge para o limite
termodinâmico s∞(e) como:

    s(e, L) = s∞(e) + a(e)/L² + b(e)/L⁴ + ...

A expansão é em potências de 1/L² (e não 1/L) porque a rede quadrada 2D
com PBC é bipartida e simétrica — termos de ordem ímpar em 1/L se cancelam.

Para cada ponto e na grade comum:
  - Coleta s(e, L) para todos os L disponíveis
  - Ajusta regressão linear em x = 1/L²: s = s∞ + a·x + b·x²
  - Armazena os coeficientes como splines cúbicas em e

Uso
---
    from fss_model import FSSModel
    model = FSSModel().fit(data)          # data = {L: (E, lng)}
    E, lng = model.predict_dos(128)       # extrapola para L=128
"""

import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp
from pathlib import Path
from typing import Dict, Tuple


class FSSModel:
    """
    Ajuste e predição FSS da DOS do Ising 2D.

    Parameters
    ----------
    n_terms : int
        Número de termos FSS: 1 → só s∞; 2 → s∞ + a/L²; 3 → s∞ + a/L² + b/L⁴
        O padrão 3 requer ao menos 3 L's no conjunto de treino.
    n_grid : int
        Número de pontos na grade comum de e para o ajuste.
    """

    def __init__(self, n_terms: int = 3, n_grid: int = 600):
        self.n_terms = n_terms
        self.n_grid  = n_grid
        self._fitted = False

    # ── ajuste ───────────────────────────────────────────────────────────────
    def fit(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> "FSSModel":
        """
        Ajusta o modelo FSS sobre todos os L's disponíveis.

        Parameters
        ----------
        data : dict  {L: (E, lng)}
            E  : energias físicas (não double-counted)
            lng: ln g(E) correspondente
        """
        L_arr = np.array(sorted(data.keys()), dtype=float)
        n_L   = len(L_arr)

        n_terms = min(self.n_terms, n_L)
        if n_terms < self.n_terms:
            warnings.warn(f"Apenas {n_L} L's disponíveis — usando {n_terms} termos FSS.")

        # ── grade comum em e ─────────────────────────────────────────────────
        # Intersecção dos intervalos de e de todos os L's (evita extrapolação)
        e_lo = max(data[int(L)][0].min() / L**2 for L in L_arr)
        e_hi = min(data[int(L)][0].max() / L**2 for L in L_arr)
        e_grid = np.linspace(e_lo, e_hi, self.n_grid)

        # ── interpola s(e,L) de cada L na grade comum ────────────────────────
        s_grid = np.full((n_L, self.n_grid), np.nan)
        for k, L in enumerate(L_arr):
            E, lng = data[int(L)]
            e = E / L**2
            s = lng / L**2
            cs = CubicSpline(e, s, extrapolate=False)
            s_grid[k] = cs(e_grid)

        # ── regressão FSS em cada ponto da grade ─────────────────────────────
        x   = 1.0 / L_arr**2          # variável de regressão principal
        x2  = x**2                     # segundo termo

        coeffs = np.full((self.n_grid, 3), np.nan)   # [s∞, a, b]

        for i in range(self.n_grid):
            sv = s_grid[:, i]
            ok = ~np.isnan(sv)
            if ok.sum() < n_terms:
                continue

            sv_ok = sv[ok]
            xv    = x[ok]
            x2v   = x2[ok]

            if n_terms == 1:
                A = np.ones((ok.sum(), 1))
            elif n_terms == 2:
                A = np.column_stack([np.ones(ok.sum()), xv])
            else:
                A = np.column_stack([np.ones(ok.sum()), xv, x2v])

            c, _, _, _ = np.linalg.lstsq(A, sv_ok, rcond=None)
            coeffs[i, :len(c)] = c

        # ── splines dos coeficientes ─────────────────────────────────────────
        valid = ~np.isnan(coeffs[:, 0])
        self._e_grid    = e_grid
        self._sp_sinf   = CubicSpline(e_grid[valid], coeffs[valid, 0])

        if n_terms >= 2:
            vv = ~np.isnan(coeffs[:, 1])
            self._sp_a = CubicSpline(e_grid[vv], coeffs[vv, 1])
        else:
            self._sp_a = None

        if n_terms >= 3:
            vv = ~np.isnan(coeffs[:, 2])
            self._sp_b = CubicSpline(e_grid[vv], coeffs[vv, 2])
        else:
            self._sp_b = None

        self._n_terms_fit = n_terms
        self._L_arr       = L_arr
        self._fitted      = True
        return self

    # ── predição ─────────────────────────────────────────────────────────────
    def predict_s(self, e: np.ndarray, L: int) -> np.ndarray:
        """Prediz s(e, L) = ln g / L² para um L alvo."""
        self._check_fitted()
        e_clip = np.clip(e, self._e_grid[0], self._e_grid[-1])
        s = self._sp_sinf(e_clip)
        if self._sp_a is not None:
            s = s + self._sp_a(e_clip) / L**2
        if self._sp_b is not None:
            s = s + self._sp_b(e_clip) / L**4
        return s

    def predict_dos(self, L: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz (E, lng) para um L alvo, com simetria e regra de soma aplicadas.

        Returns
        -------
        E   : energias físicas, passo 4
        lng : ln g(E) estimado
        """
        self._check_fitted()
        N  = L * L
        E  = np.arange(-2*N, 2*N + 1, 4, dtype=float)
        e  = E / N

        # Predição na faixa treinada; fora dela usa s∞ (sem correção finita)
        in_range = (e >= self._e_grid[0]) & (e <= self._e_grid[-1])
        s = np.zeros(len(E))
        s[ in_range] = self.predict_s(e[ in_range], L)
        s[~in_range] = self._sp_sinf(
            np.clip(e[~in_range], self._e_grid[0], self._e_grid[-1])
        )

        lng = s * N
        lng = _enforce_symmetry(E, lng)
        lng = _enforce_sum_rule(E, lng, L)
        return E, lng

    # ── diagnóstico ──────────────────────────────────────────────────────────
    def residuals(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> dict:
        """
        Computa resíduos (s_previsto - s_real) para todos os L's do treino.
        Útil para checar a qualidade do ajuste.
        """
        self._check_fitted()
        res = {}
        for L in sorted(data.keys()):
            E, lng = data[L]
            e = E / L**2
            s_real = lng / L**2
            s_pred = self.predict_s(e, L)
            res[L] = {
                "e":      e,
                "rmse":   float(np.sqrt(np.mean((s_pred - s_real)**2))),
                "maxerr": float(np.max(np.abs(s_pred - s_real))),
            }
        return res

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict().")


# ── funções auxiliares de pós-processamento ───────────────────────────────────
def _enforce_symmetry(E: np.ndarray, lng: np.ndarray) -> np.ndarray:
    order = np.argsort(E)
    Es = E[order]; ls = lng[order]
    idx = np.clip(np.searchsorted(Es, -Es), 0, len(Es)-1)
    idx0 = np.clip(idx-1, 0, len(Es)-1)
    best = np.where(np.abs(Es[idx] + Es) <= np.abs(Es[idx0] + Es), idx, idx0)
    ls_sym = 0.5*(ls + ls[best])
    out = np.empty_like(lng)
    out[order] = ls_sym
    return out

def _enforce_sum_rule(E: np.ndarray, lng: np.ndarray, L: int) -> np.ndarray:
    target = L**2 * np.log(2.0)
    return lng + (target - logsumexp(lng))


# ── utilitário: carrega arquivo .dat ─────────────────────────────────────────
def load_dat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Lê E e log[g(E)] de um arquivo WL no formato padrão."""
    raw = np.loadtxt(path, comments="#")
    E, lng = raw[:, 1].astype(float), raw[:, 2].astype(float)
    order = np.argsort(E)
    return E[order], lng[order]
