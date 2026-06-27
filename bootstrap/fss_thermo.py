"""
fss_thermo.py
=============
Extrai T_c e expoentes críticos do modelo de Ising 2D.

Estratégia em duas partes:
  A) Binder cumulante U_L(T) a partir dos dados WL REAIS (L=8..82)
       → T_c do cruzamento, ν do slope de dU/dT
  B) χ(T,L) e C_V(T,L) via FSS sintético para L até 1024
       → γ/ν de χ_max(L) ~ L^(γ/ν)

Por que separar: m⁴ varia como L^(-3/2) (→0 para L grande), tornando o
FSS de m⁴ mal condicionado. m² escala melhor. Binder usa dados reais.
"""

import re, sys, warnings
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.special import logsumexp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fss_model import _enforce_symmetry, _enforce_sum_rule

DATA_DIR = Path("../data")
OUT_DIR  = Path("../out/thermo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))   # 2.26919

# ─── Carregamento ─────────────────────────────────────────────────────────────
def load_dat(path, need_mag=False):
    raw  = np.loadtxt(path, comments="#")
    E    = raw[:, 1].astype(float)
    lng  = raw[:, 2].astype(float)
    m1   = raw[:, 4].astype(float)   # <|M|>/N
    m2   = raw[:, 5].astype(float)   # <M²>/N²
    m4   = raw[:, 6].astype(float)   # <M⁴>/N⁴ (pode ser corrompido)
    cols = [E, lng, m1, m2, m4]
    if need_mag:
        ok = np.all([np.isfinite(c) for c in cols], axis=0)
    else:
        ok = np.isfinite(E) & np.isfinite(lng)
    E, lng, m1, m2, m4 = [c[ok] for c in cols]
    order = np.argsort(E)
    return E[order], lng[order], m1[order], m2[order], m4[order]

pat  = re.compile(r"ising_DOS_L_(\d+)\.dat$")
data_s   = {}   # {L: (E, lng)}           — todos os L's
data_mag = {}   # {L: (E, lng, m2, m4)}   — só L's com m4 válido

def _m4_valid(E, m4):
    """Verifica se m4 é fisicamente correto: m4(E_min) deve ser ≈ 1."""
    gs_mask = E == E.min()
    if not gs_mask.any():
        return False
    gs_val = m4[gs_mask][0]
    return np.isfinite(gs_val) and abs(gs_val - 1.0) < 0.01

for f in sorted(DATA_DIR.glob("ising_DOS_L_*.dat")):
    m_ = pat.match(f.name)
    if not m_ or "_predicted" in f.name or "_fss" in f.name:
        continue
    L = int(m_.group(1))
    E, lng, m1, m2, m4 = load_dat(f, need_mag=False)
    if len(E) > 4:
        data_s[L] = (E, lng)
    E, lng, m1, m2, m4 = load_dat(f, need_mag=True)
    if len(E) > 4 and _m4_valid(E, m4):
        data_mag[L] = (E, lng, m1, m2, m4)

L_real = sorted(data_s.keys())
L_mag  = sorted(data_mag.keys())
print(f"L's DOS:    {L_real}")
print(f"L's magnet: {L_mag}")

# ─── FSS genérico ─────────────────────────────────────────────────────────────
class GenericFSS:
    def __init__(self, n_terms=3, n_grid=500):
        self.n_terms, self.n_grid = n_terms, n_grid

    def fit(self, data_Q):
        L_arr = np.array(sorted(data_Q.keys()), dtype=float)
        n_t   = min(self.n_terms, len(L_arr))
        e_lo  = max(data_Q[int(L)][0].min() / L**2 for L in L_arr if len(data_Q[int(L)][0]) > 0)
        e_hi  = min(data_Q[int(L)][0].max() / L**2 for L in L_arr if len(data_Q[int(L)][0]) > 0)
        e_grid = np.linspace(e_lo, e_hi, self.n_grid)
        Q_grid = np.full((len(L_arr), self.n_grid), np.nan)
        for k, L in enumerate(L_arr):
            E, Q = data_Q[int(L)]
            e    = E / L**2
            ok   = np.isfinite(Q) & np.isfinite(e)
            e_ok, Q_ok = e[ok], Q[ok]
            _, u = np.unique(e_ok, return_index=True)
            if len(u) < 4:
                continue
            cs = CubicSpline(e_ok[u], Q_ok[u], extrapolate=False)
            Q_grid[k] = cs(e_grid)
        x  = 1.0 / L_arr**2;  x2 = x**2
        coeffs = np.full((self.n_grid, 3), np.nan)
        for i in range(self.n_grid):
            qv = Q_grid[:, i];  ok = ~np.isnan(qv)
            if ok.sum() < n_t:
                continue
            if n_t == 1:   A = np.ones((ok.sum(), 1))
            elif n_t == 2: A = np.column_stack([np.ones(ok.sum()), x[ok]])
            else:          A = np.column_stack([np.ones(ok.sum()), x[ok], x2[ok]])
            c, *_ = np.linalg.lstsq(A, qv[ok], rcond=None)
            coeffs[i, :len(c)] = c
        valid = ~np.isnan(coeffs[:, 0])
        self.e_grid  = e_grid
        self._sp_inf = CubicSpline(e_grid[valid], coeffs[valid, 0])
        self._sp_a   = CubicSpline(e_grid[valid], coeffs[valid, 1]) if n_t >= 2 else None
        self._sp_b   = CubicSpline(e_grid[valid], coeffs[valid, 2]) if n_t >= 3 else None
        return self

    def predict(self, e, L):
        e_c = np.clip(e, self.e_grid[0], self.e_grid[-1])
        Q   = self._sp_inf(e_c)
        if self._sp_a: Q = Q + self._sp_a(e_c) / L**2
        if self._sp_b: Q = Q + self._sp_b(e_c) / L**4
        return Q


print("Ajustando FSS (s, |m|, m²)...")
fss_s  = GenericFSS().fit({L: (E, lng/L**2) for L,(E,lng)         in data_s.items()})
fss_m1 = GenericFSS().fit({L: (E, m1)       for L,(E,lng,m1,m2,m4) in data_mag.items()})
fss_m2 = GenericFSS().fit({L: (E, m2)       for L,(E,lng,m1,m2,m4) in data_mag.items()})
print("  OK")

# ─── Médias canônicas ─────────────────────────────────────────────────────────
def canonical(beta, E, lng, m1=None, m2=None, m4=None):
    w  = lng - beta * E;  w -= w.max()
    ew = np.exp(w);       Z  = ew.sum();  p = ew / Z
    avgE  = E  @ p
    avgE2 = (E**2) @ p
    avm1  = (m1 @ p) if m1 is not None else None
    avm2  = (m2 @ p) if m2 is not None else None
    avm4  = (m4 @ p) if m4 is not None else None
    return avgE, avgE2, avm1, avm2, avm4

def thermo_sweep(T_arr, E, lng, L, m1=None, m2=None, m4=None):
    N    = L * L
    Cv   = np.zeros(len(T_arr))
    chi  = np.full(len(T_arr), np.nan)
    U    = np.full(len(T_arr), np.nan)
    for i, T in enumerate(T_arr):
        b = 1.0 / T
        avgE, avgE2, avm1, avm2, avm4 = canonical(b, E, lng, m1, m2, m4)
        Cv[i] = b**2 * (avgE2 - avgE**2) / N
        # Susceptibilidade CONECTADA: χ = β×N×(<m²> - <|m|>²)
        if avm2 is not None and avm1 is not None:
            chi[i] = b * N * (avm2 - avm1**2)
        if avm2 is not None and avm4 is not None and avm2 > 1e-30:
            U[i] = 1.0 - avm4 / (3.0 * avm2**2)
    return Cv, chi, U

# ─── A) Dados REAIS: Binder + C_V + χ ────────────────────────────────────────
T_arr  = np.linspace(1.90, 2.60, 600)
L_binder = [L for L in L_mag if L >= 20]  # L's com dados magnéticos válidos

print(f"Calculando U_L(T) com dados reais para L = {L_binder}...")
res_real = {}
for L in L_binder:
    E, lng, m1, m2, m4 = data_mag[L]
    Cv, chi, U = thermo_sweep(T_arr, E, lng, L, m1=m1, m2=m2, m4=m4)
    res_real[L] = dict(Cv=Cv, chi=chi, U=U)

# ─── B) FSS sintético: χ e C_V para L grandes ────────────────────────────────
L_synth = [64, 96, 128, 192, 256, 384, 512, 768, 1024]
print(f"Calculando χ e C_V sintéticos para L = {L_synth}...")
res_synth = {}
for L in L_synth:
    N  = L * L
    E  = np.arange(-2*N, 2*N + 1, 4, dtype=float)
    e  = E / N
    s  = fss_s.predict(e, L)
    lng = _enforce_symmetry(E, s * N)
    lng = _enforce_sum_rule(E, lng, L)
    m1 = np.clip(fss_m1.predict(e, L), 0, 1)
    m2 = np.clip(fss_m2.predict(e, L), 0, 1)
    Cv, chi, _ = thermo_sweep(T_arr, E, lng, L, m1=m1, m2=m2)
    res_synth[L] = dict(Cv=Cv, chi=chi)
    print(f"  L={L:<5}  χ_max={chi.max():.1f}  C_V_max={Cv.max():.4f}")

# ─── Plots ────────────────────────────────────────────────────────────────────
cmap_r = plt.cm.Blues
cmap_s = plt.cm.Reds

# Plot 1: Binder cumulante (dados reais)
fig, ax = plt.subplots(figsize=(9, 5))
for i, L in enumerate(L_binder):
    c = cmap_r(0.4 + 0.6 * i / max(len(L_binder)-1, 1))
    ax.plot(T_arr, res_real[L]["U"], color=c, lw=1.3, label=f"L={L}")
ax.axvline(TC_EXACT, color="k", ls="--", lw=1.2, label=f"$T_c$ Onsager={TC_EXACT:.4f}")
ax.axhline(2/3,      color="gray", ls=":", lw=0.8, label="U=2/3 (ordem)")
ax.set_xlabel("T", fontsize=13); ax.set_ylabel("$U_L$", fontsize=13)
ax.set_title("Cumulante de Binder — dados WL reais", fontsize=12)
ax.legend(fontsize=7, ncol=3); ax.set_xlim(2.0, 2.55); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT_DIR / "binder.png", dpi=150); plt.close(fig)

# Plot 2: Susceptibilidade (reais + sintéticos)
fig, ax = plt.subplots(figsize=(9, 5))
for i, L in enumerate(L_binder):
    c = cmap_r(0.4 + 0.6 * i / max(len(L_binder)-1, 1))
    ax.plot(T_arr, res_real[L]["chi"], color=c, lw=1.0, ls="-")
for i, L in enumerate(L_synth):
    c = cmap_s(0.4 + 0.6 * i / max(len(L_synth)-1, 1))
    ax.plot(T_arr, res_synth[L]["chi"], color=c, lw=1.4, ls="--", label=f"L={L}*")
ax.axvline(TC_EXACT, color="k", ls=":", lw=1)
ax.set_xlabel("T", fontsize=13); ax.set_ylabel(r"$\chi$", fontsize=13)
ax.set_title(r"Susceptibilidade  (azul=real, vermelho=FSS)", fontsize=12)
ax.legend(fontsize=8, ncol=2); ax.set_xlim(2.0, 2.55); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT_DIR / "susceptibility.png", dpi=150); plt.close(fig)

# Plot 3: C_V
fig, ax = plt.subplots(figsize=(9, 5))
for i, L in enumerate(L_binder):
    c = cmap_r(0.4 + 0.6 * i / max(len(L_binder)-1, 1))
    ax.plot(T_arr, res_real[L]["Cv"], color=c, lw=1.0)
for i, L in enumerate(L_synth):
    c = cmap_s(0.4 + 0.6 * i / max(len(L_synth)-1, 1))
    ax.plot(T_arr, res_synth[L]["Cv"], color=c, lw=1.4, ls="--", label=f"L={L}*")
ax.axvline(TC_EXACT, color="k", ls=":", lw=1)
ax.set_xlabel("T", fontsize=13); ax.set_ylabel("$C_V/N$", fontsize=13)
ax.set_title("Calor específico  (azul=real, vermelho=FSS)", fontsize=12)
ax.legend(fontsize=8, ncol=2); ax.set_xlim(2.0, 2.55); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT_DIR / "specific_heat.png", dpi=150); plt.close(fig)

# ─── T_c do cruzamento de Binder ─────────────────────────────────────────────
crossings = []
L_list = sorted(res_real.keys())
for L1, L2 in zip(L_list[:-1], L_list[1:]):
    diff = res_real[L1]["U"] - res_real[L2]["U"]
    idx  = np.where(np.diff(np.sign(diff)))[0]
    for i in idx:
        t1, t2 = T_arr[i], T_arr[i+1]
        tc = t1 - diff[i] * (t2-t1) / (diff[i+1]-diff[i])
        if 2.10 < tc < 2.45:
            crossings.append((L1, L2, tc))

if crossings:
    tc_vals = [tc for _, _, tc in crossings]
    T_c_binder = np.median(tc_vals)
    print(f"\nCruzamentos Binder: {len(crossings)} pares")
    print(f"  T_c mediana = {T_c_binder:.5f}  (Onsager: {TC_EXACT:.5f})")
else:
    T_c_binder = TC_EXACT
    print("\nSem cruzamentos Binder encontrados — usando T_c de Onsager")

# ─── Expoentes críticos — dados REAIS para scaling ──────────────────────────
# O FSS de m² não captura m²→0 em L→∞ na transição (m²∞ finito no fit),
# logo χ sintético ~ L² em vez de L^(γ/ν). Usamos dados reais para os expoentes.
def power_law(L, a, exp): return a * L**exp
def tc_shift(L, Tc, c, nu): return Tc + c * L**(-1.0/nu)

def peak_T(chi_curve, T_arr, n_fit=7):
    """Posição do pico via ajuste parabólico local (interpola sub-grade)."""
    idx = chi_curve.argmax()
    lo  = max(0, idx - n_fit//2)
    hi  = min(len(T_arr), idx + n_fit//2 + 1)
    t   = T_arr[lo:hi];  c = chi_curve[lo:hi]
    p   = np.polyfit(t, c, 2)
    if p[0] < 0:
        return -p[1] / (2 * p[0])
    return T_arr[idx]

L_real_arr = np.array(L_list)
chi_real   = np.array([res_real[L]["chi"].max()        for L in L_list])
Tpk_real   = np.array([peak_T(res_real[L]["chi"], T_arr) for L in L_list])

# γ/ν de χ_max(L) ~ L^(γ/ν) — dados reais
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    popt_chi, _ = curve_fit(power_law, L_real_arr, chi_real, p0=[1.0, 1.75])
gamma_over_nu = popt_chi[1]

# T_c e ν via T_χ,max(L) → T_c + c L^(-1/ν)
# Usa apenas L >= 50 (exclui L=34 que é outlier por distância ao regime assintótico)
mask_large = L_real_arr >= 50
try:
    popt_tc, _ = curve_fit(tc_shift,
                           L_real_arr[mask_large], Tpk_real[mask_large],
                           p0=[T_c_binder, 0.5, 1.0],
                           bounds=([2.0, 0, 0.2], [2.5, 5.0, 3.0]),
                           maxfev=10000)
    T_c_fss, nu_est = popt_tc[0], popt_tc[2]
except Exception as ex:
    print(f"  Fit T_c(L) falhou: {ex}")
    T_c_fss, nu_est = np.nan, np.nan

# ν de |dU/dT| avaliado em T_c (não no máximo global que pode ser em T errado)
# dU/dT < 0 (U decresce com T) → pegamos valor em i_tc
i_tc = np.argmin(np.abs(T_arr - T_c_binder))
dU_tc = np.array([abs(np.gradient(res_real[L]["U"], T_arr)[i_tc]) for L in L_list])

# usa somente L >= 30 (curvas de L pequenos têm menor resolução em T)
mask_big = L_real_arr >= 30
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    popt_dU, _ = curve_fit(power_law, L_real_arr[mask_big], dU_tc[mask_big],
                           p0=[0.1, 1.0])
nu_binder = 1.0 / popt_dU[1]

# L's sintéticos para referência (só χ; expoente não confiável)
L_synth_arr = np.array(L_synth)
chi_synth   = np.array([res_synth[L]["chi"].max()           for L in L_synth])
Tpk_synth   = np.array([T_arr[res_synth[L]["chi"].argmax()] for L in L_synth])

print(f"\n{'='*55}")
print(f"Expoentes críticos  (2D Ising exato: ν=1, γ/ν=1.75)")
print(f"  γ/ν   χ_max ~ L^(γ/ν) [real]   = {gamma_over_nu:.4f}  (exato 1.750)")
print(f"  ν     T_χ(L)→∞ [real]          = {nu_est:.4f}  (exato 1.000)")
print(f"  ν     |dU/dT|@T_c ~ L^(1/ν)    = {nu_binder:.4f}  (exato 1.000)")
print(f"  T_c   cruzamento Binder         = {T_c_binder:.5f}  (Onsager {TC_EXACT:.5f})")
print(f"  T_c   T_χ(L)→L→∞               = {T_c_fss:.5f}  (Onsager {TC_EXACT:.5f})")
print(f"{'='*55}")

# ─── Plot FSS: χ_max, T_peak, dU/dT vs L ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.loglog(L_real_arr,  chi_real,  "o", color="tab:blue", ms=7, label="WL real")
ax.loglog(L_synth_arr, chi_synth, "s", color="tab:red",  ms=5, alpha=0.6,
          label="FSS sintético (ref.)")
Lf = np.linspace(L_real_arr.min(), L_synth_arr.max(), 300)
ax.loglog(Lf, power_law(Lf, *popt_chi), "b--",
          label=f"fit real: $L^{{{gamma_over_nu:.3f}}}$  (exato 1.75)")
ax.set_xlabel("L"); ax.set_ylabel(r"$\chi_{max}$")
ax.set_title(r"Escala de $\chi_{max}$ (WL real)"); ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.plot(1.0/L_real_arr,  Tpk_real,  "o", color="tab:blue", ms=7, label="WL real")
ax.plot(1.0/L_synth_arr, Tpk_synth, "s", color="tab:red",  ms=5, alpha=0.6,
        label="FSS sintético (ref.)")
L_inv = np.linspace(0, 1.1/L_real_arr.min(), 300)
if not np.isnan(T_c_fss):
    Lv = np.where(L_inv > 0, 1.0/L_inv, 1e10)
    ax.plot(L_inv, tc_shift(Lv, *popt_tc), "b--",
            label=f"fit: $T_c$={T_c_fss:.5f},  ν={nu_est:.3f}")
ax.axhline(TC_EXACT, ls=":", color="gray", label=f"Onsager {TC_EXACT:.5f}")
ax.set_xlabel("$1/L$"); ax.set_ylabel("$T_{\\chi,max}(L)$")
ax.set_title("Extrapolação $T_c$"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
ax.loglog(L_real_arr[mask_big], dU_tc[mask_big], "o", color="tab:green",
          ms=7, label="$|dU/dT|$ @ $T_c$ real (L≥30)")
Lf2 = np.linspace(L_real_arr[mask_big].min(), L_real_arr[mask_big].max(), 200)
ax.loglog(Lf2, power_law(Lf2, *popt_dU), "k--",
          label=f"$\\sim L^{{{popt_dU[1]:.3f}}}$  → ν={nu_binder:.3f}")
ax.set_xlabel("L"); ax.set_ylabel(r"$|dU_L/dT|_{T_c}$")
ax.set_title(r"Escala de $|dU_L/dT|$"); ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

fig.suptitle(f"FSS — Ising 2D  |  T_c(Binder)={T_c_binder:.5f}  "
             f"γ/ν={gamma_over_nu:.3f}  ν(dU)={nu_binder:.3f}  ν(T_χ)={nu_est:.3f}",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "fss_exponents.png", dpi=150)
plt.close(fig)

print(f"\nPlots salvos em {OUT_DIR}/")
