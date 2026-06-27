"""
fss_predict.py
==============
Ajusta FSS em todos os dados reais disponíveis e prediz s(e) para um L alvo.
Gera plot de data-collapse + curva predita e salva arquivo de warm-start.

Uso:
    python fss_predict.py --target 256
    python fss_predict.py --target 126 --lnf-start 1e-3
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fss_model import FSSModel, load_dat, _enforce_symmetry, _enforce_sum_rule

p = argparse.ArgumentParser()
p.add_argument("--data",    default="../data")
p.add_argument("--out",     default="../out/fss_predict")
p.add_argument("--target",  type=int, required=True)
p.add_argument("--lnf-start", type=float, default=1e-3, dest="lnf_start")
args = p.parse_args()

DATA_DIR = Path(args.data)
OUT_DIR  = Path(args.out)
L_TARGET = args.target
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── carrega dados reais ───────────────────────────────────────────────────────
pat  = re.compile(r"ising_DOS_L_(\d+)\.dat$")
data = {}
for f in sorted(DATA_DIR.glob("ising_DOS_L_*.dat")):
    m = pat.match(f.name)
    if m and "_predicted" not in f.name and "_fss" not in f.name:
        L = int(m.group(1))
        data[L] = load_dat(f)

L_list = sorted(data.keys())
print(f"Dados carregados: L = {L_list}")
print(f"L alvo: {L_TARGET}  (ratio = {L_TARGET/max(L_list):.2f}× L_max)")

# ── ajuste FSS ───────────────────────────────────────────────────────────────
model = FSSModel(n_terms=3).fit(data)

# Resíduo no maior L para avaliar qualidade
L_max = max(L_list)
res   = model.residuals({L_max: data[L_max]})
print(f"Resíduo FSS em L={L_max}: RMSE={res[L_max]['rmse']:.3e}, "
      f"max={res[L_max]['maxerr']:.3e}")

# ── predição para L_TARGET ────────────────────────────────────────────────────
E_pred, lng_pred = model.predict_dos(L_TARGET)
e_pred = E_pred / L_TARGET**2
s_pred = lng_pred / L_TARGET**2

# ── plot: data collapse + curva predita ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

cmap   = plt.cm.plasma
n      = len(L_list)
for k, L in enumerate(L_list):
    E, lng = data[L]
    ax.plot(E / L**2, lng / L**2,
            color=cmap(k / (n - 1)), lw=0.8, alpha=0.7)

# curva s∞ (L→∞)
e_grid = model._e_grid
s_inf  = model._sp_sinf(e_grid)
ax.plot(e_grid, s_inf, "k--", lw=1.5, label=r"$s_\infty(e)$  (FSS fit)")

# curva predita para L_TARGET
ax.plot(e_pred, s_pred, "r-", lw=2.0, label=f"FSS predito  L={L_TARGET}")

# colorbar manual para os L's de treino
sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(vmin=min(L_list), vmax=max(L_list)))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.01)
cb.set_label("L (dados reais)")

ax.set_xlabel(r"$e = E/L^2$",  fontsize=13)
ax.set_ylabel(r"$s = \ln g / L^2$", fontsize=13)
ax.set_title(f"Data collapse FSS  —  predição L={L_TARGET}", fontsize=13)
ax.legend(fontsize=11)
fig.tight_layout()

plot_path = OUT_DIR / f"fss_collapse_L{L_TARGET}.png"
fig.savefig(plot_path, dpi=150)
print(f"Plot salvo: {plot_path}")

# ── salva warm-start ──────────────────────────────────────────────────────────
ws_path = OUT_DIR / f"warmstart_L{L_TARGET}.txt"
header  = (f"WL warm-start  L={L_TARGET}  gerado por fss_predict.py\n"
           f"lnf_start sugerido: {args.lnf_start}\nE  ln_g")
np.savetxt(ws_path, np.column_stack([E_pred, lng_pred]),
           fmt=["%.1f", "%.10f"], header=header)
print(f"Warm-start salvo: {ws_path}")
print(f"\nPara rodar WL:\n"
      f"  ./Ising/wl_warmstart {L_TARGET} {args.lnf_start} {ws_path.resolve()}")
