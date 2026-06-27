"""
plot_lng.py  —  plota ln g(E) vs E para L's reais e preditos via FSS
"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fss_model import FSSModel, load_dat

DATA_DIR = Path("../data")
OUT_DIR  = Path("../out/fss_predict")
OUT_DIR.mkdir(parents=True, exist_ok=True)

L_PLOT = [32, 64, 128, 256]

# ── carrega todos os dados reais ──────────────────────────────────────────────
pat  = re.compile(r"ising_DOS_L_(\d+)\.dat$")
data = {}
for f in sorted(DATA_DIR.glob("ising_DOS_L_*.dat")):
    m = pat.match(f.name)
    if m and "_predicted" not in f.name and "_fss" not in f.name:
        data[int(m.group(1))] = load_dat(f)

# ── ajusta FSS para os L's não simulados ─────────────────────────────────────
model = FSSModel(n_terms=3).fit(data)

# ── plot ──────────────────────────────────────────────────────────────────────
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
fig, ax = plt.subplots(figsize=(10, 6))

for color, L in zip(colors, L_PLOT):
    if L in data:
        E, lng = data[L]
        label = f"L={L}  (WL real)"
        ls, lw = "-", 1.8
    else:
        E, lng = model.predict_dos(L)
        label = f"L={L}  (FSS predito)"
        ls, lw = "--", 2.0

    N = L * L
    ax.plot(E / N, lng, color=color, ls=ls, lw=lw, label=label)

ax.set_xlabel(r"$E/N$", fontsize=14)
ax.set_ylabel(r"$\ln g(E)$", fontsize=14)
ax.set_title(r"Densidade de estados: $\ln g(E)$ vs $E/N$", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = OUT_DIR / "lng_vs_E_multiL.png"
fig.savefig(out, dpi=150)
print(f"Salvo: {out}")
