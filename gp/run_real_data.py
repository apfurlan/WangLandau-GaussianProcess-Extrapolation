"""
run_real_data.py
================
Pipeline completo de extrapolação GP para o modelo de Ising 2D.

Estrutura esperada do repositório
----------------------------------
    WangLandau-GaussianProcess-Extrapolation/
    ├── data/               ←  arquivos ising_DOS_L_<L>.dat
    ├── gp/
    │   ├── run_real_data.py   ← este script
    │   ├── data_utils.py
    │   ├── gp_model.py
    │   └── wl_postprocess.py
    └── ...

Execute a partir da pasta gp/:
    cd gp/
    python run_real_data.py

Ou passe caminhos como argumentos:
    python run_real_data.py --data ../data --out ../out
"""

import argparse
import sys
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from data_utils import build_training_table, to_reduced, from_reduced
from gp_model import DOSExtrapolator, build_kernel, leave_one_L_out, predict_dense
from wl_postprocess import postprocess, write_wl_input

# ── Argumentos de linha de comando ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="GP extrapolation of WL density of states")
parser.add_argument("--data", default="../data",  help="Pasta com os arquivos .dat")
parser.add_argument("--out",  default="../out",   help="Pasta de saída")
args = parser.parse_args()

DATA_DIR = Path(args.data)
OUT_DIR  = Path(args.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Detecção automática dos L's disponíveis ───────────────────────────────
# Lê todos os ising_DOS_L_<L>.dat, excluindo arquivos _predicted
pat = re.compile(r"ising_DOS_L_(\d+)\.dat$")
L_list = sorted(
    int(m.group(1))
    for f in DATA_DIR.glob("ising_DOS_L_*.dat")
    if (m := pat.match(f.name)) and "_predicted" not in f.name
)

if not L_list:
    sys.exit(f"Nenhum arquivo ising_DOS_L_*.dat encontrado em {DATA_DIR}")

print(f"L's encontrados ({len(L_list)}): {L_list}")
L_holdout = L_list[-1]       # maior L disponível → holdout na validação
L_targets = []               # L's para extrapolar (preenchido abaixo)

# Extrapola para o próximo passo além do maior L simulado,
# dentro do limite de confiança (~1.5 × L_max)
L_max = L_list[-1]
step  = L_list[-1] - L_list[-2]   # inferido dos dados
for L_t in range(L_max + step, int(1.5 * L_max) + 1, step):
    L_targets.append(L_t)
L_targets = L_targets[:3]    # no máximo 3 alvos para não poluir os plots

print(f"L holdout (validação): {L_holdout}")
print(f"L's alvo (extrapolação): {L_targets}\n")

# ── 2. Carregamento ──────────────────────────────────────────────────────────
def load_dat(path: Path):
    """Carrega E e log[g(E)] de um arquivo WL no formato padrão."""
    raw  = np.loadtxt(path, comments="#")
    E    = raw[:, 1].astype(float)
    lng  = raw[:, 2].astype(float)
    order = np.argsort(E)
    return E[order], lng[order]

data = {}
print("=== Carregando dados ===")
for L in L_list:
    E, lng = load_dat(DATA_DIR / f"ising_DOS_L_{L}.dat")
    data[L] = (E, lng)
    print(f"  L={L:4d}: {len(E):5d} energias, "
          f"lng_max={lng.max():.4f}, "
          f"lng(E_min)={lng[0]:.4f}")

# ── 3. Leave-one-out (holdout = maior L) ────────────────────────────────────
print(f"\n=== Leave-one-out: holdout L={L_holdout} ===")
loo = leave_one_L_out(
    data, L_holdout,
    kernel=build_kernel(),
    max_points_per_L=40,
)
print(f"  RMSE (s por sítio)      = {loo['rmse']:.3e}")
print(f"  max |erro| (por sítio)  = {loo['max_abs_err']:.3e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(loo["e"], loo["s_true"], "k.", ms=4, label=f"WL real (L={L_holdout})")
ax1.plot(loo["e"], loo["s_pred"], "r-", lw=1.5, label="GP previsto")
ax1.fill_between(loo["e"],
                 loo["s_pred"] - 2*loo["s_std"],
                 loo["s_pred"] + 2*loo["s_std"],
                 color="r", alpha=0.18, label="IC 95%")
ax1.set_xlabel("e = E/L²"); ax1.set_ylabel("s = ln g / L²")
ax1.set_title(f"Leave-one-out: L={L_holdout} retido"); ax1.legend(fontsize=8)

ax2.plot(loo["e"], loo["s_pred"] - loo["s_true"], lw=1.2)
ax2.axhline(0, color="k", lw=0.5, ls="--")
ax2.set_xlabel("e = E/L²"); ax2.set_ylabel("previsto – real")
ax2.set_title("Resíduos")
fig.tight_layout()
fig.savefig(OUT_DIR / f"01_leave_one_out_L{L_holdout}.png", dpi=150)
plt.close(fig)
print(f"  → 01_leave_one_out_L{L_holdout}.png")

# ── 4. Data collapse ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.plasma
for i, L in enumerate(L_list):
    e, s = to_reduced(*data[L], L)
    ax.plot(e, s, color=cmap(i / max(len(L_list)-1, 1)), lw=0.8,
            label=f"L={L}" if L in (L_list[0], L_list[-1]) else None)
ax.set_xlabel("e = E/L²"); ax.set_ylabel("s = ln g / L²")
ax.set_title("Data collapse – dados reais de WL")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "02_data_collapse.png", dpi=150)
plt.close(fig)
print("  → 02_data_collapse.png")

# ── 5. Fit final do GP (todos os L's) ────────────────────────────────────────
print(f"\n=== Ajuste final do GP ({len(L_list)} L's) ===")
X, y, _, _ = build_training_table(data, max_points_per_L=40)
print(f"  Pontos de treino: {X.shape[0]}")
model = DOSExtrapolator(kernel=build_kernel(), n_restarts_optimizer=5, normalize_y=True)
model.fit(X, y)
print(f"  Kernel ajustado: {model.fitted_kernel_}")

# ── 6. Extrapolação ───────────────────────────────────────────────────────────
print(f"\n=== Extrapolação ===")
fig, axes = plt.subplots(1, len(L_targets), figsize=(6 * len(L_targets), 4.5))
if len(L_targets) == 1:
    axes = [axes]

for ax, L_t in zip(axes, L_targets):
    N_t     = L_t ** 2
    E_t     = np.arange(-2*N_t, 2*N_t + 1, 4, dtype=float)
    e_t     = E_t / N_t
    s_pred, s_std = predict_dense(model, e_t, L_t, n_coarse=400)
    _, lng_raw    = from_reduced(e_t, s_pred, L_t)
    lng_final     = postprocess(E_t, lng_raw, L_t)

    check  = logsumexp(lng_final) - N_t * np.log(2)
    gs_dev = abs(lng_final[0] - np.log(2))
    print(f"  L={L_t}:")
    print(f"    std GP médio (por sítio) = {s_std.mean():.3e}")
    print(f"    std GP max  (por sítio)  = {s_std.max():.3e}")
    print(f"    regra de soma (≈0)       = {check:.4f}")
    print(f"    |lng(E_min) - ln2|       = {gs_dev:.4f}")

    # Salva arquivos de chute inicial
    txt_path = OUT_DIR / f"ln_g_initial_L{L_t}.txt"
    write_wl_input(txt_path, E_t, lng_final, L_t)

    dat_path = OUT_DIR / f"ising_DOS_L_{L_t}_predicted.dat"
    idx = (2 * (E_t - E_t.min())).astype(int)
    with open(dat_path, "w") as fh:
        fh.write("#i \tE(i) \t \tlog[g(E)] \t  H(E) \t \t<M> \t \t<M^2> \t \t<M^4>\n")
        fh.write("#" + "-"*100 + "\n")
        for iv, Ei, lngi in zip(idx, E_t, lng_final):
            fh.write(f"{iv} \t {Ei:.2f} \t{lngi:.6f}       0 \t \t0.000000 \t0.000000 \t0.000000 \n")

    print(f"    → {txt_path.name}  +  {dat_path.name}")

    ax.plot(e_t, lng_raw / N_t,   "b-", lw=1, alpha=0.4, label="GP bruto")
    ax.plot(e_t, lng_final / N_t, "r-", lw=1.3, label="pós-processado")
    ax.fill_between(e_t,
                    (lng_raw - 2*N_t*s_std) / N_t,
                    (lng_raw + 2*N_t*s_std) / N_t,
                    color="b", alpha=0.10, label="IC 95%")
    ax.set_xlabel("e = E/L²"); ax.set_ylabel("s = ln g / L²")
    ax.set_title(f"Extrapolação → L={L_t}")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "03_extrapolation.png", dpi=150)
plt.close(fig)

print("\n=== Concluído ===")