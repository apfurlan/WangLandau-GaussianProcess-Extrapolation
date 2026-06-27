"""
gp_wl_loop.py
=============
Loop iterativo GP + Wang-Landau para estimar ln g(E) em L's grandes.

Algoritmo por iteração:
  1. Ajusta GP em todos os dados disponíveis (reais + WL anteriores)
  2. Prediz s(e, L_new) com L_new = round(1.5 * L_max) ao par mais próximo,
     limitado a L_target
  3. Aplica pós-processamento (simetria + regra de soma)
  4. Salva warm-start e roda wl_warmstart
  5. Carrega resultado do WL → novo L_max
  6. Repete até atingir L_target

Vantagem sobre bootstrap FSS:
  Cada passo avança ~1.5× em vez de +2, exigindo muito menos WL runs.
  De L=82 a L=256: 3 WL runs. De L=82 a L=512: 5 WL runs.

Uso:
    python gp_wl_loop.py --target 256
    python gp_wl_loop.py --target 512 --lnf-start 1e-2
    python gp_wl_loop.py --target 256 --dry-run   # só GP, sem rodar WL
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
GP_DIR = ROOT / "gp"
sys.path.insert(0, str(GP_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import build_training_table, to_reduced, from_reduced, select_training_L
from gp_model import DOSExtrapolator, build_kernel, predict_dense
from wl_postprocess import postprocess, write_wl_input
from fss_model import load_dat

# ── argumentos ───────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data",      default="../data")
p.add_argument("--out",       default="../out/gp_loop")
p.add_argument("--wl",        default="../Ising/wl_warmstart")
p.add_argument("--target",    type=int, required=True)
p.add_argument("--lnf-start", type=float, default=1e-3, dest="lnf_start")
p.add_argument("--max-train-L", type=int, default=30, dest="max_train_L",
               help="Máx de L's usados no treino GP (evita O(n^3) explosão)")
p.add_argument("--dry-run",   action="store_true",
               help="Só GP, não roda WL (usa predição GP como dado)")
args = p.parse_args()

DATA_DIR  = Path(args.data)
OUT_DIR   = Path(args.out)
WL_BIN    = Path(args.wl)
L_TARGET  = args.target
LNF_START = args.lnf_start
DRY_RUN   = args.dry_run

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE  = OUT_DIR / "gp_loop.log"

if not DRY_RUN and not WL_BIN.exists():
    sys.exit(f"Binário WL não encontrado: {WL_BIN}\n"
             f"Compile: cd Ising && g++ -O3 -march=native -std=c++14 "
             f"-o wl_warmstart wl_warmstart.cpp mt.cpp nnlist.cpp "
             f"init.cpp print_lattice.cpp")

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")

def nearest_even(x: float) -> int:
    n = int(round(x))
    return n if n % 2 == 0 else n + 1

def next_L(L_max: int, L_target: int) -> int:
    """Próximo L alvo: 1.5 × L_max arredondado ao par, limitado a L_target."""
    return min(nearest_even(1.5 * L_max), L_target)

def fit_gp(data: dict) -> DOSExtrapolator:
    train_Ls   = select_training_L(sorted(data.keys()), max_L=args.max_train_L)
    train_data = {L: data[L] for L in train_Ls}
    X, y, _, _ = build_training_table(train_data, max_points_per_L=40)
    log(f"  GP treino: {len(train_Ls)} L's, {X.shape[0]} pontos")
    model = DOSExtrapolator(
        kernel=build_kernel(), n_restarts_optimizer=5, normalize_y=True
    )
    model.fit(X, y)
    log(f"  Kernel: {model.fitted_kernel_}")
    return model

def predict_lng(model: DOSExtrapolator, L: int):
    N      = L * L
    E      = np.arange(-2*N, 2*N + 1, 4, dtype=float)
    e      = E / N
    s_pred, s_std = predict_dense(model, e, L, n_coarse=400)
    _, lng_raw    = from_reduced(e, s_pred, L)
    lng            = postprocess(E, lng_raw, L)
    return E, lng, s_std

def run_wl(L: int, ws_path: Path) -> Path:
    cmd = [str(WL_BIN.resolve()), str(L),
           str(LNF_START), str(ws_path.resolve())]
    log(f"  WL: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(OUT_DIR),
                            capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"  ERRO WL L={L}:\n{result.stderr}")
        raise RuntimeError(f"WL falhou para L={L}")
    log(f"  WL L={L} concluído em {elapsed/3600:.2f}h")
    last = result.stdout.strip().splitlines()
    if last:
        log(f"  {last[-1]}")
    return OUT_DIR / f"ising_DOS_L_{L}.dat"

def plot_iteration(data: dict, L_new: int, E_pred: np.ndarray,
                   lng_pred: np.ndarray, s_std: np.ndarray, iteration: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Data collapse
    ax = axes[0]
    cmap  = plt.cm.plasma
    L_all = sorted(data.keys())
    for k, L in enumerate(L_all):
        E, lng = data[L]
        col = cmap(k / max(len(L_all)-1, 1))
        lw  = 1.8 if L == L_new else 0.7
        ax.plot(E/L**2, lng/L**2, color=col, lw=lw,
                label=f"L={L}*" if L == L_new else f"L={L}")
    ax.set_xlabel("e = E/L²"); ax.set_ylabel("s = ln g / L²")
    ax.set_title(f"Data collapse — iter {iteration}  (* = novo WL)")
    handles, labels = ax.get_legend_handles_labels()
    shown = {labels[0], labels[-1]}
    new_h = [h for h, l in zip(handles, labels) if l in shown or "*" in l]
    new_l = [l for l in labels if l in shown or "*" in l]
    ax.legend(new_h, new_l, fontsize=7, ncol=2)

    # GP predição para L_new com IC
    ax = axes[1]
    N   = L_new**2
    e_p = E_pred / N
    s_p = lng_pred / N
    ax.plot(e_p, s_p, "r-", lw=1.5, label=f"GP predito L={L_new}")
    ax.fill_between(e_p, s_p - 2*s_std, s_p + 2*s_std,
                    color="r", alpha=0.15, label="IC 95%")
    ax.set_xlabel("e = E/L²"); ax.set_ylabel("s = ln g / L²")
    ax.set_title(f"Predição GP → L={L_new}  "
                 f"(std médio={s_std.mean():.2e})")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / f"gp_iter{iteration:03d}_L{L_new}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"  Plot: {path.name}")

# ── carrega dados reais iniciais ──────────────────────────────────────────────
pat  = re.compile(r"ising_DOS_L_(\d+)\.dat$")
data = {}
for f in sorted(DATA_DIR.glob("ising_DOS_L_*.dat")):
    m = pat.match(f.name)
    if m and "_predicted" not in f.name and "_fss" not in f.name:
        data[int(m.group(1))] = load_dat(f)

if not data:
    sys.exit(f"Nenhum .dat encontrado em {DATA_DIR}")

log(f"Dados reais: L = {sorted(data.keys())}")
log(f"L_TARGET={L_TARGET}, LNF_START={LNF_START}, DRY_RUN={DRY_RUN}")
log("-" * 60)

# ── loop principal ────────────────────────────────────────────────────────────
iteration = 0
while max(data.keys()) < L_TARGET:
    L_max = max(data.keys())
    L_new = next_L(L_max, L_TARGET)

    log(f"\n=== Iteração {iteration} | L_max={L_max} → L_new={L_new} "
        f"(ratio={L_new/L_max:.2f}×) ===")

    # 1. Ajusta GP
    log("  Ajustando GP...")
    t0    = time.time()
    model = fit_gp(data)
    log(f"  GP ajustado em {time.time()-t0:.1f}s")

    # 2. Prediz DOS via GP
    E_pred, lng_pred, s_std = predict_lng(model, L_new)
    log(f"  GP std médio (por sítio) = {s_std.mean():.3e}")
    log(f"  GP std max  (por sítio)  = {s_std.max():.3e}")

    # 3. Salva warm-start
    ws_path = OUT_DIR / f"warmstart_L{L_new}.txt"
    write_wl_input(ws_path, E_pred, lng_pred, L_new)
    log(f"  Warm-start: {ws_path.name}")

    # 4. Plot da iteração
    plot_iteration(data, L_new, E_pred, lng_pred, s_std, iteration)

    if DRY_RUN:
        data[L_new] = (E_pred, lng_pred)
        log(f"  [dry-run] Usando predição GP como dado para L={L_new}")
    else:
        # 5. Roda WL com warm-start
        dat_path = run_wl(L_new, ws_path)
        if dat_path.exists():
            data[L_new] = load_dat(dat_path)
            log(f"  Carregado WL L={L_new}: {len(data[L_new][0])} energias")
        else:
            log(f"  AVISO: WL não gerou {dat_path.name} — usando predição GP")
            data[L_new] = (E_pred, lng_pred)

    iteration += 1

log("\n" + "="*60)
log(f"Loop concluído. L's disponíveis: {sorted(data.keys())}")
log(f"Saída em: {OUT_DIR}")

# ── resumo das iterações ──────────────────────────────────────────────────────
log(f"\nPara rodar WL no L final com warm-start gerado:")
log(f"  {WL_BIN} {L_TARGET} {LNF_START} "
    f"{(OUT_DIR / f'warmstart_L{L_TARGET}.txt').resolve()}")
