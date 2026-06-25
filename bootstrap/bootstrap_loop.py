"""
bootstrap_loop.py
=================
Loop de bootstrapping FSS + Wang-Landau para estimar a DOS do Ising 2D
em L's arbitrariamente grandes.

Fluxo a cada iteração
---------------------
  1. Ajusta FSSModel em todos os dados reais disponíveis
  2. Prediz DOS para os próximos BATCH_SIZE L's
  3. Salva warm-start e roda WL com --lnf-start (pula iterações grosseiras)
  4. Carrega resultado do WL e adiciona ao pool de dados reais
  5. Repete até atingir L_TARGET

Uso
---
    python bootstrap_loop.py \\
        --data    ../data \\
        --out     ../out/bootstrap \\
        --wl      ../Ising/wl_warmstart \\
        --target  4096 \\
        --batch   4 \\
        --lnf-start 1e-3

Argumentos
----------
--data       : pasta com os arquivos ising_DOS_L_*.dat reais
--out        : pasta de saída (criada se não existir)
--wl         : caminho para o binário wl_warmstart compilado
--target     : L máximo a atingir (default 4096)
--batch      : L's estimados por iteração antes de rodar WL (default 4)
--lnf-start  : valor de ln f em que o WL começa (default 1e-3)
               Reduzir acelera mas exige warm-start mais preciso.
--step       : passo entre L's consecutivos (default 2, i.e. dL=2)
--fss-terms  : número de termos FSS: 1, 2 ou 3 (default 3)
--dry-run    : só extrapola FSS, não roda WL (útil para testar)
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

from fss_model import FSSModel, load_dat, _enforce_symmetry, _enforce_sum_rule

# ── argumentos ───────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data",      default="../data")
p.add_argument("--out",       default="../out/bootstrap")
p.add_argument("--wl",        default="../Ising/wl_warmstart")
p.add_argument("--target",    type=int,   default=4096)
p.add_argument("--batch",     type=int,   default=4)
p.add_argument("--lnf-start", type=float, default=1e-3, dest="lnf_start")
p.add_argument("--step",      type=int,   default=2)
p.add_argument("--fss-terms", type=int,   default=3,    dest="fss_terms")
p.add_argument("--dry-run",   action="store_true")
args = p.parse_args()

DATA_DIR   = Path(args.data)
OUT_DIR    = Path(args.out)
WL_BIN     = Path(args.wl)
L_TARGET   = args.target
BATCH_SIZE = args.batch
LNF_START  = args.lnf_start
DL         = args.step
FSS_TERMS  = args.fss_terms
DRY_RUN    = args.dry_run

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE   = OUT_DIR / "bootstrap.log"

if not DRY_RUN and not WL_BIN.exists():
    sys.exit(f"Binário WL não encontrado: {WL_BIN}\n"
             f"Compile com:\n"
             f"  cd ../Ising && g++ -O3 -march=native -std=c++14 \\\n"
             f"      -o wl_warmstart wl_warmstart.cpp mt.cpp nnlist.cpp "
             f"init.cpp print_lattice.cpp")

# ── funções auxiliares ───────────────────────────────────────────────────────
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")

def write_warmstart(path: Path, E: np.ndarray, lng: np.ndarray, L: int):
    """Salva arquivo de warm-start no formato lido pelo wl_warmstart.cpp."""
    header = f"WL warm-start  L={L}  gerado por bootstrap_loop.py\nE  ln_g"
    np.savetxt(path, np.column_stack([E, lng]),
               fmt=["%.1f", "%.10f"], header=header)

def run_wl(L: int, warmstart_file: Path) -> Path:
    """
    Roda o WL com warm-start e retorna o caminho do .dat gerado.
    O binário é executado dentro de OUT_DIR para que o .dat seja salvo lá.
    """
    cmd = [str(WL_BIN.resolve()), str(L),
           str(LNF_START), str(warmstart_file.resolve())]
    log(f"  Rodando WL: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(OUT_DIR),
                            capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"  ERRO WL L={L}:\n{result.stderr}")
        raise RuntimeError(f"WL falhou para L={L}")
    log(f"  WL L={L} concluído em {elapsed:.1f}s")
    log(f"  {result.stdout.strip().splitlines()[-1]}")   # última linha do WL
    return OUT_DIR / f"ising_DOS_L_{L}.dat"

def save_fss_dat(E: np.ndarray, lng: np.ndarray, L: int) -> Path:
    """
    Salva a DOS estimada por FSS no formato .dat padrão.
    Coluna i = 2*(E - E_min); colunas H, M, M², M⁴ = 0.
    """
    E_min = E.min()
    idx   = (2*(E - E_min)).astype(int)
    path  = OUT_DIR / f"ising_DOS_L_{L}_fss.dat"
    with open(path, "w") as fh:
        fh.write("#i \tE(i) \t \tlog[g(E)] \t  H(E) \t \t<M> \t \t<M^2> \t \t<M^4>\n")
        fh.write("#" + "-"*100 + "\n")
        for iv, Ei, lngi in zip(idx, E, lng):
            fh.write(f"{iv} \t {Ei:.2f} \t{lngi:.6f}       0 "
                     f"\t \t0.000000 \t0.000000 \t0.000000 \n")
    return path

def plot_step(data: dict, L_new_list: list, iteration: int):
    """Salva um plot do data-collapse após cada batch."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.cm.plasma
    all_L = sorted(data.keys())
    for k, L in enumerate(all_L):
        E, lng = data[L]
        e = E / L**2;  s = lng / L**2
        color = cmap(k / max(len(all_L)-1, 1))
        lw    = 1.8 if L in L_new_list else 0.7
        ax.plot(e, s, color=color, lw=lw,
                label=f"L={L}*" if L in L_new_list else f"L={L}")
    ax.set_xlabel("e = E/L²"); ax.set_ylabel("s = lng/L²")
    ax.set_title(f"Data collapse — iteração {iteration}  "
                 f"(* = WL com warm-start)")
    # Legenda apenas para o primeiro e último L de cada grupo
    handles, labels = ax.get_legend_handles_labels()
    shown = {labels[0], labels[-1]}
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l in shown or "*" in l:
            new_h.append(h); new_l.append(l)
    ax.legend(new_h, new_l, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"collapse_iter{iteration:03d}.png", dpi=120)
    plt.close(fig)

# ── carrega dados reais iniciais ─────────────────────────────────────────────
pat = re.compile(r"ising_DOS_L_(\d+)\.dat$")
data = {}
for f in sorted(DATA_DIR.glob("ising_DOS_L_*.dat")):
    m = pat.match(f.name)
    if m and "_predicted" not in f.name and "_fss" not in f.name:
        L = int(m.group(1))
        data[L] = load_dat(f)

if not data:
    sys.exit(f"Nenhum .dat encontrado em {DATA_DIR}")

log(f"Dados reais carregados: L = {sorted(data.keys())}")
log(f"L_TARGET={L_TARGET}, BATCH={BATCH_SIZE}, DL={DL}, "
    f"LNF_START={LNF_START}, DRY_RUN={DRY_RUN}")
log("-" * 60)

# ── loop principal ────────────────────────────────────────────────────────────
iteration = 0
while max(data.keys()) < L_TARGET:
    L_max  = max(data.keys())
    batch  = [L_max + DL*(i+1) for i in range(BATCH_SIZE)
              if L_max + DL*(i+1) <= L_TARGET]
    if not batch:
        break

    log(f"\n=== Iteração {iteration} | L_max={L_max} → batch {batch} ===")

    # 1. Ajusta FSS em todos os dados disponíveis
    log(f"  Ajustando FSS com {len(data)} L's...")
    model = FSSModel(n_terms=FSS_TERMS).fit(data)

    # Diagnóstico de resíduo no maior L real
    res = model.residuals({L_max: data[L_max]})
    log(f"  Resíduo FSS em L={L_max}: "
        f"RMSE={res[L_max]['rmse']:.3e}, max={res[L_max]['maxerr']:.3e}")

    wl_results = []
    for L_new in batch:
        log(f"\n  → L={L_new}")

        # 2. Prediz DOS via FSS
        E_pred, lng_pred = model.predict_dos(L_new)
        save_fss_dat(E_pred, lng_pred, L_new)

        if DRY_RUN:
            # Sem WL: usa estimativa FSS diretamente
            data[L_new] = (E_pred, lng_pred)
            log(f"    [dry-run] DOS FSS salva, WL pulado")
            wl_results.append(L_new)
            continue

        # 3. Salva warm-start e roda WL
        ws_path = OUT_DIR / f"warmstart_L{L_new}.txt"
        write_warmstart(ws_path, E_pred, lng_pred, L_new)

        dat_path = run_wl(L_new, ws_path)

        # 4. Carrega resultado do WL
        if dat_path.exists():
            data[L_new] = load_dat(dat_path)
            wl_results.append(L_new)
            log(f"    Carregado: {len(data[L_new][0])} energias")
        else:
            log(f"    AVISO: {dat_path} não encontrado — usando estimativa FSS")
            data[L_new] = (E_pred, lng_pred)
            wl_results.append(L_new)

    # 5. Plot do data-collapse após o batch
    plot_step(data, wl_results, iteration)
    log(f"  → collapse_iter{iteration:03d}.png salvo")

    iteration += 1

log("\n" + "="*60)
log(f"Loop concluído. L's simulados: {sorted(data.keys())}")
log(f"Arquivos em: {OUT_DIR}")
