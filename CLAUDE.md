# CLAUDE.md — WangLandau-GaussianProcess-Extrapolation

Este arquivo é lido automaticamente pelo Claude Code ao trabalhar neste repositório.
Contém o contexto completo do projeto, decisões de design e instruções de uso.

---

## Visão geral do projeto

Estimativa da densidade de estados (DOS) do **modelo de Ising 2D** em redes
quadradas de tamanho arbitrário, combinando:

1. **Simulação Wang-Landau (WL)** em C++ para L pequenos e médios
2. **Extrapolação por Gaussian Process (GP)** para L's próximos do maior simulado
3. **Extrapolação por Finite-Size Scaling (FSS)** + loop de bootstrapping
   para L muito grandes (L=128 até L=4096+)

O objetivo final é gerar um chute inicial de ln g(E) para acelerar simulações
WL em L's grandes — o WL com warm-start pula as iterações grosseiras e
começa diretamente na fase de refinamento, reduzindo o tempo de simulação
significativamente.

---

## Estrutura do repositório

```
.
├── Ising/                    ← código C++ do Wang-Landau
│   ├── main.cpp              ← WL padrão (versão otimizada)
│   ├── wl_warmstart.cpp      ← WL com warm-start e lnf inicial configurável
│   ├── mt.cpp / mt.hpp       ← Mersenne Twister RNG
│   ├── nnlist.cpp / nnlist.hpp  ← matriz de adjacência (vizinhos)
│   ├── init.cpp / init.hpp   ← inicialização de configuração
│   ├── print_lattice.cpp / print_lattice.hpp
│   └── run.sh                ← script para rodar múltiplos L's em paralelo
│
├── data/                     ← arquivos de saída do WL: ising_DOS_L_<L>.dat
│   └── logs/                 ← logs das simulações
│
├── gp/                       ← pipeline de extrapolação por GP
│   ├── run_real_data.py      ← script principal (detecta L's automaticamente)
│   ├── data_utils.py         ← carregamento, variáveis reduzidas, subamostragem
│   ├── gp_model.py           ← DOSExtrapolator (GP com kernel Matérn)
│   ├── wl_postprocess.py     ← simetria, regra de soma, escrita de arquivo
│   └── requirements.txt
│
├── bootstrap/                ← pipeline de bootstrapping FSS+WL
│   ├── fss_model.py          ← FSSModel: ajuste e predição FSS
│   └── bootstrap_loop.py     ← loop principal de bootstrapping
│
├── data_process/
│   └── exploratory.ipynb
│
└── README.md
```

---

## Formato dos arquivos de dados

**Arquivo:** `data/ising_DOS_L_<L>.dat`

```
#i    E(i)       log[g(E)]    H(E)       <M>        <M^2>      <M^4>
#----...
0     -128.00    0.000000     10912      1.000000   1.000000   1.000000
16    -120.00    4.142179     11541      0.968750   0.938477   0.880738
...
```

- Coluna `i`: índice de energia — fórmula: `i = 2*(E_double - E_min_double)` = `4*b` onde `b` é o bin comprimido
- Coluna `E(i)`: energia física (não double-counted) = `E_double / 2`
- Coluna `log[g(E)]`: ln g(E), normalizado com `lng(E_min) = 0`
- `H(E)`, `<M>`, `<M^2>`, `<M^4>`: histograma e observáveis magnéticos
- Arquivos `*_predicted.dat` e `*_fss.dat`: estimativas (não simulações reais) — ignorar em treino

---

## Física do modelo

**Hamiltoniana:** `H = -J Σ_{<i,j>} s_i s_j`, com `J=1`, `s_i = ±1`

**Energia double-counted** (usada internamente no WL):
- `E_d ∈ [-4L², +4L²]`, passo mínimo 4 (na prática 8 para estados acessíveis)
- Índice comprimido: `b = (E_d - E0_d) / 4`, `Nbins = 2L² + 1`
- A rede quadrada 2D com PBC é bipartida: correções FSS em potências de `1/L²`

**Variáveis reduzidas** (usadas no GP e FSS):
- `e = E / L²` (energia por sítio)
- `s = ln g(E,L) / L²` (entropia por sítio)
- Data collapse: `s(e, L) → s∞(e)` quando `L → ∞`

**Restrições exatas conhecidas:**
- Simetria: `g(E) = g(-E)` (invariância sob flip global de spins)
- Estado fundamental: `g(E_min) = g(E_max) = 2` (degenerescência ferromagnética)
- Regra de soma: `Σ_E g(E) = 2^(L²)` (número total de microestados)

---

## Algoritmo Wang-Landau (implementação C++)

### Parâmetros
| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `lnf` inicial | 1.0 | Modificador de refinamento (f = e) |
| `lnfmin` | 1e-8 | Critério de convergência |
| `dec_pow` | 0.5 | lnf → lnf/2 a cada iteração (27 iterações total) |
| `flatness` | 0.80 | Histograma plano quando min H > 80% da média |
| `flatcheck_sweeps` | max(10000, vol) | Varreduras entre checagens |

### Otimizações implementadas (vs. versão original)
1. **Indexação comprimida**: `Nbins = 2L²+1` em vez de `8L²+1` → 4× menos memória, melhor cache
2. **Array flat para vizinhos**: `nn[i*4+j]` em vez de `nn[i][j]` → sem double pointer dereference
3. **Magnetização incremental**: `O(1)` por flip em vez de `O(L²)`
4. **Checagem de planicidade fora do loop**: uma vez por N varreduras, não por flip
5. **Energia como `int`**: eliminadas conversões float desnecessárias
6. **RNG semeado uma vez**: evita seeds repetidas entre iterações

### Compilação
```bash
cd Ising/
# WL padrão
g++ -O3 -march=native -std=c++14 -o wl \
    main.cpp mt.cpp nnlist.cpp init.cpp print_lattice.cpp

# WL com warm-start
g++ -O3 -march=native -std=c++14 -o wl_warmstart \
    wl_warmstart.cpp mt.cpp nnlist.cpp init.cpp print_lattice.cpp
```

### Uso
```bash
./wl 32                                    # WL padrão para L=32
./wl_warmstart 32                          # warm-start sem arquivo (idem ao padrão)
./wl_warmstart 32 1e-3                     # começa de lnf=1e-3 (pula ~20 iterações)
./wl_warmstart 32 1e-3 warmstart_L32.txt  # com warm-start de arquivo
```

### Rodar múltiplos L's em paralelo
```bash
for L in 32 34 36 38; do ./wl $L > logs/ising_L${L}.log 2>&1 & done
wait
```

---

## Pipeline GP (extrapolação para L próximos)

**Quando usar:** L alvo até ~1.5× o maior L simulado.
Com L_max=82, confiável até L≈120; L=128 é o limite.

**Variáveis do modelo:** GP sobre `(e, 1/L)` → prediz `s(e, 1/L_target)`

**Kernel:** Matérn(ν=1.5) anisotrópico — escala diferente para `e` e `1/L`

**Validação:** leave-one-out obrigatória antes de usar qualquer predição.
RMSE < ~1e-3 por sítio é aceitável para warm-start.

### Uso
```bash
cd gp/
pip install -r requirements.txt

python run_real_data.py --data ../data --out ../out
# Detecta automaticamente todos os ising_DOS_L_*.dat em ../data
# Exclui automaticamente *_predicted.dat e *_fss.dat
# Leave-one-out no maior L; extrapola até 1.5×L_max
```

### Outputs gerados
- `out/01_leave_one_out_L<N>.png` — validação
- `out/02_data_collapse.png` — colapso das curvas
- `out/03_extrapolation.png` — curvas extrapoladas com IC 95%
- `out/ln_g_initial_L<N>.txt` — arquivo de warm-start (formato: E  lng)
- `out/ising_DOS_L_<N>_predicted.dat` — formato .dat padrão com zeros em H,M

---

## Pipeline Bootstrap FSS+WL (extrapolação para L grandes)

**Quando usar:** L alvo > 1.5× L_max simulado (L=128 até L=4096+)

**Teoria FSS:**
```
s(e, L) = s∞(e) + a(e)/L² + b(e)/L⁴ + ...
```
Correções em potências de `1/L²` (rede bipartida com PBC simétrica).

**Loop de bootstrapping:**
```
Dados reais L=8..82
    ↓
FSS extrapola → DOS para L=84,86,88,90
    ↓
WL warm-start nesses 4 L's (começa de lnf_start, pula iterações grosseiras)
    ↓
Dados reais L=8..90
    ↓
FSS extrapola → L=92,94,96,98
    ↓
... repete até L_TARGET
```

### Uso
```bash
cd bootstrap/

# Loop completo até L=4096
python bootstrap_loop.py \
    --data    ../data \
    --out     ../out/bootstrap \
    --wl      ../Ising/wl_warmstart \
    --target  4096 \
    --batch   4 \
    --lnf-start 1e-3 \
    --step    2

# Só FSS, sem rodar WL (teste rápido)
python bootstrap_loop.py --data ../data --out ../out/bootstrap --dry-run
```

### Parâmetro `--lnf-start`
| Valor | Iterações puladas | Requer |
|-------|------------------|--------|
| `1e-2` | ~17 de 27 | Warm-start razoável |
| `1e-3` | ~20 de 27 | Warm-start bom (recomendado para FSS) |
| `1e-4` | ~23 de 27 | Warm-start muito bom |
| `1e-5` | ~27 de 27 | Warm-start quase perfeito |

À medida que o loop avança e L's maiores são adicionados ao treino, o FSS
melhora e é possível reduzir `--lnf-start` progressivamente.

---

## Pós-processamento da DOS predita

Toda DOS gerada por GP ou FSS passa por três correções (em `wl_postprocess.py`):

1. **Simetria** `g(E) = g(-E)` — média de `lng(E)` e `lng(-E)`
2. **Regra de soma** `Σ g(E) = 2^(L²)` — shift aditivo global em log-space
3. **Pin do estado fundamental** (opcional, `pin_edges=False` por padrão)

**Por que não forçar o estado fundamental por padrão:**
Um viés de ~1e-4 por sítio no pico da entropia, multiplicado por N=L² sítios,
gera um shift de centenas de unidades em ln g — inconsistente com g(E_min)=2.
Para warm-start de WL isso não importa: a aceitação depende só de razões
g(E_old)/g(E_new), nunca de constante aditiva absoluta.

---

## Artigo de referência

**Landinez Borda, E. J. & Rubenstein, B. M.**
*Gaussian Processes for Finite Size Extrapolation of Many-Body Simulations*
arXiv: 2112.10334 (2021) — Faraday Discussions, 2024. DOI: 10.1039/D4FD00051J

---

## Decisões de design importantes

- **GP vs rede neural:** GP escolhido por funcionar bem com poucos dados (10-40 L's),
  dar incerteza calibrada analiticamente e ser interpretável via kernel.
- **FSS vs GP para L grande:** GP extrapola livremente em `1/L`; FSS usa o
  conhecimento físico da forma funcional `s∞ + a/L² + b/L⁴`, sendo muito mais
  confiável fora do domínio de treino.
- **Bootstrapping em batches de 4:** cada passo é uma extrapolação curta
  (erro mínimo); acumular muitos passos sem refinamento WL aumentaria o erro.
- **`n_terms=3` no FSS:** usa s∞, a/L², b/L⁴. Requer ao menos 3 L's.
  Para batches iniciais com poucos L's, o FSSModel reduz automaticamente.
- **`flatcheck_sweeps = max(10000, vol)`:** evita checagens muito frequentes
  em L grandes (vol = L² pode ser enorme).