/*
 * =============================================================================
 * Wang-Landau com Warm-Start – Modelo de Ising 2D
 * =============================================================================
 *
 * DESCRIÇÃO
 * ---------
 * Variante do Wang-Landau que aceita uma estimativa inicial de ln g(E)
 * (warm-start) e um valor inicial de ln f, permitindo pular as iterações
 * grosseiras e iniciar diretamente na fase de refinamento.
 *
 * Quando a DOS inicial é uma boa aproximação (e.g. estimada por FSS a partir
 * de L's vizinhos), o histograma fica plano muito mais rapidamente nas
 * primeiras iterações, reduzindo o tempo total de simulação.
 *
 * USO
 * ---
 *   ./wl_warmstart <L>
 *   ./wl_warmstart <L> <lnf_start>
 *   ./wl_warmstart <L> <lnf_start> <warmstart_file>
 *
 * Argumentos
 * ----------
 *   L              : tamanho linear da rede
 *   lnf_start      : valor de ln f em que o WL começa (padrão: 1.0)
 *                    Use 1e-3 para pular ~20 iterações grosseiras.
 *   warmstart_file : arquivo texto com duas colunas (E  lng), linhas # ignoradas
 *                    Gerado por bootstrap_loop.py ou wl_postprocess.py.
 *
 * SAÍDA
 * -----
 *   ising_DOS_L_<L>.dat   (mesmo formato do wl padrão)
 *
 * COMPILAÇÃO
 * ----------
 *   g++ -O3 -march=native -std=c++14 \
 *       -o wl_warmstart wl_warmstart.cpp mt.cpp nnlist.cpp \
 *       init.cpp print_lattice.cpp
 *
 * REFERÊNCIAS
 * -----------
 *   Wang & Landau (2001). Phys. Rev. Lett. 86, 2050.
 *   Wang & Landau (2001). Phys. Rev. E 64, 056101.
 * =============================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <vector>

#include "nnlist.hpp"
#include "mt.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <L> [lnf_start] [warmstart_file]\n", argv[0]);
        return 1;
    }

    // ── Parâmetros do sistema ─────────────────────────────────────────────
    const int L      = atoi(argv[1]);
    const int ndim   = 2;
    const int ncoord = 4;
    const int vol    = L * L;

    // ── Parâmetros do Wang-Landau ─────────────────────────────────────────
    double lnf                 = (argc >= 3) ? atof(argv[2]) : 1.0;
    const double lnfmin        = 1.0e-8;
    const double dec_pow       = 0.5;
    const double flatness      = 0.80;
    const int flatcheck_sweeps = std::max(10000, vol);

    const char *warmstart_file = (argc >= 4) ? argv[3] : nullptr;

    // ── Energias e indexação comprimida ───────────────────────────────────
    // b = (E_double - E0_double) / 4,  Nbins = 2*vol + 1
    const int E0_d  = -ncoord * vol;
    const int Nbins =  2 * vol + 1;

    // ── Matriz de adjacência flat ─────────────────────────────────────────
    std::vector<int> nn_flat(vol * ncoord);
    {
        int **nn_tmp = nnlist(L, ndim, ncoord, vol);
        for (int i = 0; i < vol; i++)
            for (int j = 0; j < ncoord; j++)
                nn_flat[i * ncoord + j] = nn_tmp[i][j];
        for (int i = 0; i < vol; i++) delete[] nn_tmp[i];
        delete[] nn_tmp;
    }
    const int *nn = nn_flat.data();

    // ── Arrays ───────────────────────────────────────────────────────────
    std::vector<long>   H   (Nbins, 0);
    std::vector<double> lng (Nbins, 1.0);   // inicializado em 1.0
    std::vector<double> mag (Nbins, 0.0);
    std::vector<double> mag2(Nbins, 0.0);
    std::vector<double> mag4(Nbins, 0.0);

    long   *pH   = H.data();
    double *plng = lng.data();
    double *pmag = mag.data();
    double *pmag2= mag2.data();
    double *pmag4= mag4.data();

    // ── Carrega warm-start (se fornecido) ─────────────────────────────────
    // Formato do arquivo: linhas começando com # são ignoradas;
    // demais linhas têm dois campos:  E_fisico   lng_valor
    int n_loaded = 0;
    if (warmstart_file != nullptr) {
        FILE *ws = fopen(warmstart_file, "r");
        if (ws == nullptr) {
            fprintf(stderr, "Aviso: não foi possível abrir %s — iniciando sem warm-start.\n",
                    warmstart_file);
        } else {
            char line[256];
            while (fgets(line, sizeof(line), ws)) {
                if (line[0] == '#' || line[0] == '\n') continue;
                double E_phys, lng_val;
                if (sscanf(line, "%lf %lf", &E_phys, &lng_val) == 2) {
                    // Converte energia física → bin comprimido
                    int E_d = (int)round(2.0 * E_phys);
                    int b   = (E_d - E0_d) / 4;
                    if (b >= 0 && b < Nbins)
                        plng[b] = lng_val, n_loaded++;
                }
            }
            fclose(ws);
            printf("Warm-start carregado: %d bins de %s\n", n_loaded, warmstart_file);
        }
    }

    printf("L=%d  vol=%d  Nbins=%d  lnf_start=%.2e  warm-start=%s\n",
           L, vol, Nbins, lnf, warmstart_file ? warmstart_file : "nenhum");

    // ── RNG ───────────────────────────────────────────────────────────────
    MersenneTwister mt;
    mt.init_genrand(static_cast<unsigned long>(time(NULL)));

    std::vector<int> s(vol);
    int count     = 0;
    int oldlevels = 0;

    const clock_t tStart = clock();

    // ════════════════════════════════════════════════════════════════════════
    // Loop principal do Wang-Landau
    // ════════════════════════════════════════════════════════════════════════
    while (lnf > lnfmin)
    {
        // Configuração inicial aleatória
        for (int i = 0; i < vol; i++)
            s[i] = (mt.random() < 0.5) ? 1 : -1;

        // Energia inicial
        int En_d = 0;
        for (int i = 0; i < vol; i++)
            for (int j = 0; j < ncoord; j++)
                En_d -= s[i] * s[nn[i * ncoord + j]];
        int obin = std::max(0, std::min(Nbins-1, (En_d - E0_d) / 4));

        // Magnetização incremental
        int total_mag = 0;
        for (int i = 0; i < vol; i++) total_mag += s[i];

        // Reset do histograma (lng NÃO é resetado — acumula entre iterações)
        std::fill(H.begin(),    H.end(),    0L);
        std::fill(mag.begin(),  mag.end(),  0.0);
        std::fill(mag2.begin(), mag2.end(), 0.0);
        std::fill(mag4.begin(), mag4.end(), 0.0);

        bool   flat    = false;
        long   mcsteps = 0, sweeps = 0;
        int    newlevels = 0;
        double flatcomp  = 0.0;

        const bool collect_obs = (lnf * dec_pow <= lnfmin);

        // ── Loop de Monte Carlo ───────────────────────────────────────────
        do {
            for (int k = 0; k < vol; k++)
            {
                int ii = static_cast<int>(vol * mt.random());
                if (ii >= vol) ii = vol - 1;

                const int sii  = s[ii];
                const int base = ii * ncoord;
                const int nnsum = s[nn[base]] + s[nn[base+1]]
                                + s[nn[base+2]] + s[nn[base+3]];

                const int delta_bin = sii * nnsum;
                const int nbin      = obin + delta_bin;

                if (plng[nbin] <= plng[obin] ||
                    mt.random() < exp(plng[obin] - plng[nbin]))
                {
                    s[ii]      = -sii;
                    total_mag -= 2 * sii;
                    obin       = nbin;
                }

                plng[obin] += lnf;
                pH[obin]   += 1;
                mcsteps++;

                if (collect_obs) {
                    const double m = std::abs((double)total_mag);
                    pmag [obin] += m;
                    pmag2[obin] += m * m;
                    pmag4[obin] += m * m * m * m;
                }
            }
            sweeps++;

            // Checagem de planicidade
            if (sweeps % flatcheck_sweeps == 0) {
                double avgH = 0.0;
                newlevels   = 0;
                long minH = 1L<<50, maxH = 0;
                for (int b = 0; b < Nbins; b++) {
                    if (pH[b] > 0) {
                        avgH += pH[b]; newlevels++;
                        if (pH[b] < minH) minH = pH[b];
                        if (pH[b] > maxH) maxH = pH[b];
                    }
                }
                avgH /= newlevels;
                if (newlevels >= oldlevels) {
                    flatcomp = std::max((maxH - avgH)/avgH, (avgH - minH)/avgH);
                    if (minH > avgH*flatness && maxH < (2.0-flatness)*avgH) {
                        oldlevels = newlevels;
                        flat = true;
                    }
                }
            }
        } while (!flat);

        // Normaliza lng[0] = 0
        const double lng0 = plng[0];
        for (int b = 0; b < Nbins; b++) plng[b] -= lng0;

        printf("%2d\t %.2e\t %8ld\t %.3f\t %.3f\t %4d\n",
               count, lnf, mcsteps, flatness, flatcomp, newlevels);
        count++;
        lnf *= dec_pow;
    }

    const double tTotal =
        static_cast<double>(clock()-tStart) / CLOCKS_PER_SEC / 3600.0;

    // ── Saída ─────────────────────────────────────────────────────────────
    char fname[64];
    sprintf(fname, "ising_DOS_L_%d.dat", L);
    FILE *out_file = fopen(fname, "w");
    fputs("#i \tE(i) \t \tlog[g(E)] \t  H(E) \t \t<M> \t \t<M^2> \t \t<M^4>\n",
          out_file);
    fputs("#", out_file);
    for (int i = 0; i < 100; i++) fputs("-", out_file);
    fputs("\n", out_file);

    const double vol2 = (double)vol * vol;
    const double vol4 = vol2 * vol2;
    for (int b = 0; b < Nbins; b++) {
        if (pH[b] > 0) {
            const double Hb = (double)pH[b];
            fprintf(out_file,
                    "%i \t %6.2f \t%4.6f %10ld \t \t%4.6f \t%4.6f \t%4.6f \n",
                    4*b, (E0_d + 4*b)/2.0, plng[b], pH[b],
                    pmag [b]/(vol *Hb),
                    pmag2[b]/(vol2*Hb),
                    pmag4[b]/(vol4*Hb));
        }
    }
    fclose(out_file);

    printf("\nTempo total: %.4f h\n", tTotal);
    return 0;
}
