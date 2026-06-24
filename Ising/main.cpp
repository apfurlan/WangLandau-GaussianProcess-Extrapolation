/*
 * =============================================================================
 * Wang-Landau para o Modelo de Ising 2D em Rede Quadrada
 * =============================================================================
 *
 * DESCRIÇÃO
 * ---------
 * Implementa o algoritmo de Wang-Landau (WL) para estimar a densidade de
 * estados g(E) — ou equivalentemente ln g(E) — do modelo de Ising 2D com
 * spins s_i = ±1 em rede quadrada L×L com condições de contorno periódicas
 * (PBC) e acoplamento ferromagnético J = 1.
 *
 * A Hamiltoniana é:
 *
 *     H = -J Σ_{<i,j>} s_i s_j       (soma sobre pares de primeiros vizinhos)
 *
 * ALGORITMO DE WANG-LANDAU
 * ------------------------
 * O WL constrói iterativamente uma estimativa de ln g(E) de modo que a
 * probabilidade de aceitar um flip do spin ii de sii para -sii seja:
 *
 *     P(E_old → E_new) = min(1,  g(E_old) / g(E_new))
 *                      = min(1,  exp(lng[b_old] - lng[b_new]))
 *
 * A cada passo aceito ou rejeitado, atualiza-se:
 *
 *     lng[b_current] += ln f        (modificador de refinamento)
 *     H[b_current]   += 1           (histograma de visitas)
 *
 * Quando o histograma H[b] é suficientemente plano sobre todos os níveis
 * acessíveis (critério: min H > flatness * <H> e max H < (2-flatness)*<H>),
 * o modificador é reduzido:
 *
 *     ln f ← ln f × dec_pow
 *
 * O processo repete até ln f < lnfmin (convergência).
 *
 * INDEXAÇÃO DE ENERGIA
 * --------------------
 * A energia double-counted E_d = -Σ_i Σ_{j∈nn(i)} s_i s_j é usada
 * internamente como inteiro, com intervalo [-4L², +4L²] e passo mínimo 4.
 * Isso permite um índice comprimido:
 *
 *     b = (E_d - E0_d) / 4,     E0_d = -4L²
 *     Número de bins: Nbins = 2L² + 1
 *
 * A energia física (não double-counted) impressa no arquivo é E_d / 2.
 * A coluna de índice i no arquivo obedece i = 4*b (compatível com o
 * formato padrão de saída do simulador).
 *
 * OBSERVÁVEIS
 * -----------
 * Na última iteração WL (lnf mais refinado), acumula-se:
 *
 *     <|M|>    = Σ |Σ_i s_i|           / (vol × H[b])
 *     <M²>     = Σ (Σ_i s_i)²          / (vol² × H[b])
 *     <M⁴>     = Σ (Σ_i s_i)⁴         / (vol⁴ × H[b])
 *
 * A magnetização total é mantida incrementalmente a custo O(1) por flip:
 * ao flipar o spin sii, atualiza-se total_mag += -2 * sii.
 *
 * PARÂMETROS DO ALGORITMO
 * -----------------------
 *   lnf         : modificador inicial = 1.0  (corresponde a f = e)
 *   lnfmin      : critério de convergência = 1e-8
 *   dec_pow     : fator de redução de lnf = 0.5  (raiz quadrada a cada iteração)
 *   flatness    : limiar de planicidade = 0.80  (80%)
 *   flatcheck_sweeps : número de varreduras entre checagens de planicidade
 *                      = max(10000, vol)
 *
 * COMPILAÇÃO
 * ----------
 *   g++ -O3 -march=native -std=c++14 -o wl main.cpp mt.cpp nnlist.cpp \
 *       init.cpp print_lattice.cpp
 *
 * USO
 *   ./wl <L>
 *
 *   Exemplo: ./wl 16
 *
 * SAÍDA
 * -----
 * Arquivo texto: ising_DOS_L_<L>.dat
 * Colunas (tab-separadas):
 *   i       – índice de energia (4*b)
 *   E(i)    – energia física = E_d / 2
 *   lng(E)  – estimativa de ln g(E), normalizada com lng(E_min) = 0
 *   H(E)    – número de visitas ao nível E na última iteração
 *   <|M|>   – magnetização média por sítio (módulo)
 *   <M²>    – momento quadrático da magnetização por sítio²
 *   <M⁴>    – momento de quarta ordem da magnetização por sítio⁴
 *
 * REFERÊNCIAS
 * -----------
 *   Wang, F. & Landau, D. P. (2001). Phys. Rev. Lett. 86, 2050.
 *   Wang, F. & Landau, D. P. (2001). Phys. Rev. E 64, 056101.
 * =============================================================================
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <vector>

#include "nnlist.hpp"
#include "mt.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <L>\n", argv[0]);
        return 1;
    }

    // ── Parâmetros do sistema ─────────────────────────────────────────────
    const int L      = atoi(argv[1]);
    const int ndim   = 2;
    const int ncoord = 4;
    const int vol    = L * L;

    // ── Parâmetros do Wang-Landau ─────────────────────────────────────────
    double lnf                  = 1.0;
    const double lnfmin         = 1.0e-8;
    const double dec_pow        = 0.5;
    const double flatness       = 0.80;
    const int flatcheck_sweeps  = std::max(10000, vol);

    // ── Energias e indexação comprimida ───────────────────────────────────
    const int E0_d  = -ncoord * vol;   // -4*vol
    const int Nbins =  2 * vol + 1;

    // ── Matriz de adjacência — array flat (contíguo na memória) ──────────
    // Layout: nn_flat[ii * ncoord + j] = índice do j-ésimo vizinho de ii
    // Reduz cache misses em relação ao int** (dois deferencionamentos por acesso)
    std::vector<int> nn_flat(vol * ncoord);
    {
        int **nn_tmp = nnlist(L, ndim, ncoord, vol);
        for (int i = 0; i < vol; i++)
            for (int j = 0; j < ncoord; j++)
                nn_flat[i * ncoord + j] = nn_tmp[i][j];
        for (int i = 0; i < vol; i++) delete[] nn_tmp[i];
        delete[] nn_tmp;
    }
    const int *nn = nn_flat.data();   // ponteiro cru para o loop quente

    // ── Arrays principais ─────────────────────────────────────────────────
    std::vector<long>   H   (Nbins, 0);
    std::vector<double> lng (Nbins, 1.0);
    std::vector<double> mag (Nbins, 0.0);
    std::vector<double> mag2(Nbins, 0.0);
    std::vector<double> mag4(Nbins, 0.0);

    // Ponteiros crus para o loop quente (elimina overhead de bounds check em debug)
    long   *pH   = H.data();
    double *plng = lng.data();
    double *pmag = mag.data();
    double *pmag2= mag2.data();
    double *pmag4= mag4.data();

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

        // Energia inicial (double-counted, inteira)
        int En_d = 0;
        for (int i = 0; i < vol; i++)
            for (int j = 0; j < ncoord; j++)
                En_d -= s[i] * s[nn[i * ncoord + j]];
        int obin = std::max(0, std::min(Nbins - 1, (En_d - E0_d) / 4));

        // Magnetização total (tracking incremental O(1) por flip)
        int total_mag = 0;
        for (int i = 0; i < vol; i++) total_mag += s[i];

        // Reset do histograma e observáveis
        std::fill(H.begin(),    H.end(),    0L);
        std::fill(mag.begin(),  mag.end(),  0.0);
        std::fill(mag2.begin(), mag2.end(), 0.0);
        std::fill(mag4.begin(), mag4.end(), 0.0);

        bool flat      = false;
        long mcsteps   = 0;
        long sweeps    = 0;
        int  newlevels = 0;
        double flatcomp = 0.0;

        const bool collect_obs = (lnf * dec_pow <= lnfmin);

        // ── Loop de Monte Carlo ───────────────────────────────────────────
        do {
            // Uma varredura: vol tentativas de flip
            for (int k = 0; k < vol; k++)
            {
                int ii = static_cast<int>(vol * mt.random());
                if (ii >= vol) ii = vol - 1;

                const int sii  = s[ii];
                const int base = ii * ncoord;

                int nnsum = s[nn[base    ]]
                          + s[nn[base + 1]]
                          + s[nn[base + 2]]
                          + s[nn[base + 3]];

                const int delta_bin = sii * nnsum;   // ΔE_d/4
                const int nbin      = obin + delta_bin;

                if (plng[nbin] <= plng[obin] || mt.random() < exp(plng[obin] - plng[nbin]))
                {
                    s[ii]      = -sii;
                    total_mag -= 2 * sii;
                    obin       = nbin;
                }

                plng[obin] += lnf;
                pH[obin]   += 1;
                mcsteps++;

                if (collect_obs) {
                    const double m = std::abs(static_cast<double>(total_mag));
                    pmag [obin] += m;
                    pmag2[obin] += m * m;
                    pmag4[obin] += m * m * m * m;
                }
            }
            sweeps++;

            // Checagem de planicidade (uma vez por flatcheck_sweeps varreduras)
            if (sweeps % flatcheck_sweeps == 0)
            {
                double avgH = 0.0;
                newlevels   = 0;
                long minH   = 1L << 50;
                long maxH   = 0;

                for (int b = 0; b < Nbins; b++) {
                    if (pH[b] > 0) {
                        avgH += pH[b];
                        newlevels++;
                        if (pH[b] < minH) minH = pH[b];
                        if (pH[b] > maxH) maxH = pH[b];
                    }
                }
                avgH /= newlevels;

                if (newlevels >= oldlevels) {
                    flatcomp = std::max((maxH - avgH) / avgH,
                                       (avgH - minH) / avgH);
                    if (minH > avgH * flatness && maxH < (2.0 - flatness) * avgH) {
                        oldlevels = newlevels;
                        flat = true;
                    }
                }
            }

        } while (!flat);

        // Normaliza lng[bin_E_min] = 0
        const double lng0 = plng[0];
        for (int b = 0; b < Nbins; b++) plng[b] -= lng0;

        printf("%2d\t %.2e\t %8ld\t %.3f\t %.3f\t %4d\n",
               count, lnf, mcsteps, flatness, flatcomp, newlevels);
        count++;
        lnf *= dec_pow;
    }

    const double tTotal = static_cast<double>(clock() - tStart) / CLOCKS_PER_SEC / 3600.0;

    // ── Saída ─────────────────────────────────────────────────────────────
    char fname[64];
    sprintf(fname, "ising_DOS_L_%d.dat", L);
    FILE *out_file = fopen(fname, "w");
    fputs("#i \tE(i) \t \tlog[g(E)] \t  H(E) \t \t<M> \t \t<M^2> \t \t<M^4>\n", out_file);
    fputs("#", out_file);
    for (int i = 0; i < 100; i++) fputs("-", out_file);
    fputs("\n", out_file);

    const double vol2 = static_cast<double>(vol) * vol;
    const double vol4 = vol2 * vol2;

    for (int b = 0; b < Nbins; b++) {
        if (pH[b] > 0) {
            const int    i_out = 4 * b;
            const double E_out = (E0_d + 4 * b) / 2.0;
            const double Hb    = static_cast<double>(pH[b]);
            fprintf(out_file,
                    "%i \t %6.2f \t%4.6f %10ld \t \t%4.6f \t%4.6f \t%4.6f \n",
                    i_out, E_out, plng[b], pH[b],
                    pmag [b] / (vol  * Hb),
                    pmag2[b] / (vol2 * Hb),
                    pmag4[b] / (vol4 * Hb));
        }
    }
    fclose(out_file);

    printf("\nTempo total: %.4f h\n", tTotal);
    return 0;
}
