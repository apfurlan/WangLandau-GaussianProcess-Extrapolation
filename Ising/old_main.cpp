#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <algorithm>
#include <fstream>

// My functions and classes

#include "nnlist.hpp"        // generate adjacency matrix       (func)
#include "init.hpp"          // set initial configuration       (func)
#include "print_lattice.hpp" // print lattice configuration     (subr)
#include "mt.hpp"            // Mersenne-Twister Random Number  (class)

int main(int argc, char *argv[])
{

  // Scalar variables and their meaning

  int ii;     // sites will be their positions exchanged
  int oindex; // energy index before old-index
  int nindex; // energy index after  new-index
  int nsii;
  double newEn, iran; // energy level after  changing
  double oldEn;       // energy level before changing
  long int minH;      // minimum value of histogram
  long int maxH;      // maximum value of histogram
  long int mcsteps;   // Monte-Carlo steps

  bool flat; // if histogram is flat returns true

  // Two-dimensional array and their meaning
  int **nn; // adjacency matrix

  // Classes - Mersenne Twister random number
  MersenneTwister mt; // Random number class

  // output files
  char fname[50];
  FILE *out_file;

  clock_t tStart, tEnd;
  tStart = clock(); // computing the time of simulation

  if (argc < 2) {
    fprintf(stderr, "Uso: %s <L>\n", argv[0]);
    return 1;
  }
  int L = atoi(argv[1]); // linear size
  // unsigned int Npart=atoi(argv[2]);  // number of particles

  int ndim = 2;   // number of dimension
  int ncoord = 4; // coordination number (square lattice)

  double f = 2.7182818;  // refinement factor
  double minf = 1.0e-8;  // minimum value of f
  double dec_pow = .5;   // power of decreasing f .5 = square root
  double flatness = .80; // threshold of flat criteria

  int flatcheck = 10000; // steps between each check
  // unsigned int nstates=2;            // number of states permited to a particle
  int vol = pow((double)L, (int)ndim); // volume of system

  minf += 1.;
  double lnfmin = log(minf);
  double lnf = log(f);

  int E0 = -ncoord * vol; // energia mínima (double-counted, todos alinhados)
  int Em =  ncoord * vol; // energia máxima (double-counted, todos anti-alinhados)
  int Enbins = Em - E0 + 1;

  int *s = new int[vol];           // state of particles, s0,s1,..,s(vol)
  double *En = new double[Enbins]; // array of energies E0,E1,...,Em
  long int *H = new long int[Enbins];
  double *lng = new double[Enbins];

  double *mag = new double[Enbins];
  double *mag2 = new double[Enbins];
  double *mag4 = new double[Enbins];

  for (int i = 0; i < Enbins; i++)
    En[i] = E0 + i;

  nn = new int *[vol];
  for (int i = 0; i < vol; i++)
    nn[i] = new int[ncoord];

  nn = nnlist(L, ndim, ncoord, vol); // Adjacency matrix

  for (int i = 0; i < Enbins; i++)
  {
    H[i] = 0;
    lng[i] = 1;
  }

  //=============================================================================
  //==================== At this point, the simulation is started ===============
  //=============================================================================
  int count = 0;
  int nnsum = 0;

  int newlevels = 0;
  int oldlevels = 0;

  double flatcomp = 0.;

  while (lnf > lnfmin)
  {

    mt.init_genrand(time(NULL));

    for (int i = 0; i < vol; i++)
      s[i] = (mt.random() < 0.5) ? 1 : -1;

    oldEn = 0;
    for (int i = 0; i < vol; i++)
      for (int j = 0; j < ncoord; j++)
        oldEn -= s[i] * s[nn[i][j]]; // H = -J Σ sᵢsⱼ (double-counted)

    for (int i = 0; i < Enbins; i++)
    {
      H[i] = 0;
      mag[i] = 0.;
      mag2[i] = 0.;
      mag4[i] = 0.;
    }

    flat = false;
    mcsteps = 0;

    do
    {

      for (int k = 0; k < vol; k++)
      {

        ii = (int)(vol * mt.random());
        if (ii >= vol)
          ii = vol - 1;

        int sii = s[ii];
        nsii = -sii; // Ising: flip spin

        int nnsum = 0;
        for (int j = 0; j < ncoord; j++)
          nnsum += s[nn[ii][j]];

        // ΔH_double = 2*(sii - nsii)*nnsum = 4*sii*nnsum  (nsii = -sii)
        newEn = oldEn + 4 * sii * nnsum;

        oindex = abs(E0) + oldEn;
        nindex = abs(E0) + newEn;
        double prob = exp(lng[oindex] - lng[nindex]);

        if (prob >= 1. || mt.random() < prob)
        {

          oldEn = newEn;
          oindex = nindex;

          s[ii] = nsii;
        }

        lng[oindex] += lnf;
        H[oindex] += 1;

        mcsteps++;

        if (count == 26)
        {
          double imag = 0.;

          for (int i = 0; i < vol; i++)
            imag += s[i];

          imag = abs((double)imag);

          mag[oindex] += imag;
          mag2[oindex] += imag * imag;
          mag4[oindex] += imag * imag * imag * imag;
        }

        //===============================================================================
        //=========================== Check the flatness ================================
        //===============================================================================
        if (mcsteps % flatcheck == 0)
        {

          double avgH = 0.;
          newlevels = 0;
          for (int i = 0; i < Enbins; i++)
          {
            if (H[i] > 0)
            {
              avgH += H[i];
              newlevels += 1;
            }
          }

          avgH = avgH / (double)newlevels;

          if (newlevels >= oldlevels)
          {

            minH = (long int)pow(10, 10);
            maxH = 0;
            for (int i = 1; i < Enbins; i++)
            {
              if (H[i] > 0)
              {
                if (H[i] < minH)
                  minH = H[i];
                if (H[i] > maxH)
                  maxH = H[i];
              }
            }

            flatcomp = max((abs(maxH) - avgH) / avgH, (avgH - abs(minH)) / avgH);

            if (minH > avgH * flatness && maxH < (2. - flatness) * avgH)
            {
              oldlevels = newlevels;
              flat = true;
            }
          }
        }
      }
    } while (!flat);

    printf("%2d\t %.2e\t %8lu\t %.3f\t %.3f\t %4u \n", count,
           lnf, mcsteps, flatness, flatcomp, newlevels);

    for (int i = 1; i < Enbins; i++)
      lng[i] -= lng[0];
    lng[0] = 0.;

    count++;
    lnf *= dec_pow;
  }

  tEnd = clock();

  double tTotal = ((double)(tEnd - tStart)) / CLOCKS_PER_SEC;
  tTotal /= 3600.; // Total time of simulation

  sprintf(fname, "ising_DOS_L_%u.dat", L);
  out_file = fopen(fname, "w");
  fputs("#i \tE(i) \t \tlog[g(E)] \t  H(E) \t \t<M> \t \t<M^2> \t \t<M^4>\n",
        out_file);
  fputs("#", out_file);
  for (int i = 0; i < 100; i++)
    fputs("-", out_file);
  fputs("\n", out_file);
  for (int i = 0; i < Enbins; i++)
  {
    if (H[i] > 0)
    {
      fprintf(out_file,
              "%i \t %6.2lf \t%4.6lf %10li \t \t%4.6lf \t%4.6lf \t%4.6lf \n",
              i, En[i] / 2., lng[i], H[i], mag[i] / (vol * H[i]),
              mag2[i] / (vol * vol * H[i]), mag4[i] / (vol * vol * vol * vol * H[i]));
    }
  }

  if (out_file != stdout)
    fclose(out_file);

  delete[] s;
  delete[] H;

  return 0;
}
