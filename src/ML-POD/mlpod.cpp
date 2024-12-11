/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https:
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Ngoc Cuong Nguyen (MIT) and Andrew Rohskopf (SNL)
------------------------------------------------------------------------- */

#include <cmath>

#include "comm.h"
#include "error.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "mlpod.h"
#include "rbpod.h"
#include "tokenizer.h"

using namespace LAMMPS_NS;
using MathConst::MY_PI;
using MathSpecial::cube;
using MathSpecial::powint;

#define MAXLINE 1024

MLPOD::podstruct::podstruct()
    : twobody{4, 8, 6},
      threebody{4, 8, 5, 4},
      fourbody{0, 0, 0, 0},
      pbc(nullptr),
      elemindex(nullptr),
      quadratic22{0, 0},
      quadratic23{0, 0},
      quadratic24{0, 0},
      quadratic33{0, 0},
      quadratic34{0, 0},
      quadratic44{0, 0},
      cubic234{0, 0, 0},
      cubic333{0, 0, 0},
      cubic444{0, 0, 0},
      besselparams(nullptr),
      coeff(nullptr),
      Phi2(nullptr),
      Phi3(nullptr),
      Phi4(nullptr),
      Lambda2(nullptr),
      Lambda3(nullptr),
      Lambda4(nullptr),
      snapelementradius{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
      snapelementweight{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0} {
  snaptwojmax = 0;
  snapchemflag = 0;
  snaprfac0 = 0.99363;
}

MLPOD::podstruct::~podstruct() {
  delete[] pbc;
  delete[] elemindex;
  delete[] besselparams;
}

MLPOD::MLPOD(LAMMPS *_lmp, const std::string &pod_file,
             const std::string &coeff_file)
    : Pointers(_lmp) {
  nClusters = 1;
  nComponents = 1;

  read_pod(pod_file);

  rbpodptr = new RBPOD(lmp, pod_file);

  if (coeff_file != "") read_coeff_file(coeff_file);
}

MLPOD::~MLPOD() {
  memory->destroy(pod.coeff);
  if (pod.ns2 > 0) {
    memory->destroy(pod.Phi2);
    memory->destroy(pod.Lambda2);
  }
  if (pod.ns3 > 0) {
    memory->destroy(pod.Phi3);
    memory->destroy(pod.Lambda3);
  }
  if (pod.ns4 > 0) {
    memory->destroy(pod.Phi4);
    memory->destroy(pod.Lambda4);
  }
}

void MLPOD::podMatMul(double *c, double *a, double *b, int r1, int c1, int c2) {
  int i, j, k;

  for (j = 0; j < c2; j++)
    for (i = 0; i < r1; i++) c[i + r1 * j] = 0.0;

  for (j = 0; j < c2; j++)
    for (i = 0; i < r1; i++)
      for (k = 0; k < c1; k++) c[i + r1 * j] += a[i + r1 * k] * b[k + c1 * j];
}

void MLPOD::podArrayFill(int *output, int start, int length) {
  for (int j = 0; j < length; ++j) output[j] = start + j;
}

void MLPOD::podArraySetValue(double *y, double a, int n) {
  for (int i = 0; i < n; i++) y[i] = a;
}

void MLPOD::podArrayCopy(double *y, double *x, int n) {
  for (int i = 0; i < n; i++) y[i] = x[i];
}

void podsnapshots(double *rbf, double *xij, double *besselparams, double rin,
                  double rcut, int besseldegree, int inversedegree,
                  int nbesselpars, int N) {
  double rmax = rcut - rin;
  for (int n = 0; n < N; n++) {
    double dij = xij[n];

    double r = dij - rin;
    double y = r / rmax;
    double y2 = y * y;
    double y3 = 1.0 - y2 * y;
    double y4 = y3 * y3 + 1e-6;
    double y5 = sqrt(y4);
    double y6 = exp(-1.0 / y5);
    double fcut = y6 / exp(-1.0);

    for (int j = 0; j < nbesselpars; j++) {
      double alpha = besselparams[j];
      if (fabs(alpha) <= 1.0e-6) alpha = 1e-3;
      double x = (1.0 - exp(-alpha * r / rmax)) / (1.0 - exp(-alpha));

      for (int i = 0; i < besseldegree; i++) {
        double a = (i + 1) * MY_PI;
        double b = (sqrt(2.0 / (rmax)) / (i + 1));
        int nij = n + N * i + N * besseldegree * j;
        rbf[nij] = b * fcut * sin(a * x) / r;
      }
    }

    for (int i = 0; i < inversedegree; i++) {
      int p = besseldegree * nbesselpars + i;
      int nij = n + N * p;
      double a = powint(dij, i + 1);
      rbf[nij] = fcut / a;
    }
  }
}

void MLPOD::read_pod(const std::string &pod_file) {
  pod.nbesselpars = 3;
  delete[] pod.besselparams;
  pod.besselparams = new double[3];
  delete[] pod.pbc;
  pod.pbc = new int[3];

  pod.besselparams[0] = 1e-3;
  pod.besselparams[1] = 2.0;
  pod.besselparams[2] = 4.0;

  pod.nelements = 0;
  pod.onebody = 1;
  pod.besseldegree = 3;
  pod.inversedegree = 6;
  pod.quadraticpod = 0;
  pod.rin = 0.5;
  pod.rcut = 4.6;

  pod.snaptwojmax = 0;
  pod.snapchemflag = 0;
  pod.snaprfac0 = 0.99363;

  std::string podfilename = pod_file;
  FILE *fppod;
  if (comm->me == 0) {
    fppod = utils::open_potential(podfilename, lmp, nullptr);
    if (fppod == nullptr)
      error->one(FLERR, "Cannot open POD coefficient file {}: ", podfilename,
                 utils::getsyserror());
  }

  char line[MAXLINE], *ptr;
  int eof = 0;
  while (true) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fppod);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fppod);
      }
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) break;
    MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);

    std::vector<std::string> words;
    try {
      words = Tokenizer(utils::trim_comment(line), "\"' \t\n\r\f").as_vector();
    } catch (TokenizerException &) {
    }

    if (words.size() == 0) continue;

    auto keywd = words[0];

    if (keywd == "species") {
      pod.nelements = words.size() - 1;
      for (int ielem = 1; ielem <= pod.nelements; ielem++) {
        pod.species.push_back(words[ielem]);
      }
    }

    if (keywd == "pbc") {
      if (words.size() != 4)
        error->one(FLERR, "Improper POD file.", utils::getsyserror());
      pod.pbc[0] = utils::inumeric(FLERR, words[1], false, lmp);
      pod.pbc[1] = utils::inumeric(FLERR, words[2], false, lmp);
      pod.pbc[2] = utils::inumeric(FLERR, words[3], false, lmp);
    }

    if ((keywd != "#") && (keywd != "species") && (keywd != "pbc")) {
      if (words.size() != 2)
        error->one(FLERR, "Improper POD file.", utils::getsyserror());

      if (keywd == "rin") pod.rin = utils::numeric(FLERR, words[1], false, lmp);
      if (keywd == "rcut")
        pod.rcut = utils::numeric(FLERR, words[1], false, lmp);
      if (keywd == "number_of_environment_clusters")
        nClusters = utils::inumeric(FLERR, words[1], false, lmp);
      if (keywd == "number_of_principal_components")
        nComponents = utils::inumeric(FLERR, words[1], false, lmp);
      if (keywd == "bessel_polynomial_degree")
        pod.besseldegree = utils::inumeric(FLERR, words[1], false, lmp);
      if (keywd == "inverse_polynomial_degree")
        pod.inversedegree = utils::inumeric(FLERR, words[1], false, lmp);
      if (keywd == "onebody")
        pod.onebody = utils::inumeric(FLERR, words[1], false, lmp);
      if (keywd == "twobody_number_radial_basis_functions")
        pod.twobody[2] = utils::inumeric(FLERR, words[1], false, lmp);
      if (keywd == "threebody_number_radial_basis_functions")
        pod.threebody[2] = utils::inumeric(FLERR, words[1], false, lmp);
      if (keywd == "threebody_number_angular_basis_functions")
        pod.threebody[3] = utils::inumeric(FLERR, words[1], false, lmp) - 1;
      if (keywd == "fourbody_number_radial_basis_functions")
        pod.fourbody[2] = utils::inumeric(FLERR, words[1], false, lmp);
    }
  }

  pod.twobody[0] = pod.besseldegree;
  pod.twobody[1] = pod.inversedegree;
  pod.threebody[0] = pod.besseldegree;
  pod.threebody[1] = pod.inversedegree;

  pod.ns2 = pod.nbesselpars * pod.twobody[0] + pod.twobody[1];
  pod.ns3 = pod.nbesselpars * pod.threebody[0] + pod.threebody[1];
  pod.ns4 = pod.nbesselpars * pod.fourbody[0] + pod.fourbody[1];

  for (int i = 0; i < pod.nbesselpars; i++)
    if (fabs(pod.besselparams[i]) < 1e-3) pod.besselparams[i] = 1e-3;

  pod.nc2 = pod.nelements * (pod.nelements + 1) / 2;
  pod.nc3 = pod.nelements * pod.nelements * (pod.nelements + 1) / 2;

  if (pod.onebody == 1) {
    pod.nbf1 = 1;
    pod.nd1 = pod.nelements;
  } else {
    pod.nbf1 = 0;
    pod.nd1 = 0;
  }

  pod.nbf2 = pod.twobody[2];
  pod.nd2 = pod.nbf2 * pod.nc2;

  pod.nrbf3 = pod.threebody[2];
  pod.nabf3 = pod.threebody[3];
  //pod.nbf3 = pod.nrbf3 * (1 + pod.nabf3);
  pod.nbf3 = pod.nrbf3 * pod.nrbf3 * (1 + 2*pod.nabf3);
  pod.nd3 = pod.nbf3 * pod.nc3;
  pod.nd4 = 0;

  pod.nd = pod.nd1 + pod.nd2 + pod.nd3 + pod.nd4;
  pod.nd1234 = pod.nd1 + pod.nd2 + pod.nd3 + pod.nd4;

  int nelements = pod.nelements;
  delete[] pod.elemindex;
  pod.elemindex = new int[nelements * nelements];

  int k = 1;
  for (int i = 0; i < nelements; i++) {
    for (int j = i; j < nelements; j++) {
      pod.elemindex[i + nelements * j] = k;
      pod.elemindex[j + nelements * i] = k;
      k += 1;
    }
  }

  if (comm->me == 0) {
    utils::logmesg(
        lmp, "**************** Begin of POD Potentials ****************\n");
    utils::logmesg(lmp, "species: ");
    for (int i = 0; i < pod.nelements; i++)
      utils::logmesg(lmp, "{} ", pod.species[i]);
    utils::logmesg(lmp, "\n");
    utils::logmesg(lmp, "periodic boundary conditions: {} {} {}\n", pod.pbc[0],
                   pod.pbc[1], pod.pbc[2]);
    utils::logmesg(lmp, "inner cut-off radius: {}\n", pod.rin);
    utils::logmesg(lmp, "outer cut-off radius: {}\n", pod.rcut);
    utils::logmesg(lmp, "bessel polynomial degree: {}\n", pod.besseldegree);
    utils::logmesg(lmp, "inverse polynomial degree: {}\n", pod.inversedegree);
    utils::logmesg(lmp, "one-body potential: {}\n", pod.onebody);
    utils::logmesg(lmp, "two-body potential: {} {} {}\n", pod.twobody[0],
                   pod.twobody[1], pod.twobody[2]);
    utils::logmesg(lmp, "three-body potential: {} {} {} {}\n", pod.threebody[0],
                   pod.threebody[1], pod.threebody[2], pod.threebody[3] + 1);
    utils::logmesg(lmp, "number of basis functions for one-body potential: {}\n",
                   pod.nbf1);
    utils::logmesg(lmp, "number of basis functions for two-body potential: {}\n",
                   pod.nbf2);
    utils::logmesg(lmp, "number of basis functions for three-body potential: {}\n",
                   pod.nbf3);
    utils::logmesg(lmp, "number of descriptors for one-body potential: {}\n",
                   pod.nd1);
    utils::logmesg(lmp, "number of descriptors for two-body potential: {}\n",
                   pod.nd2);
    utils::logmesg(lmp, "number of descriptors for three-body potential: {}\n",
                   pod.nd3);
    utils::logmesg(lmp, "total number of descriptors for all potentials: {}\n",
                   pod.nd);
    utils::logmesg(
        lmp, "**************** End of POD Potentials ****************\n\n");
  }
}

void MLPOD::read_coeff_file(const std::string &coeff_file) {
  std::string coefffilename = coeff_file;
  FILE *fpcoeff;
  if (comm->me == 0) {
    fpcoeff = utils::open_potential(coefffilename, lmp, nullptr);
    if (fpcoeff == nullptr)
      error->one(FLERR, "Cannot open POD coefficient file {}: ", coefffilename,
                 utils::getsyserror());
  }

  char line[MAXLINE], *ptr;
  int eof = 0;
  int nwords = 0;
  while (nwords == 0) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fpcoeff);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fpcoeff);
      }
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) break;
    MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);

    nwords = utils::count_words(utils::trim_comment(line));
  }

  if (nwords != 2)
    error->all(FLERR, "Incorrect format in POD coefficient file");

  int ncoeffall;
  std::string tmp_str;
  try {
    ValueTokenizer words(utils::trim_comment(line), "\"' \t\n\r\f");
    tmp_str = words.next_string();
    ncoeffall = words.next_int();
  } catch (TokenizerException &e) {
    error->all(FLERR, "Incorrect format in POD coefficient file: {}", e.what());
  }

  memory->create(pod.coeff, ncoeffall, "pod:pod_coeff");

  for (int icoeff = 0; icoeff < ncoeffall; icoeff++) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fpcoeff);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fpcoeff);
      }
    }

    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) error->all(FLERR, "Incorrect format in POD coefficient file");
    MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);

    try {
      ValueTokenizer coeff(utils::trim_comment(line));
      if (coeff.count() != 1)
        error->all(FLERR, "Incorrect format in POD coefficient file");

      pod.coeff[icoeff] = coeff.next_double();
    } catch (TokenizerException &e) {
      error->all(FLERR, "Incorrect format in POD coefficient file: {}",
                 e.what());
    }
  }
  if (comm->me == 0) {
    if (!eof) fclose(fpcoeff);
  }
}

void MLPOD::linear_descriptors(double *gd, double *efatom, double *y,
                               double *tmpmem, int *atomtype, int *alist,
                               int *pairlist, int *, int *pairnumsum,
                               int *tmpint, int natom, int Nij) {
  int dim = 3;
  int nelements = pod.nelements;
  int nbesselpars = pod.nbesselpars;
  int nrbf2 = pod.nbf2;
  int nabf3 = pod.nabf3;
  int nrbf3 = pod.nrbf3;
  int nd1 = pod.nd1;
  int nd2 = pod.nd2;
  int nd3 = pod.nd3;
  int nd4 = pod.nd4;
  int nd1234 = nd1 + nd2 + nd3 + nd4;
  int *pdegree2 = pod.twobody;
  int *elemindex = pod.elemindex;
  double rin = pod.rin;
  double rcut = pod.rcut;
  double *Phi2 = pod.Phi2;
  double *besselparams = pod.besselparams;

  double *fatom1 = &efatom[0];
  double *fatom2 = &efatom[dim * natom * (nd1)];
  double *fatom3 = &efatom[dim * natom * (nd1 + nd2)];
  double *fatom4 = &efatom[dim * natom * (nd1 + nd2 + nd3)];
  double *eatom1 = &efatom[dim * natom * (nd1 + nd2 + nd3 + nd4)];
  double *eatom2 = &efatom[dim * natom * (nd1 + nd2 + nd3 + nd4) + natom * nd1];
  double *eatom3 = &efatom[dim * natom * (nd1 + nd2 + nd3 + nd4) + natom * (nd1 + nd2)];
  double *eatom4 = &efatom[dim * natom * (nd1 + nd2 + nd3 + nd4) + natom * (nd1 + nd2 + nd3)];

  podArraySetValue(fatom1, 0.0, (1 + dim) * natom * (nd1 + nd2 + nd3 + nd4));

  double *rij = &tmpmem[0];
  int *ai = &tmpint[0];
  int *aj = &tmpint[Nij];
  int *ti = &tmpint[2 * Nij];
  int *tj = &tmpint[3 * Nij];
  podNeighPairs(rij, y, ai, aj, ti, tj, pairlist, pairnumsum, atomtype, alist,
                natom, dim);

  poddesc(eatom1, fatom1, eatom2, fatom2, eatom3, fatom3, rij, Phi2,
          besselparams, &tmpmem[3 * Nij], rin, rcut, pairnumsum, atomtype, ai,
          aj, ti, tj, elemindex, pdegree2, nbesselpars, nrbf2, nrbf3, nabf3,
          nelements, Nij, natom);

  podArraySetValue(tmpmem, 1.0, natom);

  char cht = 'T';
  double one = 1.0, zero = 0.0;
  int inc1 = 1;
  DGEMV(&cht, &natom, &nd1234, &one, eatom1, &natom, tmpmem, &inc1, &zero, gd,
        &inc1);
}

void MLPOD::podNeighPairs(double *xij, double *x, int *ai, int *aj, int *ti,
                          int *tj, int *pairlist, int *pairnumsum,
                          int *atomtype, int *alist, int inum, int dim) {
  for (int ii = 0; ii < inum; ii++) {
    int i = ii;
    int itype = atomtype[i];
    int start = pairnumsum[ii];
    int m = pairnumsum[ii + 1] - start;
    for (int l = 0; l < m; l++) {
      int j = pairlist[l + start];
      int k = start + l;
      ai[k] = i;
      aj[k] = alist[j];
      ti[k] = itype;
      tj[k] = atomtype[alist[j]];
      for (int d = 0; d < dim; d++)
        xij[k * dim + d] = x[j * dim + d] - x[i * dim + d];
    }
  }
};

void MLPOD::podtally2b(double *eatom, double *fatom, double *eij, double *fij,
                       int *ai, int *aj, int *ti, int *tj, int *elemindex,
                       int nelements, int nbf, int natom, int N) {
  int nelements2 = nelements * (nelements + 1) / 2;
  for (int n = 0; n < N; n++) {
    int i1 = ai[n];
    int j1 = aj[n];
    int typei = ti[n] - 1;
    int typej = tj[n] - 1;
    for (int m = 0; m < nbf; m++) {
      int im = i1 + natom * ((elemindex[typei + typej * nelements] - 1) +
                             nelements2 * m);
      int jm = j1 + natom * ((elemindex[typei + typej * nelements] - 1) +
                             nelements2 * m);
      int nm = n + N * m;
      eatom[im] += eij[nm];
      fatom[0 + 3 * im] += fij[0 + 3 * nm];
      fatom[1 + 3 * im] += fij[1 + 3 * nm];
      fatom[2 + 3 * im] += fij[2 + 3 * nm];
      fatom[0 + 3 * jm] -= fij[0 + 3 * nm];
      fatom[1 + 3 * jm] -= fij[1 + 3 * nm];
      fatom[2 + 3 * jm] -= fij[2 + 3 * nm];
    }
  }
}

void MLPOD::pod1body(double *eatom, double *fatom, int *atomtype, int nelements,
                     int natom) {
  for (int m = 1; m <= nelements; m++)
    for (int i = 0; i < natom; i++)
      eatom[i + natom * (m - 1)] = (atomtype[i] == m) ? 1.0 : 0.0;

  for (int i = 0; i < 3 * natom * nelements; i++) fatom[i] = 0.0;
}

void MLPOD::pod3body(double *eatom, double *fatom, double *yij, double *e2ij,
                     double *f2ij, double *tmpmem, int *elemindex,
                     int *pairnumsum, int *ai, int *aj, int *ti, int *tj,
                     int nrbf, int nabf, int nelements, int natom, int Nij) {
  int dim = 3;
  int nelements2 = nelements * (nelements + 1) / 2;
  int n, nijk, nijk3, typei, typej, typek, ij, ik, i, j, k;

  double xij1, xij2, xij3, xik1, xik2, xik3;
  double xdot, rijsq, riksq, rij, rik;
  double costhe, sinthe, theta, dtheta;
  double tm, tm1, tm2, dct1, dct2, dct3, dct4, dct5, dct6;
  double uj, uk, rbf, drbf1, drbf2, drbf3, drbf4, drbf5, drbf6;
  double eijk, fj1, fj2, fj3, fk1, fk2, fk3;
  
  int nabf1 = nabf + 1;
  int nabf2 = 2*nabf + 1;
  double *abf = &tmpmem[0];
  double *dabf1 = &tmpmem[nabf2];
  double *dabf2 = &tmpmem[2 * nabf2];
  double *dabf3 = &tmpmem[3 * nabf2];
  double *dabf4 = &tmpmem[4 * nabf2];
  double *dabf5 = &tmpmem[5 * nabf2];
  double *dabf6 = &tmpmem[6 * nabf2];

  for (int ii = 0; ii < natom; ii++) {
    int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
    int s = pairnumsum[ii];
    for (int lj = 0; lj < numneigh; lj++) {
      ij = lj + s;
      i = ai[ij];
      j = aj[ij];
      typei = ti[ij] - 1;
      typej = tj[ij] - 1;
      xij1 = yij[0 + dim * ij];
      xij2 = yij[1 + dim * ij];
      xij3 = yij[2 + dim * ij];
      rijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
      rij = sqrt(rijsq);
      for (int lk = lj + 1; lk < numneigh; lk++) {
        ik = lk + s;
        k = aj[ik];
        typek = tj[ik] - 1;
        xik1 = yij[0 + dim * ik];
        xik2 = yij[1 + dim * ik];
        xik3 = yij[2 + dim * ik];
        riksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
        rik = sqrt(riksq);

        xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
        costhe = xdot / (rij * rik);
        costhe = costhe > 1.0 ? 1.0 : costhe;
        costhe = costhe < -1.0 ? -1.0 : costhe;
        xdot = costhe * (rij * rik);

        sinthe = sqrt(1.0 - costhe * costhe);
        sinthe = sinthe > 1e-12 ? sinthe : 1e-12;
        theta = acos(costhe);
        dtheta = -1.0 / sinthe;

        tm1 = 1.0 / (rij * rijsq * rik);
        tm2 = 1.0 / (rij * riksq * rik);
        dct1 = (xik1 * rijsq - xij1 * xdot) * tm1;
        dct2 = (xik2 * rijsq - xij2 * xdot) * tm1;
        dct3 = (xik3 * rijsq - xij3 * xdot) * tm1;
        dct4 = (xij1 * riksq - xik1 * xdot) * tm2;
        dct5 = (xij2 * riksq - xik2 * xdot) * tm2;
        dct6 = (xij3 * riksq - xik3 * xdot) * tm2;

        for (int p = 0; p < nabf1; p++) {
          abf[p] = cos(p * theta);
          tm = -p * sin(p * theta) * dtheta;
          dabf1[p] = tm * dct1;
          dabf2[p] = tm * dct2;
          dabf3[p] = tm * dct3;
          dabf4[p] = tm * dct4;
          dabf5[p] = tm * dct5;
          dabf6[p] = tm * dct6;
        }

        for (int p = 1; p < nabf1; p++) {
          int np = nabf+p;
          abf[np] = sin(p * theta);
          tm = p * cos(p * theta) * dtheta;
          dabf1[np] = tm * dct1;
          dabf2[np] = tm * dct2;
          dabf3[np] = tm * dct3;
          dabf4[np] = tm * dct4;
          dabf5[np] = tm * dct5;
          dabf6[np] = tm * dct6;
        }
        
        for (int m = 0; m < nrbf; m++) 
        for (int q = 0; q < nrbf; q++) {
          uj = e2ij[lj + s + Nij * m];
          uk = e2ij[lk + s + Nij * q];
          rbf = uj * uk;
          drbf1 = f2ij[0 + dim * (lj + s) + dim * Nij * m] * uk;
          drbf2 = f2ij[1 + dim * (lj + s) + dim * Nij * m] * uk;
          drbf3 = f2ij[2 + dim * (lj + s) + dim * Nij * m] * uk;
          drbf4 = f2ij[0 + dim * (lk + s) + dim * Nij * q] * uj;
          drbf5 = f2ij[1 + dim * (lk + s) + dim * Nij * q] * uj;
          drbf6 = f2ij[2 + dim * (lk + s) + dim * Nij * q] * uj;

          for (int p = 0; p < nabf2; p++) {
            eijk = rbf * abf[p];
            fj1 = drbf1 * abf[p] + rbf * dabf1[p];
            fj2 = drbf2 * abf[p] + rbf * dabf2[p];
            fj3 = drbf3 * abf[p] + rbf * dabf3[p];
            fk1 = drbf4 * abf[p] + rbf * dabf4[p];
            fk2 = drbf5 * abf[p] + rbf * dabf5[p];
            fk3 = drbf6 * abf[p] + rbf * dabf6[p];

            n = p + (nabf2)*q + nabf2*nrbf*m;
            nijk = natom * ((elemindex[typej + typek * nelements] - 1) +
                            nelements2 * typei + nelements2 * nelements * n);
            eatom[i + nijk] += eijk;

            nijk3 = 3 * i + 3 * nijk;
            fatom[0 + nijk3] += fj1 + fk1;
            fatom[1 + nijk3] += fj2 + fk2;
            fatom[2 + nijk3] += fj3 + fk3;

            nijk3 = 3 * j + 3 * nijk;
            fatom[0 + nijk3] -= fj1;
            fatom[1 + nijk3] -= fj2;
            fatom[2 + nijk3] -= fj3;

            nijk3 = 3 * k + 3 * nijk;
            fatom[0 + nijk3] -= fk1;
            fatom[1 + nijk3] -= fk2;
            fatom[2 + nijk3] -= fk3;
          }
        }
      }
    }
  }
}

void MLPOD::poddesc(double *eatom1, double *fatom1, double *eatom2,
                    double *fatom2, double *eatom3, double *fatom3, double *rij,
                    double *Phi, double *besselparams, double *tmpmem,
                    double rin, double rcut, int *pairnumsum, int *atomtype,
                    int *ai, int *aj, int *ti, int *tj, int *elemindex,
                    int *pdegree, int nbesselpars, int nrbf2, int nrbf3,
                    int nabf, int nelements, int Nij, int natom) {
  int nrbf = MAX(nrbf2, nrbf3);
  int ns = pdegree[0] * nbesselpars + pdegree[1];

  double *e2ij = &tmpmem[0];
  double *f2ij = &tmpmem[Nij * nrbf];
  double *e2ijt = &tmpmem[4 * Nij * nrbf];
  double *f2ijt = &tmpmem[4 * Nij * nrbf + Nij * ns];

  rbpodptr->femradialbasis(e2ij, f2ij, rij, Nij);

  pod1body(eatom1, fatom1, atomtype, nelements, natom);

  podtally2b(eatom2, fatom2, e2ij, f2ij, ai, aj, ti, tj, elemindex, nelements,
             nrbf2, natom, Nij);

  pod3body(eatom3, fatom3, rij, e2ij, f2ij, &tmpmem[4 * Nij * nrbf], elemindex,
           pairnumsum, ai, aj, ti, tj, nrbf3, nabf, nelements, natom, Nij);
}

void MLPOD::podNeighPairs(double *rij, double *x, int *idxi, int *ai, int *aj,
                          int *ti, int *tj, int *pairnumsum, int *atomtype,
                          int *jlist, int *alist, int inum) {
  for (int ii = 0; ii < inum; ii++) {
    int gi = ii;
    int itype = atomtype[gi];
    int start = pairnumsum[ii];
    int m = pairnumsum[ii + 1] - start;
    for (int l = 0; l < m; l++) {
      int k = start + l;
      int gj = jlist[k];
      idxi[k] = ii;
      ai[k] = alist[gi];
      aj[k] = alist[gj];
      ti[k] = itype;
      tj[k] = atomtype[aj[k]];
      rij[k * 3 + 0] = x[gj * 3 + 0] - x[gi * 3 + 0];
      rij[k * 3 + 1] = x[gj * 3 + 1] - x[gi * 3 + 1];
      rij[k * 3 + 2] = x[gj * 3 + 2] - x[gi * 3 + 2];
    }
  }
};

int MLPOD::lammpsNeighPairs(double *rij, double **x, double rcutsq, int *idxi,
                            int *ai, int *aj, int *ti, int *tj, int *pairnumsum,
                            int *atomtype, int *numneigh, int *ilist,
                            int **jlist, int inum) {
  int ninside = 0;
  for (int ii = 0; ii < inum; ii++) {
    int gi = ilist[ii];
    int itype = atomtype[gi];
    int m = numneigh[gi];
    pairnumsum[ii + 1] = 0;
    for (int l = 0; l < m; l++) {
      int gj = jlist[gi][l];
      double delx = x[gj][0] - x[gi][0];
      double dely = x[gj][1] - x[gi][1];
      double delz = x[gj][2] - x[gi][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < rcutsq && rsq > 1e-20) {
        rij[ninside * 3 + 0] = delx;
        rij[ninside * 3 + 1] = dely;
        rij[ninside * 3 + 2] = delz;
        idxi[ninside] = ii;
        ai[ninside] = gi;
        aj[ninside] = gj;
        ti[ninside] = itype;
        tj[ninside] = atomtype[gj];
        ninside++;
        pairnumsum[ii + 1] += 1;
      }
    }
  }

  pairnumsum[0] = 0;
  for (int ii = 0; ii < inum; ii++)
    pairnumsum[ii + 1] = pairnumsum[ii + 1] + pairnumsum[ii];

  return ninside;
};

void MLPOD::podtally2b(double *eatom, double *eij, int *idxi, int *ti, int *tj,
                       int *elemindex, int nelements, int nbf, int natom,
                       int N) {
  int nelements2 = nelements * (nelements + 1) / 2;
  for (int n = 0; n < N; n++) {
    int i1 = idxi[n];
    int typei = ti[n] - 1;
    int typej = tj[n] - 1;
    for (int m = 0; m < nbf; m++) {
      int im = i1 + natom * ((elemindex[typei + typej * nelements] - 1) +
                             nelements2 * m);
      int nm = n + N * m;
      eatom[im] += eij[nm];
    }
  }
}

void MLPOD::pod1body(double *eatom, int *atomtype, int nelements, int natom) {
  for (int m = 1; m <= nelements; m++)
    for (int i = 0; i < natom; i++)
      eatom[i + natom * (m - 1)] = (atomtype[i] == m) ? 1.0 : 0.0;
}

void MLPOD::pod3body(double *eatom, double *yij, double *e2ij, double *tmpmem,
                     int *elemindex, int *pairnumsum, int *, int *ti, int *tj,
                     int nrbf, int nabf, int nelements, int natom, int Nij) {
  int dim = 3;
  int nabf1 = nabf + 1;
  int nabf2 = 2*nabf + 1;
  int nelements2 = nelements * (nelements + 1) / 2;
  int n, nijk, typei, typej, typek, ij, ik;

  double xij1, xij2, xij3, xik1, xik2, xik3;
  double xdot, rijsq, riksq, rij, rik;
  double costhe, theta;
  double uj, uk, rbf;

  double *abf = &tmpmem[0];
  double *etm = &tmpmem[nabf2];

  for (int ii = 0; ii < natom; ii++) {
    int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
    int s = pairnumsum[ii];

    for (int m = 0; m < pod.nd3; m++)
      etm[m] = 0.0;

    for (int lj = 0; lj < numneigh; lj++) {
      ij = lj + s;
      typei = ti[ij] - 1;
      typej = tj[ij] - 1;
      xij1 = yij[0 + dim * ij];
      xij2 = yij[1 + dim * ij];
      xij3 = yij[2 + dim * ij];
      rijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
      rij = sqrt(rijsq);
      for (int lk = lj + 1; lk < numneigh; lk++) {
        ik = lk + s;
        typek = tj[ik] - 1;
        xik1 = yij[0 + dim * ik];
        xik2 = yij[1 + dim * ik];
        xik3 = yij[2 + dim * ik];
        riksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
        rik = sqrt(riksq);

        xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
        costhe = xdot / (rij * rik);
        costhe = costhe > 1.0 ? 1.0 : costhe;
        costhe = costhe < -1.0 ? -1.0 : costhe;
        theta = acos(costhe);

        for (int p = 0; p < nabf1; p++) abf[p] = cos(p * theta);
        for (int p = 1; p < nabf1; p++) abf[nabf + p] = sin(p * theta);
                  
        for (int m = 0; m < nrbf; m++) 
        for (int q = 0; q < nrbf; q++) {
          uj = e2ij[lj + s + Nij * m];
          uk = e2ij[lk + s + Nij * q];
          rbf = uj * uk;
          for (int p = 0; p < nabf2; p++) {
            n = p + (nabf2)*q + nabf2*nrbf*m;
            nijk = (elemindex[typej + typek * nelements] - 1) +
                   nelements2 * typei + nelements2 * nelements * n;
            etm[nijk] += rbf * abf[p];
          }
        }
      }
    }
    for (int m = 0; m < pod.nd3; m++)
      eatom[ii + natom * m] += etm[m];
  }
}

void MLPOD::poddesc_ij(double *eatom1, double *eatom2, double *eatom3,
                       double *rij, double *Phi, double *besselparams,
                       double *tmpmem, double rin, double rcut, int *pairnumsum,
                       int *atomtype, int *idxi, int *ti, int *tj,
                       int *elemindex, int *pdegree, int nbesselpars, int nrbf2,
                       int nrbf3, int nabf, int nelements, int Nij, int natom) {
  int nrbf = MAX(nrbf2, nrbf3);
  int ns = pdegree[0] * nbesselpars + pdegree[1];

  double *e2ij = &tmpmem[0];
  double *e2ijt = &tmpmem[Nij * nrbf];

  rbpodptr->femradialbasis(e2ij, rij, Nij);
  
  pod1body(eatom1, atomtype, nelements, natom);

  podtally2b(eatom2, e2ij, idxi, ti, tj, elemindex, nelements, nrbf2, natom,
             Nij);

  pod3body(eatom3, rij, e2ij, &tmpmem[Nij * nrbf], elemindex, pairnumsum, idxi,
           ti, tj, nrbf3, nabf, nelements, natom, Nij);
}

void MLPOD::linear_descriptors_ij(double *gd, double *eatom, double *rij,
                                  double *tmpmem, int *pairnumsum,
                                  int *atomtype, int *idxi, int *ti, int *tj,
                                  int natom, int Nij) {
  int nelements = pod.nelements;
  int nbesselpars = pod.nbesselpars;
  int nrbf2 = pod.nbf2;
  int nabf3 = pod.nabf3;
  int nrbf3 = pod.nrbf3;
  int nd1 = pod.nd1;
  int nd2 = pod.nd2;
  int nd3 = pod.nd3;
  int nd4 = pod.nd4;
  int nd1234 = nd1 + nd2 + nd3 + nd4;
  int *pdegree2 = pod.twobody;
  int *elemindex = pod.elemindex;
  double rin = pod.rin;
  double rcut = pod.rcut;
  double *Phi2 = pod.Phi2;
  double *besselparams = pod.besselparams;

  double *eatom1 = &eatom[0];
  double *eatom2 = &eatom[0 + natom * nd1];
  double *eatom3 = &eatom[0 + natom * (nd1 + nd2)];
  double *eatom4 = &eatom[0 + natom * (nd1 + nd2 + nd3)];

  podArraySetValue(eatom1, 0.0, natom * nd1234);

  poddesc_ij(eatom1, eatom2, eatom3, rij, Phi2, besselparams, tmpmem, rin, rcut,
             pairnumsum, atomtype, idxi, ti, tj, elemindex, pdegree2,
             nbesselpars, nrbf2, nrbf3, nabf3, nelements, Nij, natom);

  podArraySetValue(tmpmem, 1.0, natom);

  char cht = 'T';
  double one = 1.0;
  int inc1 = 1;
  DGEMV(&cht, &natom, &nd1234, &one, eatom, &natom, tmpmem, &inc1, &one, gd, &inc1);
}

double MLPOD::calculate_energy(double *effectivecoeff, double *gd, double *coeff) {
  double energy = 0.0;
  for (int i = 0; i < pod.nd; i++) {
    effectivecoeff[i] = coeff[i];
    energy += coeff[i] * gd[i];
  }

  return energy;
}

void MLPOD::pod2body_force(double *force, double *fij, double *coeff2, int *ai,
                           int *aj, int *ti, int *tj, int *elemindex,
                           int nelements, int nbf, int, int N) {
  int nelements2 = nelements * (nelements + 1) / 2;
  for (int n = 0; n < N; n++) {
    int i1 = ai[n];
    int j1 = aj[n];
    int typei = ti[n] - 1;
    int typej = tj[n] - 1;
    for (int m = 0; m < nbf; m++) {
      int im = 3 * i1;
      int jm = 3 * j1;
      int nm = n + N * m;
      int km = (elemindex[typei + typej * nelements] - 1) + nelements2 * m;
      double ce = coeff2[km];
      force[0 + im] += fij[0 + 3 * nm] * ce;
      force[1 + im] += fij[1 + 3 * nm] * ce;
      force[2 + im] += fij[2 + 3 * nm] * ce;
      force[0 + jm] -= fij[0 + 3 * nm] * ce;
      force[1 + jm] -= fij[1 + 3 * nm] * ce;
      force[2 + jm] -= fij[2 + 3 * nm] * ce;
    }
  }
}

void MLPOD::pod3body_force(double *force, double *yij, double *e2ij,
                           double *f2ij, double *coeff3, double *tmpmem,
                           int *elemindex, int *pairnumsum, int *ai, int *aj,
                           int *ti, int *tj, int nrbf, int nabf, int nelements,
                           int natom, int Nij) {
  int dim = 3;
  int nelements2 = nelements * (nelements + 1) / 2;
  int n, c, nijk3, typei, typej, typek, ij, ik, i, j, k;

  double xij1, xij2, xij3, xik1, xik2, xik3;
  double xdot, rijsq, riksq, rij, rik;
  double costhe, sinthe, theta, dtheta;
  double tm, tm1, tm2, dct1, dct2, dct3, dct4, dct5, dct6;

  int nabf1 = nabf + 1;
  int nabf2 = 2*nabf + 1;
  double *abf = &tmpmem[0];
  double *dabf1 = &tmpmem[nabf2];
  double *dabf2 = &tmpmem[2 * nabf2];
  double *dabf3 = &tmpmem[3 * nabf2];
  double *dabf4 = &tmpmem[4 * nabf2];
  double *dabf5 = &tmpmem[5 * nabf2];
  double *dabf6 = &tmpmem[6 * nabf2];
  
  for (int ii = 0; ii < natom; ii++) {
    int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
    int s = pairnumsum[ii];
    for (int lj = 0; lj < numneigh; lj++) {
      ij = lj + s;
      i = ai[ij];
      j = aj[ij];
      typei = ti[ij] - 1;
      typej = tj[ij] - 1;
      xij1 = yij[0 + dim * ij];
      xij2 = yij[1 + dim * ij];
      xij3 = yij[2 + dim * ij];
      rijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
      rij = sqrt(rijsq);

      double fixtmp, fiytmp, fiztmp;
      fixtmp = fiytmp = fiztmp = 0.0;
      double fjxtmp, fjytmp, fjztmp;
      fjxtmp = fjytmp = fjztmp = 0.0;
      for (int lk = lj + 1; lk < numneigh; lk++) {
        ik = lk + s;
        k = aj[ik];
        typek = tj[ik] - 1;
        xik1 = yij[0 + dim * ik];
        xik2 = yij[1 + dim * ik];
        xik3 = yij[2 + dim * ik];
        riksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
        rik = sqrt(riksq);

        xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
        costhe = xdot / (rij * rik);
        costhe = costhe > 1.0 ? 1.0 : costhe;
        costhe = costhe < -1.0 ? -1.0 : costhe;
        xdot = costhe * (rij * rik);

        sinthe = sqrt(1.0 - costhe * costhe);
        sinthe = sinthe > 1e-12 ? sinthe : 1e-12;
        theta = acos(costhe);
        dtheta = -1.0 / sinthe;

        tm1 = 1.0 / (rij * rijsq * rik);
        tm2 = 1.0 / (rij * riksq * rik);
        dct1 = (xik1 * rijsq - xij1 * xdot) * tm1;
        dct2 = (xik2 * rijsq - xij2 * xdot) * tm1;
        dct3 = (xik3 * rijsq - xij3 * xdot) * tm1;
        dct4 = (xij1 * riksq - xik1 * xdot) * tm2;
        dct5 = (xij2 * riksq - xik2 * xdot) * tm2;
        dct6 = (xij3 * riksq - xik3 * xdot) * tm2;

        for (int p = 0; p < nabf1; p++) {
          abf[p] = cos(p * theta);
          tm = -p * sin(p * theta) * dtheta;
          dabf1[p] = tm * dct1;
          dabf2[p] = tm * dct2;
          dabf3[p] = tm * dct3;
          dabf4[p] = tm * dct4;
          dabf5[p] = tm * dct5;
          dabf6[p] = tm * dct6;
        }
        
        for (int p = 1; p < nabf1; p++) {
          int np = nabf+p;
          abf[np] = sin(p * theta);
          tm = p * cos(p * theta) * dtheta;
          dabf1[np] = tm * dct1;
          dabf2[np] = tm * dct2;
          dabf3[np] = tm * dct3;
          dabf4[np] = tm * dct4;
          dabf5[np] = tm * dct5;
          dabf6[np] = tm * dct6;
        }
        
        double fjx = 0.0, fjy = 0.0, fjz = 0.0;
        double fkx = 0.0, fky = 0.0, fkz = 0.0;

        for (int m = 0; m < nrbf; m++) 
        for (int q = 0; q < nrbf; q++) {
          double uj = e2ij[lj + s + Nij * m];
          double uk = e2ij[lk + s + Nij * q];
          double rbf = uj * uk;
          double drbf1 = f2ij[0 + dim * (lj + s) + dim * Nij * m] * uk;
          double drbf2 = f2ij[1 + dim * (lj + s) + dim * Nij * m] * uk;
          double drbf3 = f2ij[2 + dim * (lj + s) + dim * Nij * m] * uk;
          double drbf4 = f2ij[0 + dim * (lk + s) + dim * Nij * q] * uj;
          double drbf5 = f2ij[1 + dim * (lk + s) + dim * Nij * q] * uj;
          double drbf6 = f2ij[2 + dim * (lk + s) + dim * Nij * q] * uj;

          for (int p = 0; p < nabf2; p++) {
            tm = abf[p];
            double fj1 = drbf1 * tm + rbf * dabf1[p];
            double fj2 = drbf2 * tm + rbf * dabf2[p];
            double fj3 = drbf3 * tm + rbf * dabf3[p];
            double fk1 = drbf4 * tm + rbf * dabf4[p];
            double fk2 = drbf5 * tm + rbf * dabf5[p];
            double fk3 = drbf6 * tm + rbf * dabf6[p];

            n = p + (nabf2)*q + nabf2*nrbf*m;
            c = (elemindex[typej + typek * nelements] - 1) +
                nelements2 * typei + nelements2 * nelements * n;
            tm = coeff3[c];

            fjx += fj1 * tm;
            fjy += fj2 * tm;
            fjz += fj3 * tm;
            fkx += fk1 * tm;
            fky += fk2 * tm;
            fkz += fk3 * tm;
          }
        }
        nijk3 = 3 * k;
        force[0 + nijk3] -= fkx;
        force[1 + nijk3] -= fky;
        force[2 + nijk3] -= fkz;
        fjxtmp += fjx;
        fjytmp += fjy;
        fjztmp += fjz;
        fixtmp += fjx + fkx;
        fiytmp += fjy + fky;
        fiztmp += fjz + fkz;
      }
      nijk3 = 3 * j;
      force[0 + nijk3] -= fjxtmp;
      force[1 + nijk3] -= fjytmp;
      force[2 + nijk3] -= fjztmp;
      nijk3 = 3 * i;
      force[0 + nijk3] += fixtmp;
      force[1 + nijk3] += fiytmp;
      force[2 + nijk3] += fiztmp;
    }
  }
}

void MLPOD::calculate_force(double *force, double *effectivecoeff, double *rij,
                            double *tmpmem, int *pairnumsum, int *atomtype,
                            int *idxi, int *ai, int *aj, int *ti, int *tj,
                            int natom, int Nij) {
  int nelements = pod.nelements;
  int nbesselpars = pod.nbesselpars;
  int nrbf2 = pod.nbf2;
  int nabf3 = pod.nabf3;
  int nrbf3 = pod.nrbf3;
  int nd1 = pod.nd1;
  int nd2 = pod.nd2;
  int nd3 = pod.nd3;
  int *pdegree = pod.twobody;
  int *elemindex = pod.elemindex;
  double rin = pod.rin;
  double rcut = pod.rcut;
  double *Phi = pod.Phi2;
  double *besselparams = pod.besselparams;

  double *coeff2 = &effectivecoeff[nd1];
  double *coeff3 = &effectivecoeff[nd1 + nd2];
  double *coeff4 = &effectivecoeff[nd1 + nd2 + nd3];

  int nrbf = MAX(nrbf2, nrbf3);
  int ns = pdegree[0] * nbesselpars + pdegree[1];
  double *e2ij = &tmpmem[0];
  double *f2ij = &tmpmem[Nij * nrbf];
  double *e2ijt = &tmpmem[4 * Nij * nrbf];
  double *f2ijt = &tmpmem[4 * Nij * nrbf + Nij * ns];

  rbpodptr->femradialbasis(e2ij, f2ij, rij, Nij);
  
  pod2body_force(force, f2ij, coeff2, ai, aj, ti, tj, elemindex, nelements,
                 nrbf2, natom, Nij);

  pod3body_force(force, rij, e2ij, f2ij, coeff3, &tmpmem[4 * Nij * nrbf],
                 elemindex, pairnumsum, ai, aj, ti, tj, nrbf3, nabf3, nelements,
                 natom, Nij);
}

double MLPOD::energyforce_calculation(double *force, double *podcoeff,
                                      double *effectivecoeff, double *gd,
                                      double *rij, double *tmpmem,
                                      int *pairnumsum, int *atomtype, int *idxi,
                                      int *ai, int *aj, int *ti, int *tj,
                                      int natom, int Nij) {
  double *eatom = &tmpmem[0];
  podArraySetValue(gd, 0.0, pod.nd);
  linear_descriptors_ij(gd, eatom, rij, &tmpmem[natom * pod.nd], pairnumsum,
                        atomtype, idxi, ti, tj, natom, Nij);

  double energy = calculate_energy(effectivecoeff, gd, podcoeff);

  podArraySetValue(force, 0.0, 3 * natom);

  calculate_force(force, effectivecoeff, rij, tmpmem, pairnumsum, atomtype,
                  idxi, ai, aj, ti, tj, natom, Nij);

  return energy;
}
