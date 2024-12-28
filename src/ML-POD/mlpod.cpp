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
      besselparams(nullptr) {
}

MLPOD::podstruct::~podstruct() {
  delete[] pbc;
  delete[] elemindex;
  delete[] besselparams;
}

MLPOD::MLPOD(LAMMPS *_lmp, const std::string &pod_file, const std::string &coeff_file)
    : Pointers(_lmp), podcoeffs(nullptr), femcoeffs(nullptr) {
      
  nClusters = 1;
  nComponents = 1;
  femdegree = 0;
  nelemrbf = 0;
  nelemabf = 0;
  
  read_pod(pod_file);

  rbpodptr = new RBPOD(lmp, pod_file);

  npelem = (femdegree+1)*(femdegree+1)*(femdegree+1);
  nfemelem = nelemabf*nelemrbf*nelemrbf; 
  nfemcoeffs = npelem*4*nfemelem*pod.nc3;
  nfemfuncs = pod.nc3;    
  
  if (coeff_file != "") read_coeff_file(coeff_file);
  npodcoeffs = pod.nd;          
}
    
MLPOD::~MLPOD() {
  memory->destroy(podcoeffs);
  memory->destroy(femcoeffs);  
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
  pod.rin = 0.5;
  pod.rcut = 4.6;

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
  
    if (keywd == "fem_approximation_threebody") {
      if (words.size() != 4)
        error->one(FLERR, "Improper POD file.", utils::getsyserror());
      femdegree = utils::inumeric(FLERR, words[1], false, lmp);
      nelemrbf = utils::inumeric(FLERR, words[2], false, lmp);
      nelemabf = utils::inumeric(FLERR, words[3], false, lmp);
      npelem = (femdegree+1)*(femdegree+1)*(femdegree+1);
      nfemelem = nelemabf*nelemrbf*nelemrbf; 
      nfemcoeffs = npelem*4*nfemelem*pod.nc3;
    }
    
    if ((keywd != "#") && (keywd != "species") && (keywd != "pbc") && (keywd != "fem_approximation_threebody")) {
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
    utils::logmesg(lmp, "**************** Begin of POD Potentials ****************\n");
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

  memory->create(podcoeffs, ncoeffall, "pod:pod_coeff");

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

      podcoeffs[icoeff] = coeff.next_double();
    } catch (TokenizerException &e) {
      error->all(FLERR, "Incorrect format in POD coefficient file: {}",
                 e.what());
    }
  }
  
  if (comm->me == 0) {    
    if (!eof) fclose(fpcoeff);
  }
      
  if ((femdegree>0) && (nelemrbf>0) && (nelemabf>0))
    memory->create(femcoeffs, nfemcoeffs, "mlpod:femcoeffs");
    
  if (comm->me == 0) {              
    if ((femdegree>0) && (nelemrbf>0) && (nelemabf>0)) {
      utils::logmesg(lmp, "**************** Begin of FEM Approximation ****************\n");
      polyfit3body(&podcoeffs[pod.nd1 + pod.nd2], 0);      
      utils::logmesg(lmp, "**************** End of FEM Approximation ****************\n");  
    }
  }      
  
  if ((femdegree>0) && (nelemrbf>0) && (nelemabf>0))
    MPI_Bcast(femcoeffs, nfemcoeffs, MPI_DOUBLE, 0, world);    
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
  //double *Phi2 = pod.Phi2;
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

  poddesc(eatom1, fatom1, eatom2, fatom2, eatom3, fatom3, rij, 
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
                    double *besselparams, double *tmpmem,
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

double MLPOD::pod2body_energyforce(double *fij, double *ei, double *rij, double *rbf, double *drbfdr, 
                    double *coeff2, double *tmpmem, int *elemindex, int *pairnumsum, int *ti, int *tj, 
                    int nelements, int nrbf, int natom, int Nij) {    
  double energy = 0.0;
  int nelements2 = nelements * (nelements + 1) / 2;
  
  for (int ii = 0; ii < natom; ii++) {
    int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
    int s = pairnumsum[ii];
    double en = 0.0;
    for (int lj = 0; lj < numneigh; lj++) {
      int n = lj + s;
      int typei = ti[n] - 1;
      int typej = tj[n] - 1;          
      double fn = 0.0;    
      for (int m = 0; m < nrbf; m++) {      
        int nm = n + Nij * m;
        int km = (elemindex[typei + typej * nelements] - 1) + nelements2 * m;
        double ce = coeff2[km];
        en += ce * rbf[nm];
        fn += ce * drbfdr[nm];      
      }

      double xij1 = rij[0+3*n];
      double xij2 = rij[1+3*n];
      double xij3 = rij[2+3*n];
      double dij = sqrt(xij1 * xij1 + xij2 * xij2 + xij3 * xij3);
      double tn = fn/dij;
      fij[0+3*n] += tn*xij1;
      fij[1+3*n] += tn*xij2;
      fij[2+3*n] += tn*xij3;                      
    }
    ei[ii] += en;
    energy += en;
  }
  
  return energy;
}


double MLPOD::pod3body_energyforce(double *fij, double *ei, double *rij, double *rbf, double *drbfdr, 
                    double *coeff3, double *tmpmem, int *elemindex, int *pairnumsum, int *ti, int *tj, 
                    int nelements, int nrbf, int nabf, int natom, int Nij) {
  int dim = 3;
  int nelements2 = nelements * (nelements + 1) / 2;
  int n, c, typei, typej, typek, ij, ik;

  double xij1, xij2, xij3, xik1, xik2, xik3;
  double xdot, dijsq, diksq, dij, dik;
  double costhe, sinthe, theta, dtheta;
  double tm, tm1, tm2, dct1, dct2, dct3, dct4, dct5, dct6;

  int nabf1 = nabf + 1;
  int nabf2 = 2*nabf + 1;
  double *abf = &tmpmem[0];
  double *dabf = &tmpmem[nabf2];
  
  double energy = 0.0;
  for (int ii = 0; ii < natom; ii++) {
    int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
    int s = pairnumsum[ii];    
    for (int lj = 0; lj < numneigh; lj++) {
      ij = lj + s;
      typei = ti[ij] - 1;
      typej = tj[ij] - 1;
      xij1 = rij[0 + dim * ij];
      xij2 = rij[1 + dim * ij];
      xij3 = rij[2 + dim * ij];
      dijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
      dij = sqrt(dijsq);
      
      double en = 0.0, fjx = 0.0, fjy = 0.0, fjz = 0.0;
      for (int lk = lj + 1; lk < numneigh; lk++) {
        ik = lk + s;
        typek = tj[ik] - 1;
        xik1 = rij[0 + dim * ik];
        xik2 = rij[1 + dim * ik];
        xik3 = rij[2 + dim * ik];
        diksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
        dik = sqrt(diksq);

        xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
        tm = dij * dik;
        costhe = xdot / tm;
        costhe = costhe > 1.0 ? 1.0 : costhe;
        costhe = costhe < -1.0 ? -1.0 : costhe;
        xdot = costhe * tm;
        theta = acos(costhe);

        sinthe = sqrt(1.0 - costhe * costhe);
        sinthe = sinthe > 1e-12 ? sinthe : 1e-12;        
        dtheta = -1.0 / sinthe;

        tm1 = dtheta / (dijsq * tm);
        tm2 = dtheta / (diksq * tm);
        dct1 = (xik1 * dijsq - xij1 * xdot) * tm1;
        dct2 = (xik2 * dijsq - xij2 * xdot) * tm1;
        dct3 = (xik3 * dijsq - xij3 * xdot) * tm1;
        dct4 = (xij1 * diksq - xik1 * xdot) * tm2;
        dct5 = (xij2 * diksq - xik2 * xdot) * tm2;
        dct6 = (xij3 * diksq - xik3 * xdot) * tm2;

        for (int p = 0; p < nabf1; p++) {
          abf[p] = cos(p * theta);
          dabf[p] = -p * sin(p * theta);          
        }
        
        for (int p = 1; p < nabf1; p++) {
          int np = nabf+p;
          abf[np] = sin(p * theta);
          dabf[np] = p * cos(p * theta);
        }
        
        double fn = 0.0, fm = 0.0, fq = 0.0, fp = 0.0;
        for (int m = 0; m < nrbf; m++) {
          for (int q = 0; q < nrbf; q++) {
            double uj = rbf[ij + Nij * m];
            double uk = rbf[ik + Nij * q];
            double ujk = uj * uk;
            double dujkdrj = drbfdr[ij + Nij * m] * uk;
            double dujkdrk = drbfdr[ik + Nij * q] * uj;

            for (int p = 0; p < nabf2; p++) {
              n = p + (nabf2)*q + nabf2*nrbf*m;
              c = (elemindex[typej + typek * nelements] - 1) +
                  nelements2 * typei + nelements2 * nelements * n;            
              tm1 = coeff3[c];            
              tm = abf[p];          

              fn += tm1 * ujk * tm;
              fm += tm1 * dujkdrj * tm;
              fq += tm1 * dujkdrk * tm;
              fp += tm1 * ujk * dabf[p];            
            }
          }
        }
        
        en += fn;
        tm1 = fm/dij;
        tm2 = fq/dik;        
        fjx += (tm1*xij1 + fp * dct1);
        fjy += (tm1*xij2 + fp * dct2);
        fjz += (tm1*xij3 + fp * dct3);                                
        fij[0+3*ik] += (tm2*xik1 + fp * dct4);
        fij[1+3*ik] += (tm2*xik2 + fp * dct5);
        fij[2+3*ik] += (tm2*xik3 + fp * dct6);                                        
      }      
      energy += en;
      ei[ii] += en;
      fij[0+3*ij] += fjx;
      fij[1+3*ij] += fjy;
      fij[2+3*ij] += fjz;                                      
    }        
  }
    
  return energy;
}

double MLPOD::pod123body_energyforce(double *fij, double *ei, double *rij, double *podcoeff, 
                       double *tmpmem, int *pairnumsum, int *typeai, int *ti, int *tj, int natom, int Nij) 
{
  int nelements = pod.nelements;
  int nrbf2 = pod.nbf2;
  int nabf3 = pod.nabf3;
  int nrbf3 = pod.nrbf3;
  int nd1 = pod.nd1;
  int nd2 = pod.nd2;
  int nd3 = pod.nd3;
  int *elemindex = pod.elemindex;

  double *coeff1 = &podcoeff[0];
  double *coeff2 = &podcoeff[nd1];
  double *coeff3 = &podcoeff[nd1 + nd2];

  double *rbf = &tmpmem[0];
  double *drbfdr = &tmpmem[Nij * nrbf2];
  double *tmp = &tmpmem[2 * Nij * nrbf2];
  
  double energy = 0.0;  
  podArraySetValue(fij, 0.0, 3 * Nij);
  
  for (int i = 0; i < natom; i++)
    for (int m = 1; m <= nelements; m++) {   
      double e = coeff1[m-1] * ((typeai[i] == m) ? 1.0 : 0.0);
      ei[i] = e;
      energy += e;
    }
    
  rbpodptr->femdrbfdr(rbf, drbfdr, rij, Nij);
  
  energy += pod2body_energyforce(fij, ei, rij, rbf, drbfdr, coeff2, tmp, elemindex, 
                                 pairnumsum, ti, tj, nelements, nrbf2, natom, Nij);

  energy += pod3body_energyforce(fij, ei, rij, rbf, drbfdr, coeff3, tmp, elemindex, 
                               pairnumsum, ti, tj, nelements, nrbf3, nabf3, natom, Nij);
    
  return energy;
}

void MLPOD::tallyforce(double *force, double *fij, int *ai, int *aj, int N)
{
  for (int n=0; n<N; n++) {
    int im =  ai[n];
    int jm =  aj[n];
    int nm = 3*n;
    force[0 + 3*im] += fij[0 + nm];
    force[1 + 3*im] += fij[1 + nm];
    force[2 + 3*im] += fij[2 + nm];
    force[0 + 3*jm] -= fij[0 + nm];
    force[1 + 3*jm] -= fij[1 + nm];
    force[2 + 3*jm] -= fij[2 + nm];
  }
}


double MLPOD::energyforce_calculation(double *force, double *fij, double *rij, double *podcoeff, double *tmpmem, 
        int *pairnumsum, int *typeai, int *ai, int *aj, int *ti, int *tj, int natom, int Nij) 
{
  
  double energy = 0.0;
    
  double *ei = &tmpmem[0];
  double *tmp = &tmpmem[natom];
  
  energy = pod123body_energyforce(fij, ei, rij, podcoeff, tmp, pairnumsum, typeai, ti, tj, natom, Nij);
  
  podArraySetValue(force, 0.0, 3 * natom);
  tallyforce(force, fij, ai, aj, Nij);
  
  return energy;
}

void MLPOD::femrbf(double *rbf, double *drbfdr, double rin, double rcut, int nrbf, int nelem, int p)
{    
  int n = p + 1;  
  
  double *relem, *xi, *y;
  memory->create(relem, nelem+1, "mlpod:relem");    
  memory->create(xi, n, "mlpod:xi");
  memory->create(y, n, "mlpod:y");  
  
  rbpodptr->xchenodes(xi, p); 
  for (int i=0; i<nelem+1; i++)
    relem[i] = rin+1e-3 + (rcut-rin-1e-3)*(i*1.0/nelem);
  
  for (int i=0; i<nelem; i++) {
    rbpodptr->ref2dom(y, xi, relem[i], relem[i+1], n);          
    rbpodptr->fem1drbf(&rbf[n*nrbf*i], &drbfdr[n*nrbf*i], y, nrbf, n);
  }  
  
  memory->destroy(relem);
  memory->destroy(xi);
  memory->destroy(y);
}

void MLPOD::femabf(double *abf, double *dabf, int nabf, int nelem, int p)
{
  int n = p + 1;  
  int nabf1 = nabf + 1;
  int nabf2 = 2*nabf + 1;
  
  double *relem, *xi, *y;
  memory->create(relem, nelem+1, "mlpod:relem");    
  memory->create(xi, n, "mlpod:xi");
  memory->create(y, n, "mlpod:y");  
  
  rbpodptr->xchenodes(xi, p); 
  for (int i=0; i<nelem+1; i++)
    relem[i] = i*M_PI/nelem;
  
  for (int i=0; i<nelem; i++) {
    rbpodptr->ref2dom(y, xi, relem[i], relem[i+1], n);          
    
    for (int j=0; j<n; j++) {
      double theta = y[j];
      double costhe = cos(theta);
        
      for (int k = 0; k < nabf1; k++) {
        abf[j + n*k + n*nabf2*i] = cos(k * theta);
        dabf[j + n*k + n*nabf2*i] = -k * sin(k * theta);          
      }

      for (int k = 1; k < nabf1; k++) {
        int m = nabf+k;
        abf[j + n*m + n*nabf2*i] = sin(k * theta);
        dabf[j + n*m + n*nabf2*i] = k * cos(k * theta);
      }
    }
  }  
  
  memory->destroy(relem);
  memory->destroy(xi);
  memory->destroy(y);  
}

void MLPOD::femapproximation3body(double *cphi, double *coeff, double rin, 
        double rcut, int nrbf, int nelemr, int nabf, int nelema, int p)
{  
  int n = p + 1;
  int nabf2 = 2*nabf + 1;  
  
  double *rbf, *drbf, *abf, *dabf;    
  memory->create(rbf, n*nrbf*nelemr, "mlpod:rbf");    
  memory->create(drbf, n*nrbf*nelemr, "mlpod:drbf");    
  memory->create(abf, n*nabf2*nelema, "mlpod:abf");    
  memory->create(dabf, n*nabf2*nelema, "mlpod:dabf");        
  
 femrbf(rbf, drbf, rin, rcut, nrbf, nelemr, p);
 femabf(abf, dabf, nabf, nelema, p);
    
  int dim = 3, four = 4;
  int np = n*n*n;  
  char chn = 'N';  
  double alpha = 1.0, beta = 0.0;
  
  double *phi, *xi, *x, *A, *Ainv, *work;
  memory->create(xi, n, "mlpod:xi");      
  memory->create(x, np * dim, "mlpod:x");      
  memory->create(phi, np * four, "mlpod:phi");   
  memory->create(A, np * np, "mlpod:A");      
  memory->create(Ainv, np * np, "mlpod:Ainv");   
  memory->create(work, np*np, "mlpod:work");  
  int *ipiv;  
  memory->create(ipiv, np * np, "mlpod:ipiv");      
    
  rbpodptr->xchenodes(xi, p);
  for (int i1=0; i1<n; i1++)
    for (int i2=0; i2<n; i2++)
      for (int i3=0; i3<n; i3++) {
        x[i3 + n*i2 + n*n*i1] = xi[i3];
        x[i3 + n*i2 + n*n*i1 + n*n*n] = xi[i2];
        x[i3 + n*i2 + n*n*i1 + n*n*n*2] = xi[i1];
      }
  rbpodptr->tensorpolynomials(A, x, p, np, dim);
    
  int info;
  int lwork = np*np;
  for (int i=0; i<np*np; i++) Ainv[i] = A[i];      
  DGETRF(&np,&np,Ainv,&np,ipiv,&info);
  DGETRI(&np,Ainv,&np,ipiv,work,&lwork,&info);
      
  for (int e1=0; e1<nelemr; e1++) {
    double *rbf1 = &rbf[n*nrbf*e1];
    double *drbf1 = &drbf[n*nrbf*e1];
    
    for (int e2=0; e2<nelemr; e2++) {
      double *rbf2 = &rbf[n*nrbf*e2];
      double *drbf2 = &drbf[n*nrbf*e2];    
      
      for (int e3=0; e3<nelema; e3++) {
        double *abf3 = &abf[n*nabf2*e3];
        double *dabf3 = &dabf[n*nabf2*e3];
        
        int e = e3 + nelema*e2 + nelema*nelemr*e1;
        
        if ((e % 25000) == 0) {
          if (comm->me == 0) utils::logmesg(lmp, "Finite Element: # {}\n", e + 1);
        }
        
        for (int i1=0; i1<n; i1++)
          for (int i2=0; i2<n; i2++)
            for (int i3=0; i3<n; i3++) {
              
              double fn = 0.0, fm = 0.0, fq = 0.0, fp = 0.0;
              for (int j1 = 0; j1 < nrbf; j1++) {
                for (int j2 = 0; j2 < nrbf; j2++) {
                  double uj = rbf1[i1 + n * j1];
                  double uk = rbf2[i2 + n * j2];
                  double ujk = uj * uk;
                  double dujkdrj = drbf1[i1 + n * j1] * uk;
                  double dujkdrk = drbf2[i2 + n * j2] * uj;

                  for (int j3 = 0; j3 < nabf2; j3++) {                    
                    double tm1 = coeff[j3 + nabf2*j2 + nabf2*nrbf*j1];            
                    double tm = abf3[i3 + n*j3];          

                    fn += tm1 * ujk * tm;
                    fm += tm1 * dujkdrj * tm;
                    fq += tm1 * dujkdrk * tm;
                    fp += tm1 * ujk * dabf3[i3 + n*j3];            
                  }
                }
              }
              
              phi[i3 + n*i2 + n*n*i1 + np*0] = fn;
              phi[i3 + n*i2 + n*n*i1 + np*1] = fm;
              phi[i3 + n*i2 + n*n*i1 + np*2] = fq;
              phi[i3 + n*i2 + n*n*i1 + np*3] = fp;
            }
        
        DGEMM(&chn, &chn, &np, &four, &np, &alpha, Ainv, &np, phi, &np, &beta, &cphi[np*four*e], &np);            
      }
    }
  }
  
  memory->destroy(rbf);  
  memory->destroy(drbf);  
  memory->destroy(abf);  
  memory->destroy(dabf);    
  
  memory->destroy(phi); 
  memory->destroy(xi); 
  memory->destroy(x); 
  memory->destroy(A);
  memory->destroy(Ainv); 
  memory->destroy(work); 
  memory->destroy(ipiv); 
}

void MLPOD::polyfit3body(double *coeff3, int memoryallocate)
{  
  if ((nfemcoeffs==0) || (femdegree==0)) return;
  
  double *coeff;
  int nabf = pod.nabf3;
  int nrbf = pod.nrbf3;
  int nabf2 = 2*nabf + 1;  
  
  if (memoryallocate==1) memory->create(femcoeffs, nfemcoeffs, "mlpod:femcoeffs");  
  
  memory->create(coeff, nabf2*nrbf*nrbf, "mlpod:coeff");  
  
  for (int e=0; e<pod.nc3; e++) {
    
    for (int j1 = 0; j1 < nrbf; j1++) 
      for (int j2 = 0; j2 < nrbf; j2++) 
        for (int j3 = 0; j3 < nabf2; j3++) {
          int k = j3 + (nabf2)*j2 + nabf2*nrbf*j1;          
          coeff[k] = coeff3[e + pod.nc3 * k];            
        }
  
    double *cphi = &femcoeffs[npelem*4*nfemelem*e];
    femapproximation3body(cphi, coeff, pod.rin, pod.rcut, nrbf, nelemrbf, nabf, nelemabf, femdegree);
  }

  memory->destroy(coeff); 
}

void MLPOD::fempod_energyforce(double *fij, double *ei, double *rij, double *podcoeff, 
                       double *tmpmem, int *idxi, int *numij, int *typeai, int *ti, int *tj, int natom, int Nij) 
{  
  int dim = 3;
  int nelements = pod.nelements;
  int nelements2 = nelements * (nelements + 1) / 2;
  int typei, typej, typek, ii, ij, ik, e1, e2, e3;
  int nabf = pod.nabf3;
  int nabf1 = nabf + 1;
  int nabf2 = 2*nabf + 1;
  int nrbf2 = pod.nbf2;
  int nrbf3 = pod.nrbf3;
  int nd1 = pod.nd1;
  int nd2 = pod.nd2;
  int *elemindex = pod.elemindex;
  
  double xij1, xij2, xij3, xik1, xik2, xik3;
  double xdot, dijsq, diksq, dij, dik;
  double costhe, sinthe, theta, dtheta;
  double tm, tm1, tm2, dct1, dct2, dct3, dct4, dct5, dct6;
  double fm, fn, fp, fq, x1, x2, x3;
    
  int p1 = femdegree + 1;
  int n4  = p1*4;
  int nsq4 = p1*p1*4;
  double *c1 = &tmpmem[0];
  double *c2 = &tmpmem[nsq4];
  double *rbf, *drbfdr, *abf, *dabf;  
  
  if (femdegree==0) {    
    rbf = &tmpmem[0];
    drbfdr = &tmpmem[Nij * nrbf2];  
    abf = &tmpmem[2 * Nij * nrbf2];
    dabf = &tmpmem[2 * Nij * nrbf2 + nabf2];     
    rbpodptr->femdrbfdr(rbf, drbfdr, rij, Nij);
  }
  
  int nelemr = nelemrbf;
  int nelema = nelemabf;
  double rcut = pod.rcut;
  double rin = pod.rin;
  double dr = (rcut-rin-1e-3)/nelemr;
  double fr = 2.0/dr;    
  double dt = M_PI/nelema;
  double ft = 2.0/dt;  

  double dr2 = (rcut-rin-1e-3)/rbpodptr->nfemelem;
  double fr2 = 2.0/dr2;    
      
  double *coeff1 = &podcoeff[0];
  double *coeff2 = &podcoeff[nd1];
  double *coeff3 = &podcoeff[nd1 + nd2];
  
  for (int i = 0; i < natom; i++)
    for (int m = 1; m <= nelements; m++) {   
      ei[i] = coeff1[m-1] * ((typeai[i] == m) ? 1.0 : 0.0);      
    }
  
  podArraySetValue(fij, 0.0, 3 * Nij);
  for (int n=0; n<Nij; n++) {          
      ij = n;
      typei = ti[ij] - 1;
      typej = tj[ij] - 1;
      xij1 = rij[0 + dim * ij];
      xij2 = rij[1 + dim * ij];
      xij3 = rij[2 + dim * ij];
      dijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
      dij = sqrt(dijsq);
      
      ii = idxi[n];
      int numneigh = numij[ii + 1] - numij[ii];
      int s = numij[ii];          
      int lj = n - numij[ii];   
      
      e1 = (dij-rin-1e-3)/dr2;        
      e1 = (e1 > (rbpodptr->nfemelem-1)) ? (rbpodptr->nfemelem-1) : e1;        
      tm1 = rin+1e-3 + e1*dr2;        
      x1 = fr2 * (dij  - tm1) - 1;       
      tm2 = x1*x1;
      x2 = 1.5*tm2 - 0.5;
      x3 = (2.5*tm2 - 1.5)*x1;                          
          
      double *crbf2 = &rbpodptr->crbf[4*nrbf2*e1];    
      double *drbf2 = &rbpodptr->drbf[4*nrbf2*e1];                      
      double en = 0.0, tn = 0.0;      
      for (int m = 0; m < nrbf2; m++) {      
        int nm = n + Nij * m;
        int km = (elemindex[typei + typej * nelements] - 1) + nelements2 * m;        
        int pm = 4*m;
        en += coeff2[km] * (crbf2[0+pm] + crbf2[1+pm]*x1 + crbf2[2+pm]*x2 + crbf2[3+pm]*x3);
        tn += coeff2[km] * (drbf2[0+pm] + drbf2[1+pm]*x1 + drbf2[2+pm]*x2 + drbf2[3+pm]*x3);
      }          
      tn = tn/dij;     
      double fjx = tn*xij1;
      double fjy = tn*xij2;
      double fjz = tn*xij3;                      
      
      for (int lk = lj + 1; lk < numneigh; lk++) {
        ik = lk + s;
        typek = tj[ik] - 1;
        xik1 = rij[0 + dim * ik];
        xik2 = rij[1 + dim * ik];
        xik3 = rij[2 + dim * ik];
        diksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
        dik = sqrt(diksq);

        xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
        tm = dij * dik;
        costhe = xdot / tm;
        costhe = costhe > 1.0 ? 1.0 : costhe;
        costhe = costhe < -1.0 ? -1.0 : costhe;
        xdot = costhe * tm;
        theta = acos(costhe);

        sinthe = sqrt(1.0 - costhe * costhe);
        sinthe = sinthe > 1e-12 ? sinthe : 1e-12;
        dtheta = -1.0 / sinthe;
        
        tm1 = dtheta / (dijsq * tm);
        tm2 = dtheta / (diksq * tm);
        dct1 = (xik1 * dijsq - xij1 * xdot) * tm1;
        dct2 = (xik2 * dijsq - xij2 * xdot) * tm1;
        dct3 = (xik3 * dijsq - xij3 * xdot) * tm1;
        dct4 = (xij1 * diksq - xik1 * xdot) * tm2;
        dct5 = (xij2 * diksq - xik2 * xdot) * tm2;
        dct6 = (xij3 * diksq - xik3 * xdot) * tm2;
        
        if (femdegree==0) {
          for (int p = 0; p < nabf1; p++) {
            abf[p] = cos(p * theta);
            dabf[p] = -p * sin(p * theta);          
          }

          for (int p = 1; p < nabf1; p++) {            
            abf[nabf+p] = sin(p * theta);
            dabf[nabf+p] = p * cos(p * theta);
          }

          fn = 0.0, fm = 0.0, fq = 0.0, fp = 0.0;
          for (int m = 0; m < nrbf3; m++) {
            for (int q = 0; q < nrbf3; q++) {
              x1 = rbf[ij + Nij * m];
              x2 = rbf[ik + Nij * q];
              x3 = x1 * x2;
              costhe = drbfdr[ij + Nij * m] * x2;
              sinthe = drbfdr[ik + Nij * q] * x1;

              for (int p = 0; p < nabf2; p++) {
                e1 = p + (nabf2)*q + nabf2*nrbf3*m;
                e2 = (elemindex[typej + typek * nelements] - 1) +
                    nelements2 * typei + nelements2 * nelements * e1;            
                tm1 = coeff3[e2];            
                tm = abf[p];          

                fn += tm1 * x3 * tm;
                fm += tm1 * costhe * tm;
                fq += tm1 * sinthe * tm;
                fp += tm1 * x3 * dabf[p];            
              }
            }
          }          
        }
        else {
          e1 = (dij-rin-1e-3)/dr;        
          e1 = (e1 > (nelemr-1)) ? (nelemr-1) : e1;                
          e2 = (dik-rin-1e-3)/dr;        
          e2 = (e2 > (nelemr-1)) ? (nelemr-1) : e2;        
          e3 = theta/dt;        
          e3 = (e3 > (nelema-1)) ? (nelema-1) : e3;    

          int e = e3 + nelema*e2 + nelema*nelemr*e1;
          int idxe = (elemindex[typej + typek * nelements] - 1) + nelements2 * typei;
          double *c = &femcoeffs[npelem*4*(e + nfemelem*idxe)];        

          if (femdegree==3) {
            tm1 = e3*dt;    
            x1 = ft * (theta  - tm1) - 1;   
            tm2 = x1*x1;
            x2 = 1.5*tm2 - 0.5;
            x3 = (2.5*tm2 - 1.5)*x1;                  
            for (int i=0; i<nsq4; i++)           
              c1[i] = c[0 + p1*i] + x1*c[1 + p1*i] + x2*c[2 + p1*i] + x3*c[3 + p1*i];

            tm1 = rin+1e-3 + e2*dr;    
            x1 = fr * (dik  - tm1) - 1;   
            tm2 = x1*x1;
            x2 = 1.5*tm2 - 0.5;
            x3 = (2.5*tm2 - 1.5)*x1;                  
            for (int i=0; i<n4; i++)
              c2[i] = c1[0 + p1*i] + x1*c1[1 + p1*i] + x2*c1[2 + p1*i] + x3*c1[3 + p1*i];

            tm1 = rin+1e-3 + e1*dr;        
            x1 = fr * (dij  - tm1) - 1;       
            tm2 = x1*x1;
            x2 = 1.5*tm2 - 0.5;
            x3 = (2.5*tm2 - 1.5)*x1;                          
            fn = c2[0 + p1*0] + x1*c2[1 + p1*0] + x2*c2[2 + p1*0] + x3*c2[3 + p1*0];
            fm = c2[0 + p1*1] + x1*c2[1 + p1*1] + x2*c2[2 + p1*1] + x3*c2[3 + p1*1];
            fq = c2[0 + p1*2] + x1*c2[1 + p1*2] + x2*c2[2 + p1*2] + x3*c2[3 + p1*2];
            fp = c2[0 + p1*3] + x1*c2[1 + p1*3] + x2*c2[2 + p1*3] + x3*c2[3 + p1*3];
          }
          else if (femdegree==2) {
            tm1 = e3*dt;    
            x1 = ft * (theta  - tm1) - 1;   
            tm2 = x1*x1;
            x2 = 1.5*tm2 - 0.5;
            for (int i=0; i<nsq4; i++)           
              c1[i] = c[0 + p1*i] + x1*c[1 + p1*i] + x2*c[2 + p1*i];

            tm1 = rin+1e-3 + e2*dr;    
            x1 = fr * (dik  - tm1) - 1;   
            tm2 = x1*x1;
            x2 = 1.5*tm2 - 0.5;
            for (int i=0; i<n4; i++)
              c2[i] = c1[0 + p1*i] + x1*c1[1 + p1*i] + x2*c1[2 + p1*i];

            tm1 = rin+1e-3 + e1*dr;        
            x1 = fr * (dij  - tm1) - 1;       
            tm2 = x1*x1;
            x2 = 1.5*tm2 - 0.5;
            fn = c2[0 + p1*0] + x1*c2[1 + p1*0] + x2*c2[2 + p1*0];
            fm = c2[0 + p1*1] + x1*c2[1 + p1*1] + x2*c2[2 + p1*1];
            fq = c2[0 + p1*2] + x1*c2[1 + p1*2] + x2*c2[2 + p1*2];
            fp = c2[0 + p1*3] + x1*c2[1 + p1*3] + x2*c2[2 + p1*3];
          }
        }
        
        en += fn;
        tm1 = fm/dij;
        tm2 = fq/dik;        
        fjx += (tm1*xij1 + fp * dct1);
        fjy += (tm1*xij2 + fp * dct2);
        fjz += (tm1*xij3 + fp * dct3);                                
        fij[0+3*ik] += (tm2*xik1 + fp * dct4);
        fij[1+3*ik] += (tm2*xik2 + fp * dct5);
        fij[2+3*ik] += (tm2*xik3 + fp * dct6);                                        
      }
      
      ei[ii] += en;
      fij[0+3*ij] += fjx;
      fij[1+3*ij] += fjy;
      fij[2+3*ij] += fjz;                                                  
  }        
}

double MLPOD::energyforce_calculation(double *force, double *fij, double *rij, double *podcoeff, double *tmpmem, 
        int *idxi, int *numij, int *typeai, int *ai, int *aj, int *ti, int *tj, int natom, int Nij) 
{      
  double *ei = &tmpmem[0];
  double *tmp = &tmpmem[natom];
  
  fempod_energyforce(fij, ei, rij, podcoeff, tmp, idxi, numij, typeai, ti, tj, natom, Nij);
  
  double energy = 0.0;  
  for (int i = 0; i < natom; i++)
    energy += ei[i];
    
  podArraySetValue(force, 0.0, 3 * natom);
  tallyforce(force, fij, ai, aj, Nij);
  
  return energy;
}




// double MLPOD::fem3body_energyforce(double *fij, double *ei, double *rij, double *rbf, double *drbfdr, 
//                     double *coeff3, double *tmpmem, int *elemindex, int *pairnumsum, int *ti, int *tj, 
//                     int nelements, int nrbf, int nabf, int natom, int Nij) {
//   int dim = 3;
//   int nelements2 = nelements * (nelements + 1) / 2;
//   int typei, typej, typek, ij, ik;
// 
//   double xij1, xij2, xij3, xik1, xik2, xik3;
//   double xdot, dijsq, diksq, dij, dik;
//   double costhe, sinthe, theta, dtheta;
//   double tm, tm1, tm2, dct1, dct2, dct3, dct4, dct5, dct6;
//   double fm, fn, fp, fq, x1, x2, x3;
//   
//   int nabf1 = nabf + 1;
//   int nabf2 = 2*nabf + 1;
//   
//   int p1 = femdegree + 1;
//   int n4  = p1*4;
//   int nsq4 = p1*p1*4;
//   double *c1 = &tmpmem[0];
//   double *c2 = &tmpmem[nsq4];
//     
//   int nelemr = nelemrbf;
//   int nelema = nelemabf;
//   double rcut = pod.rcut;
//   double rin = pod.rin;
//   double dr = (rcut-rin-1e-3)/nelemr;
//   double fr = 2.0/dr;    
//   double dt = M_PI/nelema;
//   double ft = 2.0/dt;  
//   
//   double energy = 0.0;
//   for (int ii = 0; ii < natom; ii++) {
//     int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
//     int s = pairnumsum[ii];    
//     for (int lj = 0; lj < numneigh; lj++) {
//       ij = lj + s;
//       typei = ti[ij] - 1;
//       typej = tj[ij] - 1;
//       xij1 = rij[0 + dim * ij];
//       xij2 = rij[1 + dim * ij];
//       xij3 = rij[2 + dim * ij];
//       dijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
//       dij = sqrt(dijsq);
//       
//       double en = 0.0, fjx = 0.0, fjy = 0.0, fjz = 0.0;
//       for (int lk = lj + 1; lk < numneigh; lk++) {
//         ik = lk + s;
//         typek = tj[ik] - 1;
//         xik1 = rij[0 + dim * ik];
//         xik2 = rij[1 + dim * ik];
//         xik3 = rij[2 + dim * ik];
//         diksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
//         dik = sqrt(diksq);
// 
//         xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
//         tm = dij * dik;
//         costhe = xdot / tm;
//         costhe = costhe > 1.0 ? 1.0 : costhe;
//         costhe = costhe < -1.0 ? -1.0 : costhe;
//         xdot = costhe * tm;
//         theta = acos(costhe);
// 
//         sinthe = sqrt(1.0 - costhe * costhe);
//         sinthe = sinthe > 1e-12 ? sinthe : 1e-12;
//         dtheta = -1.0 / sinthe;
//         
//         tm1 = dtheta / (dijsq * tm);
//         tm2 = dtheta / (diksq * tm);
//         dct1 = (xik1 * dijsq - xij1 * xdot) * tm1;
//         dct2 = (xik2 * dijsq - xij2 * xdot) * tm1;
//         dct3 = (xik3 * dijsq - xij3 * xdot) * tm1;
//         dct4 = (xij1 * diksq - xik1 * xdot) * tm2;
//         dct5 = (xij2 * diksq - xik2 * xdot) * tm2;
//         dct6 = (xij3 * diksq - xik3 * xdot) * tm2;
//         
//         int e1 = (dij-rin-1e-3)/dr;        
//         e1 = (e1 > (nelemr-1)) ? (nelemr-1) : e1;        
//         int e2 = (dik-rin-1e-3)/dr;        
//         e2 = (e2 > (nelemr-1)) ? (nelemr-1) : e2;        
//         int e3 = theta/dt;        
//         e3 = (e3 > (nelema-1)) ? (nelema-1) : e3;    
//         
//         int e = e3 + nelema*e2 + nelema*nelemr*e1;
//         int idxe = (elemindex[typej + typek * nelements] - 1) + nelements2 * typei;
//         double *c = &femcoeffs[npelem*4*(e + nfemelem*idxe)];
//         
//         if (femdegree==3) {
//           tm1 = e3*dt;    
//           x1 = ft * (theta  - tm1) - 1;   
//           tm2 = x1*x1;
//           x2 = 1.5*tm2 - 0.5;
//           x3 = (2.5*tm2 - 1.5)*x1;                  
//           for (int i=0; i<nsq4; i++)           
//             c1[i] = c[0 + p1*i] + x1*c[1 + p1*i] + x2*c[2 + p1*i] + x3*c[3 + p1*i];
// 
//           tm1 = rin+1e-3 + e2*dr;    
//           x1 = fr * (dik  - tm1) - 1;   
//           tm2 = x1*x1;
//           x2 = 1.5*tm2 - 0.5;
//           x3 = (2.5*tm2 - 1.5)*x1;                  
//           for (int i=0; i<n4; i++)
//             c2[i] = c1[0 + p1*i] + x1*c1[1 + p1*i] + x2*c1[2 + p1*i] + x3*c1[3 + p1*i];
// 
//           tm1 = rin+1e-3 + e1*dr;        
//           x1 = fr * (dij  - tm1) - 1;       
//           tm2 = x1*x1;
//           x2 = 1.5*tm2 - 0.5;
//           x3 = (2.5*tm2 - 1.5)*x1;                          
//           fn = c2[0 + p1*0] + x1*c2[1 + p1*0] + x2*c2[2 + p1*0] + x3*c2[3 + p1*0];
//           fm = c2[0 + p1*1] + x1*c2[1 + p1*1] + x2*c2[2 + p1*1] + x3*c2[3 + p1*1];
//           fq = c2[0 + p1*2] + x1*c2[1 + p1*2] + x2*c2[2 + p1*2] + x3*c2[3 + p1*2];
//           fp = c2[0 + p1*3] + x1*c2[1 + p1*3] + x2*c2[2 + p1*3] + x3*c2[3 + p1*3];
//         }
//         else if (femdegree==2) {
//           tm1 = e3*dt;    
//           x1 = ft * (theta  - tm1) - 1;   
//           tm2 = x1*x1;
//           x2 = 1.5*tm2 - 0.5;
//           for (int i=0; i<nsq4; i++)           
//             c1[i] = c[0 + p1*i] + x1*c[1 + p1*i] + x2*c[2 + p1*i];
// 
//           tm1 = rin+1e-3 + e2*dr;    
//           x1 = fr * (dik  - tm1) - 1;   
//           tm2 = x1*x1;
//           x2 = 1.5*tm2 - 0.5;
//           for (int i=0; i<n4; i++)
//             c2[i] = c1[0 + p1*i] + x1*c1[1 + p1*i] + x2*c1[2 + p1*i];
// 
//           tm1 = rin+1e-3 + e1*dr;        
//           x1 = fr * (dij  - tm1) - 1;       
//           tm2 = x1*x1;
//           x2 = 1.5*tm2 - 0.5;
//           fn = c2[0 + p1*0] + x1*c2[1 + p1*0] + x2*c2[2 + p1*0];
//           fm = c2[0 + p1*1] + x1*c2[1 + p1*1] + x2*c2[2 + p1*1];
//           fq = c2[0 + p1*2] + x1*c2[1 + p1*2] + x2*c2[2 + p1*2];
//           fp = c2[0 + p1*3] + x1*c2[1 + p1*3] + x2*c2[2 + p1*3];
//         }
//         
//         en += fn;
//         tm1 = fm/dij;
//         tm2 = fq/dik;        
//         fjx += (tm1*xij1 + fp * dct1);
//         fjy += (tm1*xij2 + fp * dct2);
//         fjz += (tm1*xij3 + fp * dct3);                                
//         fij[0+3*ik] += (tm2*xik1 + fp * dct4);
//         fij[1+3*ik] += (tm2*xik2 + fp * dct5);
//         fij[2+3*ik] += (tm2*xik3 + fp * dct6);                                        
//       }
//       
//       energy += en;
//       ei[ii] += en;
//       fij[0+3*ij] += fjx;
//       fij[1+3*ij] += fjy;
//       fij[2+3*ij] += fjz;                                      
//     }        
//   }
//     
//   return energy;
// }

// int MLPOD::lammpsNeighPairs(double *rij, double **x, double rcutsq, int *idxi,
//                             int *ai, int *aj, int *ti, int *tj, int *pairnumsum,
//                             int *atomtype, int *numneigh, int *ilist,
//                             int **jlist, int inum) {
//   int ninside = 0;
//   for (int ii = 0; ii < inum; ii++) {
//     int gi = ilist[ii];
//     int itype = atomtype[gi];
//     int m = numneigh[gi];
//     pairnumsum[ii + 1] = 0;
//     for (int l = 0; l < m; l++) {
//       int gj = jlist[gi][l];
//       double delx = x[gj][0] - x[gi][0];
//       double dely = x[gj][1] - x[gi][1];
//       double delz = x[gj][2] - x[gi][2];
//       double rsq = delx * delx + dely * dely + delz * delz;
//       if (rsq < rcutsq && rsq > 1e-20) {
//         rij[ninside * 3 + 0] = delx;
//         rij[ninside * 3 + 1] = dely;
//         rij[ninside * 3 + 2] = delz;
//         idxi[ninside] = ii;
//         ai[ninside] = gi;
//         aj[ninside] = gj;
//         ti[ninside] = itype;
//         tj[ninside] = atomtype[gj];
//         ninside++;
//         pairnumsum[ii + 1] += 1;
//       }
//     }
//   }
// 
//   pairnumsum[0] = 0;
//   for (int ii = 0; ii < inum; ii++)
//     pairnumsum[ii + 1] = pairnumsum[ii + 1] + pairnumsum[ii];
// 
//   return ninside;
// };
// 
// void MLPOD::podtally2b(double *eatom, double *eij, int *idxi, int *ti, int *tj,
//                        int *elemindex, int nelements, int nbf, int natom,
//                        int N) {
//   int nelements2 = nelements * (nelements + 1) / 2;
//   for (int n = 0; n < N; n++) {
//     int i1 = idxi[n];
//     int typei = ti[n] - 1;
//     int typej = tj[n] - 1;
//     for (int m = 0; m < nbf; m++) {
//       int im = i1 + natom * ((elemindex[typei + typej * nelements] - 1) +
//                              nelements2 * m);
//       int nm = n + N * m;
//       eatom[im] += eij[nm];
//     }
//   }
// }
// 
// void MLPOD::pod1body(double *eatom, int *atomtype, int nelements, int natom) {
//   for (int m = 1; m <= nelements; m++)
//     for (int i = 0; i < natom; i++)
//       eatom[i + natom * (m - 1)] = (atomtype[i] == m) ? 1.0 : 0.0;
// }
// 
// void MLPOD::pod3body(double *eatom, double *yij, double *e2ij, double *tmpmem,
//                      int *elemindex, int *pairnumsum, int *, int *ti, int *tj,
//                      int nrbf, int nabf, int nelements, int natom, int Nij) {
//   int dim = 3;
//   int nabf1 = nabf + 1;
//   int nabf2 = 2*nabf + 1;
//   int nelements2 = nelements * (nelements + 1) / 2;
//   int n, nijk, typei, typej, typek, ij, ik;
// 
//   double xij1, xij2, xij3, xik1, xik2, xik3;
//   double xdot, rijsq, riksq, rij, rik;
//   double costhe, theta;
//   double uj, uk, rbf;
// 
//   double *abf = &tmpmem[0];
//   double *etm = &tmpmem[nabf2];
// 
//   for (int ii = 0; ii < natom; ii++) {
//     int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
//     int s = pairnumsum[ii];
// 
//     for (int m = 0; m < pod.nd3; m++)
//       etm[m] = 0.0;
// 
//     for (int lj = 0; lj < numneigh; lj++) {
//       ij = lj + s;
//       typei = ti[ij] - 1;
//       typej = tj[ij] - 1;
//       xij1 = yij[0 + dim * ij];
//       xij2 = yij[1 + dim * ij];
//       xij3 = yij[2 + dim * ij];
//       rijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
//       rij = sqrt(rijsq);
//       for (int lk = lj + 1; lk < numneigh; lk++) {
//         ik = lk + s;
//         typek = tj[ik] - 1;
//         xik1 = yij[0 + dim * ik];
//         xik2 = yij[1 + dim * ik];
//         xik3 = yij[2 + dim * ik];
//         riksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
//         rik = sqrt(riksq);
// 
//         xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
//         costhe = xdot / (rij * rik);
//         costhe = costhe > 1.0 ? 1.0 : costhe;
//         costhe = costhe < -1.0 ? -1.0 : costhe;
//         theta = acos(costhe);
// 
//         for (int p = 0; p < nabf1; p++) abf[p] = cos(p * theta);
//         for (int p = 1; p < nabf1; p++) abf[nabf + p] = sin(p * theta);
//                   
//         for (int m = 0; m < nrbf; m++) {
//           for (int q = 0; q < nrbf; q++) {
//             uj = e2ij[lj + s + Nij * m];
//             uk = e2ij[lk + s + Nij * q];
//             rbf = uj * uk;
//             for (int p = 0; p < nabf2; p++) {
//               n = p + (nabf2)*q + nabf2*nrbf*m;
//               nijk = (elemindex[typej + typek * nelements] - 1) +
//                      nelements2 * typei + nelements2 * nelements * n;
//               etm[nijk] += rbf * abf[p];              
//             }
//           }
//         }                
//       }
//     }
//     for (int m = 0; m < pod.nd3; m++)
//       eatom[ii + natom * m] += etm[m];
//   }
// }
// 
// void MLPOD::poddesc_ij(double *eatom1, double *eatom2, double *eatom3,
//                        double *rij, double *besselparams,
//                        double *tmpmem, double rin, double rcut, int *pairnumsum,
//                        int *atomtype, int *idxi, int *ti, int *tj,
//                        int *elemindex, int *pdegree, int nbesselpars, int nrbf2,
//                        int nrbf3, int nabf, int nelements, int Nij, int natom) {
//   int nrbf = MAX(nrbf2, nrbf3);
//   int ns = pdegree[0] * nbesselpars + pdegree[1];
// 
//   double *e2ij = &tmpmem[0];
//   double *e2ijt = &tmpmem[Nij * nrbf];
// 
//   rbpodptr->femradialbasis(e2ij, rij, Nij);
//   
//   pod1body(eatom1, atomtype, nelements, natom);
// 
//   podtally2b(eatom2, e2ij, idxi, ti, tj, elemindex, nelements, nrbf2, natom,
//              Nij);
// 
//   pod3body(eatom3, rij, e2ij, &tmpmem[Nij * nrbf], elemindex, pairnumsum, idxi,
//            ti, tj, nrbf3, nabf, nelements, natom, Nij);
// }

// void MLPOD::linear_descriptors_ij(double *gd, double *eatom, double *rij,
//                                   double *tmpmem, int *pairnumsum,
//                                   int *atomtype, int *idxi, int *ti, int *tj,
//                                   int natom, int Nij) {
//   int nelements = pod.nelements;
//   int nbesselpars = pod.nbesselpars;
//   int nrbf2 = pod.nbf2;
//   int nabf3 = pod.nabf3;
//   int nrbf3 = pod.nrbf3;
//   int nd1 = pod.nd1;
//   int nd2 = pod.nd2;
//   int nd3 = pod.nd3;
//   int nd4 = pod.nd4;
//   int nd1234 = nd1 + nd2 + nd3 + nd4;
//   int *pdegree2 = pod.twobody;
//   int *elemindex = pod.elemindex;
//   double rin = pod.rin;
//   double rcut = pod.rcut;
//   //double *Phi2 = pod.Phi2;
//   double *besselparams = pod.besselparams;
// 
//   double *eatom1 = &eatom[0];
//   double *eatom2 = &eatom[0 + natom * nd1];
//   double *eatom3 = &eatom[0 + natom * (nd1 + nd2)];
//   double *eatom4 = &eatom[0 + natom * (nd1 + nd2 + nd3)];
// 
//   podArraySetValue(eatom1, 0.0, natom * nd1234);
// 
//   poddesc_ij(eatom1, eatom2, eatom3, rij, besselparams, tmpmem, rin, rcut,
//              pairnumsum, atomtype, idxi, ti, tj, elemindex, pdegree2,
//              nbesselpars, nrbf2, nrbf3, nabf3, nelements, Nij, natom);
// 
//   podArraySetValue(tmpmem, 1.0, natom);
// 
//   char cht = 'T';
//   double one = 1.0;
//   int inc1 = 1;
//   DGEMV(&cht, &natom, &nd1234, &one, eatom, &natom, tmpmem, &inc1, &one, gd, &inc1);
// }
// 
// double MLPOD::calculate_energy(double *effectivecoeff, double *gd, double *coeff) {
//   double energy = 0.0;
//   for (int i = 0; i < pod.nd; i++) {
//     effectivecoeff[i] = coeff[i];
//     energy += coeff[i] * gd[i];
//   }
// 
//   return energy;
// }
// 
// void MLPOD::pod2body_force(double *force, double *fij, double *coeff2, int *ai,
//                            int *aj, int *ti, int *tj, int *elemindex,
//                            int nelements, int nbf, int, int N) {
//   int nelements2 = nelements * (nelements + 1) / 2;
//   for (int n = 0; n < N; n++) {
//     int i1 = ai[n];
//     int j1 = aj[n];
//     int typei = ti[n] - 1;
//     int typej = tj[n] - 1;
//     for (int m = 0; m < nbf; m++) {
//       int im = 3 * i1;
//       int jm = 3 * j1;
//       int nm = n + N * m;
//       int km = (elemindex[typei + typej * nelements] - 1) + nelements2 * m;
//       double ce = coeff2[km];
//       force[0 + im] += fij[0 + 3 * nm] * ce;
//       force[1 + im] += fij[1 + 3 * nm] * ce;
//       force[2 + im] += fij[2 + 3 * nm] * ce;
//       force[0 + jm] -= fij[0 + 3 * nm] * ce;
//       force[1 + jm] -= fij[1 + 3 * nm] * ce;
//       force[2 + jm] -= fij[2 + 3 * nm] * ce;
//     }
//   }
// }
// 
// void MLPOD::pod3body_force(double *force, double *yij, double *e2ij,
//                            double *f2ij, double *coeff3, double *tmpmem,
//                            int *elemindex, int *pairnumsum, int *ai, int *aj,
//                            int *ti, int *tj, int nrbf, int nabf, int nelements,
//                            int natom, int Nij) {
//   int dim = 3;
//   int nelements2 = nelements * (nelements + 1) / 2;
//   int n, c, nijk3, typei, typej, typek, ij, ik, i, j, k;
// 
//   double xij1, xij2, xij3, xik1, xik2, xik3;
//   double xdot, rijsq, riksq, rij, rik;
//   double costhe, sinthe, theta, dtheta;
//   double tm, tm1, tm2, dct1, dct2, dct3, dct4, dct5, dct6;
// 
//   int nabf1 = nabf + 1;
//   int nabf2 = 2*nabf + 1;
//   double *abf = &tmpmem[0];
//   double *dabf1 = &tmpmem[nabf2];
//   double *dabf2 = &tmpmem[2 * nabf2];
//   double *dabf3 = &tmpmem[3 * nabf2];
//   double *dabf4 = &tmpmem[4 * nabf2];
//   double *dabf5 = &tmpmem[5 * nabf2];
//   double *dabf6 = &tmpmem[6 * nabf2];
//   
//   for (int ii = 0; ii < natom; ii++) {
//     int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
//     int s = pairnumsum[ii];
//     for (int lj = 0; lj < numneigh; lj++) {
//       ij = lj + s;
//       i = ai[ij];
//       j = aj[ij];
//       typei = ti[ij] - 1;
//       typej = tj[ij] - 1;
//       xij1 = yij[0 + dim * ij];
//       xij2 = yij[1 + dim * ij];
//       xij3 = yij[2 + dim * ij];
//       rijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
//       rij = sqrt(rijsq);
// 
//       double fixtmp, fiytmp, fiztmp;
//       fixtmp = fiytmp = fiztmp = 0.0;
//       double fjxtmp, fjytmp, fjztmp;
//       fjxtmp = fjytmp = fjztmp = 0.0;
//       for (int lk = lj + 1; lk < numneigh; lk++) {
//         ik = lk + s;
//         k = aj[ik];
//         typek = tj[ik] - 1;
//         xik1 = yij[0 + dim * ik];
//         xik2 = yij[1 + dim * ik];
//         xik3 = yij[2 + dim * ik];
//         riksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
//         rik = sqrt(riksq);
// 
//         xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
//         costhe = xdot / (rij * rik);
//         costhe = costhe > 1.0 ? 1.0 : costhe;
//         costhe = costhe < -1.0 ? -1.0 : costhe;
//         xdot = costhe * (rij * rik);
// 
//         sinthe = sqrt(1.0 - costhe * costhe);
//         sinthe = sinthe > 1e-12 ? sinthe : 1e-12;
//         theta = acos(costhe);
//         dtheta = -1.0 / sinthe;
// 
//         tm1 = 1.0 / (rij * rijsq * rik);
//         tm2 = 1.0 / (rij * riksq * rik);
//         dct1 = (xik1 * rijsq - xij1 * xdot) * tm1;
//         dct2 = (xik2 * rijsq - xij2 * xdot) * tm1;
//         dct3 = (xik3 * rijsq - xij3 * xdot) * tm1;
//         dct4 = (xij1 * riksq - xik1 * xdot) * tm2;
//         dct5 = (xij2 * riksq - xik2 * xdot) * tm2;
//         dct6 = (xij3 * riksq - xik3 * xdot) * tm2;
// 
//         for (int p = 0; p < nabf1; p++) {
//           abf[p] = cos(p * theta);
//           tm = -p * sin(p * theta) * dtheta;
//           dabf1[p] = tm * dct1;
//           dabf2[p] = tm * dct2;
//           dabf3[p] = tm * dct3;
//           dabf4[p] = tm * dct4;
//           dabf5[p] = tm * dct5;
//           dabf6[p] = tm * dct6;
//         }
//         
//         for (int p = 1; p < nabf1; p++) {
//           int np = nabf+p;
//           abf[np] = sin(p * theta);
//           tm = p * cos(p * theta) * dtheta;
//           dabf1[np] = tm * dct1;
//           dabf2[np] = tm * dct2;
//           dabf3[np] = tm * dct3;
//           dabf4[np] = tm * dct4;
//           dabf5[np] = tm * dct5;
//           dabf6[np] = tm * dct6;
//         }
//         
//         double fjx = 0.0, fjy = 0.0, fjz = 0.0;
//         double fkx = 0.0, fky = 0.0, fkz = 0.0;
// 
//         for (int m = 0; m < nrbf; m++) 
//         for (int q = 0; q < nrbf; q++) {
//           double uj = e2ij[lj + s + Nij * m];
//           double uk = e2ij[lk + s + Nij * q];
//           double rbf = uj * uk;
//           double drbf1 = f2ij[0 + dim * (lj + s) + dim * Nij * m] * uk;
//           double drbf2 = f2ij[1 + dim * (lj + s) + dim * Nij * m] * uk;
//           double drbf3 = f2ij[2 + dim * (lj + s) + dim * Nij * m] * uk;
//           double drbf4 = f2ij[0 + dim * (lk + s) + dim * Nij * q] * uj;
//           double drbf5 = f2ij[1 + dim * (lk + s) + dim * Nij * q] * uj;
//           double drbf6 = f2ij[2 + dim * (lk + s) + dim * Nij * q] * uj;
// 
//           for (int p = 0; p < nabf2; p++) {
//             tm = abf[p];
//             double fj1 = drbf1 * tm + rbf * dabf1[p];
//             double fj2 = drbf2 * tm + rbf * dabf2[p];
//             double fj3 = drbf3 * tm + rbf * dabf3[p];
//             double fk1 = drbf4 * tm + rbf * dabf4[p];
//             double fk2 = drbf5 * tm + rbf * dabf5[p];
//             double fk3 = drbf6 * tm + rbf * dabf6[p];
// 
//             n = p + (nabf2)*q + nabf2*nrbf*m;
//             c = (elemindex[typej + typek * nelements] - 1) +
//                 nelements2 * typei + nelements2 * nelements * n;
//             tm = coeff3[c];
// 
//             fjx += fj1 * tm;
//             fjy += fj2 * tm;
//             fjz += fj3 * tm;
//             fkx += fk1 * tm;
//             fky += fk2 * tm;
//             fkz += fk3 * tm;
//           }
//         }
//         nijk3 = 3 * k;
//         force[0 + nijk3] -= fkx;
//         force[1 + nijk3] -= fky;
//         force[2 + nijk3] -= fkz;
//         fjxtmp += fjx;
//         fjytmp += fjy;
//         fjztmp += fjz;
//         fixtmp += fjx + fkx;
//         fiytmp += fjy + fky;
//         fiztmp += fjz + fkz;
//       }
//       nijk3 = 3 * j;
//       force[0 + nijk3] -= fjxtmp;
//       force[1 + nijk3] -= fjytmp;
//       force[2 + nijk3] -= fjztmp;
//       nijk3 = 3 * i;
//       force[0 + nijk3] += fixtmp;
//       force[1 + nijk3] += fiytmp;
//       force[2 + nijk3] += fiztmp;
//     }
//   }
// }
// 
// void MLPOD::calculate_force(double *force, double *effectivecoeff, double *rij,
//                             double *tmpmem, int *pairnumsum, int *atomtype,
//                             int *idxi, int *ai, int *aj, int *ti, int *tj,
//                             int natom, int Nij) {
//   int nelements = pod.nelements;
//   int nbesselpars = pod.nbesselpars;
//   int nrbf2 = pod.nbf2;
//   int nabf3 = pod.nabf3;
//   int nrbf3 = pod.nrbf3;
//   int nd1 = pod.nd1;
//   int nd2 = pod.nd2;
//   int nd3 = pod.nd3;
//   int *pdegree = pod.twobody;
//   int *elemindex = pod.elemindex;
//   double rin = pod.rin;
//   double rcut = pod.rcut;
//   //double *Phi = pod.Phi2;
//   double *besselparams = pod.besselparams;
// 
//   double *coeff2 = &effectivecoeff[nd1];
//   double *coeff3 = &effectivecoeff[nd1 + nd2];
//   double *coeff4 = &effectivecoeff[nd1 + nd2 + nd3];
// 
//   int nrbf = MAX(nrbf2, nrbf3);
//   int ns = pdegree[0] * nbesselpars + pdegree[1];
//   double *e2ij = &tmpmem[0];
//   double *f2ij = &tmpmem[Nij * nrbf];
//   double *e2ijt = &tmpmem[4 * Nij * nrbf];
//   double *f2ijt = &tmpmem[4 * Nij * nrbf + Nij * ns];
// 
//   rbpodptr->femradialbasis(e2ij, f2ij, rij, Nij);
//   
//   pod2body_force(force, f2ij, coeff2, ai, aj, ti, tj, elemindex, nelements,
//                  nrbf2, natom, Nij);
// 
//   pod3body_force(force, rij, e2ij, f2ij, coeff3, &tmpmem[4 * Nij * nrbf],
//                  elemindex, pairnumsum, ai, aj, ti, tj, nrbf3, nabf3, nelements,
//                  natom, Nij);
// }

// double MLPOD::energyforce_calculation(double *force, double *podcoeff,
//                                       double *effectivecoeff, double *gd,
//                                       double *rij, double *tmpmem,
//                                       int *pairnumsum, int *atomtype, int *idxi,
//                                       int *ai, int *aj, int *ti, int *tj,
//                                       int natom, int Nij) {
//   double *eatom = &tmpmem[0];
//   podArraySetValue(gd, 0.0, pod.nd);
//   linear_descriptors_ij(gd, eatom, rij, &tmpmem[natom * pod.nd], pairnumsum,
//                         atomtype, idxi, ti, tj, natom, Nij);
// 
//   double energy = calculate_energy(effectivecoeff, gd, podcoeff);
//   
// //   for (int i=pod.nd1+pod.nd2; i<pod.nd; i++)
// //     printf("\%g ", gd[i]); 
// //   printf("\n");
//   
//   podArraySetValue(force, 0.0, 3 * natom);
// 
//   calculate_force(force, effectivecoeff, rij, tmpmem, pairnumsum, atomtype,
//                   idxi, ai, aj, ti, tj, natom, Nij);
// 
//   return energy;
// }

// double MLPOD::pod2body_energyforce(double *fij, double *ei, double *rij, double *rbf, 
//                         double *drbfdr, double *coeff2, int *idxi, int *ti, int *tj, 
//                         int *elemindex, int nelements, int nbf, int N) {    
//   int energy = 0.0;
//   int nelements2 = nelements * (nelements + 1) / 2;
//   for (int n = 0; n < N; n++) {
//     int typei = ti[n] - 1;
//     int typej = tj[n] - 1;    
//     double en = 0.0;
//     double fn = 0.0;    
//     for (int m = 0; m < nbf; m++) {      
//       int nm = n + N * m;
//       int km = (elemindex[typei + typej * nelements] - 1) + nelements2 * m;
//       double ce = coeff2[km];
//       en += ce * rbf[nm];
//       fn += ce * drbfdr[nm];      
//     }
//     
//     ei[idxi[n]] += en;
//     energy += en;
//         
//     double xij1 = rij[0+3*n];
//     double xij2 = rij[1+3*n];
//     double xij3 = rij[2+3*n];
//     double dij = sqrt(xij1 * xij1 + xij2 * xij2 + xij3 * xij3);
//     double tn = fn/dij;
//     fij[0+3*n] += tn*xij1;
//     fij[1+3*n] += tn*xij2;
//     fij[2+3*n] += tn*xij3;                
//   }
//   return energy;
// }

// void pod2body_energyforce(double *fij, double *ei, double *rij, double *rbf, double *drbfdr, 
//                     double *coeff2, int *elemindex, int *idxi, int *ti, int *tj, 
//                     int nelements, int nrbf, int Nij) {    
//   
//   int nelements2 = nelements * (nelements + 1) / 2;
//   
//   for (int n=0; n<Nij; n++) {
//     int ii = idxi[n];
//     int typei = ti[n] - 1;
//     int typej = tj[n] - 1;          
//     double en = 0.0;
//     double fn = 0.0;    
//     for (int m = 0; m < nrbf; m++) {      
//       int nm = n + Nij * m;
//       int km = (elemindex[typei + typej * nelements] - 1) + nelements2 * m;
//       double ce = coeff2[km];
//       en += ce * rbf[nm];
//       fn += ce * drbfdr[nm];      
//     }
// 
//     double xij1 = rij[0+3*n];
//     double xij2 = rij[1+3*n];
//     double xij3 = rij[2+3*n];
//     double dij = sqrt(xij1 * xij1 + xij2 * xij2 + xij3 * xij3);
//     double tn = fn/dij;
//     fij[0+3*n] += tn*xij1;
//     fij[1+3*n] += tn*xij2;
//     fij[2+3*n] += tn*xij3;                      
//     ei[ii] += en;
//   }        
// }

// void MLPOD::femgrid3body(double *phi, double *dphi1, double *dphi2, double *dphi3, double *coeff3, 
//         int *elemindex, double rin, double rcut, int nrbf, int nelemr, int nabf, int nelema, int p, 
//         int nelements, int typei, int typej, int typek)
// {  
//   int n = p + 1;
//   int nabf2 = 2*nabf + 1;
//   int nelements2 = nelements * (nelements + 1) / 2;
//   
//   double *rbf, *drbf, *abf, *dabf, *coeff;    
//   memory->create(rbf, n*nrbf*nelemr, "mlpod:rbf");    
//   memory->create(drbf, n*nrbf*nelemr, "mlpod:drbf");    
//   memory->create(abf, n*nabf2*nelema, "mlpod:abf");    
//   memory->create(dabf, n*nabf2*nelema, "mlpod:dabf");    
//   memory->create(coeff, nabf2*nrbf*nrbf, "mlpod:coeff");    
//   
//   femrbf(rbf, drbf, rin, rcut, nrbf, nelemr, p);
//   femabf(abf, dabf, nabf, nelema, p);
//   
//   for (int j1 = 0; j1 < nrbf; j1++) 
//     for (int j2 = 0; j2 < nrbf; j2++) 
//       for (int j3 = 0; j3 < nabf2; j3++) {
//         int k = j3 + (nabf2)*j2 + nabf2*nrbf*j2;
//         int c = (elemindex[typej + typek * nelements] - 1) +
//                 nelements2 * typei + nelements2 * nelements * k;            
//         coeff[j3 + nabf2*j2 + nabf2*nrbf*j1] = coeff3[c];            
//       }
//   
//   for (int e1=0; e1<nelemr; e1++) {
//     double *rbf1 = &rbf[n*nrbf*e1];
//     double *drbf1 = &drbf[n*nrbf*e1];
//     
//     for (int e2=0; e2<nelemr; e2++) {
//       double *rbf2 = &rbf[n*nrbf*e2];
//       double *drbf2 = &drbf[n*nrbf*e2];    
//       
//       for (int e3=0; e3<nelema; e3++) {
//         double *abf3 = &abf[n*nabf2*e3];
//         double *dabf3 = &dabf[n*nabf2*e3];
//         
//         int e = e3 + nelema*e2 + nelema*nelemr*e1;
//         
//         for (int i1=0; i1<n; i1++)
//           for (int i2=0; i2<n; i2++)
//             for (int i3=0; i3<n; i3++) {
//               
//               double fn = 0.0, fm = 0.0, fq = 0.0, fp = 0.0;
//               for (int j1 = 0; j1 < nrbf; j1++) {
//                 for (int j2 = 0; j2 < nrbf; j2++) {
//                   double uj = rbf1[i1 + n * j1];
//                   double uk = rbf2[i2 + n * j2];
//                   double ujk = uj * uk;
//                   double dujkdrj = drbf1[i1 + n * j1] * uk;
//                   double dujkdrk = drbf2[i2 + n * j2] * uj;
// 
//                   for (int j3 = 0; j3 < nabf2; j3++) {                    
//                     double tm1 = coeff[j3 + nabf2*j2 + nabf2*nrbf*j1];            
//                     double tm = abf3[i3 + n*j3];          
// 
//                     fn += tm1 * ujk * tm;
//                     fm += tm1 * dujkdrj * tm;
//                     fq += tm1 * dujkdrk * tm;
//                     fp += tm1 * ujk * dabf3[i3 + n*j3];            
//                   }
//                 }
//               }
//               
//               phi[i3 + n*i2 + n*n*i1 + n*n*n*e] = fn;
//               dphi1[i3 + n*i2 + n*n*i1 + n*n*n*e] = fm;
//               dphi2[i3 + n*i2 + n*n*i1 + n*n*n*e] = fq;
//               dphi3[i3 + n*i2 + n*n*i1 + n*n*n*e] = fp;
//             }
//         
//       }
//     }
//   }
//   
//   memory->destroy(rbf);  
//   memory->destroy(drbf);  
//   memory->destroy(abf);  
//   memory->destroy(dabf);  
//   memory->destroy(coeff);  
// }

// 
// void MLPOD::femevaluation3body(double *phi, double *cphi, double *x, double *tmp,
//         double rin, double rcut, int nelemr, int nelema, int p, int N)
// {  
//   int one = 1;  
//   int four = 4;
//   int p1 = p + 1;
//   int n4 = p1*four;
//   int nsq = p1*p1;  
//   int nsq4 = p1*p1*four;    
//   int np = p1*p1*p1;
//   
//   char chn = 'N';  
//   double alpha = 1.0, beta = 0.0, tm;
//   
//   double dr = (rcut-rin-1e-3)/nelemr;
//   double fr = 2.0/dr;    
//   double dt = M_PI/nelema;
//   double ft = 2.0/dt;  
//   
//   double *xi = &tmp[0];
//   xi[0] = 1;
//   double *c1 = &tmp[p1];
//   double *c2 = &tmp[p1 + nsq4];
//   
//   for (int n=0; n<N; n++) {
//     double th = x[0 + 3*n];    
//     double rk = x[1 + 3*n];
//     double rj = x[2 + 3*n];
//     
//     int e1 = (rj-rin-1e-3)/dr;        
//     if (e1 > (nelemr-1)) e1 = nelemr-1;        
//     
//     int e2 = (rk-rin-1e-3)/dr;        
//     if (e2 > (nelemr-1)) e2 = nelemr-1;        
//     
//     int e3 = th/dt;        
//     if (e3 > (nelema-1)) e3 = nelema-1;    
//     
//     int e = e3 + nelema*e2 + nelema*nelemr*e1;
//     double *c = &cphi[np*four*e];
//             
//     double yt = e3*dt;    
//     xi[1] = ft * (th  - yt) - 1;   
//     double xi2 = xi[1]*xi[1];
//     xi[2] = 1.5*xi2 - 0.5;
//     xi[3] = (2.5*xi2 - 1.5)*xi[1];                  
//     //DGEMM(&chn, &chn, &one, &nsq4, &n, &alpha, xi, &one, c, &n, &beta, c1, &one);     
//     for (int i=0; i<nsq4; i++) {
//       tm = 0;
//       for (int j=0; j<n; j++)
//         tm += xi[j]*c[j + n*i];
//       c1[i] = tm;
//     }
//     
//     double yk = rin+1e-3 + e2*dr;    
//     xi[1] = fr * (rk  - yk) - 1;   
//     xi2 = xi[1]*xi[1];
//     xi[2] = 1.5*xi2 - 0.5;
//     xi[3] = (2.5*xi2 - 1.5)*xi[1];                          
//     //DGEMM(&chn, &chn, &one, &n4, &n, &alpha, xi, &one, c1, &n, &beta, c2, &one);   
//     for (int i=0; i<n4; i++) {
//       tm = 0;
//       for (int j=0; j<n; j++)
//         tm += xi[j]*c1[j + n*i];
//       c2[i] = tm;
//     }
//     
//     double yj = rin+1e-3 + e1*dr;        
//     xi[1] = fr * (rj  - yj) - 1;       
//     xi2 = xi[1]*xi[1];
//     xi[2] = 1.5*xi2 - 0.5;
//     xi[3] = (2.5*xi2 - 1.5)*xi[1];                              
//     //DGEMM(&chn, &chn, &one, &four, &n, &alpha, xi, &one, c2, &n, &beta, &phi[four*n], &one);          
//     for (int i=0; i<four; i++) {
//       tm = 0;
//       for (int j=0; j<n; j++)
//         tm += xi[j]*c2[j + n*i];
//       phi[i + four*n] = tm;
//     }
//   }    
// }


// double MLPOD::fem3body_energyforce(double *fij, double *ei, double *rij, double *rbf, double *drbfdr, 
//                     double *coeff3, double *tmpmem, int *elemindex, int *pairnumsum, int *ti, int *tj, 
//                     int nelements, int nrbf, int nabf, int natom, int Nij) {
//   int dim = 3;
//   int nelements2 = nelements * (nelements + 1) / 2;
//   int c, typei, typej, typek, ij, ik;
// 
//   double xij1, xij2, xij3, xik1, xik2, xik3;
//   double xdot, dijsq, diksq, dij, dik;
//   double costhe, sinthe, theta, dtheta;
//   double tm, tm1, tm2, dct1, dct2, dct3, dct4, dct5, dct6;
// 
//   int nabf1 = nabf + 1;
//   int nabf2 = 2*nabf + 1;
//   
//   int p1 = femdegree + 1;
//   int n4  = p1*4;
//   int nsq4 = p1*p1*4;
//   double *xi = &tmpmem[0];
//   xi[0] = 1;
//   double *c1 = &tmpmem[p1];
//   double *c2 = &tmpmem[p1 + nsq4];
//   
//   double *abf = &tmpmem[p1 + 2*nsq4];
//   double *dabf = &tmpmem[p1 + 2*nsq4 + nabf2];
//   
//   int nelemr = nelemrbf;
//   int nelema = nelemabf;
//   double rcut = pod.rcut;
//   double rin = pod.rin;
//   double dr = (rcut-rin-1e-3)/nelemr;
//   double fr = 2.0/dr;    
//   double dt = M_PI/nelema;
//   double ft = 2.0/dt;  
//   
//   double energy = 0.0;
//   for (int ii = 0; ii < natom; ii++) {
//     int numneigh = pairnumsum[ii + 1] - pairnumsum[ii];
//     int s = pairnumsum[ii];    
//     for (int lj = 0; lj < numneigh; lj++) {
//       ij = lj + s;
//       typei = ti[ij] - 1;
//       typej = tj[ij] - 1;
//       xij1 = rij[0 + dim * ij];
//       xij2 = rij[1 + dim * ij];
//       xij3 = rij[2 + dim * ij];
//       dijsq = xij1 * xij1 + xij2 * xij2 + xij3 * xij3;
//       dij = sqrt(dijsq);
//       
//       double en = 0.0, fjx = 0.0, fjy = 0.0, fjz = 0.0;
//       for (int lk = lj + 1; lk < numneigh; lk++) {
//         ik = lk + s;
//         typek = tj[ik] - 1;
//         xik1 = rij[0 + dim * ik];
//         xik2 = rij[1 + dim * ik];
//         xik3 = rij[2 + dim * ik];
//         diksq = xik1 * xik1 + xik2 * xik2 + xik3 * xik3;
//         dik = sqrt(diksq);
// 
//         xdot = xij1 * xik1 + xij2 * xik2 + xij3 * xik3;
//         tm = dij * dik;
//         costhe = xdot / tm;
//         costhe = costhe > 1.0 ? 1.0 : costhe;
//         costhe = costhe < -1.0 ? -1.0 : costhe;
//         xdot = costhe * tm;
//         theta = acos(costhe);
// 
//         sinthe = sqrt(1.0 - costhe * costhe);
//         sinthe = sinthe > 1e-12 ? sinthe : 1e-12;
//         dtheta = -1.0 / sinthe;
//         
//         tm1 = dtheta / (dijsq * tm);
//         tm2 = dtheta / (diksq * tm);
//         dct1 = (xik1 * dijsq - xij1 * xdot) * tm1;
//         dct2 = (xik2 * dijsq - xij2 * xdot) * tm1;
//         dct3 = (xik3 * dijsq - xij3 * xdot) * tm1;
//         dct4 = (xij1 * diksq - xik1 * xdot) * tm2;
//         dct5 = (xij2 * diksq - xik2 * xdot) * tm2;
//         dct6 = (xij3 * diksq - xik3 * xdot) * tm2;
//                                 
// //         for (int p = 0; p < nabf1; p++) {
// //           abf[p] = cos(p * theta);
// //           dabf[p] = -p * sin(p * theta);          
// //         }
// //         
// //         for (int p = 1; p < nabf1; p++) {
// //           int np = nabf+p;
// //           abf[np] = sin(p * theta);
// //           dabf[np] = p * cos(p * theta);
// //         }
//         
// //         double fn1 = 0.0, fm1 = 0.0, fq1 = 0.0, fp1 = 0.0;
// //         for (int m = 0; m < nrbf; m++) {
// //           for (int q = 0; q < nrbf; q++) {
// //             double uj = rbf[ij + Nij * m];
// //             double uk = rbf[ik + Nij * q];
// //             double ujk = uj * uk;
// //             double dujkdrj = drbfdr[ij + Nij * m] * uk;
// //             double dujkdrk = drbfdr[ik + Nij * q] * uj;
// // 
// //             for (int p = 0; p < nabf2; p++) {
// //               int na = p + (nabf2)*q + nabf2*nrbf*m;
// //               int nb = (elemindex[typej + typek * nelements] - 1) +
// //                   nelements2 * typei + nelements2 * nelements * na;            
// //               tm1 = coeff3[nb];            
// //               tm = abf[p];          
// // 
// //               fn1 += tm1 * ujk * tm;
// //               fm1 += tm1 * dujkdrj * tm;
// //               fq1 += tm1 * dujkdrk * tm;
// //               fp1 += tm1 * ujk * dabf[p];            
// //             }
// //           }
// //         }
//         
//         int e1 = (dij-rin-1e-3)/dr;        
//         if (e1 > (nelemr-1)) e1 = nelemr-1;        
//         int e2 = (dik-rin-1e-3)/dr;        
//         if (e2 > (nelemr-1)) e2 = nelemr-1;        
//         int e3 = theta/dt;        
//         if (e3 > (nelema-1)) e3 = nelema-1;    
//         
//         int e = e3 + nelema*e2 + nelema*nelemr*e1;
//         int idxe = (elemindex[typej + typek * nelements] - 1) + nelements2 * typei;
//         double *c = &femcoeffs[npelem*4*(e + nfemelem*idxe)];
//         
//         double yt = e3*dt;    
//         xi[1] = ft * (theta  - yt) - 1;   
//         double xi2 = xi[1]*xi[1];
//         xi[2] = 1.5*xi2 - 0.5;
//         xi[3] = (2.5*xi2 - 1.5)*xi[1];                  
//         for (int i=0; i<nsq4; i++) {
//           tm = 0;
//           for (int j=0; j<p1; j++)
//             tm += xi[j]*c[j + p1*i];
//           c1[i] = tm;
//         }
// 
//         yt = rin+1e-3 + e2*dr;    
//         xi[1] = fr * (dik  - yt) - 1;   
//         xi2 = xi[1]*xi[1];
//         xi[2] = 1.5*xi2 - 0.5;
//         xi[3] = (2.5*xi2 - 1.5)*xi[1];                          
//         for (int i=0; i<n4; i++) {
//           tm = 0;
//           for (int j=0; j<p1; j++)
//             tm += xi[j]*c1[j + p1*i];
//           c2[i] = tm;
//         }
// 
//         yt = rin+1e-3 + e1*dr;        
//         xi[1] = fr * (dij  - yt) - 1;       
//         xi2 = xi[1]*xi[1];
//         xi[2] = 1.5*xi2 - 0.5;
//         xi[3] = (2.5*xi2 - 1.5)*xi[1];                              
//         
//         double fn = 0.0, fm = 0.0, fq = 0.0, fp = 0.0;
//         for (int j=0; j<p1; j++) {
//           fn += xi[j]*c2[j + p1*0];       
//           fm += xi[j]*c2[j + p1*1];       
//           fq += xi[j]*c2[j + p1*2];       
//           fp += xi[j]*c2[j + p1*3];       
//         }
//         
// //         if ((fabs(fn-fn1)>1e-2) || (fabs(fm-fm1)>1e-2) || (fabs(fq-fq1)>1e-2) || (fabs(fp-fp1)>1e-2)) 
// //         {
// //           printf("FEM:  %g    %g     %g     %g\n", fn, fm, fq, fp); 
// //           printf("ORG:  %g    %g     %g     %g\n", fn1, fm1, fq1, fp1); 
// //         }
//         
//         en += fn;
//         tm1 = fm/dij;
//         tm2 = fq/dik;        
//         fjx += (tm1*xij1 + fp * dct1);
//         fjy += (tm1*xij2 + fp * dct2);
//         fjz += (tm1*xij3 + fp * dct3);                                
//         fij[0+3*ik] += (tm2*xik1 + fp * dct4);
//         fij[1+3*ik] += (tm2*xik2 + fp * dct5);
//         fij[2+3*ik] += (tm2*xik3 + fp * dct6);                                        
//       }
//       
//       energy += en;
//       ei[ii] += en;
//       fij[0+3*ij] += fjx;
//       fij[1+3*ij] += fjy;
//       fij[2+3*ij] += fjz;                                      
//     }        
//   }
//     
//   return energy;
// }

