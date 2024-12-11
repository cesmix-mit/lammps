// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Ngoc Cuong Nguyen (MIT)
------------------------------------------------------------------------- */

// LAMMPS header files

#include "comm.h"
#include "error.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "tokenizer.h"

#include <cmath>

// header file. Moved down here to avoid polluting other headers with its defines
#include "rbpod.h"

using namespace LAMMPS_NS;
using MathConst::MY_PI;
using MathSpecial::cube;
using MathSpecial::powint;

static constexpr int MAXLINE=1024;

// constructor
RBPOD::RBPOD(LAMMPS *_lmp, const std::string &pod_file) :
    Pointers(_lmp), Phi(nullptr), Lambda(nullptr), crbf(nullptr), drbf(nullptr)
{
  rin = 0.5;
  rcut = 5.0;
  rmax = rcut - rin;
  cutofftype = 1;
  besseldegree = 4;
  inversedegree = 8;
  nbesselpars = 3;
  ngaussianfuncs = 0;
  ns = nbesselpars*besseldegree + inversedegree;
  nrbfmax = 8;
  pdegree[0] = besseldegree;
  pdegree[1] = inversedegree;
  besselparams[0] = 1e-3;
  besselparams[1] = 2.0;
  besselparams[2] = 4.0;
    
  read_pod_file(pod_file);
  
  nfemelem = 1000;
  nfemdegree = 3;  
  femapproximation(nfemelem, nfemdegree);
}

// destructor
RBPOD::~RBPOD()
{
  memory->destroy(Phi);
  memory->destroy(Lambda);
  memory->destroy(crbf);
  memory->destroy(drbf);  
}

void RBPOD::read_pod_file(std::string pod_file)
{
  int nrbf2, nrbf3, nrbf4, nrbf33, nrbf34, nrbf44;
  
  std::string podfilename = pod_file;
  FILE *fppod;
  if (comm->me == 0) {

    fppod = utils::open_potential(podfilename,lmp,nullptr);
    if (fppod == nullptr)
      error->one(FLERR,"Cannot open POD coefficient file {}: ",
                                   podfilename, utils::getsyserror());
  }

  // loop through lines of POD file and parse keywords

  char line[MAXLINE],*ptr;
  int eof = 0;

  while (true) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fppod);
      if (ptr == nullptr) {
        eof = 1;
        fclose(fppod);
      }
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    // words = ptrs to all words in line
    // strip single and double quotes from words

    std::vector<std::string> words;
    try {
      words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();
    } catch (TokenizerException &) {
      // ignore
    }

    if (words.size() == 0) continue;

    auto keywd = words[0];

    if (keywd == "gaussian_exponents") {
      ngaussianfuncs = words.size()-1;      
      for (int i = 0; i<ngaussianfuncs; i++)
        gaussianexponents[i] = utils::numeric(FLERR,words[i+1],false,lmp);      
    }

    if (keywd == "polynomial_degrees") {
      if (ngaussianfuncs != words.size()-1)
        error->one(FLERR,"Number of polynomial degrees does not match number of gaussian exponents.", utils::getsyserror());
      for (int i = 0; i<ngaussianfuncs; i++)
        polydegrees[i] = utils::inumeric(FLERR,words[i+1],false,lmp);      
    }
    
    if ((keywd != "#") && (keywd != "species") && (keywd != "pbc") && (keywd != "gaussian_exponents") && (keywd != "polynomial_degrees")) {

      if (words.size() != 2)
        error->one(FLERR,"Improper POD file.", utils::getsyserror());      
              
      if (keywd == "rin") rin = utils::numeric(FLERR,words[1],false,lmp);
      if (keywd == "rcut") rcut = utils::numeric(FLERR,words[1],false,lmp);
      if (keywd == "cutoff_function_type")
        cutofftype = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "bessel_polynomial_degree")
        besseldegree = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "inverse_polynomial_degree")
        inversedegree = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "twobody_number_radial_basis_functions")
        nrbf2 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "threebody_number_radial_basis_functions")
        nrbf3 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "fourbody_number_radial_basis_functions")
        nrbf4 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "fivebody_number_radial_basis_functions")
        nrbf33 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "sixbody_number_radial_basis_functions")
        nrbf34 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "sevenbody_number_radial_basis_functions")
        nrbf44 = utils::inumeric(FLERR,words[1],false,lmp);
    }
  }
    
  if (nrbf2 < nrbf3) error->all(FLERR,"number of three-body radial basis functions must be equal or less than number of two-body radial basis functions");
  if (nrbf3 < nrbf4) error->all(FLERR,"number of four-body radial basis functions must be equal or less than number of three-body radial basis functions");
  if (nrbf4 < nrbf33) error->all(FLERR,"number of five-body radial basis functions must be equal or less than number of four-body radial basis functions");
  if (nrbf4 < nrbf34) error->all(FLERR,"number of six-body radial basis functions must be equal or less than number of four-body radial basis functions");
  if (nrbf4 < nrbf44) error->all(FLERR,"number of seven-body radial basis functions must be equal or less than number of four-body radial basis functions");
  nrbfmax = (nrbf2 < nrbf3) ? nrbf3 : nrbf2;
  nrbfmax = (nrbfmax < nrbf4) ? nrbf4 : nrbfmax;
  nrbfmax = (nrbfmax < nrbf33) ? nrbf33 : nrbfmax;
  nrbfmax = (nrbfmax < nrbf34) ? nrbf34 : nrbfmax;
  nrbfmax = (nrbfmax < nrbf44) ? nrbf44 : nrbfmax;

  ns = nbesselpars*besseldegree + inversedegree;  
  if (ngaussianfuncs > 0) {
    ns = ngaussianfuncs;    
    nrbfmax = ngaussianfuncs;
    if (ngaussianfuncs < nrbf2) error->all(FLERR,"number of two-body radial basis functions must be equal or less than number of gaussian functions");    
  }
  
  rmax = rcut - rin;
  init2body();  
}

/**
 * @brief Calculates the radial basis functions and their derivatives.
 *
 * @param rbf           Pointer to the array of radial basis functions.
 * @param rbfx          Pointer to the array of derivatives of radial basis functions with respect to x.
 * @param rbfy          Pointer to the array of derivatives of radial basis functions with respect to y.
 * @param rbfz          Pointer to the array of derivatives of radial basis functions with respect to z.
 * @param rij           Pointer to the relative positions of neighboring atoms and atom i.
 * @param besselparams  Pointer to the array of Bessel function parameters.
 * @param rin           Minimum distance for radial basis functions.
 * @param rmax          Maximum distance for radial basis functions.
 * @param besseldegree  Degree of Bessel functions.
 * @param inversedegree Degree of inverse distance functions.
 * @param nbesselpars   Number of Bessel function parameters.
 * @param N             Number of neighboring atoms.
 */
void RBPOD::radialbasis(double *rbf, double *rbfx, double *rbfy, double *rbfz, double *rij, int N)
{
  // Loop over all neighboring atoms
  for (int n=0; n<N; n++) {
    double xij1 = rij[0+3*n];
    double xij2 = rij[1+3*n];
    double xij3 = rij[2+3*n];

    double dij = sqrt(xij1*xij1 + xij2*xij2 + xij3*xij3);
    double dr1 = xij1/dij;
    double dr2 = xij2/dij;
    double dr3 = xij3/dij;

    double r = dij - rin;
    double y = r/rmax;
    double y2 = y*y;

    double fcut, dfcut;
    if (cutofftype==0) {
      fcut = 1;
      dfcut = 0;
    }
    else if (cutofftype==1) {
      double y3 = 1.0 - y2*y;
      double y4 = y3*y3 + 1e-6;
      double y5 = sqrt(y4);
      double y6 = exp(-1.0/y5);
      double y7 = y4*sqrt(y4);

      // Calculate the final cutoff function as y6/exp(-1)
      fcut = y6/exp(-1.0);

      // Calculate the derivative of the final cutoff function
      dfcut = ((3.0/(rmax*exp(-1.0)))*(y2)*y6*(y*y2 - 1.0))/y7;
    }
    else if (cutofftype==2) {
      fcut = 0.5*cos(MY_PI*y) + 0.5;
      dfcut = -0.5*(MY_PI*sin(MY_PI*y));
    }
    else if (cutofftype==3) {    
      double y3 = y2*y;
      double y4 = y2*y2;
      fcut = (y*(y*(20*y - 70) + 84) - 35)*y4 + 1;
      dfcut = 140*y3*(y - 1)*(y - 1)*(y - 1)/rmax;
    }
    else if (cutofftype==4) {    
      double y4 = y2*y2;
      double y5 = y*y4;
      double y6 = (y - 1)*(y - 1);  
      fcut = (y*(y*((315 - 70*y)*y - 540) + 420) - 126)*y5 + 1;
      dfcut = -630*y4*y6*y6/rmax;    
    }
    
    // Calculate fcut/r, fcut/r^2, and dfcut/r
    double f1 = fcut/r;
    double f2 = f1/r;
    double df1 = dfcut/r;

    double alpha = besselparams[0];
    double t1 = (1.0-exp(-alpha));
    double t2 = exp(-alpha*r/rmax);
    double x0 =  (1.0 - t2)/t1;
    double dx0 = (alpha/rmax)*t2/t1;

    if (nbesselpars==1) {
      for (int i=0; i<besseldegree; i++) {
        double a = (i+1)*MY_PI;
        double b = (sqrt(2.0/(rmax))/(i+1));
        double af1 = a*f1;

        double sinax = sin(a*x0);
        int nij = n + N*i;

        rbf[nij] = b*f1*sinax;

        double drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);
        rbfx[nij] = drbfdr*dr1;
        rbfy[nij] = drbfdr*dr2;
        rbfz[nij] = drbfdr*dr3;
      }
    }
    else if (nbesselpars==2) {
      alpha = besselparams[1];
      t1 = (1.0-exp(-alpha));
      t2 = exp(-alpha*r/rmax);
      double x1 =  (1.0 - t2)/t1;
      double dx1 = (alpha/rmax)*t2/t1;
      for (int i=0; i<besseldegree; i++) {
        double a = (i+1)*MY_PI;
        double b = (sqrt(2.0/(rmax))/(i+1));
        double af1 = a*f1;

        double sinax = sin(a*x0);
        int nij = n + N*i;

        rbf[nij] = b*f1*sinax;

        double drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);
        rbfx[nij] = drbfdr*dr1;
        rbfy[nij] = drbfdr*dr2;
        rbfz[nij] = drbfdr*dr3;

        sinax = sin(a*x1);
        nij = n + N*i + N*besseldegree*1;
        rbf[nij] = b*f1*sinax;

        drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x1)*dx1);
        rbfx[nij] = drbfdr*dr1;
        rbfy[nij] = drbfdr*dr2;
        rbfz[nij] = drbfdr*dr3;
      }
    }
    else if (nbesselpars==3) {
      alpha = besselparams[1];
      t1 = (1.0-exp(-alpha));
      t2 = exp(-alpha*r/rmax);
      double x1 =  (1.0 - t2)/t1;
      double dx1 = (alpha/rmax)*t2/t1;

      alpha = besselparams[2];
      t1 = (1.0-exp(-alpha));
      t2 = exp(-alpha*r/rmax);
      double x2 =  (1.0 - t2)/t1;
      double dx2 = (alpha/rmax)*t2/t1;
      for (int i=0; i<besseldegree; i++) {
        double a = (i+1)*MY_PI;
        double b = (sqrt(2.0/(rmax))/(i+1));
        double af1 = a*f1;

        double sinax = sin(a*x0);
        int nij = n + N*i;

        rbf[nij] = b*f1*sinax;
        double drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);
        rbfx[nij] = drbfdr*dr1;
        rbfy[nij] = drbfdr*dr2;
        rbfz[nij] = drbfdr*dr3;

        sinax = sin(a*x1);
        nij = n + N*i + N*besseldegree*1;

        rbf[nij] = b*f1*sinax;
        drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x1)*dx1);
        rbfx[nij] = drbfdr*dr1;
        rbfy[nij] = drbfdr*dr2;
        rbfz[nij] = drbfdr*dr3;

        sinax = sin(a*x2);
        nij = n + N*i + N*besseldegree*2;
        rbf[nij] = b*f1*sinax;
        drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x2)*dx2);
        rbfx[nij] = drbfdr*dr1;
        rbfy[nij] = drbfdr*dr2;
        rbfz[nij] = drbfdr*dr3;
      }
    }

    // Calculate fcut/dij and dfcut/dij
    f1 = fcut/dij;
    for (int i=0; i<inversedegree; i++) {
      int p = besseldegree*nbesselpars + i;
      int nij = n + N*p;
      double a = powint(dij, i+1);

      rbf[nij] = fcut/a;

      double drbfdr = (dfcut - (i+1.0)*f1)/a;
      rbfx[nij] = drbfdr*dr1;
      rbfy[nij] = drbfdr*dr2;
      rbfz[nij] = drbfdr*dr3;
    }
  }
}

void RBPOD::gaussianbasis(double *rbf, double *rbfx, double *rbfy, double *rbfz, double *rij, int N)
{
  // Loop over all neighboring atoms
  for (int n=0; n<N; n++) {
    double xij1 = rij[0+3*n];
    double xij2 = rij[1+3*n];
    double xij3 = rij[2+3*n];

    double dij = sqrt(xij1*xij1 + xij2*xij2 + xij3*xij3);
    double dr1 = xij1/dij;
    double dr2 = xij2/dij;
    double dr3 = xij3/dij;

    double r = dij - rin;
    double y = r/rmax;
    double y2 = y*y;

    double fcut, dfcut;
    if (cutofftype==0) {
      fcut = 1;
      dfcut = 0;
    }
    else if (cutofftype==1) {
      double y3 = 1.0 - y2*y;
      double y4 = y3*y3 + 1e-6;
      double y5 = sqrt(y4);
      double y6 = exp(-1.0/y5);
      double y7 = y4*sqrt(y4);

      // Calculate the final cutoff function as y6/exp(-1)
      fcut = y6/exp(-1.0);

      // Calculate the derivative of the final cutoff function
      dfcut = ((3.0/(rmax*exp(-1.0)))*(y2)*y6*(y*y2 - 1.0))/y7;
    }
    else if (cutofftype==2) {
      fcut = 0.5*cos(MY_PI*y) + 0.5;
      dfcut = -0.5*(MY_PI*sin(MY_PI*y));
    }
    else if (cutofftype==3) {    
      double y3 = y2*y;
      double y4 = y2*y2;
      fcut = (y*(y*(20*y - 70) + 84) - 35)*y4 + 1;
      dfcut = 140*y3*(y - 1)*(y - 1)*(y - 1)/rmax;
    }
    else if (cutofftype==4) {    
      double y4 = y2*y2;
      double y5 = y*y4;
      double y6 = (y - 1)*(y - 1);  
      fcut = (y*(y*((315 - 70*y)*y - 540) + 420) - 126)*y5 + 1;
      dfcut = -630*y4*y6*y6/rmax;    
    }
    
    for (int i=0; i<ngaussianfuncs; i++) {      
      double alpha = gaussianexponents[i];  // gaussian exponents
      int beta = polydegrees[i];       // polynomial degrees
      
      double a = powint(y, beta);       // a = y^beta      
      double g = a * exp(-alpha * y2 ); // g = y^beta * exp(-alpha * y^2 );
      // dg = y^beta*exp(-alpha*y^2)*(- 2*alpha*y + beta/y)
      double dg = a*exp(-alpha*y2)*(- 2*alpha*y + beta/y); 

      int nij = n + N*i;
      rbf[nij] = fcut * g;
      double drbfdr = (dfcut * g + fcut * dg);
      rbfx[nij] = drbfdr*dr1;
      rbfy[nij] = drbfdr*dr2;
      rbfz[nij] = drbfdr*dr3;
    }
  }
}

void RBPOD::podradialbasis(double *rbf, double *rbfx, double *rbfy, double *rbfz, double *rij, double *temp, int N)
{    
  double *rbft = &temp[0]; // Nj*ns
  double *rbfxt = &temp[N*ns]; // Nj*ns
  double *rbfyt = &temp[2*N*ns]; // Nj*ns
  double *rbfzt = &temp[3*N*ns]; // Nj*ns
 
  if (ngaussianfuncs==0)
    radialbasis(rbft, rbfxt, rbfyt, rbfzt, rij, N);
  else
    gaussianbasis(rbft, rbfxt, rbfyt, rbfzt, rij, N);
      
  char chn = 'N';
  double alpha = 1.0, beta = 0.0;
  DGEMM(&chn, &chn, &N, &nrbfmax, &ns, &alpha, rbft, &N, Phi, &ns, &beta, rbf, &N);
  DGEMM(&chn, &chn, &N, &nrbfmax, &ns, &alpha, rbfxt, &N, Phi, &ns, &beta, rbfx, &N);
  DGEMM(&chn, &chn, &N, &nrbfmax, &ns, &alpha, rbfyt, &N, Phi, &ns, &beta, rbfy, &N);
  DGEMM(&chn, &chn, &N, &nrbfmax, &ns, &alpha, rbfzt, &N, Phi, &ns, &beta, rbfz, &N);
}

/**
 * @brief Compute the radial basis function (RBF) for each atom.
 *
 * @param rbf Pointer to the output array for the RBF.
 * @param xij Pointer to the array of distances between each pair of atoms.
 * @param N Number of points in the interval [rin, rcut].
 */
void RBPOD::snapshots(double *rbf, double *xij, int N)
{
  // Compute the maximum distance between two atoms
  double rmax = rcut-rin;

  // Loop over all atoms
  for (int n=0; n<N; n++) {
    double dij = xij[n];

    // Compute the distance between two atoms
    double r = dij - rin;

    // Compute the normalized distance
    double y = r/rmax;
    double y2 = y*y;
    
    double fcut;
    if (cutofftype==0) {
      fcut = 1;      
    }
    else if (cutofftype==1) {
      double y3 = 1.0 - y2*y;
      double y4 = y3*y3 + 1e-6;
      double y5 = sqrt(y4);
      double y6 = exp(-1.0/y5);

      // Calculate the final cutoff function as y6/exp(-1)
      fcut = y6/exp(-1.0);
    }
    else if (cutofftype==2) {
      fcut = 0.5*cos(MY_PI*y) + 0.5;
    }
    else if (cutofftype==3) {    
      double y4 = y2*y2;
      fcut = (y*(y*(20*y - 70) + 84) - 35)*y4 + 1;
    }
    else if (cutofftype==4) {    
      double y4 = y2*y2;
      double y5 = y*y4;
      fcut = (y*(y*((315 - 70*y)*y - 540) + 420) - 126)*y5 + 1;
    }

    // Loop over all Bessel parameters
    for (int j=0; j<nbesselpars; j++) {
      double alpha = besselparams[j];
      if (fabs(alpha) <= 1.0e-6) alpha = 1e-3;
      double x =  (1.0 - exp(-alpha*r/rmax))/(1.0-exp(-alpha));

      // Loop over all Bessel degrees
      for (int i=0; i<besseldegree; i++) {
        double a = (i+1)*MY_PI;
        double b = (sqrt(2.0/(rmax))/(i+1));
        int nij = n + N*i + N*besseldegree*j;

        // Compute the RBF
        rbf[nij] = b*fcut*sin(a*x)/r;
      }
    }

    // Loop over all polynomial degrees of the radial inverse functions
    for (int i=0; i<inversedegree; i++) {
      int p = besseldegree*nbesselpars + i;
      int nij = n + N*p;
      //double a = pow(dij, (double) (i+1.0));
      double a = powint(dij, i+1);

      // Compute the RBF
      rbf[nij] = fcut/a;
    }
  }
}

void RBPOD::gaussiansnapshots(double *rbf, double *rij, int N)
{
  // Loop over all neighboring atoms
  for (int n=0; n<N; n++) {
    double xij1 = rij[0+3*n];
    double xij2 = rij[1+3*n];
    double xij3 = rij[2+3*n];

    double dij = sqrt(xij1*xij1 + xij2*xij2 + xij3*xij3);
    double r = dij - rin;
    double y = r/rmax;
    double y2 = y*y;

    double fcut;
    if (cutofftype==0) {
      fcut = 1;      
    }
    else if (cutofftype==1) {
      double y3 = 1.0 - y2*y;
      double y4 = y3*y3 + 1e-6;
      double y5 = sqrt(y4);
      double y6 = exp(-1.0/y5);

      // Calculate the final cutoff function as y6/exp(-1)
      fcut = y6/exp(-1.0);
    }
    else if (cutofftype==2) {
      fcut = 0.5*cos(MY_PI*y) + 0.5;
    }
    else if (cutofftype==3) {    
      double y4 = y2*y2;
      fcut = (y*(y*(20*y - 70) + 84) - 35)*y4 + 1;
    }
    else if (cutofftype==4) {    
      double y4 = y2*y2;
      double y5 = y*y4;
      fcut = (y*(y*((315 - 70*y)*y - 540) + 420) - 126)*y5 + 1;
    }
    
    for (int i=0; i<ngaussianfuncs; i++) {      
      double alpha = gaussianexponents[i];  // gaussian exponents
      int beta = polydegrees[i];            // polynomial degrees 
      
      double a = powint(y, beta);       // a = y^beta      
      double g = a * exp(-alpha * y2 ); // g = y^beta * exp(-alpha * y^2 );

      int nij = n + N*i;
      rbf[nij] = fcut * g;
    }
  }
}

/**
 * @brief Perform eigenvalue decomposition of the snapshots matrix S and return the eigenvectors and eigenvalues.
 *
 * @param Phi Pointer to the output array for the eigenvectors.
 * @param Lambda Pointer to the output array for the eigenvalues.
 * @param N Number of points in the interval [rin, rcut].
 */
void RBPOD::eigenvaluedecomposition(int N)
{
  int ns = besseldegree*nbesselpars + inversedegree;

  double *xij;
  double *S;
  double *Q;
  double *A;
  double *work;
  double *b;

  memory->create(xij, N, "rbpod:xij");
  memory->create(S, N*ns, "rbpod:S");
  memory->create(Q, N*ns, "rbpod:Q");
  memory->create(A, ns*ns, "rbpod:A");
  memory->create(work, ns*ns, "rbpod:work");
  memory->create(b, ns, "rbpod:ns");

  // Generate the xij array
  for (int i=0; i<N; i++)
    xij[i] = (rin+1e-6) + (rcut-rin-1e-6)*(i*1.0/(N-1));

  // Compute the snapshots matrix S
  if (ngaussianfuncs==0)
    snapshots(S, xij, N);
  else
    gaussiansnapshots(S, xij, N);
            
  // Compute the matrix A = S^T * S
  char chn = 'N';
  char cht = 'T';
  double alpha = 1.0, beta = 0.0;
  DGEMM(&cht, &chn, &ns, &ns, &N, &alpha, S, &N, S, &N, &beta, A, &ns);

  // Normalize the matrix A by dividing by N
  for (int i=0; i<ns*ns; i++)
    A[i] = A[i]*(1.0/N);

  // Compute the eigenvectors and eigenvalues of A
  int lwork = ns * ns;  // the length of the array work, lwork >= max(1,3*N-1)
  int info = 1;     // = 0:  successful exit
  //double work[ns*ns];
  char chv = 'V';
  char chu = 'U';
  DSYEV(&chv, &chu, &ns, A, &ns, b, work, &lwork, &info);

  // Order eigenvalues and eigenvectors from largest to smallest
  for (int j=0; j<ns; j++)
    for (int i=0; i<ns; i++)
      Phi[i + ns*(ns-j-1)] = A[i + ns*j];

  for (int i=0; i<ns; i++)
    Lambda[(ns-i-1)] = b[i];

  // Compute the matrix Q = S * Phi
  DGEMM(&chn, &chn, &N, &ns, &ns, &alpha, S, &N, Phi, &ns, &beta, Q, &N);

  // Compute the area of each snapshot and normalize the eigenvectors
  for (int i=0; i<(N-1); i++)
    xij[i] = xij[i+1] - xij[i];
  double area;
  for (int m=0; m<ns; m++) {
    area = 0.0;
    for (int i=0; i<(N-1); i++)
      area += 0.5*xij[i]*(Q[i + N*m]*Q[i + N*m] + Q[i+1 + N*m]*Q[i+1 + N*m]);
    for (int i=0; i<ns; i++)
      Phi[i + ns*m] = Phi[i + ns*m]/sqrt(area);
  }

  // Enforce consistent signs for the eigenvectors
  for (int m=0; m<ns; m++) {
    if (Phi[m + ns*m] < 0.0) {
      for (int i=0; i<ns; i++)
        Phi[i + ns*m] = -Phi[i + ns*m];
    }
  }

  // Free temporary arrays
  memory->destroy(xij);
  memory->destroy(S);
  memory->destroy(A);
  memory->destroy(work);
  memory->destroy(b);
  memory->destroy(Q);
}

/**
 * @brief Initialize the two-body coefficients.
 *
 * @param None
 */
void RBPOD::init2body()
{
  // Set the degree of the Bessel function and the inverse distance function
  pdegree[0] = besseldegree;
  pdegree[1] = inversedegree;

  // Compute the total number of snapshots
  ns = nbesselpars * pdegree[0] + pdegree[1];

  // Allocate memory for the eigenvectors and eigenvalues
  memory->create(Phi, ns * ns, "Phi");
  memory->create(Lambda, ns, "Lambda");

  // Perform eigenvalue decomposition of the snapshots matrix S and store the eigenvectors and eigenvalues
  eigenvaluedecomposition(2000);
}

void RBPOD::xchenodes(double* xi, int p) 
{
  int n = p + 1;
  for (int k = 1; k <= n; k++) {
    xi[k - 1] = -cos((2 * k - 1) * M_PI / (2 * n)) / cos(M_PI / (2 * n));
  }    
}

void RBPOD::ref2dom(double* y, double* xi, double ymin, double ymax, int n) 
{
  double dy = 0.5 * (ymax - ymin);
  for (int i=0; i < n; i++) 
    y[i] = ymin + dy * (xi[i] + 1);    
}

void RBPOD::dom2ref(double* xi, double* y, double ymin, double ymax, int n) 
{
  double dy = 2.0/(ymax - ymin);
  for (int i=0; i < n; i++) 
    xi[i] = dy * (y[i] - ymin) - 1;    
}

void RBPOD::legendrepolynomials(double* poly, double* xi, int p, int n) 
{
  int p1 = p + 1;
  for (int j = 0; j < n; j++) {
    poly[j] = 1.0;        
    poly[j + n] = xi[j];        
  }
  for (int i = 2; i <= p; i++) {
      double b = (i - 1.0) / i;
      double a = 1.0 + b;
      for (int j = 0; j < n; j++) {
          poly[j + n*i] = a * xi[j] * poly[j + n*(i - 1)] - b * poly[j + n*(i - 2)];
      }
  }
}

int RBPOD::tensorpolyfit(double* c, double* xi, double* A, double* y, double* f, int* ipiv, double ymin, double ymax, int p, int n, int nrhs)
{
    dom2ref(xi, y, ymin, ymax, n);               
    legendrepolynomials(A, xi, p, n);    
    for (int i = 0; i < n*nrhs; i++) c[i] = f[i];
    
    int info;    
    DGESV(&n, &nrhs, A, &n, ipiv, c, &n, &info);   
    
    return info;
}

void RBPOD::tensorpolyeval(double* f, double* c, double* xi, double* A, double* y, double ymin, double ymax, int p, int n, int nrhs)
{
    dom2ref(xi, y, ymin, ymax, n);               
    legendrepolynomials(A, xi, p, n);   
    
    char chn = 'N';  
    double alpha = 1.0, beta = 0.0;
    DGEMM(&chn, &chn, &n, &nrhs, &n, &alpha, A, &n, c, &n, &beta, f, &n);    
}

void RBPOD::radialbasis(double *rbf, double *drbfdr, double *rij, int N)
{
  // Loop over all neighboring atoms
  for (int n=0; n<N; n++) {
    double dij = rij[n];
    double r = dij - rin;
    double y = r/rmax;
    double y2 = y*y;

    double fcut, dfcut;
    if (cutofftype==0) {
      fcut = 1;
      dfcut = 0;
    }
    else if (cutofftype==1) {
      double y3 = 1.0 - y2*y;
      double y4 = y3*y3 + 1e-6;
      double y5 = sqrt(y4);
      double y6 = exp(-1.0/y5);
      double y7 = y4*sqrt(y4);

      // Calculate the final cutoff function as y6/exp(-1)
      fcut = y6/exp(-1.0);

      // Calculate the derivative of the final cutoff function
      dfcut = ((3.0/(rmax*exp(-1.0)))*(y2)*y6*(y*y2 - 1.0))/y7;
    }
    else if (cutofftype==2) {
      fcut = 0.5*cos(MY_PI*y) + 0.5;
      dfcut = -0.5*(MY_PI*sin(MY_PI*y));
    }
    else if (cutofftype==3) {    
      double y3 = y2*y;
      double y4 = y2*y2;
      fcut = (y*(y*(20*y - 70) + 84) - 35)*y4 + 1;
      dfcut = 140*y3*(y - 1)*(y - 1)*(y - 1)/rmax;
    }
    else if (cutofftype==4) {    
      double y4 = y2*y2;
      double y5 = y*y4;
      double y6 = (y - 1)*(y - 1);  
      fcut = (y*(y*((315 - 70*y)*y - 540) + 420) - 126)*y5 + 1;
      dfcut = -630*y4*y6*y6/rmax;    
    }
    
    // Calculate fcut/r, fcut/r^2, and dfcut/r
    double f1 = fcut/r;
    double f2 = f1/r;
    double df1 = dfcut/r;

    double alpha = besselparams[0];
    double t1 = (1.0-exp(-alpha));
    double t2 = exp(-alpha*r/rmax);
    double x0 =  (1.0 - t2)/t1;
    double dx0 = (alpha/rmax)*t2/t1;

    if (nbesselpars==1) {
      for (int i=0; i<besseldegree; i++) {
        double a = (i+1)*MY_PI;
        double b = (sqrt(2.0/(rmax))/(i+1));
        double af1 = a*f1;

        double sinax = sin(a*x0);
        int nij = n + N*i;

        rbf[nij] = b*f1*sinax;
        drbfdr[nij] = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);
      }
    }
    else if (nbesselpars==2) {
      alpha = besselparams[1];
      t1 = (1.0-exp(-alpha));
      t2 = exp(-alpha*r/rmax);
      double x1 =  (1.0 - t2)/t1;
      double dx1 = (alpha/rmax)*t2/t1;
      for (int i=0; i<besseldegree; i++) {
        double a = (i+1)*MY_PI;
        double b = (sqrt(2.0/(rmax))/(i+1));
        double af1 = a*f1;

        double sinax = sin(a*x0);
        int nij = n + N*i;

        rbf[nij] = b*f1*sinax;
        drbfdr[nij] = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);

        sinax = sin(a*x1);
        nij = n + N*i + N*besseldegree*1;
        rbf[nij] = b*f1*sinax;
        drbfdr[nij] = b*(df1*sinax - f2*sinax + af1*cos(a*x1)*dx1);        
      }
    }
    else if (nbesselpars==3) {
      alpha = besselparams[1];
      t1 = (1.0-exp(-alpha));
      t2 = exp(-alpha*r/rmax);
      double x1 =  (1.0 - t2)/t1;
      double dx1 = (alpha/rmax)*t2/t1;

      alpha = besselparams[2];
      t1 = (1.0-exp(-alpha));
      t2 = exp(-alpha*r/rmax);
      double x2 =  (1.0 - t2)/t1;
      double dx2 = (alpha/rmax)*t2/t1;
      for (int i=0; i<besseldegree; i++) {
        double a = (i+1)*MY_PI;
        double b = (sqrt(2.0/(rmax))/(i+1));
        double af1 = a*f1;

        double sinax = sin(a*x0);
        int nij = n + N*i;
        rbf[nij] = b*f1*sinax;
        drbfdr[nij] = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);
        
        sinax = sin(a*x1);
        nij = n + N*i + N*besseldegree*1;
        rbf[nij] = b*f1*sinax;
        drbfdr[nij] = b*(df1*sinax - f2*sinax + af1*cos(a*x1)*dx1);

        sinax = sin(a*x2);
        nij = n + N*i + N*besseldegree*2;
        rbf[nij] = b*f1*sinax;
        drbfdr[nij] = b*(df1*sinax - f2*sinax + af1*cos(a*x2)*dx2);        
      }
    }

    // Calculate fcut/dij and dfcut/dij
    f1 = fcut/dij;
    for (int i=0; i<inversedegree; i++) {
      int p = besseldegree*nbesselpars + i;
      int nij = n + N*p;
      double a = powint(dij, i+1);

      rbf[nij] = fcut/a;
      drbfdr[nij] = (dfcut - (i+1.0)*f1)/a;      
    }
  }
}

void RBPOD::gaussianbasis(double *rbf, double *drbfdr, double *rij, int N)
{
  // Loop over all neighboring atoms
  for (int n=0; n<N; n++) {
    double dij = rij[n];
    double r = dij - rin;
    double y = r/rmax;
    double y2 = y*y;

    double fcut, dfcut;
    if (cutofftype==0) {
      fcut = 1;
      dfcut = 0;
    }
    else if (cutofftype==1) {
      double y3 = 1.0 - y2*y;
      double y4 = y3*y3 + 1e-6;
      double y5 = sqrt(y4);
      double y6 = exp(-1.0/y5);
      double y7 = y4*sqrt(y4);

      // Calculate the final cutoff function as y6/exp(-1)
      fcut = y6/exp(-1.0);

      // Calculate the derivative of the final cutoff function
      dfcut = ((3.0/(rmax*exp(-1.0)))*(y2)*y6*(y*y2 - 1.0))/y7;
    }
    else if (cutofftype==2) {
      fcut = 0.5*cos(MY_PI*y) + 0.5;
      dfcut = -0.5*(MY_PI*sin(MY_PI*y));
    }
    else if (cutofftype==3) {    
      double y3 = y2*y;
      double y4 = y2*y2;
      fcut = (y*(y*(20*y - 70) + 84) - 35)*y4 + 1;
      dfcut = 140*y3*(y - 1)*(y - 1)*(y - 1)/rmax;
    }
    else if (cutofftype==4) {    
      double y4 = y2*y2;
      double y5 = y*y4;
      double y6 = (y - 1)*(y - 1);  
      fcut = (y*(y*((315 - 70*y)*y - 540) + 420) - 126)*y5 + 1;
      dfcut = -630*y4*y6*y6/rmax;    
    }
    
    for (int i=0; i<ngaussianfuncs; i++) {      
      double alpha = gaussianexponents[i];  // gaussian exponents
      int beta = polydegrees[i];       // polynomial degrees
      
      double a = powint(y, beta);       // a = y^beta      
      double g = a * exp(-alpha * y2 ); // g = y^beta * exp(-alpha * y^2 );
      // dg = y^beta*exp(-alpha*y^2)*(- 2*alpha*y + beta/y)
      double dg = a*exp(-alpha*y2)*(- 2*alpha*y + beta/y); 

      int nij = n + N*i;
      rbf[nij] = fcut * g;
      drbfdr[nij] = (dfcut * g + fcut * dg);      
    }
  }
}


void RBPOD::podradialbasis(double *rbf, double *drbfdr, double *rij, double *temp, int N)
{  
  double *rbft = &temp[0]; // Nj*ns
  double *drbft = &temp[N*ns]; // Nj*ns
  
  if (ngaussianfuncs==0)
    radialbasis(rbft, drbft, rij, N);
  else
    gaussianbasis(rbft, drbft, rij, N);
      
  char chn = 'N';
  double alpha = 1.0, beta = 0.0;
  DGEMM(&chn, &chn, &N, &nrbfmax, &ns, &alpha, rbft, &N, Phi, &ns, &beta, rbf, &N);
  DGEMM(&chn, &chn, &N, &nrbfmax, &ns, &alpha, drbft, &N, Phi, &ns, &beta, drbfdr, &N);  
}

double maxerror(double *a, double *b, int n)
{
  double e = 0.0;
  for (int i=0; i<n; i++) {
    double d = fabs(a[i] - b[i]);
    e = (e > d) ? e : d;
  }
  return e;
}

void RBPOD::femapproximation(int nelem, int p)
{    
  int n = p + 1;
  int ntest = 2*n;  
  int info;
  int lwork = n*n;
  char chn = 'N';  
  double alpha = 1.0, beta = 0.0;
    
  int ipiv[10];  
  double *Ainv;
  double *work;
  double *xi;
  double *y;
  double *A;  
  double *rbf;
  double *drbfdr;  
  double *temp;  
  double *xitest;
  double *ytest;
  double *Atest;  
  double *rbftest;
  double *drbfdrtest;  
  double *temptest;    
  
  memory->create(crbf, n*nrbfmax*nelem, "rbpod:crbf");
  memory->create(drbf, n*nrbfmax*nelem, "rbpod:drbf");  
    
  memory->create(A, n*n, "rbpod:A");
  memory->create(Ainv, n*n, "rbpod:Ainv");
  memory->create(work, n*n, "rbpod:work");
  memory->create(relem, nelem+1, "rbpod:elem");  
  
  memory->create(xi, n, "rbpod:xi");
  memory->create(y, n, "rbpod:y");  
  memory->create(rbf, n*nrbfmax, "rbpod:rbf");
  memory->create(drbfdr, n*nrbfmax, "rbpod:rbfx");  
  memory->create(temp, 2*n*ns, "rbpod:rbfz");  
  
  memory->create(xitest, ntest, "rbpod:xi");
  memory->create(ytest, ntest, "rbpod:y");
  memory->create(Atest, ntest*n, "rbpod:A");
  memory->create(rbftest, ntest*nrbfmax, "rbpod:rbf");
  memory->create(drbfdrtest, ntest*nrbfmax, "rbpod:rbfx");  
  memory->create(temptest, 2*ntest*ns, "rbpod:rbfz");  
  
  xchenodes(xi, p);
  legendrepolynomials(A, xi, p, n); 
  for (int i=0; i<n*n; i++) Ainv[i] = A[i];
      
  DGETRF(&n,&n,Ainv,&n,ipiv,&info);
  DGETRI(&n,Ainv,&n,ipiv,work,&lwork,&info);
    
  xchenodes(xitest, ntest-1);
  legendrepolynomials(Atest, xitest, p, ntest);   
  
//   for (int i=0; i<n; i++) {
//     for (int j=0; j<n; j++)
//       printf("%g ", A[i + n*j]);
//     printf("\n");    
//   }
//   
//   for (int i=0; i<n; i++) {
//     for (int j=0; j<n; j++)
//       printf("%g ", Ainv[i + n*j]);
//     printf("\n");    
//   }
  
  for (int i=0; i<nelem+1; i++)
    relem[i] = rin+1e-3 + (rcut-rin-1e-3)*(i*1.0/nelem);
  
//   double a = rin+1e-3; 
//   double b = rcut;
//   double c = 1.0;
//   for (int i=0; i<nelem+1; i++)
//     relem[i] = a + (b-a)*(exp(c*(relem[i]-a)/(b-a))-1)/(exp(c)-1);
  
  double err[2] = {0, 0};
  for (int i=0; i<nelem; i++) {
    ref2dom(y, xi, relem[i], relem[i+1], n); 
    podradialbasis(rbf, drbfdr, y, temp, n);        
    DGEMM(&chn, &chn, &n, &nrbfmax, &n, &alpha, Ainv, &n, rbf, &n, &beta, &crbf[n*nrbfmax*i], &n);    
    DGEMM(&chn, &chn, &n, &nrbfmax, &n, &alpha, Ainv, &n, drbfdr, &n, &beta, &drbf[n*nrbfmax*i], &n);        
    
    ref2dom(ytest, xitest, relem[i], relem[i+1], ntest); 
    podradialbasis(rbftest, drbfdrtest, ytest, temptest, ntest);            
    DGEMM(&chn, &chn, &ntest, &nrbfmax, &n, &alpha, Atest, &ntest, &crbf[n*nrbfmax*i], &n, &beta, temptest, &ntest);        
    DGEMM(&chn, &chn, &ntest, &nrbfmax, &n, &alpha, Atest, &ntest, &drbf[n*nrbfmax*i], &n, &beta, &temptest[ntest*nrbfmax], &ntest);        
    
//     if (i==nelem-1) {
//       printf("Element: %d, [%g  %g] \n", i, relem[i], relem[i+1]); 
//       for (int k=0; k<ntest; k++) {
//         for (int j=0; j<nrbfmax; j++)
//           printf("%g ", rbftest[k + ntest*j]);
//         printf("\n");    
//       } 
//       
//       printf("\n");          
//       for (int k=0; k<n; k++) {
//         for (int j=0; j<nrbfmax; j++)
//           printf("%g ", crbf[k + n*j + n*nrbfmax*i]);
//         printf("\n");    
//       } 
//     }
 
    double e0 = 0, e1 = 0;
    for (int j=0; j<ntest*nrbfmax; j++) {
      double de = fabs(temptest[j] - rbftest[j]);
      double fe = fabs(temptest[ntest*nrbfmax+j] - drbfdrtest[j]);
      e0 = (e0 > de) ? e0 : de;
      e1 = (e1 > fe) ? e1 : fe;
    }    
    err[0] = (err[0] > e0) ? err[0] : e0;
    err[1] = (err[1] > e1) ? err[1] : e1;
    
//    printf("Element: %d, [%g  %g], %g,  %g \n", i, relem[i], relem[i+1], e0, e1); 
//     for (int k=0; k<ntest; k++) {
//       for (int j=0; j<nrbfmax; j++)
//         printf("%g ", drbfdrtest[k + ntest*j]);
//       printf("\n");    
//     }
//     for (int k=0; k<ntest; k++) {
//       for (int j=0; j<nrbfmax; j++)
//         printf("%g ", temptest[ntest*nrbfmax + k + ntest*j]);
//       printf("\n");    
//     }    
  }  
  printf("Maximum absolute errors: |rbf - rbffem|_infty = %g,   |drbfdr - drbfdrfem|_infty = %g\n", err[0], err[1]);
    
  int K = 10000;
  int N = K*nrbfmax;
  double *rij;
  double *rbf1;
  double *rbfx1;
  double *rbfy1;
  double *rbfz1;
  double *rbf2;
  double *rbfx2;
  double *rbfy2;
  double *rbfz2;
  double *tmp;
  
  memory->create(rij, 3*K, "rbpod:rij");
  memory->create(rbf1, N, "rbpod:rbf1");
  memory->create(rbfx1, N, "rbpod:rbfx1");
  memory->create(rbfy1, N, "rbpod:rbfy1");
  memory->create(rbfz1, N, "rbpod:rbfz1");
  memory->create(rbf2, N, "rbpod:rbf2");
  memory->create(rbfx2, N, "rbpod:rbfx2");
  memory->create(rbfy2, N, "rbpod:rbfy2");
  memory->create(rbfz2, N, "rbpod:rbfz2");  
  memory->create(tmp, 4*K*ns, "rbpod:tmp");
  
  for(int i=0; i<K; i++) {    
    // Generate random values for the vector components
    double x = -1.0 + (2.0) * ((double)rand() / RAND_MAX);
    double y = -1.0 + (2.0) * ((double)rand() / RAND_MAX);
    double z = -1.0 + (2.0) * ((double)rand() / RAND_MAX);
    double norm = sqrt(x * x + y * y + z * z);
    double target_norm = rin+1e-3 + (rcut-rin-1e-3) * ((double)rand() / RAND_MAX);
    rij[0 + 3*i] = x * (target_norm / norm);
    rij[1 + 3*i] = y * (target_norm / norm);
    rij[2 + 3*i] = z * (target_norm / norm);    
//     double normrij = sqrt(rij[0 + 3*i]*rij[0 + 3*i] + rij[1 + 3*i]*rij[1 + 3*i] + rij[2 + 3*i]*rij[2 + 3*i]);
//     if (i<100) printf("%d :    %g    %g    %g    %g\n", i, rij[0 + 3*i], rij[1 + 3*i], rij[2 + 3*i], normrij);
  }
  
  clock_t start, end;
  double time_pod, time_fem;
    
//  podradialbasis(rbf1, rbfx1, rbfy1, rbfz1, rij, tmp, K);  
//  femradialbasis(rbf2, rbfx2, rbfy2, rbfz2, rij, K);
  start = clock();
  podradialbasis(rbf1, rbfx1, rbfy1, rbfz1, rij, tmp, K);
  end = clock();
  time_pod = ((double)(end - start)) / CLOCKS_PER_SEC;
    
  start = clock();
  femradialbasis(rbf2, rbfx2, rbfy2, rbfz2, rij, K);
  end = clock();
  time_fem = ((double)(end - start)) / CLOCKS_PER_SEC;
    
  printf("Maximum absolute error: |rbf - rbffem|_infty = %g\n", maxerror(rbf1, rbf2, K*nrbfmax));
  printf("Maximum absolute error: |rbfx - rbfxfem|_infty = %g\n", maxerror(rbfx1, rbfx2, K*nrbfmax));
  printf("Maximum absolute error: |rbfy - rbfyfem|_infty = %g\n", maxerror(rbfy1, rbfy2, K*nrbfmax));
  printf("Maximum absolute error: |rbfz - rbfzfem|_infty = %g\n", maxerror(rbfz1, rbfz2, K*nrbfmax));
  
  // Print the results
  printf("Time taken by podradialbasis: %f seconds\n", time_pod);
  printf("Time taken by femradialbasis: %f seconds\n", time_fem);
    
  memory->destroy(xi);
  memory->destroy(y);
  memory->destroy(A);
  memory->destroy(Ainv);
  memory->destroy(work);
  memory->destroy(rbf);
  memory->destroy(drbfdr);  
  memory->destroy(temp);
  
  memory->destroy(xitest);
  memory->destroy(ytest);
  memory->destroy(Atest);
  memory->destroy(rbftest);
  memory->destroy(drbfdrtest);  
  memory->destroy(temptest);
  
  memory->destroy(rij);
  memory->destroy(rbf1);
  memory->destroy(rbfx1);
  memory->destroy(rbfy1);
  memory->destroy(rbfz1);  
  memory->destroy(rbfx2);
  memory->destroy(rbfy2);
  memory->destroy(rbfz2);  
  memory->destroy(tmp);
}

void RBPOD::femradialbasis(double *rbf, double *rbfx, double *rbfy, double *rbfz, double *rij, int N)
{
  int p1 = nfemdegree + 1;
  double dr = (rcut-rin-1e-3)/nfemelem;
  
  for (int n=0; n<N; n++) {
    double xij1 = rij[0+3*n];
    double xij2 = rij[1+3*n];
    double xij3 = rij[2+3*n];

    double dij = sqrt(xij1*xij1 + xij2*xij2 + xij3*xij3);
    double dr1 = xij1/dij;
    double dr2 = xij2/dij;
    double dr3 = xij3/dij;

    int i = (dij-rin-1e-3)/dr;        
    if (i > (nfemelem-1)) i = nfemelem-1;        
    
    double ymin = relem[i];
    double ymax = relem[i+1];                
    double dy = 2.0/(ymax - ymin);  
    double xi = dy * (dij  - ymin) - 1;    
    double xi1 = xi*xi;
    double xi2 = 1.5*xi1 - 0.5;
    double xi3 = (2.5*xi1 - 1.5)*xi;
            
    double *c = &crbf[p1*nrbfmax*i];    
    double *d = &drbf[p1*nrbfmax*i];                
    for (int j=0; j<nrbfmax; j++) {
      int m = p1*j;
      rbf[n + N*j] = c[0+m] + c[1+m]*xi + c[2+m]*xi2 + c[3+m]*xi3;
      double drbfdr = d[0+m] + d[1+m]*xi + d[2+m]*xi2 + d[3+m]*xi3;
      rbfx[n + N*j] = drbfdr*dr1;
      rbfy[n + N*j] = drbfdr*dr2;
      rbfz[n + N*j] = drbfdr*dr3;      
    }    
  }
}

void RBPOD::femradialbasis(double *rbf, double *drbfx, double *rij, int N)
{
  int p1 = nfemdegree + 1;
  double dr = (rcut-rin-1e-3)/nfemelem;
  
  for (int n=0; n<N; n++) {
    double xij1 = rij[0+3*n];
    double xij2 = rij[1+3*n];
    double xij3 = rij[2+3*n];

    double dij = sqrt(xij1*xij1 + xij2*xij2 + xij3*xij3);
    double dr1 = xij1/dij;
    double dr2 = xij2/dij;
    double dr3 = xij3/dij;

    int i = (dij-rin-1e-3)/dr;        
    if (i > (nfemelem-1)) i = nfemelem-1;        
                
    double ymin = relem[i];
    double ymax = relem[i+1];                
    double dy = 2.0/(ymax - ymin);  
    double xi = dy * (dij  - ymin) - 1;    
    double xi1 = xi*xi;
    double xi2 = 1.5*xi1 - 0.5;
    double xi3 = (2.5*xi1 - 1.5)*xi;
            
    double *c = &crbf[p1*nrbfmax*i];    
    double *d = &drbf[p1*nrbfmax*i];                
    for (int j=0; j<nrbfmax; j++) {
      int m = p1*j;
      rbf[n + N*j] = c[0+m] + c[1+m]*xi + c[2+m]*xi2 + c[3+m]*xi3;
      double drbfdr = d[0+m] + d[1+m]*xi + d[2+m]*xi2 + d[3+m]*xi3;
      drbfx[0 + 3*n + 3*N*j] = drbfdr*dr1;
      drbfx[1 + 3*n + 3*N*j] = drbfdr*dr2;
      drbfx[2 + 3*n + 3*N*j] = drbfdr*dr3;      
//       if (fabs(dij-rcut)<1e-6) {
//         printf("%d   %g    %g    %g    %g    %g    %g    %g    %g\n", i, xij1, xij2, xij3, c[0+m], c[2+m], c[2+m], c[3+m], rbf[n + N*j]);
//       }    
    }            
  }
}

void RBPOD::femradialbasis(double *rbf, double *rij, int N)
{
  int p1 = nfemdegree + 1;
  double dr = (rcut-rin-1e-3)/nfemelem;
  
  for (int n=0; n<N; n++) {
    double xij1 = rij[0+3*n];
    double xij2 = rij[1+3*n];
    double xij3 = rij[2+3*n];

    double dij = sqrt(xij1*xij1 + xij2*xij2 + xij3*xij3);
    double dr1 = xij1/dij;
    double dr2 = xij2/dij;
    double dr3 = xij3/dij;

    int i = (dij-rin-1e-3)/dr;        
    if (i > (nfemelem-1)) i = nfemelem-1;        
    
    double ymin = relem[i];
    double ymax = relem[i+1];                
    double dy = 2.0/(ymax - ymin);  
    double xi = dy * (dij  - ymin) - 1;    
    double xi1 = xi*xi;
    double xi2 = 1.5*xi1 - 0.5;
    double xi3 = (2.5*xi1 - 1.5)*xi;
            
    double *c = &crbf[p1*nrbfmax*i];        
    for (int j=0; j<nrbfmax; j++) {
      int m = p1*j;
      rbf[n + N*j] = c[0+m] + c[1+m]*xi + c[2+m]*xi2 + c[3+m]*xi3;      
    }    
  }
}
