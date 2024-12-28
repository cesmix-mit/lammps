/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_RBPOD_H
#define LMP_RBPOD_H

#include "pointers.h"

#define DDOT ddot_
#define DGEMV dgemv_
#define DGEMM dgemm_
#define DGETRF dgetrf_
#define DGETRI dgetri_
#define DSYEV dsyev_
#define DPOSV dposv_
#define DGESV dgesv_

extern "C" {
double DDOT(int *, double *, int *, double *, int *);
void DGEMV(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *,
           int *);
void DGEMM(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *,
           double *, double *, int *);
void DGETRF(int *, int *, double *, int *, int *, int *);
void DGETRI(int *, double *, int *, int *, double *, int *, int *);
void DSYEV(char *, char *, int *, double *, int *, double *, double *, int *, int *);
void DPOSV(char *, int *, int *, double *, int *, double *, int *, int *);
void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
}

namespace LAMMPS_NS {

class RBPOD : protected Pointers {
 private:  
  void init2body();
  void eigenvaluedecomposition(int N);
  void snapshots(double *rbf, double *xij, int N);
  void gaussiansnapshots(double *rbf, double *rij, int N);  
  void radialbasis(double *rbf, double *drbf, double *rij, int N);      
  void gaussianbasis(double *rbf, double *drbf, double *rij, int N);  
  void podradialbasis(double *rbf, double *drbf, double *rij, double *temp, int N);    
  void femapproximation(int nelem, int p);
  
 public:
  double rin;
  double rcut;
  double rmax;
  int cutofftype;
  int besseldegree;
  int inversedegree;
  int pdegree[2];
  int nbesselpars;
  int ngaussianfuncs;
  double gaussianexponents[100];
  int polydegrees[100];
  double besselparams[3];
  double *Phi;       // eigenvectors
  double *Lambda;    // eigenvalues  
  int ns;            // number of snapshots for radial basis functions
  int nrbfmax;       // number of radial basis functions

  int nfemelem;
  int nfemdegree;  
  double *relem;    
  double *crbf;
  double *drbf;  
          
  RBPOD(LAMMPS *, const std::string &pod_file);

  RBPOD(LAMMPS *lmp) : Pointers(lmp){};
    
  void read_pod_file(std::string pod_file);
  void femradialbasis(double *rbf, double *rbfx, double *rbfy, double *rbfz, double *rij, int N);
  void femradialbasis(double *rbf, double *drbfx, double *rij, int N);
  void femdrbfdr(double *rbf, double *drbfdr, double *rij, int N);
  void fem1drbf(double *rbf, double *drbfdr, double *x, int nrbf, int N);
  
  void xchenodes(double* xi, int p); 
  void ref2dom(double* y, double* xi, double ymin, double ymax, int n); 
  void dom2ref(double* xi, double* y, double ymin, double ymax, int n); 
  void legendrepolynomials(double* poly, double* xi, int p, int n);
  void tensorpolynomials(double* A, double* x, int p, int n, int dim);
  int tensorpolyfit(double* c, double* xi, double* A, double* y, double* f, int* ipiv, double ymin, double ymax, int p, int n, int nrhs);
  void tensorpolyeval(double* f, double* c, double* xi, double* A, double* y, double ymin, double ymax, int p, int n, int nrhs);
    
  ~RBPOD() override;
};

}    // namespace LAMMPS_NS

#endif
