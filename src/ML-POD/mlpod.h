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

#ifndef LMP_MLPOD_H
#define LMP_MLPOD_H

#include "pointers.h"

#define DDOT ddot_
#define DGEMV dgemv_
#define DGEMM dgemm_
#define DGETRF dgetrf_
#define DGETRI dgetri_
#define DSYEV dsyev_
#define DPOSV dposv_

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
}

namespace LAMMPS_NS {

class MLPOD : protected Pointers {

 private:
  // functions for reading input files

  void read_pod(const std::string &pod_file);
  void read_coeff_file(const std::string &coeff_file);

  // functions for calculating/collating POD descriptors/coefficients for energies

  void pod1body(double *eatom, double *fatom, int *atomtype, int nelements, int natom);
  void podtally2b(double *eatom, double *fatom, double *eij, double *fij, int *ai, int *aj, int *ti,
                  int *tj, int *elemindex, int nelements, int nbf, int natom, int N);
  void pod3body(double *eatom, double *fatom, double *rij, double *e2ij, double *f2ij,
                double *tmpmem, int *elemindex, int *pairnumsum, int *ai, int *aj, int *ti, int *tj,
                int nrbf, int nabf, int nelements, int natom, int Nij);
  void poddesc(double *eatom1, double *fatom1, double *eatom2, double *fatom2, double *eatom3,
               double *fatom3, double *rij, double *besselparams, double *tmpmem,
               double rin, double rcut, int *pairnumsum, int *atomtype, int *ai, int *aj, int *ti,
               int *tj, int *elemindex, int *pdegree, int nbesselpars, int nrbf2, int nrbf3,
               int nabf, int nelements, int Nij, int natom);
  
  double pod2body_energyforce(double *fij, double *ei, double *rij, double *rbf, double *drbfdr, 
                      double *coeff2, double *tmpmem, int *elemindex, int *pairnumsum, int *ti, 
                      int *tj, int nelements, int nrbf, int natom, int Nij);   

  double pod3body_energyforce(double *fij, double *ei, double *rij, double *rbf, double *drbfdr, 
                      double *coeff3, double *tmpmem, int *elemindex, int *pairnumsum, int *ti, 
                      int *tj, int nelements, int nrbf, int nabf, int natom, int Nij);
  
 public:
  MLPOD(LAMMPS *, const std::string &pod_file, const std::string &coeff_file);

  MLPOD(LAMMPS *lmp) : Pointers(lmp){};
  ~MLPOD() override;

  struct podstruct {
    podstruct();
    virtual ~podstruct();

    std::vector<std::string> species;
    int twobody[3];
    int threebody[4];
    int fourbody[4];
    int *pbc;
    int *elemindex;

    int nelements;
    int onebody;
    int besseldegree;
    int inversedegree;

    double rin;
    double rcut;
    double *besselparams;    

    // variables declaring number of descriptors and combinations
    int nbesselpars = 3;
    int nc2, nc3, nc4;             // number of chemical  combinations for linear POD potentials
    int nbf1, nbf2, nbf3, nbf4;    // number of basis functions for linear POD potentials
    int nd1, nd2, nd3, nd4;        // number of descriptors for linear POD potentials
    int nrbf3, nabf3, nrbf4, nabf4;
    int nd, nd1234;
  };

  podstruct pod;
  class RBPOD *rbpodptr;

  // environmental variables
  int nClusters;      // number of environment clusters
  int nComponents;    // number of principal components

  double *podcoeffs; // POD potential coefficients
  int npodcoeffs;
          
  int femdegree;
  int nelemrbf;
  int nelemabf;
  int npelem;
  int nfemelem;
  int nfemfuncs;
  int nfemcoeffs;
  double *femcoeffs;
  
  // functions for collecting/collating arrays

  void podMatMul(double *c, double *a, double *b, int r1, int c1, int c2);
  void podArraySetValue(double *y, double a, int n);
  void podArrayCopy(double *y, double *x, int n);
  void podArrayFill(int *output, int start, int length);

  // functions for calculating energy and force descriptors

  void podNeighPairs(double *xij, double *x, int *ai, int *aj, int *ti, int *tj, int *pairlist,
                     int *pairnumsum, int *atomtype, int *alist, int inum, int dim);
  void linear_descriptors(double *gd, double *efatom, double *y, double *tmpmem, int *atomtype,
                          int *alist, int *pairlist, int *pairnum, int *pairnumsum, int *tmpint,
                          int natom, int Nij);
  
  // functions for calculating energies and forces

  void podNeighPairs(double *rij, double *x, int *idxi, int *ai, int *aj, int *ti, int *tj,
                     int *pairnumsum, int *atomtype, int *jlist, int *alist, int inum);

  void linear_descriptors_ij(double *gd, double *eatom, double *rij, double *tmpmem,
                             int *pairnumsum, int *atomtype, int *ai, int *ti, int *tj, int natom,
                             int Nij);

  double pod123body_energyforce(double *fij, double *ei, double *rij, double *podcoeff, 
                       double *tmpmem, int *pairnumsum, int *typeai, int *ti, int *tj, int natom, int Nij);   

  void fempod_energyforce(double *fij, double *ei, double *rij, double *podcoeff, 
                       double *tmpmem, int *idxi, int *numij, int *typeai, int *ti, int *tj, int natom, int Nij);   
  
  void fempod3_energyforce(double *fij, double *ei, double *rij, double *podcoeff, 
                       double *tmpmem, int *idxi, int *numij, int *typeai, 
                       int *ti, int *tj, int natom, int Nij);   
  
  void tallyforce(double *force, double *fij, int *ai, int *aj, int N);
  
  double energyforce_calculation(double *force, double *fij, double *rij, double *podcoeff, double *tmpmem, 
        int *pairnumsum, int *typeai, int *ai, int *aj, int *ti, int *tj, int natom, int Nij); 
  
  double energyforce_calculation(double *force, double *fij, double *rij, double *podcoeff, double *tmpmem, 
        int *idxi, int *numij, int *typeai, int *ai, int *aj, int *ti, int *tj, int natom, int Nij); 
  
  void femrbf(double *rbf, double *drbfdr, double rin, double rcut, int nrbf, int nelem, int p);
  
  void femabf(double *abf, double *dabf, int nabf, int nelem, int p);
  
  void femapproximation3body(double *cphi, double *coeff, double rin, 
        double rcut, int nrbf, int nelemr, int nabf, int nelema, int p);
  
  void polyfit3body(double *coeff3, int memoryallocate);  
};

}    // namespace LAMMPS_NS

#endif

