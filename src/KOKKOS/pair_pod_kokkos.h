/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pod/kk,PairPODKokkos<LMPDeviceType>);
PairStyle(pod/kk/device,PairPODKokkos<LMPDeviceType>);
PairStyle(pod/kk/host,PairPODKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_POD_KOKKOS_H
#define LMP_PAIR_POD_KOKKOS_H

#include "eapod.h"
#include "pair_pod.h"
#include "kokkos_type.h"
#include "pair_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairPODKokkos : public PairPOD {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairPODKokkos(class LAMMPS *);
  ~PairPODKokkos() override;

  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  
// protected:
  int inum;
  int host_flag;

  int eflag, vflag;
  int neighflag;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d d_ilist;
  typename AT::t_int_1d d_numneigh;  
//   typename AT::t_int_1d_randomread d_ilist;
//   typename AT::t_int_1d_randomread d_numneigh;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  typedef Kokkos::DualView<F_FLOAT**, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq, k_scale;
  typedef Kokkos::View<F_FLOAT**, DeviceType> t_fparams;
  t_fparams d_cutsq, d_scale;
  typename AT::t_int_1d d_map;

  friend void pair_virial_fdotr_compute<PairPODKokkos>(PairPODKokkos*);

  void grow(int, int);  
  void copy_from_pod_class(EAPOD *podptr);
  void divideInterval(int *intervals, int N, int M);
  int calculateNumberOfIntervals(int N, int intervalSize);     
  void grow_atoms(int Ni);
  void grow_pairs(int Nij);
   
  void allocate() override;
  double memory_usage() override;

  typedef Kokkos::View<int*, DeviceType> t_pod_1i;
  typedef Kokkos::View<int**, DeviceType> t_pod_2i;
  typedef Kokkos::View<double*, DeviceType> t_pod_1d;
  typedef Kokkos::View<double**, DeviceType> t_pod_2d;
  typedef Kokkos::View<double**[3], DeviceType> t_pod_3d3;
  
    
  int atomBlockSize;        // size of each atom block
  int nAtomBlocks;          // number of atoms blocks
  int atomBlocks[101];      // atom blocks

  int ni;            // number of atoms i in the current atom block 
  int nij;           // number of pairs (i,j) in the current atom block 
  int nimax;         // maximum number of atoms i
  int nijmax;        // maximum number of pairs (i,j) 
  
  int nelements; // number of elements 
  int onebody;   // one-body descriptors
  int besseldegree; // degree of Bessel functions
  int inversedegree; // degree of inverse functions
  int nbesselpars;  // number of Bessel parameters
  int nCoeffPerElement; // number of coefficients per element = (nl1 + Mdesc*nClusters)
  int ns;      // number of snapshots for radial basis functions
  int nl1, nl2, nl3, nl4, nl23, nl33, nl34, nl44, n23, n32, nl;   // number of local descriptors
  int nrbf2, nrbf3, nrbf4, nrbfmax;            // number of radial basis functions
  int nabf3, nabf4;                            // number of angular basis functions  
  int K3, K4, Q4;                                  // number of monomials
    
  // environmental variables
  int nClusters; // number of environment clusters
  int nComponents; // number of principal components
  int Mdesc; // number of base descriptors 

  double rin;  // inner cut-off radius
  double rcut; // outer cut-off radius
  double rmax; // rcut - rin  
  
  t_pod_1d rij;         // (xj - xi) for all pairs (I, J)
  t_pod_1d fij;         // force for all pairs (I, J)
  t_pod_1d ei;          // energy for each atom I
  t_pod_1i typeai;         // types of atoms I only
  t_pod_1i numij;          // number of pairs (I, J) for each atom I   
  t_pod_1i idxi;           // storing linear indices of atom I for all pairs (I, J)
  t_pod_1i ai;             // IDs of atoms I for all pairs (I, J)
  t_pod_1i aj;             // IDs of atoms J for all pairs (I, J)
  t_pod_1i ti;             // types of atoms I for all pairs (I, J)
  t_pod_1i tj;             // types of atoms J for all pairs (I, J)  

  t_pod_1d besselparams;
  t_pod_1d Phi;  // eigenvectors matrix ns x ns
  t_pod_1d rbf;  // radial basis functions nij x nrbfmax  
  t_pod_1d rbfx; // x-derivatives of radial basis functions nij x nrbfmax 
  t_pod_1d rbfy; // y-derivatives of radial basis functions nij x nrbfmax
  t_pod_1d rbfz; // z-derivatives of radial basis functions nij x nrbfmax   
  t_pod_1d abf;  // angular basis functions nij x K3
  t_pod_1d abfx; // x-derivatives of angular basis functions nij x K3
  t_pod_1d abfy; // y-derivatives of angular basis functions nij x K3  
  t_pod_1d abfz; // z-derivatives of angular basis functions nij x K3
  t_pod_1d abftm;  // temp array for angular basis functions K3
  t_pod_1d abftmx; // temp array for x-derivatives of angular basis functions K3
  t_pod_1d abftmy; // temp array for y-derivatives of angular basis functions K3  
  t_pod_1d abftmz; // temp array for z-derivatives of angular basis functions K3
  t_pod_1d sumU; // sum of radial basis functions ni x K3 x nrbfmax x nelements
  t_pod_1d Proj; // PCA Projection matrix
  t_pod_1d Centroids; // centroids of the clusters  
  t_pod_1d bd;   // base descriptors ni x Mdesc
  t_pod_1d bdd;  // base descriptors derivatives 3 x nij x Mdesc 
  t_pod_1d pd;   // environment probability descriptors ni x nClusters
  t_pod_1d pdd;  // environment probability descriptors derivatives 3 x nij x nClusters
  t_pod_1d coefficients; // coefficients nCoeffPerElement x nelements
  t_pod_1i pq3, pn3, pc3; // arrays to compute 3-body angular basis functions
  t_pod_1i pa4, pb4, pc4; // arrays to compute 4-body angular basis functions  
  t_pod_1i ind23; // n23 
  t_pod_1i ind32; // n32
  t_pod_1i ind33l, ind33r; // nl33
  t_pod_1i ind34l, ind34r; // nl34
  t_pod_1i ind44l, ind44r; // nl44
  t_pod_1i elemindex;  
  
  void NeighborCount(double rcutsq, int gi1, int Ni);
  
  int numberOfNeighbors(int Ni);
  
  void NeighborList(double rcutsq, int gi1, int Ni);
  
  void radialbasis(t_pod_1d rbft, t_pod_1d rbftx, t_pod_1d rbfty, t_pod_1d rbftz, int Nij);    
  
  void matrixMultiply(t_pod_1d a, t_pod_1d b, t_pod_1d c, int r1, int c1, int c2); 
  
  void angularbasis(t_pod_1d tm, t_pod_1d tmu, t_pod_1d tmv, t_pod_1d tmw, int Nij);  
  
  void radialangularsum(const int Ni, const int Nij); 
  
  void twobodydescderiv(t_pod_1d d2, t_pod_1d dd2, const int Ni, const int Nij); 
  
  void threebodydesc(t_pod_1d d3, const int Ni);
  
  void threebodydescderiv(t_pod_1d dd3, int Ni, int Nij);
  
  void extractsumU(int Ni);
  
  void fourbodydesc(t_pod_1d d4, int Ni);
    
  void fourbodydescderiv(t_pod_1d dd4, int Ni, int Nij);
  
  void fourbodydesc23(t_pod_1d d23, t_pod_1d d2, t_pod_1d d3, int Ni);
  
  void fourbodydescderiv23(t_pod_1d dd23, t_pod_1d d2, t_pod_1d d3, t_pod_1d dd2, t_pod_1d dd3, int Ni, int Nij);
  
  void crossdesc(t_pod_1d d12, t_pod_1d d1, t_pod_1d d2, t_pod_1i ind1, t_pod_1i ind2, int n12, int Ni);
  
  void crossdescderiv(t_pod_1d dd12, t_pod_1d d1, t_pod_1d d2, t_pod_1d dd1, t_pod_1d dd2,
        t_pod_1i ind1, t_pod_1i ind2, int n12, int Ni, int Nij);
  
  void blockatom_base_descriptors(int Ni, int Nij);
      
  void blockatomenergyforce(int Ni, int Nij);
  
  void tallyforce(int Nij);
  
  void tallyenergy(int istart, int Ni);

  void tallystress(int Nij);  
  
  void savematrix2binfile(std::string filename, t_pod_1d d_A, int nrows, int ncols);
  void saveintmatrix2binfile(std::string filename, t_pod_1i d_A, int nrows, int ncols);
  void savedatafordebugging();
};
}    // namespace LAMMPS_NS

#endif
#endif
