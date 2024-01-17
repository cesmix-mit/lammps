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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pod,PairPOD);
// clang-format on
#else

#ifndef LMP_PAIR_POD_H
#define LMP_PAIR_POD_H

#include "pair.h"

namespace LAMMPS_NS {

class PairPOD : public Pair {
 public:
  PairPOD(class LAMMPS *);
  ~PairPOD() override;
  void compute(int, int) override;

  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double memory_usage() override;

  void lammpsNeighborList(double **x, int **firstneigh, int *atomtype, int *map, int *numneigh,
                        double rcutsq, int i);
  void lammpsNeighborList(double **x, int **firstneigh, int *atomtype, int *map, int *numneigh,
                        double rcutsq, int i1, int i2);
  void tallyforce(double **force, double *fij,  int *ai, int *aj, int N);
  void divideInterval(int *intervals, int N, int M);
  int calculateNumberOfIntervals(int N, int intervalSize); 
  int numberOfNeighbors(int *numneigh, int gi1, int gi2);
  int maximumNumberOfNeighbors(int *numneigh);
  void free_temp_memory();
  void allocate_temp_memory(int N);
 protected:
  int dim;    // typically 3  
  int ni;            // number of atoms i
  int nij;           //  number of atom pairs
  int nijmax;        // maximum number of atom pairs
  int szd;           // size of tmpmem
  
  int atomBlockSize;        // size of each atom block
  int nAtomBlocks;          // number of atoms blocks
  int atomBlocks[100];      // atom blocks
  int nNeighbors;           // number of neighbors in the current block 
  int numNeighMax;          // maximum number of neighbors so far
  
  //class MLPOD *podptr;
  class EAPOD *fastpodptr;

  // temporary arrays for computation blocks

  double *tmpmem;      // temporary memory
  int *typeai;         // types of atoms I only
  //int *numneighsum;    // cumulative sum for an array of numbers of neighbors
  double *rij;         // (xj - xi) for all pairs (I, J)
  double *fij;         // force for all pairs (I, J)
  int *idxi;           // storing linear indices of atom I for all pairs (I, J)
  int *ai;             // IDs of atoms I for all pairs (I, J)
  int *aj;             // IDs of atoms J for all pairs (I, J)
  int *ti;             // types of atoms I for all pairs (I, J)
  int *tj;             // types of atoms J  for all pairs (I, J)  

  bool peratom_warn;    // print warning about missing per-atom energies or stresses
};

}    // namespace LAMMPS_NS

#endif
#endif
