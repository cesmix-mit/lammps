// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   aE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "eapod.h" 
#include "pair_pod_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;
using MathSpecial::powint;

enum{FS,FS_SHIFTEDSCALED};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairPODKokkos<DeviceType>::PairPODKokkos(LAMMPS *lmp) : PairPOD(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  ni = 0;
  nimax = 0;
  nij = 0;
  nijmax = 0;  
  atomBlockSize = 1024;
  nAtomBlocks = 0;
  
  host_flag = (execution_space == Host);
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<class DeviceType>
PairPODKokkos<DeviceType>::~PairPODKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom,eatom);
  memoryKK->destroy_kokkos(k_vatom,vatom);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::init_style()
{
  if (host_flag) {
    if (lmp->kokkos->nthreads > 1)
      error->all(FLERR,"Pair style pod/kk can currently only run on a single "
                         "CPU thread");

    PairPOD::init_style();
    return;
  }

  if (atom->tag_enable == 0) error->all(FLERR, "Pair style POD requires atom IDs");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style POD requires newton pair on");

  // neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL)
    error->all(FLERR,"Must use half neighbor list style with pair pace/kk");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairPODKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairPOD::init_one(i,j);

//   k_scale.h_view(i,j) = k_scale.h_view(j,i) = scale[i][j];
//   k_scale.template modify<LMPHostType>();

  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();

  return cutone;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::coeff(int narg, char **arg)
{
  if (narg < 7) utils::missing_cmd_args(FLERR, "pair_coeff", error);
    
  PairPOD::coeff(narg,arg); // create a PairPOD object
  
  printf("nelements %d, Mdesc %d\n", PairPOD::fastpodptr->nelements, PairPOD::fastpodptr->Mdesc);
  
  copy_from_pod_class(PairPOD::fastpodptr); // copy parameters and arrays from pod class 

  int n = atom->ntypes + 1;
  MemKK::realloc_kokkos(d_map, "pair_pod:map", n);

  MemKK::realloc_kokkos(k_cutsq, "pair_pod:cutsq", n, n);
  d_cutsq = k_cutsq.template view<DeviceType>();

  MemKK::realloc_kokkos(k_scale, "pair_pod:scale", n, n);
  d_scale = k_scale.template view<DeviceType>();
  
  // Set up element lists  
  
  auto h_map = Kokkos::create_mirror_view(d_map);
  
  for (int i = 1; i <= atom->ntypes; i++)
    h_map(i) = map[i];
  
  Kokkos::deep_copy(d_map,h_map);      
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::allocate()
{
  PairPOD::allocate();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
//   if (host_flag) {
//     printf("atomBlockSize: %d\n", atomBlockSize);
//     atomKK->sync(Host,X_MASK|TYPE_MASK);
//     PairPOD::compute(eflag_in,vflag_in);
//     atomKK->modified(Host,F_MASK);
//     return;
//   }
  
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  //printf("eflag_atom %d, vflag_atom %d\n", eflag_atom, vflag_atom);
  
  // reallocate per-atom arrays if necessary
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  copymode = 1;  
  int newton_pair = force->newton_pair;
  if (newton_pair == false)
    error->all(FLERR,"PairPODKokkos requires 'newton on'");
  
  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);    
  x = atomKK->k_x.view<DeviceType>();    
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  //k_cutsq.template sync<DeviceType>();
 
  //printf("type %d, type %d\n", type[0], type[1]);
  
  if (host_flag) {        
    inum = list->inum;    
    d_numneigh = typename ArrayTypes<DeviceType>::t_int_1d("pair_pod:numneigh",inum);
    for (int i=0; i<inum; i++) d_numneigh(i) = list->numneigh[i];    
    d_ilist = typename ArrayTypes<DeviceType>::t_int_1d("pair_pod:ilist",inum);
    for (int i=0; i<inum; i++) d_ilist(i) = list->ilist[i];    
    
    int maxn = 0;
    for (int i=0; i<inum; i++) 
      if (maxn < list->numneigh[i]) maxn = list->numneigh[i];    
    MemoryKokkos::realloc_kokkos(d_neighbors,"neighlist:neighbors",inum,maxn);
    for (int i=0; i<inum; i++) {
      int gi = list->ilist[i];
      int m = list->numneigh[gi];      
      for (int l = 0; l < m; l++) {           // loop over each atom around atom i
        int gj = list->firstneigh[gi][l];           // atom j
        d_neighbors(gi, l) = list->firstneigh[gi][l];
      }
    }    
    //error->all(FLERR,"here");            
  }
  else {
    NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);  
    d_numneigh = k_list->d_numneigh; 
    d_neighbors = k_list->d_neighbors;
    d_ilist = k_list->d_ilist;
    inum = list->inum;
  }
  
  // determine the number of atom blocks and divide atoms into blocks
  nAtomBlocks = calculateNumberOfIntervals(inum, atomBlockSize);
  if (nAtomBlocks > 100) nAtomBlocks = 100; 
  divideInterval(atomBlocks, inum, nAtomBlocks);
    
  int nmax = 0;
  for (int block=0; block<nAtomBlocks; block++) {    
    int n = atomBlocks[block+1] - atomBlocks[block]; 
    if (nmax < n) nmax = n;
  }
  grow_atoms(nmax); // reallocate memory only if necessary  

  double rcutsq = rcut*rcut;
  
  for (int block=0; block<nAtomBlocks; block++) {
    int gi1 = atomBlocks[block]-1;
    int gi2 = atomBlocks[block+1]-1;
    ni = gi2 - gi1; // total number of atoms in the current atom block
    
    //printf("block %d, ni %d\n", block, ni);

    // calculate the total number of pairs (i,j) in the current atom block        
    NeighborCount(rcutsq, gi1, ni);                    
    nij = numberOfNeighbors(ni); // total number of pairs (i,j) in the current atom block        
    grow_pairs(nij); // reallocate memory only if necessary
       
    // obtain the neighbors within rcut 
    NeighborList(rcutsq, gi1, ni);   
    
    // compute atomic energy and force for the current atom block
    blockatomenergyforce(ni, nij);

    // tally atomic energy to global energy
    tallyenergy(gi1, ni);

    // tally atomic force to global force
    tallyforce(nij);

    // tally atomic stress
    if (vflag) tallystress(nij);
    
    //savedatafordebugging();
  }    
  
  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  atomKK->modified(execution_space,F_MASK);

  copymode = 0;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::copy_from_pod_class(EAPOD *podptr) 
{
  nelements = podptr->nelements; // number of elements 
  onebody = podptr->onebody;   // one-body descriptors
  besseldegree = podptr->besseldegree; // degree of Bessel functions
  inversedegree = podptr->inversedegree; // degree of inverse functions
  nbesselpars = podptr->nbesselpars;  // number of Bessel parameters
  nCoeffPerElement = podptr->nCoeffPerElement; // number of coefficients per element = (nl1 + Mdesc*nClusters)
  ns = podptr->ns;      // number of snapshots for radial basis functions
  nl1 = podptr->nl1;  // number of one-body descriptors
  nl2 = podptr->nl2;  // number of two-body descriptors
  nl3 = podptr->nl3;  // number of three-body descriptors
  nl4 = podptr->nl4;  // number of four-body descriptors
  nl23 = podptr->nl23; // number of two-body x three-body descriptors
  nl33 = podptr->nl33; // number of three-body x three-body descriptors
  nl34 = podptr->nl34; // number of three-body x four-body descriptors
  nl44 = podptr->nl44; // number of four-body x four-body descriptors
  n23 = podptr->n23;
  n32 = podptr->n32;
  nl = podptr->nl;   // number of local descriptors
  nrbf2 = podptr->nrbf2;
  nrbf3 = podptr->nrbf3;
  nrbf4 = podptr->nrbf4;
  nrbfmax = podptr->nrbfmax; // number of radial basis functions
  nabf3 = podptr->nabf3;     // number of three-body angular basis functions
  nabf4 = podptr->nabf4;     // number of four-body angular basis functions  
  K3 = podptr->K3;           // number of three-body monomials
  K4 = podptr->K4;           // number of four-body monomials
  Q4 = podptr->Q4;           // number of four-body monomial coefficients
  nClusters = podptr->nClusters; // number of environment clusters
  nComponents = podptr->nComponents; // number of principal components
  Mdesc = podptr->Mdesc; // number of base descriptors 

  rin = podptr->rin;
  rcut = podptr->rcut;
  rmax = rcut - rin;  
    
  MemKK::realloc_kokkos(besselparams, "pair_pod:besselparams", 3); 
  auto h_besselparams = Kokkos::create_mirror_view(besselparams);  
  h_besselparams[0] = podptr->besselparams[0];
  h_besselparams[1] = podptr->besselparams[1];
  h_besselparams[2] = podptr->besselparams[2];   
  Kokkos::deep_copy(besselparams, h_besselparams);    
   
  MemKK::realloc_kokkos(abftm, "pair_pod:abftm", K3);
  MemKK::realloc_kokkos(abftmx, "pair_pod:abftmx", K3);
  MemKK::realloc_kokkos(abftmy, "pair_pod:abftmy", K3);
  MemKK::realloc_kokkos(abftmz, "pair_pod:abftmz", K3);
  Kokkos::deep_copy(abftm, 1.0);
  Kokkos::deep_copy(abftmx, 0.0);
  Kokkos::deep_copy(abftmy, 0.0);
  Kokkos::deep_copy(abftmz, 0.0);       
  
  MemKK::realloc_kokkos(elemindex, "pair_pod:elemindex", nelements*nelements);
  auto h_elemindex = Kokkos::create_mirror_view(elemindex);
  for (int i=0; i<nelements*nelements; i++) h_elemindex[i] = podptr->elemindex[i];
  Kokkos::deep_copy(elemindex, h_elemindex);
  
  MemKK::realloc_kokkos(Phi, "pair_pod:Phi", ns*ns);
  auto h_Phi = Kokkos::create_mirror_view(Phi);
  for (int i=0; i<ns*ns; i++) h_Phi[i] = podptr->Phi[i];
  Kokkos::deep_copy(Phi, h_Phi);

  MemKK::realloc_kokkos(coefficients, "pair_pod:coefficients", nCoeffPerElement * nelements);
  auto h_coefficients = Kokkos::create_mirror_view(coefficients);
  for (int i=0; i<nCoeffPerElement * nelements; i++) h_coefficients[i] = podptr->coeff[i];
  Kokkos::deep_copy(coefficients, h_coefficients);

  MemKK::realloc_kokkos(pn3, "pair_pod:pn3", nabf3+1); // array stores the number of monomials for each degree
  MemKK::realloc_kokkos(pq3, "pair_pod:pq3", K3*2); // array needed for the recursive computation of the angular basis functions
  MemKK::realloc_kokkos(pc3, "pair_pod:pc3", K3);   // array needed for the computation of the three-body descriptors
  MemKK::realloc_kokkos(pa4, "pair_pod:pa4", nabf4+1); // this array is a subset of the array {0, 1, 4, 10, 19, 29, 47, 74, 89, 119, 155, 209, 230, 275, 335, 425, 533, 561, 624, 714, 849, 949, 1129, 1345}
  MemKK::realloc_kokkos(pb4, "pair_pod:pb4", Q4*3); // array stores the indices of the monomials needed for the computation of the angular basis functions
  MemKK::realloc_kokkos(pc4, "pair_pod:pc4", Q4);   // array of monomial coefficients needed for the computation of the four-body descriptors  

  auto h_pn3 = Kokkos::create_mirror_view(pn3);
  for (int i=0; i<nabf3+1; i++) h_pn3[i] = podptr->pn3[i];
  Kokkos::deep_copy(pn3, h_pn3);

  auto h_pq3 = Kokkos::create_mirror_view(pq3);
  for (int i = 0; i < K3*2; i++) h_pq3[i] = podptr->pq3[i];
  Kokkos::deep_copy(pq3, h_pq3);

  auto h_pc3 = Kokkos::create_mirror_view(pc3);
  for (int i = 0; i < K3; i++) h_pc3[i] = podptr->pc3[i];
  Kokkos::deep_copy(pc3, h_pc3);

  auto h_pa4 = Kokkos::create_mirror_view(pa4);
  for (int i = 0; i < nabf4+1; i++) h_pa4[i] = podptr->pa4[i];
  Kokkos::deep_copy(pa4, h_pa4);

  auto h_pb4 = Kokkos::create_mirror_view(pb4);
  for (int i = 0; i < Q4*3; i++) h_pb4[i] = podptr->pb4[i];
  Kokkos::deep_copy(pb4, h_pb4);

  auto h_pc4 = Kokkos::create_mirror_view(pc4);
  for (int i = 0; i < Q4; i++) h_pc4[i] = podptr->pc4[i];
  Kokkos::deep_copy(pc4, h_pc4);

  MemKK::realloc_kokkos(ind23, "pair_pod:ind23", n23);
  MemKK::realloc_kokkos(ind32, "pair_pod:ind32", n32);
  MemKK::realloc_kokkos(ind33l, "pair_pod:ind33l", nl33);
  MemKK::realloc_kokkos(ind33r, "pair_pod:ind33r", nl33);
  MemKK::realloc_kokkos(ind34l, "pair_pod:ind34l", nl34);
  MemKK::realloc_kokkos(ind34r, "pair_pod:ind34r", nl34);
  MemKK::realloc_kokkos(ind44l, "pair_pod:ind44l", nl44);
  MemKK::realloc_kokkos(ind44r, "pair_pod:ind44r", nl44);
  
  auto h_ind23 = Kokkos::create_mirror_view(ind23);
  for (int i = 0; i < n23; i++) h_ind23[i] = podptr->ind23[i];
  Kokkos::deep_copy(ind23, h_ind23);

  auto h_ind32 = Kokkos::create_mirror_view(ind32);
  for (int i = 0; i < n32; i++) h_ind32[i] = podptr->ind32[i];
  Kokkos::deep_copy(ind32, h_ind32);

  auto h_ind33l = Kokkos::create_mirror_view(ind33l);
  for (int i = 0; i < nl33; i++) h_ind33l[i] = podptr->ind33l[i];
  Kokkos::deep_copy(ind33l, h_ind33l);

  auto h_ind33r = Kokkos::create_mirror_view(ind33r);
  for (int i = 0; i < nl33; i++) h_ind33r[i] = podptr->ind33r[i];
  Kokkos::deep_copy(ind33r, h_ind33r);

  auto h_ind34l = Kokkos::create_mirror_view(ind34l);
  for (int i = 0; i < nl34; i++) h_ind34l[i] = podptr->ind34l[i];
  Kokkos::deep_copy(ind34l, h_ind34l);

  auto h_ind34r = Kokkos::create_mirror_view(ind34r);
  for (int i = 0; i < nl34; i++) h_ind34r[i] = podptr->ind34r[i];
  Kokkos::deep_copy(ind34r, h_ind34r);

  auto h_ind44l = Kokkos::create_mirror_view(ind44l);
  for (int i = 0; i < nl44; i++) h_ind44l[i] = podptr->ind44l[i];
  Kokkos::deep_copy(ind44l, h_ind44l);

  auto h_ind44r = Kokkos::create_mirror_view(ind44r);
  for (int i = 0; i < nl44; i++) h_ind44r[i] = podptr->ind44r[i];
  Kokkos::deep_copy(ind44r, h_ind44r); 
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::divideInterval(int *intervals, int N, int M) 
{
  int intervalSize = N / M; // Basic size of each interval
  int remainder = N % M;    // Remainder to distribute
  intervals[0] = 1;         // Start of the first interval
  for (int i = 1; i <= M; i++) {
    intervals[i] = intervals[i - 1] + intervalSize + (remainder > 0 ? 1 : 0);
    if (remainder > 0) {
      remainder--;
    }
  }  
}

template<class DeviceType>
int PairPODKokkos<DeviceType>::calculateNumberOfIntervals(int N, int intervalSize) 
{
  if (intervalSize <= 0) {
    printf("Interval size must be a positive integer.\n");
    return -1;
  }

  int M = N / intervalSize;
  if (N % intervalSize != 0) {
    M++; // Add an additional interval to cover the remainder
  }

  return M;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::grow_atoms(int Ni)
{
  if (Ni > nimax) {
    nimax = Ni;
    MemKK::realloc_kokkos(numij, "pair_pod:numij", nimax+1);        
    MemKK::realloc_kokkos(ei, "pair_pod:ei", nimax);
    MemKK::realloc_kokkos(typeai, "pair_pod:typeai", nimax);
    MemKK::realloc_kokkos(sumU, "pair_pod:sumU", nimax * nelements * K3 * nrbfmax);
    MemKK::realloc_kokkos(bd, "pair_pod:bd", nimax * Mdesc);
    MemKK::realloc_kokkos(pd, "pair_pod:bd", nimax * nClusters);
    
    Kokkos::deep_copy(numij, 0);       
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::grow_pairs(int Nij)
{
  if (Nij > nijmax) {
    nijmax = Nij;
    MemKK::realloc_kokkos(rij, "pair_pod:r_ij", 3 * nijmax);
    MemKK::realloc_kokkos(fij, "pair_pod:f_ij", 3 * nijmax);  
    MemKK::realloc_kokkos(idxi, "pair_pod:idxi", nijmax);
    MemKK::realloc_kokkos(ai, "pair_pod:ai", nijmax);
    MemKK::realloc_kokkos(aj, "pair_pod:aj", nijmax);
    MemKK::realloc_kokkos(ti, "pair_pod:ti", nijmax);
    MemKK::realloc_kokkos(tj, "pair_pod:tj", nijmax);
    MemKK::realloc_kokkos(rbf, "pair_pod:rbf", nijmax * nrbfmax);
    MemKK::realloc_kokkos(rbfx, "pair_pod:rbfx", nijmax * nrbfmax);
    MemKK::realloc_kokkos(rbfy, "pair_pod:rbfy", nijmax * nrbfmax);
    MemKK::realloc_kokkos(rbfz, "pair_pod:rbfz", nijmax * nrbfmax);
    int kmax = (K3 > ns) ? K3 : ns;
    MemKK::realloc_kokkos(abf, "pair_pod:abf", nijmax * kmax);
    MemKK::realloc_kokkos(abfx, "pair_pod:abfx", nijmax * kmax);
    MemKK::realloc_kokkos(abfy, "pair_pod:abfy", nijmax * kmax);
    MemKK::realloc_kokkos(abfz, "pair_pod:abfz", nijmax * kmax);
    MemKK::realloc_kokkos(bdd, "pair_pod:bdd", 3 * nijmax * Mdesc);
    MemKK::realloc_kokkos(pdd, "pair_pod:pdd", 3 * nijmax * nClusters);    
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::NeighborCount(double rcutsq, int gi1, int Ni)
{
  Kokkos::parallel_for("NeighborCount", Ni, KOKKOS_LAMBDA(int i) {
    int gi = d_ilist[gi1 + i];
    X_FLOAT xi0 = x(gi, 0);    
    X_FLOAT xi1 = x(gi, 1);    
    X_FLOAT xi2 = x(gi, 2);        
    int m = d_numneigh(gi);
    int n = 0;
    for (int l = 0; l < m; l++) { // loop over each atom around atom i
      int gj = d_neighbors(gi, l); // atom j
      X_FLOAT delx = x(gj, 0) - xi0; // xj - xi
      X_FLOAT dely = x(gj, 1) - xi1; // yj - yi
      X_FLOAT delz = x(gj, 2) - xi2; // zj - zi
      X_FLOAT rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < rcutsq && rsq > 1e-20) n++;
    }
    numij(1 + i) = n; // Assuming numij(0) is reserved or used for something else
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::NeighborList(double rcutsq, int gi1, int Ni)
{
  Kokkos::parallel_for("NeighborCount", Ni, KOKKOS_LAMBDA(int i) {
    int gi = d_ilist[gi1 + i];
    X_FLOAT xi0 = x(gi, 0);    
    X_FLOAT xi1 = x(gi, 1);    
    X_FLOAT xi2 = x(gi, 2);        
    int itype = d_map(type(gi)) + 1; //map[atomtypes[gi]] + 1;
    typeai(i) = itype;    
    int m = d_numneigh(gi);
    int nij0 = numij(i);    
    int k = 0;
    for (int l = 0; l < m; l++) { // loop over each atom around atom i
      int gj = d_neighbors(gi, l); // atom j
      X_FLOAT delx = x(gj, 0) - xi0; // xj - xi
      X_FLOAT dely = x(gj, 1) - xi1; // yj - yi
      X_FLOAT delz = x(gj, 2) - xi2; // zj - zi
      X_FLOAT rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < rcutsq && rsq > 1e-20) {
        int nij1 = nij0 + k;
        rij[nij1 * 3 + 0] = delx;
        rij[nij1 * 3 + 1] = dely;
        rij[nij1 * 3 + 2] = delz;
        idxi[nij1] = i;
        ai[nij1] = gi;
        aj[nij1] = gj;
        ti[nij1] = itype;
        tj[nij1] = d_map(type(gj)) + 1; //map[atomtypes[gj]] + 1;
        k++;        
      }
    }
  });
}

template<class DeviceType>
int PairPODKokkos<DeviceType>::numberOfNeighbors(int Ni) {
  // Perform an inclusive scan on numij
  Kokkos::parallel_scan("InclusivePrefixSum", Ni + 1, KOKKOS_LAMBDA(int i, int& update, const bool final) {
    if (i > 0) { // Skip the first element as the original loop starts from 1
      update += numij(i);
      if (final) {
        numij(i) = update;
      }
    }
  });

  // Retrieve the total number of neighbors from the last element of numij
  int total_neighbors = 0;
  Kokkos::deep_copy(Kokkos::View<int>(&total_neighbors, 1), Kokkos::subview(numij, Ni));

  return total_neighbors;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialbasis(t_pod_1d rbft, t_pod_1d rbftx, t_pod_1d rbfty, t_pod_1d rbftz, int Nij) 
{
  Kokkos::parallel_for("ComputeRadialBasis", Nij, KOKKOS_LAMBDA(int n) {
    double xij1 = rij(0+3*n);
    double xij2 = rij(1+3*n);
    double xij3 = rij(2+3*n);

    double dij = sqrt(xij1*xij1 + xij2*xij2 + xij3*xij3);
    double dr1 = xij1/dij;
    double dr2 = xij2/dij;
    double dr3 = xij3/dij;

    double r = dij - rin;
    double y = r/rmax;
    double y2 = y*y;

    double y3 = 1.0 - y2*y;
    double y4 = y3*y3 + 1e-6;
    double y5 = sqrt(y4);
    double y6 = exp(-1.0/y5);
    double y7 = y4*sqrt(y4);

    // Calculate the final cutoff function as y6/exp(-1)
    double fcut = y6/exp(-1.0);

    // Calculate the derivative of the final cutoff function
    double dfcut = ((3.0/(rmax*exp(-1.0)))*(y2)*y6*(y*y2 - 1.0))/y7;

    // Calculate fcut/r, fcut/r^2, and dfcut/r
    double f1 = fcut/r;
    double f2 = f1/r;
    double df1 = dfcut/r;

    double alpha = besselparams(0);
    double t1 = (1.0-exp(-alpha));
    double t2 = exp(-alpha*r/rmax);
    double x0 =  (1.0 - t2)/t1;
    double dx0 = (alpha/rmax)*t2/t1;

    alpha = besselparams(1);
    t1 = (1.0-exp(-alpha));
    t2 = exp(-alpha*r/rmax);
    double x1 =  (1.0 - t2)/t1;
    double dx1 = (alpha/rmax)*t2/t1;

    alpha = besselparams(2);
    t1 = (1.0-exp(-alpha));
    t2 = exp(-alpha*r/rmax);
    double x2 =  (1.0 - t2)/t1;
    double dx2 = (alpha/rmax)*t2/t1;
    for (int i=0; i<besseldegree; i++) {
      double a = (i+1)*MY_PI;
      double b = (sqrt(2.0/(rmax))/(i+1));
      double af1 = a*f1;

      double sinax = sin(a*x0);
      int idxni = n + Nij*i;

      rbft(idxni) = b*f1*sinax;
      double drbftdr = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);
      rbftx(idxni) = drbftdr*dr1;
      rbfty(idxni) = drbftdr*dr2;
      rbftz(idxni) = drbftdr*dr3;

      sinax = sin(a*x1);
      idxni = n + Nij*i + Nij*besseldegree*1;

      rbft(idxni) = b*f1*sinax;
      drbftdr = b*(df1*sinax - f2*sinax + af1*cos(a*x1)*dx1);
      rbftx(idxni) = drbftdr*dr1;
      rbfty(idxni) = drbftdr*dr2;
      rbftz(idxni) = drbftdr*dr3;

      sinax = sin(a*x2);
      idxni = n + Nij*i + Nij*besseldegree*2;
      rbft(idxni) = b*f1*sinax;
      drbftdr = b*(df1*sinax - f2*sinax + af1*cos(a*x2)*dx2);
      rbftx(idxni) = drbftdr*dr1;
      rbfty(idxni) = drbftdr*dr2;
      rbftz(idxni) = drbftdr*dr3;
    }
  
    // Calculate fcut/dij and dfcut/dij
    f1 = fcut/dij;
    for (int i=0; i<inversedegree; i++) {
      int p = besseldegree*nbesselpars + i;
      int idxni = n + Nij*p;      
      double a = powint(dij, i+1);

      rbft(idxni) = fcut/a;

      double drbftdr = (dfcut - (i+1.0)*f1)/a;
      rbftx(idxni) = drbftdr*dr1;
      rbfty(idxni) = drbftdr*dr2;
      rbftz(idxni) = drbftdr*dr3;
    }
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::matrixMultiply(t_pod_1d a, t_pod_1d b, t_pod_1d c, int r1, int c1, int c2) 
{
    Kokkos::parallel_for("MatrixMultiply", r1 * c2, KOKKOS_LAMBDA(int idx) {
        int j = idx / r1;  // Calculate column index
        int i = idx % r1;  // Calculate row index
        double sum = 0.0;
        for (int k = 0; k < c1; ++k) {
            sum += a(i + r1*k) * b(k + c1*j);  // Manually calculate the 1D index
        }
        c(i + r1*j) = sum;  // Manually calculate the 1D index for c
    });        
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::angularbasis(t_pod_1d tm, t_pod_1d tmu,
                  t_pod_1d tmv, t_pod_1d tmw, int N) 
{  
  Kokkos::parallel_for("AngularBasis", N, KOKKOS_LAMBDA(int j) {
    double x = rij(j*3 + 0);
    double y = rij(j*3 + 1);
    double z = rij(j*3 + 2);

    double xx = x*x;
    double yy = y*y;
    double zz = z*z;
    double xy = x*y;
    double xz = x*z;
    double yz = y*z;

    double dij = sqrt(xx + yy + zz);
    double u = x / dij;
    double v = y / dij;
    double w = z / dij;

    double dij3 = dij * dij * dij;
    double dudx = (yy + zz) / dij3;
    double dudy = -xy / dij3;
    double dudz = -xz / dij3;

    double dvdx = -xy / dij3;
    double dvdy = (xx + zz) / dij3;
    double dvdz = -yz / dij3;

    double dwdx = -xz / dij3;
    double dwdy = -yz / dij3;
    double dwdz = (xx + yy) / dij3;

    abf(j) = tm(0);
    abfx(j) = 0.0;
    abfy(j) = 0.0;
    abfz(j) = 0.0;

    for (int n = 1; n < K3; n++) {
      int m = pq3(n) - 1;
      int d = pq3(n + K3);

      if (d == 1) {
        tm(n) = tm(m) * u;
        tmu(n) = tmu(m) * u + tm(m);
        tmv(n) = tmv(m) * u;
        tmw(n) = tmw(m) * u;
      } else if (d == 2) {
        tm(n) = tm(m) * v;
        tmu(n) = tmu(m) * v;
        tmv(n) = tmv(m) * v + tm(m);
        tmw(n) = tmw(m) * v;
      } else if (d == 3) {
        tm(n) = tm(m) * w;
        tmu(n) = tmu(m) * w;
        tmv(n) = tmv(m) * w;
        tmw(n) = tmw(m) * w + tm(m);
      }
      abf(j + N * n) = tm(n);
      abfx(j + N * n) = tmu(n) * dudx + tmv(n) * dvdx + tmw(n) * dwdx;
      abfy(j + N * n) = tmu(n) * dudy + tmv(n) * dvdy + tmw(n) * dwdy;
      abfz(j + N * n) = tmu(n) * dudz + tmv(n) * dvdz + tmw(n) * dwdz;
    }
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialangularsum(const int Ni, const int Nij) 
{  
  // Initialize sumU to zero
  Kokkos::deep_copy(sumU, 0.0);

  int totalIterations = nrbf3 * K3 * Nij;
  Kokkos::parallel_for("RadialAngularSum", totalIterations, KOKKOS_LAMBDA(int idx) {
      int n = idx % Nij;
      int temp = idx / Nij;
      int k = temp % K3;
      int m = temp / K3;

      int ia = n + Nij * k;
      int ib = n + Nij * m;
      int ii = ia + Nij * K3 * m;

      // Update sumU with atomtype adjustment
      int tn = tj(n) - 1; // offset the atom type by 1, since atomtype is 1-based
      Kokkos::atomic_add(&sumU(idxi(n) + Ni * (tn + nelements * k + nelements * K3 * m)), rbf(ib) * abf(ia));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobodydescderiv(t_pod_1d d2, t_pod_1d dd2, const int Ni, const int Nij) 
{
  int totalIterations = nrbf2 * Nij;

  Kokkos::parallel_for("TwoBodyDescDeriv", totalIterations, KOKKOS_LAMBDA(int idx) {
    int m = idx / Nij; // Recalculate m
    int n = idx % Nij; // Recalculate n

    int i2 = n + Nij * m; // Index of the radial basis function for atom n and RBF m
    int i1 = n + Nij * m + Nij * nrbf2 * (tj(n) - 1); // Index of the descriptor for atom n, RBF m, and atom type tj[n]

    Kokkos::atomic_add(&d2(idxi(n) + Ni * (m + nrbf2 * (tj(n) - 1))), rbf(i2)); // Add the radial basis function to the corresponding descriptor
    dd2(0 + 3 * i1) = rbfx(i2); // Assign the derivative with respect to x
    dd2(1 + 3 * i1) = rbfy(i2); // Assign the derivative with respect to y
    dd2(2 + 3 * i1) = rbfz(i2); // Assign the derivative with respect to z
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::threebodydesc(t_pod_1d d3, const int Ni) 
{

  int totalIterations = nrbf3 * Ni;
  Kokkos::parallel_for("ThreeBodyDesc", totalIterations, KOKKOS_LAMBDA(int idx) {
    int i = idx % Ni;
    int m = idx / Ni;
    for (int p = 0; p < nabf3; p++) {
      int n1 = pn3(p);
      int n2 = pn3(p + 1);
      int nn = n2 - n1;
      for (int q = 0; q < nn; q++) {
        int k = 0;
        for (int i1 = 0; i1 < nelements; i1++) {
          double t1 = pc3(n1 + q) * sumU(i + Ni * (i1 + nelements * (n1 + q) + nelements * K3 * m));
          for (int i2 = i1; i2 < nelements; i2++) {
            int index = i + Ni * (p + nabf3 * m + nabf3 * nrbf3 * k);
            d3(index) += t1 * sumU(i + Ni * (i2 + nelements * (n1 + q) + nelements * K3 * m));
            k += 1;
          }
        }
      }
    }
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::threebodydescderiv(t_pod_1d dd3, int Ni, int Nij)
{
  int totalIterations = nrbf3 * Nij;
  if (nelements==1) {
    Kokkos::parallel_for("ThreeBodyDescDeriv1", totalIterations, KOKKOS_LAMBDA(int idx) {
      int j = idx / nrbf3;       // Calculate j using integer division
      int m = idx % nrbf3;       // Calculate m using modulo operation
      int idxR = j + Nij * m;  // Pre-compute the index for rbf
      double rbfBase = rbf[idxR];
      double rbfxBase = rbfx[idxR];
      double rbfyBase = rbfy[idxR];
      double rbfzBase = rbfz[idxR];

      for (int p = 0; p < nabf3; p++) {
        int n1 = pn3[p];
        int n2 = pn3[p + 1];
        int nn = n2 - n1;
        int baseIdx = 3 * j + 3 * Nij * (p + nabf3 * m);  // Pre-compute the base index for dd3
        int idxU = idxi[j] + Ni * (K3 * m);
        for (int q = 0; q < nn; q++) {                  
          int idxNQ = n1 + q;  // Combine n1 and q into a single index for pc3 and sumU
          double t1 = pc3[idxNQ] * sumU[idxU + Ni * idxNQ];
          double f = 2.0 * t1;          
          int idxA = j + Nij * idxNQ;  // Pre-compute the index for abf          
          double abfA = abf[idxA];  

          // Use the pre-computed indices to update dd3
          dd3[baseIdx]     += f * (abfx[idxA] * rbfBase + rbfxBase * abfA);
          dd3[baseIdx + 1] += f * (abfy[idxA] * rbfBase + rbfyBase * abfA);
          dd3[baseIdx + 2] += f * (abfz[idxA] * rbfBase + rbfzBase * abfA);          
        }
      }
    });
  }
  else {
    int N3 = 3 * Nij *  nabf3 * nrbf3;
    Kokkos::parallel_for("ThreeBodyDescDeriv2", totalIterations, KOKKOS_LAMBDA(int idx) {
      int j = idx / nrbf3;  // Derive the original j value
      int m = idx % nrbf3;  // Derive the original m value
      int idxR = j + Nij * m;  // Pre-compute the index for rbf
      double rbfBase = rbf[idxR];
      double rbfxBase = rbfx[idxR];
      double rbfyBase = rbfy[idxR];
      double rbfzBase = rbfz[idxR];

      for (int p = 0; p < nabf3; p++) {
        int n1 = pn3[p];
        int n2 = pn3[p + 1];
        int nn = n2 - n1;
        int jmp = 3 * j + 3 * Nij * (p + nabf3 * m);
        for (int q = 0; q < nn; q++) {
          int idxNQ = n1 + q;  // Combine n1 and q into a single index
          int idxU = idxi[j] + Ni * (nelements * idxNQ + nelements * K3 * m);
          int idxA = j + Nij * idxNQ;  // Pre-compute the index for abf      
          double abfA = abf[idxA];   
          double abfxA = abfx[idxA];
          double abfyA = abfy[idxA];
          double abfzA = abfz[idxA];

          for (int i1 = 0; i1 < nelements; i1++) {
            double t1 = pc3[idxNQ] * sumU[idxU + Ni*i1];
            int i2 = tj[j] - 1;
            int k = elemindex[i2 + nelements * i1];
            double f = (i1 == i2) ? 2.0 * t1 : t1;
            int ii = jmp + N3 * k;                     

            // Update dd3
            dd3[0 + ii] += f * (abfxA * rbfBase + rbfxBase * abfA);
            dd3[1 + ii] += f * (abfyA * rbfBase + rbfyBase * abfA);
            dd3[2 + ii] += f * (abfzA * rbfBase + rbfzBase * abfA);          
          }
        }
      }
    });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::extractsumU(int Ni)
{
  int totalIterations = nrbf4 * K4 * nelements * Ni;
  Kokkos::parallel_for("extractsumU", totalIterations, KOKKOS_LAMBDA(int idx) {
    int n = idx % Ni;
    int temp = idx / Ni;
    int i = temp % nelements;
    temp = temp / nelements;
    int k = temp % K4;
    int m = temp / K4;

    int indexDst = n + Ni * i + Ni * nelements * k + Ni * nelements * K4 * m;
    int indexSrc = n + Ni * i + Ni * nelements * k + Ni * nelements * K3 * m;
    sumU[indexDst] = sumU[indexSrc];
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::fourbodydesc(t_pod_1d d4, int Ni)
{
  int totalIterations = nrbf4 * Ni;
  Kokkos::parallel_for("fourbodydesc", totalIterations, KOKKOS_LAMBDA(int idx) {
    int i = idx % Ni;
    int m = idx / Ni;
    int idxU = i + Ni * nelements * K4 * m;
    for (int p = 0; p < nabf4; p++) {
      int n1 = pa4[p];
      int n2 = pa4[p + 1];
      int nn = n2 - n1;
      for (int q = 0; q < nn; q++) {
        int c = pc4[n1 + q];
        int j1 = pb4[n1 + q];
        int j2 = pb4[n1 + q + Q4];
        int j3 = pb4[n1 + q + 2 * Q4];
        int k = 0;
        for (int i1 = 0; i1 < nelements; i1++) {
          double c1 =  sumU[idxU + Ni * (i1 + nelements * j1)];
          for (int i2 = i1; i2 < nelements; i2++) {
            double c2 = sumU[idxU + Ni * (i2 + nelements * j2)];
            double t12 = c * c1 * c2;
            for (int i3 = i2; i3 < nelements; i3++) {
              double c3 = sumU[idxU + Ni * (i3 + nelements * j3)];
              int kk = p + nabf4 * m + nabf4 * nrbf4 * k;
              d4[i + Ni * kk] += t12 * c3;
              k += 1;
            }
          }
        }
      }
    }
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::fourbodydescderiv(t_pod_1d dd4, int Ni, int Nij)
{
  int totalIterations = nrbf4 * Nij;
  if (nelements==1) {
    Kokkos::parallel_for("fourbodydescderiv1", totalIterations, KOKKOS_LAMBDA(int idx) {
      int j = idx / nrbf4;  // Derive the original j value
      int m = idx % nrbf4;  // Derive the original m value
      int idxU = idxi[j] + Ni * (K4 * m);
      int baseIdxJ = j + Nij * m; // Common index for rbf, rbfx, rbfy, rbfz
      double rbfBase = rbf[baseIdxJ];
      double rbfxBase = rbfx[baseIdxJ];
      double rbfyBase = rbfy[baseIdxJ];
      double rbfzBase = rbfz[baseIdxJ];

      for (int p = 0; p < nabf4; p++) {
        int n1 = pa4[p];
        int n2 = pa4[p + 1];
        int nn = n2 - n1;
        int kk = p + nabf4 * m;
        int ii = 3 * Nij * kk;
        int baseIdx = 3 * j + ii;

        for (int q = 0; q < nn; q++) {
          int idxNQ = n1 + q;  // Combine n1 and q into a single index
          int c = pc4[idxNQ];
          int j1 = pb4[idxNQ];
          int j2 = pb4[idxNQ + Q4];
          int j3 = pb4[idxNQ + 2 * Q4];
          double c1 = sumU[idxU + Ni * (j1)];
          double c2 = sumU[idxU + Ni * (j2)];
          double c3 = sumU[idxU + Ni * (j3)];
          double t12 = c * c1 * c2;          
          double t13 = c * c1 * c3;
          double t23 = c * c2 * c3;
          
          // Pre-calculate commonly used indices          
          int baseIdxJ3 = j + Nij * j3; // Common index for j3 terms
          int baseIdxJ2 = j + Nij * j2; // Common index for j2 terms
          int baseIdxJ1 = j + Nij * j1; // Common index for j1 terms

          // Temporary variables to store repeated calculations
          double abfBaseJ1 = abf[baseIdxJ1];
          double abfBaseJ2 = abf[baseIdxJ2];
          double abfBaseJ3 = abf[baseIdxJ3];

          // Update dd4 using pre-computed indices
          dd4[baseIdx]     += t12 * (abfx[baseIdxJ3] * rbfBase + rbfxBase * abfBaseJ3)
                            + t13 * (abfx[baseIdxJ2] * rbfBase + rbfxBase * abfBaseJ2)
                            + t23 * (abfx[baseIdxJ1] * rbfBase + rbfxBase * abfBaseJ1);
          dd4[baseIdx + 1] += t12 * (abfy[baseIdxJ3] * rbfBase + rbfyBase * abfBaseJ3)
                            + t13 * (abfy[baseIdxJ2] * rbfBase + rbfyBase * abfBaseJ2)
                            + t23 * (abfy[baseIdxJ1] * rbfBase + rbfyBase * abfBaseJ1);
          dd4[baseIdx + 2] += t12 * (abfz[baseIdxJ3] * rbfBase + rbfzBase * abfBaseJ3)
                            + t13 * (abfz[baseIdxJ2] * rbfBase + rbfzBase * abfBaseJ2)
                            + t23 * (abfz[baseIdxJ1] * rbfBase + rbfzBase * abfBaseJ1);
        }
      }
    });
  }
  else {        
    int N3 = 3*Nij * nabf4 * nrbf4;
    Kokkos::parallel_for("fourbodydescderiv2", totalIterations, KOKKOS_LAMBDA(int idx) {
      int j = idx / nrbf4;  // Derive the original j value
      int m = idx % nrbf4;  // Derive the original m value
    
      int idxM = j + Nij * m;
      // Temporary variables to store frequently used products
      double rbfM = rbf[idxM];
      double rbfxM = rbfx[idxM];
      double rbfyM = rbfy[idxM];
      double rbfzM = rbfz[idxM];
      int typej = tj[j] - 1;

      for (int p = 0; p < nabf4; p++)  {
        int n1 = pa4[p];
        int n2 = pa4[p + 1];
        int nn = n2 - n1;
        int jpm = 3 * j + 3 * Nij * (p + nabf4 * m);

        for (int q = 0; q < nn; q++) {
          int c = pc4[n1 + q];
          int j1 = pb4[n1 + q];
          int j2 = pb4[n1 + q + Q4];
          int j3 = pb4[n1 + q + 2 * Q4];
          // Pre-calculate commonly used indices for j3, j2, j1, and m
          int idxJ3 = j + Nij * j3;
          int idxJ2 = j + Nij * j2;
          int idxJ1 = j + Nij * j1;
          int idx1 = idxi[j] + Ni * (nelements * j1 + nelements * K4 * m);
          int idx2 = idxi[j] + Ni * (nelements * j2 + nelements * K4 * m);
          int idx3 = idxi[j] + Ni * (nelements * j3 + nelements * K4 * m);

          // Temporary variables to store repeated calculations
          double abfJ1 = abf[idxJ1];
          double abfJ2 = abf[idxJ2];
          double abfJ3 = abf[idxJ3];
          double abfxJ1 = abfx[idxJ1];
          double abfxJ2 = abfx[idxJ2];
          double abfxJ3 = abfx[idxJ3];
          double abfyJ1 = abfy[idxJ1];
          double abfyJ2 = abfy[idxJ2];
          double abfyJ3 = abfy[idxJ3];
          double abfzJ1 = abfz[idxJ1];
          double abfzJ2 = abfz[idxJ2];
          double abfzJ3 = abfz[idxJ3];

          int k = 0;          
          for (int i1 = 0; i1 < nelements; i1++) {            
            double c1 = sumU[idx1 + Ni*i1];
            for (int i2 = i1; i2 < nelements; i2++) {
              double c2 = sumU[idx2 + Ni*i2];
              double t12 = c*(c1 * c2);  
              for (int i3 = i2; i3 < nelements; i3++) {                                                
                double c3 = sumU[idx3 + Ni*i3];                
                double t13 = c*(c1 * c3);
                double t23 = c*(c2 * c3);
                int baseIdx = jpm + N3 * k;
                
                // Compute contributions for each condition
                if (typej == i3) {
                    dd4[0 + baseIdx] += t12 * (abfxJ3 * rbfM + rbfxM * abfJ3);
                    dd4[1 + baseIdx] += t12 * (abfyJ3 * rbfM + rbfyM * abfJ3);
                    dd4[2 + baseIdx] += t12 * (abfzJ3 * rbfM + rbfzM * abfJ3);
                }
                if (typej == i2) {
                    dd4[0 + baseIdx] += t13 * (abfxJ2 * rbfM + rbfxM * abfJ2);
                    dd4[1 + baseIdx] += t13 * (abfyJ2 * rbfM + rbfyM * abfJ2);
                    dd4[2 + baseIdx] += t13 * (abfzJ2 * rbfM + rbfzM * abfJ2);
                }
                if (typej == i1) {
                    dd4[0 + baseIdx] += t23 * (abfxJ1 * rbfM + rbfxM * abfJ1);
                    dd4[1 + baseIdx] += t23 * (abfyJ1 * rbfM + rbfyM * abfJ1);
                    dd4[2 + baseIdx] += t23 * (abfzJ1 * rbfM + rbfzM * abfJ1);
                }
                k += 1;
              }
            }
          }
        }
      }
    });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::fourbodydesc23(t_pod_1d d23, t_pod_1d d2, t_pod_1d d3, int Ni)
{
  int totalIterations = n32 * n23 * Ni;
  Kokkos::parallel_for("fourbodydesc23", totalIterations, KOKKOS_LAMBDA(int idx) {
    int n = idx % Ni;
    int temp = idx / Ni;
    int i = temp % n23;
    int j = temp / n23;

    int indexDst = n + Ni * i + Ni * n23 * j;
    int indexSrc2 = n + Ni * ind23[i];
    int indexSrc3 = n + Ni * ind32[j];
    d23[indexDst] = d2[indexSrc2] * d3[indexSrc3];
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::fourbodydescderiv23(t_pod_1d dd23, t_pod_1d d2, t_pod_1d d3, t_pod_1d dd2, t_pod_1d dd3, int Ni, int N)
{
  int totalIterations = n32 * n23 * N;
  Kokkos::parallel_for("fourbodydescderiv23", totalIterations, KOKKOS_LAMBDA(int idx) {
    int n = idx % N;
    int temp = idx / N;
    int i = temp % n23;
    int j = temp / n23;

    int k = 3 * (n + N * i + N * n23 * j);        
    int k1 = 3 * n + 3 * N * ind23[i];
    int k2 = 3 * n + 3 * N * ind32[i];
    int m1 = idxi[n] + Ni * ind23[i];
    int m2 = idxi[n] + Ni * ind32[i];
    dd23[0 + k] = d2[m1] * dd3[0 + k2] + dd2[0 + k1] * d3[m2];
    dd23[1 + k] = d2[m1] * dd3[1 + k2] + dd2[1 + k1] * d3[m2];
    dd23[2 + k] = d2[m1] * dd3[2 + k2] + dd2[2 + k1] * d3[m2];
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::crossdesc(t_pod_1d d12, t_pod_1d d1, t_pod_1d d2, t_pod_1i ind1, t_pod_1i ind2, int n12, int Ni)
{
  int totalIterations = n12 * Ni;
  Kokkos::parallel_for("crossdesc", totalIterations, KOKKOS_LAMBDA(int idx) {
    int n = idx % Ni;
    int i = idx / Ni;

    d12[n + Ni * i] = d1[n + Ni * ind1[i]] * d2[n + Ni * ind2[i]];
  });
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION        
void PairPODKokkos<DeviceType>::crossdescderiv(t_pod_1d dd12, t_pod_1d d1, t_pod_1d d2, t_pod_1d dd1, t_pod_1d dd2,
        t_pod_1i ind1, t_pod_1i ind2, int n12, int Ni, int Nij)
{  
  int totalIterations = n12 * Nij;
  Kokkos::parallel_for("crossdescderiv", totalIterations, KOKKOS_LAMBDA(int idx) {
    int n = idx % Nij;
    int i = idx / Nij;

    int k = 3 * n + 3 * Nij * i;
    int k1 = 3 * n + 3 * Nij * ind1[i];
    int k2 = 3 * n + 3 * Nij * ind2[i];
    int m1 = idxi[n] + Ni * ind1[i];
    int m2 = idxi[n] + Ni * ind2[i];

    dd12[0 + k] = d1[m1] * dd2[0 + k2] + dd1[0 + k1] * d2[m2];
    dd12[1 + k] = d1[m1] * dd2[1 + k2] + dd1[1 + k1] * d2[m2];
    dd12[2 + k] = d1[m1] * dd2[2 + k2] + dd1[2 + k1] * d2[m2];
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatom_base_descriptors(int Ni, int Nij)
{  
  Kokkos::deep_copy(bd, 0.0);
  Kokkos::deep_copy(bdd, 0.0);
   
  auto d2 = Kokkos::subview(bd, std::make_pair(0, Ni * nl2));
  auto d3 = Kokkos::subview(bd, std::make_pair(Ni * nl2, Ni * (nl2 + nl3)));
  auto d4 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3), Ni * (nl2 + nl3 + nl4)));
  auto d23 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4), Ni * (nl2 + nl3 + nl4 + nl23)));
  auto d33 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl23), Ni * (nl2 + nl3 + nl4 + nl23 + nl33)));
  auto d34 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl23 + nl33), Ni * (nl2 + nl3 + nl4 + nl23 + nl33 + nl34)));
  auto d44 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl23 + nl33 + nl34), Ni * (nl2 + nl3 + nl4 + nl23 + nl33 + nl34 + nl44)));

  auto dd2 = Kokkos::subview(bdd, std::make_pair(0, 3 * Nij * nl2));
  auto dd3 = Kokkos::subview(bdd, std::make_pair(3 * Nij * nl2, 3 * Nij * (nl2 + nl3)));
  auto dd4 = Kokkos::subview(bdd, std::make_pair(3 * Nij * (nl2 + nl3), 3 * Nij * (nl2 + nl3 + nl4)));
  auto dd23 = Kokkos::subview(bdd, std::make_pair(3 * Nij * (nl2 + nl3 + nl4), 3 * Nij * (nl2 + nl3 + nl4 + nl23)));
  auto dd33 = Kokkos::subview(bdd, std::make_pair(3 * Nij * (nl2 + nl3 + nl4 + nl23), 3 * Nij * (nl2 + nl3 + nl4 + nl23 + nl33)));
  auto dd34 = Kokkos::subview(bdd, std::make_pair(3 * Nij * (nl2 + nl3 + nl4 + nl23 + nl33), 3 * Nij * (nl2 + nl3 + nl4 + nl23 + nl33 + nl34)));
  auto dd44 = Kokkos::subview(bdd, std::make_pair(3 * Nij * (nl2 + nl3 + nl4 + nl23 + nl33 + nl34), 3 * Nij * (nl2 + nl3 + nl4 + nl23 + nl33 + nl34 + nl44)));

  radialbasis(abf, abfx, abfy, abfz, Nij);
  matrixMultiply(abf, Phi, rbf, Nij, ns, nrbfmax);   
  matrixMultiply(abfx, Phi, rbfx, Nij, ns, nrbfmax);   
  matrixMultiply(abfy, Phi, rbfy, Nij, ns, nrbfmax);   
  matrixMultiply(abfz, Phi, rbfz, Nij, ns, nrbfmax);   
  
  if ((nl2>0) && (Nij>0)) {
    twobodydescderiv(d2, dd2, Ni, Nij);
  }

  if ((nl3 > 0) && (Nij>1)) {
    angularbasis(abftm, abftmx, abftmy, abftmz, Nij);
    radialangularsum(Ni, Nij);

    threebodydesc(d3, Ni);
    threebodydescderiv(dd3, Ni, Nij);

    if ((nl23>0) && (Nij>2)) {
      fourbodydesc23(d23, d2, d3, Ni);
      fourbodydescderiv23(dd23, d2, d3, dd2, dd3, Ni, Nij);
    }

    if ((nl33>0) && (Nij>3)) {
      crossdesc(d33, d3, d3, ind33l, ind33r, nl33, Ni);
      crossdescderiv(dd33, d3, d3, dd3, dd3, ind33l, ind33r, nl33, Ni, Nij);
    }
    
    if ((nl4 > 0) && (Nij>2)) {
      if (K4 < K3) {
        extractsumU(Ni);                
        fourbodydesc(d4, Ni);
        fourbodydescderiv(dd4, Ni, Nij);        
      }

      if ((nl34>0) && (Nij>4)) {
        crossdesc(d34, d3, d4, ind34l, ind34r, nl34, Ni);
        crossdescderiv(dd34, d3, d4, dd3, dd4, ind34l, ind34r, nl34, Ni, Nij);
      }

      if ((nl44>0) && (Nij>5)) {
        crossdesc(d44, d4, d4, ind44l, ind44r, nl44, Ni);
        crossdescderiv(dd44, d4, d4, dd4, dd4, ind44l, ind44r, nl44, Ni, Nij);
      }
    }
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatomenergyforce(int Ni, int Nij)
{  
  // calculate base descriptors and their derivatives with respect to atom coordinates
  blockatom_base_descriptors(Ni, Nij);  

  Kokkos::parallel_for("compute_ei", Ni, KOKKOS_LAMBDA(const int n) {
      double sum = coefficients(0 + nCoeffPerElement * (typeai(n) - 1));
      for (int m = 0; m < Mdesc; ++m) {
          sum += coefficients(1 + m + nCoeffPerElement * (typeai(n) - 1)) * bd(n + Ni * m);
      }
      ei(n) = sum;
  });
  
  Kokkos::parallel_for("compute_fij", Nij, KOKKOS_LAMBDA(const int n) {
    int n3 = 3 * n;
    int nc = nCoeffPerElement * (ti(n) - 1);  // Assuming ti is a 1-D Kokkos::View
    int N3 = 3 * Nij;
    fij(n3) = 0.0;
    fij(n3 + 1) = 0.0;
    fij(n3 + 2) = 0.0;
    for (int m = 0; m < Mdesc; m++) {
      fij(n3)     += coefficients(1 + m + nc) * bdd(n3 + N3 * m);
      fij(n3 + 1) += coefficients(1 + m + nc) * bdd(n3 + 1 + N3 * m);
      fij(n3 + 2) += coefficients(1 + m + nc) * bdd(n3 + 2 + N3 * m);
    }
  });    
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::tallyforce(int Nij) {
  Kokkos::parallel_for("TallyForce", Nij, KOKKOS_LAMBDA(int n) {
    int im = ai(n);
    int jm = aj(n);
    Kokkos::atomic_add(&f(im, 0), fij(3*n + 0));
    Kokkos::atomic_add(&f(im, 1), fij(3*n + 1));
    Kokkos::atomic_add(&f(im, 2), fij(3*n + 2));
    Kokkos::atomic_sub(&f(jm, 0), fij(3*n + 0));
    Kokkos::atomic_sub(&f(jm, 1), fij(3*n + 1));
    Kokkos::atomic_sub(&f(jm, 2), fij(3*n + 2));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::tallyenergy(int istart, int Ni) {
  
  auto policy = Kokkos::RangePolicy<DeviceType>(0, Ni);
  
  // For global energy tally
  if (eflag_global) {
    E_FLOAT local_eng_vdwl = 0.0;
    Kokkos::parallel_reduce("GlobalEnergyTally", policy, KOKKOS_LAMBDA(int k, E_FLOAT& update) {
        update += ei(k);
      }, local_eng_vdwl);

    // Update global energy on the host after the parallel region
    eng_vdwl += local_eng_vdwl;
  }

  // For per-atom energy tally
  if (eflag_atom) {
    Kokkos::parallel_for("PerAtomEnergyTally", policy, KOKKOS_LAMBDA(int k) {
        d_eatom(istart + k) += ei(k);
      });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::tallystress(int Nij) {
  
  // Define the execution policy with DeviceType
  auto policy = Kokkos::RangePolicy<DeviceType>(0, Nij);
  
  if (vflag_global) {
    for (int j=0; j<3; j++) {
      F_FLOAT sum = 0.0;  
      Kokkos::parallel_reduce("GlobalStressTally", policy, KOKKOS_LAMBDA(int k, F_FLOAT& update) {          
          int k3 = 3*k;
          update += rij(j + k3) * fij(j + k3);
        }, sum);
      virial[j] -= sum;    
    }

    F_FLOAT sum = 0.0;  
    Kokkos::parallel_reduce("GlobalStressTally", policy, KOKKOS_LAMBDA(int k, F_FLOAT& update) {
        int k3 = 3*k;
        update += rij(k3) * fij(1 + k3);
      }, sum);
    virial[3] -= sum;    
    
    sum = 0.0;  
    Kokkos::parallel_reduce("GlobalStressTally", policy, KOKKOS_LAMBDA(int k, F_FLOAT& update) {
        int k3 = 3*k;
        update += rij(k3) * fij(2 + k3);
      }, sum);
    virial[4] -= sum;    
    
    sum = 0.0;  
    Kokkos::parallel_reduce("GlobalStressTally", policy, KOKKOS_LAMBDA(int k, F_FLOAT& update) {
        int k3 = 3*k;
        update += rij(1+k3) * fij(2+k3);
      }, sum);
    virial[5] -= sum;    
  }

  if (vflag_atom) {
    Kokkos::parallel_for("PerAtomStressTally", policy, KOKKOS_LAMBDA(int k) {
        int i = ai(k);
        int j = aj(k);
        int k3 = 3*k;
        F_FLOAT v_local[6];
        v_local[0] = -rij(k3) * fij(k3 + 0);
        v_local[1] = -rij(k3 + 1) * fij(k3 + 1);
        v_local[2] = -rij(k3 + 2) * fij(k3 + 2);
        v_local[3] = -rij(k3 + 0) * fij(k3 + 1);
        v_local[4] = -rij(k3 + 0) * fij(k3 + 2);
        v_local[5] = -rij(k3 + 1) * fij(k3 + 2);
        
        for (int d = 0; d < 6; ++d) {
          Kokkos::atomic_add(&d_vatom(i, d), 0.5 * v_local[d]);
        }

        for (int d = 0; d < 6; ++d) {
          Kokkos::atomic_add(&d_vatom(j, d), 0.5 * v_local[d]);
        }
        
      });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::savematrix2binfile(std::string filename, t_pod_1d d_A, int nrows, int ncols)
{
  auto A = Kokkos::create_mirror_view(d_A);  
  Kokkos::deep_copy(A, d_A);          
  
  FILE *fp = fopen(filename.c_str(), "wb");
  double sz[2];
  sz[0] = (double) nrows;
  sz[1] = (double) ncols;
  fwrite( reinterpret_cast<char*>( sz ), sizeof(double) * (2), 1, fp);
  fwrite( reinterpret_cast<char*>( A.data() ), sizeof(double) * (nrows*ncols), 1, fp);
  fclose(fp);
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::saveintmatrix2binfile(std::string filename, t_pod_1i d_A, int nrows, int ncols)
{
  auto A = Kokkos::create_mirror_view(d_A);  
  Kokkos::deep_copy(A, d_A);          
  
  FILE *fp = fopen(filename.c_str(), "wb");
  int sz[2];
  sz[0] = nrows;
  sz[1] = ncols;
  fwrite( reinterpret_cast<char*>( sz ), sizeof(int) * (2), 1, fp);
  fwrite( reinterpret_cast<char*>( A.data() ), sizeof(int) * (nrows*ncols), 1, fp);
  fclose(fp);
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::savedatafordebugging()
{
  saveintmatrix2binfile("podkktypeai.bin", typeai, ni, 1);  
  saveintmatrix2binfile("podkknumij.bin", numij, ni+1, 1);  
  saveintmatrix2binfile("podkkai.bin", ai, nij, 1);  
  saveintmatrix2binfile("podkkaj.bin", aj, nij, 1);  
  saveintmatrix2binfile("podkkti.bin", ai, nij, 1);  
  saveintmatrix2binfile("podkktj.bin", aj, nij, 1);  
  saveintmatrix2binfile("podkkidxi.bin", idxi, nij, 1);     
  savematrix2binfile("podkkrbf.bin", rbf, nij,nrbfmax);
  savematrix2binfile("podkkrbfx.bin", rbfx, nij,nrbfmax);
  savematrix2binfile("podkkrbfy.bin", rbfy, nij,nrbfmax);
  savematrix2binfile("podkkrbfz.bin", rbfz, nij,nrbfmax);      
  int kmax = (K3 > ns) ? K3 : ns;
  savematrix2binfile("podkkabf.bin", abf, nij,kmax);
  savematrix2binfile("podkkabfx.bin", abfx, nij,kmax);
  savematrix2binfile("podkkabfy.bin", abfy, nij,kmax);
  savematrix2binfile("podkkabfz.bin", abfz, nij,kmax);            
  savematrix2binfile("podkkbdd.bin", bdd, 3*nij, Mdesc);      
  savematrix2binfile("podkkbd.bin", bd, ni, Mdesc);      
  savematrix2binfile("podkksumU.bin", sumU, ni, nelements * K3 * nrbfmax);      
  savematrix2binfile("podkkrij.bin", rij, 3*nij, 1);
  savematrix2binfile("podkkfij.bin", fij, 3*nij, 1);
  savematrix2binfile("podkkei.bin", ei, ni, 1);      
  
  error->all(FLERR, "Save data and stop the run for debugging");
}

/* ----------------------------------------------------------------------
   memory usage of arrays
------------------------------------------------------------------------- */

template<class DeviceType>
double PairPODKokkos<DeviceType>::memory_usage()
{
  double bytes = 0;

  return bytes;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairPODKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairPODKokkos<LMPHostType>;
#endif
}
