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
   Contributing authors: Ngoc Cuong Nguyen (MIT) and Andrew Rohskopf (SNL)
------------------------------------------------------------------------- */

#include "pair_pod.h"

#include "eapod.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using MathConst::MY_PI;
using MathSpecial::powint;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairPOD::PairPOD(LAMMPS *lmp) : Pair(lmp), fastpodptr(nullptr)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  peratom_warn = false;

  ni = 0;
  nimax = 0;
  nij = 0;
  nijmax = 0;  
  atomBlockSize = 4096;
  nAtomBlocks = 0;

  rij = nullptr;
  fij = nullptr;
  ei = nullptr;
  typeai = nullptr;
  numij =  nullptr;
  idxi = nullptr;
  ai = nullptr;
  aj = nullptr;
  ti = nullptr;
  tj = nullptr;
  Phi = nullptr;
  rbf = nullptr;
  rbfx = nullptr;
  rbfy = nullptr;
  rbfz = nullptr;
  abf = nullptr;
  abfx = nullptr;
  abfy = nullptr;
  abfz = nullptr;
  sumU = nullptr;
  Centroids = nullptr;
  Proj = nullptr;
  bd = nullptr;
  bdd = nullptr;
  pd = nullptr;
  pdd = nullptr;
  coefficients = nullptr;
  pn3 = nullptr;
  pc3 = nullptr;
  pa4 = nullptr;
  pb4 = nullptr;
  pc4 = nullptr;
  ind23 = nullptr;
  ind32 = nullptr;
  ind33l = nullptr;
  ind33r = nullptr;
  ind34l = nullptr;
  ind34r = nullptr;
  ind44l = nullptr;
  ind44r = nullptr;
  elemindex = nullptr;  
}

/* ---------------------------------------------------------------------- */

PairPOD::~PairPOD()
{
  memory->destroy(rij);
  memory->destroy(fij);
  memory->destroy(ei);
  memory->destroy(typeai);
  memory->destroy(numij);
  memory->destroy(idxi);
  memory->destroy(ai);
  memory->destroy(aj);
  memory->destroy(ti);
  memory->destroy(tj);
  memory->destroy(Phi);
  memory->destroy(rbf);
  memory->destroy(rbfx);
  memory->destroy(rbfy);
  memory->destroy(rbfz);
  memory->destroy(abf);
  memory->destroy(abfx);
  memory->destroy(abfy);
  memory->destroy(abfz);
  memory->destroy(sumU);
  memory->destroy(Centroids);
  memory->destroy(Proj);
  memory->destroy(bd);
  memory->destroy(bdd);
  memory->destroy(pd);
  memory->destroy(pdd);
  memory->destroy(coefficients);
  memory->destroy(pn3);
  memory->destroy(pc3);
  memory->destroy(pa4);
  memory->destroy(pb4);
  memory->destroy(pc4);
  memory->destroy(ind23);
  memory->destroy(ind32);
  memory->destroy(ind33l);
  memory->destroy(ind33r);
  memory->destroy(ind34l);
  memory->destroy(ind34r);
  memory->destroy(ind44l);
  memory->destroy(ind44r);
  memory->destroy(elemindex);

  delete fastpodptr;
  
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairPOD::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // we must enforce using F dot r, since we have no energy or stress tally calls.
  vflag_fdotr = 1;

  if (peratom_warn && (vflag_atom || eflag_atom)) {
    peratom_warn = false;
    if (comm->me == 0)
      error->warning(FLERR, "Pair style pod does not support per-atom energies or stresses");
  }

  double **x = atom->x;
  double **f = atom->f;
  int **firstneigh = list->firstneigh;
  int *numneigh = list->numneigh;
  int *type = atom->type;
  int *ilist = list->ilist;
  int inum = list->inum;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  // initialize global descriptors to zero

  double rcutsq = rcut*rcut;
  double evdwl = 0.0;

//     fastpodptr->timing = 1;
//     if (fastpodptr->timing == 1)
//       for (int i=0; i<20; i++) fastpodptr->comptime[i] = 0;

  int blockMode = 1;
  if (blockMode==0) {
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int jnum = numneigh[i];

    // allocate temporary memory
    if (nijmax < jnum) {
      nijmax = MAX(nijmax, jnum);
      fastpodptr->free_temp_memory();
      fastpodptr->allocate_temp_memory(nijmax);
    }

    double *rij1 = &fastpodptr->tmpmem[0];    
    double *fij1 = &fastpodptr->tmpmem[3*nijmax];   
    double *tmp = &fastpodptr->tmpmem[6*nijmax]; 
    int *ai1 = &fastpodptr->tmpint[0];      
    int *aj1 = &fastpodptr->tmpint[nijmax]; 
    int *ti1 = &fastpodptr->tmpint[2*nijmax];
    int *tj1 = &fastpodptr->tmpint[3*nijmax];    
    lammpsNeighborList(rij1, ai1, aj1, ti1, tj1, x, firstneigh, type, map, numneigh, rcutsq, i);
    
    evdwl = fastpodptr->peratomenergyforce(fij1, rij1, &fastpodptr->tmpmem[6*nijmax], ti1, tj1, nij);

    // tally atomic energy to global energy
    ev_tally_full(i,2.0*evdwl,0.0,0.0,0.0,0.0,0.0);

    // tally atomic force to global force
    tallyforce(f, fij1, ai1, aj1, nij);

    // tally atomic stress
    if (vflag) {
      for (int jj = 0; jj < nij; jj++) {
        int j = aj1[jj];
        ev_tally_xyz(i,j,nlocal,newton_pair,0.0,0.0,
                    fij1[0 + 3*jj],fij1[1 + 3*jj],fij1[2 + 3*jj],
                    -rij1[0 + 3*jj], -rij1[1 + 3*jj], -rij1[2 + 3*jj]);
      }
    }    
  }
  }
  else if (blockMode == 1) {
 // determine the number of atom blocks and divide atoms into blocks
  nAtomBlocks = calculateNumberOfIntervals(inum, atomBlockSize);
  if (nAtomBlocks > 100) nAtomBlocks = 100; 
  divideInterval(atomBlocks, inum, nAtomBlocks);

  int nmax = 0;
  for (int block =0; block<nAtomBlocks; block++) {    
    int n = atomBlocks[block+1] - atomBlocks[block]; 
    if (nmax < n) nmax = n;
  }
  grow_atoms(nmax); // reallocate memory only if necessary  
  
  for (int block =0; block<nAtomBlocks; block++) {
    int gi1 = atomBlocks[block]-1;
    int gi2 = atomBlocks[block+1]-1;
    ni = gi2 - gi1; // total number of atoms in the current atom block
        
    NeighborCount(x, firstneigh, ilist, numneigh, rcutsq, gi1, gi2);                    
    nij = numberOfNeighbors(); // total number of pairs (i,j) in the current atom block        
    grow_pairs(nij); // reallocate memory only if necessary

    // get neighbor list for atoms i in the current atom block
    NeighborList(x, firstneigh, type, map, ilist, numneigh, rcutsq, gi1, gi2);       
    
    // compute atomic energy and force for the current atom block
    blockatomenergyforce(ei, fij, ni, nij);

    // tally atomic energy to global energy
    tallyenergy(ei, gi1, ni);

    // tally atomic force to global force
    tallyforce(f, fij, ai, aj, nij);

    // tally atomic stress
    if (vflag) tallystress(fij, rij, ai, aj, nlocal, nij);
        
    //savedatafordebugging();
  }    
  }

//   if (fastpodptr->timing == 1) {
//     for (int i=0; i<20; i++) printf("%g  ", fastpodptr->comptime[i]);
//     printf("\n");
//   }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPOD::settings(int narg, char ** /* arg */)
{
  if (narg > 0) error->all(FLERR, "Pair style pod accepts no arguments");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPOD::coeff(int narg, char **arg)
{
  const int np1 = atom->ntypes + 1;
  memory->destroy(setflag);
  memory->destroy(cutsq);
  memory->create(setflag, np1, np1, "pair:setflag");
  memory->create(cutsq, np1, np1, "pair:cutsq");
  delete[] map;
  map = new int[np1];
  allocated = 1;

  if (narg < 7) utils::missing_cmd_args(FLERR, "pair_coeff", error);
  
  std::string pod_file = std::string(arg[2]);      // pod input file
  std::string coeff_file = std::string(arg[3]);    // coefficient input file
  std::string proj_file = std::string(arg[4]);     // projection matrix file
  std::string centroid_file = std::string(arg[5]); // centroid matrix file
  map_element2type(narg - 6, arg + 6);    
  
  delete fastpodptr;
  fastpodptr = new EAPOD(lmp, pod_file, coeff_file, proj_file, centroid_file);

  copy_data_from_pod_class();
  rcut = fastpodptr->rcut;
  
  memory->destroy(fastpodptr->tmpmem);
  memory->destroy(fastpodptr->tmpint);

  for (int ii = 0; ii < np1; ii++)
    for (int jj = 0; jj < np1; jj++) cutsq[ii][jj] = fastpodptr->rcut * fastpodptr->rcut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairPOD::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "Pair style pod requires newton pair on");

  // need a full neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL);

  // reset flag to print warning about per-atom energies or stresses
  peratom_warn = false;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPOD::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  double rcut = 0.0;
  rcut = fastpodptr->rcut;

  return rcut;
}

void PairPOD::allocate()
{
  allocated = 1;
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double PairPOD::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes;
}

void PairPOD::lammpsNeighborList(double *rij1, int *ai1, int *aj1, int *ti1, int *tj1, 
                               double **x, int **firstneigh, int *atomtypes, int *map,
                               int *numneigh, double rcutsq, int gi)
{
  nij = 0;
  int itype = map[atomtypes[gi]] + 1;
  int m = numneigh[gi];
  for (int l = 0; l < m; l++) {           // loop over each atom around atom i
    int gj = firstneigh[gi][l];           // atom j
    double delx = x[gj][0] - x[gi][0];    // xj - xi
    double dely = x[gj][1] - x[gi][1];    // xj - xi
    double delz = x[gj][2] - x[gi][2];    // xj - xi
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq < rcutsq && rsq > 1e-20) {
      rij1[nij * 3 + 0] = delx;
      rij1[nij * 3 + 1] = dely;
      rij1[nij * 3 + 2] = delz;
      ai1[nij] = gi;
      aj1[nij] = gj;
      ti1[nij] = itype;
      tj1[nij] = map[atomtypes[gj]] + 1;
      nij++;
    }
  }
}

void PairPOD::NeighborCount(double **x, int **firstneigh, int *ilist, int *numneigh, double rcutsq, int gi1, int gi2)
{  
  for (int i=0; i<ni; i++) {
    int gi = ilist[gi1 + i];
    double xi0 = x[gi][0];    
    double xi1 = x[gi][1];    
    double xi2 = x[gi][2];        
    int m = numneigh[gi];
    int n = 0;
    for (int l = 0; l < m; l++) {           // loop over each atom around atom i
      int gj = firstneigh[gi][l];           // atom j
      double delx = x[gj][0] - xi0;    // xj - xi
      double dely = x[gj][1] - xi1;    // xj - xi
      double delz = x[gj][2] - xi2;    // xj - xi
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < rcutsq && rsq > 1e-20) n++;      
    }
    numij[1+i] = n;
  }  
}

int PairPOD::numberOfNeighbors()
{
  int n = 0;
  for (int i=1; i<=ni; i++) {
    n += numij[i];  
    numij[i] += numij[i-1];   
  }
  return n;
}

void PairPOD::NeighborList(double **x, int **firstneigh, int *atomtypes, int *map,
                               int *ilist, int *numneigh, double rcutsq, int gi1, int gi2)
{  
  for (int i=0; i<ni; i++) {
    int gi = ilist[gi1 + i];
    double xi0 = x[gi][0];    
    double xi1 = x[gi][1];    
    double xi2 = x[gi][2];    
    int itype = map[atomtypes[gi]] + 1;
    typeai[i] = itype;    
    int m = numneigh[gi];
    int nij0 = numij[i];    
    int k = 0;    
    for (int l = 0; l < m; l++) {           // loop over each atom around atom i
      int gj = firstneigh[gi][l];           // atom j
      double delx = x[gj][0] - xi0;    // xj - xi
      double dely = x[gj][1] - xi1;    // xj - xi
      double delz = x[gj][2] - xi2;    // xj - xi
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < rcutsq && rsq > 1e-20) {
        int nij1 = nij0 + k;
        rij[nij1 * 3 + 0] = delx;
        rij[nij1 * 3 + 1] = dely;
        rij[nij1 * 3 + 2] = delz;
        idxi[nij1] = i;
        ai[nij1] = gi;
        aj[nij1] = gj;
        ti[nij1] = itype;
        tj[nij1] = map[atomtypes[gj]] + 1;
        k++;
      }
    }
  }  
}

void PairPOD::tallyforce(double **force, double *fij,  int *ai, int *aj, int N)
{
  for (int n=0; n<N; n++) {
    int im =  ai[n];
    int jm =  aj[n];
    int nm = 3*n;
    force[im][0] += fij[0 + nm];
    force[im][1] += fij[1 + nm];
    force[im][2] += fij[2 + nm];
    force[jm][0] -= fij[0 + nm];
    force[jm][1] -= fij[1 + nm];
    force[jm][2] -= fij[2 + nm];
  }
}

void PairPOD::tallyenergy(double *ei, int istart, int Ni)
{
  if (eflag_global)       
    for (int k = 0; k < Ni; k++) eng_vdwl += ei[k];      

  if (eflag_atom) 
    for (int k = 0; k < Ni; k++) eatom[istart+k] += ei[k];            
}

/* ----------------------------------------------------------------------
   tally eng_vdwl and virial into global or per-atom accumulators
   for virial, have delx,dely,delz and fx,fy,fz
------------------------------------------------------------------------- */

void PairPOD::tallystress(double *fij, double *rij, int *ai, int *aj, int nlocal, int N)
{
  double v[6];

  if (vflag_global) {      
    for (int k = 0; k < N; k++) {
      int k3 = 3*k;
      v[0] = -rij[0 + k3]*fij[0 + k3]; // delx*fx;
      v[1] = -rij[1 + k3]*fij[1 + k3]; // dely*fy;
      v[2] = -rij[2 + k3]*fij[2 + k3]; // delz*fz;
      v[3] = -rij[0 + k3]*fij[1 + k3]; // delx*fy;
      v[4] = -rij[0 + k3]*fij[2 + k3]; // delx*fz;  
      v[5] = -rij[1 + k3]*fij[2 + k3]; // dely*fz;
      virial[0] += v[0];
      virial[1] += v[1];
      virial[2] += v[2];
      virial[3] += v[3];
      virial[4] += v[4];
      virial[5] += v[5];
    }      
  }

  if (vflag_atom) {
    for (int k = 0; k < N; k++) {
      int i = ai[k];
      int j = aj[k];
      int k3 = k3;
      v[0] = -rij[0 + k3]*fij[0 + k3]; // delx*fx;
      v[1] = -rij[1 + k3]*fij[1 + k3]; // dely*fy;
      v[2] = -rij[2 + k3]*fij[2 + k3]; // delz*fz;
      v[3] = -rij[0 + k3]*fij[1 + k3]; // delx*fy;
      v[4] = -rij[0 + k3]*fij[2 + k3]; // delx*fz;  
      v[5] = -rij[1 + k3]*fij[2 + k3]; // dely*fz;

      if (i < nlocal) {
        vatom[i][0] += 0.5*v[0];
        vatom[i][1] += 0.5*v[1];
        vatom[i][2] += 0.5*v[2];
        vatom[i][3] += 0.5*v[3];
        vatom[i][4] += 0.5*v[4];
        vatom[i][5] += 0.5*v[5];
      }
      if (j < nlocal) {
        vatom[j][0] += 0.5*v[0];
        vatom[j][1] += 0.5*v[1];
        vatom[j][2] += 0.5*v[2];
        vatom[j][3] += 0.5*v[3];
        vatom[j][4] += 0.5*v[4];
        vatom[j][5] += 0.5*v[5];
      }
    }
  }
}

void PairPOD::copy_data_from_pod_class() 
{
  nelements = fastpodptr->nelements; // number of elements 
  onebody = fastpodptr->onebody;   // one-body descriptors
  besseldegree = fastpodptr->besseldegree; // degree of Bessel functions
  inversedegree = fastpodptr->inversedegree; // degree of inverse functions
  nbesselpars = fastpodptr->nbesselpars;  // number of Bessel parameters
  nCoeffPerElement = fastpodptr->nCoeffPerElement; // number of coefficients per element = (nl1 + Mdesc*nClusters)
  ns = fastpodptr->ns;      // number of snapshots for radial basis functions
  nl1 = fastpodptr->nl1;  // number of one-body descriptors
  nl2 = fastpodptr->nl2;  // number of two-body descriptors
  nl3 = fastpodptr->nl3;  // number of three-body descriptors
  nl4 = fastpodptr->nl4;  // number of four-body descriptors
  nl23 = fastpodptr->nl23; // number of two-body x three-body descriptors
  nl33 = fastpodptr->nl33; // number of three-body x three-body descriptors
  nl34 = fastpodptr->nl34; // number of three-body x four-body descriptors
  nl44 = fastpodptr->nl44; // number of four-body x four-body descriptors
  n23 = fastpodptr->n23;
  n32 = fastpodptr->n32;
  nl = fastpodptr->nl;   // number of local descriptors
  nrbf2 = fastpodptr->nrbf2;
  nrbf3 = fastpodptr->nrbf3;
  nrbf4 = fastpodptr->nrbf4;
  nrbfmax = fastpodptr->nrbfmax; // number of radial basis functions
  nabf3 = fastpodptr->nabf3;     // number of three-body angular basis functions
  nabf4 = fastpodptr->nabf4;     // number of four-body angular basis functions  
  K3 = fastpodptr->K3;           // number of three-body monomials
  K4 = fastpodptr->K4;           // number of four-body monomials
  Q4 = fastpodptr->Q4;           // number of four-body monomial coefficients
  nClusters = fastpodptr->nClusters; // number of environment clusters
  nComponents = fastpodptr->nComponents; // number of principal components
  Mdesc = fastpodptr->Mdesc; // number of base descriptors 

  rin = fastpodptr->rin;
  rcut = fastpodptr->rcut;
  rmax = rcut - rin;
  besselparams[0] = fastpodptr->besselparams[0];
  besselparams[1] = fastpodptr->besselparams[1];
  besselparams[2] = fastpodptr->besselparams[2];
  
  memory->create(abftm, 4*K3, "abftm");
  memory->create(elemindex, nelements*nelements, "elemindex");
  for (int i=0; i<nelements*nelements; i++) elemindex[i] = fastpodptr->elemindex[i];

  memory->create(Phi, ns * ns, "pair_pod:Phi");
  for (int i=0; i<ns*ns; i++)    
      Phi[i] = fastpodptr->Phi[i];

  memory->create(coefficients, nCoeffPerElement * nelements, "pair_pod:coefficients");
  for (int i=0; i<nCoeffPerElement * nelements; i++)
      coefficients[i] = fastpodptr->coeff[i];

  memory->create(pn3, nabf3+1, "pn3"); // array stores the number of monomials for each degree
  memory->create(pq3, K3*2, "pq3"); // array needed for the recursive computation of the angular basis functions
  memory->create(pc3, K3, "pc3");   // array needed for the computation of the three-body descriptors
  memory->create(pa4, nabf4+1, "pa4"); // this array is a subset of the array {0, 1, 4, 10, 19, 29, 47, 74, 89, 119, 155, 209, 230, 275, 335, 425, 533, 561, 624, 714, 849, 949, 1129, 1345}
  memory->create(pb4, Q4*3, "pb4"); // array stores the indices of the monomials needed for the computation of the angular basis functions
  memory->create(pc4, Q4, "pc4");   // array of monomial coefficients needed for the computation of the four-body descriptors
  for (int i=0; i<nabf3+1; i++) pn3[i] = fastpodptr->pn3[i];
  for (int i=0; i<K3; i++) pc3[i] = fastpodptr->pc3[i];
  for (int i=0; i<K3*2; i++) pq3[i] = fastpodptr->pq3[i];
  for (int i=0; i<nabf4+1; i++) pa4[i] = fastpodptr->pa4[i];
  for (int i=0; i<Q4*3; i++) pb4[i] = fastpodptr->pb4[i];
  for (int i=0; i<Q4; i++) pc4[i] = fastpodptr->pc4[i];  
  
  memory->create(ind23, n23, "pair_pod:ind23");
  memory->create(ind32, n32, "pair_pod:ind32");
  memory->create(ind33l, nl33, "pair_pod:ind33l");
  memory->create(ind33r, nl33, "pair_pod:ind33r");
  memory->create(ind34l, nl34, "pair_pod:ind34l");
  memory->create(ind34r, nl34, "pair_pod:ind34r");
  memory->create(ind44l, nl44, "pair_pod:ind44l");
  memory->create(ind44r, nl44, "pair_pod:ind44r");
  for (int i=0; i<n23; i++) ind23[i] = fastpodptr->ind23[i];
  for (int i=0; i<n32; i++) ind32[i] = fastpodptr->ind32[i];
  for (int i=0; i<nl33; i++) ind33l[i] = fastpodptr->ind33l[i];
  for (int i=0; i<nl33; i++) ind33r[i] = fastpodptr->ind33r[i];
  for (int i=0; i<nl34; i++) ind34l[i] = fastpodptr->ind34l[i];
  for (int i=0; i<nl34; i++) ind34r[i] = fastpodptr->ind34r[i];
  for (int i=0; i<nl44; i++) ind44l[i] = fastpodptr->ind44l[i];
  for (int i=0; i<nl44; i++) ind44r[i] = fastpodptr->ind44r[i];  
}

void PairPOD::grow_atoms(int Ni)
{
  if (Ni > nimax) {
    memory->destroy(ei);
    memory->destroy(typeai);
    memory->destroy(numij);
    memory->destroy(sumU);
    memory->destroy(bd);
    memory->destroy(pd);
    nimax = Ni;
    memory->create(ei, nimax, "pair_pod:ei");
    memory->create(typeai, nimax, "pair_pod:typeai");
    memory->create(numij, nimax+1, "pair_pod:typeai");
    memory->create(sumU, nimax * nelements * K3 * nrbfmax , "pair_pod:sumU");
    memory->create(bd, nimax * Mdesc, "pair_pod:bd");
    memory->create(pd, nimax * nClusters, "pair_pod:pd");    
    
    for (int i=0; i<=nimax; i++) numij[i] = 0;
  }
}

void PairPOD::grow_pairs(int Nij)
{
  if (Nij > nijmax) {
    memory->destroy(rij);
    memory->destroy(fij);
    memory->destroy(idxi);
    memory->destroy(ai);
    memory->destroy(aj);
    memory->destroy(ti);
    memory->destroy(tj);
    memory->destroy(rbf);
    memory->destroy(rbfx);
    memory->destroy(rbfy);
    memory->destroy(rbfz);
    memory->destroy(abf);
    memory->destroy(abfx);
    memory->destroy(abfy);
    memory->destroy(abfz);    
    memory->destroy(bdd);
    memory->destroy(pdd);        
    nijmax = Nij;
    memory->create(rij, 3 * nijmax,  "pair_pod:r_ij");
    memory->create(fij, 3 * nijmax,  "pair_pod:f_ij");  
    memory->create(idxi, nijmax, "pair_pod:idxi");
    memory->create(ai, nijmax, "pair_pod:ai");
    memory->create(aj, nijmax, "pair_pod:aj");
    memory->create(ti, nijmax, "pair_pod:ti");
    memory->create(tj, nijmax, "pair_pod:tj");
    memory->create(rbf, nijmax * nrbfmax, "pair_pod:rbf");
    memory->create(rbfx, nijmax * nrbfmax, "pair_pod:rbfx");
    memory->create(rbfy, nijmax * nrbfmax, "pair_pod:rbfy");
    memory->create(rbfz, nijmax * nrbfmax, "pair_pod:rbfz");
    int kmax = (K3 > ns) ? K3 : ns;
    memory->create(abf, nijmax * kmax, "pair_pod:abf");
    memory->create(abfx, nijmax * kmax, "pair_pod:abfx");
    memory->create(abfy, nijmax * kmax, "pair_pod:abfy");
    memory->create(abfz, nijmax * kmax, "pair_pod:abfz");  
    memory->create(bdd, 3 * nijmax  * Mdesc, "pair_pod:bdd");
    memory->create(pdd, 3 * nijmax * nClusters, "pair_pod:pdd");      
  }
}

void PairPOD::divideInterval(int *intervals, int N, int M) 
{
//   int *intervals = malloc((M + 1) * sizeof(int));
//   if (intervals == NULL) {
//     perror("Failed to allocate memory");
//     exit(EXIT_FAILURE);
//   }
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

int PairPOD::calculateNumberOfIntervals(int N, int intervalSize) 
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

void PairPOD::radialbasis(double *rbft, double *rbftx, double *rbfty, double *rbftz, double *rij, int Nij)
{
  // Loop over all neighboring atoms
  for (int n=0; n<Nij; n++) {
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

    double alpha = besselparams[0];
    double t1 = (1.0-exp(-alpha));
    double t2 = exp(-alpha*r/rmax);
    double x0 =  (1.0 - t2)/t1;
    double dx0 = (alpha/rmax)*t2/t1;

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
      //int idxni = n + Nij*i;
      int idxni = i + ns*n;

      rbft[idxni] = b*f1*sinax;
      double drbftdr = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);
      rbftx[idxni] = drbftdr*dr1;
      rbfty[idxni] = drbftdr*dr2;
      rbftz[idxni] = drbftdr*dr3;

      sinax = sin(a*x1);
      //idxni = n + Nij*i + Nij*besseldegree*1;
      idxni = i + besseldegree + ns*n;

      rbft[idxni] = b*f1*sinax;
      drbftdr = b*(df1*sinax - f2*sinax + af1*cos(a*x1)*dx1);
      rbftx[idxni] = drbftdr*dr1;
      rbfty[idxni] = drbftdr*dr2;
      rbftz[idxni] = drbftdr*dr3;

      sinax = sin(a*x2);
      //idxni = n + Nij*i + Nij*besseldegree*2;
      idxni = i + besseldegree*2 + ns*n;
      rbft[idxni] = b*f1*sinax;
      drbftdr = b*(df1*sinax - f2*sinax + af1*cos(a*x2)*dx2);
      rbftx[idxni] = drbftdr*dr1;
      rbfty[idxni] = drbftdr*dr2;
      rbftz[idxni] = drbftdr*dr3;
    }
  
    // Calculate fcut/dij and dfcut/dij
    f1 = fcut/dij;
    for (int i=0; i<inversedegree; i++) {
      int p = besseldegree*nbesselpars + i;
      //int idxni = n + Nij*p;      
      int idxni = p + ns*n;
      double a = powint(dij, i+1);

      rbft[idxni] = fcut/a;

      double drbftdr = (dfcut - (i+1.0)*f1)/a;
      rbftx[idxni] = drbftdr*dr1;
      rbfty[idxni] = drbftdr*dr2;
      rbftz[idxni] = drbftdr*dr3;
    }
  }
}

// void matrixMultiply(double *a, double *b, double *c, int r1, int c1, int c2) 
// {
//   for (int idx=0; idx<r1*c2*; ++)  
//     int j = idx / r1;  // Calculate column index
//     int i = idx % r1;  // Calculate row index
//     double sum = 0.0;
//     for (int k = 0; k < c1; ++k) {
//         sum += a[i + r1*k] * b[k + c1*j];  // Manually calculate the 1D index
//     }
//     c[i + r1*j] = sum;  // Manually calculate the 1D index for c
//   }        
// }

void matrixMultiply(double *Phi, double *rbft, double *rbf, int nrbfmax, int ns, int Nij) 
{
  for (int idx=0; idx<nrbfmax*Nij; idx++)  {
    int j = idx / nrbfmax;  // pair index index
    int i = idx % nrbfmax;  // basis function index
    double sum = 0.0;
    for (int k = 0; k < ns; ++k) {
        sum += rbft[k + ns*j] * Phi[k + ns*i];  // Manually calculate the 1D index
    }
    rbf[i + nrbfmax*j] = sum;  // Manually calculate the 1D index for c
  }        
}

void PairPOD::orthogonalradialbasis(int Nij)
{
  radialbasis(abf, abfx, abfy, abfz, rij, Nij);
  matrixMultiply(Phi, abf, rbf, nrbfmax, ns,  Nij); 
  matrixMultiply(Phi, abfx, rbfx, nrbfmax, ns,  Nij); 
  matrixMultiply(Phi, abfy, rbfy, nrbfmax, ns,  Nij); 
  matrixMultiply(Phi, abfz, rbfz, nrbfmax, ns,  Nij); 

//   char chn = 'N';
//   double alpha = 1.0, beta = 0.0;
//   DGEMM(&chn, &chn, &Nij, &nrbfmax, &ns, &alpha, abf, &Nij, Phi, &ns, &beta, rbf, &Nij);
//   DGEMM(&chn, &chn, &Nij, &nrbfmax, &ns, &alpha, abfx, &Nij, Phi, &ns, &beta, rbfx, &Nij);
//   DGEMM(&chn, &chn, &Nij, &nrbfmax, &ns, &alpha, abfy, &Nij, Phi, &ns, &beta, rbfy, &Nij);
//   DGEMM(&chn, &chn, &Nij, &nrbfmax, &ns, &alpha, abfz, &Nij, Phi, &ns, &beta, rbfz, &Nij);    
}

void PairPOD::angularbasis(double *tm, double *tmu, double *tmv, double *tmw, int N)
{  
  // Initialize first angular basis function and its derivatives
  tm[0] = 1.0;
  tmu[0] = 0.0;
  tmv[0] = 0.0;
  tmw[0] = 0.0;

  // Loop over all neighboring atoms
  for (int j=0; j<N; j++) {
    // Calculate relative positions of neighboring atoms and atom i
    double x = rij[0+3*j];
    double y = rij[1+3*j];
    double z = rij[2+3*j];

    // Calculate various terms for derivatives
    double xx = x*x;
    double yy = y*y;
    double zz = z*z;
    double xy = x*y;
    double xz = x*z;
    double yz = y*z;

    // Calculate distance between neighboring atoms and unit vectors
    double dij = sqrt(xx + yy + zz);
    double u = x/dij;
    double v = y/dij;
    double w = z/dij;

    // Calculate derivatives of unit vectors
    double dij3 = dij*dij*dij;
    double dudx = (yy+zz)/dij3;
    double dudy = -xy/dij3;
    double dudz = -xz/dij3;

    double dvdx = -xy/dij3;
    double dvdy = (xx+zz)/dij3;
    double dvdz = -yz/dij3;

    double dwdx = -xz/dij3;
    double dwdy = -yz/dij3;
    double dwdz = (xx+yy)/dij3;

    // Initialize first angular basis function and its derivatives
//     abf[j] = tm[0];
//     abfx[j] = 0.0;
//     abfy[j] = 0.0;
//     abfz[j] = 0.0;
    int idxa = 0 + K3*j;
    abf[idxa] = tm[0];
    abfx[idxa] = 0.0;
    abfy[idxa] = 0.0;
    abfz[idxa] = 0.0;

    // Loop over all angular basis functions
    for (int n=1; n<K3; n++) {
      // Get indices for angular basis function
      int m = pq3[n]-1;
      int d = pq3[n + K3];

      // Calculate angular basis function and its derivatives using recursion relation
      if (d==1) {
        tm[n] = tm[m]*u;
        tmu[n] = tmu[m]*u + tm[m];
        tmv[n] = tmv[m]*u;
        tmw[n] = tmw[m]*u;
      }
      else if (d==2) {
        tm[n] = tm[m]*v;
        tmu[n] = tmu[m]*v;
        tmv[n] = tmv[m]*v + tm[m];
        tmw[n] = tmw[m]*v;
      }
      else if (d==3) {
        tm[n] = tm[m]*w;
        tmu[n] = tmu[m]*w;
        tmv[n] = tmv[m]*w;
        tmw[n] = tmw[m]*w + tm[m];
      }
//       abf[j + N*n] = tm[n];
//       abfx[j + N*n] = tmu[n]*dudx + tmv[n]*dvdx + tmw[n]*dwdx;
//       abfy[j + N*n] = tmu[n]*dudy + tmv[n]*dvdy + tmw[n]*dwdy;
//       abfz[j + N*n] = tmu[n]*dudz + tmv[n]*dvdz + tmw[n]*dwdz;
      idxa = n + K3*j;
      abf[idxa] = tm[n];
      abfx[idxa] = tmu[n]*dudx + tmv[n]*dvdx + tmw[n]*dwdx;
      abfy[idxa] = tmu[n]*dudy + tmv[n]*dvdy + tmw[n]*dwdy;
      abfz[idxa] = tmu[n]*dudz + tmv[n]*dvdz + tmw[n]*dwdz;      
    }
  }
}

void PairPOD::radialangularsum(int Ni, int Nij)
{
  // Initialize sumU to zero
  std::fill(sumU, sumU + Ni * nelements * K3 * nrbf3, 0.0);

  int totalIterations = nrbf3 * K3 * Nij;
  for (int idx = 0; idx < totalIterations; idx++) {
//       int n = idx % Nij;
//       int temp = idx / Nij;
//       int k = temp % K3;
//       int m = temp / K3;
    int k = idx % K3;
    int temp = idx / K3;
    int m = temp % nrbf3;
    int n = temp / nrbf3;

//       int ia = n + Nij * k;
//       int ib = n + Nij * m;
//       int ii = ia + Nij * K3 * m;
    int ia = k + K3 * n;
    int ib = m + nrbfmax * n;      

      // Update sumU with atomtype adjustment
    int tn = tj[n] - 1; // offset the atom type by 1, since atomtype is 1-based
    //sumU[idxi[n] + Ni * (tn + nelements * k + nelements * K3 * m)] += rbf[ib] * abf[ia];      
    sumU[tn + nelements*k + nelements*K3*m + nelements*K3*nrbf3*idxi[n]] += rbf[ib] * abf[ia];
  }
}

void PairPOD::twobodydescderiv(double *d2, double *dd2, int Ni, int Nij)
{
  // Calculate the two-body descriptors and their derivatives
  int totalIterations = nrbf2 * Nij;
  for (int idx = 0; idx < totalIterations; idx++) {
//       int m = idx / Nij; // Recalculate m
//       int n = idx % Nij; // Recalculate n
    int n = idx / nrbf2; // Recalculate m
    int m = idx % nrbf2; // Recalculate n

      //int i2 = n + Nij * m; // Index of the radial basis function for atom n and RBF m
    int i2 = m + nrbfmax * n; // Index of the radial basis function for atom n and RBF m
    int i1 = 3*(n + Nij * m + Nij * nrbf2 * (tj[n] - 1)); // Index of the descriptor for atom n, RBF m, and atom type tj[n]
    d2[idxi[n] + Ni * (m + nrbf2 * (tj[n] - 1))] += rbf[i2]; // Add the radial basis function to the corresponding descriptor
    dd2[0 + i1] = rbfx[i2]; // Add the derivative with respect to x to the corresponding descriptor derivative
    dd2[1 + i1] = rbfy[i2]; // Add the derivative with respect to y to the corresponding descriptor derivative
    dd2[2 + i1] = rbfz[i2]; // Add the derivative with respect to z to the corresponding descriptor derivative
//       int i2 = m + nrbfmax * n; // Index of the radial basis function for atom n and RBF m
//       int i1 = m + nrbf2 * (tj[n] - 1) + nrbf2*nelements*n; // Index of the descriptor for atom n, RBF m, and atom type tj[n]
//       d2[m + nrbf2 * (tj[n] - 1) + nrbf2*nelements*idxi[n]] += rbf[i2]; // Add the radial basis function to the corresponding descriptor
//       dd2[0 + 3 * i1] += rbfx[i2]; // Add the derivative with respect to x to the corresponding descriptor derivative
//       dd2[1 + 3 * i1] += rbfy[i2]; // Add the derivative with respect to y to the corresponding descriptor derivative
//       dd2[2 + 3 * i1] += rbfz[i2]; // Add the derivative with respect to z to the corresponding descriptor derivative      
  }
}

void PairPOD::threebodydesc(double *d3, int Ni)
{
  int totalIterations = nrbf3 * Ni;
  for (int idx = 0; idx < totalIterations; idx++) {
//     int i = idx % Ni;
//     int m = idx / Ni;
    int m = idx % nrbf3;
    int i = idx / nrbf3;    
    for (int p = 0; p < nabf3; p++) {   
      int n1 = pn3[p];
      int n2 = pn3[p + 1];
      int nn = n2 - n1;
      for (int q = 0; q < nn; q++) {
        int k = 0;
        for (int i1 = 0; i1 < nelements; i1++) {
          //double t1 = pc3[n1 + q] * sumU[i + Ni * (i1 + nelements * (n1 + q) + nelements * K3 * m)];
          double t1 = pc3[n1 + q] * sumU[i1 + nelements * (n1 + q) + nelements * K3 * m + nelements * K3 * nrbf3*i];
          for (int i2 = i1; i2 < nelements; i2++) {
            //d3[i + Ni * (p + nabf3 * m + nabf3 * nrbf3 * k)] += t1 * sumU[i + Ni * (i2 + nelements * (n1 + q) + nelements * K3 * m)];
            d3[i + Ni * (p + nabf3 * m + nabf3 * nrbf3 * k)] += t1 * sumU[i2 + nelements * (n1 + q) + nelements * K3 * m + nelements * K3 * nrbf3*i];
            //d3[p + nabf3 * m + nabf3 * nrbf3 * k + nabf3 * nrbf3* nelements*(nelements+1)/2*i] += t1 * sumU[i2 + nelements * (n1 + q) + nelements * K3 * m + nelements * K3 * nrbf3*i];
            k += 1;
          }
        }
      }
     }
  }
}

void PairPOD::threebodydescderiv(double *dd3, int Ni, int Nij)
{
  int totalIterations = nrbf3 * Nij;
  if (nelements==1) {
    for (int idx = 0; idx < totalIterations; ++idx) {
      int j = idx / nrbf3;       // Calculate j using integer division
      int m = idx % nrbf3;       // Calculate m using modulo operation
      //int idxR = j + Nij * m;  // Pre-compute the index for rbf
      int idxR = m + nrbfmax * j;  // Pre-compute the index for rbf
      double rbfBase = rbf[idxR];
      double rbfxBase = rbfx[idxR];
      double rbfyBase = rbfy[idxR];
      double rbfzBase = rbfz[idxR];

      for (int p = 0; p < nabf3; p++) {
        int n1 = pn3[p];
        int n2 = pn3[p + 1];
        int nn = n2 - n1;
        int baseIdx = 3 * j + 3 * Nij * (p + nabf3 * m);  // Pre-compute the base index for dd3
        //int baseIdx = 3 * (p + nabf3 * m + nabf3*nrbf3*j);  // Pre-compute the base index for dd3
        //int idxU = idxi[j] + Ni * (K3 * m);
        int idxU = K3 * m + K3*nrbf3*idxi[j];
        double tmp1 = 0;
        double tmp2 = 0;
        double tmp3 = 0;
        for (int q = 0; q < nn; q++) {                  
          int idxNQ = n1 + q;  // Combine n1 and q into a single index for pc3 and sumU
          //double t1 = pc3[idxNQ] * sumU[idxU + Ni * idxNQ];
          double t1 = pc3[idxNQ] * sumU[idxNQ + idxU];
          double f = 2.0 * t1;          
          //int idxA = j + Nij * idxNQ;  // Pre-compute the index for abf          
          int idxA = idxNQ + K3 * j;  // Pre-compute the index for abf          
          double abfA = abf[idxA];  

          // Use the pre-computed indices to update dd3
          tmp1 += f * (abfx[idxA] * rbfBase + rbfxBase * abfA);
          tmp2 += f * (abfy[idxA] * rbfBase + rbfyBase * abfA);
          tmp3 += f * (abfz[idxA] * rbfBase + rbfzBase * abfA);          
        }
        dd3[baseIdx]     = tmp1;
        dd3[baseIdx + 1] = tmp2;
        dd3[baseIdx + 2] = tmp3;                  
      }
    }
  }
  else {
    int N3 = 3 * Nij *  nabf3 * nrbf3;
    for (int idx = 0; idx < totalIterations; ++idx) {
      int j = idx / nrbf3;  // Derive the original j value
      int m = idx % nrbf3;  // Derive the original m value
      //int idxR = j + Nij * m;  // Pre-compute the index for rbf
      int idxR = m + nrbfmax * j;  // Pre-compute the index for rbf
      double rbfBase = rbf[idxR];
      double rbfxBase = rbfx[idxR];
      double rbfyBase = rbfy[idxR];
      double rbfzBase = rbfz[idxR];
      for (int p = 0; p < nabf3; p++) {
        int n1 = pn3[p];
        int n2 = pn3[p + 1];
        int nn = n2 - n1;
        int jmp = 3 * j + 3 * Nij * (p + nabf3 * m);
        //int jmp = 3 * (p + nabf3 * m + nabf3*nrbf3*nelements*(nelements+1)/2*j);
        for (int q = 0; q < nn; q++) {
          int idxNQ = n1 + q;  // Combine n1 and q into a single index
          //int idxU = idxi[j] + Ni * (nelements * idxNQ + nelements * K3 * m);
          int idxU = nelements * idxNQ + nelements * K3 * m + nelements*K3*nrbf3*idxi[j];
          //int idxA = j + Nij * idxNQ;  // Pre-compute the index for abf      
          int idxA = idxNQ + K3 * j;  // Pre-compute the index for abf          
          double abfA = abf[idxA];   
          double abfxA = abfx[idxA];
          double abfyA = abfy[idxA];
          double abfzA = abfz[idxA];
          for (int i1 = 0; i1 < nelements; i1++) {
            //double t1 = pc3[idxNQ] * sumU[idxU + Ni*i1];
            double t1 = pc3[idxNQ] * sumU[i1 + idxU];
            int i2 = tj[j] - 1;
            int k = elemindex[i2 + nelements * i1];
            double f = (i1 == i2) ? 2.0 * t1 : t1;
            int ii = jmp + N3 * k;                     
            //int ii = jmp +  3 * nabf3 * nrbf3 * k;                     

            // Update dd3
            dd3[0 + ii] += f * (abfxA * rbfBase + rbfxBase * abfA);
            dd3[1 + ii] += f * (abfyA * rbfBase + rbfyBase * abfA);
            dd3[2 + ii] += f * (abfzA * rbfBase + rbfzBase * abfA);          
          }
        }
      }
    }
  }
}

void PairPOD::extractsumU(int Ni)
{
//   int totalIterations = nrbf4 * K4 * nelements * Ni;
//   for (int idx = 0; idx < totalIterations; idx++) {
//     int n = idx % Ni;
//     int temp = idx / Ni;
//     int i = temp % nelements;
//     temp = temp / nelements;
//     int k = temp % K4;
//     int m = temp / K4;
// 
// //     int indexDst = n + Ni * i + Ni * nelements * k + Ni * nelements * K4 * m;
// //     int indexSrc = n + Ni * i + Ni * nelements * k + Ni * nelements * K3 * m;    
//     int indexDst =  i +  nelements * k + nelements * K4 * m + nelements * K4 * nrbf4 * n;
//     int indexSrc =  i +  nelements * k + nelements * K3 * m + nelements * K3 * nrbf3 * n;        
//     sumU[indexDst] = sumU[indexSrc];
//   }
}

void PairPOD::fourbodydesc(double *d4, int Ni)
{
  int totalIterations = nrbf4 * Ni;
  int ne4 = nelements*(nelements+1)*(nelements+2)/6;
  for (int idx = 0; idx < totalIterations; idx++) {
//     int i = idx % Ni;
//     int m = idx / Ni;
    int m = idx % nrbf4;
    int i = idx / nrbf4;    
    //int idxU = i + Ni * nelements * K4 * m;
    int idxU = nelements * K3 * m + nelements * K3 * nrbf3 * i;
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
          //double c1 =  sumU[idxU + Ni * (i1 + nelements * j1)];
          double c1 =  sumU[idxU + i1 + nelements * j1];
          for (int i2 = i1; i2 < nelements; i2++) {
            //double c2 = sumU[idxU + Ni * (i2 + nelements * j2)];
            double c2 = sumU[idxU + i2 + nelements * j2];
            double t12 = c * c1 * c2;
            for (int i3 = i2; i3 < nelements; i3++) {
              //double c3 = sumU[idxU + Ni * (i3 + nelements * j3)];
              double c3 = sumU[idxU + i3 + nelements * j3];
              int kk = p + nabf4 * m + nabf4 * nrbf4 * k;
              d4[i + Ni * kk] += t12 * c3;              
              //d4[p + nabf4 * m + nabf4 * nrbf4 * k + nabf4 * nrbf4 * ne4 * i] += t12 * c3;              
              k += 1;
            }
          }
        }
      }
    }
  }
}

void PairPOD::fourbodydescderiv(double *dd4, int Ni, int Nij)
{
  if (nelements==1) {
    for (int idx = 0; idx < Nij * nrbf4; ++idx) {
      int j = idx / nrbf4;  // Derive the original j value
      int m = idx % nrbf4;  // Derive the original m value
      //int idxU = idxi[j] + Ni * (K4 * m);
      int idxU = K3 * m + K3*nrbf3*idxi[j];
      //int baseIdxJ = j + Nij * m; // Common index for rbf, rbfx, rbfy, rbfz
      int baseIdxJ = m + nrbfmax * j;  // Pre-compute the index for rbf
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
        //int baseIdx = 3*(p + nabf4*m + nabf4*nrbf4*j);
        double tmp1 = 0;
        double tmp2 = 0;
        double tmp3 = 0;
        for (int q = 0; q < nn; q++) {
          int idxNQ = n1 + q;  // Combine n1 and q into a single index
          int c = pc4[idxNQ];
          int j1 = pb4[idxNQ];
          int j2 = pb4[idxNQ + Q4];
          int j3 = pb4[idxNQ + 2 * Q4];
//           double c1 = sumU[idxU + Ni * (j1)];
//           double c2 = sumU[idxU + Ni * (j2)];
//           double c3 = sumU[idxU + Ni * (j3)];
          double c1 = sumU[idxU + j1];
          double c2 = sumU[idxU + j2];
          double c3 = sumU[idxU + j3];          
          double t12 = c * c1 * c2;          
          double t13 = c * c1 * c3;
          double t23 = c * c2 * c3;
          
          // Pre-calculate commonly used indices          
//           int baseIdxJ3 = j + Nij * j3; // Common index for j3 terms
//           int baseIdxJ2 = j + Nij * j2; // Common index for j2 terms
//           int baseIdxJ1 = j + Nij * j1; // Common index for j1 terms
          int baseIdxJ3 = j3 + K3 * j; // Common index for j3 terms
          int baseIdxJ2 = j2 + K3 * j; // Common index for j2 terms
          int baseIdxJ1 = j1 + K3 * j; // Common index for j1 terms

          // Temporary variables to store repeated calculations
          double abfBaseJ1 = abf[baseIdxJ1];
          double abfBaseJ2 = abf[baseIdxJ2];
          double abfBaseJ3 = abf[baseIdxJ3];

          // Update dd4 using pre-computed indices
          tmp1 += t12 * (abfx[baseIdxJ3] * rbfBase + rbfxBase * abfBaseJ3)
                            + t13 * (abfx[baseIdxJ2] * rbfBase + rbfxBase * abfBaseJ2)
                            + t23 * (abfx[baseIdxJ1] * rbfBase + rbfxBase * abfBaseJ1);
          tmp2 += t12 * (abfy[baseIdxJ3] * rbfBase + rbfyBase * abfBaseJ3)
                            + t13 * (abfy[baseIdxJ2] * rbfBase + rbfyBase * abfBaseJ2)
                            + t23 * (abfy[baseIdxJ1] * rbfBase + rbfyBase * abfBaseJ1);
          tmp3 += t12 * (abfz[baseIdxJ3] * rbfBase + rbfzBase * abfBaseJ3)
                            + t13 * (abfz[baseIdxJ2] * rbfBase + rbfzBase * abfBaseJ2)
                            + t23 * (abfz[baseIdxJ1] * rbfBase + rbfzBase * abfBaseJ1);
        }
        dd4[baseIdx]     = tmp1;
        dd4[baseIdx + 1] = tmp2;
        dd4[baseIdx + 2] = tmp3;                          
      }
    }
  }
  else {        
    int N3 = 3*Nij * nabf4 * nrbf4;
    int ne4 = nelements*(nelements+1)*(nelements+2)/6;
    int totalIterations = nrbf4 * Nij;
    for (int idx = 0; idx < totalIterations; idx++) {
      int j = idx / nrbf4;  // Derive the original j value
      int m = idx % nrbf4;  // Derive the original m value
    
      //int idxM = j + Nij * m;
      int idxM = m + nrbfmax * j;
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
        //int jpm = 3 * (p + nabf4 * m + nabf4 * nrbf4 * ne4 * j);

        for (int q = 0; q < nn; q++) {
          int c = pc4[n1 + q];
          int j1 = pb4[n1 + q];
          int j2 = pb4[n1 + q + Q4];
          int j3 = pb4[n1 + q + 2 * Q4];
          // Pre-calculate commonly used indices for j3, j2, j1, and m
//           int idxJ3 = j + Nij * j3;
//           int idxJ2 = j + Nij * j2;
//           int idxJ1 = j + Nij * j1;
          int idxJ3 = j3 + K3 * j;
          int idxJ2 = j2 + K3 * j;
          int idxJ1 = j1 + K3 * j;          
//           int idx1 = idxi[j] + Ni * (nelements * j1 + nelements * K4 * m);
//           int idx2 = idxi[j] + Ni * (nelements * j2 + nelements * K4 * m);
//           int idx3 = idxi[j] + Ni * (nelements * j3 + nelements * K4 * m);
          int idx1 = nelements * j1 + nelements * K3 * m + nelements * K3 * nrbf3 * idxi[j];
          int idx2 = nelements * j2 + nelements * K3 * m + nelements * K3 * nrbf3 * idxi[j];
          int idx3 = nelements * j3 + nelements * K3 * m + nelements * K3 * nrbf3 * idxi[j];

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
            //double c1 = sumU[idx1 + Ni*i1];
            double c1 = sumU[idx1 + i1];
            for (int i2 = i1; i2 < nelements; i2++) {
              //double c2 = sumU[idx2 + Ni*i2];
              double c2 = sumU[idx2 + i2];
              double t12 = c*(c1 * c2);  
              for (int i3 = i2; i3 < nelements; i3++) {                                                
                //double c3 = sumU[idx3 + Ni*i3];                
                double c3 = sumU[idx3 + i3];                
                double t13 = c*(c1 * c3);
                double t23 = c*(c2 * c3);
                int baseIdx = jpm + N3 * k;
                //int baseIdx = jpm +  3 * nabf4 * nrbf4 * k;
                
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
    }
  }
}

void PairPOD::fourbodydesc23(double *d23, double *d2, double *d3, int Ni)
{
  int totalIterations = n32 * n23 * Ni;
  for (int idx = 0; idx < totalIterations; idx++) {
    int n = idx % Ni;
    int temp = idx / Ni;
    int i = temp % n23;
    int j = temp / n23;

    int indexDst = n + Ni * i + Ni * n23 * j;
    int indexSrc2 = n + Ni * ind23[i];
    int indexSrc3 = n + Ni * ind32[j];
    d23[indexDst] = d2[indexSrc2] * d3[indexSrc3];
//     int indexDst = i + n23 * j + n23*n32*n;
//     int indexSrc2 = ind23[i] + nl2*n;
//     int indexSrc3 = ind32[j] + nl3*n;
//     d23[indexDst] = d2[indexSrc2] * d3[indexSrc3];    
  }
}

void PairPOD::fourbodydescderiv23(double* dd23, double *d2, double *d3, double *dd2, double *dd3, int *idxi, int Ni, int N)
{
  int totalIterations = n32 * n23 * N;
  for (int idx = 0; idx < totalIterations; idx++) {
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
//     int k = 3 * (i + n23 * j + n23*n32*n);        
//     int k1 = 3 * (ind23[i] + nl2*n);
//     int k2 = 3 * (ind32[i] + nl3*n );
//     int m1 = ind23[i] + nl2*idxi[n];
//     int m2 = ind32[i] + nl3*idxi[n];
//     dd23[0 + k] = d2[m1] * dd3[0 + k2] + dd2[0 + k1] * d3[m2];
//     dd23[1 + k] = d2[m1] * dd3[1 + k2] + dd2[1 + k1] * d3[m2];
//     dd23[2 + k] = d2[m1] * dd3[2 + k2] + dd2[2 + k1] * d3[m2];    
  }
}

void PairPOD::crossdesc(double *d12, double *d1, double *d2, int *ind1, int *ind2, int n12, int Ni)
{
  int totalIterations = n12 * Ni;
  for (int idx = 0; idx < totalIterations; idx++) {
    int n = idx % Ni;
    int i = idx / Ni;

    d12[n + Ni * i] = d1[n + Ni * ind1[i]] * d2[n + Ni * ind2[i]];
  }
}

void PairPOD::crossdescderiv(double *dd12, double *d1, double *d2, double *dd1, double *dd2,
        int *ind1, int *ind2, int *idxi, int n12, int Ni, int Nij)
{  
  int totalIterations = n12 * Nij;
  for (int idx = 0; idx < totalIterations; idx++) {
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
  }
}

void PairPOD::crossdesc(double *d12, double *d1, double *d2, int *ind1, int *ind2, 
        int n12, int nd1, int nd2, int Ni)
{
  int totalIterations = n12 * Ni;
  for (int idx = 0; idx < totalIterations; idx++) {
    int i = idx % n12;
    int n = idx / n12;

    d12[i + n12*n] = d1[ind1[i] + nd1*n] * d2[ind2[i] + nd2*n];
  }
}

void PairPOD::crossdescderiv(double *dd12, double *d1, double *d2, double *dd1, double *dd2,
        int *ind1, int *ind2, int *idxi, int n12, int nd1, int nd2, int Ni, int Nij)
{  
  int totalIterations = n12 * Nij;
  for (int idx = 0; idx < totalIterations; idx++) {
    int i = idx % n12;
    int n = idx / n12;
    
    int k = 3 * i + 3 * n12 * n;
    int k1 = 3 * ind1[i] + 3 * nd1 * n;
    int k2 = 3 * ind2[i] + 3 * nd2 * n;
    int m1 = ind1[i] + nd1*idxi[n];
    int m2 = ind2[i] + nd2*idxi[n];

    dd12[0 + k] = d1[m1] * dd2[0 + k2] + dd1[0 + k1] * d2[m2];
    dd12[1 + k] = d1[m1] * dd2[1 + k2] + dd1[1 + k1] * d2[m2];
    dd12[2 + k] = d1[m1] * dd2[2 + k2] + dd1[2 + k1] * d2[m2];
  }
}

void PairPOD::blockatombase_descriptors(double *bd1, double *bdd1, int Ni, int Nij)
{
  for (int i=0; i<Ni*Mdesc; i++) bd1[i] = 0.0;
  for (int i=0; i<3*Nij*Mdesc; i++) bdd1[i] = 0.0;

  double *d2 =  &bd1[0]; // nl2
  double *d3 =  &bd1[Ni*nl2]; // nl3
  double *d4 =  &bd1[Ni*(nl2 + nl3)]; // nl4
  double *d23 =  &bd1[Ni*(nl2 + nl3 + nl4)]; // nl23
  double *d33 =  &bd1[Ni*(nl2 + nl3 + nl4 + nl23)]; // nl33
  double *d34 =  &bd1[Ni*(nl2 + nl3 + nl4 + nl23 + nl33)]; // nl34
  double *d44 =  &bd1[Ni*(nl2 + nl3 + nl4 + nl23 + nl33 + nl34)]; // nl44

  double *dd2 = &bdd1[0]; // 3*Nj*nl2  
  double *dd3 = &bdd1[3*Nij*nl2]; // 3*Nj*nl3  
  double *dd4 = &bdd1[3*Nij*(nl2+nl3)]; // 3*Nj*nl4
  double *dd23 = &bdd1[3*Nij*(nl2+nl3+nl4)]; // 3*Nj*nl23
  double *dd33 = &bdd1[3*Nij*(nl2+nl3+nl4+nl23)]; // 3*Nj*nl33
  double *dd34 = &bdd1[3*Nij*(nl2+nl3+nl4+nl23+nl33)]; // 3*Nj*nl34
  double *dd44 = &bdd1[3*Nij*(nl2+nl3+nl4+nl23+nl33+nl34)]; // 3*Nj*nl44

  orthogonalradialbasis(Nij);

  if ((nl2>0) && (Nij>0)) {
    twobodydescderiv(d2, dd2, Ni, Nij);
  }

  if ((nl3 > 0) && (Nij>1)) {
    angularbasis(abftm, &abftm[K3], &abftm[2*K3], &abftm[3*K3], Nij);
    radialangularsum(Ni, Nij);

    threebodydesc(d3, Ni);
    threebodydescderiv(dd3, Ni, Nij);

    if ((nl23>0) && (Nij>2)) {
      fourbodydesc23(d23, d2, d3, Ni);
      fourbodydescderiv23(dd23, d2, d3, dd2, dd3, idxi, Ni, Nij);
    }

    if ((nl33>0) && (Nij>3)) {
      crossdesc(d33, d3, d3, ind33l, ind33r, nl33, Ni);
      crossdescderiv(dd33, d3, d3, dd3, dd3, ind33l, ind33r, idxi, nl33, Ni, Nij);
//       crossdesc(d33, d3, d3, ind33l, ind33r, nl33, nl3, nl3, Ni);
//       crossdescderiv(dd33, d3, d3, dd3, dd3, ind33l, ind33r, idxi, nl33, nl3, nl3, Ni, Nij);      
    }
    
    if ((nl4 > 0) && (Nij>2)) {
      if (K4 < K3) {
        extractsumU(Ni);                
        fourbodydesc(d4, Ni);
        fourbodydescderiv(dd4, Ni, Nij);        
      }

      if ((nl34>0) && (Nij>4)) {
        crossdesc(d34, d3, d4, ind34l, ind34r, nl34, Ni);
        crossdescderiv(dd34, d3, d4, dd3, dd4, ind34l, ind34r, idxi, nl34, Ni, Nij);
//         crossdesc(d34, d3, d4, ind34l, ind34r, nl34, nl3, nl4, Ni);
//         crossdescderiv(dd34, d3, d4, dd3, dd4, ind34l, ind34r, idxi, nl34, nl3, nl4, Ni, Nij);        
      }

      if ((nl44>0) && (Nij>5)) {
        crossdesc(d44, d4, d4, ind44l, ind44r, nl44, Ni);
        crossdescderiv(dd44, d4, d4, dd4, dd4, ind44l, ind44r, idxi, nl44, Ni, Nij);
//         crossdesc(d44, d4, d4, ind44l, ind44r, nl44, nl4, nl4, Ni);
//         crossdescderiv(dd44, d4, d4, dd4, dd4, ind44l, ind44r, idxi, nl44, nl4, nl4, Ni, Nij);        
      }
    }
  }
}

void PairPOD::blockatomenergyforce(double *ei, double *fij, int Ni, int Nij)
{  
  // calculate base descriptors and their derivatives with respect to atom coordinates
  blockatombase_descriptors(bd, bdd, Ni, Nij);  
  
  for (int n=0; n<Ni; n++) {
    ei[n] = coefficients[0 + nCoeffPerElement*(typeai[n]-1)];
    for (int m=0; m<Mdesc; m++)     
      ei[n] += coefficients[1 + m + nCoeffPerElement*(typeai[n]-1)]*bd[n + Ni*m];
  }

  //for (int n=0; n<3*Nij; n++) fij[n] = 0.0;
  for (int n=0; n<Nij; n++) {
    int n3 = 3*n;
    int nc = nCoeffPerElement*(ti[n]-1);
    int N3 = 3*Nij;
    fij[0 + n3] = 0.0;
    fij[1 + n3] = 0.0;
    fij[2 + n3] = 0.0;
    for (int m=0; m<Mdesc; m++) {    
      fij[0 + n3] += coefficients[1 + m + nc]*bdd[0 + n3 + N3*m];
      fij[1 + n3] += coefficients[1 + m + nc]*bdd[1 + n3 + N3*m];
      fij[2 + n3] += coefficients[1 + m + nc]*bdd[2 + n3 + N3*m];
    }
  }      
}

void PairPOD::savematrix2binfile(std::string filename, double *A, int nrows, int ncols)
{
  FILE *fp = fopen(filename.c_str(), "wb");
  double sz[2];
  sz[0] = (double) nrows;
  sz[1] = (double) ncols;
  fwrite( reinterpret_cast<char*>( sz ), sizeof(double) * (2), 1, fp);
  fwrite( reinterpret_cast<char*>( A ), sizeof(double) * (nrows*ncols), 1, fp);
  fclose(fp);
}

void PairPOD::saveintmatrix2binfile(std::string filename, int *A, int nrows, int ncols)
{
  FILE *fp = fopen(filename.c_str(), "wb");
  int sz[2];
  sz[0] = nrows;
  sz[1] = ncols;
  fwrite( reinterpret_cast<char*>( sz ), sizeof(int) * (2), 1, fp);
  fwrite( reinterpret_cast<char*>( A ), sizeof(int) * (nrows*ncols), 1, fp);
  fclose(fp);
}

void PairPOD::savedatafordebugging()
{
  saveintmatrix2binfile("podtypeai.bin", typeai, ni, 1);  
  saveintmatrix2binfile("podnumij.bin", numij, ni+1, 1);  
  saveintmatrix2binfile("podai.bin", ai, nij, 1);  
  saveintmatrix2binfile("podaj.bin", aj, nij, 1);  
  saveintmatrix2binfile("podti.bin", ai, nij, 1);  
  saveintmatrix2binfile("podtj.bin", aj, nij, 1);  
  saveintmatrix2binfile("podidxi.bin", idxi, nij, 1);     

  savematrix2binfile("podrbf.bin", rbf, nij,nrbfmax);
  savematrix2binfile("podrbfx.bin", rbfx, nij,nrbfmax);
  savematrix2binfile("podrbfy.bin", rbfy, nij,nrbfmax);
  savematrix2binfile("podrbfz.bin", rbfz, nij,nrbfmax);      
  int kmax = (K3 > ns) ? K3 : ns;
  savematrix2binfile("podabf.bin", abf, nij,kmax);
  savematrix2binfile("podabfx.bin", abfx, nij,kmax);
  savematrix2binfile("podabfy.bin", abfy, nij,kmax);
  savematrix2binfile("podabfz.bin", abfz, nij,kmax);            
  savematrix2binfile("podbdd.bin", bdd, 3*nij, Mdesc);      
  savematrix2binfile("podbd.bin", bd, ni, Mdesc);      
  savematrix2binfile("podsumU.bin", sumU, ni, nelements * K3 * nrbfmax);      

  savematrix2binfile("podrij.bin", rij, 3*nij, 1);
  savematrix2binfile("podfij.bin", fij, 3*nij, 1);
  savematrix2binfile("podei.bin", ei, ni, 1);           
  error->all(FLERR, "Save data and stop the run for debugging");
}

