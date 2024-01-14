// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_pod.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "eapod.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

enum{SCALAR,VECTOR,ARRAY};

ComputePOD::ComputePOD(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), list(nullptr), pod(nullptr)
{  
  int nargmin = 4;

  if (narg < nargmin) error->all(FLERR, "Illegal compute {} command", style);
  
  std::string pod_file = std::string(arg[3]);      // pod input file
  std::string coeff_file = "";    // coefficient input file
  std::string proj_file = "";
  std::string centroid_file = "";
  if (narg>5) {
    proj_file = std::string(arg[4]);    // coefficient input file
    centroid_file = std::string(arg[5]);    // coefficient input file    
  } 
  
  podptr = new EAPOD(lmp, pod_file, coeff_file, proj_file, centroid_file);
  
  size_array_rows = 1 + 3*atom->natoms;  
  size_array_cols = podptr->nCoeffAll;
  cutmax = podptr->rcut;
    
  //size_peratom = 3*podptr->nCoeffPerElement*atom->ntypes;
  
  nmax = 0;
  nijmax = 0;
  pod = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputePOD::~ComputePOD()
{
  memory->destroy(pod);
  //memory->destroy(pod_peratom);
  delete podptr;
}

/* ---------------------------------------------------------------------- */

void ComputePOD::init()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Compute pod requires a pair style be defined");

  if (cutmax > force->pair->cutforce)
    error->all(FLERR,"Compute pod cutoff is longer than pairwise cutoff");

  // need an occasional full neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);

  if (modify->get_compute_by_style("pod").size() > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute pod");
  
 // allocate memory for global array
  memory->create(pod,size_array_rows,size_array_cols,
                 "pod:pod");
  array = pod;
}

/* ---------------------------------------------------------------------- */

void ComputePOD::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputePOD::compute_array()
{
  // int ntotal = atom->nlocal + atom->nghost;
  
  invoked_peratom = update->ntimestep;

//   // grow snap_peratom array if necessary
// 
//   if (atom->nmax > nmax) {
//     memory->destroy(pod_peratom);
//     nmax = atom->nmax;
//     memory->create(pod_peratom,nmax,size_peratom,
//                    "snap:snap_peratom");
//   }
  
  // clear global array

  for (int irow = 0; irow < size_array_rows; irow++)
    for (int icoeff = 0; icoeff < size_array_cols; icoeff++)
      pod[irow][icoeff] = 0.0;
  
  // clear local peratom array

//   for (int i = 0; i < ntotal; i++)
//     for (int icoeff = 0; icoeff < size_peratom; icoeff++)
//       pod_peratom[i][icoeff] = 0.0;
  
  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);
  
  double **x = atom->x;
  int **firstneigh = list->firstneigh;
  int *numneigh = list->numneigh;
  int *type = atom->type;
  int *ilist = list->ilist;
  int inum = list->inum;
  int nlocal = atom->nlocal;
  
  int nClusters = podptr->nClusters;
  int Mdesc = podptr->Mdesc;
  int nCoeffPerElement = podptr->nCoeffPerElement;
  double *bd = &podptr->bd[0];
  double *bdd = &podptr->bdd[0];
  
  double rcutsq = podptr->rcut*podptr->rcut;
  
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int jnum = numneigh[i];

    // allocate temporary memory
    if (nijmax < jnum) {
      nijmax = MAX(nijmax, jnum);
      podptr->free_temp_memory();
      podptr->allocate_temp_memory(nijmax);
    }
    
    rij = &podptr->tmpmem[0];    
    tmpmem = &podptr->tmpmem[3*nijmax]; 
    ai = &podptr->tmpint[0];      
    aj = &podptr->tmpint[nijmax]; 
    ti = &podptr->tmpint[2*nijmax];
    tj = &podptr->tmpint[3*nijmax];
        
    // get neighbor list for atom i
    lammpsNeighborList(x, firstneigh, atom->tag, type, numneigh, rcutsq, i);
    
    // peratom base descriptors
    podptr->peratombase_descriptors(bd, bdd, rij, tmpmem, ti, tj, nij);        
        
    pod[0][nCoeffPerElement*(ti[0]-1)] += 1.0; // one-body descriptor
    
    if (nClusters>1) {
      // peratom env descriptors
      double *pd = &podptr->pd[0];
      double *pdd = &podptr->pdd[0];
      podptr->peratomenvironment_descriptors(pd, pdd, bd, bdd, tmpmem, ti[0] - 1,  nij);    
      
      for (int j = 0; j < nClusters; j++) {
        for (int m=0; m<Mdesc; m++) {
          int k = nCoeffPerElement*(ti[0]-1) + 1 + m + j*Mdesc; // increment by 1 because of the one-body descriptor
          pod[0][k] += pd[j]*bd[m];
          int k3 = 3*nClusters*Mdesc*(ti[0]-1) + 3*(m + j*Mdesc); 
          for (int n=0; n<nij; n++) {
            int ain = 3*ai[n];
            int ajn = 3*aj[n];     
            int nm = 3*n + 3*nij*m;
            int nj = 3*n + 3*nij*j;
            pod[1+ain][k] += bdd[0 + nm]*pd[j] + bd[m]*pdd[0 + nj];
            pod[2+ain][k] += bdd[1 + nm]*pd[j] + bd[m]*pdd[1 + nj];
            pod[3+ain][k] += bdd[2 + nm]*pd[j] + bd[m]*pdd[2 + nj];
            pod[1+ajn][k] -= bdd[0 + nm]*pd[j] + bd[m]*pdd[0 + nj];
            pod[2+ajn][k] -= bdd[1 + nm]*pd[j] + bd[m]*pdd[1 + nj];
            pod[3+ajn][k] -= bdd[2 + nm]*pd[j] + bd[m]*pdd[2 + nj];
          }
        }
      }      
    }
    else {
      for (int m=0; m<Mdesc; m++) {
        int k = nCoeffPerElement*(ti[0]-1) + 1 + m; // increment by 1 because of the one-body descriptor
        pod[0][k] += bd[m];
        for (int n=0; n<nij; n++) {
          int ain = 3*ai[n];
          int ajn = 3*aj[n];     
          int nm = 3*n + 3*nij*m;
          pod[1+ain][k] += bdd[0 + nm];
          pod[2+ain][k] += bdd[1 + nm];
          pod[3+ain][k] += bdd[2 + nm];
          pod[1+ajn][k] -= bdd[0 + nm];
          pod[2+ajn][k] -= bdd[1 + nm];
          pod[3+ajn][k] -= bdd[2 + nm];
        }
      }      
    }    
  }
  
//   int nvalues = Mdesc*nClusters;
//   
//   for (int itype = 0; itype < atom->ntypes; itype++) {
//     int typeoffset_local = 3*nvalues*itype;
//     int typeoffset_global = (1+nvalues)*itype;
//     for (int icoeff = 0; icoeff < nvalues; icoeff++) {
//       for (int i = 0; i < ntotal; i++) {
//         int iglobal = atom->tag[i];
//         int irow = 3*(iglobal-1)+1;
//         pod[irow++][1+icoeff+typeoffset_global] += pod_peratom[i][0 + 3*icoeff + typeoffset_local];
//         pod[irow++][1+icoeff+typeoffset_global] += pod_peratom[i][1 + 3*icoeff + typeoffset_local];
//         pod[irow][1+icoeff+typeoffset_global] += pod_peratom[i][2 + 3*icoeff + typeoffset_local];
//       }
//     }
//   }
  
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputePOD::memory_usage()
{
  double bytes = 0.0;

  return bytes;
}


void ComputePOD::lammpsNeighborList(double **x, int **firstneigh, int *atomid, int *atomtypes, 
                               int *numneigh, double rcutsq, int gi)
{
  nij = 0;
  int itype = atomtypes[gi];
  int m = numneigh[gi];
  for (int l = 0; l < m; l++) {           // loop over each atom around atom i
    int gj = firstneigh[gi][l];           // atom j
    double delx = x[gj][0] - x[gi][0];    // xj - xi
    double dely = x[gj][1] - x[gi][1];    // xj - xi
    double delz = x[gj][2] - x[gi][2];    // xj - xi
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq < rcutsq && rsq > 1e-20) {
      rij[nij * 3 + 0] = delx;
      rij[nij * 3 + 1] = dely;
      rij[nij * 3 + 2] = delz;
      ai[nij] = atomid[gi];
      aj[nij] = atomid[gj];
      ti[nij] = itype;
      tj[nij] = atomtypes[gj];
      nij++;
    }
  }
}
