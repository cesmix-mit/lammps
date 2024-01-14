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

#include "compute_pod_atom.h"

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

ComputePODAtom::ComputePODAtom(LAMMPS *lmp, int narg, char **arg) :
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
  
  cutmax = podptr->rcut;
  
  nmax = 0;
  nijmax = 0;
  pod = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputePODAtom::~ComputePODAtom()
{
  memory->destroy(pod);
  delete podptr;
}

/* ---------------------------------------------------------------------- */

void ComputePODAtom::init()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Compute pod requires a pair style be defined");

  if (cutmax > force->pair->cutforce)
    error->all(FLERR,"Compute pod cutoff is longer than pairwise cutoff");

  // need an occasional full neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);

  if (modify->get_compute_by_style("pod").size() > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute pod");
}


/* ---------------------------------------------------------------------- */

void ComputePODAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputePODAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow pod array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(pod);
    nmax = atom->nmax;
    int numdesc = podptr->Mdesc * podptr->nClusters;
    memory->create(pod,1 + 3*nmax, nmax*numdesc,"sna/atom:sna");
    array_atom = pod;
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);
  
  double **x = atom->x;
  int **firstneigh = list->firstneigh;
  int *numneigh = list->numneigh;
  int *type = atom->type;
  int *ilist = list->ilist;
  int inum = list->inum;
  int nlocal = atom->nlocal;
  int natoms = atom->natoms;
  
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
        
    if (nClusters>1) {
      // peratom env descriptors
      double *pd = &podptr->pd[0];
      double *pdd = &podptr->pdd[0];
      podptr->peratomenvironment_descriptors(pd, pdd, bd, bdd, tmpmem, ti[0] - 1,  nij);    
      for (int k = 0; k < nClusters; k++)
        for (int m = 0; m < Mdesc; m++) {
          int imk = m + Mdesc*k +  Mdesc*nClusters*i;
          pod[0][imk] = pd[k]*bd[m];     
          for (int n=0; nij; n++) {
            int ain = 3*ai[n];
            int ajn = 3*aj[n];
            int nm = 3*n + 3*nij*m;
            int nk = 3*n + 3*nij*k;
            pod[1 + ain][imk] += bdd[0 + nm]*pd[k] + bd[m]*pdd[0+nk];
            pod[2 + ain][imk] += bdd[1 + nm]*pd[k] + bd[m]*pdd[1+nk];
            pod[3 + ain][imk] += bdd[2 + nm]*pd[k] + bd[m]*pdd[2+nk];
            pod[1 + ajn][imk] -= bdd[0 + nm]*pd[k] + bd[m]*pdd[0+nk];
            pod[2 + ajn][imk] -= bdd[1 + nm]*pd[k] + bd[m]*pdd[1+nk];
            pod[3 + ajn][imk] -= bdd[2 + nm]*pd[k] + bd[m]*pdd[2+nk];
          }                  
        }
    }
    else {
      for (int m = 0; m < Mdesc; m++) {
       int im = m + Mdesc*i;
       pod[0][im] = bd[m];
       for (int n=0; nij; n++) {
          int ain = 3*ai[n];
          int ajn = 3*aj[n];
          int nm = 3*n + 3*nij*m;
          pod[1 + ain][im] += bdd[0 + nm];
          pod[2 + ain][im] += bdd[1 + nm];
          pod[3 + ain][im] += bdd[2 + nm];
          pod[1 + ajn][im] -= bdd[0 + nm];
          pod[2 + ajn][im] -= bdd[1 + nm];
          pod[3 + ajn][im] -= bdd[2 + nm];
        }       
      }
    }    
  }  
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputePODAtom::memory_usage()
{
  double bytes = 0.0;

  return bytes;
}


void ComputePODAtom::lammpsNeighborList(double **x, int **firstneigh, int *atomid, int *atomtypes, 
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
