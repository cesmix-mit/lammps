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
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairPOD::PairPOD(LAMMPS *lmp) :
    Pair(lmp), fastpodptr(nullptr), tmpmem(nullptr),  rij(nullptr), fij(nullptr),
    ai(nullptr), aj(nullptr), ti(nullptr), tj(nullptr)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  peratom_warn = true;

  dim = 3;
  nablockmax = 1;
  nij = 0;
  nijmax = 0;
  szd = 0;
}

/* ---------------------------------------------------------------------- */

PairPOD::~PairPOD()
{
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

  double rcutsq = fastpodptr->rcut*fastpodptr->rcut;
  double evdwl = 0.0;

//     fastpodptr->timing = 1;
//     if (fastpodptr->timing == 1)
//       for (int i=0; i<20; i++) fastpodptr->comptime[i] = 0;

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int jnum = numneigh[i];

    // allocate temporary memory
    if (nijmax < jnum) {
      nijmax = MAX(nijmax, jnum);
      fastpodptr->free_temp_memory();
      fastpodptr->allocate_temp_memory(nijmax);
    }
    
    rij = &fastpodptr->tmpmem[0];    
    fij = &fastpodptr->tmpmem[3*nijmax];   
    tmpmem = &fastpodptr->tmpmem[6*nijmax]; 
    ai = &fastpodptr->tmpint[0];      
    aj = &fastpodptr->tmpint[nijmax]; 
    ti = &fastpodptr->tmpint[2*nijmax];
    tj = &fastpodptr->tmpint[3*nijmax];

    // get neighbor list for atom i
    lammpsNeighborList(x, firstneigh, type, map, numneigh, rcutsq, i);

    // compute atomic energy and force for atom i
    evdwl = fastpodptr->peratomenergyforce(fij, rij, tmpmem, ti, tj, nij);

    // tally atomic energy to global energy
    ev_tally_full(i,2.0*evdwl,0.0,0.0,0.0,0.0,0.0);

    // tally atomic force to global force
    tallyforce(f, fij, ai, aj, nij);

    // tally atomic stress
    if (vflag) {
      for (int jj = 0; jj < nij; jj++) {
        int j = aj[jj];
        ev_tally_xyz(i,j,nlocal,newton_pair,0.0,0.0,
                    fij[0 + 3*jj],fij[1 + 3*jj],fij[2 + 3*jj],
                    -rij[0 + 3*jj], -rij[1 + 3*jj], -rij[2 + 3*jj]);
      }
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
  
//   std::string pod_file = std::string(arg[2]);      // pod input file
//   std::string coeff_file = std::string(arg[3]);    // coefficient input file
//   std::string proj_file = "";
//   std::string centroid_file = "";
//   if (narg>5) {
//     proj_file = std::string(arg[4]);    // coefficient input file
//     centroid_file = std::string(arg[5]);    // coefficient input file
//     //printf("proj_file = %s\n", proj_file.c_str());
//     //printf("centroid_file = %s\n", centroid_file.c_str());
//     map_element2type(narg - 6, arg + 6);    
//   }
//   else {
//     map_element2type(narg - 4, arg + 4);
//   }

  std::string pod_file = std::string(arg[2]);      // pod input file
  std::string coeff_file = std::string(arg[3]);    // coefficient input file
  std::string proj_file = std::string(arg[4]);
  std::string centroid_file = std::string(arg[5]);
  map_element2type(narg - 6, arg + 6);    
  
  delete fastpodptr;
  fastpodptr = new EAPOD(lmp, pod_file, coeff_file, proj_file, centroid_file);

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
  peratom_warn = true;
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

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double PairPOD::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes;
}

void PairPOD::lammpsNeighborList(double **x, int **firstneigh, int *atomtypes, int *map,
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
      rij[nij * 3 + 0] = delx;
      rij[nij * 3 + 1] = dely;
      rij[nij * 3 + 2] = delz;
      ai[nij] = gi;
      aj[nij] = gj;
      ti[nij] = itype;
      tj[nij] = map[atomtypes[gj]] + 1;
      nij++;
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
