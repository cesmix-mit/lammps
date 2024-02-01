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

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::grow(int natom, int maxneigh)
{
  
  //if (((int)fr.extent(0) < natom) || ((int)fr.extent(1) < maxneigh)) {
    // short neigh list
    MemKK::realloc_kokkos(d_ncount, "pod:ncount", natom);
    MemKK::realloc_kokkos(d_mu, "pod:mu", natom, maxneigh);
    MemKK::realloc_kokkos(d_rhats, "pod:rhats", natom, maxneigh);
    MemKK::realloc_kokkos(d_rnorms, "pod:rnorms", natom, maxneigh);
    MemKK::realloc_kokkos(d_nearest, "pod:nearest", natom, maxneigh);

    MemKK::realloc_kokkos(f_ij, "pod:f_ij", natom, maxneigh);
  //}
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::init_style()
{
  if (host_flag) {
    if (lmp->kokkos->nthreads > 1)
      error->all(FLERR,"Pair style pace/kk can currently only run on a single "
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
  PairPOD::coeff(narg,arg);

  int n = atom->ntypes + 1;
  MemKK::realloc_kokkos(d_map, "pod:map", n);

  MemKK::realloc_kokkos(k_cutsq, "pod:cutsq", n, n);
  d_cutsq = k_cutsq.template view<DeviceType>();

  MemKK::realloc_kokkos(k_scale, "pod:scale", n, n);
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
struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType>* nl): k_list(*nl) {}
  ~FindMaxNumNeighs() {k_list.copymode = 1;}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii, int& maxneigh) const {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (maxneigh < num_neighs) maxneigh = num_neighs;
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  if (host_flag) {
    atomKK->sync(Host,X_MASK|TYPE_MASK);
    PairPOD::compute(eflag_in,vflag_in);
    atomKK->modified(Host,F_MASK);
    return;
  }

  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

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
  if (!force->newton_pair)
    error->all(FLERR,"PairPODKokkos requires 'newton on'");

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  k_scale.template sync<DeviceType>();
  k_cutsq.template sync<DeviceType>();

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
  }

  maxneigh = 0;
  Kokkos::parallel_reduce("pod::find_maxneigh", inum, FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(maxneigh));

  int vector_length_default = 1;
  int team_size_default = 1;
  if (!host_flag)
    team_size_default = 32;

  int chunksize = 1000;
  chunk_size = MIN(chunksize,inum); // "chunksize" variable is set by user
  chunk_offset = 0;

  grow(chunk_size, maxneigh);

  EV_FLOAT ev;

  while (chunk_offset < inum) { // chunk up loop to prevent running out of memory

    EV_FLOAT ev_tmp;

    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    //Neigh
    {
      int vector_length = vector_length_default;
      int team_size = team_size_default;
      check_team_size_for<TagPairPODComputeNeigh>(chunk_size,team_size,vector_length);
      int scratch_size = scratch_size_helper<int>(team_size * maxneigh);
      typename Kokkos::TeamPolicy<DeviceType, TagPairPODComputeNeigh> policy_neigh(chunk_size,team_size,vector_length);
      policy_neigh = policy_neigh.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
      Kokkos::parallel_for("ComputeNeigh",policy_neigh,*this);
    }

    //ComputeForce
    {
      if (evflag) {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairPODComputeForce<HALF,1> > policy_force(0,chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairPODComputeForce<HALFTHREAD,1> > policy_force(0,chunk_size);
          Kokkos::parallel_reduce("ComputeForce",policy_force, *this, ev_tmp);
        }
      } else {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType,TagPairPODComputeForce<HALF,0> > policy_force(0,chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType,TagPairPODComputeForce<HALFTHREAD,0> > policy_force(0,chunk_size);
          Kokkos::parallel_for("ComputeForce",policy_force, *this);
        }
      }
    }
    ev += ev_tmp;

    chunk_offset += chunk_size;
  } // end while

  if (need_dup)
    Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup)
      Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  atomKK->modified(execution_space,F_MASK);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f     = decltype(dup_f)();
    dup_vatom = decltype(dup_vatom)();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::operator() (TagPairPODComputeNeigh,const typename Kokkos::TeamPolicy<DeviceType, TagPairPODComputeNeigh>::member_type& team) const
{
  const int ii = team.league_rank();
  const int i = d_ilist[ii + chunk_offset];
  const int itype = type[i];
  const X_FLOAT xtmp = x(i,0);
  const X_FLOAT ytmp = x(i,1);
  const X_FLOAT ztmp = x(i,2);
  const int jnum = d_numneigh[i];
  const int mu_i = d_map(type(i));

  // get a pointer to scratch memory
  // This is used to cache whether or not an atom is within the cutoff
  // If it is, inside is assigned to 1, otherwise -1
  const int team_rank = team.team_rank();
  const int scratch_shift = team_rank * maxneigh; // offset into pointer for entire team
  int* inside = (int*)team.team_shmem().get_shmem(team.team_size() * maxneigh * sizeof(int), 0) + scratch_shift;

  // loop over list of all neighbors within force cutoff
  // distsq[] = distance sq to each
  // rlist[] = distance vector to each
  // nearest[] = atom indices of neighbors

  int ncount = 0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,jnum),
      [&] (const int jj, int& count) {
    int j = d_neighbors(i,jj);
    j &= NEIGHMASK;

    const int jtype = type(j);

    const F_FLOAT delx = xtmp - x(j,0);
    const F_FLOAT dely = ytmp - x(j,1);
    const F_FLOAT delz = ztmp - x(j,2);
    const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    inside[jj] = -1;
    if (rsq < d_cutsq(itype,jtype)) {
     inside[jj] = 1;
     count++;
    }
  },ncount);

  d_ncount(ii) = ncount;

  Kokkos::parallel_scan(Kokkos::TeamThreadRange(team,jnum),
      [&] (const int jj, int& offset, bool final) {

    if (inside[jj] < 0) return;

    if (final) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      const F_FLOAT delx = xtmp - x(j,0);
      const F_FLOAT dely = ytmp - x(j,1);
      const F_FLOAT delz = ztmp - x(j,2);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      const F_FLOAT r = sqrt(rsq);
      const F_FLOAT rinv = 1.0/r;
      const int mu_j = d_map(type(j));
      d_mu(ii,offset) = mu_j;
      d_rnorms(ii,offset) = r;
      d_rhats(ii,offset,0) = -delx*rinv;
      d_rhats(ii,offset,1) = -dely*rinv;
      d_rhats(ii,offset,2) = -delz*rinv;
      d_nearest(ii,offset) = j;
    }
    offset++;
  });
  
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::operator() (TagPairPODComputeForce<NEIGHFLAG,EVFLAG>, const int& ii, EV_FLOAT& ev) const
{
  // The f array is duplicated for OpenMP, atomic for CUDA, and neither for Serial
  const auto v_f = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_f),decltype(ndup_f)>::get(dup_f,ndup_f);
  const auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const int i = d_ilist[ii + chunk_offset];
  const int itype = type(i);
  const double scale = d_scale(itype,itype);

  const int ncount = d_ncount(ii);

  F_FLOAT fitmp[3] = {0.0,0.0,0.0};
  for (int jj = 0; jj < ncount; jj++) {
    int j = d_nearest(ii,jj);

    double r_hat[3];
    r_hat[0] = d_rhats(ii, jj, 0);
    r_hat[1] = d_rhats(ii, jj, 1);
    r_hat[2] = d_rhats(ii, jj, 2);
    const double r = d_rnorms(ii, jj);
    const double delx = -r_hat[0]*r;
    const double dely = -r_hat[1]*r;
    const double delz = -r_hat[2]*r;

    const double fpairx = f_ij(ii, jj, 0);
    const double fpairy = f_ij(ii, jj, 1);
    const double fpairz = f_ij(ii, jj, 2);

    fitmp[0] += fpairx;
    fitmp[1] += fpairy;
    fitmp[2] += fpairz;
    a_f(j,0) -= fpairx;
    a_f(j,1) -= fpairy;
    a_f(j,2) -= fpairz;

    // tally per-atom virial contribution
    if (EVFLAG && vflag_either)
      v_tally_xyz<NEIGHFLAG>(ev, i, j, fpairx, fpairy, fpairz, delx, dely, delz);
  }

  a_f(i,0) += fitmp[0];
  a_f(i,1) += fitmp[1];
  a_f(i,2) += fitmp[2];

  // tally energy contribution
  if (EVFLAG && eflag_either) {
    const double evdwl = e_atom(ii);
    //ev_tally_full(i, 2.0 * evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (eflag_global) ev.evdwl += evdwl;
    if (eflag_atom) d_eatom[i] += evdwl;
  }
}

template<class DeviceType>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::operator() (TagPairPODComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const {
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG,EVFLAG>(TagPairPODComputeForce<NEIGHFLAG,EVFLAG>(), ii, ev);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
{
  // The vatom array is duplicated for OpenMP, atomic for CUDA, and neither for Serial

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vatom),decltype(ndup_vatom)>::get(dup_vatom,ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG,DeviceType>>();

  const E_FLOAT v0 = delx*fx;
  const E_FLOAT v1 = dely*fy;
  const E_FLOAT v2 = delz*fz;
  const E_FLOAT v3 = delx*fy;
  const E_FLOAT v4 = delx*fz;
  const E_FLOAT v5 = dely*fz;

  if (vflag_global) {
    ev.v[0] += v0;
    ev.v[1] += v1;
    ev.v[2] += v2;
    ev.v[3] += v3;
    ev.v[4] += v4;
    ev.v[5] += v5;
  }

  if (vflag_atom) {
    a_vatom(i,0) += 0.5*v0;
    a_vatom(i,1) += 0.5*v1;
    a_vatom(i,2) += 0.5*v2;
    a_vatom(i,3) += 0.5*v3;
    a_vatom(i,4) += 0.5*v4;
    a_vatom(i,5) += 0.5*v5;
    a_vatom(j,0) += 0.5*v0;
    a_vatom(j,1) += 0.5*v1;
    a_vatom(j,2) += 0.5*v2;
    a_vatom(j,3) += 0.5*v3;
    a_vatom(j,4) += 0.5*v4;
    a_vatom(j,5) += 0.5*v5;
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::tallyforce(Kokkos::View<F_FLOAT**> fij, Kokkos::View<int*> ai, Kokkos::View<int*> aj, int Nij) {
  // Define the execution policy with DeviceType
  auto policy = Kokkos::RangePolicy<DeviceType>(0, Nij);

  Kokkos::parallel_for("TallyForce", policy, KOKKOS_LAMBDA(int n) {
    int im = ai(n);
    int jm = aj(n);
    Kokkos::atomic_add(&f(im, 0), fij(n , 0));
    Kokkos::atomic_add(&f(im, 1), fij(n , 1));
    Kokkos::atomic_add(&f(im, 2), fij(n , 2));
    Kokkos::atomic_sub(&f(jm, 0), fij(n , 0));
    Kokkos::atomic_sub(&f(jm, 1), fij(n , 1));
    Kokkos::atomic_sub(&f(jm, 2), fij(n , 2));
  });
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::tallystress(Kokkos::View<F_FLOAT**> fij, Kokkos::View<F_FLOAT**> rij, Kokkos::View<int*> ai, Kokkos::View<int*> aj, int Nij) {
  
  // Define the execution policy with DeviceType
  auto policy = Kokkos::RangePolicy<DeviceType>(0, Nij);
  
  if (vflag_global) {
    for (int j=0; j<3; j++) {
      F_FLOAT sum = 0.0;  
      Kokkos::parallel_reduce("GlobalStressTally", policy, KOKKOS_LAMBDA(const int& k, F_FLOAT& update) {          
          update += rij(k, j) * fij(k, j);
        }, sum);
      virial[j] -= sum;    
    }

    F_FLOAT sum = 0.0;  
    Kokkos::parallel_reduce("GlobalStressTally", policy, KOKKOS_LAMBDA(const int& k, F_FLOAT& update) {
        update += rij(k, 0) * fij(k, 1);
      }, sum);
    virial[3] -= sum;    
    
    sum = 0.0;  
    Kokkos::parallel_reduce("GlobalStressTally", policy, KOKKOS_LAMBDA(const int& k, F_FLOAT& update) {
        int k3 = 3*k;
        update += rij(k, 0) * fij(k, 2);
      }, sum);
    virial[4] -= sum;    
    
    sum = 0.0;  
    Kokkos::parallel_reduce("GlobalStressTally", policy, KOKKOS_LAMBDA(const int& k, F_FLOAT& update) {
        int k3 = 3*k;
        update += rij(k, 1) * fij(k, 2);
      }, sum);
    virial[4] -= sum;    
  }

  if (vflag_atom) {
    Kokkos::parallel_for("PerAtomStressTally", policy, KOKKOS_LAMBDA(const int& k) {
        int i = ai(k);
        int j = aj(k);
        F_FLOAT v_local[6];
        v_local[0] = -rij(k, 0) * fij(k, 0);
        v_local[1] = -rij(k, 1) * fij(k, 1);
        v_local[2] = -rij(k, 2) * fij(k, 2);
        v_local[3] = -rij(k, 0) * fij(k, 1);
        v_local[4] = -rij(k, 0) * fij(k, 2);
        v_local[5] = -rij(k, 1) * fij(k, 2);
        
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
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::tallyenergy(Kokkos::View<E_FLOAT*> ei, int istart, int Ni) {
  
  auto policy = Kokkos::RangePolicy<DeviceType>(0, Ni);
  
  // For global energy tally
  if (eflag_global) {
    E_FLOAT local_eng_vdwl = 0.0;
    Kokkos::parallel_reduce("GlobalEnergyTally", policy, KOKKOS_LAMBDA(const int& k, E_FLOAT& update) {
        update += ei(k);
      }, local_eng_vdwl);

    // Update global energy on the host after the parallel region
    eng_vdwl += local_eng_vdwl;
  }

  // For per-atom energy tally
  if (eflag_atom) {
    Kokkos::parallel_for("PerAtomEnergyTally", policy, KOKKOS_LAMBDA(const int& k) {
        d_eatom(istart + k) += ei(k);
      });
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::matrixMultiply(const Kokkos::View<double**, DeviceType>& A,
                    const Kokkos::View<double**, DeviceType>& B, Kokkos::View<double**, DeviceType>& C,
                    int N, int K, int M) {
  using ExecutionSpace = typename DeviceType::execution_space;

  Kokkos::parallel_for("MatrixMultiply", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {N, M}),
    KOKKOS_LAMBDA(const int i, const int j) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    });
}

// // void EAPOD::radialbasis(double *rbf, double *rbfx, double *rbfy, double *rbfz, double *rij, double *besselparams, double rin,
// //         double rmax, int besseldegree, int inversedegree, int nbesselpars, int N)
// void EAPOD::radialbasis(double *rbf, double *rbfx, double *rbfy, double *rbfz, double *rij, int N)

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::radialbasis(Kokkos::View<double**, DeviceType>& rbf,
                 Kokkos::View<double**, DeviceType>& rbfx,
                 Kokkos::View<double**, DeviceType>& rbfy,
                 Kokkos::View<double**, DeviceType>& rbfz,
                 const Kokkos::View<double**, DeviceType>& rij,
                 double besselparams0, double besselparams1, double besselparams2,
                 double rin,  double rmax, int besseldegree, int inversedegree, int nbesselpars,
                 int N) {
  
  auto policy = Kokkos::RangePolicy<DeviceType>(0, N);
  
  Kokkos::parallel_for("RadialBasisFunction", policy, KOKKOS_LAMBDA(const int n) {
    double xij1 = rij(n, 0);
    double xij2 = rij(n, 1);
    double xij3 = rij(n, 2);

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

    double alpha = besselparams0;
    double t1 = (1.0-exp(-alpha));
    double t2 = exp(-alpha*r/rmax);
    double x0 =  (1.0 - t2)/t1;
    double dx0 = (alpha/rmax)*t2/t1;

    alpha = besselparams1;
    t1 = (1.0-exp(-alpha));
    t2 = exp(-alpha*r/rmax);
    double x1 =  (1.0 - t2)/t1;
    double dx1 = (alpha/rmax)*t2/t1;

    alpha = besselparams2;
    t1 = (1.0-exp(-alpha));
    t2 = exp(-alpha*r/rmax);
    double x2 =  (1.0 - t2)/t1;
    double dx2 = (alpha/rmax)*t2/t1;
    for (int i=0; i<besseldegree; i++) {
      double a = (i+1)*MY_PI;
      double b = (sqrt(2.0/(rmax))/(i+1));
      double af1 = a*f1;

      double sinax = sin(a*x0);

      rbf(n,i) = b*f1*sinax;
      double drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x0)*dx0);
      rbfx(n,i) = drbfdr*dr1;
      rbfy(n,i) = drbfdr*dr2;
      rbfz(n,i) = drbfdr*dr3;

      sinax = sin(a*x1);

      rbf(n,i + besseldegree) = b*f1*sinax;
      drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x1)*dx1);
      rbfx(n,i + besseldegree) = drbfdr*dr1;
      rbfy(n,i + besseldegree) = drbfdr*dr2;
      rbfz(n,i + besseldegree) = drbfdr*dr3;

      sinax = sin(a*x2);
      rbf(n,i + besseldegree*2) = b*f1*sinax;
      drbfdr = b*(df1*sinax - f2*sinax + af1*cos(a*x2)*dx2);
      rbfx(n,i + besseldegree*2) = drbfdr*dr1;
      rbfy(n,i + besseldegree*2) = drbfdr*dr2;
      rbfz(n,i + besseldegree*2) = drbfdr*dr3;
    }

    // Calculate fcut/dij and dfcut/dij
    f1 = fcut/dij;
    for (int i=0; i<inversedegree; i++) {
      int p = besseldegree*nbesselpars + i;
      double a = powint(dij, i+1);

      rbf(n,p) = fcut/a;

      double drbfdr = (dfcut - (i+1.0)*f1)/a;
      rbfx(n,p) = drbfdr*dr1;
      rbfy(n,p) = drbfdr*dr2;
      rbfz(n,p) = drbfdr*dr3;
    }            
  });
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::orthogonalradialbasis(Kokkos::View<double**, DeviceType>& rbf,
                 Kokkos::View<double**, DeviceType>& rbfx,
                 Kokkos::View<double**, DeviceType>& rbfy,
                 Kokkos::View<double**, DeviceType>& rbfz,
                 Kokkos::View<double**, DeviceType>& rbft,
                 Kokkos::View<double**, DeviceType>& rbfxt,
                 Kokkos::View<double**, DeviceType>& rbfyt,
                 Kokkos::View<double**, DeviceType>& rbfzt,               
                 const Kokkos::View<double**, DeviceType>& rij,
                 const Kokkos::View<double**, DeviceType>& Phi,
                 double besselparams0, double besselparams1, double besselparams2,
                 double rin,  double rmax, int besseldegree, int inversedegree, int nbesselpars,
                 int nrbfmax, int N) {
  
  radialbasis(rbft, rbfxt, rbfyt, rbfzt, rij, besselparams0, besselparams1, besselparams2, 
          rin, rmax, besseldegree, inversedegree, nbesselpars, N);
  
  int ns = inversedegree + besseldegree*nbesselpars;  
  matrixMultiply(rbft, Phi, rbf, N, ns, nrbfmax);
  matrixMultiply(rbfxt, Phi, rbfx, N, ns, nrbfmax);
  matrixMultiply(rbfyt, Phi, rbfy, N, ns, nrbfmax);
  matrixMultiply(rbfzt, Phi, rbfz, N, ns, nrbfmax);
}

// void EAPOD::orthogonalradialbasis(double *rbf, double *rbfx, double *rbfy, double *rbfz, double *rij, double *temp, int Nj)
// {
//   int n3 = Nj*ns;
//   double *rbft = &temp[0]; // Nj*ns
//   double *rbfxt = &temp[n3]; // Nj*ns
//   double *rbfyt = &temp[2*n3]; // Nj*ns
//   double *rbfzt = &temp[3*n3]; // Nj*ns
//   radialbasis(rbft, rbfxt, rbfyt, rbfzt, rij, Nj);
// 
//   char chn = 'N';
//   double alpha = 1.0, beta = 0.0;
//   DGEMM(&chn, &chn, &Nj, &nrbfmax, &ns, &alpha, rbft, &Nj, Phi, &ns, &beta, rbf, &Nj);
//   DGEMM(&chn, &chn, &Nj, &nrbfmax, &ns, &alpha, rbfxt, &Nj, Phi, &ns, &beta, rbfx, &Nj);
//   DGEMM(&chn, &chn, &Nj, &nrbfmax, &ns, &alpha, rbfyt, &Nj, Phi, &ns, &beta, rbfy, &Nj);
//   DGEMM(&chn, &chn, &Nj, &nrbfmax, &ns, &alpha, rbfzt, &Nj, Phi, &ns, &beta, rbfz, &Nj);
// }

            
template<class DeviceType>
template<class TagStyle>
void PairPODKokkos<DeviceType>::check_team_size_for(int inum, int &team_size, int vector_length) {
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelForTag());

  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
template<class TagStyle>
void PairPODKokkos<DeviceType>::check_team_size_reduce(int inum, int &team_size, int vector_length) {
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType,TagStyle>(inum,Kokkos::AUTO).team_size_max(*this,Kokkos::ParallelReduceTag());

  if (team_size*vector_length > team_size_max)
    team_size = team_size_max/vector_length;
}

template<class DeviceType>
template<typename scratch_type>
int PairPODKokkos<DeviceType>::scratch_size_helper(int values_per_team) {
  typedef Kokkos::View<scratch_type*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > ScratchViewType;

  return ScratchViewType::shmem_size(values_per_team);
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
