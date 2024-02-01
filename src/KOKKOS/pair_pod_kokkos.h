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

#include "pair_pod.h"
#include "kokkos_type.h"
#include "pair_kokkos.h"

class SplineInterpolator;

namespace LAMMPS_NS {

template<class DeviceType>
class PairPODKokkos : public PairPOD {
 public:
  struct TagPairPODComputeNeigh{};

  template<int NEIGHFLAG, int EVFLAG>
  struct TagPairPODComputeForce{};

  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairPODKokkos(class LAMMPS *);
  ~PairPODKokkos() override;

  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPODComputeNeigh,const typename Kokkos::TeamPolicy<DeviceType, TagPairPODComputeNeigh>::member_type& team) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPODComputeForce<NEIGHFLAG,EVFLAG>,const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPairPODComputeForce<NEIGHFLAG,EVFLAG>,const int& ii, EV_FLOAT&) const;
  
 protected:
  int inum, maxneigh, chunk_size, chunk_offset, idx_ms_combs_max, idx_sph_max;
  int host_flag;

  int eflag, vflag;

  int neighflag, max_ndensity;
  int nelements, lmax, nradmax, nradbase;

  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

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
  t_fparams d_cut_in, d_dcut_in; // inner cutoff

  typename AT::t_int_1d d_map;

  int need_dup;

  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template<typename DataType, typename Layout>
  using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template<typename DataType, typename Layout>
  using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> dup_f;
  DupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> dup_vatom;

  NonDupScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout> ndup_f;
  NonDupScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout> ndup_vatom;

  friend void pair_virial_fdotr_compute<PairPODKokkos>(PairPODKokkos*);

  void grow(int, int);
  void allocate() override;
  double memory_usage() override;

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const;

  KOKKOS_INLINE_FUNCTION
  void matrixMultiply(const Kokkos::View<double**, DeviceType>& A,
                    const Kokkos::View<double**, DeviceType>& B, Kokkos::View<double**, DeviceType>& C,
                    int N, int K, int M);

  KOKKOS_INLINE_FUNCTION
  void radialbasis(Kokkos::View<double**, DeviceType>& rbf, Kokkos::View<double**, DeviceType>& rbfx,
                   Kokkos::View<double**, DeviceType>& rbfy, Kokkos::View<double**, DeviceType>& rbfz,
                   const Kokkos::View<double**, DeviceType>& rij, double besselparams0, 
                   double besselparams1, double besselparams2, double rin,  double rmax, 
                   int besseldegree, int inversedegree, int nbesselpars, int N);
  
  KOKKOS_INLINE_FUNCTION
  void orthogonalradialbasis(Kokkos::View<double**, DeviceType>& rbf,
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
                 int nrbfmax, int N);
  
  KOKKOS_INLINE_FUNCTION
  void tallyforce(Kokkos::View<F_FLOAT**> fij, Kokkos::View<int*> ai, Kokkos::View<int*> aj, int N);

  KOKKOS_INLINE_FUNCTION
  void tallyenergy(Kokkos::View<E_FLOAT*> ei, int istart, int Ni);

  KOKKOS_INLINE_FUNCTION
  void tallystress(Kokkos::View<F_FLOAT**> fij, Kokkos::View<F_FLOAT**> rij, Kokkos::View<int*> ai, Kokkos::View<int*> aj, int N);
        
  KOKKOS_INLINE_FUNCTION
  void inner_cutoff(const double, const double, const double, double &, double &) const;
  
  template<class TagStyle>
  void check_team_size_for(int, int&, int);

  template<class TagStyle>
  void check_team_size_reduce(int, int&, int);

  // Utility routine which wraps computing per-team scratch size requirements for
  // ComputeNeigh, ComputeUi, and ComputeFusedDeidrj
  template <typename scratch_type>
  int scratch_size_helper(int values_per_team);

  typedef Kokkos::View<int*, DeviceType> t_pod_1i;
  typedef Kokkos::View<int**, DeviceType> t_pod_2i;
  typedef Kokkos::View<double*, DeviceType> t_pod_1d;
  typedef Kokkos::View<double**, DeviceType> t_pod_2d;
  typedef Kokkos::View<double**[3], DeviceType> t_pod_3d3;
  
  // short neigh list
  t_pod_1i d_ncount;
  t_pod_2i d_nearest;
  t_pod_2d d_mu;
  t_pod_2d d_rnorms;
  t_pod_3d3 d_rhats;  
  
  t_pod_1d e_atom;
  t_pod_3d3 f_ij;
  
  bool is_zbl; 
};
}    // namespace LAMMPS_NS

#endif
#endif
