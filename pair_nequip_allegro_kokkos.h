/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(allegro/kk,PairAllegroKokkos<false>)
PairStyle(nequip/kk,PairAllegroKokkos<true>)

#else

#ifndef LMP_PAIR_ALLEGRO_KOKKOS_H
#define LMP_PAIR_ALLEGRO_KOKKOS_H

#include "pair_nequip_allegro.h"
#include "kokkos_base.h"
#include <pair_kokkos.h>


namespace LAMMPS_NS {

template<bool nequip_mode>
class PairAllegroKokkos : public PairNequIPAllegro<nequip_mode>, public KokkosBase {
 public:
  typedef PairNequIPAllegro<nequip_mode> super;
  using DeviceType = LMPDeviceType;
  using MemberType = typename Kokkos::TeamPolicy<DeviceType>::member_type;
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef LMPDeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairAllegroKokkos(class LAMMPS *);
  virtual ~PairAllegroKokkos();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  virtual void init_style();

  // === Route 2b device-resident per-layer feature halo (M6) ===
  // Bridge overrides: keep features ON the GPU, exchange via CommKokkos::*_comm_device.
  torch::Tensor forward_exchange_t(const torch::Tensor &node_features) override;
  torch::Tensor reverse_exchange_t(const torch::Tensor &grad_features) override;
  // === M10 async-overlap split of the forward halo ===
  // `start` records the "owned features ready" event on the model stream; `finish` runs the halo
  // on the Kokkos comm stream so it overlaps the owned-source TP-scatter in between.
  torch::Tensor forward_exchange_start_t(const torch::Tensor &node_features) override;
  torch::Tensor forward_exchange_finish_t(const torch::Tensor &node_features) override;
  // KokkosBase device comm callbacks: gather/scatter feature rows of the on-GPU tensor.
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &, int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &) override;

  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom; 
 protected:
  typedef Kokkos::DualView<int***,DeviceType> tdual_int_3d;
  typedef typename tdual_int_3d::t_dev_const_randomread t_int_3d_randomread;
  typedef typename tdual_int_3d::t_host t_host_int_3d;

  typename AT::t_kkfloat_1d_3_lr_randomread x; 
  typename AT::t_kkacc_1d_3 f; 
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  DAT::ttransform_kkacc_1d k_eatom; 
  DAT::ttransform_kkacc_1d_6 k_vatom;  

  using inputtype = typename super::inputtype;
  using outputtype = typename super::outputtype;

  using IntView1D = Kokkos::View<int*, Kokkos::LayoutRight, DeviceType>;
  using IntView2D = Kokkos::View<int**, Kokkos::LayoutRight, DeviceType>;
  using LongView1D = Kokkos::View<long*, Kokkos::LayoutRight, DeviceType>;
  using LongView2D = Kokkos::View<long**, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView1D = Kokkos::View<outputtype*, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView2D = Kokkos::View<outputtype**, Kokkos::LayoutRight, DeviceType>;
  using View1D = Kokkos::View<KK_ACC_FLOAT*, Kokkos::LayoutRight, DeviceType>;
  using View2D = Kokkos::View<KK_ACC_FLOAT**, Kokkos::LayoutRight, DeviceType>;
  using InputFloatView2D = Kokkos::View<inputtype**, Kokkos::LayoutRight, DeviceType>;



  IntView1D d_type_mapper;
  LongView1D d_ij2type;
  LongView2D d_edges;
  // async-overlap (is_async): edge list re-partitioned owned-source-first into this scratch
  // buffer (real owned-src edges, then real ghost-src edges, then the fake pad edges) so the
  // model can slice the owned-src TP-scatter off the front. Same shape as `d_edges`.
  LongView2D d_edges_async;
  InputFloatView2D d_xfloat;

  View2D d_cutoff_matrix;

  // device-resident feature-halo state, valid only for the duration of one
  // comm->forward_comm/reverse_comm(this, F) call (no reentrancy: the per-layer exchanges run
  // sequentially inside a single model call). Features are F64 (inputtype==double). We index the
  // tensor's raw device pointer directly in the pack/unpack kernels (the mliap/kk pattern), which
  // keeps everything on the device stream and needs no torch<->Kokkos fence (proven by mliap/kk).
  double *exch_ptr_kk = nullptr;   // device ptr to the live [ntotal_padded, F] feature buffer
  torch::Tensor exch_hold;         // keeps that tensor's storage alive across the comm call
  int exch_ncol_kk = 0;            // per-node feature width F for the in-flight exchange

  // === M10 async-overlap state (HIP) ===
  // A dedicated, non-default torch stream the AOT model runs on (set in `compute` for is_async),
  // so the per-layer halo `Comm::forward_comm` on the Kokkos comm stream overlaps the owned-source
  // TP-scatter on this stream. Two events synchronize the two streams across the `start`/`finish`
  // split. Stored as `void*` to keep HIP/c10 types out of this header; created lazily.
  void *m10_model_stream = nullptr;   // c10::hip::HIPStream* (heap)
  void *m10_event_ready = nullptr;    // hipEvent_t: owned features ready (recorded in `start`)
  void *m10_event_done = nullptr;     // hipEvent_t: halo complete (recorded in `finish`)
  void ensure_m10_async_state();      // lazily create the stream + events



  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  Kokkos::View<int**,DeviceType> d_neighbors_short;
  Kokkos::View<int*,DeviceType> d_numneigh_short, d_cumsum_numneigh_short;


  friend void pair_virial_fdotr_compute<PairAllegroKokkos>(PairAllegroKokkos*);
};
}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use chosen neighbor list style with pair_allegro/kk

Self-explanatory.

*/
