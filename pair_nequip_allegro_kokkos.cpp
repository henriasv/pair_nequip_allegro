/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include <cmath>
#include "kokkos.h"
#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "error.h"
#include "atom_masks.h"
#include "math_const.h"

#include <pair_nequip_allegro_kokkos.h>
#include <torch/torch.h>
#include <torch/script.h>

#ifdef KOKKOS_ENABLE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#ifdef KOKKOS_ENABLE_HIP
// M10 async-overlap: run the AOT model on a dedicated torch stream so the per-layer halo
// (`Comm::forward_comm` on the Kokkos comm stream) overlaps the owned-source TP-scatter.
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#endif

using namespace LAMMPS_NS;
using namespace MathConst;
namespace Kokkos {
  template <>
  struct reduction_identity<s_FEV_FLOAT> {
    KOKKOS_FORCEINLINE_FUNCTION static s_FEV_FLOAT sum() {
      return s_FEV_FLOAT();
    }
  };
}

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

template<bool nequip_mode>
PairAllegroKokkos<nequip_mode>::PairAllegroKokkos(LAMMPS *lmp) : PairNequIPAllegro<nequip_mode>(lmp)
{
  this->respa_enable = 0;


  this->atomKK = (AtomKokkos *) this->atom;
  this->execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  this->datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  this->datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  // Route the per-layer feature reverse_comm through CommKokkos' device path (forward already
  // routes on execution_space==Device). Without this, reverse_comm(Pair*) falls back to host.
  this->reverse_comm_device = 1;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<bool nequip_mode>
PairAllegroKokkos<nequip_mode>::~PairAllegroKokkos()
{
  if (!this->copymode) {
    this->memoryKK->destroy_kokkos(k_eatom,this->eatom);
    this->memoryKK->destroy_kokkos(k_vatom,this->vatom);
    this->eatom = NULL;
    this->vatom = NULL;
  }
#ifdef KOKKOS_ENABLE_HIP
  // release the M10 async-overlap stream + events (created lazily in `ensure_m10_async_state`)
  if (m10_event_ready) hipEventDestroy((hipEvent_t) m10_event_ready);
  if (m10_event_done) hipEventDestroy((hipEvent_t) m10_event_done);
  if (m10_model_stream) delete static_cast<c10::hip::HIPStream *>(m10_model_stream);
  m10_event_ready = m10_event_done = m10_model_stream = nullptr;
#endif
}

/* ---------------------------------------------------------------------- */

template<bool nequip_mode>
void PairAllegroKokkos<nequip_mode>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) this->no_virial_fdotr_compute = 1;

  this->ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (this->eflag_atom) {
    this->memoryKK->destroy_kokkos(k_eatom,this->eatom);
    this->memoryKK->create_kokkos(k_eatom,this->eatom,this->maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (this->vflag_atom) {
    this->memoryKK->destroy_kokkos(k_vatom,this->vatom);
    this->memoryKK->create_kokkos(k_vatom,this->vatom,this->maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  this->atomKK->sync(this->execution_space,this->datamask_read);
  if (eflag || vflag) this->atomKK->modified(this->execution_space,this->datamask_modify);
  else this->atomKK->modified(this->execution_space,F_MASK);

  x = this->atomKK->k_x.template view<DeviceType>();
  f = this->atomKK->k_f.template view<DeviceType>();
  tag = this->atomKK->k_tag.template view<DeviceType>();
  type = this->atomKK->k_type.template view<DeviceType>();
  nlocal = this->atom->nlocal;
  newton_pair = this->force->newton_pair;
  nall = this->atom->nlocal + this->atom->nghost;

  const int inum = this->list->inum;
  const int ignum = inum + this->atom->nghost;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(this->list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  if (inum==0) return; // empty domain

  this->copymode = 1;


  // build short neighbor list

  const int max_neighs = d_neighbors.extent(1);


  if(d_numneigh_short.extent(0) < inum){
    d_numneigh_short = decltype(d_numneigh_short)();
    d_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::numneighs_short") ,inum);
    d_cumsum_numneigh_short = decltype(d_cumsum_numneigh_short)();
    d_cumsum_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::cumsum_numneighs_short") ,inum);
  }
  if(d_neighbors_short.extent(0) < inum || d_neighbors_short.extent(1) < max_neighs){
    d_neighbors_short = decltype(d_neighbors_short)();
    d_neighbors_short = Kokkos::View<int**,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("FLARE::neighbors_short") ,inum,max_neighs);
    //c10::cuda::CUDACachingAllocator::emptyCache();
  }

  // compute short neighbor list
  auto d_numneigh_short = this->d_numneigh_short;
  auto d_neighbors_short = this->d_neighbors_short;
  auto d_cumsum_numneigh_short = this->d_cumsum_numneigh_short;
  double cutoff = this->cutoff;
  auto x = this->x;
  auto d_type = this->type;
  auto d_ilist = this->d_ilist;
  auto d_numneigh = this->d_numneigh;
  auto d_neighbors = this->d_neighbors;
  auto f = this->f;
  auto d_eatom = this->d_eatom;
  auto d_type_mapper = this->d_type_mapper;
  auto d_cutoff_matrix = this->d_cutoff_matrix;

  Kokkos::parallel_for("Allegro: Short neighlist", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii){
      const int i = d_ilist[ii];
      const KK_FLOAT xtmp = x(i,0); 
      const KK_FLOAT ytmp = x(i,1); 
      const KK_FLOAT ztmp = x(i,2); 

      const int si = d_type[i] - 1;

      const int jnum = d_numneigh[i];
      int inside = 0;
      for (int jj = 0; jj < jnum; jj++) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;

        const int sj = d_type[j] - 1;
        const double ijcut = d_cutoff_matrix(si, sj); //TODO

        //printf("i=%3d j=%3d ti=%d tj=%d cut=%.2f\n", i, j, d_type[i], d_type[j], ijcut);

        const KK_FLOAT delx = xtmp - x(j,0); 
        const KK_FLOAT dely = ytmp - x(j,1); 
        const KK_FLOAT delz = ztmp - x(j,2); 
        const KK_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < ijcut*ijcut) {
          d_neighbors_short(ii,inside) = j;
          inside++;
        }
      }
      d_numneigh_short(ii) = inside;
  });
  Kokkos::deep_copy(d_cumsum_numneigh_short, d_numneigh_short);

  Kokkos::parallel_scan("Allegro: cumsum shortneighs", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii, int& update, const bool is_final){
      const int curr_val = d_cumsum_numneigh_short(ii);
      update += curr_val;
      if(is_final) d_cumsum_numneigh_short(ii) = update;
  });
  int nedges = 0;
  Kokkos::View<int*, Kokkos::HostSpace> nedges_view("Allegro: nedges",1);
  Kokkos::deep_copy(nedges_view, Kokkos::subview(d_cumsum_numneigh_short, Kokkos::make_pair(inum-1, inum)));
  nedges = nedges_view(0);

  //auto nn = Kokkos::create_mirror_view(d_numneigh_short);
  //Kokkos::deep_copy(nn, d_numneigh_short);
  //auto cs = Kokkos::create_mirror_view(d_cumsum_numneigh_short);
  //Kokkos::deep_copy(cs, d_cumsum_numneigh_short);
  //printf("INUM=%d, GNUM=%d, IGNUM=%d\n", inum, list->gnum, ignum);
  //printf("NEDGES: %d\nnumneigh_short cumsum\n",nedges);
  //for(int i = 0; i < inum; i++){
  //  printf("%d %d\n", nn(i), cs(i));
  //}

  double padding_factor = 1.05;

  if(d_edges.extent(1) < nedges || nedges*padding_factor*padding_factor < d_edges.extent(1)){
    d_edges = decltype(d_edges)();
    d_edges = decltype(d_edges)("Allegro: edges", 2, padding_factor*nedges);
  }
  if(d_ij2type.extent(0) < ignum+2 || (ignum+2)*padding_factor*padding_factor < d_ij2type.extent(0)){
    d_ij2type = decltype(d_ij2type)();
    d_ij2type = decltype(d_ij2type)("Allegro: ij2type", padding_factor*ignum+2);
    d_xfloat = decltype(d_xfloat)();
    d_xfloat = decltype(d_xfloat)("Allegro: xfloat", padding_factor*ignum+2, 3);
  }

  auto d_edges = this->d_edges;
  auto d_ij2type = this->d_ij2type;
  auto d_xfloat = this->d_xfloat;

  Kokkos::parallel_for("Allegro: store type mask and x", Kokkos::RangePolicy<DeviceType>(0, ignum), KOKKOS_LAMBDA(const int i){
      d_ij2type(i) = d_type_mapper(d_type(i)-1);
      d_xfloat(i,0) = x(i,0);
      d_xfloat(i,1) = x(i,1);
      d_xfloat(i,2) = x(i,2);
  });

  int max_atoms = d_ij2type.extent(0);
  Kokkos::parallel_for("Allegro: store fake atoms", Kokkos::RangePolicy<DeviceType>(ignum, max_atoms), KOKKOS_LAMBDA(const int i){
      d_ij2type(i) = d_type_mapper(0);
      d_xfloat(i,0) = i==max_atoms-1 ? 100.0 : 0.0;
      d_xfloat(i,1) = 0.0;
      d_xfloat(i,2) = 0.0;
  });

  Kokkos::parallel_for("Allegro: create edges", Kokkos::TeamPolicy<DeviceType>(inum, Kokkos::AUTO()), KOKKOS_LAMBDA(const MemberType team_member){
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      const int startedge = ii==0 ? 0 : d_cumsum_numneigh_short(ii-1);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, d_numneigh_short(ii)), [&] (const int jj){
          d_edges(0, startedge + jj) = i;
          d_edges(1, startedge + jj) = d_neighbors_short(ii,jj);
      });
  });

  int max_edges = d_edges.extent(1);
  Kokkos::parallel_for("Allegro: store fake edges", Kokkos::RangePolicy<DeviceType>(nedges, max_edges), KOKKOS_LAMBDA(const int i){
      d_edges(0, i) = max_atoms-2;
      d_edges(1, i) = max_atoms-1;
  });

  // === async-overlap edge partition (is_async) ===
  // Reorder the REAL edges `[0, nedges)` owned-source-first (owned-src = `edge_src = d_edges(1,e)
  // < nlocal`) into the `d_edges_async` scratch, then copy the fake pad edges `[nedges, max_edges)`
  // unchanged (their src `max_atoms-1 >= nlocal` makes them legitimately ghost-side). The model
  // then slices the per-edge tensors at `n_owned_edges`: the owned-src TP-scatter (needing no
  // freshly-exchanged ghost features) runs while the halo is in flight. The partition uses two
  // atomic cursors (order within each partition is irrelevant -- scatter-add is commutative).
  long *edges_data = d_edges.data();
  long edges_stride = d_edges.extent(1);
  int n_owned_edges = 0;
  if (this->is_async) {
    if (this->d_edges_async.extent(1) != d_edges.extent(1)) {
      this->d_edges_async = decltype(this->d_edges_async)();
      this->d_edges_async = decltype(this->d_edges_async)("Allegro: edges_async", 2, d_edges.extent(1));
    }
    auto d_edges_async = this->d_edges_async;
    const int nloc = nlocal;
    Kokkos::parallel_reduce("Allegro: count owned edges",
        Kokkos::RangePolicy<DeviceType>(0, nedges),
        KOKKOS_LAMBDA(const int e, int &sum){ if (d_edges(1, e) < nloc) sum++; },
        n_owned_edges);
    Kokkos::View<int*, DeviceType> cursors("Allegro: part cursors", 2);
    auto h_cursors = Kokkos::create_mirror_view(cursors);
    h_cursors(0) = 0;             // owned-src cursor -> fills [0, n_owned_edges)
    h_cursors(1) = n_owned_edges; // ghost-src cursor -> fills [n_owned_edges, nedges)
    Kokkos::deep_copy(cursors, h_cursors);
    Kokkos::parallel_for("Allegro: partition edges owned-first",
        Kokkos::RangePolicy<DeviceType>(0, nedges), KOKKOS_LAMBDA(const int e){
          const int dst = d_edges(0, e);
          const int src = d_edges(1, e);
          const int slot = (src < nloc)
              ? Kokkos::atomic_fetch_add(&cursors(0), 1)
              : Kokkos::atomic_fetch_add(&cursors(1), 1);
          d_edges_async(0, slot) = dst;
          d_edges_async(1, slot) = src;
        });
    const int max_edges_l = d_edges.extent(1);
    Kokkos::parallel_for("Allegro: copy fake edges async",
        Kokkos::RangePolicy<DeviceType>(nedges, max_edges_l), KOKKOS_LAMBDA(const int e){
          d_edges_async(0, e) = d_edges(0, e);
          d_edges_async(1, e) = d_edges(1, e);
        });
    edges_data = this->d_edges_async.data();
    edges_stride = this->d_edges_async.extent(1);
  }

  torch::Tensor ij2type_tensor = torch::from_blob(d_ij2type.data(), {max_atoms}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
  torch::Tensor edges_tensor = torch::from_blob(edges_data, {2,max_edges}, {edges_stride,1}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
  torch::Tensor pos_tensor = torch::from_blob(d_xfloat.data(), {max_atoms,3}, {3,1}, torch::TensorOptions().device(this->device).dtype(this->inputtorchtype));

  if (this->debug_mode) {
    printf("Allegro edges: i j rij\n");
    for (long i = 0; i < nedges; i++) {
      printf(
        "%ld %ld %.10g\n",
        edges_tensor.index({0, i}).item<long>(),
        edges_tensor.index({1, i}).item<long>(),
        (pos_tensor[edges_tensor.index({0, i}).item<long>()] - pos_tensor[edges_tensor.index({1, i}).item<long>()]).square().sum().sqrt().item<inputtype>()
      );
    }
    printf("end Allegro edges\n");
  }

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor);
  input.insert("edge_index", edges_tensor);
  input.insert("atom_types", ij2type_tensor);

  if (this->is_multirank) {
    // The multirank model also consumes `cell`, a zero `edge_cell_shift` (ghost positions are
    // real -> no periodic reconstruction), and `num_local_ghost_atoms` (drives the owned-energy
    // mask). The per-layer `ghost_exchange` op fired from inside the model reaches this pair via
    // the thread-local bridge -> CommKokkos device exchange.
    torch::Tensor cell_tensor = this->get_cell().unsqueeze(0).to(this->device);
    torch::Tensor shift_tensor = torch::zeros(
        {max_edges, 3}, torch::TensorOptions().dtype(this->inputtorchtype).device(this->device));
    torch::Tensor nlg_tensor =
        torch::tensor({static_cast<int64_t>(this->atom->nlocal),
                       static_cast<int64_t>(this->atom->nghost)},
                      torch::TensorOptions().dtype(torch::kInt64))
            .to(this->device);
    // truncate-to-nlocal marker `(nlocal,)`: carries the owned count as the model's backed
    // `nlocal` dynamic dim so per-node ops are computed on owned atoms only (contents unused).
    torch::Tensor marker_tensor =
        torch::zeros({static_cast<int64_t>(this->atom->nlocal)},
                     torch::TensorOptions().dtype(torch::kInt64))
            .to(this->device);
    input.insert("cell", cell_tensor);
    input.insert("edge_cell_shift", shift_tensor);
    input.insert("num_local_ghost_atoms", nlg_tensor);
    input.insert("num_local_nodes_marker", marker_tensor);
    if (this->is_async) {
      // owned-src-edge marker `(n_owned_edges,)`: carries the backed split point at which the
      // model slices the per-edge tensors (the edge list above was partitioned owned-src-first).
      // Contents unused -- only the size-0 dim. `call()` places it positionally via the model's
      // declared `nequip_aoti_inputs` order.
      torch::Tensor owned_edges_marker =
          torch::zeros({static_cast<int64_t>(n_owned_edges)},
                       torch::TensorOptions().dtype(torch::kInt64))
              .to(this->device);
      input.insert("num_owned_edges_marker", owned_edges_marker);
    }
    LAMMPS_NS::active_nequip_bridge = this;
  }

#ifdef KOKKOS_ENABLE_HIP
  // M10 async overlap: run the AOT model on a dedicated torch stream so the per-layer halo
  // (`Comm::forward_comm` on the Kokkos comm stream) overlaps the owned-source TP-scatter. The
  // model inputs were built on the previous (default) torch stream + the Kokkos comm stream;
  // drain both before the model reads them on the dedicated stream (one-off per step, cheap).
  c10::optional<c10::hip::HIPStreamGuard> m10_guard;
  if (this->is_async) {
    this->ensure_m10_async_state();
    hipStreamSynchronize(c10::hip::getCurrentHIPStream().stream());
    DeviceType().fence();
    m10_guard.emplace(*static_cast<c10::hip::HIPStream *>(m10_model_stream));
  }
#endif
  auto output = this->call(input);
#ifdef KOKKOS_ENABLE_HIP
  if (this->is_async) {
    // make the model outputs (produced on the dedicated stream) visible to the Kokkos force-store
    // reduce below (which runs on the comm stream) before the guard restores the previous stream.
    hipStreamSynchronize(static_cast<c10::hip::HIPStream *>(m10_model_stream)->stream());
  }
#endif
  if (this->is_multirank) LAMMPS_NS::active_nequip_bridge = nullptr;
  torch::Tensor forces_tensor = output.at("forces");
  torch::Tensor atomic_energy_tensor = output.at("atomic_energy");

  UnmanagedFloatView1D d_atomic_energy(atomic_energy_tensor.data_ptr<outputtype>(), inum);
  UnmanagedFloatView2D d_forces(forces_tensor.data_ptr<outputtype>(), ignum, 3);

  //std::cout << "NequIP model output:\n";
  //std::cout << "forces:\n" << forces_tensor.cpu() << "\n";
  //std::cout << "atomic_energy:\n" << atomic_energy_tensor.cpu() << "\n";

  this->eng_vdwl = 0.0;
  auto eflag_atom = this->eflag_atom;
  Kokkos::parallel_reduce("Allegro: store forces",
      Kokkos::RangePolicy<DeviceType>(0, ignum),
      KOKKOS_LAMBDA(const int i, double &eng_vdwl){
        f(i,0) += d_forces(i,0);
        f(i,1) += d_forces(i,1);
        f(i,2) += d_forces(i,2);
        if(eflag_atom && i < inum){
          d_eatom(i) = d_atomic_energy(i);
        }
        if(i < inum){
          eng_vdwl += d_atomic_energy(i);
        }
      },
      this->eng_vdwl
      );

  if (eflag_atom) {
    // if (need_dup)
    //   Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if(vflag){
    torch::Tensor v_tensor = output.at("virial").cpu();
    auto v = v_tensor.accessor<outputtype, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // First [0] index on v is batch
    this->virial[0] = v[0][0][0];
    this->virial[1] = v[0][1][1];
    this->virial[2] = v[0][2][2];
    this->virial[3] = v[0][0][1];
    this->virial[4] = v[0][0][2];
    this->virial[5] = v[0][1][2];
  }
  if(this->vflag_atom) {
    this->error->all(FLERR,"Pair style Allegro does not support per-atom virial");
  }

  if (this->vflag_fdotr) pair_virial_fdotr_compute(this);

  for(const std::string &output_name : this->custom_output_names){
    this->custom_output.insert_or_assign(output_name, output.at(output_name).detach());
  }


  this->copymode = 0;

}






/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<bool nequip_mode>
void PairAllegroKokkos<nequip_mode>::coeff(int narg, char **arg)
{
  super::coeff(narg,arg);
  int ntypes = this->atom->ntypes;

  d_type_mapper = IntView1D("Allegro: type_mapper", this->type_mapper.size());
  auto h_type_mapper = Kokkos::create_mirror_view(d_type_mapper);
  for(int i = 0; i < this->type_mapper.size(); i++){
    h_type_mapper(i) = this->type_mapper[i];
  }
  Kokkos::deep_copy(d_type_mapper, h_type_mapper);

  d_cutoff_matrix = View2D("Allegro: cutoff_matrix", ntypes, ntypes);
  auto h_cutoff_matrix = Kokkos::create_mirror_view(d_cutoff_matrix);
  for(int i = 0; i < ntypes; i++){
    for(int j = 0; j < ntypes; j++){
      h_cutoff_matrix(i,j) = this->cutoff_matrix[i][j];
      if (this->comm->me==0) printf("ti=%d tj=%d cut=%.2f\n", i, j, h_cutoff_matrix(i,j));
    }
  }
  Kokkos::deep_copy(d_cutoff_matrix, h_cutoff_matrix); // TODO: Check
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<bool nequip_mode>
void PairAllegroKokkos<nequip_mode>::init_style()
{
  super::init_style();

  auto request = this->neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
      !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);

  neighflag = this->lmp->kokkos->neighflag;
  if (neighflag == FULL) {
    this->error->all(FLERR,"pair style allegro/kk requires the 'neigh half' flag due to 'newton on'");
  }
}



/* ----------------------------------------------------------------------
   Route 2b device-resident per-layer feature halo (M6).
   Mirrors the mliap/kk pattern: gather/scatter feature rows of the on-GPU tensor by the device
   sendlist, exchanged through CommKokkos::*_comm_device (GPU-aware MPI on device buffers). No
   host staging, no explicit torch<->Kokkos fence (the kernels run on the device stream, same as
   mliap/kk). Features are F64; we index the raw device pointer directly.
------------------------------------------------------------------------- */

template<bool nequip_mode>
torch::Tensor PairAllegroKokkos<nequip_mode>::forward_exchange_t(const torch::Tensor &node_features)
{
  // Stay on the GPU. Exchange in F64 (the comm buffer is double); convert back to the model's
  // feature dtype after. `.clone()` guarantees a private, contiguous buffer (the op is functional)
  // even when the input is already contiguous F64.
  auto f = node_features.to(torch::kDouble).contiguous().clone();
  const int64_t ntot = f.size(0);
  const int F = ntot > 0 ? static_cast<int>(f.numel() / ntot) : 0;
  this->comm_forward = F;
  exch_ptr_kk = f.data_ptr<double>();
  exch_hold = f;
  exch_ncol_kk = F;
#ifdef KOKKOS_ENABLE_HIP
  // Cross-stream safety: in the normal (non-async) path the model and the Kokkos comm share one
  // stream, so `f` (cloned on the model stream) is already ordered before `forward_comm`'s pack.
  // But if a single-op (`ghost_exchange`) model is ever run under the async model-stream guard,
  // the two streams differ and we must order them explicitly (blocking; no overlap). Zero overhead
  // when the streams coincide.
  hipStream_t fx_cur = c10::hip::getCurrentHIPStream().stream();
  hipStream_t fx_k = DeviceType().hip_stream();
  hipEvent_t fx_ev = nullptr;
  if (fx_cur != fx_k) {
    hipEventCreateWithFlags(&fx_ev, hipEventDisableTiming);
    hipEventRecord(fx_ev, fx_cur);            // `f` produced on the model stream
    hipStreamWaitEvent(fx_k, fx_ev, 0);       // comm stream waits for it
  }
#endif
  this->comm->forward_comm(this, F);   // -> CommKokkos::forward_comm_device -> pack/unpack below
#ifdef KOKKOS_ENABLE_HIP
  if (fx_cur != fx_k) {
    hipEventRecord(fx_ev, fx_k);              // ghost rows written on the comm stream
    hipStreamWaitEvent(fx_cur, fx_ev, 0);     // model stream waits before it reads them
    hipEventDestroy(fx_ev);
  }
#endif
  exch_ptr_kk = nullptr;
  exch_ncol_kk = 0;
  exch_hold = torch::Tensor();
  return f.to(node_features.scalar_type()).view_as(node_features);
}

/* ----------------------------------------------------------------------
   M10 async-overlap split of the forward halo. `start` records a "owned features ready" event on
   the model stream (before the owned-source TP-scatter is launched); `finish` runs the halo on the
   Kokkos comm stream so it overlaps that TP. See the design in M10_async_overlap_design.md.
------------------------------------------------------------------------- */
#ifdef KOKKOS_ENABLE_HIP
template<bool nequip_mode>
void PairAllegroKokkos<nequip_mode>::ensure_m10_async_state()
{
  if (m10_model_stream != nullptr) return;
  // Dedicated, non-default torch stream from the pool: distinct from the Kokkos comm stream
  // (`DeviceType().hip_stream()`), so the owned-source TP-scatter (run on this stream) and the
  // deferred halo (`Comm::forward_comm` on the comm stream) overlap instead of serializing.
  m10_model_stream = new c10::hip::HIPStream(c10::hip::getStreamFromPool());
  hipEvent_t ev_ready, ev_done;
  hipEventCreateWithFlags(&ev_ready, hipEventDisableTiming);
  hipEventCreateWithFlags(&ev_done, hipEventDisableTiming);
  m10_event_ready = (void *) ev_ready;
  m10_event_done = (void *) ev_done;
}

template<bool nequip_mode>
torch::Tensor PairAllegroKokkos<nequip_mode>::forward_exchange_start_t(const torch::Tensor &node_features)
{
  // Phase 1: clone to a private buffer that the owned-source TP-scatter (next, on the model stream)
  // and the deferred halo (`finish`, on the comm stream) both read, and record the "owned features
  // ready" event on the model stream -- BEFORE that TP is launched. `finish` makes the comm stream
  // wait on this event, so the halo starts without waiting for the owned-edge TP (the overlap).
  ensure_m10_async_state();
  auto C = node_features.contiguous().clone();
  hipEventRecord((hipEvent_t) m10_event_ready, c10::hip::getCurrentHIPStream().stream());
  return C;
}

template<bool nequip_mode>
torch::Tensor PairAllegroKokkos<nequip_mode>::forward_exchange_finish_t(const torch::Tensor &node_features)
{
  // Phase 2: complete the halo on the Kokkos comm stream (overlapping the owned-source TP still in
  // flight on the model stream). `node_features` is the `start` clone (owned rows valid, ghost rows
  // zero). Copy it to a fresh output ON THE COMM STREAM (gated on the ready event, so the copy does
  // not wait for the owned-edge TP), then `forward_comm` fills the ghost rows -- all on the comm
  // stream. Functional (returns the new output), so the in-model force/backward pass matches the
  // single-op path.
  ensure_m10_async_state();
  auto Cf = node_features.to(torch::kDouble).contiguous();
  const int64_t ntot = Cf.size(0);
  const int F = ntot > 0 ? static_cast<int>(Cf.numel() / ntot) : 0;
  auto out = torch::empty_like(Cf);

  hipStream_t kstream = DeviceType().hip_stream();
  hipStream_t mstream = c10::hip::getCurrentHIPStream().stream();
  // comm stream waits for the model stream's production of the `start` clone (the pre-owned-TP
  // event), NOT for the owned-edge TP itself
  hipStreamWaitEvent(kstream, (hipEvent_t) m10_event_ready, 0);
  // copy owned + zero-ghost rows on the comm stream; the halo below overwrites the ghost rows
  hipMemcpyAsync(out.data_ptr<double>(), Cf.data_ptr<double>(),
                 static_cast<size_t>(Cf.numel()) * sizeof(double),
                 hipMemcpyDeviceToDevice, kstream);
  this->comm_forward = F;
  exch_ptr_kk = out.data_ptr<double>();
  exch_hold = out;
  exch_ncol_kk = F;
  this->comm->forward_comm(this, F);   // pack/unpack on the Kokkos comm stream; fills ghost rows
  exch_ptr_kk = nullptr;
  exch_ncol_kk = 0;
  exch_hold = torch::Tensor();
  // model stream (next op: ghost-source TP-scatter) waits for the halo to land
  hipEventRecord((hipEvent_t) m10_event_done, kstream);
  hipStreamWaitEvent(mstream, (hipEvent_t) m10_event_done, 0);
  return out.to(node_features.scalar_type()).view_as(node_features);
}
#else
template<bool nequip_mode>
void PairAllegroKokkos<nequip_mode>::ensure_m10_async_state() {}
template<bool nequip_mode>
torch::Tensor PairAllegroKokkos<nequip_mode>::forward_exchange_start_t(const torch::Tensor &node_features)
{ return node_features.clone(); }
template<bool nequip_mode>
torch::Tensor PairAllegroKokkos<nequip_mode>::forward_exchange_finish_t(const torch::Tensor &node_features)
{ return forward_exchange_t(node_features); }
#endif

template<bool nequip_mode>
torch::Tensor PairAllegroKokkos<nequip_mode>::reverse_exchange_t(const torch::Tensor &grad_features)
{
  auto g = grad_features.to(torch::kDouble).contiguous().clone();
  const int64_t ntot = g.size(0);
  const int F = ntot > 0 ? static_cast<int>(g.numel() / ntot) : 0;
  this->comm_reverse = F;
  exch_ptr_kk = g.data_ptr<double>();
  exch_hold = g;
  exch_ncol_kk = F;
#ifdef KOKKOS_ENABLE_HIP
  // Cross-stream safety for the in-model backward (force pass): under the async model-stream guard
  // `g` is cloned on the model stream but `reverse_comm` + the zero-ghost kernel run on the Kokkos
  // comm stream. Order them (blocking; the reverse halo is not overlapped until Stage 3). Zero
  // overhead when the two streams coincide (the non-async path).
  hipStream_t rx_cur = c10::hip::getCurrentHIPStream().stream();
  hipStream_t rx_k = DeviceType().hip_stream();
  hipEvent_t rx_ev = nullptr;
  if (rx_cur != rx_k) {
    hipEventCreateWithFlags(&rx_ev, hipEventDisableTiming);
    hipEventRecord(rx_ev, rx_cur);            // `g` produced on the model stream
    hipStreamWaitEvent(rx_k, rx_ev, 0);       // comm stream waits before reverse_comm reads it
  }
#endif
  this->comm->reverse_comm(this, F);   // accumulates ghost-row grads onto owners (atomic +=)
  // dy[ghost] = 0: the forward halo output's ghost rows do not depend on ghost inputs.
  const int nloc = this->atom->nlocal;
  const int nrow = static_cast<int>(ntot);
  double *p = exch_ptr_kk;
  Kokkos::parallel_for("nequip:zero_ghost_grad",
      Kokkos::RangePolicy<DeviceType>(nloc, nrow), KOKKOS_LAMBDA(const int i) {
        for (int c = 0; c < F; c++) p[(long) i * F + c] = 0.0;
      });
#ifdef KOKKOS_ENABLE_HIP
  if (rx_cur != rx_k) {
    hipEventRecord(rx_ev, rx_k);              // grads written on the comm stream
    hipStreamWaitEvent(rx_cur, rx_ev, 0);     // model stream waits before it reads them
    hipEventDestroy(rx_ev);
  }
#endif
  exch_ptr_kk = nullptr;
  exch_ncol_kk = 0;
  exch_hold = torch::Tensor();
  return g.to(grad_features.scalar_type()).view_as(grad_features);
}

template<bool nequip_mode>
int PairAllegroKokkos<nequip_mode>::pack_forward_comm_kokkos(
    int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &k_buf, int /*pbc_flag*/, int * /*pbc*/)
{
  auto idx = k_sendlist.template view<DeviceType>();
  auto buf = k_buf.template view<DeviceType>();
  const int F = exch_ncol_kk;
  double *p = exch_ptr_kk;
  Kokkos::parallel_for("nequip:pack_fwd", Kokkos::RangePolicy<DeviceType>(0, n * F),
      KOKKOS_LAMBDA(const int s) {
        const int i = s / F, c = s % F;
        buf(s) = p[(long) idx(i) * F + c];
      });
  k_buf.template modify<DeviceType>();
  return n * F;
}

template<bool nequip_mode>
void PairAllegroKokkos<nequip_mode>::unpack_forward_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &k_buf)
{
  auto buf = k_buf.template view<DeviceType>();
  const int F = exch_ncol_kk;
  double *p = exch_ptr_kk;
  Kokkos::parallel_for("nequip:unpack_fwd", Kokkos::RangePolicy<DeviceType>(0, n * F),
      KOKKOS_LAMBDA(const int s) {
        const int i = s / F, c = s % F;
        p[(long) (first + i) * F + c] = buf(s);
      });
}

template<bool nequip_mode>
int PairAllegroKokkos<nequip_mode>::pack_reverse_comm_kokkos(
    int n, int first, DAT::tdual_double_1d &k_buf)
{
  auto buf = k_buf.template view<DeviceType>();
  const int F = exch_ncol_kk;
  double *p = exch_ptr_kk;
  Kokkos::parallel_for("nequip:pack_rev", Kokkos::RangePolicy<DeviceType>(0, n * F),
      KOKKOS_LAMBDA(const int s) {
        const int i = s / F, c = s % F;
        buf(s) = p[(long) (first + i) * F + c];
      });
  k_buf.template modify<DeviceType>();
  return n * F;
}

template<bool nequip_mode>
void PairAllegroKokkos<nequip_mode>::unpack_reverse_comm_kokkos(
    int n, DAT::tdual_int_1d k_recvlist, DAT::tdual_double_1d &k_buf)
{
  auto idx = k_recvlist.template view<DeviceType>();
  auto buf = k_buf.template view<DeviceType>();
  const int F = exch_ncol_kk;
  double *p = exch_ptr_kk;
  Kokkos::parallel_for("nequip:unpack_rev", Kokkos::RangePolicy<DeviceType>(0, n * F),
      KOKKOS_LAMBDA(const int s) {
        const int i = s / F, c = s % F;
        Kokkos::atomic_add(&p[(long) idx(i) * F + c], buf(s));
      });
}

namespace LAMMPS_NS {
template class PairAllegroKokkos<false>;
template class PairAllegroKokkos<true>;
}

