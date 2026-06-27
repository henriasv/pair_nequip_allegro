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

PairStyle(nequip,PairNequIPAllegro<true>)
PairStyle(allegro,PairNequIPAllegro<false>)

#else

#ifndef LMP_PAIR_NEQUIP_ALLEGRO_H
#define LMP_PAIR_NEQUIP_ALLEGRO_H

#include "pair.h"

#include <torch/torch.h>
#ifdef NEQUIP_AOT_COMPILE
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#endif

#include <vector>
#include <type_traits>
#include <map>
#include <string>


namespace LAMMPS_NS {

// === Route 2b: native multi-rank per-layer ghost exchange ===
// The compiled `.pt2` calls the globally-registered ops `nequip_lammps::ghost_exchange`
// (forward halo) and `ghost_exchange_reverse` (in-model autograd / force pass). They reach the
// live pair instance through a `thread_local` pointer to this non-template bridge so the
// registration carries no LAMMPS handle as an argument (the obstacle that made the mliap op
// eager-only). The pair implements the two methods via `Comm::forward_comm`/`reverse_comm`.
struct NequIPGhostExchangeBridge {
  // Forward halo: return a copy of `[ntotal, F]` node features with ghost rows filled from their
  // owners. The pair decides host-staged (plain) vs device-resident (Kokkos) internally.
  virtual torch::Tensor forward_exchange_t(const torch::Tensor &node_features) = 0;
  // Reverse (in-model autograd / force pass): accumulate ghost-row grads onto owners, zero ghosts.
  virtual torch::Tensor reverse_exchange_t(const torch::Tensor &grad_features) = 0;

  // === M10 async-overlap split of the forward halo ===
  // `start` returns the features unchanged (ghost rows still zero) but records a "owned features
  // ready" marker on the model stream, BEFORE the owned-source TP-scatter is launched; `finish`
  // runs the halo on a comm stream that overlaps that TP and returns the ghost-filled features.
  // Default (plain / host-staged pair, or any non-overlapping impl): `start` is a clone and
  // `finish` is the blocking `forward_exchange_t` — correct, just not overlapped. The Kokkos pair
  // overrides both with the device-stream + event implementation.
  virtual torch::Tensor forward_exchange_start_t(const torch::Tensor &node_features) {
    return node_features.clone();
  }
  virtual torch::Tensor forward_exchange_finish_t(const torch::Tensor &node_features) {
    return forward_exchange_t(node_features);
  }

 protected:
  // Non-virtual, protected destructor: instances are only ever owned/deleted through `Pair*`
  // (LAMMPS owns the pair), never through this mix-in pointer, so no virtual destructor is needed.
  // This also avoids a multiple-inheritance exception-specification clash with `Pair`'s
  // (noexcept) virtual destructor when `PairNequIPAllegro` derives from both.
  ~NequIPGhostExchangeBridge() = default;
};
// Set to the active pair for the duration of a model call (one per thread).
extern thread_local NequIPGhostExchangeBridge *active_nequip_bridge;

template<bool nequip_mode>
class PairNequIPAllegro : public Pair, public NequIPGhostExchangeBridge {
 public:
  PairNequIPAllegro(class LAMMPS *);
  virtual ~PairNequIPAllegro();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();

  // per-layer feature halo (Route 2b) -- LAMMPS Comm callbacks
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  // NequIPGhostExchangeBridge entry points (called by the registered ops). The plain pair stages
  // through host double buffers + Comm::forward_comm/reverse_comm (CommBrick).
  torch::Tensor forward_exchange_t(const torch::Tensor &node_features) override;
  torch::Tensor reverse_exchange_t(const torch::Tensor &grad_features) override;

  double cutoff;
  torch::Device device = torch::kCPU;
  std::vector<int> type_mapper;

  std::string model_path;

  // `use_aot` bool present even if NEQUIP_AOT_COMPILE not used at compile-time
  // cleaner logic where we always have an outer condition `use_aot` and an inner `ifndef` condition on whether NEQUIP_AOT_COMPILE was used at compile-time
  // errors out if `use_aot` is true, but NEQUIP_AOT_COMPILE wasn't used at compile-time
  bool use_aot;

  // keep separate `torchscript_model` and `aot_model` declarations since which is used is only determined at runtime
  torch::jit::Module torchscript_model;

#ifdef NEQUIP_AOT_COMPILE
  // both torchscript or AOT model can be used
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> aot_model;
  std::vector<std::string> model_input_order;
  std::vector<std::string> model_output_order;
#endif

  // In nequip >= 0.7.0, input and output are always F64
  typedef double inputtype;
  typedef double outputtype;

  torch::ScalarType inputtorchtype = torch::CppTypeToScalarType<inputtype>();
  torch::ScalarType outputtorchtype = torch::CppTypeToScalarType<outputtype>();

  std::vector<std::string> custom_output_names;
  std::map<std::string, torch::Tensor> custom_output;
  void add_custom_output(std::string);
  // dlopen custom Torch op libraries (e.g. OEQ's libtorch_tp_jit.so) the compiled
  // model depends on, from libraries embedded in the model's `.nequip.pt2` zip, a
  // "<model>.oplibs" sidecar, and/or NEQUIP_OP_LIBRARIES.
  void load_extra_op_libraries(const std::string &model_path);
  // Extract custom-op `.so` libraries embedded in the model's `.nequip.pt2` zip (under
  // the "nequip_custom_op_libs/" prefix, STORED) to temporary files; returns their paths.
  std::vector<std::string> extract_embedded_op_libraries(const std::string &model_path);

 protected:
  int debug_mode = 0;

  double** cutoff_matrix;

  // multi-rank native pair_nequip: model takes `num_local_ghost_atoms` and the per-layer
  // ghost exchange runs across ranks (set in `coeff` from the model's declared input order).
  bool is_multirank = false;
  // async-overlap multi-rank pair_nequip: model also takes `num_owned_edges_marker`; the Kokkos
  // pair partitions the edge list owned-source-first and feeds that marker so the model can
  // overlap the owned-src TP-scatter with the in-flight feature halo (set alongside is_multirank).
  bool is_async = false;
  // bridge state for the in-flight feature exchange (valid during a `comm->*_comm(this, F)`)
  double *exch_buf = nullptr;   // [ntotal * exch_ncol], row-major (atom-major)
  int exch_ncol = 0;            // flattened per-node feature width F for the current layer

  c10::Dict<std::string, torch::Tensor> preprocess();
  // multi-rank build: all `ntotal` atoms in atom-index order, real ghost positions, edges from
  // local atoms only (real neighbor indices, no tag2i remap, zero cell shift), plus the
  // `num_local_ghost_atoms` count the per-layer exchange uses.
  c10::Dict<std::string, torch::Tensor> preprocess_multirank();
  c10::Dict<std::string, torch::Tensor> call(c10::Dict<std::string, torch::Tensor>);

  torch::Tensor get_cell();
  void get_tag2i(std::vector<int>&);
};

}

#endif
#endif

