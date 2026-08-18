# Fork delta vs `mir-group/pair_nequip_allegro`

**Fork:** `henriasv/pair_nequip_allegro`, branch `pr/multigpu`
**Upstream base:** `mir-group/pair_nequip_allegro` `main` @ `2e19360` (fully up to date at the time of writing — the branch is 0 commits behind upstream)
**Delta:** 759 insertions / 22 deletions, confined to the four pair-style sources
(`pair_nequip_allegro.{cpp,h}`, `pair_nequip_allegro_kokkos.{cpp,h}`).

This document is the complete, annotated description of what this fork changes, written
for upstream review. Companion: the `nequip` fork's `UPSTREAM_DELTA.md`
(`henriasv/nequip`, branch `pr/multigpu`) — the two deltas form one feature and are
intended to be reviewed together.

---

## What this fork adds

### 1. Native multi-GPU (multi-rank) `pair_style nequip`

Upstream `pair_nequip` is restricted to a single MPI rank because a NequIP
(message-passing) model needs features of atoms owned by neighboring ranks at every
interaction layer. This fork removes that restriction with a **per-layer ghost-feature
exchange**: the compiled model itself calls back into LAMMPS communication between
layers.

**How it works, end to end:**

- The `nequip` fork compiles models with a new `pair_nequip_multirank` AOTInductor
  target. Such models (a) declare two extra inputs, `num_local_ghost_atoms` (`[nlocal,
  nghost]`) and `num_local_nodes_marker` (a `(nlocal,)` tensor whose *size* carries the
  owned-atom count as a backed dynamic dimension), and (b) call a registered Torch custom
  op `nequip_lammps::ghost_exchange` (+ its autograd transpose
  `ghost_exchange_reverse`) before each interaction layer's tensor-product scatter.
- On the C++ side (this fork), `TORCH_LIBRARY(nequip_lammps, ...)` registers the *real*
  implementations of those ops in the LAMMPS process. The ops carry **no LAMMPS handle as
  an argument** — they reach the live pair instance through a `thread_local` pointer to a
  small abstract base (`NequIPGhostExchangeBridge`) that the pair arms for the duration
  of each model call. (Passing a handle through the op signature is exactly what made the
  ML-IAP ghost exchange eager-only; this design is what lets the exchange survive
  AOTInductor compilation.) With no bridge armed — single rank, export tracing — the ops
  are the identity.
- The bridge has two implementations:
  - **Plain (`pair_style nequip`)**: stages the `[ntotal, F]` feature tensor through host
    double buffers and LAMMPS `Comm::forward_comm(this, F)` / `reverse_comm(this, F)`,
    via the standard `pack/unpack_{forward,reverse}_comm` pair callbacks.
  - **Kokkos (`pair_style nequip/kk`, new)**: keeps features **on the GPU**. The pair now
    also derives from `KokkosBase` and implements
    `pack/unpack_{forward,reverse}_comm_kokkos`, which gather/scatter feature rows of the
    live device tensor by the device sendlist; the exchange goes through
    `CommKokkos::*_comm_device` (GPU-aware MPI on device buffers), no host staging. The
    reverse path accumulates with `Kokkos::atomic_add` and then zeroes ghost rows (the
    forward output's ghost rows do not depend on ghost inputs).
- **Multi-rank input build** (`preprocess_multirank()`): all `ntotal` atoms in LAMMPS
  atom-index order (so tensor rows line up with what `Comm` expects), *real* ghost
  positions (hence `edge_cell_shift ≡ 0` — no periodic reconstruction), edges from local
  atoms only with real neighbor indices (no `tag2i` remap). The Kokkos compute path
  builds the same extra inputs on device.
- **Force/energy handling**: the model masks atomic energies to owned atoms; forces come
  back for all `ntotal` atoms in atom order and ghost forces (`dE_owned/dx_ghost`) go
  home through LAMMPS' standard Newton reverse communication, exactly like the `allegro`
  path. `newton pair on` is therefore required (enforced with a clear error).
- **Model detection is automatic**: `coeff()` inspects the compiled model's declared
  input order for `num_local_ghost_atoms`. Multi-rank models enable the path;
  single-rank models keep the exact upstream behavior. A `pair_nequip_multirank`
  metadata stamp written by `nequip-compile` is cross-checked against the declared
  inputs, with a warning if they disagree (inputs win).

### 2. Custom Torch-op library loading (OpenEquivariance etc.)

Compiled models that use accelerated tensor-product kernels (e.g. OpenEquivariance)
reference custom Torch ops. In Python those ops are registered by `import
openequivariance`; a standalone C++ LAMMPS process has no interpreter, so loading such a
`.pt2` dies with "Could not find schema for libtorch_tp_jit::…". This fork makes those
models load: at `pair_coeff` time, before the model is loaded, the pair `dlopen`s
(`RTLD_NOW | RTLD_GLOBAL`) the op libraries collected from, in order:

1. **`.so` binaries embedded in the model's `.nequip.pt2`** by `nequip-compile` (see the
   nequip fork: stored uncompressed under `nequip_custom_op_libs/`; a ~100-line
   dependency-free ZIP central-directory reader extracts them to temp files) — the
   self-contained default;
2. a `<model>.oplibs` sidecar file (one path per line, `#` comments, relative paths
   resolved against the model's directory);
3. the `NEQUIP_OP_LIBRARIES` environment variable (colon-separated), as an explicit
   override.

This is independent of multi-rank: it also fixes single-rank `pair_nequip`/`pair_allegro`
with OEQ-accelerated models.

### 3. Robustness guards (error messages instead of crashes/silent wrong)

- **Single-rank model on N ranks**: upstream errored in the constructor for `nequip` mode
  with >1 rank, before it could know whether the model is multi-rank capable. The check
  is moved to `init_style()` (after `coeff()` has loaded the model): multi-rank models
  pass; single-rank models get an actionable error that includes the exact
  `nequip-compile` command to produce a multi-rank build.
- **Second in-process model load** (e.g. after `clear`): a second AOTInductor model load
  re-enters process-global Torch/custom-op state and segfaults. Now trips a clean
  `error->all` explaining what happened and what to do instead (change the system in
  place / `rerun` / one process per setup). The flag is deliberately process-scoped.
- **Metadata/inputs cross-check** on load (see feature 1).

---

## Backward-compatibility audit

Claim: **every pre-existing feature is preserved; the multi-rank machinery is inert
unless the loaded model itself declares the multi-rank input.** `is_multirank` can only
become true for a model compiled with the new `pair_nequip_multirank` target; for every
model that existed before this fork it is false and the code takes the upstream paths.

All touches of *shared* (pre-existing) code paths, exhaustively:

| # | Change | Effect on existing behavior |
|---|--------|------------------------------|
| 1 | Comm-buffer sizing: constructor sets a bounded default (`comm_forward/reverse = 2048`); `coeff()` refines it once the model is known — **exact** width from the model's `pair_nequip_feature_width` metadata stamp (written by `nequip-compile`) for multirank models, **zero** for everything else | None on results. Non-multirank models (all pre-existing ones) get `comm_forward = comm_reverse = 0`, i.e. exactly upstream's memory behavior. Multirank models get exactly-sized buffers; only a multirank model compiled before the stamp existed falls back to the bounded default, still guarded by a hard-error (never overflow) in the exchange. |
| 2 | Single-rank restriction moved from constructor to `init_style()` | Same protection, later (but still before any run), better message. Required because multi-rank capability is only known after `coeff()`. |
| 3 | `AOTIModelPackageLoader` now constructed with explicit `device_index` | On 1 GPU: identical (index 0). On multi-GPU it is a correctness fix: the default places every rank's weights on `cuda:0` while inputs live on `cuda:r` → cross-device fault. |
| 4 | `c10::OptionalDeviceGuard` pinning the rank's device around each model call | Same rationale as 3; no-op on single GPU. |
| 5 | Force/energy write-back restructured into `if (is_multirank) {...} else {...}` | The `else` branch is the upstream loop, verbatim. |
| 6 | `load_extra_op_libraries()` called in `coeff()` for every model | No-op unless the model embeds libraries / a sidecar / env var exists (returns empty and loads nothing otherwise). |
| 7 | Second in-process AOT load now a clean error | Previously this segfaulted; no working use case is removed. |
| 8 | Kokkos constructor sets `reverse_comm_device = 1` | Routes *pair-initiated* `reverse_comm(Pair*)` through the device path. The pre-existing `allegro/kk` path never issues such a call, so this is inert outside the feature. |
| 9 | Four commented-out debug `std::cout` lines removed from the Kokkos compute | Dead code removal. |
| 10 | New pair style `nequip/kk` = `PairAllegroKokkos<true>` instantiation | Pure addition (upstream only instantiated `<false>` for Kokkos). |

Everything else in the diff is new code (new methods, new blocks) reached only when
`is_multirank == true`.

---

## Verification

Validated on LUMI-G (MI250X, ROCm 6.3.4, torch 2.9, LAMMPS `patch_30Mar2026`), 1–32
GCDs, real production models (OAM-S/M and a fine-tuned OAM-M):

- **Correctness gate** (script in the campaign repo, run for every compiled model):
  single-rank `pair_nequip` vs multi-rank build at 1 rank, |ΔPE|/atom < 1e-4 (measured
  ≈ exact); multi-GPU PE rank-invariance 1 vs 8 vs 32 GCDs, ΔPE/atom ≈ 1e-6; max |ΔF| vs
  single-rank reference ≈ 1e-5 eV/Å; cross-engine agreement with `mliap/kk` to 1e-3 eV
  total PE on a 2880-atom cell.
- **Benchmarks** (bound-corrected, OEQ on both paths): parity with `mliap/kk` when
  compute-bound (≥ ~2800 atoms/GCD); ~2.2–2.6× higher strong-scaling ceiling in the
  small-per-GCD regime (e.g. MgO+water 2880 atoms: 77 vs 35 ts/s; Cu 2048: 98 vs 38);
  markedly lower run-to-run variance multi-node.

Since resolved on this branch: exact comm-buffer sizing (audit row 1); a README usage
section for multi-GPU `pair_nequip`; a CPU-runnable regression test of the export path
(in the `nequip` fork's test suite); and the branch itself compiles and passes the
correctness gate + `clear`-guard repro on LUMI.

Still open:
- compile verification is LUMI/ROCm + CUDA only so far (no CPU-Kokkos build tested);
- the deployed frozen production module predates this branch (by design — it is never
  rebuilt in place); a new module suffix would be cut from this branch when needed.
