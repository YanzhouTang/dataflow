# 3D Gaussian Splatting (3DGS) — Latency

3DGS is a rasterization pipeline. This document tracks the **faithful**
reproduction of every functional point of the pipeline as CGRA-mapped kernels
with measured IIs. Algorithm reference: GSCore (ASPLOS'24) for the pipeline
stages, *hierarchical sorting*, *shape-aware intersection*, and *sub-tile
skipping*; the memory/compute decomposition follows RAICHU §4.

See [README.md](README.md) for the latency model and caveats.

## Pipeline stages → CGRA

GSCore §3.1 divides rendering into **Frustum Culling → Feature Computation →
Gaussian Sorting → Rasterization**. RAICHU maps compute-dense bodies to the
C-CGRA (every PE has fused mul-add + FVCU) and irregular memory reorganization
to the M-CGRA (gather/scatter). Note the M-CGRA has *no* `fmul_fadd`, so any
fused-multiply-add body must target the C-CGRA.

## Faithful functional-point coverage (all measured, real mapper runs)

Flow: `clang++ -O3 → mlir-translate → assign-accelerator → lower-llvm-to-neura
→ fuse-pattern → promote-input-arg-to-const → fold-constant → canonicalize-return
→ canonicalize-live-in → leverage-predicated-value → transform-ctrl-to-data-flow
→ insert-data-mov → map-to-accelerator` (`backtrack-config=simple max-steps=8`
unless noted). Sources in `test/c2llvm2mlir/render/`.

### 1. Frustum Culling

| functional point | source | HW | II |
|---|---|---|---|
| view transform + depth + frustum/screen test → keep mask | `frustum_cull.cpp` | C-CGRA | **12** |

### 2. Feature Computation

| functional point | source | HW | II |
|---|---|---|---|
| 3D covariance: `M = R(quat)·S` | `rot_m_from_quat.cpp` | C-CGRA | **9** |
| 3D covariance: `Σ₃D = M·Mᵀ` | `sigma3d_from_m.cpp` | C-CGRA | **7** |
| EWA Jacobian projection `Σ₂D = J·Σ₃D·Jᵀ` | `cov2d_project.cpp` | C-CGRA | **13** |
| perspective projection + conic `= inv(Σ₂D)` | `preprocess_full.cpp` | C-CGRA | **11** |
| bbox radius `r = 3√λ` (eigenvalues, Eq.3; FVCU G4 fuses the λ min/max) | `bbox_radius.cpp` | C-CGRA | **10** |
| SH → RGB color (degree-2, exact 3DGS constants) | `sh_color.cpp` | C-CGRA | **14** |

The covariance derivation is a 3-stage pipeline (`quat→M → Σ₃D → Σ₂D`) because
the fused single-kernel body is 152 ops (over the mapper's ~40-op ceiling).

`bbox_radius` uses the FVCU **G4** compare-swap (`raichu-fuse-fvcu` fuses the
eigenvalue `fmin(λ1,λ2)/fmax(λ1,λ2)` into one `fvc_cmpswap`); the II stays 10
because it is bound by the `fsqrt`/`fdiv` critical path, not the compare. See
[ray_tracing.md](ray_tracing.md) for the full FVCU gate-mode set (G1–G4).

### 3. Gaussian Sorting (GSCore hierarchical sort)

| functional point | source | HW | II | rec_mii |
|---|---|---|---|---|
| tile-range coverage count (per-tile duplication factor) | `tile_range_binning.cpp` | M-CGRA | **10** | — |
| sort-key generation `(tile_id, depth)` | `sort_binning.cpp` | C-CGRA | **7** | — |
| approximate sort — **pivot partition** (Stage-1) | `sort_partition.cpp` | M-CGRA | **7** | 6 |
| precise sort — compare-swap network stage (Stage-2) | `sort_stage.cpp` | M-CGRA | **5** | 5 |

- **Pivot partition is GSCore's key contribution** (Fig 7/8): compare each
  Gaussian's depth to a pivot and **scatter** into a lower/higher chunk via
  running cursors (`rec_mii=6`). Precise sorting runs only on the lower chunk;
  early termination skips the higher chunk.
- **Tile-range duplication** (a Gaussian's bbox spans a rectangle of tiles →
  one key per covered tile) is the 3-step primitive: (1) coverage count [mapped
  above], (2) prefix-sum of counts → offsets, (3) cursor-driven scatter (same
  pattern as `sort_partition`). The dynamic per-Gaussian nested loop is *not*
  CGRA-mappable directly; count+scan+scatter is the faithful decomposition.
- A full sort composes `O(log²N)` (bitonic) or `O(N)` (odd-even) compare-swap
  stages, each at `II=5`.

### 4. Rasterization / Blending

| functional point | source | HW | II | rec_mii |
|---|---|---|---|---|
| α-computation: gather + conic power + **real `exp`** + clamp | `blend_alpha_exp.cpp` | C-CGRA | **12** | — |
| compositing: 3-channel accumulate + `T *= (1−α)` recurrence | `blend_stage2.cpp` | C-CGRA | **7** | 5 |
| early ray termination (`T < t_min` → loop exit) | `blend_early_term.cpp` | C-CGRA | **11** | 5 |

- **`rec_mii=5` in compositing is the transmittance recurrence** — the
  fundamental blending bottleneck, matching the paper.
- α now uses the **true exponential** (`neura.fexp`), not the earlier 2nd-order
  polynomial approximation.

### 5. GSCore optimizations

| functional point | source | HW | II |
|---|---|---|---|
| shape-aware intersection test (AABB/OBB selection) | `intersect_shape.cpp` | C-CGRA | **14** |
| sub-tile skipping bitmap (2×2 subtile overlap → 4-bit mask) | `subtile_bitmap.cpp` | C-CGRA | **13** |

## New ops added this round (Neura dialect)

To reproduce the real math (previously approximated / avoided):

- **`neura.fexp`** (`exp`) and **`neura.fsqrt`** (`sqrt`) — unary transcendental
  ops on the `fdiv` FU. Added: op defs (`NeuraOps.td`), `OperationKind`
  `IFExp/IFSqrt` + FU mapping (`Architecture.h`), classifier (`mapping_util.cpp`),
  and `llvm.call @expf/@sqrtf → neura.fexp/fsqrt` lowering (`LlvmToNeuraPass.cpp`).
- **`llvm.sitofp → neura.cast`** lowering (int→float, for tile-index math);
  `fptosi` already existed.

## Mapper knobs (from the previous round)

- **`map-to-accelerator max-steps=<N>`** (default 10): per-II time-step search
  window (`slots = II·N`). Smaller = faster; too small fails if the schedule is
  longer than the window.
- **`NEURA_MAP_VERBOSE=1`**: gates the mapper's per-candidate/per-route logging
  (default off; removes hot-path I/O).
- **Architecture assignment as a config**: compute-dense (fused mul-add) kernels
  must target the C-CGRA; forcing them onto the M-CGRA causes unbounded
  backtracking.

## Latency

### Deliverable to the simulator: per-kernel latency

The primary hand-off to the simulator is the **per-kernel latency model**:

```
latency(kernel) = II · (trip_count − 1) + steps
```

where `II` (initiation interval) and `steps` (schedule makespan = max
`time_step`+1) are read from the mapper and are **trip-count-independent**
hardware figures; only `trip_count` is a workload parameter the simulator
supplies. At 1 GHz, 1 cycle = 1 ns. The table below reports II/steps for every
kernel (the demonstrator `trip_count` is 32 Gaussians, 16 pairs for the
compare-swap stage).

| stage | kernel | II | steps | source | HW |
|---|---|---|---|---|---|
| Culling | frustum_cull | 12 | 20 | `frustum_cull.cpp` | C-CGRA |
| Feature | rot_m_from_quat | 9 | 11 | `rot_m_from_quat.cpp` | C-CGRA |
| Feature | sigma3d_from_m | 7 | 10 | `sigma3d_from_m.cpp` | C-CGRA |
| Feature | cov2d_project | 13 | 16 | `cov2d_project.cpp` | C-CGRA |
| Feature | proj + conic | 11 | 20 | `preprocess_full.cpp` | C-CGRA |
| Feature | bbox_radius | 10 | 15 | `bbox_radius.cpp` | C-CGRA |
| Feature | sh_color | 14 | 26 | `sh_color.cpp` | C-CGRA |
| Sort | tile_coverage | 10 | 17 | `tile_range_binning.cpp` | M-CGRA |
| Sort | sort_binning | 7 | 13 | `sort_binning.cpp` | C-CGRA |
| Sort | sort_partition | 7 | 11 | `sort_partition.cpp` | M-CGRA |
| Sort | sort_stage (1 stage) | 5 | 12 | `sort_stage.cpp` | M-CGRA |
| Blend | blend_alpha_exp | 12 | 16 | `blend_alpha_exp.cpp` | C-CGRA |
| Blend | blend_composite | 7 | 13 | `blend_stage2.cpp` | C-CGRA |
| Blend | blend_early_term* | 11 | 22 | `blend_early_term.cpp` | C-CGRA |
| Opt | intersect_shape | 14 | 22 | `intersect_shape.cpp` | C-CGRA |
| Opt | subtile_bitmap | 13 | 19 | `subtile_bitmap.cpp` | C-CGRA |

\* alternative to `blend_composite` when early-termination is enabled.

Example (demonstrator, `trip_count=32`; sort_stage `=16`):
`sh_color = 14·31+26 = 460`, `blend_composite = 7·31+13 = 230`,
`sort_stage = 5·15+12 = 87` cycles.

### Per-stage trip counts the simulator must apply

Per-kernel latency alone is not the frame time — each stage runs with a very
different `trip_count`, and that weighting (not the raw per-kernel latency) is
what reproduces the paper's stage breakdown. The simulator should scale each
kernel by its real iteration count:

| stage | iteration count (per frame) | driver |
|---|---|---|
| Culling | `N_total` (all Gaussians) | one pass over the scene |
| Feature | `N′` (visible Gaussians) | per-visible-Gaussian |
| Sorting | `N′ · tiles_per_gaussian` keys + sort passes | tile-range duplication |
| Blending | `pixels · avg_gaussians_per_pixel` | per-pixel × sorted-list depth |

Blending's iteration count is **orders of magnitude larger** than Feature's, so
even though per-body IIs are similar (7–14), blending dominates the frame.

**Weighted frame estimate** (representative: `N_total=500K`, `N′=200K`,
1280×720 = 0.92 M px, ~4 tiles/Gaussian, ~20 composited Gaussians/pixel after
early termination):

| stage | per-elem cost | iters | cycles | share |
|---|---|---|---|---|
| Culling | 12 | 500 K | 6.0 M | ~3% |
| Feature | ~64 (Σ sub-kernels) | 200 K | 12.8 M | ~7% |
| Sorting | ~24/key + sort | ~0.8 M keys | ~30 M | ~17% |
| **Blending** | 7 | 0.92 M × 20 | **~129 M** | **~72%** |
| **frame** | | | ~178 M ≈ 0.18 ms @1GHz | 100% |

This matches GSCore Fig 2b (Blending 55–75%, Sort 20–35%, Preprocess 4–15%).
The earlier equal-`trip_count=32` per-kernel table is a pure throughput view and
must **not** be read as a stage-share breakdown.

### Notes for the simulator

- **Use II/steps as the fixed per-kernel cost**; feed real `trip_count` per
  stage (table above) to get frame latency.
- **Full sort** chains many compare-swap stages: bitonic `≈ log²(N)` or odd-even
  `≈ N` stages, each at the `sort_stage` cost.
- IIs for the larger bodies are `backtrack-config=simple` quality (greedy times
  out), so they are **upper bounds** — the true II is ≤ the reported value.

## Status

**Every functional point of the GSCore/3DGS pipeline is now reproduced as a
CGRA-mapped kernel with a measured II** — frustum culling, full feature
computation (quaternion covariance + EWA projection + conic + eigenvalue radius
+ SH color), hierarchical sorting (tile-range binning + key gen + approximate
pivot partition + precise compare-swap), rasterization (real-`exp` α +
transmittance compositing + early termination), and GSCore's three optimizations
(shape-aware intersection, sub-tile skipping). Large fused bodies are split into
faithful pipeline sub-stages (covariance ×3, blending ×2) to fit the mapper's
~40-op ceiling; the split points are dataflow boundaries that preserve all
functional points.

### Remaining approximations (honest)

- SH is degree-2 for one channel (RGB replicates ×3; degree-3 adds 7 basis).
- EWA Jacobian assumes a view-axis-aligned frame (`W = I`).
- Sort keys use f32 arithmetic packing instead of true bit-packed radix keys.
- Compare-swap is one network stage; a full sort chains many.
