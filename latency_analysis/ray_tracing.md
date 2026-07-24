# Ray Tracing — Latency

Ray tracing is the highest-fidelity / lowest-speed rendering method in RAICHU
(Table 1). This document tracks the faithful reproduction of the RT pipeline as
CGRA-mapped kernels with measured IIs. Reference: RAICHU §RT (Fig 1: rays
interact with a **BVH** via **intersection tests**). See [README.md](README.md)
for the latency model and caveats.

## Pipeline stages → CGRA

RAICHU's memory/compute decoupling maps the memory-bound BVH traversal /
compaction to the M-CGRA and the compute-bound intersection / shading to the
C-CGRA (whose PEs carry the FVCU used for the AABB slab reduction).

| Stage | role | CGRA | rationale |
|---|---|---|---|
| Ray generation | camera → primary ray dirs (normalize) | **C-CGRA** | per-pixel compute (`fsqrt`) |
| BVH AABB slab test | ray-box intersection, min/max reduce | **C-CGRA** | FVCU `fmin/fmax` reduction (RAICHU AABB primitive) |
| Hit-node compaction | pack hit nodes into dense task list | **M-CGRA** | data-dependent **scatter** with write cursor |
| Ray-triangle (Möller-Trumbore) | edge/cross/dot + barycentric + closest hit | **C-CGRA** | cross/dot-heavy compute + `best_t` recurrence |
| Shading / accumulation | Lambertian + distance falloff → pixel | **C-CGRA** | per-hit compute |

## Measured IIs (all real mapper runs)

Flow: `clang++ -O3 → mlir-translate → assign-accelerator → lower-llvm-to-neura
→ fuse-pattern → promote-input-arg-to-const → fold-constant → canonicalize-return
→ canonicalize-live-in → leverage-predicated-value → transform-ctrl-to-data-flow
→ insert-data-mov → map-to-accelerator` (`backtrack-config=simple`, `max-steps=8`
unless noted). Sources in `test/c2llvm2mlir/ray_tracing/`.

| stage | kernel | source | HW | II | steps |
|---|---|---|---|---|---|
| Ray gen | rtRayGen | `rt_ray_gen.cpp` | C-CGRA | **11** | 16 |
| BVH AABB (FVCU G3) | rtAABB | `rt_aabb.cpp` | C-CGRA | **10** | 16 |
| Compaction | rtCompact | `rt_compact.cpp` | M-CGRA | **7** | 11 |
| Triangle · setup | rtTriSetup | `rt_triangle_setup.cpp` | C-CGRA | **12** | 14 |
| Triangle · v/t test | rtTriVT | `rt_tri_vt.cpp` | C-CGRA | **13** | 24 (max-steps=6) |
| Triangle · closest hit | rtTriReduce | `rt_tri_reduce.cpp` | C-CGRA | **7** | 12 |
| Shading | rtShade | `rt_shade.cpp` | C-CGRA | **9** | 15 |

- **AABB slab test uses the FVCU G3 + G4 modes** (RAICHU §4.4). `raichu-fuse-fvcu`
  rewrites the whole slab test into FVCU compare primitives: the 3 per-axis
  `fmin(t1,t2)/fmax(t1,t2)` pairs become **3 `neura.fvc_cmpswap`** (G4 pairwise
  compare-and-swap = interval tightening), and the `tnear`/`tfar` reductions
  become **`neura.fvc_max3`/`neura.fvc_min3`** (G3 comparison-and-interval
  update). The measured AABB II is **10** (down from 11), and the slab test now
  runs entirely on the FVCU compare datapath — the faithful hardware path.
- **Compaction** is the same cursor-driven scatter as the 3DGS sort partition —
  irregular memory orchestration on the M-CGRA.
- The Möller-Trumbore body is 110 ops; split into setup / v-t / closest-hit
  (over the mapper's ~40-op ceiling). `rtTriReduce`'s `rec_mii` is the closest-
  hit `best_t` recurrence, analogous to blending's transmittance recurrence.

## Latency

### Deliverable to the simulator: per-kernel latency

`latency = II·(trip_count − 1) + steps`; II/steps are trip-count-independent
hardware figures, `trip_count` is a workload parameter. Demonstrator
`trip_count = 32`; at 1 GHz, 1 cycle = 1 ns.

| stage | kernel | II | steps | latency @N=32 (cyc) |
|---|---|---|---|---|
| Ray gen | rtRayGen | 11 | 16 | 357 |
| BVH AABB (FVCU G3) | rtAABB | 10 | 16 | 326 |
| Compaction | rtCompact | 7 | 11 | 228 |
| Triangle · setup | rtTriSetup | 12 | 14 | 386 |
| Triangle · v/t | rtTriVT | 13 | 24 | 427 |
| Triangle · reduce | rtTriReduce | 7 | 12 | 229 |
| Shading | rtShade | 9 | 15 | 294 |
| Triangle (setup+vt+reduce) | | | | 1042 |

### Per-stage trip counts the simulator must apply

RT latency is dominated by **BVH traversal** (many nodes per ray) and
**triangle intersection** (many candidate primitives per ray), which is why the
paper rates RT as the slowest / most memory-bound method. The simulator should
weight each kernel by its real iteration count:

| stage | iteration count (per frame) | driver |
|---|---|---|
| Ray gen | `pixels` | one primary ray per pixel |
| BVH AABB + compaction | `rays · nodes_visited` | traversal depth ≈ `O(log #prims)` × branching |
| Triangle intersection | `rays · leaf_prims_tested` | candidate triangles per ray |
| Shading | `rays · hits` (≈ rays) | one shade per closest hit |

BVH traversal + triangle tests iterate far more than ray-gen/shading, so they
dominate — the AABB slab test (II=11) and triangle v/t (II=13), amplified by
traversal/candidate counts, set the frame time. Secondary rays (shadows,
reflections) multiply the traversal/intersection counts further.

### Notes for the simulator

- Use II/steps as the fixed per-kernel cost; feed real `trip_count`
  (rays × traversal/candidate counts) per stage.
- The BVH traversal is iterative (AABB test → compact → descend), so its total
  cost = `Σ over levels (rays_active · II_aabb + compaction)`.
- IIs are `backtrack-config=simple` quality (greedy times out on the larger
  bodies), so they are **upper bounds** — the true II is ≤ the reported value.

## FVCU gate modes (complete set, RAICHU §4.4)

`raichu-fuse-fvcu` now implements all four FVCU gate modes as fusion patterns:

| mode | primitive | fuses | used by |
|---|---|---|---|
| G1 | `fvc_partial_dot6` | six-lane FMA (op defined) | NeRF / L-NeRF matrix-vector |
| G2 | `fvc_reduce6` | 6-value sum `a+…+f` → 1 reduce | NeRF / L-NeRF dot completion |
| G3 | `fvc_max3` / `fvc_min3` | nested `fmax`/`fmin` (AABB `tnear`/`tfar`) | RT AABB slab test |
| G4 | `fvc_cmpswap` | `fmin(a,b)`+`fmax(a,b)` → (lo,hi) | RT slab per-axis / interval bounds / 3DGS ordering |

Each has: op def (`NeuraOps.td`), `OperationKind` + `fvcu` FU mapping
(`Architecture.h`), classifier (`mapping_util.cpp`), fusion pattern
(`RaichuFuseFvcuPass.cpp`), and latency modeling (`RaichuModelFvcuLatencyPass.cpp`).

### FVCU fusion — II impact (measured, `--raichu-fuse-fvcu`)

| kernel | II w/o FVCU | II w/ FVCU | fused ops | note |
|---|---|---|---|---|
| `rt_aabb` (slab test) | 11 | **10** | 3×`fvc_cmpswap` + `fvc_max3` + `fvc_min3` | whole slab test on the FVCU compare datapath |
| `bbox_radius` (3DGS) | 10 | 10 | 1×`fvc_cmpswap` (λ min/max) | II bound by `fsqrt`, not the compare |
| 6-sum (L-NeRF proxy) | — | **6** | 1×`fvc_reduce6` | G2 reduce fires; for NeRF/L-NeRF dots |

**Finding:** the FVCU fusion collapses the min/max/reduce trees into single FVCU
compare ops (faithful hardware path, fewer/denser ops), but the **II improvement
is marginal** — only `rt_aabb` drops (11→10). On kernels like `bbox_radius` the
II is set by the `fdiv`/`fsqrt` critical path, so fusing the compares doesn't
move it. `intersect_shape`'s eigenvalue min/max is constant-folded away by clang
(`max(mid+disc, mid−disc) = mid+disc` since `disc = sqrt(...) ≥ 0`), so there is
nothing to fuse there. The FVCU modes' main value is faithful HW modeling and
op-count reduction (mapping tractability), not large II wins on these kernels.

## Other lowering support added for RT

- `llvm.intr.sqrt` / `llvm.intr.exp` intrinsic forms → `neura.fsqrt` /
  `neura.fexp` (clang emits the intrinsic for sqrt/exp of *computed* values;
  the libm-call forms `@sqrtf`/`@expf` were already handled). Ray generation's
  `1/sqrt(len²)` normalize needed the intrinsic form.

## Status

**Every stage of the RT pipeline is reproduced as a CGRA-mapped kernel with a
measured II** — ray generation, BVH AABB slab test, hit-node compaction,
Möller-Trumbore ray-triangle intersection (setup + v/t + closest-hit), and
shading. Large bodies (Möller-Trumbore, 110 ops) are split into faithful
pipeline sub-stages to fit the mapper's ~40-op ceiling; split points are
dataflow boundaries that preserve all functional points.

### Remaining (honest)

- BVH traversal control (the iterative descend + node fetch that chains AABB →
  compaction across levels) is modeled per-kernel (AABB test + compaction), not
  as a single fused traversal loop (pointer-chasing dynamic loops are not
  CGRA-mappable directly — same reason as 3DGS tile-range binning).
- Single BVH (no TLAS/BLAS two-level); primary rays only (no secondary-ray
  recursion, which the simulator models by multiplying traversal counts).
