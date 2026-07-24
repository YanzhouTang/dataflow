# RAICHU Latency Analysis

Per-rendering-method latency records for the RAICHU compiler reproduction.
Each rendering method has its own file; every kernel of that method is mapped
onto the RAICHU M-CGRA / C-CGRA and its Initiation Interval (II) is read back
from the mapper, then composed into a latency estimate.

## Rendering methods

| Method | File | Status |
|---|---|---|
| Hash Grid NeRF (Instant-NGP style) | [hash_nerf.md](hash_nerf.md) | ✅ implemented |
| MLP-based NeRF | [mlp_nerf.md](mlp_nerf.md) | ✅ full pipeline (sampling, Fourier encoding, MLP, accumulation) |
| Low-Rank NeRF (TensoRF) | [low_rank_nerf.md](low_rank_nerf.md) | ✅ full pipeline (factor gather, VM product, rank reduce via FVCU G2, MLP decode) |
| 3D Gaussian Splatting | [3dgs.md](3dgs.md) | ✅ full pipeline (culling, feature comp, hierarchical sort, blending + GSCore opts) |
| Ray Tracing | [ray_tracing.md](ray_tracing.md) | ✅ full pipeline (ray gen, BVH AABB, compaction, Möller-Trumbore, shading) |

All five RAICHU rendering methods have their kernels developed, compiled, mapped,
and measured (real `II`/`steps` from the mapper).

## Compiler-optimization coverage (RAICHU §4.2 / §4.4)

| Optimization | Pass / mechanism | Status | Applied to |
|---|---|---|---|
| Operator classification (M/C-CGRA) | `raichu-classify-target` (name-based) | ✅ reproduced, validated on all 5 methods | all (tag→spec auto-selection = orchestration, TODO) |
| Constant folding | `fold-constant` | ✅ | all |
| Op fusion `fmul_fadd`/`fadd_fadd` | `fuse-pattern` | ✅ | all |
| FVCU G1 six-element MAC | `vector_load` + `fvc_dot3` (vector-fed) | ✅ | L-NeRF / Hash-NeRF (width-parameterized) |
| FVCU G2 reduce6 | `raichu-fuse-fvcu` | ✅ | L-NeRF rank reduce |
| FVCU G3 AABB slab | `raichu-fuse-fvcu` (`fvc_max3`/`fvc_min3`) | ✅ | Ray Tracing |
| FVCU G4 compare-swap | `raichu-fuse-fvcu` (`fvc_cmpswap`) | ✅ | Ray Tracing / 3DGS |
| Control→dataflow, predication | `transform-ctrl-to-data-flow`, `leverage-predicated-value` | ✅ | all |
| Vector loads feeding FVCU | `neura.vector_load` | ✅ | Hash-NeRF / L-NeRF |
| Heuristic mapping + II search | `map-to-accelerator` (+ `max-steps`, `NEURA_MAP_VERBOSE`) | ✅ | all |
| GA / FVCU width latency model | `raichu-model-ga-latency`, `raichu-model-fvcu-latency` | ✅ | all |
| Transcendental ops (real math) | `neura.fexp/fsqrt/fsin/fcos` + lowering | ✅ | 3DGS α, radius; NeRF encoding/accum |
| M-CGRA gather/scatter primitives | `neura.gather`/`neura.scatter` + `raichu-model-ga-latency` | ✅ (kernels authored manually) | Hash-NeRF, L-NeRF, 3DGS sort, RT compaction |
| — auto `N loads → 1 gather` folding | — | ⛔ out of scope (manual gather is supported) | — |
| DFG batch manipulation / shared-buffer bank assignment | partial (`AffineToTaskflow`) | ⏳ partial (DAE/orchestration side) | — |

**Verdict:** the compiler pipeline (classification → fusion incl. all four FVCU
modes → predication/dataflow → heuristic mapping → width-latency modeling →
real transcendental ops) is reproduced and applied across all five methods,
producing real `II` for every kernel. The M-CGRA gather/scatter primitives and
their GA-latency model exist and are used; **gather kernels are written manually
by design** (the automatic `load→gather` consolidation pass is intentionally out
of scope). The only remaining automation piece is tag→arch-spec orchestration
(multi-CGRA Taskflow), which does not affect the per-kernel `II`/`steps`
deliverable.

## Methodology

### 1. Per-kernel latency model

Each kernel is lowered to the Neura dialect and mapped by `map-to-accelerator`,
which reports `compiled_ii` (II). The single-kernel latency for `T` iterations
(loop trip count, or number of pipelined work items) is:

```
latency = II * (T - 1) + steps
```

- `II` (compiled_ii): steady-state initiation interval, from the mapper.
- `steps`: schedule makespan of one iteration = (max `time_step` in the mapping) + 1.
- `T` (trip_count): number of iterations / pipelined items.

### 2. Macro-pipeline composition

RAICHU runs the M-CGRA and C-CGRA as a producer-consumer macro-pipeline
(Section 4.3). For a linear pipeline of stages `i` processing `N` work items,
the composed latency is modeled as:

```
latency_total = (N - 1) * max_i(II_i) + sum_i(steps_i)
```

- Steady-state throughput is bounded by the slowest stage: `1 / max_i(II_i)`.
- `sum_i(steps_i)` approximates the pipeline fill+drain depth.
- Cycles are converted to time at the target 1 GHz (1 cycle = 1 ns).

### 3. Caveats (read before citing numbers)

- **II and steps are measured** by the compiler/mapper; they are solid.
- **Absolute latencies are illustrative**: kernels use small demonstrator
  dimensions (e.g., 4x8 matmul, 1 point / 8 corners for encoding), not the full
  NeRF Synthetic dimensions. Scaling dims changes `T` and (mildly) `steps`/`II`.
- **Granularity differs per stage**: encoding kernels are per-sample bodies
  (pipeline over `N` samples); MLP kernels loop internally over their matrix
  dimensions. The composition treats each as one pipeline stage.
- **Functional correctness is not validated here** — these are mapping/timing
  metrics; numerical validation is the external simulator's job.
- **Mapper config**: medium kernels use `--backtrack-config=greedy` (see the
  mapper-performance note in the project TODOs).
