# RAICHU Compiler — Artifact Evaluation (AE)

This directory is the **compiler** subproject of the RAICHU artifact. RAICHU's
artifact has three parts — **simulator**, **RTL**, and **compiler** — and this
repository is the compiler part: it lowers each rendering-method kernel to the
Neura dialect and maps it onto the RAICHU heterogeneous CGRA (M-CGRA / C-CGRA),
producing the per-kernel timing the simulator consumes.

## What this AE reproduces

For every kernel of the **five RAICHU rendering methods** it reports the mapper's
- **`II`** (Initiation Interval, steady-state throughput of the pipelined loop), and
- **`steps`** (schedule makespan of one iteration = max `time_step` + 1).

These are the compiler's deliverable to the simulator. The end-to-end frame
latency is composed downstream as

```
latency(kernel) = II * (trip_count - 1) + steps
```

where `trip_count` is a workload parameter (number of Gaussians / samples / rays
/ ranks …) supplied by the scene and model configuration — see the per-method
notes in `../latency_analysis/`.

## Methods and kernels

| Method | kernels | fabric |
|---|---|---|
| **M-NeRF** (classic MLP NeRF) | ray sampling, Fourier positional encoding (`sin/cos`), MLP layer, volume accumulation | C-CGRA |
| **H-NeRF** (hash-grid / Instant-NGP) | hash + batched gather (M-CGRA), trilinear interp, vectorized MLP dot | M/C-CGRA |
| **L-NeRF** (TensoRF, low-rank) | factor gather, VM product, rank reduce (FVCU G2), appearance projection, MLP decode | M/C-CGRA |
| **3DGS** | covariance (quat→M→Σ₃D→Σ₂D), conic, bbox radius, SH color, tile binning, sort partition/compare-swap, α (real `exp`), 3-ch compositing, early termination, shape-aware intersection, sub-tile bitmap | M/C-CGRA |
| **Ray Tracing** | ray gen, BVH AABB slab test (FVCU G3/G4), hit compaction, Möller-Trumbore (setup / v-t / closest-hit), shading | M/C-CGRA |

Kernel sources: `../test/c2llvm2mlir/{mlp_nerf,low_rank_nerf,render,ray_tracing}/*.cpp`
and the hand-written `../test/raichu/*.mlir` (Hash NeRF).

## Prerequisites

Run inside the Neura dev container (`neura-dev-tangyz`), with the project built:

```bash
cd /workspace/dataflow
cmake --build build --target mlir-neura-opt      # if not already built
```

The driver expects:
- `mlir-neura-opt` at `./build/tools/mlir-neura-opt/mlir-neura-opt` (override with `OPT=`);
- `clang++` / `mlir-translate` at `/workspace/llvm-project/build/bin` (override with `LLVM=`).

## Running

```bash
# all five methods (long: ~1 hour; some kernels take a few minutes to map)
bash raichu_ae/run_ae.sh

# one method at a time (recommended)
bash raichu_ae/run_ae.sh mnerf     # M-NeRF
bash raichu_ae/run_ae.sh hnerf     # Hash NeRF
bash raichu_ae/run_ae.sh lnerf     # Low-rank NeRF
bash raichu_ae/run_ae.sh 3dgs      # 3D Gaussian Splatting
bash raichu_ae/run_ae.sh rt        # Ray Tracing
```

Environment overrides: `OPT`, `LLVM`, `TIMEOUT` (per-kernel seconds, default 900).

## Output

A table of `METHOD | KERNEL | CGRA | II | steps | check`, e.g.

```
METHOD   KERNEL                   CGRA     II       steps    check
---------------------------------------------------------------------
RT       rt_aabb                  c-cgra   10       16       ok
RT       rt_compact               m-cgra   7        11       ok
...
PASS=N  FAIL=0
```

`check` compares the measured `II` to the recorded reference value (`ok` = match,
`~E` = mapped but differs from reference `E`, `x` = failed to map). `II` is the
figure of merit; `steps` is reported for the latency formula. Exit code is 0
only if every kernel maps.

## Notes / caveats

- **II/steps are the hardware-timing deliverable**; absolute cycle counts at the
  demonstrator `trip_count` (32) are illustrative — the simulator scales by the
  real per-stage `trip_count`.
- Larger fused kernels are mapped with `backtrack-config=simple` (the exhaustive
  search is slow); the reported `II` is therefore an upper bound (true II ≤ it).
- Large single kernels are split into faithful pipeline sub-stages (covariance
  ×3, blending, Möller-Trumbore ×3, VM product+reduce) to fit the heuristic
  mapper's practical op-count ceiling; the split points are dataflow boundaries
  that preserve all functional points.
- Detailed per-method latency analysis and the reproduced compiler-optimization
  coverage table live in `../latency_analysis/`.
