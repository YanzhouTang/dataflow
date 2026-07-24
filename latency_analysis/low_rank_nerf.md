# Low-Rank NeRF (TensoRF) — Latency

Low-rank NeRF (TensoRF-style) factorizes the radiance-field volume into a sum of
rank-R low-rank components (vector-matrix / VM decomposition). This document
tracks the faithful reproduction of the L-NeRF pipeline as CGRA-mapped kernels
with measured IIs. Reference: RAICHU §4.3.2. See [README.md](README.md) for the
latency model.

## Pipeline stages → CGRA

RAICHU §4.3.2: *"Low rank NeRF map to the M-CGRA for coherent embedding
generation and task mapping ... extensive tile reuse ... the data access engine
feeds these tasks to the C-CGRA for high utilization execution."* The density /
appearance at a sample are

```
sigma = Σ_r [ vx_r · Myz_r + vy_r · Mxz_r + vz_r · Mxy_r ]     (VM decomposition)
```

| Stage | role | CGRA |
|---|---|---|
| Factor gather + line/plane interpolation | coherent embedding generation | **M-CGRA** gather (GA unit) → interp on C-CGRA |
| Per-rank VM product `vx·Myz + vy·Mxz + vz·Mxy` | the FMA / G1 stage | **C-CGRA** |
| Rank reduction `Σ_r term_r` | the **FVCU G2 reduce** stage | **C-CGRA** |
| Appearance MLP decode (feat·w + b, ReLU) | dot-product decode | **C-CGRA** |
| Volume-render accumulation | alpha compositing recurrence | **C-CGRA** (shared with NeRF/3DGS) |

The product↔reduce split mirrors RAICHU's FVCU **G1 (six-lane FMA) → G2
(pair-and-sum reduce)** decomposition of a low-rank dot product.

## Measured IIs (all real mapper runs)

Flow: `clang++ -O3 → mlir-translate → assign-accelerator → lower-llvm-to-neura
→ fuse-pattern → raichu-fuse-fvcu → raichu-model-fvcu-latency →
promote-input-arg-to-const → fold-constant → canonicalize-return →
canonicalize-live-in → leverage-predicated-value → transform-ctrl-to-data-flow
→ insert-data-mov → map-to-accelerator` (`backtrack-config=simple max-steps=8`).
Sources in `test/c2llvm2mlir/low_rank_nerf/`.

| stage | kernel | source | HW | II | steps |
|---|---|---|---|---|---|
| Factor gather + interp | lnerfLineGather | `lnerf_line_gather.cpp` | C-CGRA† | **6** | 9 |
| Per-rank VM product | lnerfVMProduct | `lnerf_vm_product.cpp` | C-CGRA | **6** | 8 |
| Rank reduction (**G2**) | lnerfVMReduce | `lnerf_vm_reduce.cpp` | C-CGRA | **6** | 9 |
| Appearance MLP decode | lnerfMLP | `lnerf_mlp.cpp` | C-CGRA | **11** | 21 |

† The raw factor gather (indexed loads) is an M-CGRA GA-unit task (as in
[hash_nerf.md](hash_nerf.md)); the linear interpolation `a + f·(b−a)` uses
`fmul_fadd`, which the M-CGRA lacks, so the interp maps on the C-CGRA. In the
real pipeline the M-CGRA gathers and the C-CGRA interpolates.

- **`lnerf_vm_reduce` fires the FVCU G2 mode**: the 6-way rank sum fuses into one
  `neura.fvc_reduce6` (verified in the mapping). This is exactly the "reduce
  stage commonly used in NeRF and low-rank NeRF" the paper assigns to G2.
- **G1 (six-element MAC)** is realized on the appearance-matrix projection
  `feat = Σ_r c_r·B_r`. The mapping-efficient form is the vector-load-fed FVCU
  dot (`lnerf_appear_proj_vec.mlir`): two 6-wide `neura.vector_load`s feed one
  `neura.fvc_dot3`, mapping at **II=2** (fan-in 2) — the paper's "FVCU supported
  by vector loads matching the FVCU layout" (the width-6 instance of the
  `vector_load + fvc_dot3` path in [hash_nerf.md](hash_nerf.md)). A scalar 6-MAC
  chain also fuses (recognized), but a 12-scalar-port op exceeds the PE input
  budget, so the vector-fed form is the one used.
- The per-rank VM product (`lnerf_vm_product`) is the G1 FMA stage; kept separate
  from the reduce so the reduction stays a pure add tree (a fused single-kernel
  VM body of 138 ops both exceeds the mapper's ~40-op ceiling and collapses the
  reduction into `fmul_fadd` MAC chains, hiding G2).

## Latency

`latency = II·(trip_count − 1) + steps`; II/steps are trip-count-independent.
Demonstrator `trip_count = 32`; at 1 GHz, 1 cycle = 1 ns.

| stage | kernel | II | steps | latency @N=32 (cyc) |
|---|---|---|---|---|
| Factor gather + interp | lnerfLineGather | 6 | 9 | 195 |
| Per-rank VM product | lnerfVMProduct | 6 | 8 | 194 |
| Rank reduction (G2) | lnerfVMReduce | 6 | 9 | 195 |
| Appearance MLP decode | lnerfMLP | 11 | 21 | 362 |

### Per-stage trip counts the simulator must apply

| stage | iteration count (per frame) | driver |
|---|---|---|
| Factor gather + interp | `samples · R · 3axes` | per-rank per-axis line/plane fetch |
| VM product + reduce | `samples · R` | per-rank contribution then reduce |
| MLP decode | `samples · hidden` | one dot per hidden unit |
| Accumulation | `rays · samples_per_ray` | alpha compositing along the ray |

L-NeRF is lighter than dense/hash NeRF per sample (the low-rank factorization
replaces a big MLP with a few rank dots), so the per-sample cost is dominated by
the factor gather (memory) plus the small decode MLP — consistent with the
paper rating L-NeRF as high-speed / high-memory-efficiency.

### Notes for the simulator

- Use II/steps as the fixed per-kernel cost; feed real `trip_count`
  (samples × rank × axes) per stage.
- IIs are `backtrack-config=simple` quality (upper bounds; true II ≤ reported).

## Status

**Every stage of the L-NeRF (TensoRF) pipeline is reproduced as a CGRA-mapped
kernel with a measured II** — factor gather + interpolation, per-rank VM product,
rank reduction (via the FVCU **G2** `fvc_reduce6`), and the appearance MLP decode
(accumulation shared with the NeRF/3DGS blending kernels). This is the pipeline
that most directly exercises the FVCU G1/G2 low-rank-dot modes.

### Remaining (honest)

- VM decomposition only (not the CP variant); R = 6 ranks (demonstrator).
- The product and reduce stages are separate kernels (faithful to the G1→G2
  split); a single fused VM body would collapse the reduce into `fmul_fadd`.
- View-dependent SH / positional encoding of the direction into the MLP is
  simplified to a plain feature·weight dot.
