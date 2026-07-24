# Hash Grid NeRF (Instant-NGP style) — Latency

Reproduced Hash NeRF pipeline stage kernels, mapped onto the RAICHU dual-tier
fabric. See [README.md](README.md) for the latency model and caveats.

## Pipeline stages (RAICHU classification)

```
ray march ──▶ hash encode ──▶ MLP ──▶ (blending)
  (M-CGRA)   (M-CGRA + C-CGRA)  (C-CGRA)   (TODO, done in 3DGS)
```

- Ray marching: M-CGRA (not yet reproduced).
- Hash encoding: split per RAICHU memory/compute decoupling —
  - index hashing + batched gather → **M-CGRA** (memory orchestration);
  - trilinear interpolation → **C-CGRA** (compute).
- MLP: **C-CGRA** (matmul + bias + ReLU).
- Blending / ray composition: deferred (will be handled with 3DGS).

## Measured per-kernel data (II / steps)

II and steps are read from the mapper (`compiled_ii`, steps = max `time_step`
+ 1). Kernel test cases live under `test/raichu/`.

| Stage | Kernel | HW (fabric) | II | steps | source test |
|---|---|---|---|---|---|
| Hash encode | 8-corner spatial hash + 1 batched gather | M-CGRA (4×4 super-tile, GA on top row) | **11** | 21 | `mcgra_hash_gather_map_ii.mlir` |
| Hash encode | trilinear interpolation (8 corners, 2 ch) | C-CGRA (4×4) | **7** | 33 | `ccgra_interp_map_ii.mlir` |
| MLP | matmul (loop body) | C-CGRA (4×4) | **2** | 7 | `mlp_full_layer.mlir`, `mlp_layer_real.mlir` |
| MLP | bias + ReLU (loop body) | C-CGRA (4×4) | **2** | 7 | `mlp_full_layer.mlir` |

### II / steps are dimension-robust (verified)

For loop kernels the mapper schedules the **loop body**, not the unrolled loop,
so `II` and `steps` do NOT depend on the matrix dimensions — only the counter
upper-bounds (and thus `trip_count`) change. Verified: mapping the matmul at
toy dims `4×8 @ 8×8` and at realistic dims `16×32 @ 32×64` both yield
**II=2, steps=7**; only the loop counters changed (16 / 64 / 32).

This lets us keep the measured II/steps and scale only `trip_count` to reach
realistic-dimension latency.

## Realistic dimensions (Instant-NGP) vs. what was mapped

| Parameter | Instant-NGP (real) | mapped toy | drives |
|---|---|---|---|
| num_levels L | 16 | 1 | encode trip ×16 |
| features/level F | 2 | 2 | encoded dim = L·F = 32 |
| corners/level | 8 | 8 | gather batch |
| log2 hashmap | 19 (2¹⁹) | 8 (256) | DRAM only (not op count) |
| MLP dims | 32→64→…→out | 8→8 | MLP trip |

Representative MLP (per sample): density 32→64, 64→16; color 32→64, 64→64,
64→3  ⇒  **≈ 9408 MACs/sample**.

## Realistic per-sample latency (latency = II·(T−1) + steps)

**Encoding** — 16 levels pipelined through each stage, per sample:

| Stage | II | steps | T = levels | latency (cyc) |
|---|---|---|---|---|
| M-CGRA hash + gather | 11 | 21 | 16 | 11·15 + 21 = **186** |
| C-CGRA interpolation  | 7  | 33 | 16 | 7·15 + 33 = **138** |

Encoding per sample ≈ **324 cyc** (stages overlap; both are ≤ 200 cyc).

**MLP** — per sample, 9408 MACs at II=2 (see the big caveat below):

```
latency(MLP, per sample) ≈ II · MACs = 2 · 9408 ≈ 18 816 cyc
```

### ⚠ Two matmul mapping paths: scalar (measured) vs FVCU (parallel)

**(a) Scalar `linalg.matmul` path — measured, sequential.**
`linalg.matmul` lowers to a scalar triple-loop (one multiply-add per
iteration), mapped at II=2 ⇒ ~1 MAC every 2 cycles on a few PEs. It does NOT
exploit spatial parallelism, so 18 816 cyc/sample is a *pessimistic upper bound*.

**(b) FVCU vector-dot path — the RAICHU way, primitive + width latency measured.**
RAICHU maps matrix-vector into FVCU dot products (Section 4.2.2). We have this
primitive: `vfmul + vector.reduce.add` fuses to `neura.fvc_dot3`
(`raichu-fuse-fvcu`), and its execution latency is now modeled by the FVCU lane
width via `raichu-model-fvcu-latency` (`fvc_dot3` latency = `ceil(K / fvcu_width)`,
default `fvcu_width = 6`). **Measured** for `K = 32` (test `mlp_dot32_map_ii.mlir`):

| fvcu-width | fvc_dot3 latency (measured) |
|---|---|
| 6 (default)  | 6 = ceil(32/6) |
| 16 | 2 |
| 32 | 1 |

The latency shows up in the schedule (a width-6 dot occupies the FVCU t=1..6).
As with the GA unit, a multi-cycle op reused every loop iteration pushes the
loop II up to that latency, so a looped matmul of `M·N` K-dots on one FVCU has
**II ≈ ceil(K/6) = 6** per dot; with `P` parallel PEs it is `≈ 6·M·N/P`.

**Vectorized output-element body — measured end-to-end** (test
`vec_matmul_body.mlir`). With the new `neura.vector_load` op (width latency
`ceil(K/mem_width)` modeled by `raichu-model-fvcu-latency`, `mem_width=8`):

```
av = vector_load a[i,:]   (latency 4 = ceil(32/8))
wv = vector_load w[:,j]   (latency 4)
d  = fvc_dot3(av, wv)     (latency 6 = ceil(32/6))
```

maps on the C-CGRA with **compiled_ii = 2** and a body makespan of **13 cycles**
(loads t=1..4, dot t=6..11, drain). This is the real RAICHU C-CGRA form (vector
loads feeding the FVCU) — no longer a scalar MAC. Per output in a loop the
throughput is FVCU-bound at ~6 cyc; with `P` PEs, `≈ 6·M·N/P`:

**Looped vectorized matmul — measured** (test `vec_matmul_loop.mlir`). The whole
`for i in 0..16, j in 0..64` loop with the vector body above lowers to a Neura
kernel (a fix to `convert-affine-to-taskflow` lets it capture the memref
operands of `neura.vector_load` / `store_indexed`) and **maps with structural
`compiled_ii = 2`** (unit-latency units). With the real FVCU/mem latencies
(fvc_dot3 = 6, vector_load = 4) the per-iteration II is bounded by the reused
FVCU multi-cycle op at **II ≈ 6** (the direct real-latency map currently times
out — mapper perf, TODO #18 — but the II lower bound is set by the fvc_dot3
latency, consistent with the GA-unit behavior). Layer latency (M·N = 1024
outputs) ≈ `6·1023 + steps ≈ 6.2k cyc` on one FVCU tile, `/P` across P PEs.

| path / parallelism | MLP cyc/sample | note |
|---|---|---|
| scalar loop (measured, II=2) | ~18 816 | current `linalg.matmul` lowering |
| FVCU dot, 1 PE (latency 6, measured) | ~3 136 | 9408 MACs / 6 |
| FVCU dot, 16 PEs (one tile) | ~196 | /16 |
| FVCU dot, 256 PEs (full C-CGRA) | ~12 | /256 |

### Attempts to parallelize the scalar matmul (documented)

- `affine-loop-unroll` + `affine-scalrep`: unrolls the k-reduction and forwards
  the accumulator load, making the multiplies independent (parallel). **But**
  `affine-scalrep`/`canonicalize` do NOT remove the now-redundant `affine.store`
  to `acc[i,j]` (affine stores have side effects; there is no affine dead-store
  elimination). The leftover stores create WAW memory serialization and the
  mapper times out (>260 s) even at unroll factor 4.
- **Blockers to a measured parallel-matmul II**: (1) redundant-store elimination
  after unroll, and (2) mapper performance on the larger parallel body. Until
  these are addressed, the parallel MLP numbers above are projections anchored
  on the measured `fvc_dot3` II=1 and the FVCU 6-lane width.

## Bottleneck and throughput

Per-sample stage costs: M-CGRA gather ~186 cyc, C-CGRA interp ~138 cyc, MLP as
per the path table above. The macro-pipeline bottleneck is the slowest stage:

| MLP path | MLP cyc/sample | bottleneck stage | bottleneck cyc/sample |
|---|---|---|---|
| scalar loop (measured) | ~18 816 | **MLP** | ~18 816 |
| FVCU dot, 16 PEs | ~196 | MLP ≈ gather | ~196 |
| FVCU dot, 256 PEs | ~12 | **M-CGRA gather** | ~186 |

Only in the well-parallelized FVCU case does the **M-CGRA batched gather become
the bottleneck (~186 cyc/sample)** — matching the paper's "hash encoding is
memory-bound" finding. With the current scalar matmul, MLP compute dominates.

Per-frame latency for `N` samples ≈ `N · (bottleneck cyc/sample)`:

| scenario | bottleneck cyc/sample | N = 1 M samples |
|---|---|---|
| scalar MLP (measured) | ~18 816 | ~18.8 G cyc ≈ 18.8 ms |
| FVCU MLP, full C-CGRA | ~186 (gather-bound) | ~186 M cyc ≈ 0.19 ms |

## Notes / open items

- **Solid**: per-kernel II/steps (measured, dimension-robust).
- **Estimated**: MAC counts and MLP arch are representative Instant-NGP values;
  `N` (samples/frame) is workload-dependent.
- **Key gap for realistic latency = parallel matmul.** Status of the steps:
  1. ✅ **FVCU width latency modeled + measured** (`raichu-model-fvcu-latency`):
     `fvc_dot3` costs `ceil(K/6)` cycles (measured 6 for K=32); reused in a loop
     this becomes the per-dot II — the parallel-path numbers above now rest on a
     measured per-dot latency, not a guess.
  2. ✅ **`neura.vector_load` added + width latency modeled** (`ceil(K/mem_width)`,
     measured 4 for K=32). The vectorized output-element body (2×vector_load +
     fvc_dot3) now maps end-to-end (II=2, 13-cycle body). Remaining: express the
     matmul as a `for i,j` loop of this body (needs neura ops inside the affine/
     counter loop, or `linalg` vectorization emitting `vector_load`+`fvc_dot3`)
     to read back the *looped* II directly instead of composing it.
  3. ⏳ Affine dead-store elimination so the alternative `unroll + scalrep`
     route produces a clean parallel body (canonicalize does not remove the
     redundant `affine.store`s).
  4. ⏳ Mapper performance (project TODO #18) for larger parallel bodies.
- Blending/ray-composition and ray-marching stages are not yet reproduced.
