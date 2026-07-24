# MLP-based NeRF (Classic NeRF) — Latency

Classic NeRF represents a radiance field with a deep MLP evaluated at sampled
3D positions and view directions. This document tracks the faithful compiler
reproduction of ray sampling, Fourier positional encoding, MLP inference, and
volume-rendering accumulation. See [README.md](README.md) for the latency model.

## Pipeline

```
ray sampling → positional encoding γ(p) → deep MLP → volume accumulation
```

| Stage | role | CGRA |
|---|---|---|
| Ray sampling | `p = o + t·d` | C-CGRA |
| Positional encoding | `sin(2^kπp), cos(2^kπp)` | C-CGRA |
| MLP | encoded position/direction → density + RGB | C-CGRA/FVCU |
| Accumulation | alpha + transmittance recurrence | C-CGRA |

## Measured II and simulator latency

All values are real mapper outputs. Sources are under
`test/c2llvm2mlir/mlp_nerf/`.

| stage | source | II | steps | latency at T=32 |
|---|---|---:|---:|---:|
| Ray sampling | `mnerf_sample.cpp` | **6** | 9 | 195 |
| Positional encoding, one coordinate, L=4 | `mnerf_posenc.cpp` | **6** | 8 | 194 |
| MLP hidden-unit body, K=8 | `mnerf_mlp.cpp` | **14** | 26 | 460 |
| Volume accumulation | `mnerf_accum.cpp` | **7** | 14 | 231 |

The simulator should use:

```
latency(kernel) = II · (trip_count - 1) + steps
```

The table's `T=32` is only the demonstrator trip count. Real iteration counts:

| stage | trip count driver |
|---|---|
| Sampling | `rays · samples_per_ray` |
| Position encoding | `samples · coordinates(3) · frequency_bands(L)` |
| MLP | `samples · layers · output_units` (dot width = input dimension) |
| Accumulation | `rays · samples_per_ray` |

The MLP is the dominant classic-NeRF stage: an 8×256 network requires many
hidden-unit dot bodies per sample. This is why classic M-NeRF is slower than
factorized/hash variants despite similar per-body II.

## New Neura operations

Faithful Fourier encoding required real trigonometric operations:

- `neura.fsin` and `neura.fcos`;
- operation kinds `IFSin` / `IFCos` on the transcendental (`fdiv`) FU;
- mapping classification;
- both libm-call (`@sinf/@cosf`) and LLVM intrinsic
  (`llvm.intr.sin/cos`) lowering.

The L=4 kernel emits four `fsin` and four `fcos` operations and maps at II=6.

## Functional coverage

- Ray sampling: complete `o+t·d` for x/y/z.
- Fourier positional encoding: real sin/cos, four demonstrator bands. Production
  classic NeRF commonly uses L=10 for position and L=4 for direction; the same
  II/steps model scales via `trip_count`.
- MLP: dot+bias+ReLU body. Production depth/width scale through layer and unit
  counts; vectorized FVCU dot support is documented in
  [hash_nerf.md](hash_nerf.md).
- Volume rendering: real `exp`, alpha, weighted color accumulation, and
  transmittance `T *= (1-alpha)` recurrence.

## Remaining approximation

The demonstrator MLP dot width is K=8 rather than the production encoded width
and hidden width. The mapper schedules one loop body, so larger dimensions
primarily scale the number and width of FVCU dot operations; simulator inputs
must use the production layer dimensions.
