// Real NeRF hash-grid encoding body (single point, two corners unrolled) mapped
// onto the RAICHU M-CGRA to obtain a real II. This is the straight-line,
// SSA-dataflow form of nerf_kernels.py :: hash_encode/trilinear_interpolation:
// normalize -> pos_scaled -> floor grid coord -> fractional clamp -> per-corner
// spatial hash (mul primes / xor / lshr / and / rem) -> gather embedding row ->
// trilinear-weighted accumulation.
//
// The imperative memref+loop kernel (hash_encode_neura.mlir) cannot pass the
// ctrl-to-dataflow transform; this captures the same real arithmetic in the
// mappable dataflow form the mapper expects.
//
// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../arch_spec/raichu_m_cgra.yaml \
// RUN:   -o %t-mapped.mlir
// RUN: FileCheck %s --input-file=%t-mapped.mlir

// CHECK: compiled_ii
// CHECK: neura.gather
func.func @hash_encode_body() -> f32 {
  // Input coordinate (one 3D point), scale and constants.
  %x0 = llvm.mlir.constant(0.25 : f32) : f32
  %x1 = llvm.mlir.constant(-0.5 : f32) : f32
  %x2 = llvm.mlir.constant(0.75 : f32) : f32
  %f0_5 = llvm.mlir.constant(0.5 : f32) : f32
  %f0 = llvm.mlir.constant(0.0 : f32) : f32
  %f1 = llvm.mlir.constant(1.0 : f32) : f32
  %scale = llvm.mlir.constant(15.0 : f32) : f32
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c256 = llvm.mlir.constant(256 : i64) : i64
  %p0 = llvm.mlir.constant(1 : i64) : i64
  %p1 = llvm.mlir.constant(2654435761 : i64) : i64
  %p2 = llvm.mlir.constant(805459861 : i64) : i64
  %table = llvm.mlir.constant(0 : i64) : i64

  // Normalize to [0,1]: norm = x*0.5 + 0.5
  %t0 = "neura.fmul"(%x0, %f0_5) : (f32, f32) -> f32
  %nr0 = "neura.fadd"(%t0, %f0_5) : (f32, f32) -> f32
  %t1 = "neura.fmul"(%x1, %f0_5) : (f32, f32) -> f32
  %nr1 = "neura.fadd"(%t1, %f0_5) : (f32, f32) -> f32
  %t2 = "neura.fmul"(%x2, %f0_5) : (f32, f32) -> f32
  %nr2 = "neura.fadd"(%t2, %f0_5) : (f32, f32) -> f32

  // pos_scaled = norm * scale + 0.5
  %ps00 = "neura.fmul"(%nr0, %scale) : (f32, f32) -> f32
  %ps0 = "neura.fadd"(%ps00, %f0_5) : (f32, f32) -> f32
  %ps10 = "neura.fmul"(%nr1, %scale) : (f32, f32) -> f32
  %ps1 = "neura.fadd"(%ps10, %f0_5) : (f32, f32) -> f32
  %ps20 = "neura.fmul"(%nr2, %scale) : (f32, f32) -> f32
  %ps2 = "neura.fadd"(%ps20, %f0_5) : (f32, f32) -> f32

  // pos_grid = floor(pos_scaled) as i64
  %fl0 = "neura.floor"(%ps0) : (f32) -> f32
  %pg0 = "neura.cast"(%fl0) <{cast_type = "fptosi"}> : (f32) -> i64
  %fl1 = "neura.floor"(%ps1) : (f32) -> f32
  %pg1 = "neura.cast"(%fl1) <{cast_type = "fptosi"}> : (f32) -> i64
  %fl2 = "neura.floor"(%ps2) : (f32) -> f32
  %pg2 = "neura.cast"(%fl2) <{cast_type = "fptosi"}> : (f32) -> i64

  // frac = clamp(pos_scaled - floor, 0, 1)
  %fpg0 = "neura.cast"(%pg0) <{cast_type = "sitofp"}> : (i64) -> f32
  %dp0 = "neura.fsub"(%ps0, %fpg0) : (f32, f32) -> f32
  %dp0c = neura.fmax<"maxnum">(%dp0, %f0 : f32) : f32 -> f32
  %pos0 = neura.fmin<"minnum">(%dp0c, %f1 : f32) : f32 -> f32
  %omp0 = "neura.fsub"(%f1, %pos0) : (f32, f32) -> f32

  // ---- Corner 0 (offset 0,0,0): hash -> gather -> weight ----
  %m0_0 = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %m0_1 = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %x0_01 = "neura.xor"(%m0_0, %m0_1) : (i64, i64) -> i64
  %m0_2 = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h0 = "neura.xor"(%x0_01, %m0_2) : (i64, i64) -> i64
  %idx0 = "neura.rem"(%h0, %c256) : (i64, i64) -> i64
  %feat0 = neura.gather %table[%idx0] : i64, i64 -> f32
  // weight = (1-pos0) for corner with d0=0
  %wf0 = "neura.fmul"(%omp0, %feat0) : (f32, f32) -> f32

  // ---- Corner 1 (offset 1,0,0): grid+1 on x, hash -> gather -> weight ----
  %cx1 = "neura.add"(%pg0, %c1) : (i64, i64) -> i64
  %m1_0 = "neura.mul"(%cx1, %p0) : (i64, i64) -> i64
  %m1_1 = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %x1_01 = "neura.xor"(%m1_0, %m1_1) : (i64, i64) -> i64
  %m1_2 = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h1 = "neura.xor"(%x1_01, %m1_2) : (i64, i64) -> i64
  %idx1 = "neura.rem"(%h1, %c256) : (i64, i64) -> i64
  %feat1 = neura.gather %table[%idx1] : i64, i64 -> f32
  // weight = pos0 for corner with d0=1
  %wf1 = "neura.fmul"(%pos0, %feat1) : (f32, f32) -> f32

  // Accumulate the two corner contributions.
  %acc = "neura.fadd"(%wf0, %wf1) : (f32, f32) -> f32
  return %acc : f32
}
