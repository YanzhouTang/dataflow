// End-to-end NeRF hash-grid encoding (single point, single level, 8 corners)
// on the RAICHU M-CGRA, combining the real spatial-hash index arithmetic with
// one real batched gather (8 embedding rows fetched in a single GA operation)
// and trilinear-weighted accumulation. This is the mappable straight-line form
// of nerf_kernels.py :: hash_encode/trilinear_interpolation for one level.
//
// Pipeline notes:
//   - memref inputs (embedding table + index buffer) are promoted to constants,
//     exactly as the e2e compute kernels handle their memref inputs.
//   - the batched gather latency is modeled from the GA unit width.
//
// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --raichu-model-ga-latency \
// RUN:   --promote-input-arg-to-const \
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
func.func @hash_encode_e2e(%embeddings: memref<256x2xf32>,
                           %idx_buf: memref<8xi64>) -> f32 {
  // ---- Constants ----
  %x0 = llvm.mlir.constant(0.25 : f32) : f32
  %x1 = llvm.mlir.constant(-0.5 : f32) : f32
  %x2 = llvm.mlir.constant(0.75 : f32) : f32
  %f0_5 = llvm.mlir.constant(0.5 : f32) : f32
  %f0 = llvm.mlir.constant(0.0 : f32) : f32
  %f1 = llvm.mlir.constant(1.0 : f32) : f32
  %scale = llvm.mlir.constant(15.0 : f32) : f32
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %c5 = llvm.mlir.constant(5 : i64) : i64
  %c6 = llvm.mlir.constant(6 : i64) : i64
  %c7 = llvm.mlir.constant(7 : i64) : i64
  %c256 = llvm.mlir.constant(256 : i64) : i64
  %p0 = llvm.mlir.constant(1 : i64) : i64
  %p1 = llvm.mlir.constant(2654435761 : i64) : i64
  %p2 = llvm.mlir.constant(805459861 : i64) : i64

  // ---- Normalize to [0,1]: norm = x*0.5 + 0.5 ----
  %t0 = "neura.fmul"(%x0, %f0_5) : (f32, f32) -> f32
  %nr0 = "neura.fadd"(%t0, %f0_5) : (f32, f32) -> f32
  %t1 = "neura.fmul"(%x1, %f0_5) : (f32, f32) -> f32
  %nr1 = "neura.fadd"(%t1, %f0_5) : (f32, f32) -> f32
  %t2 = "neura.fmul"(%x2, %f0_5) : (f32, f32) -> f32
  %nr2 = "neura.fadd"(%t2, %f0_5) : (f32, f32) -> f32

  // ---- pos_scaled = norm * scale + 0.5 ----
  %ps00 = "neura.fmul"(%nr0, %scale) : (f32, f32) -> f32
  %ps0 = "neura.fadd"(%ps00, %f0_5) : (f32, f32) -> f32
  %ps10 = "neura.fmul"(%nr1, %scale) : (f32, f32) -> f32
  %ps1 = "neura.fadd"(%ps10, %f0_5) : (f32, f32) -> f32
  %ps20 = "neura.fmul"(%nr2, %scale) : (f32, f32) -> f32
  %ps2 = "neura.fadd"(%ps20, %f0_5) : (f32, f32) -> f32

  // ---- pos_grid = floor(pos_scaled) as i64 ; frac = clamp(ps - grid, 0, 1) ----
  %fl0 = "neura.floor"(%ps0) : (f32) -> f32
  %pg0 = "neura.cast"(%fl0) <{cast_type = "fptosi"}> : (f32) -> i64
  %fl1 = "neura.floor"(%ps1) : (f32) -> f32
  %pg1 = "neura.cast"(%fl1) <{cast_type = "fptosi"}> : (f32) -> i64
  %fl2 = "neura.floor"(%ps2) : (f32) -> f32
  %pg2 = "neura.cast"(%fl2) <{cast_type = "fptosi"}> : (f32) -> i64

  %fpg0 = "neura.cast"(%pg0) <{cast_type = "sitofp"}> : (i64) -> f32
  %dp0 = "neura.fsub"(%ps0, %fpg0) : (f32, f32) -> f32
  %dp0c = neura.fmax<"maxnum">(%dp0, %f0 : f32) : f32 -> f32
  %pos0 = neura.fmin<"minnum">(%dp0c, %f1 : f32) : f32 -> f32
  %omp0 = "neura.fsub"(%f1, %pos0) : (f32, f32) -> f32

  %fpg1 = "neura.cast"(%pg1) <{cast_type = "sitofp"}> : (i64) -> f32
  %dp1 = "neura.fsub"(%ps1, %fpg1) : (f32, f32) -> f32
  %dp1c = neura.fmax<"maxnum">(%dp1, %f0 : f32) : f32 -> f32
  %pos1 = neura.fmin<"minnum">(%dp1c, %f1 : f32) : f32 -> f32
  %omp1 = "neura.fsub"(%f1, %pos1) : (f32, f32) -> f32

  %fpg2 = "neura.cast"(%pg2) <{cast_type = "sitofp"}> : (i64) -> f32
  %dp2 = "neura.fsub"(%ps2, %fpg2) : (f32, f32) -> f32
  %dp2c = neura.fmax<"maxnum">(%dp2, %f0 : f32) : f32 -> f32
  %pos2 = neura.fmin<"minnum">(%dp2c, %f1 : f32) : f32 -> f32
  %omp2 = "neura.fsub"(%f1, %pos2) : (f32, f32) -> f32

  // ---- Corner grid coordinates (grid and grid+1 per axis) ----
  %gx1 = "neura.add"(%pg0, %c1) : (i64, i64) -> i64
  %gy1 = "neura.add"(%pg1, %c1) : (i64, i64) -> i64
  %gz1 = "neura.add"(%pg2, %c1) : (i64, i64) -> i64

  // ================= Corner 0: (0,0,0) =================
  %h0a = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %h0b = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %h0c = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h0x = "neura.xor"(%h0a, %h0b) : (i64, i64) -> i64
  %h0h = "neura.xor"(%h0x, %h0c) : (i64, i64) -> i64
  %i0 = "neura.rem"(%h0h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i0 to %idx_buf [%c0 : i64] memref<8xi64> : i64

  // ================= Corner 1: (1,0,0) =================
  %h1a = "neura.mul"(%gx1, %p0) : (i64, i64) -> i64
  %h1b = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %h1c = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h1x = "neura.xor"(%h1a, %h1b) : (i64, i64) -> i64
  %h1h = "neura.xor"(%h1x, %h1c) : (i64, i64) -> i64
  %i1 = "neura.rem"(%h1h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i1 to %idx_buf [%c1 : i64] memref<8xi64> : i64

  // ================= Corner 2: (0,1,0) =================
  %h2a = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %h2b = "neura.mul"(%gy1, %p1) : (i64, i64) -> i64
  %h2c = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h2x = "neura.xor"(%h2a, %h2b) : (i64, i64) -> i64
  %h2h = "neura.xor"(%h2x, %h2c) : (i64, i64) -> i64
  %i2 = "neura.rem"(%h2h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i2 to %idx_buf [%c2 : i64] memref<8xi64> : i64

  // ================= Corner 3: (1,1,0) =================
  %h3a = "neura.mul"(%gx1, %p0) : (i64, i64) -> i64
  %h3b = "neura.mul"(%gy1, %p1) : (i64, i64) -> i64
  %h3c = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h3x = "neura.xor"(%h3a, %h3b) : (i64, i64) -> i64
  %h3h = "neura.xor"(%h3x, %h3c) : (i64, i64) -> i64
  %i3 = "neura.rem"(%h3h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i3 to %idx_buf [%c3 : i64] memref<8xi64> : i64

  // ================= Corner 4: (0,0,1) =================
  %h4a = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %h4b = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %h4c = "neura.mul"(%gz1, %p2) : (i64, i64) -> i64
  %h4x = "neura.xor"(%h4a, %h4b) : (i64, i64) -> i64
  %h4h = "neura.xor"(%h4x, %h4c) : (i64, i64) -> i64
  %i4 = "neura.rem"(%h4h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i4 to %idx_buf [%c4 : i64] memref<8xi64> : i64

  // ================= Corner 5: (1,0,1) =================
  %h5a = "neura.mul"(%gx1, %p0) : (i64, i64) -> i64
  %h5b = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %h5c = "neura.mul"(%gz1, %p2) : (i64, i64) -> i64
  %h5x = "neura.xor"(%h5a, %h5b) : (i64, i64) -> i64
  %h5h = "neura.xor"(%h5x, %h5c) : (i64, i64) -> i64
  %i5 = "neura.rem"(%h5h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i5 to %idx_buf [%c5 : i64] memref<8xi64> : i64

  // ================= Corner 6: (0,1,1) =================
  %h6a = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %h6b = "neura.mul"(%gy1, %p1) : (i64, i64) -> i64
  %h6c = "neura.mul"(%gz1, %p2) : (i64, i64) -> i64
  %h6x = "neura.xor"(%h6a, %h6b) : (i64, i64) -> i64
  %h6h = "neura.xor"(%h6x, %h6c) : (i64, i64) -> i64
  %i6 = "neura.rem"(%h6h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i6 to %idx_buf [%c6 : i64] memref<8xi64> : i64

  // ================= Corner 7: (1,1,1) =================
  %h7a = "neura.mul"(%gx1, %p0) : (i64, i64) -> i64
  %h7b = "neura.mul"(%gy1, %p1) : (i64, i64) -> i64
  %h7c = "neura.mul"(%gz1, %p2) : (i64, i64) -> i64
  %h7x = "neura.xor"(%h7a, %h7b) : (i64, i64) -> i64
  %h7h = "neura.xor"(%h7x, %h7c) : (i64, i64) -> i64
  %i7 = "neura.rem"(%h7h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i7 to %idx_buf [%c7 : i64] memref<8xi64> : i64

  // ================= One batched gather: 8 rows in a single GA op =========
  %rows = neura.gather %embeddings[%idx_buf] : memref<256x2xf32>, memref<8xi64> -> memref<8x2xf32>

  // ================= Trilinear-weighted accumulation over 8 corners ========
  // Corner 0: w = omp0*omp1*omp2
  %w0a = "neura.fmul"(%omp0, %omp1) : (f32, f32) -> f32
  %w0 = "neura.fmul"(%w0a, %omp2) : (f32, f32) -> f32
  %f0v = neura.load_indexed %rows [%c0, %c0 : i64, i64] memref<8x2xf32> : f32
  %a0 = "neura.fmul"(%w0, %f0v) : (f32, f32) -> f32

  // Corner 1: w = pos0*omp1*omp2
  %w1a = "neura.fmul"(%pos0, %omp1) : (f32, f32) -> f32
  %w1 = "neura.fmul"(%w1a, %omp2) : (f32, f32) -> f32
  %f1v = neura.load_indexed %rows [%c1, %c0 : i64, i64] memref<8x2xf32> : f32
  %a1p = "neura.fmul"(%w1, %f1v) : (f32, f32) -> f32
  %a1 = "neura.fadd"(%a0, %a1p) : (f32, f32) -> f32

  // Corner 2: w = omp0*pos1*omp2
  %w2a = "neura.fmul"(%omp0, %pos1) : (f32, f32) -> f32
  %w2 = "neura.fmul"(%w2a, %omp2) : (f32, f32) -> f32
  %f2v = neura.load_indexed %rows [%c2, %c0 : i64, i64] memref<8x2xf32> : f32
  %a2p = "neura.fmul"(%w2, %f2v) : (f32, f32) -> f32
  %a2 = "neura.fadd"(%a1, %a2p) : (f32, f32) -> f32

  // Corner 3: w = pos0*pos1*omp2
  %w3a = "neura.fmul"(%pos0, %pos1) : (f32, f32) -> f32
  %w3 = "neura.fmul"(%w3a, %omp2) : (f32, f32) -> f32
  %f3v = neura.load_indexed %rows [%c3, %c0 : i64, i64] memref<8x2xf32> : f32
  %a3p = "neura.fmul"(%w3, %f3v) : (f32, f32) -> f32
  %a3 = "neura.fadd"(%a2, %a3p) : (f32, f32) -> f32

  // Corner 4: w = omp0*omp1*pos2
  %w4a = "neura.fmul"(%omp0, %omp1) : (f32, f32) -> f32
  %w4 = "neura.fmul"(%w4a, %pos2) : (f32, f32) -> f32
  %f4v = neura.load_indexed %rows [%c4, %c0 : i64, i64] memref<8x2xf32> : f32
  %a4p = "neura.fmul"(%w4, %f4v) : (f32, f32) -> f32
  %a4 = "neura.fadd"(%a3, %a4p) : (f32, f32) -> f32

  // Corner 5: w = pos0*omp1*pos2
  %w5a = "neura.fmul"(%pos0, %omp1) : (f32, f32) -> f32
  %w5 = "neura.fmul"(%w5a, %pos2) : (f32, f32) -> f32
  %f5v = neura.load_indexed %rows [%c5, %c0 : i64, i64] memref<8x2xf32> : f32
  %a5p = "neura.fmul"(%w5, %f5v) : (f32, f32) -> f32
  %a5 = "neura.fadd"(%a4, %a5p) : (f32, f32) -> f32

  // Corner 6: w = omp0*pos1*pos2
  %w6a = "neura.fmul"(%omp0, %pos1) : (f32, f32) -> f32
  %w6 = "neura.fmul"(%w6a, %pos2) : (f32, f32) -> f32
  %f6v = neura.load_indexed %rows [%c6, %c0 : i64, i64] memref<8x2xf32> : f32
  %a6p = "neura.fmul"(%w6, %f6v) : (f32, f32) -> f32
  %a6 = "neura.fadd"(%a5, %a6p) : (f32, f32) -> f32

  // Corner 7: w = pos0*pos1*pos2
  %w7a = "neura.fmul"(%pos0, %pos1) : (f32, f32) -> f32
  %w7 = "neura.fmul"(%w7a, %pos2) : (f32, f32) -> f32
  %f7v = neura.load_indexed %rows [%c7, %c0 : i64, i64] memref<8x2xf32> : f32
  %a7p = "neura.fmul"(%w7, %f7v) : (f32, f32) -> f32
  %acc = "neura.fadd"(%a6, %a7p) : (f32, f32) -> f32

  return %acc : f32
}
