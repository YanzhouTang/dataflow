// RAICHU M-CGRA memory-orchestration kernel (faithful split): the M-CGRA does
// ONLY the memory-bound work of hash-grid encoding -- compute the 8 corner
// spatial-hash indices and issue ONE batched gather that fetches all 8
// embedding rows. The compute-bound trilinear interpolation is a separate
// C-CGRA kernel (see mlp_dot / fvc_dot3), matching RAICHU's memory/compute
// decoupling (Section 4.2). This keeps the M-CGRA kernel light, as intended.
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
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=greedy max-steps=8" \
// RUN:   --architecture-spec=%S/../arch_spec/raichu_m_cgra.yaml \
// RUN:   -o %t-mapped.mlir
// RUN: FileCheck %s --input-file=%t-mapped.mlir

// CHECK: compiled_ii
// CHECK: neura.gather
func.func @mcgra_hash_gather(%embeddings: memref<256x2xf32>,
                             %idx_buf: memref<8xi64>) -> f32 {
  %x0 = llvm.mlir.constant(0.25 : f32) : f32
  %x1 = llvm.mlir.constant(-0.5 : f32) : f32
  %x2 = llvm.mlir.constant(0.75 : f32) : f32
  %f0_5 = llvm.mlir.constant(0.5 : f32) : f32
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

  // Normalize + scale + floor -> integer grid coordinates (pg0, pg1, pg2).
  %t0 = "neura.fmul"(%x0, %f0_5) : (f32, f32) -> f32
  %nr0 = "neura.fadd"(%t0, %f0_5) : (f32, f32) -> f32
  %t1 = "neura.fmul"(%x1, %f0_5) : (f32, f32) -> f32
  %nr1 = "neura.fadd"(%t1, %f0_5) : (f32, f32) -> f32
  %t2 = "neura.fmul"(%x2, %f0_5) : (f32, f32) -> f32
  %nr2 = "neura.fadd"(%t2, %f0_5) : (f32, f32) -> f32
  %ps00 = "neura.fmul"(%nr0, %scale) : (f32, f32) -> f32
  %ps0 = "neura.fadd"(%ps00, %f0_5) : (f32, f32) -> f32
  %ps10 = "neura.fmul"(%nr1, %scale) : (f32, f32) -> f32
  %ps1 = "neura.fadd"(%ps10, %f0_5) : (f32, f32) -> f32
  %ps20 = "neura.fmul"(%nr2, %scale) : (f32, f32) -> f32
  %ps2 = "neura.fadd"(%ps20, %f0_5) : (f32, f32) -> f32
  %fl0 = "neura.floor"(%ps0) : (f32) -> f32
  %pg0 = "neura.cast"(%fl0) <{cast_type = "fptosi"}> : (f32) -> i64
  %fl1 = "neura.floor"(%ps1) : (f32) -> f32
  %pg1 = "neura.cast"(%fl1) <{cast_type = "fptosi"}> : (f32) -> i64
  %fl2 = "neura.floor"(%ps2) : (f32) -> f32
  %pg2 = "neura.cast"(%fl2) <{cast_type = "fptosi"}> : (f32) -> i64
  %gx1 = "neura.add"(%pg0, %c1) : (i64, i64) -> i64
  %gy1 = "neura.add"(%pg1, %c1) : (i64, i64) -> i64
  %gz1 = "neura.add"(%pg2, %c1) : (i64, i64) -> i64

  // 8 corner spatial hashes -> index buffer.
  %h0a = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %h0b = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %h0c = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h0x = "neura.xor"(%h0a, %h0b) : (i64, i64) -> i64
  %h0h = "neura.xor"(%h0x, %h0c) : (i64, i64) -> i64
  %i0 = "neura.rem"(%h0h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i0 to %idx_buf [%c0 : i64] memref<8xi64> : i64

  %h1a = "neura.mul"(%gx1, %p0) : (i64, i64) -> i64
  %h1b = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %h1c = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h1x = "neura.xor"(%h1a, %h1b) : (i64, i64) -> i64
  %h1h = "neura.xor"(%h1x, %h1c) : (i64, i64) -> i64
  %i1 = "neura.rem"(%h1h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i1 to %idx_buf [%c1 : i64] memref<8xi64> : i64

  %h2a = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %h2b = "neura.mul"(%gy1, %p1) : (i64, i64) -> i64
  %h2c = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h2x = "neura.xor"(%h2a, %h2b) : (i64, i64) -> i64
  %h2h = "neura.xor"(%h2x, %h2c) : (i64, i64) -> i64
  %i2 = "neura.rem"(%h2h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i2 to %idx_buf [%c2 : i64] memref<8xi64> : i64

  %h3a = "neura.mul"(%gx1, %p0) : (i64, i64) -> i64
  %h3b = "neura.mul"(%gy1, %p1) : (i64, i64) -> i64
  %h3c = "neura.mul"(%pg2, %p2) : (i64, i64) -> i64
  %h3x = "neura.xor"(%h3a, %h3b) : (i64, i64) -> i64
  %h3h = "neura.xor"(%h3x, %h3c) : (i64, i64) -> i64
  %i3 = "neura.rem"(%h3h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i3 to %idx_buf [%c3 : i64] memref<8xi64> : i64

  %h4a = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %h4b = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %h4c = "neura.mul"(%gz1, %p2) : (i64, i64) -> i64
  %h4x = "neura.xor"(%h4a, %h4b) : (i64, i64) -> i64
  %h4h = "neura.xor"(%h4x, %h4c) : (i64, i64) -> i64
  %i4 = "neura.rem"(%h4h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i4 to %idx_buf [%c4 : i64] memref<8xi64> : i64

  %h5a = "neura.mul"(%gx1, %p0) : (i64, i64) -> i64
  %h5b = "neura.mul"(%pg1, %p1) : (i64, i64) -> i64
  %h5c = "neura.mul"(%gz1, %p2) : (i64, i64) -> i64
  %h5x = "neura.xor"(%h5a, %h5b) : (i64, i64) -> i64
  %h5h = "neura.xor"(%h5x, %h5c) : (i64, i64) -> i64
  %i5 = "neura.rem"(%h5h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i5 to %idx_buf [%c5 : i64] memref<8xi64> : i64

  %h6a = "neura.mul"(%pg0, %p0) : (i64, i64) -> i64
  %h6b = "neura.mul"(%gy1, %p1) : (i64, i64) -> i64
  %h6c = "neura.mul"(%gz1, %p2) : (i64, i64) -> i64
  %h6x = "neura.xor"(%h6a, %h6b) : (i64, i64) -> i64
  %h6h = "neura.xor"(%h6x, %h6c) : (i64, i64) -> i64
  %i6 = "neura.rem"(%h6h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i6 to %idx_buf [%c6 : i64] memref<8xi64> : i64

  %h7a = "neura.mul"(%gx1, %p0) : (i64, i64) -> i64
  %h7b = "neura.mul"(%gy1, %p1) : (i64, i64) -> i64
  %h7c = "neura.mul"(%gz1, %p2) : (i64, i64) -> i64
  %h7x = "neura.xor"(%h7a, %h7b) : (i64, i64) -> i64
  %h7h = "neura.xor"(%h7x, %h7c) : (i64, i64) -> i64
  %i7 = "neura.rem"(%h7h, %c256) : (i64, i64) -> i64
  neura.store_indexed %i7 to %idx_buf [%c7 : i64] memref<8xi64> : i64

  // One batched gather: 8 embedding rows in a single GA operation.
  %rows = neura.gather %embeddings[%idx_buf] : memref<256x2xf32>, memref<8xi64> -> memref<8x2xf32>

  // Minimal consume so the gather stays live (real interp is on the C-CGRA).
  %r0 = neura.load_indexed %rows [%c0, %c0 : i64, i64] memref<8x2xf32> : f32
  %r7 = neura.load_indexed %rows [%c7, %c1 : i64, i64] memref<8x2xf32> : f32
  %out = "neura.fadd"(%r0, %r7) : (f32, f32) -> f32
  return %out : f32
}
