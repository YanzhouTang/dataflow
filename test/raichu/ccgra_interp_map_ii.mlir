// RAICHU C-CGRA compute kernel (faithful split): the trilinear interpolation
// half of hash-grid encoding. Given the 8 gathered corner feature rows (2
// channels each, produced by the M-CGRA batched gather) and the fractional
// coordinates, it computes the 8 trilinear weights and the weighted sum over
// both channels. This is the compute-bound counterpart of the M-CGRA
// hash+gather kernel, matching RAICHU's memory/compute decoupling (Section 4.2).
//
// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-input-arg-to-const \
// RUN:   --fuse-pattern \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=%S/../arch_spec/raichu_c_cgra.yaml \
// RUN:   -o %t-mapped.mlir
// RUN: FileCheck %s --input-file=%t-mapped.mlir

// CHECK: compiled_ii
func.func @ccgra_interp(%rows: memref<8x2xf32>) -> f32 {
  %pos0 = llvm.mlir.constant(0.3 : f32) : f32
  %pos1 = llvm.mlir.constant(0.6 : f32) : f32
  %pos2 = llvm.mlir.constant(0.9 : f32) : f32
  %f1 = llvm.mlir.constant(1.0 : f32) : f32
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(2 : i64) : i64
  %c3 = llvm.mlir.constant(3 : i64) : i64
  %c4 = llvm.mlir.constant(4 : i64) : i64
  %c5 = llvm.mlir.constant(5 : i64) : i64
  %c6 = llvm.mlir.constant(6 : i64) : i64
  %c7 = llvm.mlir.constant(7 : i64) : i64

  %omp0 = "neura.fsub"(%f1, %pos0) : (f32, f32) -> f32
  %omp1 = "neura.fsub"(%f1, %pos1) : (f32, f32) -> f32
  %omp2 = "neura.fsub"(%f1, %pos2) : (f32, f32) -> f32

  // Corner 0 (0,0,0): w = omp0*omp1*omp2
  %w0a = "neura.fmul"(%omp0, %omp1) : (f32, f32) -> f32
  %w0 = "neura.fmul"(%w0a, %omp2) : (f32, f32) -> f32
  %f0_0 = neura.load_indexed %rows [%c0, %c0 : i64, i64] memref<8x2xf32> : f32
  %f0_1 = neura.load_indexed %rows [%c0, %c1 : i64, i64] memref<8x2xf32> : f32
  %a0_0 = "neura.fmul"(%w0, %f0_0) : (f32, f32) -> f32
  %a1_0 = "neura.fmul"(%w0, %f0_1) : (f32, f32) -> f32

  // Corner 1 (1,0,0): w = pos0*omp1*omp2
  %w1a = "neura.fmul"(%pos0, %omp1) : (f32, f32) -> f32
  %w1 = "neura.fmul"(%w1a, %omp2) : (f32, f32) -> f32
  %f1_0 = neura.load_indexed %rows [%c1, %c0 : i64, i64] memref<8x2xf32> : f32
  %f1_1 = neura.load_indexed %rows [%c1, %c1 : i64, i64] memref<8x2xf32> : f32
  %m1_0 = "neura.fmul"(%w1, %f1_0) : (f32, f32) -> f32
  %a0_1 = "neura.fadd"(%a0_0, %m1_0) : (f32, f32) -> f32
  %m1_1 = "neura.fmul"(%w1, %f1_1) : (f32, f32) -> f32
  %a1_1 = "neura.fadd"(%a1_0, %m1_1) : (f32, f32) -> f32

  // Corner 2 (0,1,0): w = omp0*pos1*omp2
  %w2a = "neura.fmul"(%omp0, %pos1) : (f32, f32) -> f32
  %w2 = "neura.fmul"(%w2a, %omp2) : (f32, f32) -> f32
  %f2_0 = neura.load_indexed %rows [%c2, %c0 : i64, i64] memref<8x2xf32> : f32
  %f2_1 = neura.load_indexed %rows [%c2, %c1 : i64, i64] memref<8x2xf32> : f32
  %m2_0 = "neura.fmul"(%w2, %f2_0) : (f32, f32) -> f32
  %a0_2 = "neura.fadd"(%a0_1, %m2_0) : (f32, f32) -> f32
  %m2_1 = "neura.fmul"(%w2, %f2_1) : (f32, f32) -> f32
  %a1_2 = "neura.fadd"(%a1_1, %m2_1) : (f32, f32) -> f32

  // Corner 3 (1,1,0): w = pos0*pos1*omp2
  %w3a = "neura.fmul"(%pos0, %pos1) : (f32, f32) -> f32
  %w3 = "neura.fmul"(%w3a, %omp2) : (f32, f32) -> f32
  %f3_0 = neura.load_indexed %rows [%c3, %c0 : i64, i64] memref<8x2xf32> : f32
  %f3_1 = neura.load_indexed %rows [%c3, %c1 : i64, i64] memref<8x2xf32> : f32
  %m3_0 = "neura.fmul"(%w3, %f3_0) : (f32, f32) -> f32
  %a0_3 = "neura.fadd"(%a0_2, %m3_0) : (f32, f32) -> f32
  %m3_1 = "neura.fmul"(%w3, %f3_1) : (f32, f32) -> f32
  %a1_3 = "neura.fadd"(%a1_2, %m3_1) : (f32, f32) -> f32

  // Corner 4 (0,0,1): w = omp0*omp1*pos2
  %w4a = "neura.fmul"(%omp0, %omp1) : (f32, f32) -> f32
  %w4 = "neura.fmul"(%w4a, %pos2) : (f32, f32) -> f32
  %f4_0 = neura.load_indexed %rows [%c4, %c0 : i64, i64] memref<8x2xf32> : f32
  %f4_1 = neura.load_indexed %rows [%c4, %c1 : i64, i64] memref<8x2xf32> : f32
  %m4_0 = "neura.fmul"(%w4, %f4_0) : (f32, f32) -> f32
  %a0_4 = "neura.fadd"(%a0_3, %m4_0) : (f32, f32) -> f32
  %m4_1 = "neura.fmul"(%w4, %f4_1) : (f32, f32) -> f32
  %a1_4 = "neura.fadd"(%a1_3, %m4_1) : (f32, f32) -> f32

  // Corner 5 (1,0,1): w = pos0*omp1*pos2
  %w5a = "neura.fmul"(%pos0, %omp1) : (f32, f32) -> f32
  %w5 = "neura.fmul"(%w5a, %pos2) : (f32, f32) -> f32
  %f5_0 = neura.load_indexed %rows [%c5, %c0 : i64, i64] memref<8x2xf32> : f32
  %f5_1 = neura.load_indexed %rows [%c5, %c1 : i64, i64] memref<8x2xf32> : f32
  %m5_0 = "neura.fmul"(%w5, %f5_0) : (f32, f32) -> f32
  %a0_5 = "neura.fadd"(%a0_4, %m5_0) : (f32, f32) -> f32
  %m5_1 = "neura.fmul"(%w5, %f5_1) : (f32, f32) -> f32
  %a1_5 = "neura.fadd"(%a1_4, %m5_1) : (f32, f32) -> f32

  // Corner 6 (0,1,1): w = omp0*pos1*pos2
  %w6a = "neura.fmul"(%omp0, %pos1) : (f32, f32) -> f32
  %w6 = "neura.fmul"(%w6a, %pos2) : (f32, f32) -> f32
  %f6_0 = neura.load_indexed %rows [%c6, %c0 : i64, i64] memref<8x2xf32> : f32
  %f6_1 = neura.load_indexed %rows [%c6, %c1 : i64, i64] memref<8x2xf32> : f32
  %m6_0 = "neura.fmul"(%w6, %f6_0) : (f32, f32) -> f32
  %a0_6 = "neura.fadd"(%a0_5, %m6_0) : (f32, f32) -> f32
  %m6_1 = "neura.fmul"(%w6, %f6_1) : (f32, f32) -> f32
  %a1_6 = "neura.fadd"(%a1_5, %m6_1) : (f32, f32) -> f32

  // Corner 7 (1,1,1): w = pos0*pos1*pos2
  %w7a = "neura.fmul"(%pos0, %pos1) : (f32, f32) -> f32
  %w7 = "neura.fmul"(%w7a, %pos2) : (f32, f32) -> f32
  %f7_0 = neura.load_indexed %rows [%c7, %c0 : i64, i64] memref<8x2xf32> : f32
  %f7_1 = neura.load_indexed %rows [%c7, %c1 : i64, i64] memref<8x2xf32> : f32
  %m7_0 = "neura.fmul"(%w7, %f7_0) : (f32, f32) -> f32
  %a0_7 = "neura.fadd"(%a0_6, %m7_0) : (f32, f32) -> f32
  %m7_1 = "neura.fmul"(%w7, %f7_1) : (f32, f32) -> f32
  %a1_7 = "neura.fadd"(%a1_6, %m7_1) : (f32, f32) -> f32

  // Output = channel0 + channel1 (both encoded features).
  %out = "neura.fadd"(%a0_7, %a1_7) : (f32, f32) -> f32
  return %out : f32
}
