// Vectorized MLP matmul output-element body (RAICHU C-CGRA style):
//   av = vector_load a[i, :]   (K=32 contiguous)
//   wv = vector_load w[:, j]
//   d  = fvc_dot3(av, wv)      (K-length dot on the FVCU)
//   acc[i,j] = d
// vector_load latency = ceil(K/mem_width), fvc_dot3 latency = ceil(K/fvcu_width).
// This measures the realistic per-output cost of a vectorized MLP layer.
//
// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --raichu-model-fvcu-latency \
// RUN:   --promote-input-arg-to-const \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=greedy" \
// RUN:   --architecture-spec=%S/../arch_spec/raichu_c_cgra.yaml \
// RUN:   -o %t-mapped.mlir
// RUN: FileCheck %s --input-file=%t-mapped.mlir

// CHECK: compiled_ii
// CHECK: neura.vector_load
// CHECK: neura.fvc_dot3
func.func @vec_matmul_body(%a: memref<16x32xf32>, %w: memref<32x64xf32>) -> f32 {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %av = neura.vector_load %a [%c0, %c0 : i64, i64] memref<16x32xf32> : vector<32xf32>
  %wv = neura.vector_load %w [%c0, %c0 : i64, i64] memref<32x64xf32> : vector<32xf32>
  %d = "neura.fvc_dot3"(%av, %wv) : (vector<32xf32>, vector<32xf32>) -> f32
  return %d : f32
}
