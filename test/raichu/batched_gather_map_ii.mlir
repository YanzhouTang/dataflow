// Real batched gather on the RAICHU M-CGRA: a single neura.gather fetches 8
// embedding rows in one operation (NOT degenerated into per-corner loads),
// modeling the GA unit's 8-wide random-address fetch. The memref arguments
// (embedding table + index buffer) are promoted to constants before mapping,
// exactly as the e2e compute kernels handle their memref inputs.
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
func.func @batched_gather(%table: memref<256x2xf32>, %idx_buf: memref<8xi64>) -> f32 {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  // One batched gather fetches all 8 corner rows at once.
  %rows = neura.gather %table[%idx_buf] : memref<256x2xf32>, memref<8xi64> -> memref<8x2xf32>
  // Consume a couple of the gathered rows.
  %f00 = neura.load_indexed %rows [%c0, %c0 : i64, i64] memref<8x2xf32> : f32
  %f10 = neura.load_indexed %rows [%c1, %c0 : i64, i64] memref<8x2xf32> : f32
  %s = "neura.fadd"(%f00, %f10) : (f32, f32) -> f32
  return %s : f32
}
