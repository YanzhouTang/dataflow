// Realistic MLP inner-product width (K=32 = 16 levels x 2 features) done the
// RAICHU way: a vector multiply + horizontal reduce fuses to neura.fvc_dot3,
// and the FVCU lane-width latency model (raichu-model-fvcu-latency) charges it
// ceil(32/6)=6 cycles on a 6-lane FVCU. This measures the realistic per-output
// dot cost of an MLP layer on the C-CGRA.
//
// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --raichu-fuse-fvcu \
// RUN:   --raichu-model-fvcu-latency \
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
// CHECK: neura.fvc_dot3
func.func @mlp_dot32(%a: vector<32xf32>, %b: vector<32xf32>) -> f32 {
  %p = "neura.vfmul"(%a, %b) : (vector<32xf32>, vector<32xf32>) -> vector<32xf32>
  %d = "neura.vector.reduce.add"(%p) : (vector<32xf32>) -> f32
  return %d : f32
}
