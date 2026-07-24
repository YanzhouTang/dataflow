// C-CGRA mapping smoke test: an MLP-style vector dot product fuses into an FVCU
// primitive and maps onto the RAICHU C-CGRA, yielding a compiled_ii.
//
// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --raichu-fuse-fvcu \
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
// CHECK: neura.fvc_dot3
func.func @mlp_dot() -> f32 {
  %a = llvm.mlir.constant(dense<[1.0, 2.0, 3.0]> : vector<3xf32>) : vector<3xf32>
  %b = llvm.mlir.constant(dense<[4.0, 5.0, 6.0]> : vector<3xf32>) : vector<3xf32>
  %p = "neura.vfmul"(%a, %b) : (vector<3xf32>, vector<3xf32>) -> vector<3xf32>
  %d = "neura.vector.reduce.add"(%p) : (vector<3xf32>) -> f32
  return %d : f32
}
