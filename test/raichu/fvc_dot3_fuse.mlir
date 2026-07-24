// RUN: mlir-neura-opt %s --raichu-fuse-fvcu | FileCheck %s

// A vector dot product (elementwise multiply + horizontal reduce) should fuse
// into a single neura.fvc_dot3 FVCU primitive.
// CHECK-LABEL: func.func @dot3
// CHECK: neura.fvc_dot3
// CHECK-NOT: neura.vfmul
// CHECK-NOT: neura.vector.reduce.add
func.func @dot3(%a: vector<3xf32>, %b: vector<3xf32>) -> f32 {
  %p = "neura.vfmul"(%a, %b) : (vector<3xf32>, vector<3xf32>) -> vector<3xf32>
  %d = "neura.vector.reduce.add"(%p) : (vector<3xf32>) -> f32
  return %d : f32
}
