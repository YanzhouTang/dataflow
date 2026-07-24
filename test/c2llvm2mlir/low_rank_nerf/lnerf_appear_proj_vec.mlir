// L-NeRF appearance-matrix projection via the FVCU vector dot (G1 six-element
// MAC, mapping-efficient form). The R=6 rank coefficients and the B-matrix row
// are fetched by two 6-wide vector loads and reduced by the FVCU dot in one op
// (fan-in 2), which is the paper's "FVCU supported by vector loads matching the
// FVCU layout". This is the width-6 instance of the vector_load + fvc_dot3 path
// (cf. test/raichu/vec_matmul_body.mlir at width 32).
func.func @lnerf_appear_proj_vec(%c: memref<32x6xf32>, %b: memref<32x6xf32>) -> f32 {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %cv = neura.vector_load %c [%c0, %c0 : i64, i64] memref<32x6xf32> : vector<6xf32>
  %bv = neura.vector_load %b [%c0, %c0 : i64, i64] memref<32x6xf32> : vector<6xf32>
  %d = "neura.fvc_dot3"(%cv, %bv) : (vector<6xf32>, vector<6xf32>) -> f32
  return %d : f32
}
