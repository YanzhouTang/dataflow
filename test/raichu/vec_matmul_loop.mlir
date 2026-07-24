// Vectorized MLP matmul as a for-i,j loop whose body is vector_load + fvc_dot3
// (the RAICHU C-CGRA form). Mapping this reads the *looped* II directly.
//   for i in 0..16: for j in 0..64:
//     acc[i,j] = fvc_dot3( vector_load a[i,:], vector_load w[j,:] )
func.func @vec_matmul_loop(%a: memref<16x32xf32>, %w: memref<64x32xf32>,
                           %acc: memref<16x64xf32>) -> memref<16x64xf32> {
  %c0 = arith.constant 0 : index
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 64 {
      %av = neura.vector_load %a [%i, %c0 : index, index] memref<16x32xf32> : vector<32xf32>
      %wv = neura.vector_load %w [%j, %c0 : index, index] memref<64x32xf32> : vector<32xf32>
      %d = "neura.fvc_dot3"(%av, %wv) : (vector<32xf32>, vector<32xf32>) -> f32
      neura.store_indexed %d to %acc [%i, %j : index, index] memref<16x64xf32> : f32
    }
  }
  return %acc : memref<16x64xf32>
}
