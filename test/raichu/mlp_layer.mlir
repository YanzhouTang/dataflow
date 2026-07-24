// A single real MLP layer (matmul) representing one NeRF MLP dense layer.
// The accumulator/output is passed in (pre-initialized) so there is a single
// matmul loop nest (no separate fill loop). Lowered linalg -> affine -> neura
// and mapped onto the RAICHU C-CGRA.
func.func @mlp_layer(%a: tensor<4x8xf32>, %w: tensor<8x8xf32>,
                     %acc: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %out = linalg.matmul ins(%a, %w : tensor<4x8xf32>, tensor<8x8xf32>)
                       outs(%acc : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
