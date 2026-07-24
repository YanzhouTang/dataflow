// Realistic-dimension MLP layer matmul: [16 x 32] @ [32 x 64] -> [16 x 64]
// (batch 16 samples, input_dim 32 = 16 levels x 2, hidden_dim 64), to confirm
// that the loop-body II/steps are dimension-robust (only trip_count scales).
func.func @mlp_layer_real(%a: tensor<16x32xf32>, %w: tensor<32x64xf32>,
                          %acc: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %out = linalg.matmul ins(%a, %w : tensor<16x32xf32>, tensor<32x64xf32>)
                       outs(%acc : tensor<16x64xf32>) -> tensor<16x64xf32>
  return %out : tensor<16x64xf32>
}
