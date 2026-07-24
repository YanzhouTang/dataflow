// A complete NeRF MLP dense layer: matmul + bias add + ReLU activation.
//   out = relu(A @ W + bias)
// The accumulator is passed in (pre-zeroed by the caller) so the matmul is a
// single loop nest. ReLU is expressed as cmp+select so it lowers to
// neura.fcmp + neura.sel. Lowered linalg -> affine -> neura and mapped onto the
// RAICHU C-CGRA.
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#bcast = affine_map<(d0, d1) -> (d1)>
func.func @mlp_full_layer(%a: tensor<4x8xf32>, %w: tensor<8x8xf32>,
                          %bias: tensor<8xf32>,
                          %acc: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %zero = arith.constant 0.0 : f32
  // Linear part: A @ W (accumulating into acc).
  %mm = linalg.matmul ins(%a, %w : tensor<4x8xf32>, tensor<8x8xf32>)
                      outs(%acc : tensor<4x8xf32>) -> tensor<4x8xf32>
  // Bias add + ReLU (elementwise).
  %init = tensor.empty() : tensor<4x8xf32>
  %out = linalg.generic {
      indexing_maps = [#id2, #bcast, #id2],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm, %bias : tensor<4x8xf32>, tensor<8xf32>)
      outs(%init : tensor<4x8xf32>) {
  ^bb0(%x: f32, %b: f32, %o: f32):
    %s = arith.addf %x, %b : f32
    %r = arith.maximumf %s, %zero : f32
    linalg.yield %r : f32
  } -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
