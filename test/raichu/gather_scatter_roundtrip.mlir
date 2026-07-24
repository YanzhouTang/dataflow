// Round-trip parse/print check for the gather/scatter primitives.
func.func @gs(%table: tensor<16x4xf32>, %idx: tensor<8xi64>,
              %vals: tensor<8x4xf32>) -> tensor<16x4xf32> {
  %rows = neura.gather %table[%idx] : tensor<16x4xf32>, tensor<8xi64> -> tensor<8x4xf32>
  %out = neura.scatter %table[%idx], %vals : tensor<16x4xf32>, tensor<8xi64>, tensor<8x4xf32> -> tensor<16x4xf32>
  return %out : tensor<16x4xf32>
}
