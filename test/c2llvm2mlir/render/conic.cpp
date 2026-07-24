// 3DGS Preprocessing core (M-CGRA): "compute conic terms" (RAICHU §4.3.3).
// conic = inverse of the 2D covariance [a b; b c]:
//   det = a*c - b*b;  conic = (c, -b, a) / det.
// Per-gaussian, no loop-carried recurrence. Minimal size to map cleanly.
void computeConic(float *cov_a, float *cov_b, float *cov_c,
                  float *out_conic_x, float *out_conic_y, float *out_conic_z) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float a = cov_a[i];
    float b = cov_b[i];
    float c = cov_c[i];
    float det = a * c - b * b;
    float inv_det = 1.0f / det;
    out_conic_x[i] = c * inv_det;
    out_conic_y[i] = -b * inv_det;
    out_conic_z[i] = a * inv_det;
  }
}
