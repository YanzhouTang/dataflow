// 3DGS Preprocessing kernel (M-CGRA), RAICHU §4.3.3 "compute conic terms and
// generate gaussian-centric job lists". Per-gaussian (no loop-carried
// recurrence): perspective projection (2D screen position + depth) and the
// conic = inverse of the 2D covariance [a b; b c]. These feed the sorting /
// job-list stage. Kept compact to be mapper-tractable.
void preprocessGaussians(float fx, float fy, float ppx, float ppy,
                         float *mean_x, float *mean_y, float *mean_z,
                         float *cov_a, float *cov_b, float *cov_c,
                         float *out_sx, float *out_sy, float *out_depth,
                         float *out_conic_x, float *out_conic_y,
                         float *out_conic_z) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    // Perspective projection: screen = f * mean / z + principal_point.
    float z = mean_z[i];
    float inv_z = 1.0f / z;
    out_sx[i] = fx * mean_x[i] * inv_z + ppx;
    out_sy[i] = fy * mean_y[i] * inv_z + ppy;
    out_depth[i] = z;

    // Conic = inverse of the 2D covariance [a b; b c].
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
