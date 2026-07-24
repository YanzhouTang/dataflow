// 3DGS Preprocessing / Feature Computation (C-CGRA), GSCore §3.1 / RAICHU §4.
// Faithful per-Gaussian feature computation covering the key functional points:
//   1. derive the 2D covariance from scale (sx,sy) and rotation (cos/sin given
//      to avoid trig ops): M = R*S, cov = M*M^T,
//   2. perspective projection of the 3D mean to screen space + depth,
//   3. conic = inverse of the 2D covariance (feeds the alpha evaluation),
//   4. bounding-box extent proxy from the covariance trace (feeds tile binning).
// This is compute-dense (many fmul/fmul_fadd), so it maps to the C-CGRA.
void preprocessGaussians(float fx, float fy, float ppx, float ppy,
                         float *mean_x, float *mean_y, float *mean_z,
                         float *scale_x, float *scale_y,
                         float *rot_cos, float *rot_sin,
                         float *out_sx, float *out_sy, float *out_depth,
                         float *out_conic_x, float *out_conic_y,
                         float *out_conic_z, float *out_extent) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    // (1) 2D covariance from scale + rotation: cov = (R*S)(R*S)^T.
    float ca = rot_cos[i];
    float sa = rot_sin[i];
    float sxv = scale_x[i];
    float syv = scale_y[i];
    float m00 = ca * sxv;
    float m01 = -sa * syv;
    float m10 = sa * sxv;
    float m11 = ca * syv;
    float a = m00 * m00 + m01 * m01;
    float b = m00 * m10 + m01 * m11;
    float c = m10 * m10 + m11 * m11;

    // (2) Perspective projection: screen = f * mean / z + principal_point.
    float z = mean_z[i];
    float inv_z = 1.0f / z;
    out_sx[i] = fx * mean_x[i] * inv_z + ppx;
    out_sy[i] = fy * mean_y[i] * inv_z + ppy;
    out_depth[i] = z;

    // (3) Conic = inverse of the 2D covariance [a b; b c].
    float det = a * c - b * b;
    float inv_det = 1.0f / det;
    out_conic_x[i] = c * inv_det;
    out_conic_y[i] = -b * inv_det;
    out_conic_z[i] = a * inv_det;

    // (4) Bounding-box extent proxy (covariance trace ~ 3-sigma radius).
    out_extent[i] = 3.0f * (a + c);
  }
}
