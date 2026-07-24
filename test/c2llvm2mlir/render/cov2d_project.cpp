// 3DGS Feature Computation - EWA Jacobian projection Sigma3D -> Sigma2D (C-CGRA).
// Stage 2 of covariance derivation. Given the 6 entries of the symmetric 3D
// covariance and the mean position, apply the perspective Jacobian
//   J = [[fx/z, 0, -fx*X/z^2], [0, fy/z, -fy*Y/z^2]]   (view axis-aligned, W=I)
// and compute Sigma2D = J*Sigma3D*J^T -> (a,b,c), consumed by conic/extent.
void cov2dProject(float *s00, float *s01, float *s02, float *s11, float *s12,
                  float *s22, float *mx, float *my, float *mz,
                  float fx, float fy,
                  float *out_a, float *out_b, float *out_c) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float inv_z = 1.0f / mz[i];
    float inv_z2 = inv_z * inv_z;
    float j00 = fx * inv_z;
    float j02 = -fx * mx[i] * inv_z2;
    float j11 = fy * inv_z;
    float j12 = -fy * my[i] * inv_z2;

    float a00 = s00[i], a01 = s01[i], a02 = s02[i];
    float a11 = s11[i], a12 = s12[i], a22 = s22[i];

    // t = J * Sigma3D (2x3).
    float t00 = j00 * a00 + j02 * a02;
    float t01 = j00 * a01 + j02 * a12;
    float t02 = j00 * a02 + j02 * a22;
    float t10 = j11 * a01 + j12 * a02;
    float t11 = j11 * a11 + j12 * a12;
    float t12 = j11 * a12 + j12 * a22;

    // Sigma2D = t * J^T (2x2 symmetric).
    out_a[i] = t00 * j00 + t02 * j02;
    out_b[i] = t01 * j11 + t02 * j12;
    out_c[i] = t11 * j11 + t12 * j12;
  }
}
