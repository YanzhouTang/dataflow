// 3DGS Feature Computation - Sigma3D = M*M^T (C-CGRA).
// Covariance derivation stage 1b: given the 9 entries of M = R*S, compute the 6
// unique entries of the symmetric 3D covariance Sigma3D, consumed by the EWA
// Jacobian projection (cov2d_project).
void sigma3dFromM(float *m00, float *m01, float *m02, float *m10, float *m11,
                  float *m12, float *m20, float *m21, float *m22,
                  float *s00, float *s01, float *s02, float *s11, float *s12,
                  float *s22) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float a00 = m00[i], a01 = m01[i], a02 = m02[i];
    float a10 = m10[i], a11 = m11[i], a12 = m12[i];
    float a20 = m20[i], a21 = m21[i], a22 = m22[i];
    s00[i] = a00 * a00 + a01 * a01 + a02 * a02;
    s01[i] = a00 * a10 + a01 * a11 + a02 * a12;
    s02[i] = a00 * a20 + a01 * a21 + a02 * a22;
    s11[i] = a10 * a10 + a11 * a11 + a12 * a12;
    s12[i] = a10 * a20 + a11 * a21 + a12 * a22;
    s22[i] = a20 * a20 + a21 * a21 + a22 * a22;
  }
}
