// 3DGS Feature Computation - M = R(quat)*diag(scale) (C-CGRA).
// Covariance derivation stage 1a: build rotation R from the quaternion
// (w,x,y,z) and scale its columns to form M = R*S. Outputs the 9 entries of M,
// consumed by sigma3d_from_m to form Sigma3D = M*M^T.
void rotMFromQuat(float *qw, float *qx, float *qy, float *qz,
                  float *sx, float *sy, float *sz,
                  float *m00, float *m01, float *m02,
                  float *m10, float *m11, float *m12,
                  float *m20, float *m21, float *m22) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float w = qw[i], x = qx[i], y = qy[i], z = qz[i];
    float sxv = sx[i], syv = sy[i], szv = sz[i];
    m00[i] = (1.0f - 2.0f * (y * y + z * z)) * sxv;
    m01[i] = (2.0f * (x * y - w * z)) * syv;
    m02[i] = (2.0f * (x * z + w * y)) * szv;
    m10[i] = (2.0f * (x * y + w * z)) * sxv;
    m11[i] = (1.0f - 2.0f * (x * x + z * z)) * syv;
    m12[i] = (2.0f * (y * z - w * x)) * szv;
    m20[i] = (2.0f * (x * z - w * y)) * sxv;
    m21[i] = (2.0f * (y * z + w * x)) * syv;
    m22[i] = (1.0f - 2.0f * (x * x + y * y)) * szv;
  }
}
