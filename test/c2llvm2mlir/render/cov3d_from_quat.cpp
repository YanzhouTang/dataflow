// 3DGS Feature Computation - 3D covariance from quaternion+scale (C-CGRA).
// Stage 1 of covariance derivation (the full body is split for mapper
// tractability): build the rotation matrix R from the quaternion (w,x,y,z),
// form M = R*diag(sx,sy,sz), and output the 6 unique entries of the symmetric
// Sigma3D = M*M^T. These feed the EWA Jacobian projection (cov2d_project).
void cov3dFromQuat(float *qw, float *qx, float *qy, float *qz,
                   float *sx, float *sy, float *sz,
                   float *out_s00, float *out_s01, float *out_s02,
                   float *out_s11, float *out_s12, float *out_s22) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float w = qw[i], x = qx[i], y = qy[i], z = qz[i];
    float r00 = 1.0f - 2.0f * (y * y + z * z);
    float r01 = 2.0f * (x * y - w * z);
    float r02 = 2.0f * (x * z + w * y);
    float r10 = 2.0f * (x * y + w * z);
    float r11 = 1.0f - 2.0f * (x * x + z * z);
    float r12 = 2.0f * (y * z - w * x);
    float r20 = 2.0f * (x * z - w * y);
    float r21 = 2.0f * (y * z + w * x);
    float r22 = 1.0f - 2.0f * (x * x + y * y);

    float sxv = sx[i], syv = sy[i], szv = sz[i];
    float m00 = r00 * sxv, m01 = r01 * syv, m02 = r02 * szv;
    float m10 = r10 * sxv, m11 = r11 * syv, m12 = r12 * szv;
    float m20 = r20 * sxv, m21 = r21 * syv, m22 = r22 * szv;

    out_s00[i] = m00 * m00 + m01 * m01 + m02 * m02;
    out_s01[i] = m00 * m10 + m01 * m11 + m02 * m12;
    out_s02[i] = m00 * m20 + m01 * m21 + m02 * m22;
    out_s11[i] = m10 * m10 + m11 * m11 + m12 * m12;
    out_s12[i] = m10 * m20 + m11 * m21 + m12 * m22;
    out_s22[i] = m20 * m20 + m21 * m21 + m22 * m22;
  }
}
