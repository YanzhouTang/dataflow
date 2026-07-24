// 3DGS Feature Computation - 3D covariance from quaternion+scale, projected to
// 2D via the EWA Jacobian (C-CGRA). Faithful to the original 3DGS:
//   R = rotation matrix built from quaternion (w,x,y,z),
//   M = R * diag(sx,sy,sz), Sigma3D = M * M^T   (symmetric, 6 unique entries),
//   J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]  (perspective Jacobian),
//   Sigma2D = J * Sigma3D * J^T   (2x2, view assumed axis-aligned so W=I).
// Outputs the 2D covariance (a,b,c) consumed by conic/extent computation.
void cov3dJacobian(float *qw, float *qx, float *qy, float *qz,
                   float *sx, float *sy, float *sz,
                   float *mx, float *my, float *mz,
                   float fx, float fy,
                   float *out_a, float *out_b, float *out_c) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float w = qw[i], x = qx[i], y = qy[i], z = qz[i];
    // Rotation matrix from quaternion (rows r0,r1,r2).
    float r00 = 1.0f - 2.0f * (y * y + z * z);
    float r01 = 2.0f * (x * y - w * z);
    float r02 = 2.0f * (x * z + w * y);
    float r10 = 2.0f * (x * y + w * z);
    float r11 = 1.0f - 2.0f * (x * x + z * z);
    float r12 = 2.0f * (y * z - w * x);
    float r20 = 2.0f * (x * z - w * y);
    float r21 = 2.0f * (y * z + w * x);
    float r22 = 1.0f - 2.0f * (x * x + y * y);

    // M = R * S (scale columns).
    float sxv = sx[i], syv = sy[i], szv = sz[i];
    float m00 = r00 * sxv, m01 = r01 * syv, m02 = r02 * szv;
    float m10 = r10 * sxv, m11 = r11 * syv, m12 = r12 * szv;
    float m20 = r20 * sxv, m21 = r21 * syv, m22 = r22 * szv;

    // Sigma3D = M * M^T (6 unique entries).
    float s00 = m00 * m00 + m01 * m01 + m02 * m02;
    float s01 = m00 * m10 + m01 * m11 + m02 * m12;
    float s02 = m00 * m20 + m01 * m21 + m02 * m22;
    float s11 = m10 * m10 + m11 * m11 + m12 * m12;
    float s12 = m10 * m20 + m11 * m21 + m12 * m22;
    float s22 = m20 * m20 + m21 * m21 + m22 * m22;

    // Perspective Jacobian rows (W=I): j0=(fx/z,0,-fx*X/z^2), j1=(0,fy/z,-fy*Y/z^2).
    float zv = mz[i];
    float inv_z = 1.0f / zv;
    float inv_z2 = inv_z * inv_z;
    float j00 = fx * inv_z;
    float j02 = -fx * mx[i] * inv_z2;
    float j11 = fy * inv_z;
    float j12 = -fy * my[i] * inv_z2;

    // Sigma2D = J * Sigma3D * J^T.
    // t0 = J*Sigma3D (2x3): t0_k = sum_l J_kl * S_lk
    float t00 = j00 * s00 + j02 * s02;
    float t01 = j00 * s01 + j02 * s12;
    float t02 = j00 * s02 + j02 * s22;
    float t10 = j11 * s01 + j12 * s02;
    float t11 = j11 * s11 + j12 * s12;
    float t12 = j11 * s12 + j12 * s22;
    out_a[i] = t00 * j00 + t02 * j02;              // (0,0)
    out_b[i] = t00 * 0.0f + t01 * j11 + t02 * j12; // (0,1)
    out_c[i] = t11 * j11 + t12 * j12;              // (1,1)
  }
}
