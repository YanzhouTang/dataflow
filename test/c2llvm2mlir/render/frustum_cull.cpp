// 3DGS Frustum Culling (M-CGRA), GSCore §3.1 step 1.
// Iterate all N Gaussians, transform the mean into view space, compute the
// depth (z in view space), and test whether the Gaussian is inside the viewing
// frustum (near/far planes + screen bounds after projection). Only the mean is
// read here (no full feature load), so it is a light memory-streaming pass that
// emits a per-Gaussian keep mask and the depth used later for sorting.
void frustumCull(float *mean_x, float *mean_y, float *mean_z,
                 // view transform (row-major 3x4: R|t), given as scalars
                 float r00, float r01, float r02, float t0,
                 float r10, float r11, float r12, float t1,
                 float r20, float r21, float r22, float t2,
                 float fx, float fy, float near_z, float far_z,
                 float half_w, float half_h,
                 int *out_keep, float *out_depth) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float X = mean_x[i], Y = mean_y[i], Z = mean_z[i];
    // View-space coordinates.
    float vx = r00 * X + r01 * Y + r02 * Z + t0;
    float vy = r10 * X + r11 * Y + r12 * Z + t1;
    float vz = r20 * X + r21 * Y + r22 * Z + t2;

    // Projected screen position (for screen-bound test).
    float inv_z = 1.0f / vz;
    float sx = fx * vx * inv_z;
    float sy = fy * vy * inv_z;

    // Frustum test: near/far and screen bounds.
    int in_depth = (vz > near_z) & (vz < far_z);
    int in_x = (sx > -half_w) & (sx < half_w);
    int in_y = (sy > -half_h) & (sy < half_h);
    out_keep[i] = in_depth & in_x & in_y;
    out_depth[i] = vz;
  }
}
