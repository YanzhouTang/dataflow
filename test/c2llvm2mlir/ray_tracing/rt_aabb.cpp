// Ray Tracing - BVH node AABB slab test (C-CGRA FVCU), RAICHU §RT.
// For each BVH node, the ray-box slab test computes per-axis entry/exit t's and
// reduces them with min/max: tnear = max(tmin_xyz), tfar = min(tmax_xyz); the
// node is hit iff tnear <= tfar (within [t_min, t_max]). This is exactly the
// FVCU fmin/fmax reduction pattern (RAICHU FVCU AABB primitive). Per-node, no
// recurrence -> maps on the C-CGRA at the resource-bound II.
#include <math.h>
void rtAABB(float rox, float roy, float roz,
            float rdx, float rdy, float rdz, float t_min, float t_max,
            float *min_x, float *min_y, float *min_z,
            float *max_x, float *max_y, float *max_z,
            int *out_hit) {
  float invx = 1.0f / rdx;
  float invy = 1.0f / rdy;
  float invz = 1.0f / rdz;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float t1x = (min_x[i] - rox) * invx;
    float t2x = (max_x[i] - rox) * invx;
    float t1y = (min_y[i] - roy) * invy;
    float t2y = (max_y[i] - roy) * invy;
    float t1z = (min_z[i] - roz) * invz;
    float t2z = (max_z[i] - roz) * invz;

    float tnear = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
    float tfar = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
    tnear = fmaxf(tnear, t_min);
    tfar = fminf(tfar, t_max);

    out_hit[i] = tnear <= tfar ? 1 : 0;
  }
}
