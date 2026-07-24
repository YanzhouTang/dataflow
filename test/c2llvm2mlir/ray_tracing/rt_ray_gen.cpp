// Ray Tracing - primary ray generation (C-CGRA), RAICHU §RT (Fig 1 "Ray
// Marching" from the camera). For each pixel, form the ray direction from the
// camera basis (forward + u*right + v*up) and normalize it (uses the real
// sqrt via neura.fsqrt). Recurrence-free per-pixel compute -> C-CGRA.
#include <math.h>
void rtRayGen(float *u_in, float *v_in,
              float fwd_x, float fwd_y, float fwd_z,
              float right_x, float right_y, float right_z,
              float up_x, float up_y, float up_z,
              float *out_dx, float *out_dy, float *out_dz) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float u = u_in[i];
    float v = v_in[i];
    float dx = fwd_x + u * right_x + v * up_x;
    float dy = fwd_y + u * right_y + v * up_y;
    float dz = fwd_z + u * right_z + v * up_z;
    float len2 = dx * dx + dy * dy + dz * dz;
    float inv_len = 1.0f / sqrtf(len2);
    out_dx[i] = dx * inv_len;
    out_dy[i] = dy * inv_len;
    out_dz[i] = dz * inv_len;
  }
}
