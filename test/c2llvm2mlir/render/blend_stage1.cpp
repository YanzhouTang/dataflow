// 3DGS Rasterization - Blending Stage 1: alpha-computation (C-CGRA).
// GSCore §4.3 isolates alpha-computation as its own step (for alpha-pruning);
// RAICHU maps this per-Gaussian, recurrence-free body onto the C-CGRA. It
// gathers each Gaussian by its sorted index, evaluates the 2D Gaussian
// (conic power) at the pixel, and produces the clamped alpha. No loop-carried
// state, so it pipelines at the resource-bound II.
#include <math.h>
void blendAlpha(float pixel_x, float pixel_y,
                float *points_x, float *points_y,
                float *conic_x, float *conic_y, float *conic_z,
                float *opacity,
                int *point_indices,
                float *out_alpha) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    int idx = point_indices[i];

    float d_x = points_x[idx] - pixel_x;
    float d_y = points_y[idx] - pixel_y;

    float con_x = conic_x[idx];
    float con_y = conic_y[idx];
    float con_z = conic_z[idx];

    // 2D Gaussian exponent (conic form).
    float power =
        -0.5f * (con_x * d_x * d_x + con_z * d_y * d_y) - con_y * d_x * d_y;

    // alpha = opacity * exp(power), 2nd-order polynomial approximation + clamp.
    float alpha_raw = opacity[idx] * (1.0f + power + 0.5f * power * power);
    float alpha_temp = fmaxf(0.0f, alpha_raw);
    float alpha = fminf(0.99f, alpha_temp);

    out_alpha[i] = alpha;
  }
}
