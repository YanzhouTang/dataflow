// 3DGS Blending - alpha-computation with the REAL exponential (C-CGRA).
// alpha = opacity * exp(power), where power is the 2D Gaussian conic exponent.
// Uses the true exp() (lowered to neura.fexp), replacing the earlier 2nd-order
// polynomial approximation.
#include <math.h>
void blendAlphaExp(float pixel_x, float pixel_y,
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

    float power =
        -0.5f * (con_x * d_x * d_x + con_z * d_y * d_y) - con_y * d_x * d_y;

    // Real alpha with exp, then clamp to [0, 0.99].
    float alpha_raw = opacity[idx] * expf(power);
    float alpha_temp = fmaxf(0.0f, alpha_raw);
    float alpha = fminf(0.99f, alpha_temp);

    out_alpha[i] = alpha;
  }
}
