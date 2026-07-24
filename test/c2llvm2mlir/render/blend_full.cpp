// 3DGS Rasterization / Volume Rendering (Blending) kernel, RAICHU §4 /
// GSCore §3.1 "Rasterization". Full per-pixel front-to-back compositing:
//   - gather each Gaussian via its sorted index (point_indices),
//   - evaluate the 2D Gaussian (conic power) at the pixel,
//   - alpha = opacity * exp(power) approximated by 2nd-order polynomial + clamp,
//   - accumulate 3 color channels weighted by alpha*T,
//   - carry the transmittance recurrence T *= (1 - alpha) (the II bottleneck).
// This is the complete blending body (3 channels, polynomial alpha, clamp,
// transmittance recurrence) mapped onto the C-CGRA.
#include <math.h>
void renderPixel(float pixel_x, float pixel_y,
                 float *out_color,
                 float *points_x, float *points_y,
                 float *features,
                 float *conic_x, float *conic_y, float *conic_z,
                 float *opacity,
                 int *point_indices) {
  float T = 1.0f;
  float C0 = 0.0f;
  float C1 = 0.0f;
  float C2 = 0.0f;

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

    // alpha = opacity * exp(power), 2nd-order polynomial approximation.
    float alpha_raw = opacity[idx] * (1.0f + power + 0.5f * power * power);
    float alpha_temp = fmaxf(0.0f, alpha_raw);
    float alpha = fminf(0.99f, alpha_temp);

    // Front-to-back compositing of 3 color channels.
    float weight = alpha * T;
    C0 += features[idx * 3 + 0] * weight;
    C1 += features[idx * 3 + 1] * weight;
    C2 += features[idx * 3 + 2] * weight;

    // Transmittance recurrence (loop-carried).
    T = T * (1.0f - alpha);
  }

  out_color[0] = C0;
  out_color[1] = C1;
  out_color[2] = C2;
}
