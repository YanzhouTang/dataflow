// Simplified 3DGS blending kernel (C-CGRA), reduced to a mapper-tractable size
// while keeping the RAICHU-relevant structure:
//   - conic "power" = conic . [dx^2, dy^2, dx*dy]  (a 3D dot -> FVCU fvc_dot3)
//   - alpha, transmittance recurrence T *= (1 - alpha)  (loop-carried scan)
//   - single-channel weighted accumulation
// The full 3-channel + polynomial-alpha renderPixel (kernel.cpp) is too large
// for the current mapper (see project TODO #18); this captures the same
// compute pattern for reading the blending II.
void blendSimple(float pixel_x, float pixel_y, float *out_color,
                 float *points_x, float *points_y,
                 float *conic_x, float *conic_y, float *conic_z,
                 float *opacity, float *features) {
  float T = 1.0f;
  float C = 0.0f;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float d_x = points_x[i] - pixel_x;
    float d_y = points_y[i] - pixel_y;
    // conic power (3D dot form).
    float power = -0.5f * (conic_x[i] * d_x * d_x + conic_z[i] * d_y * d_y) -
                  conic_y[i] * d_x * d_y;
    float alpha = opacity[i] * power;
    float weight = alpha * T;
    C += features[i] * weight;
    T = T * (1.0f - alpha);
  }
  out_color[0] = C;
}
