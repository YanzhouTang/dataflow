// MLP NeRF (classic NeRF) - volume-rendering accumulation (C-CGRA). Along a ray
// the sampled (density sigma, color c) are composited front-to-back:
//   alpha_i = 1 - exp(-sigma_i * delta_i)
//   w_i     = alpha_i * T,  T *= (1 - alpha_i)
//   C      += w_i * c_i
// The transmittance T is the loop-carried recurrence (the NeRF rendering
// bottleneck), and alpha uses the real exp (neura.fexp).
#include <math.h>
void mnerfAccum(float *sigma, float *delta, float *color, float *out_C) {
  float T = 1.0f;
  float C = 0.0f;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float alpha = 1.0f - expf(-sigma[i] * delta[i]);
    float w = alpha * T;
    C += w * color[i];
    T = T * (1.0f - alpha);
  }
  out_C[0] = C;
}
