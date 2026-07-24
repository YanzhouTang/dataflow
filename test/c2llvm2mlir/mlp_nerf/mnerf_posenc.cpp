// MLP NeRF (classic NeRF) - positional encoding (C-CGRA), the signature NeRF
// front-end. Each coordinate p is lifted to a high-frequency Fourier feature
//   gamma(p) = [ sin(2^k * pi * p), cos(2^k * pi * p) ]  for k = 0..L-1
// so the MLP can represent high-frequency detail. This kernel emits the L=4
// frequency band (sin/cos) for one coordinate; uses the real sin/cos (lowered
// to neura.fsin / neura.fcos). Recurrence-free -> C-CGRA.
#include <math.h>
void mnerfPosEnc(float *p_in,
                 float *out_sin0, float *out_cos0,
                 float *out_sin1, float *out_cos1,
                 float *out_sin2, float *out_cos2,
                 float *out_sin3, float *out_cos3) {
  const float PI = 3.14159265f;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float p = p_in[i];
    float f0 = PI * p;         // 2^0 * pi * p
    float f1 = 2.0f * f0;      // 2^1
    float f2 = 4.0f * f0;      // 2^2
    float f3 = 8.0f * f0;      // 2^3
    out_sin0[i] = sinf(f0); out_cos0[i] = cosf(f0);
    out_sin1[i] = sinf(f1); out_cos1[i] = cosf(f1);
    out_sin2[i] = sinf(f2); out_cos2[i] = cosf(f2);
    out_sin3[i] = sinf(f3); out_cos3[i] = cosf(f3);
  }
}
