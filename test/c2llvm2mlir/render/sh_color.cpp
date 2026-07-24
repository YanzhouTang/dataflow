// 3DGS Feature Computation - Spherical Harmonics -> RGB color (C-CGRA).
// GSCore §3.1: the view-dependent color is evaluated from SH coefficients and
// the view direction (this dominates the 59-dim feature). Faithful degree-2 SH
// evaluation (9 basis) using the exact 3DGS SH constants, for one color
// channel; RGB replicates this three times (one coefficient set per channel).
void shColor(float *dir_x, float *dir_y, float *dir_z,
             float *sh0, float *sh1, float *sh2, float *sh3, float *sh4,
             float *sh5, float *sh6, float *sh7, float *sh8,
             float *out_c) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float x = dir_x[i], y = dir_y[i], z = dir_z[i];
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    // Degree-0 and degree-1.
    float c = 0.28209479f * sh0[i];
    c = c - 0.48860251f * y * sh1[i];
    c = c + 0.48860251f * z * sh2[i];
    c = c - 0.48860251f * x * sh3[i];

    // Degree-2.
    c = c + 1.09254843f * xy * sh4[i];
    c = c - 1.09254843f * yz * sh5[i];
    c = c + 0.31539157f * (2.0f * zz - xx - yy) * sh6[i];
    c = c - 1.09254843f * xz * sh7[i];
    c = c + 0.54627421f * (xx - yy) * sh8[i];

    // 3DGS adds 0.5 then clamps to non-negative.
    c = c + 0.5f;
    out_c[i] = c > 0.0f ? c : 0.0f;
  }
}
