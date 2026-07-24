// Ray Tracing - Möller-Trumbore stage 1: edge/determinant setup (C-CGRA).
// Per triangle: edges e1,e2; h = d×e2; a = e1·h; f = 1/a; and the u barycentric
// via s = o-v0. Outputs (f, u, and the reusable s,e1) for the second stage.
// Split from the full MT body (110 ops) for mapper tractability.
void rtTriSetup(float rox, float roy, float roz,
                float rdx, float rdy, float rdz,
                float *v0x, float *v0y, float *v0z,
                float *v1x, float *v1y, float *v1z,
                float *v2x, float *v2y, float *v2z,
                float *out_f, float *out_u,
                float *out_sx, float *out_sy, float *out_sz,
                float *out_e1x, float *out_e1y, float *out_e1z) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float e1x = v1x[i] - v0x[i], e1y = v1y[i] - v0y[i], e1z = v1z[i] - v0z[i];
    float e2x = v2x[i] - v0x[i], e2y = v2y[i] - v0y[i], e2z = v2z[i] - v0z[i];
    float hx = rdy * e2z - rdz * e2y;
    float hy = rdz * e2x - rdx * e2z;
    float hz = rdx * e2y - rdy * e2x;
    float a = e1x * hx + e1y * hy + e1z * hz;
    float f = 1.0f / a;
    float sx = rox - v0x[i], sy = roy - v0y[i], sz = roz - v0z[i];
    float u = f * (sx * hx + sy * hy + sz * hz);
    out_f[i] = f;
    out_u[i] = u;
    out_sx[i] = sx; out_sy[i] = sy; out_sz[i] = sz;
    out_e1x[i] = e1x; out_e1y[i] = e1y; out_e1z[i] = e1z;
  }
}
