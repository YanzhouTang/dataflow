// Ray Tracing - Möller-Trumbore stage 2: v/t barycentric + closest-hit (C-CGRA).
// Consumes stage-1 (f, u, s, e1) and needs e2 for t: q = s×e1; v = f*(d·q);
// t = f*(e2·q); accept when u,v in-range and t is the closest so far (best_t
// recurrence). Split from the full MT body for mapper tractability.
void rtTriTest(float rdx, float rdy, float rdz, float t_max,
               float *f_in, float *u_in,
               float *sx_in, float *sy_in, float *sz_in,
               float *e1x_in, float *e1y_in, float *e1z_in,
               float *e2x_in, float *e2y_in, float *e2z_in,
               float *out_best_t, int *out_best_tri) {
  float best_t = t_max;
  int best_tri = -1;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float f = f_in[i], u = u_in[i];
    float sx = sx_in[i], sy = sy_in[i], sz = sz_in[i];
    float e1x = e1x_in[i], e1y = e1y_in[i], e1z = e1z_in[i];
    float e2x = e2x_in[i], e2y = e2y_in[i], e2z = e2z_in[i];

    float qx = sy * e1z - sz * e1y;
    float qy = sz * e1x - sx * e1z;
    float qz = sx * e1y - sy * e1x;
    float vv = f * (rdx * qx + rdy * qy + rdz * qz);
    float t = f * (e2x * qx + e2y * qy + e2z * qz);

    int in_u = (u >= 0.0f) & (u <= 1.0f);
    int in_v = (vv >= 0.0f) & (u + vv <= 1.0f);
    int closer = (t > 1e-7f) & (t < best_t);
    int keep = in_u & in_v & closer;
    best_t = keep ? t : best_t;
    best_tri = keep ? i : best_tri;
  }
  out_best_t[0] = best_t;
  out_best_tri[0] = best_tri;
}
