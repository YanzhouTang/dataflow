// Ray Tracing - ray-triangle intersection, Möller-Trumbore (C-CGRA), RAICHU §RT.
// For each candidate triangle: two edge vectors, h = d×e2, a = e1·h, f = 1/a,
// barycentric u = f*(s·h), q = s×e1, v = f*(d·q), t = f*(e2·q); the hit is
// kept when u,v are in-range and t is the closest so far (best_t recurrence).
// Cross/dot-heavy compute -> C-CGRA. Predicated (u,v,t range tests) so the
// control-to-dataflow lowering handles the reject branches.
void rtTriangle(float rox, float roy, float roz,
                float rdx, float rdy, float rdz, float t_max,
                float *v0x, float *v0y, float *v0z,
                float *v1x, float *v1y, float *v1z,
                float *v2x, float *v2y, float *v2z,
                float *out_best_t, int *out_best_tri) {
  float best_t = t_max;
  int best_tri = -1;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float e1x = v1x[i] - v0x[i], e1y = v1y[i] - v0y[i], e1z = v1z[i] - v0z[i];
    float e2x = v2x[i] - v0x[i], e2y = v2y[i] - v0y[i], e2z = v2z[i] - v0z[i];

    // h = d x e2 ; a = e1 . h
    float hx = rdy * e2z - rdz * e2y;
    float hy = rdz * e2x - rdx * e2z;
    float hz = rdx * e2y - rdy * e2x;
    float a = e1x * hx + e1y * hy + e1z * hz;
    float f = 1.0f / a;

    // s = o - v0 ; u = f*(s.h)
    float sx = rox - v0x[i], sy = roy - v0y[i], sz = roz - v0z[i];
    float u = f * (sx * hx + sy * hy + sz * hz);

    // q = s x e1 ; v = f*(d.q) ; t = f*(e2.q)
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
