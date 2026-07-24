// Ray Tracing - closest-hit reduction / argmin (C-CGRA).
// Given per-triangle intersection distances t[i] and validity flags, reduce to
// the nearest valid hit (best_t, best_tri). The loop-carried best_t/best_tri is
// the recurrence that sets the II -- the classic ray-tracing "closest hit"
// scan, analogous to the transmittance recurrence in blending.
void rtTriReduce(float *t_in, int *valid_in, float t_max,
                 float *out_best_t, int *out_best_tri) {
  float best_t = t_max;
  int best_tri = -1;
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float t = t_in[i];
    int keep = valid_in[i] & (t < best_t);
    best_t = keep ? t : best_t;
    best_tri = keep ? i : best_tri;
  }
  out_best_t[0] = best_t;
  out_best_tri[0] = best_tri;
}
