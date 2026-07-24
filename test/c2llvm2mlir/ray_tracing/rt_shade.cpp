// Ray Tracing - shading / accumulation (C-CGRA), RAICHU §RT ("Accumulation /
// Intersection" -> pixel color). For each hit, Lambertian shading with a
// distance falloff: color = albedo * max(0, N·L) / (1 + k*t²). Per-hit compute,
// no recurrence -> C-CGRA.
void rtShade(float *nx, float *ny, float *nz,
             float *albedo, float *hit_t,
             float lx, float ly, float lz, float k,
             float *out_color) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    float ndotl = nx[i] * lx + ny[i] * ly + nz[i] * lz;
    float diff = ndotl > 0.0f ? ndotl : 0.0f;
    float t = hit_t[i];
    float atten = 1.0f / (1.0f + k * t * t);
    out_color[i] = albedo[i] * diff * atten;
  }
}
