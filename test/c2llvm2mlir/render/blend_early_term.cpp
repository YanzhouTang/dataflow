// 3DGS Blending with early ray termination (C-CGRA), GSCore §3.1 / §4.
// Volume rendering can stop once the accumulated transmittance T falls below a
// threshold: the remaining (farther) Gaussians cannot contribute. This adds a
// data-dependent loop exit on top of the compositing recurrence, which the
// control-to-dataflow lowering turns into a predicated loop-carried exit.
void blendEarlyTerm(float *features, float *alpha_in, int *point_indices,
                    float t_min, float *out_color) {
  float T = 1.0f;
  float C0 = 0.0f;
  float C1 = 0.0f;
  float C2 = 0.0f;

#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
    // Early ray termination.
    if (T < t_min)
      break;

    int idx = point_indices[i];
    float alpha = alpha_in[i];

    float weight = alpha * T;
    C0 += features[idx * 3 + 0] * weight;
    C1 += features[idx * 3 + 1] * weight;
    C2 += features[idx * 3 + 2] * weight;

    T = T * (1.0f - alpha);
  }

  out_color[0] = C0;
  out_color[1] = C1;
  out_color[2] = C2;
}
