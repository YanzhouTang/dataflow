// 3DGS Rasterization - Blending Stage 2: front-to-back compositing (C-CGRA).
// Consumes the per-Gaussian alpha from Stage 1 and performs the volume-
// rendering recurrence: accumulate 3 color channels weighted by alpha*T and
// carry the transmittance T *= (1 - alpha). The loop-carried T (and the 3
// accumulators) set the recurrence-bound II, which is the throughput limiter
// of the whole blending pipeline.
void blendComposite(float *features,
                    float *alpha_in,
                    int *point_indices,
                    float *out_color) {
  float T = 1.0f;
  float C0 = 0.0f;
  float C1 = 0.0f;
  float C2 = 0.0f;

#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < 32; i++) {
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
