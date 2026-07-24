// Low-Rank NeRF (TensoRF) - appearance MLP decode (C-CGRA), RAICHU §4.3.2.
// TensoRF decodes the appearance feature vector (plus the view direction) into
// RGB with a tiny MLP. This kernel is one hidden-unit dot: h = w·feat + b,
// followed by a ReLU, over a K=6 appearance feature -- the matrix-vector /
// dot-product motif the FVCU (G1/G2) targets. Looped over 32 samples.
void lnerfMLP(float *feat, float *w, float *bias, float *out_h) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int s = 0; s < 32; s++) {
    int b = 6 * s;
    // K=6 dot product feat·w + bias (dot completion = FVCU reduce).
    float acc = feat[b + 0] * w[b + 0] + feat[b + 1] * w[b + 1] +
                feat[b + 2] * w[b + 2] + feat[b + 3] * w[b + 3] +
                feat[b + 4] * w[b + 4] + feat[b + 5] * w[b + 5] + bias[s];
    // ReLU.
    out_h[s] = acc > 0.0f ? acc : 0.0f;
  }
}
