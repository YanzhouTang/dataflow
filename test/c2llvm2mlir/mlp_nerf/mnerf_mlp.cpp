// MLP NeRF (classic NeRF) - MLP hidden-layer body (C-CGRA). Classic NeRF uses a
// deep MLP (8 layers x 256) mapping the encoded position to density + features.
// This kernel is one hidden unit's dot over an 8-wide encoded input plus bias
// and ReLU; looped over 32 units. Matrix-vector dot = the FVCU / fmul_fadd
// motif that dominates NeRF MLP compute. This is the per-layer body whose II
// the simulator scales by (layers x hidden-units).
void mnerfMLP(float *x, float *w, float *bias, float *out_h) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int u = 0; u < 32; u++) {
    int b = 8 * u;
    float acc = x[b + 0] * w[b + 0] + x[b + 1] * w[b + 1] +
                x[b + 2] * w[b + 2] + x[b + 3] * w[b + 3] +
                x[b + 4] * w[b + 4] + x[b + 5] * w[b + 5] +
                x[b + 6] * w[b + 6] + x[b + 7] * w[b + 7] + bias[u];
    out_h[u] = acc > 0.0f ? acc : 0.0f; // ReLU
  }
}
