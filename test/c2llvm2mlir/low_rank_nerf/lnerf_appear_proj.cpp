// Low-Rank NeRF (TensoRF) - appearance matrix projection (C-CGRA), RAICHU
// §4.2.2 / §4.4. The appearance feature is the projection of the R rank
// coefficients onto a learned appearance-matrix row: feat = Σ_r c_r · B_r.
// This is a pure R=6 dot product (no bias), i.e. the "six-element fused
// multiply-accumulate chain" that the FVCU G1 partial_dot6 mode collapses.
void lnerfAppearProj(float *c, float *bmat, float *out_feat) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int s = 0; s < 32; s++) {
    int b = 6 * s;
    out_feat[s] = c[b + 0] * bmat[b + 0] + c[b + 1] * bmat[b + 1] +
                  c[b + 2] * bmat[b + 2] + c[b + 3] * bmat[b + 3] +
                  c[b + 4] * bmat[b + 4] + c[b + 5] * bmat[b + 5];
  }
}
