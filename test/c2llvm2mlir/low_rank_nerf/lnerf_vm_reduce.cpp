// Low-Rank NeRF (TensoRF) - rank reduction stage (C-CGRA), RAICHU §4.3.2 / §4.4.
// This is the "reduce stage commonly used in NeRF and low-rank NeRF" that the
// FVCU G2 mode targets. The per-rank VM contributions (produced by the product
// stage, materialized in the shared buffer) are summed over the R=6 rank
// components to give the density (or a feature coefficient). Because the six
// summands are materialized values (not inline products), this is a pure 6-way
// reduction and fuses into one neura.fvc_reduce6 (G2).
void lnerfVMReduce(float *rank_term, float *out_sigma) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (int s = 0; s < 32; s++) {
    int b = 6 * s;
    out_sigma[s] = rank_term[b + 0] + rank_term[b + 1] + rank_term[b + 2] +
                   rank_term[b + 3] + rank_term[b + 4] + rank_term[b + 5];
  }
}
